"""`photocull audit` / `cluster` / `calibrate` entry points.

`run_*`'s `records` parameter is the testable seam: pass real records in production
(default, opens the real library) or synthetic fixtures in tests -- either way the
wiring below is the same code path. `cluster` and `calibrate` extend that to `cache`
and `analyzers` for the same reason, so the pipeline can be driven without SQLite,
Vision or a Photos library.

**This is where `config_io` first reaches production.** Task 7b built the
defaults <- file <- CLI-flag precedence chain and validated it thoroughly in unit
tests, but nothing imported it, so the chain had never actually run. The override
flags below therefore arrive as `load_config(overrides=...)` rather than being applied
by hand -- that path is the one `config-validation-covers-every-field` hardened, and
applying them directly would skip every check it added."""

from __future__ import annotations

import argparse
import importlib
import tempfile
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

from rich.console import Console
from rich.table import Table

from .cache import AnalysisCache
from .calibrate import (
    DEFAULT_SAMPLE_SIZE,
    write_calibration_sample,
    write_contact_sheet,
)
from .albums import AlbumConfig, TripGroup, demo_pool, keeper_pool, trip_sweep
from .config import ClusterConfig
from .config_io import find_config, load_config
from .models import AuditSummary, Cluster, PhotoRecord
from .photos_source import iter_photo_records, open_library
from .photos_source import keeper_uuids as _keeper_uuids
from .report import render_table, write_json
from .summary import DEFAULT_GAP_SECONDS, build_summary
from .sweep import SweepGroup

#: Where the analysis cache lives by default. A cold run over the real library is
#: minutes of Vision work and a warm one is seconds, so the cache is worth keeping.
DEFAULT_CACHE = "out/analysis.db"

#: The review UI's dependencies live in an extra, so a plain install carries no web
#: server (`review-ui-is-a-local-web-app`). Anyone running `photocull review` on such
#: an install must be told the command that fixes it, not handed a traceback.
REVIEW_EXTRA_HINT = (
    "`photocull review` needs the review extra, which is not installed.\n"
    'Install it with:  pip install -e ".[review]"'
)


class ReviewUnavailable(RuntimeError):
    """The `[review]` extra is missing. Carries the install command, not a traceback."""


ALBUM_EXTRA_HINT = (
    "`photocull albums serve` needs the album extra, which is not installed.\n"
    'Install it with:  pip install -e ".[album]"'
)


class AlbumUIUnavailable(RuntimeError):
    """The `[album]` extra is missing. Carries the install command, not a traceback."""


def _load_album_launcher(
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> ModuleType:
    """Import the album-sequencing launcher, translating only *its own* missing
    dependencies. Same shape as `_load_launcher` above, and the same reason: this
    import stays inside a function so `test_photocull_never_imports_a_review_
    dependency`-style isolation holds for the `album` extra too, not just `review`."""
    try:
        return importer("photocull_album.launcher")
    except ModuleNotFoundError as exc:
        if exc.name in {"fastapi", "starlette", "uvicorn", "httpx"}:
            raise AlbumUIUnavailable(ALBUM_EXTRA_HINT) from exc
        raise


def _load_launcher(
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> ModuleType:
    """Import the review launcher, translating only *its own* missing dependencies.

    The `importer` seam exists so the refusal can be tested without uninstalling
    anything. The `exc.name` check is the point of the function: a `ModuleNotFoundError`
    naming `fastapi` is a missing extra and is worth an install hint, while one naming
    anything else is a bug in this repo, and reporting that as "install the extra"
    would send the reader to fix the wrong thing.

    This import is inside a function, not at module scope, because
    `tests/test_guardrails.py` asserts in a subprocess that importing `photocull.cli`
    pulls in no review-only dependency at all.
    """
    try:
        return importer("photocull_review.launcher")
    except ModuleNotFoundError as exc:
        if exc.name in {"fastapi", "starlette", "uvicorn", "photoscript"}:
            raise ReviewUnavailable(REVIEW_EXTRA_HINT) from exc
        raise


def run_audit(
    *,
    library: str | None = None,
    json_out: str | None = None,
    gap_seconds: float = DEFAULT_GAP_SECONDS,
    records: Iterable[PhotoRecord] | None = None,
    console: Console | None = None,
) -> AuditSummary:
    if records is None:
        db = open_library(library)
        records = iter_photo_records(db)

    summary = build_summary(records, gap_seconds=gap_seconds, library_path=library)
    render_table(summary, console=console)
    if json_out:
        write_json(summary, json_out)
    return summary


# --- clustering ----------------------------------------------------------------


def _resolve_config(
    config_path: str | None, threshold: float | None, gap_seconds: float | None
) -> tuple[ClusterConfig, Path | None]:
    """Shipped defaults <- config file <- CLI flags.

    `load_config` ignores None-valued overrides, so unset flags fall through to the
    file rather than stamping the defaults over it."""
    overrides = {"similarity_threshold": threshold, "gap_seconds": gap_seconds}
    return load_config(config_path, overrides), find_config(config_path)


def _progress(console: Console | None):
    # Redirected output gets nothing: the carriage-return refresh below turns into one
    # enormous line in a file or a pipe.
    if console is None or not console.is_terminal:
        return None
    state = {"stage": None}

    def report(stage: str, done: int, total: int) -> None:
        # A cold run is ~4 minutes of Vision work; silence for that long reads as a
        # hang. One line per stage, refreshed, rather than a line per photo.
        if stage != state["stage"] or done % 250 == 0 or done == total:
            state["stage"] = stage
            console.print(f"[dim]{stage}: {done}/{total}[/dim]", end="\r", highlight=False)
            if done == total:
                console.print()

    return report


def _cluster(
    *,
    cfg: ClusterConfig,
    library: str | None,
    records: Iterable[PhotoRecord] | None,
    cache,
    cache_path: str,
    analyzers,
    console: Console | None,
) -> list[Cluster]:
    from .pipeline import run_pipeline

    if records is None:
        db = open_library(library)
        # with_derivatives=True is what bridges a PhotoRecord to pixels, and it is
        # opt-in so `audit` stays pure metadata (`pipeline-seam-is-the-derivative-path`).
        records = iter_photo_records(db, with_derivatives=True)

    handle = nullcontext(cache) if cache is not None else AnalysisCache(cache_path)
    with handle as open_cache:
        return run_pipeline(
            records,
            cache=open_cache,
            config=cfg,
            analyzers=analyzers,
            progress=_progress(console),
        )


def run_cluster(
    *,
    library: str | None = None,
    config_path: str | None = None,
    threshold: float | None = None,
    gap_seconds: float | None = None,
    cache_path: str = DEFAULT_CACHE,
    json_out: str | None = None,
    records: Iterable[PhotoRecord] | None = None,
    cache=None,
    analyzers=None,
    console: Console | None = None,
) -> list[Cluster]:
    """Group the library into ranked near-duplicate clusters.

    `--json-out` writes every cluster, not a sample -- same schema `calibrate` emits,
    so the two files are directly comparable."""
    cfg, source = _resolve_config(config_path, threshold, gap_seconds)
    clusters = _cluster(
        cfg=cfg,
        library=library,
        records=records,
        cache=cache,
        cache_path=cache_path,
        analyzers=analyzers,
        console=console,
    )
    if json_out:
        write_calibration_sample(
            clusters, json_out, config=cfg, source=source, limit=len(clusters)
        )
    return clusters


def run_calibrate(
    *,
    out_dir: str,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    library: str | None = None,
    config_path: str | None = None,
    threshold: float | None = None,
    gap_seconds: float | None = None,
    cache_path: str = DEFAULT_CACHE,
    records: Iterable[PhotoRecord] | None = None,
    cache=None,
    analyzers=None,
    console: Console | None = None,
) -> dict:
    """Write the Task 9 calibration sample: `sample.json` plus a `sample.html` contact
    sheet of the same clusters, side by side, for the owner to actually look at."""
    cfg, source = _resolve_config(config_path, threshold, gap_seconds)
    clusters = _cluster(
        cfg=cfg,
        library=library,
        records=records,
        cache=cache,
        cache_path=cache_path,
        analyzers=analyzers,
        console=console,
    )
    out = Path(out_dir)
    doc = write_calibration_sample(
        clusters, out / "sample.json", config=cfg, source=source, limit=sample_size
    )
    write_contact_sheet(doc, out / "sample.html")
    if console is not None:
        pop = doc["population"]
        console.print(
            f"sampled {pop['sampled']} of {pop['clusters']} clusters "
            f"({pop['culled_photos_sampled']} of {pop['culled_photos']} staged photos)"
        )
        console.print(f"open {out / 'sample.html'}")
    return doc


# --- review --------------------------------------------------------------------


def run_review(
    *,
    library: str | None = None,
    config_path: str | None = None,
    threshold: float | None = None,
    gap_seconds: float | None = None,
    cache_path: str = DEFAULT_CACHE,
    decisions_path: str | None = None,
    demo: bool = False,
    new_session: bool = False,
    allow_write_back: bool = False,
    open_browser: bool = True,
    serve: bool = True,
    records: Iterable[PhotoRecord] | None = None,
    cache=None,
    analyzers=None,
    console: Console | None = None,
    launcher: ModuleType | None = None,
):
    """Scan, then hand the clusters to a local review server on an ephemeral port.

    `--demo` is not a flag that skips a step: it swaps the whole input for the
    synthetic 24-cluster library, so the review UI runs with no Photos library, no
    Full Disk Access and no Vision (`demo-dataset-reproduces-the-measured-rates`).
    Its decisions go to a file inside the demo's own throwaway directory, never to the
    real log — a demo cluster key is indistinguishable from a real one once written,
    and the one artifact this project cannot recompute is the decision log.

    `serve=False` builds everything and returns it without running the server, which
    is how the assembly is tested end to end without a thread.
    """
    review = launcher if launcher is not None else _load_launcher()
    cfg, source = _resolve_config(config_path, threshold, gap_seconds)

    analysed_library: str | None = None
    if demo:
        root = Path(tempfile.mkdtemp(prefix="photocull-demo-"))
        clusters = review.demo_clusters(root)
        db_path = Path(decisions_path) if decisions_path else root / "decisions.db"
    else:
        if records is None:
            db = open_library(library)
            # Read from the opened database rather than from `--library`, which is None
            # for the default library. This is the path write-back compares against
            # whatever Photos.app has open, and a comparison against None refuses.
            analysed_library = getattr(db, "library_path", None)
            records = iter_photo_records(db, with_derivatives=True)
        else:
            analysed_library = library
        # Materialised because the scan is the only place the *library's* uuids exist,
        # and `reconcile` needs them to tell "this photo is gone from the library" from
        # "this photo never clustered" — only 4,498 of 14,235 photos reach a cluster at
        # all, so inferring the library from the clusters would call two thirds of it
        # missing.
        records = list(records)
        clusters = _cluster(
            cfg=cfg,
            library=library,
            records=records,
            cache=cache,
            cache_path=cache_path,
            analyzers=analyzers,
            console=console,
        )
        db_path = Path(decisions_path) if decisions_path else None

    library_uuids = (
        {record.uuid for record in records}
        if not demo
        else {record.uuid for cluster in clusters for record in cluster.records}
    )

    # Deliberately not `with review.DecisionLog(...) as log:`. The log has to outlive
    # this function on the `serve=False` path -- every route reads it on demand -- and
    # a `with` block closes it at the `return`, handing back a `Launch` whose first
    # request dies on "Cannot operate on a closed database". That is not hypothetical:
    # it is what the live smoke hit, one tick after the same module's `check_same_thread`
    # defect (`the-decision-log-is-used-from-another-thread`). The log is opened here
    # and closed here, and `serve=False` hands ownership to the caller in writing.
    log = review.DecisionLog(db_path)
    try:
        launch = review.prepare(
            clusters,
            log=log,
            config=cfg,
            source=source,
            library_uuids=library_uuids,
            force_new_session=new_session,
            writeback=review.WritebackSettings(
                allow_write_back=allow_write_back,
                analysed_library=analysed_library,
            ),
        )
    except review.ResumeBlocked as exc:
        log.close()
        raise SystemExit(f"photocull review: {exc}") from None
    except BaseException:
        log.close()
        raise

    if console is not None:
        report = launch.resume.report()
        console.print(
            f"{len(clusters)} clusters, {report['decided_count']} already decided"
        )
        console.print(f"open {launch.url}")
    if not serve:
        # The caller owns the open log and the bound port: `launch.close()` releases
        # the port, `launch.session.log.close()` the log.
        return launch
    try:
        review.serve(launch, open_browser=open_browser)
    finally:
        launch.close()
        log.close()
    return launch


def _render_writeback_report(report: Any, out: Console) -> None:
    data = report.to_dict()
    counts = data["counts"]
    label = "DRY RUN" if data["dry_run"] else "WRITE-BACK"
    out.print(
        f"[bold]{label}[/bold] — planned {counts['planned']}, applied {counts['applied']}, "
        f"already done {counts['already_done']}, failed {counts['failed']}, "
        f"skipped {counts['skipped']}"
    )
    if data["albums"]:
        out.print(f"albums: {', '.join(data['albums'])}")
    if not report.ok:
        out.print(f"[red]{len(report.failed)} action(s) failed:[/red]")
        for failure in report.failed:
            out.print(f"  {failure.action.photo_uuid} {failure.action.kind} -> {failure.error}")


def run_sweep(
    *,
    library: str | None = None,
    screenshot_age_months: float = 12.0,
    short_video_seconds: float = 5.0,
    records: Iterable[PhotoRecord] | None = None,
    console: Console | None = None,
    now: datetime | None = None,
    dry_run: bool = False,
    allow_write_back: bool = False,
    launcher: ModuleType | None = None,
) -> tuple[SweepGroup, ...]:
    """Report the bulk-disposable categories. Reads nothing but metadata.

    `records` is the same test seam `run_cluster` offers: supply them and no library is
    opened, so the suite never needs a real Photos library.

    `dry_run`/`allow_write_back` route through the review package's own write-back —
    lazily, via `_load_launcher()`, for the same reason `run_review` does: this module
    must stay importable without `photoscript`/`fastapi` installed
    (`test_photocull_never_imports_a_review_dependency`), and neither flag is ever set
    by the tests that prove the plain report path needs no review extra at all."""
    from .sweep import SweepConfig, build_sweep

    out = console or Console()
    analysed_library = library
    if records is None:
        db = open_library(library)
        analysed_library = getattr(db, "library_path", None)
        # with_derivatives=False: the sweep reads no pixels, so it must not pay the
        # ~63s of ImageIO header reads the clustering scan needs.
        records = iter_photo_records(db, with_derivatives=False)
    records = list(records)

    cfg = SweepConfig(
        screenshot_age_months=screenshot_age_months,
        short_video_seconds=short_video_seconds,
    )
    groups = build_sweep(records, config=cfg, now=now or datetime.now(timezone.utc))

    table = Table(title="Sweep candidates")
    table.add_column("Category")
    table.add_column("Album")
    table.add_column("Items", justify="right")
    table.add_column("Why")
    for group in groups:
        table.add_row(group.category, f"Cull/{group.album}", str(len(group)), group.reason)
    out.print(table)
    out.print(
        f"[dim]{sum(len(g) for g in groups)} items would be staged. "
        "Nothing has been written — `sweep` does not touch Photos.[/dim]"
    )

    if dry_run or allow_write_back:
        review = launcher if launcher is not None else _load_launcher()
        live = {r.uuid: r.mod_date for r in records}
        plan = review.plan_sweep(groups, live)
        report = review.run_writeback(
            plan,
            analysed_library=analysed_library,
            targeted_library=review.last_opened_library(),
            dry_run=not allow_write_back,
            # Constructed only on the real-run branch: `PhotoScriptWriter.__init__`
            # launches Photos.app and can block for up to 300s
            # (`photosLibraryWaitForPhotos`) — a dry run must never pay that cost to
            # report what it is not going to do.
            writer=review.PhotoScriptWriter() if allow_write_back else None,
            ledger=review.WritebackLedger(),
            session_id=f"sweep-{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        )
        _render_writeback_report(report, out)

    return groups


def run_albums_list(
    *,
    records: Iterable[PhotoRecord] | None = None,
    keeper_uuids_set: set[str] | None = None,
    library_path: str | None = None,
    config: AlbumConfig | None = None,
    console: Console | None = None,
) -> list[TripGroup]:
    """List trip groups in the keeper pool. Pure report -- no pixels, no export, no
    write. `records`/`keeper_uuids_set` let tests supply synthetic input and assert this
    never opens a library when they do (same pattern as `run_sweep`)."""
    out = console or Console()
    if records is None:
        db = open_library(library_path)
        records = list(iter_photo_records(db, with_derivatives=False))
        keeper_uuids_set = _keeper_uuids(db)
    pool = keeper_pool(records, keeper_uuids_set or set())
    groups = trip_sweep(pool, config or AlbumConfig())

    table = Table(title=f"Trips in the keeper pool ({len(pool)} photos)")
    table.add_column("Title")
    table.add_column("Photos", justify="right")
    table.add_column("Flags")
    for g in groups:
        table.add_row(g.title, str(len(g)), "wide-ranging" if g.wide_spread else "")
    out.print(table)
    return groups


def _default_album_db_path() -> Path:
    """`~/.local/state/photocull/albums.db`, alongside `decisions.db`/`writeback.db` --
    same convention `default_db_path`/`default_ledger_path` already established."""
    return Path.home() / ".local" / "state" / "photocull" / "albums.db"


def run_albums_export(
    *,
    album_id: str,
    out_dir: Path,
    store: "AlbumStore | None" = None,
    library_path: str | None = None,
    bleed_mm: float = 0.0,
    crop_marks: bool = False,
    console: Console | None = None,
) -> "ExportReport":
    from .album_export import export_album
    from .album_store import AlbumStore as _AlbumStore

    out = console or Console()
    store = store or _AlbumStore(_default_album_db_path())
    album = store.get(album_id)
    if album is None:
        raise SystemExit(f"no album with id {album_id!r} — run `photocull albums list` first")

    db = open_library(library_path)
    photos_by_uuid = {p.uuid: p for p in db.photos() if p.uuid in album.sequence}
    report = export_album(album, photos_by_uuid, out_dir, bleed_mm=bleed_mm, crop_marks=crop_marks)

    out.print(f"[green]Exported {report.exported}/{len(album.sequence)}[/green] to {out_dir}")
    if report.failed:
        out.print(f"[red]Failed: {report.failed}[/red]")
    if report.crop_not_applied:
        out.print(
            f"[yellow]Crop not applied (used original, verify manually): {report.crop_not_applied}[/yellow]"
        )
    return report


def run_albums_serve(
    *,
    library_path: str | None = None,
    demo: bool = False,
    config: AlbumConfig | None = None,
    db_path: Path | None = None,
    open_browser: bool = True,
    serve: bool = True,
    records: Iterable[PhotoRecord] | None = None,
    keeper_uuids_set: set[str] | None = None,
    console: Console | None = None,
    launcher: ModuleType | None = None,
):
    """Open the local album-sequencing UI (Task 10's drag-to-sequence frontend).

    `--demo` is not a flag that skips a step: it swaps the whole input for
    `photocull.albums.demo_pool`'s small synthetic two-trip pool, backed by real (if
    tiny) PNGs on disk, so the UI runs with no Photos library and no Full Disk Access
    -- same convention `run_review`'s own `--demo` established. `serve=False` builds
    everything and returns the `Launch` without running the server, which is how the
    assembly is tested end to end without a thread (mirrors `run_review`'s seam).
    """
    album = launcher if launcher is not None else _load_album_launcher()
    out = console or Console()

    if demo:
        root = Path(tempfile.mkdtemp(prefix="photocull-album-demo-"))
        records = demo_pool(root)
        keeper_uuids_set = {r.uuid for r in records}

    return album.launch(
        records=records,
        keeper_uuids_set=keeper_uuids_set,
        library_path=library_path,
        config=config,
        db_path=db_path,
        open_browser=open_browser,
        serve=serve,
        console=out,
    )


# --- parser --------------------------------------------------------------------


def _add_cluster_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--library",
        default=None,
        help="Path to a .photoslibrary; default is the system's last-opened library",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a photocull.toml; default searches ./ then ~/.config/photocull/",
    )
    # Both default to None, not to the shipped value: None means "not overridden", so
    # the config file still wins. Defaulting them here would silently stamp the
    # built-in value over every file setting.
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Complete-linkage cut height; every cluster is guaranteed narrower than this",
    )
    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=None,
        help="Time-bucketing gap in seconds (default: 90)",
    )
    parser.add_argument(
        "--cache", dest="cache_path", default=DEFAULT_CACHE, help="Analysis cache path"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="photocull")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Summarize the Photos library")
    audit.add_argument(
        "--library",
        default=None,
        help="Path to a .photoslibrary; default is the system's last-opened library",
    )
    audit.add_argument(
        "--json-out", default=None, help="Write the summary as JSON to this path"
    )
    audit.add_argument(
        "--gap-seconds",
        type=float,
        default=DEFAULT_GAP_SECONDS,
        help="Near-duplicate cluster gap threshold in seconds (default: 90)",
    )

    cluster = subparsers.add_parser(
        "cluster", help="Group the library into ranked near-duplicate clusters"
    )
    _add_cluster_flags(cluster)
    cluster.add_argument(
        "--json-out", default=None, help="Write every cluster as JSON to this path"
    )

    calibrate = subparsers.add_parser(
        "calibrate", help="Export a review sample for threshold calibration"
    )
    _add_cluster_flags(calibrate)
    calibrate.add_argument(
        "--out", default="out/calibration", help="Directory for sample.json + sample.html"
    )
    calibrate.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Clusters to sample (default: {DEFAULT_SAMPLE_SIZE})",
    )

    review = subparsers.add_parser(
        "review", help="Open the local review UI to pick keepers cluster by cluster"
    )
    _add_cluster_flags(review)
    review.add_argument(
        "--decisions",
        dest="decisions_path",
        default=None,
        help="Decision log path (default: ~/.local/state/photocull/decisions.db)",
    )
    review.add_argument(
        "--demo",
        action="store_true",
        help="Review a synthetic 24-cluster library; opens no Photos library at all",
    )
    review.add_argument(
        "--new-session",
        action="store_true",
        help="Start a fresh session when the settings have moved (keeps old decisions)",
    )
    review.add_argument(
        "--no-browser",
        dest="open_browser",
        action="store_false",
        help="Print the URL instead of opening a browser",
    )
    review.add_argument(
        "--allow-write-back",
        action="store_true",
        help=(
            "Permit this session to write to Photos (favorite / album / keyword only). "
            "Without it, write-back can only be dry-run. Never deletes anything."
        ),
    )

    sweep = subparsers.add_parser(
        "sweep", help="Report bulk-disposable categories (screenshots, short videos, exact duplicates)"
    )
    sweep.add_argument(
        "--library",
        default=None,
        help="Path to a .photoslibrary; default is the system's last-opened library",
    )
    sweep.add_argument(
        "--screenshot-age-months",
        type=float,
        default=12.0,
        help="Only sweep screenshots older than this many months (default: 12)",
    )
    sweep.add_argument(
        "--short-video-seconds",
        type=float,
        default=5.0,
        help="Only sweep videos shorter than this many seconds (default: 5)",
    )
    sweep.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the write-back and report it without touching Photos",
    )
    sweep.add_argument(
        "--allow-write-back",
        action="store_true",
        help=(
            "Permit this run to modify your Photos library. Adds photos to albums under "
            "Cull/ and sets the cull-candidate keyword. Never deletes anything."
        ),
    )

    albums = subparsers.add_parser("albums", help="Build print-ready albums from the keeper pool")
    albums_sub = albums.add_subparsers(dest="albums_command", required=True)
    albums_list = albums_sub.add_parser("list", help="List trip groups in the keeper pool")
    albums_list.add_argument(
        "--library",
        default=None,
        help="Path to a .photoslibrary; default is the system's last-opened library",
    )
    albums_export = albums_sub.add_parser(
        "export", help="Export a sequenced album to JPEGs + a print-ready PDF"
    )
    albums_export.add_argument("--album-id", required=True, help="Album id from `photocull albums list`")
    albums_export.add_argument(
        "--out", dest="out_dir", default="out/albums/", help="Output directory (default: out/albums/)"
    )
    albums_export.add_argument(
        "--library",
        default=None,
        help="Path to a .photoslibrary; default is the system's last-opened library",
    )
    albums_export.add_argument(
        "--bleed-mm", type=float, default=0.0, help="Bleed margin in millimeters (default: 0.0)"
    )
    albums_export.add_argument(
        "--crop-marks", action="store_true", help="Draw print-shop crop marks around each page"
    )
    albums_serve = albums_sub.add_parser(
        "serve", help="Open the local album-sequencing UI (drag to reorder, adjust crops)"
    )
    albums_serve.add_argument(
        "--library",
        default=None,
        help="Path to a .photoslibrary; default is the system's last-opened library",
    )
    albums_serve.add_argument(
        "--demo",
        action="store_true",
        help="Sequence a small synthetic two-trip pool; opens no Photos library at all",
    )
    albums_serve.add_argument(
        "--no-browser",
        dest="open_browser",
        action="store_false",
        help="Print the URL instead of opening a browser",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "audit":
        run_audit(
            library=args.library,
            json_out=args.json_out,
            gap_seconds=args.gap_seconds,
        )
    elif args.command == "cluster":
        run_cluster(
            library=args.library,
            config_path=args.config,
            threshold=args.threshold,
            gap_seconds=args.gap_seconds,
            cache_path=args.cache_path,
            json_out=args.json_out,
            console=Console(),
        )
    elif args.command == "calibrate":
        run_calibrate(
            out_dir=args.out,
            sample_size=args.sample_size,
            library=args.library,
            config_path=args.config,
            threshold=args.threshold,
            gap_seconds=args.gap_seconds,
            cache_path=args.cache_path,
            console=Console(),
        )
    elif args.command == "review":
        try:
            run_review(
                library=args.library,
                config_path=args.config,
                threshold=args.threshold,
                gap_seconds=args.gap_seconds,
                cache_path=args.cache_path,
                decisions_path=args.decisions_path,
                demo=args.demo,
                new_session=args.new_session,
                allow_write_back=args.allow_write_back,
                open_browser=args.open_browser,
                console=Console(),
            )
        except ReviewUnavailable as exc:
            raise SystemExit(str(exc)) from None
    elif args.command == "albums":
        if args.albums_command == "list":
            run_albums_list(library_path=args.library, console=Console())
        elif args.albums_command == "export":
            run_albums_export(
                album_id=args.album_id,
                out_dir=Path(args.out_dir),
                library_path=args.library,
                bleed_mm=args.bleed_mm,
                crop_marks=args.crop_marks,
                console=Console(),
            )
        elif args.albums_command == "serve":
            try:
                run_albums_serve(
                    library_path=args.library,
                    demo=args.demo,
                    open_browser=args.open_browser,
                    console=Console(),
                )
            except AlbumUIUnavailable as exc:
                raise SystemExit(str(exc)) from None
    elif args.command == "sweep":
        try:
            run_sweep(
                library=args.library,
                screenshot_age_months=args.screenshot_age_months,
                short_video_seconds=args.short_video_seconds,
                dry_run=args.dry_run,
                allow_write_back=args.allow_write_back,
                console=Console(),
            )
        except ReviewUnavailable as exc:
            raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
