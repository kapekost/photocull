import io
import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from rich.console import Console

from photocull.cli import (
    ALBUM_EXTRA_HINT,
    REVIEW_EXTRA_HINT,
    AlbumUIUnavailable,
    ReviewUnavailable,
    _load_album_launcher,
    _load_launcher,
    build_parser,
    main,
    run_albums_export,
    run_albums_list,
    run_albums_serve,
    run_audit,
    run_calibrate,
    run_cluster,
    run_review,
    run_sweep,
)
from photocull.config_io import ConfigError

from .fixtures import make_photo_record


def test_build_parser_audit_defaults():
    parser = build_parser()
    args = parser.parse_args(["audit"])
    assert args.command == "audit"
    assert args.library is None
    assert args.json_out is None
    assert args.gap_seconds == 90.0


def test_build_parser_audit_custom_args():
    parser = build_parser()
    args = parser.parse_args(
        [
            "audit",
            "--library",
            "/some/lib.photoslibrary",
            "--json-out",
            "/tmp/out.json",
            "--gap-seconds",
            "30",
        ]
    )
    assert args.library == "/some/lib.photoslibrary"
    assert args.json_out == "/tmp/out.json"
    assert args.gap_seconds == 30.0


def test_run_audit_with_injected_records_writes_json(tmp_path):
    records = [
        make_photo_record(date=datetime(2025, 1, 1), filesize=100, is_screenshot=True),
        make_photo_record(date=datetime(2025, 1, 1), filesize=200, is_video=True, duration_seconds=3),
    ]
    out_path = tmp_path / "audit.json"
    summary = run_audit(
        library="/fake/lib.photoslibrary",
        json_out=str(out_path),
        gap_seconds=90.0,
        records=records,
    )
    assert summary.total_items == 2
    assert summary.library_path == "/fake/lib.photoslibrary"
    with open(out_path) as f:
        data = json.load(f)
    assert data["total_items"] == 2


def test_run_audit_without_json_out_does_not_write_a_file(tmp_path):
    records = [make_photo_record()]
    run_audit(records=records, json_out=None)
    assert list(tmp_path.iterdir()) == []


def test_main_dispatches_audit_command_to_run_audit(monkeypatch, tmp_path):
    captured = {}

    def fake_run_audit(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("photocull.cli.run_audit", fake_run_audit)
    out_path = str(tmp_path / "out.json")
    main(["audit", "--json-out", out_path, "--gap-seconds", "45"])
    assert captured["json_out"] == out_path
    assert captured["gap_seconds"] == 45.0
    assert captured["library"] is None


# --- Task 8b: `cluster` and `calibrate` ----------------------------------------
#
# These are the first subcommands that read `photocull.toml`. Until now nothing in
# the codebase imported `config_io` at all, so the precedence chain
# (defaults <- file <- CLI flags) had unit tests but no production caller.


def _burst(uuid_prefix, n=2, second=0):
    """n records close enough in time to share a bucket."""
    return [
        make_photo_record(
            uuid=f"{uuid_prefix}{i}",
            date=datetime(2025, 1, 1, 12, 0, 0) + timedelta(seconds=second + i),
            derivative_path=f"/deriv/{uuid_prefix}{i}.jpeg",
        )
        for i in range(n)
    ]


class _FakeCache:
    """Stands in for AnalysisCache: no SQLite file, nothing cached."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_feature_print(self, *a):
        return None

    def put_feature_print(self, *a):
        pass

    def get_scores(self, *a):
        return None

    def put_scores(self, *a):
        pass


def _analyzers(vectors):
    from photocull.pipeline import Analyzers

    return Analyzers(
        printer=lambda path: vectors.get(path),
        stats=lambda path: None,
        faces=lambda path: [],
        horizon=lambda path: None,
    )


def _identical_pair():
    records = _burst("a", 2)
    vectors = {r.derivative_path: [0.0, 0.0, 1.0] for r in records}
    return records, _analyzers(vectors)


def _two_clusters_apart_in_time():
    """Two bursts far enough apart to bucket (and cluster) separately.

    `second=100_000` is well past `gap_seconds`' default of 90, so `b` lands in its own
    time bucket rather than merging with `a` -- the two clusters this needs to tell
    "newest first" from "oldest first" apart."""
    early = _burst("a", 2, second=0)
    late = _burst("b", 2, second=100_000)
    records = early + late
    vectors = {r.derivative_path: [0.0, 0.0, 1.0] for r in records}
    return records, _analyzers(vectors)


def _no_config(tmp_path):
    """An empty TOML, i.e. pure shipped defaults.

    Not the same as passing nothing: `find_config()` searches ./photocull.toml, which
    exists in this repo and is git-ignored, so a test relying on discovery would start
    testing the owner's own calibration the moment Task 9 writes values into it. That
    file is entirely commented out today, which is exactly why the dependency would go
    unnoticed until it mattered."""
    path = tmp_path / "empty.toml"
    path.write_text("")
    return str(path)


def test_build_parser_cluster_defaults():
    args = build_parser().parse_args(["cluster"])
    assert args.command == "cluster"
    assert args.library is None
    assert args.config is None
    assert args.threshold is None
    assert args.gap_seconds is None
    assert args.json_out is None


def test_build_parser_calibrate_args(tmp_path):
    args = build_parser().parse_args(
        ["calibrate", "--out", str(tmp_path), "--sample-size", "50"]
    )
    assert args.command == "calibrate"
    assert args.out == str(tmp_path)
    assert args.sample_size == 50


def test_run_cluster_with_injected_records_groups_them(tmp_path):
    records, analyzers = _identical_pair()
    clusters = run_cluster(
        records=records,
        cache=_FakeCache(),
        analyzers=analyzers,
        config_path=_no_config(tmp_path),
    )
    assert len(clusters) == 1
    assert len(clusters[0].records) == 2


def test_run_cluster_writes_json_carrying_every_cluster(tmp_path):
    records, analyzers = _identical_pair()
    out = tmp_path / "clusters.json"
    run_cluster(
        records=records,
        cache=_FakeCache(),
        analyzers=analyzers,
        json_out=str(out),
        config_path=_no_config(tmp_path),
    )
    doc = json.loads(out.read_text())
    # `cluster` exports everything, unlike `calibrate` which samples.
    assert doc["population"]["clusters"] == 1
    assert doc["population"]["sampled"] == 1
    assert len(doc["clusters"]) == 1


def test_cli_threshold_flag_overrides_the_shipped_default(tmp_path):
    """The precedence chain's last link, exercised through the CLI for the first time.
    At a cut of 0.0 nothing can merge, so the flag is provably doing something rather
    than being accepted and dropped."""
    records, analyzers = _identical_pair()
    clusters = run_cluster(
        records=records,
        cache=_FakeCache(),
        analyzers=analyzers,
        threshold=0.0,
        config_path=_no_config(tmp_path),
    )
    assert clusters == []


def test_cli_reads_the_config_file(tmp_path):
    cfg = tmp_path / "photocull.toml"
    cfg.write_text("similarity_threshold = 0.0\n")
    records, analyzers = _identical_pair()
    clusters = run_cluster(
        records=records,
        cache=_FakeCache(),
        analyzers=analyzers,
        config_path=str(cfg),
    )
    assert clusters == []


def test_cli_flag_beats_the_config_file(tmp_path):
    """`config-is-external-defaults-are-generic` pins this order; without a production
    caller it had never been exercised end to end."""
    cfg = tmp_path / "photocull.toml"
    cfg.write_text("similarity_threshold = 0.0\n")
    records, analyzers = _identical_pair()
    clusters = run_cluster(
        records=records,
        cache=_FakeCache(),
        analyzers=analyzers,
        config_path=str(cfg),
        threshold=0.9,
    )
    assert len(clusters) == 1


def test_cli_rejects_a_bad_config_loudly(tmp_path):
    """`config-validation-covers-every-field`: a mistyped threshold that silently
    changed nothing would corrupt a calibration run with no visible symptom."""
    cfg = tmp_path / "photocull.toml"
    cfg.write_text("similarity_threshold = 'not a number'\n")
    records, analyzers = _identical_pair()
    with pytest.raises(ConfigError):
        run_cluster(
            records=records,
            cache=_FakeCache(),
            analyzers=analyzers,
            config_path=str(cfg),
        )


def test_run_calibrate_writes_both_the_sample_and_the_contact_sheet(tmp_path):
    records, analyzers = _identical_pair()
    out_dir = tmp_path / "calibration"
    run_calibrate(
        records=records,
        cache=_FakeCache(),
        analyzers=analyzers,
        out_dir=str(out_dir),
        sample_size=10,
        config_path=_no_config(tmp_path),
    )
    assert (out_dir / "sample.json").is_file()
    assert (out_dir / "sample.html").is_file()
    doc = json.loads((out_dir / "sample.json").read_text())
    assert doc["population"]["clusters"] == 1
    assert doc["config"]["source"].endswith("empty.toml")


def test_run_calibrate_honours_the_sample_size(tmp_path):
    records = []
    vectors = {}
    for group in range(4):
        pair = _burst(f"g{group}_", 2, second=group * 600)
        records.extend(pair)
        for r in pair:
            vectors[r.derivative_path] = [float(group), 0.0, 1.0]
    out_dir = tmp_path / "calibration"
    run_calibrate(
        records=records,
        cache=_FakeCache(),
        analyzers=_analyzers(vectors),
        out_dir=str(out_dir),
        sample_size=2,
        config_path=_no_config(tmp_path),
    )
    doc = json.loads((out_dir / "sample.json").read_text())
    assert doc["population"]["clusters"] == 4
    assert doc["population"]["sampled"] == 2
    assert len(doc["clusters"]) == 2


def test_main_dispatches_cluster_command(monkeypatch):
    captured = {}
    monkeypatch.setattr("photocull.cli.run_cluster", lambda **kw: captured.update(kw))
    main(["cluster", "--threshold", "0.25", "--cache", "/tmp/c.db"])
    assert captured["threshold"] == 0.25
    assert captured["cache_path"] == "/tmp/c.db"


def test_main_dispatches_calibrate_command(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr("photocull.cli.run_calibrate", lambda **kw: captured.update(kw))
    main(["calibrate", "--out", str(tmp_path), "--sample-size", "25"])
    assert captured["out_dir"] == str(tmp_path)
    assert captured["sample_size"] == 25


# --- review --------------------------------------------------------------------
#
# These run on a machine with no `[review]` extra, so nothing here may import
# `photocull_review`. The launcher arrives through the same seam production uses --
# `run_review(launcher=...)` -- which means these tests pin the *wiring* and
# `tests/review/test_launcher.py` pins the launcher itself. The split is deliberate:
# a review test that needed fastapi could not assert the CLI works without it.


class _FakeLog:
    def __init__(self, path=None):
        self.path = path
        self.closed = False

    def close(self):
        self.closed = True


class _FakeResumeBlocked(RuntimeError):
    pass


class _FakeLauncher:
    """A stand-in with the names `run_review` and `run_sweep` reach for, and no others.

    "And no others" is load-bearing rather than tidy: `photocull.cli` reaches the whole
    review package through one lazily-imported module, so this fake going red is how a
    new name added to that surface announces itself. It went red for `WritebackSettings`
    in Task 14, and for the write-back names below in Phase 2 Task 4."""

    ResumeBlocked = _FakeResumeBlocked

    def __init__(self, *, blocked: bool = False):
        self.blocked = blocked
        self.logs: list[_FakeLog] = []
        self.prepared: list[dict] = []
        self.served: list[dict] = []
        self.writeback_settings: list[dict] = []
        self.plan_sweep_calls: list[dict] = []
        self.writeback_calls: list[dict] = []
        self.writer_built = False
        self.ledger_built = False
        self.last_opened_library_value: str | None = "/fake/Last.photoslibrary"
        #: What `run_writeback` returns to the caller. A test overwrites this to shape
        #: the report `run_sweep` renders (failures, counts, albums).
        self.writeback_report = SimpleNamespace(
            ok=True,
            failed=(),
            to_dict=lambda: {
                "dry_run": True,
                "counts": {
                    "planned": 0,
                    "applied": 0,
                    "already_done": 0,
                    "failed": 0,
                    "skipped": 0,
                },
                "albums": [],
            },
        )

    def demo_clusters(self, root):
        self.demo_root = Path(root)
        return [_demo_cluster()]

    def DecisionLog(self, path=None):  # noqa: N802 - mirrors the real class's name
        log = _FakeLog(path)
        self.logs.append(log)
        return log

    def WritebackSettings(self, **kwargs):  # noqa: N802 - mirrors the real class's name
        self.writeback_settings.append(kwargs)
        return SimpleNamespace(**kwargs)

    def prepare(self, clusters, **kwargs):
        if self.blocked:
            raise _FakeResumeBlocked("similarity_threshold 0.48 -> 0.52")
        self.prepared.append({"clusters": clusters, **kwargs})
        return SimpleNamespace(
            url="http://127.0.0.1:51234/?t=tok",
            resume=SimpleNamespace(report=lambda: {"decided_count": 0}),
            close=lambda: None,
        )

    def serve(self, launch, **kwargs):
        self.served.append({"launch": launch, **kwargs})

    def plan_sweep(self, groups, live):
        self.plan_sweep_calls.append({"groups": groups, "live": live})
        return SimpleNamespace(actions=(), skipped=(), albums=(), keepers=(), staged=(), favorites=())

    def run_writeback(self, plan, **kwargs):
        self.writeback_calls.append({"plan": plan, **kwargs})
        return self.writeback_report

    def PhotoScriptWriter(self):  # noqa: N802 - mirrors the real class's name
        self.writer_built = True
        return object()

    def WritebackLedger(self):  # noqa: N802 - mirrors the real class's name
        self.ledger_built = True
        return object()

    def last_opened_library(self):
        return self.last_opened_library_value


def _demo_cluster():
    from photocull.models import Cluster

    records = [make_photo_record(uuid="demo-1"), make_photo_record(uuid="demo-2")]
    return Cluster(records=records, scores=[], winner_uuid="demo-1", is_ambiguous=False)


def test_build_parser_review_defaults():
    parser = build_parser()
    args = parser.parse_args(["review"])
    assert args.command == "review"
    assert args.decisions_path is None
    assert args.demo is False
    assert args.new_session is False
    assert args.newest_first is False
    assert args.open_browser is True
    # The precedence chain must reach `review` too, or a calibrated threshold would
    # apply to `cluster` and silently not to the UI built on it.
    assert args.threshold is None
    assert args.gap_seconds is None


def test_build_parser_review_custom_args(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "review",
            "--demo",
            "--no-browser",
            "--new-session",
            "--newest-first",
            "--decisions",
            str(tmp_path / "d.db"),
            "--threshold",
            "0.52",
        ]
    )
    assert args.demo is True
    assert args.open_browser is False
    assert args.new_session is True
    assert args.newest_first is True
    assert args.decisions_path == str(tmp_path / "d.db")
    assert args.threshold == 0.52


def test_review_demo_never_opens_a_photos_library(monkeypatch, tmp_path):
    """`--demo` must be runnable with no Photos library and no Full Disk Access."""

    def explode(*args, **kwargs):
        raise AssertionError("--demo opened the Photos library")

    monkeypatch.setattr("photocull.cli.open_library", explode)
    fake = _FakeLauncher()

    run_review(
        demo=True,
        serve=False,
        launcher=fake,
        config_path=_no_config(tmp_path),
    )

    assert len(fake.prepared) == 1


def test_review_demo_keeps_its_decisions_out_of_the_real_log(monkeypatch, tmp_path):
    """A demo decision is indistinguishable from a real one once it is written."""
    monkeypatch.setattr(
        "photocull.cli.open_library",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("opened the library")),
    )
    fake = _FakeLauncher()

    run_review(demo=True, serve=False, launcher=fake, config_path=_no_config(tmp_path))

    (log,) = fake.logs
    assert log.path is not None, "the demo fell through to the default decision log"
    assert fake.demo_root in Path(log.path).parents


def test_review_demo_honours_an_explicit_decisions_path(tmp_path):
    fake = _FakeLauncher()
    wanted = tmp_path / "demo-decisions.db"

    run_review(
        demo=True,
        serve=False,
        launcher=fake,
        decisions_path=str(wanted),
        config_path=_no_config(tmp_path),
    )

    assert fake.logs[0].path == wanted


def test_review_passes_every_library_uuid_not_only_the_clustered_ones(tmp_path):
    """`reconcile` cannot tell a deleted photo from a never-clustered one otherwise."""
    records, analyzers = _identical_pair()
    records = list(records) + [make_photo_record(uuid="lonely")]
    fake = _FakeLauncher()

    run_review(
        records=records,
        cache=_FakeCache(),
        analyzers=analyzers,
        serve=False,
        launcher=fake,
        config_path=_no_config(tmp_path),
    )

    passed = fake.prepared[0]["library_uuids"]
    assert "lonely" in passed
    assert len(passed) == 3


def test_review_defaults_to_the_launcher_s_own_decision_log(tmp_path):
    """No `--decisions` means None, so `DecisionLog` applies `default_db_path()`.

    Passing a path computed here would duplicate that default in a second place and
    let the two drift.
    """
    records, analyzers = _identical_pair()
    fake = _FakeLauncher()

    run_review(
        records=records,
        cache=_FakeCache(),
        analyzers=analyzers,
        serve=False,
        launcher=fake,
        config_path=_no_config(tmp_path),
    )

    assert fake.logs[0].path is None


def test_review_defaults_to_oldest_capture_first(tmp_path):
    records, analyzers = _two_clusters_apart_in_time()
    fake = _FakeLauncher()

    run_review(
        records=records,
        cache=_FakeCache(),
        analyzers=analyzers,
        serve=False,
        launcher=fake,
        config_path=_no_config(tmp_path),
    )

    clusters = fake.prepared[0]["clusters"]
    assert len(clusters) == 2
    assert min(r.date for r in clusters[0].records) < min(r.date for r in clusters[1].records)


def test_review_newest_first_reverses_capture_order(tmp_path):
    records, analyzers = _two_clusters_apart_in_time()
    fake = _FakeLauncher()

    run_review(
        records=records,
        cache=_FakeCache(),
        analyzers=analyzers,
        newest_first=True,
        serve=False,
        launcher=fake,
        config_path=_no_config(tmp_path),
    )

    clusters = fake.prepared[0]["clusters"]
    assert len(clusters) == 2
    assert min(r.date for r in clusters[0].records) > min(r.date for r in clusters[1].records)


def test_a_blocked_resume_exits_with_the_refusal(tmp_path):
    fake = _FakeLauncher(blocked=True)

    with pytest.raises(SystemExit) as caught:
        run_review(
            demo=True, serve=False, launcher=fake, config_path=_no_config(tmp_path)
        )

    assert "similarity_threshold" in str(caught.value)


def test_a_missing_review_extra_names_the_install_command():
    def importer(name):
        raise ModuleNotFoundError("No module named 'fastapi'", name="fastapi")

    with pytest.raises(ReviewUnavailable) as caught:
        _load_launcher(importer)

    assert 'pip install -e ".[review]"' in str(caught.value)


def test_an_unrelated_missing_module_is_not_blamed_on_the_extra():
    """Reporting our own bug as a missing extra sends the reader to fix the wrong thing."""

    def importer(name):
        raise ModuleNotFoundError("No module named 'typo'", name="typo")

    with pytest.raises(ModuleNotFoundError) as caught:
        _load_launcher(importer)

    assert not isinstance(caught.value, ReviewUnavailable)


def test_load_launcher_finds_the_real_module_when_the_extra_is_installed():
    pytest.importorskip("fastapi", reason="needs the [review] extra")

    module = _load_launcher()

    assert module.__name__ == "photocull_review.launcher"


def test_main_dispatches_review_command(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "photocull.cli.run_review", lambda **kwargs: seen.update(kwargs)
    )

    main(["review", "--demo", "--no-browser"])

    assert seen["demo"] is True
    assert seen["open_browser"] is False


def test_main_turns_a_missing_extra_into_a_message_not_a_traceback(monkeypatch):
    def explode(**kwargs):
        raise ReviewUnavailable(REVIEW_EXTRA_HINT)

    monkeypatch.setattr("photocull.cli.run_review", explode)

    with pytest.raises(SystemExit) as caught:
        main(["review"])

    assert 'pip install -e ".[review]"' in str(caught.value)


def test_review_forwards_new_session_to_the_launcher(tmp_path):
    """A flag that is parsed and then dropped is worse than one that is absent."""
    fake = _FakeLauncher()

    run_review(
        demo=True,
        new_session=True,
        serve=False,
        launcher=fake,
        config_path=_no_config(tmp_path),
    )

    assert fake.prepared[0]["force_new_session"] is True


def test_review_forwards_the_write_back_permission_to_the_session(tmp_path):
    """`--allow-write-back` is one of the two gates on the only code that mutates the
    owner's library, and a flag parsed and then dropped is worse than one that is
    absent — it reads as a safeguard while being none."""
    fake = _FakeLauncher()
    run_review(
        demo=True, serve=False, launcher=fake, config_path=_no_config(tmp_path)
    )
    assert fake.writeback_settings[0]["allow_write_back"] is False

    allowed = _FakeLauncher()
    run_review(
        demo=True,
        allow_write_back=True,
        serve=False,
        launcher=allowed,
        config_path=_no_config(tmp_path),
    )
    assert allowed.writeback_settings[0]["allow_write_back"] is True


def test_a_demo_review_can_never_name_a_library_to_write_to(tmp_path):
    """The demo opens no Photos library, so there is nothing for write-back to compare
    against what Photos has open — and an unidentified analysed library is refused
    rather than assumed. A demo therefore cannot write back, structurally."""
    fake = _FakeLauncher()
    run_review(
        demo=True,
        allow_write_back=True,
        serve=False,
        launcher=fake,
        config_path=_no_config(tmp_path),
    )
    assert fake.writeback_settings[0]["analysed_library"] is None


def test_the_analysed_library_reaches_write_back(tmp_path):
    """What write-back compares against Photos' last-opened library. Taken from the
    opened database rather than from `--library`, which is None for the default one."""
    fake = _FakeLauncher()
    run_review(
        library="/Volumes/Photos/Some.photoslibrary",
        records=[make_photo_record(uuid="u1"), make_photo_record(uuid="u2")],
        serve=False,
        launcher=fake,
        cache=_FakeCache(),
        config_path=_no_config(tmp_path),
    )
    assert (
        fake.writeback_settings[0]["analysed_library"]
        == "/Volumes/Photos/Some.photoslibrary"
    )


def test_review_forwards_no_browser_to_serve(tmp_path):
    fake = _FakeLauncher()

    run_review(
        demo=True,
        open_browser=False,
        launcher=fake,
        config_path=_no_config(tmp_path),
    )

    assert fake.served[0]["open_browser"] is False


def test_serving_closes_the_decision_log_and_the_port(tmp_path):
    """The log outlives `run_review` only on the `serve=False` path.

    Opening it in a `with` block instead closed it at the `return`, so every route
    answered "Cannot operate on a closed database" on the first request -- caught by
    the live smoke rather than by this suite, which is why both halves are pinned.
    """
    fake = _FakeLauncher()

    run_review(demo=True, launcher=fake, config_path=_no_config(tmp_path))

    assert fake.logs[0].closed is True


def test_not_serving_hands_back_an_open_log(tmp_path):
    fake = _FakeLauncher()

    run_review(demo=True, serve=False, launcher=fake, config_path=_no_config(tmp_path))

    assert fake.logs[0].closed is False


def test_a_blocked_resume_closes_the_log_it_opened(tmp_path):
    fake = _FakeLauncher(blocked=True)

    with pytest.raises(SystemExit):
        run_review(demo=True, launcher=fake, config_path=_no_config(tmp_path))

    assert fake.logs[0].closed is True


def test_sweep_parses_its_own_defaults():
    args = build_parser().parse_args(["sweep"])
    assert args.command == "sweep"
    assert args.screenshot_age_months == 12.0
    assert args.short_video_seconds == 5.0


def test_sweep_accepts_overrides():
    args = build_parser().parse_args(
        ["sweep", "--screenshot-age-months", "3", "--short-video-seconds", "2"]
    )
    assert args.screenshot_age_months == 3.0
    assert args.short_video_seconds == 2.0


def test_run_sweep_reports_every_category_from_supplied_records():
    now = datetime(2026, 8, 17, 12, 0, 0)
    shot = make_photo_record(is_screenshot=True, date=datetime(2020, 1, 1))
    clip = make_photo_record(is_video=True, duration_seconds=1.0)
    groups = run_sweep(records=[shot, clip], now=now, console=Console(file=io.StringIO()))

    counts = {g.category: len(g) for g in groups}
    assert counts == {"screenshots": 1, "short-videos": 1, "exact-duplicates": 0}


def test_run_sweep_never_opens_a_library_when_records_are_supplied():
    """The same seam `run_cluster` uses: supplying records must not touch osxphotos,
    so the suite runs with no Photos library present."""
    with mock.patch("photocull.cli.open_library") as opened:
        run_sweep(
            records=[make_photo_record(is_screenshot=True, date=datetime(2020, 1, 1))],
            now=datetime(2026, 8, 17),
            console=Console(file=io.StringIO()),
        )
    opened.assert_not_called()


def test_sweep_parses_the_writeback_flags():
    args = build_parser().parse_args(["sweep"])
    assert args.dry_run is False
    assert args.allow_write_back is False

    args = build_parser().parse_args(["sweep", "--dry-run"])
    assert args.dry_run is True

    args = build_parser().parse_args(["sweep", "--allow-write-back"])
    assert args.allow_write_back is True


def test_run_sweep_plain_report_never_touches_the_launcher():
    """No `dry_run`/`allow_write_back`: the existing report-only path, unchanged.

    A `launcher` that raises on any attribute access proves this — the two Task 2
    tests above already prove `open_library` is untouched when records are supplied,
    this proves the review package isn't reached for either."""

    class _ExplodingLauncher:
        def __getattr__(self, name):
            raise AssertionError(f"run_sweep reached for {name!r} without dry_run/allow_write_back")

    groups = run_sweep(
        records=[make_photo_record(is_screenshot=True, date=datetime(2020, 1, 1))],
        now=datetime(2026, 8, 17),
        console=Console(file=io.StringIO()),
        launcher=_ExplodingLauncher(),
    )
    assert sum(len(g) for g in groups) == 1


def test_run_sweep_dry_run_never_constructs_a_writer():
    fake = _FakeLauncher()
    record = make_photo_record(is_screenshot=True, date=datetime(2020, 1, 1))

    run_sweep(
        records=[record],
        now=datetime(2026, 8, 17),
        console=Console(file=io.StringIO()),
        dry_run=True,
        launcher=fake,
    )

    assert fake.writer_built is False
    assert fake.ledger_built is True
    assert len(fake.writeback_calls) == 1
    assert fake.writeback_calls[0]["dry_run"] is True
    assert fake.writeback_calls[0]["writer"] is None
    # plan_sweep got a live mod-date map keyed by the same uuid the record carries.
    assert record.uuid in fake.plan_sweep_calls[0]["live"]


def test_run_sweep_allow_write_back_constructs_a_writer():
    fake = _FakeLauncher()

    run_sweep(
        records=[make_photo_record(is_screenshot=True, date=datetime(2020, 1, 1))],
        now=datetime(2026, 8, 17),
        console=Console(file=io.StringIO()),
        allow_write_back=True,
        launcher=fake,
    )

    assert fake.writer_built is True
    assert fake.ledger_built is True
    assert fake.writeback_calls[0]["dry_run"] is False
    assert fake.writeback_calls[0]["writer"] is not None


def test_run_sweep_reports_failures_from_the_writeback():
    fake = _FakeLauncher()
    fake.writeback_report = SimpleNamespace(
        ok=False,
        failed=(
            SimpleNamespace(
                action=SimpleNamespace(photo_uuid="ABCD-1234", kind="album"),
                error="ValueError: Invalid photo id: ABCD-1234",
            ),
        ),
        to_dict=lambda: {
            "dry_run": False,
            "counts": {"planned": 2, "applied": 1, "already_done": 0, "failed": 1, "skipped": 0},
            "albums": ["Cull/Screenshots"],
        },
    )
    out = io.StringIO()

    run_sweep(
        records=[make_photo_record(is_screenshot=True, date=datetime(2020, 1, 1))],
        now=datetime(2026, 8, 17),
        console=Console(file=out),
        allow_write_back=True,
        launcher=fake,
    )

    printed = out.getvalue()
    assert "failed 1" in printed
    assert "ABCD-1234" in printed
    assert "Invalid photo id" in printed


def test_main_dispatches_sweep_writeback_flags(monkeypatch):
    seen = {}
    monkeypatch.setattr("photocull.cli.run_sweep", lambda **kwargs: seen.update(kwargs) or ())

    main(["sweep", "--allow-write-back"])

    assert seen["allow_write_back"] is True
    assert seen["dry_run"] is False


def test_main_turns_a_missing_extra_into_a_message_for_sweep_too(monkeypatch):
    def explode(**kwargs):
        raise ReviewUnavailable(REVIEW_EXTRA_HINT)

    monkeypatch.setattr("photocull.cli.run_sweep", explode)

    with pytest.raises(SystemExit) as caught:
        main(["sweep", "--dry-run"])

    assert 'pip install -e ".[review]"' in str(caught.value)


def test_run_albums_list_never_opens_a_library_when_records_are_supplied(monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("must not open a library")

    monkeypatch.setattr("photocull.cli.open_library", _boom)
    records = [make_photo_record(uuid="a", is_favorite=True)]
    groups = run_albums_list(
        records=records, keeper_uuids_set=set(), console=Console(file=io.StringIO())
    )
    assert groups == []  # 1 favorite, but trip_min_size default is 3 -- no group


def test_run_albums_list_reports_trips_above_the_minimum():
    records = [
        make_photo_record(uuid=f"u{i}", date=datetime(2026, 3, 15, 9, i), is_favorite=False)
        for i in range(3)
    ]
    groups = run_albums_list(
        records=records,
        keeper_uuids_set={"u0", "u1", "u2"},
        console=Console(file=io.StringIO()),
    )
    assert len(groups) == 1
    assert len(groups[0]) == 3


class _FakeAlbumPhoto:
    def __init__(self, uuid, width=1800, height=1200):
        self.uuid = uuid
        self.hasadjustments = False
        self._width = width
        self._height = height

    def export(self, dest, filename=None, edited=False, use_photos_export=False):
        from tests.render_helpers import make_test_jpeg

        path = Path(dest) / f"{self.uuid}.jpeg"
        make_test_jpeg(path, self._width, self._height)
        return [str(path)]


class _FakeAlbumDB:
    def __init__(self, photos):
        self._photos = photos

    def photos(self):
        return self._photos


def test_run_albums_export_writes_files_and_reports_progress(tmp_path, monkeypatch):
    from photocull.album_store import AlbumStore

    store = AlbumStore(tmp_path / "albums.db")
    album = store.create("Test Trip", "4x6", ["a", "b"])
    fake_db = _FakeAlbumDB([_FakeAlbumPhoto("a"), _FakeAlbumPhoto("b"), _FakeAlbumPhoto("c")])
    monkeypatch.setattr("photocull.cli.open_library", lambda library_path: fake_db)

    out = io.StringIO()
    report = run_albums_export(
        album_id=album.album_id,
        out_dir=tmp_path / "out",
        store=store,
        console=Console(file=out),
    )

    assert report.exported == 2
    assert (tmp_path / "out" / "a.jpeg").exists()
    assert (tmp_path / "out" / "b.jpeg").exists()
    assert "Exported 2/2" in out.getvalue()


def test_run_albums_export_raises_when_the_album_id_is_unknown(tmp_path):
    from photocull.album_store import AlbumStore

    store = AlbumStore(tmp_path / "albums.db")
    with pytest.raises(SystemExit):
        run_albums_export(
            album_id="not-a-real-id",
            out_dir=tmp_path / "out",
            store=store,
            console=Console(file=io.StringIO()),
        )


# --- albums serve (Task 10) ------------------------------------------------------
#
# Same split `run_review`'s own tests use: the launcher arrives through a seam
# (`run_albums_serve(launcher=...)`), so these tests pin the CLI wiring without
# needing fastapi installed; `tests/album/test_launcher.py` pins the launcher itself.


class _FakeAlbumLauncher:
    def __init__(self):
        self.launch_calls: list[dict] = []

    def launch(self, **kwargs):
        self.launch_calls.append(kwargs)
        return SimpleNamespace(url="http://127.0.0.1:0/?t=fake-token")


def test_run_albums_serve_demo_builds_a_synthetic_two_trip_pool(tmp_path):
    fake = _FakeAlbumLauncher()
    run_albums_serve(demo=True, serve=False, launcher=fake, console=Console(file=io.StringIO()))

    assert len(fake.launch_calls) == 1
    call = fake.launch_calls[0]
    assert call["records"] is not None and len(call["records"]) >= 6
    assert call["keeper_uuids_set"] == {r.uuid for r in call["records"]}


def test_run_albums_serve_passes_through_library_path_when_not_demo():
    fake = _FakeAlbumLauncher()
    run_albums_serve(
        library_path="/some/Real.photoslibrary",
        serve=False,
        launcher=fake,
        console=Console(file=io.StringIO()),
    )

    call = fake.launch_calls[0]
    assert call["records"] is None
    assert call["library_path"] == "/some/Real.photoslibrary"


def test_a_missing_album_extra_names_the_install_command():
    def importer(name):
        raise ModuleNotFoundError("No module named 'fastapi'", name="fastapi")

    with pytest.raises(AlbumUIUnavailable) as caught:
        _load_album_launcher(importer)

    assert 'pip install -e ".[album]"' in str(caught.value)


def test_an_unrelated_missing_module_for_albums_is_not_blamed_on_the_extra():
    def importer(name):
        raise ModuleNotFoundError("No module named 'typo'", name="typo")

    with pytest.raises(ModuleNotFoundError) as caught:
        _load_album_launcher(importer)

    assert not isinstance(caught.value, AlbumUIUnavailable)


def test_load_album_launcher_finds_the_real_module_when_the_extra_is_installed():
    pytest.importorskip("fastapi", reason="needs the [album] extra")

    module = _load_album_launcher()

    assert hasattr(module, "launch")


def test_build_parser_albums_serve_defaults():
    parser = build_parser()
    args = parser.parse_args(["albums", "serve"])
    assert args.albums_command == "serve"
    assert args.library is None
    assert args.demo is False
    assert args.open_browser is True


def test_build_parser_albums_serve_demo_and_no_browser():
    parser = build_parser()
    args = parser.parse_args(["albums", "serve", "--demo", "--no-browser"])
    assert args.demo is True
    assert args.open_browser is False


def test_main_dispatches_albums_serve_command(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "photocull.cli.run_albums_serve", lambda **kwargs: seen.update(kwargs)
    )

    main(["albums", "serve", "--demo", "--no-browser"])

    assert seen["demo"] is True
    assert seen["open_browser"] is False


def test_main_turns_a_missing_album_extra_into_a_message_not_a_traceback(monkeypatch):
    def explode(**kwargs):
        raise AlbumUIUnavailable(ALBUM_EXTRA_HINT)

    monkeypatch.setattr("photocull.cli.run_albums_serve", explode)

    with pytest.raises(SystemExit) as caught:
        main(["albums", "serve"])

    assert 'pip install -e ".[album]"' in str(caught.value)
