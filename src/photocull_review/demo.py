"""A synthetic library the whole review UI runs against — no Photos, no FDA, no Vision.

This is what makes the Playwright suite possible at all. A real library needs Full Disk
Access, is a different shape on every machine, and cannot be asserted against; a browser
test needs to know that the top-left pane holds *that* photograph and no other.

**Shaped like the real library, not tidied.** UI problems only appear at real rates: a
demo of four clean pairs would never show that a real fraction of clusters have no
local detail to examine, and a broken pick-one screen would ship because no demo
cluster was ever ambiguous. The proportions below are scaled down from what a real
library actually produces — a real share ambiguous, a real share mixed resolution, a
real share with no sharpness signal, a real share large — down to 24 clusters.

The rasters are written by `write_png`: ~40 lines of stdlib `zlib` and `struct`, so the
demo costs no Pillow, no Quartz and no npm. Each photo gets its own colour *and* its own
checker size, which is what lets a browser test assert both which image is in which pane
and that a magnified crop differs from a fitted one.

**Most photos have two files, and that is the point.** A photo's *analysis* raster
(`derivative_path`, the smallest local copy — what clustering and sharpness are measured
on) and its *display* raster (`display_path`, the largest — what the owner judges by) are
different files for the majority of a real library. The demo library used to make them
the same file for every record, which silently disarmed every test written against the
demo: the mutation "serve the analysis raster instead of the display one" survived a
full pass because no fixture could tell the two apart. They now differ for most demo
records, and the ones that don't are the photos whose only local copy is already the
small one — the real library's own shape, not a simplification of it. The two copies of
one photo share a colour and differ in checker frequency, so which *photograph* is in a
pane reads off the palette while which *raster* it came from is legible in the pixels as
well as in `naturalWidth`.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from photocull.models import Cluster, PhotoRecord, PhotoScore

from .session import build_session

#: The two raster classes the real library actually has: the analysis derivative's 480px
#: class and the largest-local-derivative's 1024px class (median 1024px, measured).
LARGE_RASTER = (1024, 768)
SMALL_RASTER = (480, 360)

#: Fixed so a failure message quotes a reproducible date, and so two runs of the demo
#: produce byte-identical records. Nothing here uses an RNG.
BASE_DATE = datetime(2026, 3, 1, 9, 0, 0)

#: 12 colours x 8 checker sizes = 96 distinct styles, against the demo's 82 photos, so
#: every photo can look different from every other one rather than merely from its own
#: cluster-mates. Sized deliberately: a browser test that mixed up two panes across
#: clusters would fail in a way that reads as a UI bug.
_PALETTE = (
    (208, 74, 62),
    (232, 158, 54),
    (214, 206, 78),
    (110, 190, 92),
    (72, 178, 168),
    (78, 142, 220),
    (128, 108, 214),
    (206, 96, 178),
    (168, 96, 72),
    (96, 128, 96),
    (176, 176, 200),
    (120, 84, 140),
)
_STRIPES = (4, 6, 8, 10, 12, 16, 20, 24)


@dataclass(frozen=True)
class Raster:
    """One PNG to write. Fully specified, so `build_demo_clusters` decides nothing.

    Carrying the style here rather than re-deriving it from the filename is what lets a
    photo own two files: the analysis raster's name is not a uuid, so a writer that
    parsed the stem would have to be taught the naming scheme to find the style.
    """

    path: Path
    width: int
    height: int
    rgb: tuple[int, int, int]
    stripe: int


def analysis_stripe(display_stripe: int) -> int:
    """The checker size the analysis copy of a photo is drawn at.

    The next size up the `_STRIPES` ladder, wrapping. Deliberately not "the same
    pattern at a smaller size": scaling a checkerboard down produces the same *image*
    at a lower resolution, so a test comparing crops could not tell a correctly-served
    small raster from a wrongly-served one without reading the header. A different
    frequency makes the two distinguishable in the pixels themselves, which is the
    axis a magnification test works on.
    """
    return _STRIPES[(_STRIPES.index(display_stripe) + 1) % len(_STRIPES)]


def _crc_chunk(tag: bytes, payload: bytes) -> bytes:
    body = tag + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))


def write_png(
    path: str | Path, *, width: int, height: int, rgb: tuple[int, int, int], stripe: int
) -> Path:
    """Write a checkerboard PNG. Stdlib only, and deterministic byte for byte.

    Built two rows at a time rather than pixel by pixel: a checkerboard has exactly two
    distinct row patterns, so a 1024x768 raster costs ~2,800 iterations instead of
    786,432. That is the difference between a test suite that runs in one second and one
    that runs in twenty, for an image no human will ever look closely at."""
    light = bytes(rgb)
    dark = bytes(channel // 3 for channel in rgb)
    rows = [
        b"".join(light if ((x // stripe) + phase) % 2 == 0 else dark for x in range(width))
        for phase in (0, 1)
    ]
    # Filter byte 0 (None) per scanline — the raw rows compress fine and skipping the
    # filter keeps this readable.
    raw = b"".join(b"\x00" + rows[(y // stripe) % 2] for y in range(height))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    blob = (
        b"\x89PNG\r\n\x1a\n"
        + _crc_chunk(b"IHDR", header)
        + _crc_chunk(b"IDAT", zlib.compress(raw, 6))
        + _crc_chunk(b"IEND", b"")
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return path


#: One row per cluster: (size, ambiguous, sharpness_available, dropped, small_rasters,
#: evicted). `small_rasters` is how many members get the 480px class — 0 means the whole
#: cluster is at 1024px, `size` means all of it is low-detail, anything between makes the
#: cluster mixed-resolution. `evicted` names a member whose file is never written, which
#: is what Photos having reclaimed a derivative looks like from here.
#:
#: **The rates below are the measured ones, not an earlier plan's estimate.** An
#: earlier estimate of the ambiguous and mixed-resolution counts was based on figures
#: since superseded by a re-measurement against a real library, at a materially
#: different rate on the single number that decides how much of the review is a
#: pick-one screen. Reproducing the old estimate would have built a demo that
#: misrepresents that experience. The actual rates below track the measured ones
#: closely, with one deliberate exception: sharpness-unavailable clusters are slightly
#: over-represented, so the plural case runs at all in a 24-cluster set.
_SPEC: tuple[tuple[int, bool, bool, tuple[str, ...], int, bool], ...] = (
    # 17 pairs — the common case, and the bulk of any real review session.
    (2, False, True, ("horizon",), 0, False),
    (2, True, True, ("horizon",), 0, False),
    (2, False, True, (), 0, False),
    (2, True, True, ("horizon", "faces", "framing"), 2, False),
    (2, False, True, ("horizon",), 1, False),
    (2, True, True, ("horizon",), 0, False),
    (2, False, False, ("horizon", "sharpness"), 0, False),
    (2, True, True, ("horizon", "faces", "framing"), 0, False),
    (2, False, True, ("horizon",), 2, False),
    (2, True, True, (), 0, False),
    (2, False, True, ("horizon",), 0, False),
    (2, True, True, ("horizon",), 0, False),
    (2, False, True, ("horizon", "faces", "framing"), 0, False),
    (2, True, True, ("horizon",), 2, False),
    (2, False, True, ("horizon",), 0, False),
    (2, True, True, ("horizon",), 0, False),
    # The evicted-derivative pair: one member has no readable file at all.
    (2, False, True, ("horizon",), 0, True),
    # 4 clusters of 3-4.
    (3, True, True, ("horizon", "faces", "framing"), 0, False),
    (3, False, True, ("horizon",), 3, False),
    # **A mixed-resolution cluster with more than two takes**, which is what makes the
    # challenger's raster *change size* as it steps. The demo carried its mixed cases
    # only as pairs, where the challenger never changes at all, so a magnification bug
    # that placed a newly-loaded take using the previous take's dimensions passed every
    # test until this case was added. A real library has clusters like this: some
    # fraction mix the two display classes, and cluster sizes run well beyond a pair.
    # The winner is position 2, so the small raster at position 0 lands among the
    # challengers rather than in the pinned keeper. Swapped with the pair above rather
    # than added, so the mixed-resolution rate, the pair count and the 24-small-raster
    # total are all exactly what they were.
    (4, False, True, ("horizon",), 1, False),
    # Two annotations at once — no sharpness signal *and* nothing above 640px to fall
    # back on. Real clusters stack, so the surface that renders them has to.
    (4, False, False, ("horizon", "sharpness"), 4, False),
    # 2 clusters of 5-9.
    (5, True, True, ("horizon", "faces", "framing"), 0, False),
    (9, False, True, ("horizon",), 9, False),
    # The one large cluster. 20 takes, one accepted proposal, 19 photos staged.
    (20, False, True, ("horizon",), 0, False),
)


def photo_style(uuid: str) -> tuple[tuple[int, int, int], int]:
    """This photo's colour and checker size, derived from its uuid alone.

    Indexed by the photo's **dense** ordinal across the whole demo, not by its position
    within its cluster: position alone would give the first photo of every cluster the
    same colour, so a browser test asserting "the left pane holds photo X" would pass
    while showing photo Y from a different cluster. With 82 photos against 96 styles the
    two axes never wrap, so every photo is distinct from every other."""
    _, cluster_part, position_part = uuid.split("-")
    ordinal = sum(spec[0] for spec in _SPEC[: int(cluster_part)]) + int(position_part)
    return (
        _PALETTE[ordinal % len(_PALETTE)],
        _STRIPES[(ordinal // len(_PALETTE)) % len(_STRIPES)],
    )


def _cluster_from_spec(
    index: int, spec: tuple[int, bool, bool, tuple[str, ...], int, bool], root: Path
) -> tuple[Cluster, list[Raster]]:
    size, ambiguous, sharpness_available, dropped, small_rasters, evicted = spec
    records: list[PhotoRecord] = []
    scores: list[PhotoScore] = []
    rasters: list[Raster] = []

    for position in range(size):
        uuid = f"demo-{index:02d}-{position:03d}"
        display_path = root / f"{uuid}.png"
        rgb, stripe = photo_style(uuid)
        # A photo whose largest local copy is already the small class has exactly one
        # derivative, so both selectors return the same file — a meaningful share of
        # these demo records, matching a real library's own shape rather than
        # pretending every photo has two.
        only_one_copy = position < small_rasters
        analysis_path = display_path if only_one_copy else root / f"{uuid}-analysis.png"

        is_evicted = evicted and position == size - 1
        if not is_evicted:
            display_size = SMALL_RASTER if only_one_copy else LARGE_RASTER
            rasters.append(
                Raster(display_path, display_size[0], display_size[1], rgb, stripe)
            )
            if not only_one_copy:
                rasters.append(
                    Raster(
                        analysis_path,
                        SMALL_RASTER[0],
                        SMALL_RASTER[1],
                        rgb,
                        analysis_stripe(stripe),
                    )
                )
        records.append(
            PhotoRecord(
                uuid=uuid,
                date=BASE_DATE + timedelta(days=index, seconds=position * 3),
                width=4032,
                height=3024,
                derivative_path=str(analysis_path),
                display_path=str(display_path),
            )
        )
        # Ambiguous clusters get a near-tie; confident ones a clear gap. The scorer's
        # own `is_ambiguous` is not recomputed here — the demo states it — but the
        # totals have to agree with it or the evidence panel would contradict the badge.
        gap = 0.004 if ambiguous else 0.09
        # And so must the sub-scores -- an earlier version of this demo did not keep
        # them consistent with the totals. `sharpness` used to fall monotonically with
        # position while `total` peaked at `best`, so for most of these clusters the
        # winner was NOT the sharpest take and the evidence panel explained every pick
        # with the reason it lost. It also gave no photo a sharpness of 1.0, which real
        # data cannot do: `cluster_sharpness` emits a ratio to the sharpest take, so
        # exactly one member is always 1.00.
        #
        # Fixed by peaking sharpness where the total peaks, and by stating the *spread*
        # rather than a per-step decrement -- a fixed step made the softest take of the
        # largest cluster far softer than its keeper, well beyond what a real burst
        # group's spread actually looks like. The softest take is therefore pinned to a
        # spread close to what real burst groups show, tighter for a confident cluster
        # and tighter still for an ambiguous one, so a close call looks close at every
        # cluster size.
        softest = 0.98 if ambiguous else 0.74
        # The best take is deliberately NOT the first record. With a monotonically
        # decreasing total, sorting the scores leaves them in record order and the
        # fixture silently stops reproducing the one thing it exists to reproduce —
        # `rank_cluster` emits records and scores in *different* orders, and a consumer
        # that joins them by index gives every photo another photo's sub-scores.
        # Never position 0, so the trap is live in all 24 clusters rather than in the
        # (size-1)/size of them a uniform pick would cover. The realism cost is small
        # and one-directional; the fixture value is the whole point of the fixture.
        best = 1 + (index * 7) % (size - 1)
        # The take furthest from the winner is the softest, whichever side it sits on.
        span = max(best, size - 1 - best) or 1
        scores.append(
            PhotoScore(
                uuid=uuid,
                sub_scores={
                    "sharpness": None
                    if not sharpness_available
                    else round(1.0 - (1.0 - softest) * abs(position - best) / span, 4),
                    "exposure": 0.61 + position * 0.01,
                    "faces": None if "faces" in dropped else 0.44,
                    "framing": None if "framing" in dropped else 0.58,
                    "horizon": None if "horizon" in dropped else 0.81,
                },
                total=round(0.70 - abs(position - best) * gap, 6),
            )
        )

    # `rank_cluster` emits scores sorted by descending total while records keep bucket
    # order. Reproduced faithfully, because a demo that kept them aligned would hide the
    # one join bug this document's builder exists to avoid.
    scores.sort(key=lambda s: (-s.total, s.uuid))
    winner = scores[0].uuid
    cluster = Cluster(
        records=records,
        scores=scores,
        winner_uuid=winner,
        is_ambiguous=ambiguous,
        distances_to_winner={
            r.uuid: round(0.11 + 0.02 * i, 4) for i, r in enumerate(records) if r.uuid != winner
        },
        diameter=round(0.11 + 0.02 * size, 4),
        median_distance=round(0.09 + 0.01 * size, 4),
        sharpness_available=sharpness_available,
        dropped_criteria=sorted(dropped),
    )
    return cluster, rasters


def build_demo_clusters(root: str | Path, *, write_files: bool = True) -> list[Cluster]:
    """The 24 demo clusters, with their rasters written under `root`.

    `write_files=False` builds the records without touching the filesystem, for tests
    that only care about the shape."""
    root = Path(root)
    clusters: list[Cluster] = []
    for index, spec in enumerate(_SPEC):
        cluster, rasters = _cluster_from_spec(index, spec, root)
        if write_files:
            for raster in rasters:
                write_png(
                    raster.path,
                    width=raster.width,
                    height=raster.height,
                    rgb=raster.rgb,
                    stripe=raster.stripe,
                )
        clusters.append(cluster)
    return clusters


def build_demo_session(root: str | Path) -> dict[str, Any]:
    """The demo clusters through the real `build_session`.

    Deliberately not a second hand-written document: a fixture that reimplements the
    thing it stands in for drifts from it, and the drift shows up as a UI bug rather than
    a test failure."""
    return build_session(build_demo_clusters(root))


def demo_records(clusters: Sequence[Cluster]) -> dict[str, PhotoRecord]:
    """uuid -> record, for the image endpoint to resolve without a library."""
    return {record.uuid: record for cluster in clusters for record in cluster.records}
