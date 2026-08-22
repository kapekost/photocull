"""Task 3: the demo dataset — the whole review UI, runnable with no Photos library.

This is what makes the Playwright suite (Tasks 10–13) possible at all: a real Photos
library needs Full Disk Access, is a different shape on every machine, and cannot be
asserted against. The demo is shaped like the *real* library at the live 0.48 cut,
because UI problems only appear at real rates — a demo of four tidy pairs would never
show that a quarter of clusters have no local detail to zoom into.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from photocull.imaging import gray_bitmap, image_dimensions
from photocull_review import demo
from photocull_review.session import build_session


# --- write_png ------------------------------------------------------------------


def test_write_png_produces_a_file_a_real_decoder_accepts(tmp_path):
    """Round-tripped through ImageIO, not through this module's own parser: a
    hand-rolled writer that only its own reader accepts is worth nothing to a browser."""
    path = tmp_path / "a.png"
    demo.write_png(path, width=1024, height=768, rgb=(200, 60, 60), stripe=8)
    assert image_dimensions(str(path)) == (1024, 768)


def test_write_png_handles_the_small_raster_class_too(tmp_path):
    path = tmp_path / "b.png"
    demo.write_png(path, width=480, height=360, rgb=(60, 120, 200), stripe=4)
    assert image_dimensions(str(path)) == (480, 360)


def test_different_styles_produce_different_images(tmp_path):
    a, b, c = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"
    demo.write_png(a, width=480, height=360, rgb=(200, 60, 60), stripe=8)
    demo.write_png(b, width=480, height=360, rgb=(60, 200, 60), stripe=8)
    demo.write_png(c, width=480, height=360, rgb=(200, 60, 60), stripe=4)
    assert a.read_bytes() != b.read_bytes()
    assert a.read_bytes() != c.read_bytes()


def test_write_png_really_is_a_checkerboard_in_both_axes(tmp_path):
    """Decoded and sampled, because nothing else here would notice vertical stripes.

    A writer that emitted the same row for every scanline passes every byte-comparison
    test in this file — different colours and different checker sizes still produce
    different bytes. It would also still magnify visibly, so no downstream assertion
    would catch it either. The vertical alternation is what makes a crop from the top of
    a photo distinguishable from a crop of its middle, which is Task 12's whole subject."""
    path = tmp_path / "check.png"
    demo.write_png(path, width=64, height=64, rgb=(240, 240, 240), stripe=16)
    bitmap = gray_bitmap(str(path))

    def pixel(x, y):
        return bitmap.data[y * bitmap.stride + x]

    assert pixel(4, 4) != pixel(4, 20)  # same column, next band down
    assert pixel(4, 4) != pixel(20, 4)  # same row, next band across
    assert pixel(4, 4) == pixel(20, 20)  # diagonal neighbour is the same phase


def test_write_png_is_deterministic(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    for path in (a, b):
        demo.write_png(path, width=480, height=360, rgb=(10, 20, 30), stripe=6)
    assert a.read_bytes() == b.read_bytes()


# --- the dataset ----------------------------------------------------------------


def test_the_demo_is_shaped_like_the_real_library(tmp_path):
    clusters = demo.build_demo_clusters(tmp_path)
    sizes = Counter(len(c.records) for c in clusters)

    assert len(clusters) == 24
    assert sizes[2] == 17  # 16 plain pairs + the one with an evicted derivative
    assert sum(n for size, n in sizes.items() if 3 <= size <= 4) == 4
    assert sum(n for size, n in sizes.items() if 5 <= size <= 9) == 2
    assert sizes[20] == 1


def test_the_demo_reproduces_the_real_rates_that_drive_the_ui():
    """Measured over all 1,807 clusters of the real library at the live 0.48 cut: 40.1%
    ambiguous, 90.3% with criteria dropped, 4.5% with no sharpness signal. A demo of
    confident pairs would let a broken pick-one screen ship.

    These are **not** the Phase 1b plan's numbers. It asked for ~13 ambiguous of 24,
    which is 52.7% — the rate at the superseded 0.40 cut. 13 would overstate how much of
    the review is a manual pick by 13 percentage points, on the one axis that decides
    what the UI mostly does."""
    clusters = demo.build_demo_clusters(Path("/nonexistent"), write_files=False)
    assert sum(1 for c in clusters if c.is_ambiguous) == 10  # 41.7% vs 40.1%
    assert sum(1 for c in clusters if c.dropped_criteria) == 22  # 91.7% vs 90.3%
    # Deliberately over-represented (8.3% vs 4.5%) so the case runs more than once.
    assert sum(1 for c in clusters if not c.sharpness_available) == 2


def test_the_demo_winner_is_the_sharpest_take_it_has():
    """The fixture's sub-scores must agree with the winner its totals name, or anything
    built on them explains the pick with the reason it lost.

    Found by Task 11: `sharpness` fell monotonically with position while `total` peaked
    elsewhere, so the winner was NOT the sharpest take in **22 of these 24 clusters**,
    and the evidence panel said "softer than the sharpest take" under every proposal.
    Nothing caught it because nothing had ever read the sub-scores as a *statement about
    the photograph* — the score table renders whatever it is handed."""
    clusters = demo.build_demo_clusters(Path("/nonexistent"), write_files=False)
    checked = 0
    for cluster in clusters:
        values = {s.uuid: s.sub_scores["sharpness"] for s in cluster.scores}
        if any(v is None for v in values.values()):
            continue
        checked += 1
        assert max(values, key=lambda u: values[u]) == cluster.winner_uuid
        # `cluster_sharpness` is a ratio to the sharpest take, so exactly one member of
        # a real cluster is 1.00. A fixture where nobody is has stopped being one.
        assert sum(1 for v in values.values() if v == 1.0) == 1, values
    assert checked == 22


def test_the_demo_keeps_a_close_call_looking_close():
    """An ambiguous cluster whose sub-scores differ visibly would put "1.3x the detail of
    the softest" directly above "too close to call"."""
    clusters = demo.build_demo_clusters(Path("/nonexistent"), write_files=False)
    for cluster in clusters:
        if not cluster.is_ambiguous:
            continue
        values = [s.sub_scores["sharpness"] for s in cluster.scores]
        if any(v is None for v in values):
            continue
        assert max(values) - min(values) < 0.10, (cluster.winner_uuid, values)


def test_the_demo_has_both_raster_classes_and_one_evicted_derivative(tmp_path):
    clusters = demo.build_demo_clusters(tmp_path)
    long_sides = set()
    missing = 0
    for cluster in clusters:
        for record in cluster.records:
            path = Path(record.display_path)
            if not path.exists():
                missing += 1
                continue
            long_sides.add(max(image_dimensions(str(path))))
    assert {480, 1024} <= long_sides
    assert missing == 1


def test_most_demo_photos_have_a_different_file_for_analysis_and_for_display(tmp_path):
    """The fixture defect Task 9 found, closed.

    All 82 records used to carry `derivative_path == display_path`, so the mutation
    "serve the analysis raster instead of the display one" survived — not because the
    tests were weak but because the fixture could not tell the two apart
    (`the-demo-library-cannot-tell-the-two-derivatives-apart`). On the real library the
    two differ for **62.2%** of photos; here they differ for 58 of 82 (**70.7%**), and
    the remainder are the photos whose only local copy is already the small one, which
    is the real library's own shape rather than a simplification.
    """
    clusters = demo.build_demo_clusters(tmp_path)
    records = [r for c in clusters for r in c.records]
    differ = [r for r in records if r.derivative_path != r.display_path]

    assert len(records) == 82
    assert len(differ) == 58, "the split must cover most, but deliberately not all, photos"
    # Both cases have to exist, or the fixture stops reproducing one of them.
    assert 0 < len(differ) < len(records)


def test_where_the_two_rasters_differ_the_display_one_is_bigger(tmp_path):
    """A wrong-raster bug is a resolution bug, so the fixture's split is a size split."""
    clusters = demo.build_demo_clusters(tmp_path)
    checked = 0
    for record in (r for c in clusters for r in c.records):
        if record.derivative_path == record.display_path:
            continue
        analysis, display = Path(record.derivative_path), Path(record.display_path)
        if not (analysis.exists() and display.exists()):
            continue  # the evicted member has neither
        assert max(image_dimensions(str(analysis))) == 480
        assert max(image_dimensions(str(display))) == 1024
        checked += 1
    # 58 records carry a split; 57 of them have files, because the evicted member is in
    # a two-copy cluster and eviction takes both of its copies.
    assert checked == 57


def test_the_two_rasters_of_one_photo_differ_in_pixels_not_only_in_dimensions(tmp_path):
    """Detectable by a test that reads pixels, not only by one that reads `naturalWidth`.

    Same colour, so "which photograph is in this pane" still reads off the palette;
    different checker frequency, so "which raster is this" is legible in the pixels
    themselves. A crop comparison that never looks at the image header can still catch
    the wrong file.
    """
    clusters = demo.build_demo_clusters(tmp_path)
    record = next(
        r
        for c in clusters
        for r in c.records
        if r.derivative_path != r.display_path and Path(r.derivative_path).exists()
    )
    analysis_rgb, analysis_stripe = demo.photo_style(record.uuid)
    display_rgb, display_stripe = demo.photo_style(record.uuid)

    assert analysis_rgb == display_rgb
    assert demo.analysis_stripe(analysis_stripe) != display_stripe
    assert Path(record.derivative_path).read_bytes() != Path(record.display_path).read_bytes()


def test_the_evicted_member_has_neither_raster_on_disk(tmp_path):
    """Eviction takes the whole photo, not one of its two copies."""
    clusters = demo.build_demo_clusters(tmp_path)
    gone = [
        r
        for c in clusters
        for r in c.records
        if not Path(r.display_path).exists()
    ]
    assert len(gone) == 1
    assert not Path(gone[0].derivative_path).exists()


def test_every_demo_photo_looks_different_from_every_other(tmp_path):
    """Playwright must be able to assert *which* photo is in *which* pane, and that a
    magnified crop differs from a fitted view — so colour and stripe frequency are both
    per-photo, and no two photos may share a style."""
    clusters = demo.build_demo_clusters(tmp_path)
    uuids = [r.uuid for c in clusters for r in c.records]
    styles = [demo.photo_style(u) for u in uuids]
    assert len(set(styles)) == len(styles)
    assert len({stripe for _, stripe in styles}) > 1


def test_the_demo_is_deterministic(tmp_path):
    a = demo.build_demo_clusters(tmp_path / "a")
    b = demo.build_demo_clusters(tmp_path / "b")
    assert [[r.uuid for r in c.records] for c in a] == [[r.uuid for r in c.records] for c in b]
    assert [c.winner_uuid for c in a] == [c.winner_uuid for c in b]


def test_the_demo_builds_a_real_session_document(tmp_path):
    doc = demo.build_demo_session(tmp_path)
    assert len(doc["clusters"]) == 24
    blob = json.dumps(doc)
    assert str(tmp_path) not in blob
    assert json.loads(blob)["population"]["clusters"] == 24


def test_the_demo_session_exercises_every_annotation_code(tmp_path):
    """A demo that never triggers an annotation cannot test the surface that shows it."""
    doc = demo.build_demo_session(tmp_path)
    seen = {note["code"] for c in doc["clusters"] for note in c["annotations"]}
    assert seen == {
        "ambiguous",
        "no-display",
        "no-sharpness",
        "low-detail",
        "mixed-resolution",
        "large-cluster",
    }


def test_the_demo_session_has_ambiguous_clusters_that_still_suggest(tmp_path):
    """`every-cluster-opens-with-a-suggestion`: a near-tie proposes its winner like any
    other cluster, and declares itself a close call in prose instead."""
    doc = demo.build_demo_session(tmp_path)
    ambiguous = [c for c in doc["clusters"] if c["is_ambiguous"]]
    assert len(ambiguous) == 10
    for entry in ambiguous:
        assert entry["proposed"][entry["winner_uuid"]] == "keep"
        assert "ambiguous" in [n["code"] for n in entry["annotations"]]


def test_the_demo_annotates_at_the_measured_rates(tmp_path):
    """The attention set covers 34.6% of real clusters. If the demo annotated most of
    them, every later task would be built and eyeballed against a screen that is mostly
    warning banner — which is the failure `annotations` exists to avoid."""
    doc = demo.build_demo_session(tmp_path)
    codes = Counter(note["code"] for c in doc["clusters"] for note in c["annotations"])
    assert codes["low-detail"] == 6  # 25.0% vs a measured 25.3%
    assert codes["mixed-resolution"] == 2  # 8.3% vs 7.7%
    assert codes["no-sharpness"] == 2
    assert codes["large-cluster"] == 1
    assert codes["no-display"] == 1
    attention = [c for c in doc["clusters"] if any(n["code"] != "ambiguous" for n in c["annotations"])]
    assert len(attention) == 11
    # 12 notes over 11 clusters: one cluster stacks two, because real ones do and the
    # surface that renders them must handle a list rather than a single banner.
    assert max(len(c["annotations"]) for c in doc["clusters"]) == 2

    # `ambiguous` is deliberately NOT counted in the attention rate above. It fires on
    # 10 of 24 here (41.7%, against a measured 40.1%), so folding it into the same
    # banner would annotate 19 of 24 clusters — the "mostly warning banner" screen this
    # test exists to prevent. It is a confidence statement about the *suggestion*, not a
    # defect in the cluster, so Task 11 must render it attached to the proposed KEEP
    # rather than in the warnings list. See `every-cluster-opens-with-a-suggestion`.
    #
    # 19 rather than 18 since Task 12 moved the second mixed-resolution cluster from a
    # pair that was *already* annotated (ambiguous) onto a four-take cluster that carried
    # nothing. The note count is unchanged; it now covers one more cluster, which is why
    # the attention rate above did not move.
    assert codes["ambiguous"] == 10
    assert sum(1 for c in doc["clusters"] if c["annotations"]) == 19


def test_the_demo_reproduces_the_records_scores_misalignment():
    """The trap this whole document's builder exists to avoid, present in the fixture.

    `rank_cluster` sorts `scores` by descending total while `records` keep bucket order,
    so the two lists routinely disagree. A demo whose winner is always the first record
    would let an index join pass every test and then hand the owner another photo's
    sub-scores against a real library. Asserted, not assumed — with a monotonic total
    this fixture *was* index-aligned when first written."""
    clusters = demo.build_demo_clusters(Path("/nonexistent"), write_files=False)
    misaligned = [
        c
        for c in clusters
        if [r.uuid for r in c.records] != [s.uuid for s in c.scores]
    ]
    assert len(misaligned) == 24
    # The winner is never the first record, deliberately: a uniform pick would leave
    # 1-in-size clusters aligned by luck, and those are exactly the ones an index join
    # would pass on.
    assert all(c.winner_uuid != c.records[0].uuid for c in clusters)


def test_build_demo_clusters_matches_the_session_document(tmp_path):
    """`build_demo_session` must be the demo clusters through the real `build_session`,
    not a second hand-written document that can drift from it."""
    clusters = demo.build_demo_clusters(tmp_path)
    assert demo.build_demo_session(tmp_path) == build_session(clusters)
