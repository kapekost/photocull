"""Task 8a: the end-to-end clustering pipeline.

Every analyzer is injected, so nothing here touches Vision, Quartz or a Photos
library. Records carry a synthetic `derivative_path` because that -- not the uuid --
is what the analyzers are keyed by in production too; see the module docstring in
`pipeline.py` for why the plan's uuid-keyed test seam was rejected.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from photocull.cache import AnalysisCache
from photocull.config import ClusterConfig
from photocull.imaging import PixelStats
from photocull.pipeline import Analyzers, run_pipeline
from photocull.vision_backend import FaceObservation
from tests.fixtures import make_photo_record

BASE = datetime(2025, 6, 1, 12, 0, 0)


def rec(i, **kw):
    kw.setdefault("derivative_path", f"/d/{i}.jpg")
    return make_photo_record(uuid=f"u{i}", date=BASE + timedelta(seconds=i), **kw)


def stats(variance=100.0, width=480, height=360, luma=0.46):
    return PixelStats(
        width=width,
        height=height,
        laplacian_variance=variance,
        mean_luma=luma,
        shadow_clipped=0.0,
        highlight_clipped=0.0,
    )


def analyzers(vectors, *, stats_by=None, faces_by=None, horizon_by=None, log=None):
    """Build an Analyzers bundle keyed by derivative path. `log` collects calls so a
    test can assert which stage touched which photo."""

    def record(stage, path):
        if log is not None:
            log.append((stage, path))

    return Analyzers(
        printer=lambda p: (record("printer", p), vectors.get(p))[1],
        stats=lambda p: (record("stats", p), (stats_by or {}).get(p, stats()))[1],
        faces=lambda p: (record("faces", p), (faces_by or {}).get(p, []))[1],
        horizon=lambda p: (record("horizon", p), (horizon_by or {}).get(p))[1],
    )


def run(records, vectors, tmp_path, **kw):
    with AnalysisCache(tmp_path / "c.db") as cache:
        return run_pipeline(records, cache=cache, analyzers=analyzers(vectors, **kw))


# --- grouping ------------------------------------------------------------------


def test_clusters_similar_photos_and_leaves_the_odd_one_out(tmp_path):
    records = [rec(0), rec(1), rec(9999)]
    vectors = {"/d/0.jpg": [0.0, 0.0], "/d/1.jpg": [0.05, 0.0], "/d/9999.jpg": [9.0, 0.0]}
    clusters = run(records, vectors, tmp_path)
    assert len(clusters) == 1
    assert sorted(r.uuid for r in clusters[0].records) == ["u0", "u1"]


def test_skips_photos_with_no_derivative_path(tmp_path):
    """A photo with no local derivative is skipped, never escalated to the original
    (CLAUDE.md hard rule #2). It must not even reach an analyzer."""
    log: list[tuple[str, str]] = []
    records = [rec(0), rec(1), rec(2, derivative_path=None)]
    vectors = {"/d/0.jpg": [0.0, 0.0], "/d/1.jpg": [0.05, 0.0], "/d/2.jpg": [0.05, 0.0]}
    with AnalysisCache(tmp_path / "c.db") as cache:
        clusters = run_pipeline(
            records, cache=cache, analyzers=analyzers(vectors, log=log)
        )
    # u2's vector is present and would have clustered, so its absence proves the
    # skip happened on the missing derivative rather than on the feature print.
    assert [sorted(r.uuid for r in c.records) for c in clusters] == [["u0", "u1"]]
    assert "/d/2.jpg" not in {p for _, p in log}


def test_a_bucket_that_cannot_form_a_pair_costs_nothing(tmp_path):
    """A lone photo in its own time bucket can never be a near-duplicate of anything,
    so it must not be feature-printed at all -- that is 23% of the real library."""
    log: list[tuple[str, str]] = []
    records = [rec(0), rec(9999)]
    vectors = {"/d/0.jpg": [0.0, 0.0], "/d/9999.jpg": [0.0, 0.0]}
    with AnalysisCache(tmp_path / "c.db") as cache:
        assert run_pipeline(records, cache=cache, analyzers=analyzers(vectors, log=log)) == []
    assert log == []


def test_skips_photos_with_no_feature_print(tmp_path):
    records = [rec(0), rec(1)]
    assert run(records, {}, tmp_path) == []


def test_clusters_come_back_in_capture_order(tmp_path):
    """Bucketing sorts by (date, uuid) and union_find orders by lowest member, so the
    pipeline must not re-sort on the way out (`clusters-ordered-by-earliest-member`)."""
    records = [rec(0), rec(1), rec(500), rec(501)]
    vectors = {
        "/d/0.jpg": [0.0, 0.0],
        "/d/1.jpg": [0.05, 0.0],
        "/d/500.jpg": [5.0, 0.0],
        "/d/501.jpg": [5.05, 0.0],
    }
    clusters = run(records, vectors, tmp_path)
    assert [sorted(r.uuid for r in c.records) for c in clusters] == [
        ["u0", "u1"],
        ["u500", "u501"],
    ]


# --- the cost lever ------------------------------------------------------------


def test_pixel_analyzers_run_only_on_photos_that_landed_in_a_cluster(tmp_path):
    """Feature prints are needed for everything in a bucket to decide the grouping;
    faces/sharpness/horizon are only needed for photos being ranked against a sibling.
    Measured on the real library that is 39% of bucketed photos, so this is the
    difference between a ~4 min cold run and a much longer one."""
    log: list[tuple[str, str]] = []
    records = [rec(0), rec(1), rec(2)]
    vectors = {
        "/d/0.jpg": [0.0, 0.0],
        "/d/1.jpg": [0.05, 0.0],
        "/d/2.jpg": [9.0, 0.0],  # same bucket, not similar
    }
    with AnalysisCache(tmp_path / "c.db") as cache:
        run_pipeline(records, cache=cache, analyzers=analyzers(vectors, log=log))
    printed = {p for stage, p in log if stage == "printer"}
    scored = {p for stage, p in log if stage != "printer"}
    assert printed == {"/d/0.jpg", "/d/1.jpg", "/d/2.jpg"}
    assert scored == {"/d/0.jpg", "/d/1.jpg"}


# --- caching -------------------------------------------------------------------


def test_a_second_run_is_served_entirely_from_the_cache(tmp_path):
    records = [rec(0), rec(1)]
    vectors = {"/d/0.jpg": [0.0, 0.0], "/d/1.jpg": [0.05, 0.0]}
    log: list[tuple[str, str]] = []
    with AnalysisCache(tmp_path / "c.db") as cache:
        a = analyzers(vectors, log=log)
        run_pipeline(records, cache=cache, analyzers=a)
        first = len(log)
        run_pipeline(records, cache=cache, analyzers=a)
    assert first > 0
    assert len(log) == first, "second run must not call a single analyzer again"


def test_a_cached_run_produces_the_same_clusters(tmp_path):
    records = [rec(0), rec(1)]
    vectors = {"/d/0.jpg": [0.0, 0.0], "/d/1.jpg": [0.05, 0.0]}
    faces_by = {
        "/d/0.jpg": [FaceObservation(bounding_box=(0.4, 0.4, 0.2, 0.2), left_eye_open=0.3,
                                     right_eye_open=0.3, smiling=0.1, capture_quality=0.5)],
        "/d/1.jpg": [FaceObservation(bounding_box=(0.4, 0.4, 0.2, 0.2), left_eye_open=0.1,
                                     right_eye_open=0.1, smiling=0.1, capture_quality=0.4)],
    }
    kw = dict(faces_by=faces_by, horizon_by={"/d/0.jpg": 0.05, "/d/1.jpg": 0.2})
    with AnalysisCache(tmp_path / "c.db") as cache:
        cold = run_pipeline(records, cache=cache, analyzers=analyzers(vectors, **kw))
        warm = run_pipeline(records, cache=cache, analyzers=analyzers(vectors, **kw))
    assert cold[0].winner_uuid == warm[0].winner_uuid
    assert cold[0].scores[0].sub_scores == pytest.approx(warm[0].scores[0].sub_scores)
    assert cold[0].scores[0].total == pytest.approx(warm[0].scores[0].total)


def test_changing_the_derivative_invalidates_the_cached_analysis(tmp_path):
    """`analysis_key` is the derivative, not just the uuid. A photo re-read from a
    different raster must be re-analyzed, never served the old numbers
    (`derivative-selection-is-smallest-class`)."""
    log: list[tuple[str, str]] = []
    vectors = {"/d/0.jpg": [0.0, 0.0], "/d/1.jpg": [0.05, 0.0], "/other/0.jpg": [0.0, 0.0]}
    with AnalysisCache(tmp_path / "c.db") as cache:
        a = analyzers(vectors, log=log)
        run_pipeline([rec(0), rec(1)], cache=cache, analyzers=a)
        before = len(log)
        run_pipeline(
            [rec(0, derivative_path="/other/0.jpg"), rec(1)], cache=cache, analyzers=a
        )
    second = log[before:]
    # Assert PER STAGE. An earlier version of this test asserted only that
    # "/other/0.jpg" appeared somewhere in the second run's log, and it passed under a
    # mutation that keyed the analysis cache on uuid -- because the feature-print
    # stage produced that path on its own. Two stages read this key and both must miss.
    assert ("printer", "/other/0.jpg") in second
    assert ("stats", "/other/0.jpg") in second, "pixel analysis must re-run on a new raster"
    assert ("faces", "/other/0.jpg") in second
    # ...and the photo that did not change must still be served from cache, so this
    # cannot pass by the cache having been discarded wholesale.
    assert not any(path == "/d/1.jpg" for _, path in second)


# --- what the calibration export needs -----------------------------------------


def test_records_each_members_distance_to_the_photo_being_kept(tmp_path):
    """`cull-safety-is-radius-to-keeper`: the reviewable number is d(culled, keeper),
    not the cluster's width."""
    records = [rec(0), rec(1), rec(2)]
    vectors = {"/d/0.jpg": [0.0, 0.0], "/d/1.jpg": [0.1, 0.0], "/d/2.jpg": [0.2, 0.0]}
    clusters = run(records, vectors, tmp_path)
    c = clusters[0]
    assert c.winner_uuid not in c.distances_to_winner
    assert sorted(c.distances_to_winner) == sorted(
        r.uuid for r in c.records if r.uuid != c.winner_uuid
    )
    expected = {"u0": 0.0, "u1": 0.1, "u2": 0.2}
    for uuid, distance in c.distances_to_winner.items():
        assert distance == pytest.approx(abs(expected[uuid] - expected[c.winner_uuid]))


def test_records_cluster_diameter_and_median_internal_distance(tmp_path):
    records = [rec(0), rec(1), rec(2)]
    vectors = {"/d/0.jpg": [0.0, 0.0], "/d/1.jpg": [0.1, 0.0], "/d/2.jpg": [0.2, 0.0]}
    c = run(records, vectors, tmp_path)[0]
    # pairwise: 0.1, 0.2, 0.1 -> median 0.1, diameter 0.2
    assert c.diameter == pytest.approx(0.2)
    assert c.median_distance == pytest.approx(0.1)


def test_flags_a_cluster_that_got_no_sharpness_signal(tmp_path):
    """7.7% of real multi-item buckets mix raster scale, so `cluster_sharpness` drops
    the criterion entirely. The owner must be able to see which ones at Task 9."""
    records = [rec(0), rec(1)]
    vectors = {"/d/0.jpg": [0.0, 0.0], "/d/1.jpg": [0.05, 0.0]}
    mixed = {"/d/0.jpg": stats(width=1524, height=360), "/d/1.jpg": stats(width=360, height=480)}
    c = run(records, vectors, tmp_path, stats_by=mixed)[0]
    assert c.sharpness_available is False
    assert "sharpness" in c.dropped_criteria
    assert all(s.sub_scores["sharpness"] is None for s in c.scores)


def test_a_comparable_cluster_keeps_its_sharpness_signal(tmp_path):
    records = [rec(0), rec(1)]
    vectors = {"/d/0.jpg": [0.0, 0.0], "/d/1.jpg": [0.05, 0.0]}
    same = {"/d/0.jpg": stats(variance=100.0), "/d/1.jpg": stats(variance=50.0)}
    c = run(records, vectors, tmp_path, stats_by=same)[0]
    assert c.sharpness_available is True
    assert "sharpness" not in c.dropped_criteria


def test_zero_weighted_criteria_are_not_reported_as_dropped(tmp_path):
    """`facing` is always None and ships at weight 0.0
    (`unvalidated-signals-ship-at-zero-weight`). Reporting it as dropped on every
    single cluster would make the annotation worthless."""
    records = [rec(0), rec(1)]
    vectors = {"/d/0.jpg": [0.0, 0.0], "/d/1.jpg": [0.05, 0.0]}
    c = run(records, vectors, tmp_path)[0]
    assert ClusterConfig().weights["facing"] == 0.0
    assert "facing" not in c.dropped_criteria


def test_reports_a_weighted_criterion_that_was_dropped(tmp_path):
    """No faces anywhere in the cluster -> the faces criterion could not be measured."""
    records = [rec(0), rec(1)]
    vectors = {"/d/0.jpg": [0.0, 0.0], "/d/1.jpg": [0.05, 0.0]}
    c = run(records, vectors, tmp_path)[0]
    assert ClusterConfig().weights["faces"] > 0.0
    assert "faces" in c.dropped_criteria


# --- plumbing ------------------------------------------------------------------


def test_honours_a_supplied_config(tmp_path):
    """A threshold tight enough to separate the pair must actually separate it --
    this is the seam Task 9's calibration turns."""
    records = [rec(0), rec(1)]
    vectors = {"/d/0.jpg": [0.0, 0.0], "/d/1.jpg": [0.3, 0.0]}
    with AnalysisCache(tmp_path / "c.db") as cache:
        loose = run_pipeline(records, cache=cache, analyzers=analyzers(vectors),
                             config=ClusterConfig(similarity_threshold=0.4))
        tight = run_pipeline(records, cache=cache, analyzers=analyzers(vectors),
                             config=ClusterConfig(similarity_threshold=0.2))
    assert len(loose) == 1
    assert tight == []


def test_reports_progress_per_stage(tmp_path):
    seen: list[tuple[str, int, int]] = []
    records = [rec(0), rec(1)]
    vectors = {"/d/0.jpg": [0.0, 0.0], "/d/1.jpg": [0.05, 0.0]}
    with AnalysisCache(tmp_path / "c.db") as cache:
        run_pipeline(records, cache=cache, analyzers=analyzers(vectors),
                     progress=lambda stage, done, total: seen.append((stage, done, total)))
    assert {stage for stage, _, _ in seen} == {"feature-print", "analyze"}
    assert all(done <= total for _, done, total in seen)


# --- two things the tests above did not actually pin ---------------------------


def test_cached_analysis_round_trips_through_json_exactly(tmp_path):
    """Goes through json.dumps/loads because that is what the cache does, and it is
    what destroys the bounding box's tuple-ness. A list unpacks in `framing_score`
    exactly like a tuple, so a warm run would score identically off an object that no
    longer equals the cold one -- passing tests, silently divergent data."""
    import json

    from photocull.pipeline import _decode, _encode

    face = FaceObservation(
        bounding_box=(0.1, 0.2, 0.3, 0.4),
        yaw=0.0,
        roll=None,
        left_eye_open=0.3,
        right_eye_open=0.25,
        smiling=-0.1,
        capture_quality=0.42,
    )
    original = stats(variance=12.5)
    payload = json.loads(json.dumps(_encode(original, [face], 0.07)))
    back_stats, back_faces, back_horizon = _decode(payload)
    assert back_stats == original
    assert back_faces == [face]
    assert isinstance(back_faces[0].bounding_box, tuple)
    assert back_horizon == pytest.approx(0.07)


def test_a_criterion_measured_for_only_one_take_cannot_decide_the_cluster(tmp_path):
    """Pins `on_common_criteria` into the pipeline, not just into scoring.py. Without
    it a criterion wins by merely being *present*: the take Vision happened to find a
    face in would out-total its twin on detection luck
    (DECISIONS.md `scoring-compares-on-common-criteria`)."""
    records = [rec(0), rec(1)]
    vectors = {"/d/0.jpg": [0.0, 0.0], "/d/1.jpg": [0.05, 0.0]}
    faces_by = {
        "/d/0.jpg": [
            FaceObservation(
                bounding_box=(0.4, 0.4, 0.2, 0.2),
                left_eye_open=0.35,
                right_eye_open=0.35,
                smiling=0.2,
                capture_quality=0.9,
            )
        ]
    }
    c = run(records, vectors, tmp_path, faces_by=faces_by)[0]
    assert "faces" in c.dropped_criteria and "framing" in c.dropped_criteria
    assert c.scores[0].total == pytest.approx(c.scores[1].total)
    assert c.is_ambiguous is True


# --- shared-album assets -------------------------------------------------------


def test_shared_album_assets_are_excluded_from_clustering(tmp_path):
    """Measured on the real library: shared assets are 6.2% of it but **27.1% of the
    photos this app stages for culling** -- a 4.4x over-representation, because shared
    albums are exactly where duplicate takes accumulate. Photos' AppleScript surface
    cannot act on them either, so clustering them is labour that produces nothing and
    proposes culling photos sitting in someone else's album."""
    records = [rec(0), rec(1, is_shared=True)]
    vectors = {r.derivative_path: [0.0, 0.0, 1.0] for r in records}
    with AnalysisCache(tmp_path / "c.db") as cache:
        clusters = run_pipeline(
            records, cache=cache, analyzers=analyzers(vectors), config=ClusterConfig()
        )
    # identical vectors one second apart: they would certainly group if both took part
    assert clusters == []


def test_shared_assets_can_be_opted_back_in(tmp_path):
    records = [rec(0), rec(1, is_shared=True)]
    vectors = {r.derivative_path: [0.0, 0.0, 1.0] for r in records}
    with AnalysisCache(tmp_path / "c.db") as cache:
        clusters = run_pipeline(
            records,
            cache=cache,
            analyzers=analyzers(vectors),
            config=ClusterConfig(include_shared=True),
        )
    assert len(clusters) == 1
    assert {r.uuid for r in clusters[0].records} == {"u0", "u1"}
