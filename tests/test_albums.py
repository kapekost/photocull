from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from photocull.albums import (
    AlbumConfig,
    TripGroup,
    demo_pool,
    haversine_km,
    keeper_pool,
    move_before,
    trip_sweep,
)
from tests.fixtures import make_photo_record

BASE = datetime(2026, 1, 1, 12, 0, 0)


def test_keeper_pool_includes_uuids_in_the_keeper_set():
    records = [make_photo_record(uuid="a"), make_photo_record(uuid="b")]
    pool = keeper_pool(records, keeper_uuids={"a"})
    assert [r.uuid for r in pool] == ["a"]


def test_keeper_pool_includes_favorites_even_if_not_in_keeper_set():
    records = [make_photo_record(uuid="a", is_favorite=True), make_photo_record(uuid="b")]
    pool = keeper_pool(records, keeper_uuids=set())
    assert [r.uuid for r in pool] == ["a"]


def test_keeper_pool_dedupes_a_photo_that_is_both():
    records = [make_photo_record(uuid="a", is_favorite=True)]
    pool = keeper_pool(records, keeper_uuids={"a"})
    assert len(pool) == 1


def test_keeper_pool_excludes_videos_even_if_keeper_or_favorite():
    """A video can never be part of a printed photo album -- Phase 3's whole reason to exist
    is a JPEG/PDF print pipeline. Confirmed live on the real library (Tick 56): 44 of 2070 real
    keeper-pool members and 57 of the real Cull/Keepers album's 2077 are is_video=True, and one
    of them reaching export crashes the whole album (see this task's own plan evidence)."""
    records = [
        make_photo_record(uuid="photo", is_favorite=True),
        make_photo_record(uuid="video-keeper", is_video=True),
        make_photo_record(uuid="video-favorite", is_video=True, is_favorite=True),
    ]
    pool = keeper_pool(records, keeper_uuids={"photo", "video-keeper"})
    assert [r.uuid for r in pool] == ["photo"]


def test_keeper_pool_preserves_input_order():
    records = [make_photo_record(uuid=u) for u in ("c", "a", "b")]
    pool = keeper_pool(records, keeper_uuids={"a", "b", "c"})
    assert [r.uuid for r in pool] == ["c", "a", "b"]


def test_trip_sweep_splits_on_a_gap_larger_than_the_threshold():
    records = [
        make_photo_record(uuid="a", date=BASE),
        make_photo_record(uuid="b", date=BASE + timedelta(hours=1)),
        make_photo_record(uuid="c", date=BASE + timedelta(hours=100)),
        make_photo_record(uuid="d", date=BASE + timedelta(hours=101)),
    ]
    groups = trip_sweep(records, AlbumConfig(trip_gap_hours=48, trip_min_size=2))
    assert [[r.uuid for r in g.records] for g in groups] == [["a", "b"], ["c", "d"]]


def test_trip_sweep_drops_groups_below_the_minimum_size():
    records = [
        make_photo_record(uuid="a", date=BASE),
        make_photo_record(uuid="b", date=BASE + timedelta(hours=100)),
        make_photo_record(uuid="c", date=BASE + timedelta(hours=101)),
    ]
    groups = trip_sweep(records, AlbumConfig(trip_gap_hours=48, trip_min_size=2))
    assert [[r.uuid for r in g.records] for g in groups] == [["b", "c"]]


def test_trip_sweep_sorts_by_date_regardless_of_input_order():
    records = [
        make_photo_record(uuid="b", date=BASE + timedelta(hours=1)),
        make_photo_record(uuid="a", date=BASE),
    ]
    groups = trip_sweep(records, AlbumConfig(trip_gap_hours=48, trip_min_size=2))
    assert [r.uuid for r in groups[0].records] == ["a", "b"]


def test_trip_sweep_default_title_is_the_date_range():
    records = [
        make_photo_record(uuid="a", date=datetime(2026, 3, 15, 9, 0)),
        make_photo_record(uuid="b", date=datetime(2026, 3, 17, 18, 0)),
    ]
    groups = trip_sweep(records, AlbumConfig(trip_gap_hours=999, trip_min_size=2))
    assert groups[0].title == "Mar 15 - 17, 2026"


def test_trip_sweep_single_day_title_has_no_range():
    records = [
        make_photo_record(uuid="a", date=datetime(2026, 3, 15, 9, 0)),
        make_photo_record(uuid="b", date=datetime(2026, 3, 15, 18, 0)),
    ]
    groups = trip_sweep(records, AlbumConfig(trip_gap_hours=999, trip_min_size=2))
    assert groups[0].title == "Mar 15, 2026"


def test_trip_sweep_flags_wide_gps_spread():
    records = [
        make_photo_record(uuid="a", date=BASE, latitude=37.7749, longitude=-122.4194),  # SF
        make_photo_record(
            uuid="b", date=BASE + timedelta(hours=1), latitude=40.7128, longitude=-74.0060
        ),  # NYC, ~4,130 km away
    ]
    groups = trip_sweep(records, AlbumConfig(trip_gap_hours=48, trip_min_size=2, gps_spread_flag_km=150))
    assert groups[0].wide_spread is True


def test_trip_sweep_does_not_flag_a_tight_gps_spread():
    records = [
        make_photo_record(uuid="a", date=BASE, latitude=37.7749, longitude=-122.4194),
        make_photo_record(
            uuid="b", date=BASE + timedelta(hours=1), latitude=37.8044, longitude=-122.2712
        ),  # Oakland, ~13 km away
    ]
    groups = trip_sweep(records, AlbumConfig(trip_gap_hours=48, trip_min_size=2, gps_spread_flag_km=150))
    assert groups[0].wide_spread is False


def test_trip_sweep_ignores_gps_spread_when_too_few_located_members():
    # 2 photos, only one has GPS -- can't measure a spread, must not flag.
    records = [
        make_photo_record(uuid="a", date=BASE, latitude=37.7749, longitude=-122.4194),
        make_photo_record(uuid="b", date=BASE + timedelta(hours=1)),
    ]
    groups = trip_sweep(records, AlbumConfig(trip_gap_hours=48, trip_min_size=2, gps_spread_flag_km=150))
    assert groups[0].wide_spread is False


def test_haversine_km_known_distance_sf_to_nyc():
    # Great-circle SF->NYC is ~4,129 km; assert within 1% rather than pinning a float exactly.
    d = haversine_km(37.7749, -122.4194, 40.7128, -74.0060)
    assert 4088 < d < 4170


def test_haversine_km_zero_for_identical_point():
    assert haversine_km(10.0, 20.0, 10.0, 20.0) == 0.0


# --- move_before (Task 10 Step 3) -------------------------------------------------
#
# The reorder rule `app.js`'s drag-and-drop mirrors, pinned in Python so the JS's
# two-line splice logic is proven once rather than trusted by inspection. Table given
# verbatim by the plan (docs/plans/2026-08-19-phase-3-album-builder.md, Task 10 Step 3).


def test_move_before_reinserts_before_the_target():
    assert move_before(["a", "b", "c"], "c", "a") == ["c", "a", "b"]


def test_move_before_moving_forward():
    assert move_before(["a", "b", "c"], "a", "c") == ["b", "a", "c"]


def test_move_before_no_op_for_identical_ids():
    assert move_before(["a", "b"], "a", "a") == ["a", "b"]


def test_move_before_no_op_for_unknown_id():
    assert move_before(["a", "b"], "z", "a") == ["a", "b"]


# --- demo_pool (Task 10's CLI/demo scope gap) --------------------------------------
#
# Task 10's plan text asks for `photocull albums serve --demo` but no earlier step in
# Tasks 1-10 builds a synthetic-data path for photocull_album (unlike
# `photocull review --demo`, which `photocull_review/demo.py` already provides). This
# is that generator: a handful of synthetic PhotoRecords backed by real, tiny PNG files
# on disk (so `images.py`'s endpoint has something real to serve), spread across at
# least two distinct trips (so the trip picker has something to show), with at least
# one image whose aspect ratio matches a `PRINT_SIZES` entry exactly (no crop needed)
# and at least one non-square image whose ratio matches none of them (crop needed) --
# so Step 9's crop-overlay check is meaningful either way.


def test_demo_pool_writes_real_png_files_for_every_record(tmp_path):
    records = demo_pool(tmp_path)
    assert len(records) >= 6
    for record in records:
        path = Path(record.display_path)
        assert path.is_file()
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_demo_pool_spans_at_least_two_trips(tmp_path):
    records = demo_pool(tmp_path)
    groups = trip_sweep(records, AlbumConfig())
    assert len(groups) >= 2


def test_demo_pool_has_a_ratio_matching_photo_and_a_non_matching_one(tmp_path):
    from photocull.printlayout import PRINT_SIZES

    records = demo_pool(tmp_path)
    ratios = {round(r.width / r.height, 4) for r in records}
    print_ratios = {round(size.width_px / size.height_px, 4) for size in PRINT_SIZES.values()}
    assert ratios & print_ratios, "no demo photo matches a print size's aspect ratio"
    non_matching_non_square = [
        r for r in records if round(r.width / r.height, 4) not in print_ratios and r.width != r.height
    ]
    assert non_matching_non_square, "no non-square demo photo needs a crop"


def test_demo_pool_uuids_are_unique(tmp_path):
    records = demo_pool(tmp_path)
    uuids = [r.uuid for r in records]
    assert len(uuids) == len(set(uuids))
