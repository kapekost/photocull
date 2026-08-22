from datetime import datetime

from photocull.models import Cluster, CountSize, PhotoRecord


def test_photo_record_defaults_are_all_false_and_photo_not_video():
    r = PhotoRecord(uuid="abc", date=datetime(2026, 1, 1))
    assert r.is_video is False
    assert r.filesize == 0
    assert r.duration_seconds is None
    assert r.is_screenshot is False
    assert r.is_screen_recording is False
    assert r.is_selfie is False
    assert r.is_burst is False
    assert r.is_live_photo is False
    assert r.is_slow_mo is False
    assert r.is_time_lapse is False
    assert r.is_panorama is False
    assert r.is_raw is False
    assert r.is_favorite is False
    assert r.is_hidden is False
    assert r.album_count == 0


def test_photo_record_overrides_stick():
    r = PhotoRecord(uuid="abc", date=None, is_video=True, duration_seconds=12.5)
    assert r.is_video is True
    assert r.duration_seconds == 12.5


def test_count_size_defaults_to_zero():
    c = CountSize()
    assert c.count == 0
    assert c.size_bytes == 0


def test_count_size_add_accumulates_both_fields():
    c = CountSize()
    c.add(1500)
    c.add(2500)
    assert c.count == 2
    assert c.size_bytes == 4000


def test_photo_record_has_phase1_fields_with_safe_defaults():
    r = PhotoRecord(uuid="u", date=datetime(2025, 1, 1))
    assert r.latitude is None and r.longitude is None
    assert r.device is None and r.burst_key is None
    assert r.mod_date is None
    assert r.width == 0 and r.height == 0


def test_photo_record_accepts_phase1_fields():
    r = PhotoRecord(
        uuid="u",
        date=datetime(2025, 1, 1),
        latitude=37.9,
        longitude=23.7,
        device="iPhone 15 Pro",
        burst_key="burst-1",
        width=4032,
        height=3024,
    )
    assert (r.latitude, r.longitude) == (37.9, 23.7)
    assert r.device == "iPhone 15 Pro"
    assert r.burst_key == "burst-1"
    assert (r.width, r.height) == (4032, 3024)


def test_photo_record_carries_both_derivatives_independently():
    """Two paths to the same photo's pixels, for two different jobs. `derivative_path`
    is the analysis raster and doubles as the analysis cache's key; `display_path` is
    the largest local raster and is only ever put on screen. They default to None
    together (the audit path never resolves either) and are set together, but nothing
    may collapse them into one field -- they name different files for 62.2% of the
    real library."""
    r = PhotoRecord(uuid="u", date=datetime(2025, 1, 1))
    assert r.derivative_path is None and r.display_path is None

    stamped = PhotoRecord(
        uuid="u",
        date=datetime(2025, 1, 1),
        derivative_path="/thumbs/small.jpeg",
        display_path="/thumbs/big.jpeg",
    )
    assert stamped.derivative_path == "/thumbs/small.jpeg"
    assert stamped.display_path == "/thumbs/big.jpeg"


def test_cluster_self_assessment_defaults_are_not_shared_between_instances():
    """`distances_to_winner` and `dropped_criteria` are mutable and the pipeline
    assigns into them per cluster; a bare `{}`/`[]` default would make every cluster
    in a run share one object."""
    a, b = Cluster(), Cluster()
    assert (a.diameter, a.median_distance) == (None, None)
    assert a.sharpness_available is True
    a.distances_to_winner["u1"] = 0.5
    a.dropped_criteria.append("faces")
    assert b.distances_to_winner == {}
    assert b.dropped_criteria == []
