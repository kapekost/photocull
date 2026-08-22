from datetime import datetime

from photocull.summary import build_summary

from .fixtures import make_photo_record


def test_build_summary_totals_and_split():
    records = [
        make_photo_record(date=datetime(2025, 1, 1), filesize=100, is_video=False),
        make_photo_record(date=datetime(2025, 1, 1), filesize=200, is_video=True, duration_seconds=3),
        make_photo_record(date=datetime(2025, 1, 1), filesize=300, is_video=True, duration_seconds=40),
    ]
    summary = build_summary(records, library_path="/fake/lib.photoslibrary")
    assert summary.total_items == 3
    assert summary.total_photos == 1
    assert summary.total_videos == 2
    assert summary.total_size_bytes == 600
    assert summary.library_path == "/fake/lib.photoslibrary"
    assert summary.gap_seconds == 90.0


def test_build_summary_wires_in_by_year_by_type_and_duration_buckets():
    records = [
        make_photo_record(date=datetime(2024, 1, 1), is_screenshot=True, filesize=10),
        make_photo_record(date=datetime(2025, 1, 1), is_video=True, duration_seconds=3, filesize=20),
    ]
    summary = build_summary(records)
    assert summary.by_year[2024].count == 1
    assert summary.by_year[2025].count == 1
    assert summary.by_type["screenshot"].count == 1
    assert summary.videos_by_duration["<5s"].count == 1


def test_build_summary_wires_in_favorites_no_album_hidden():
    records = [
        make_photo_record(is_favorite=True, album_count=0, is_hidden=True),
    ]
    summary = build_summary(records)
    assert summary.favorites_count == 1
    assert summary.no_album_count == 1
    assert summary.hidden_count == 1


def test_build_summary_wires_in_cluster_estimate_with_configurable_gap():
    close = [
        make_photo_record(date=datetime(2025, 1, 1, 12, 0, 0)),
        make_photo_record(date=datetime(2025, 1, 1, 12, 0, 10)),
    ]
    summary = build_summary(close, gap_seconds=90)
    assert summary.estimated_near_dup_clusters == 1

    summary_tight = build_summary(close, gap_seconds=5)
    assert summary_tight.estimated_near_dup_clusters == 0
    assert summary_tight.gap_seconds == 5


def test_build_summary_handles_empty_library():
    summary = build_summary([])
    assert summary.total_items == 0
    assert summary.total_photos == 0
    assert summary.total_videos == 0
    assert summary.total_size_bytes == 0
    assert summary.estimated_near_dup_clusters == 0
    assert summary.by_year == {}
