from datetime import datetime

from photocull.aggregate import (
    TYPE_FLAGS,
    by_type,
    by_year,
    duration_bucket,
    hidden_count,
    no_album_count,
    favorites_count,
    videos_by_duration,
)

from .fixtures import make_photo_record


def test_by_year_groups_count_and_size():
    records = [
        make_photo_record(date=datetime(2024, 6, 1), filesize=1000),
        make_photo_record(date=datetime(2024, 12, 1), filesize=500),
        make_photo_record(date=datetime(2025, 1, 1), filesize=2000),
    ]
    result = by_year(records)
    assert result[2024].count == 2
    assert result[2024].size_bytes == 1500
    assert result[2025].count == 1
    assert result[2025].size_bytes == 2000


def test_by_year_ignores_records_with_no_date():
    records = [make_photo_record(date=None, filesize=1000)]
    result = by_year(records)
    assert result == {}


def test_type_flags_covers_exactly_the_spec_categories():
    assert set(TYPE_FLAGS) == {
        "screenshot",
        "screen_recording",
        "selfie",
        "burst",
        "live_photo",
        "slow_mo",
        "time_lapse",
        "panorama",
        "raw",
    }


def test_by_type_counts_each_flag_independently_per_item():
    records = [
        make_photo_record(is_screenshot=True, filesize=100),
        make_photo_record(is_screenshot=True, is_selfie=True, filesize=200),
        make_photo_record(filesize=50),
    ]
    result = by_type(records)
    assert result["screenshot"].count == 2
    assert result["screenshot"].size_bytes == 300
    assert result["selfie"].count == 1
    assert result["selfie"].size_bytes == 200
    assert result["burst"].count == 0
    assert result["burst"].size_bytes == 0


def test_duration_bucket_boundaries():
    assert duration_bucket(0) == "<5s"
    assert duration_bucket(4.9) == "<5s"
    assert duration_bucket(5.0) == "5-30s"
    assert duration_bucket(29.9) == "5-30s"
    assert duration_bucket(30.0) == "30s-2m"
    assert duration_bucket(119.9) == "30s-2m"
    assert duration_bucket(120.0) == ">2m"
    assert duration_bucket(600) == ">2m"


def test_videos_by_duration_only_counts_videos_with_known_duration():
    records = [
        make_photo_record(is_video=True, duration_seconds=3, filesize=10),
        make_photo_record(is_video=True, duration_seconds=45, filesize=20),
        make_photo_record(is_video=True, duration_seconds=None, filesize=999),  # unknown, skipped
        make_photo_record(is_video=False, duration_seconds=3, filesize=999),  # not a video, skipped
    ]
    result = videos_by_duration(records)
    assert result["<5s"].count == 1
    assert result["<5s"].size_bytes == 10
    assert result["5-30s"].count == 0
    assert result["30s-2m"].count == 1
    assert result["30s-2m"].size_bytes == 20
    assert result[">2m"].count == 0


def test_favorites_no_album_hidden_counts():
    records = [
        make_photo_record(is_favorite=True, album_count=1, is_hidden=False),
        make_photo_record(is_favorite=False, album_count=0, is_hidden=True),
        make_photo_record(is_favorite=True, album_count=0, is_hidden=True),
    ]
    assert favorites_count(records) == 2
    assert no_album_count(records) == 2
    assert hidden_count(records) == 2
