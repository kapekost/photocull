from datetime import datetime, timedelta

import pytest

from photocull.sweep import (
    SWEEP_EXACT_DUPLICATES,
    SWEEP_SCREENSHOTS,
    SWEEP_SHORT_VIDEOS,
    SweepConfig,
    build_sweep,
    exact_duplicate_sweep,
    screenshot_sweep,
    short_video_sweep,
)
from tests.fixtures import make_photo_record

NOW = datetime(2026, 8, 17, 12, 0, 0)


def test_screenshots_older_than_the_cutoff_are_swept():
    old = make_photo_record(is_screenshot=True, date=NOW - timedelta(days=400))
    recent = make_photo_record(is_screenshot=True, date=NOW - timedelta(days=10))
    group = screenshot_sweep([old, recent], older_than_months=12, now=NOW)
    assert [r.uuid for r in group.records] == [old.uuid]


def test_a_favorited_screenshot_is_never_swept():
    kept = make_photo_record(
        is_screenshot=True, is_favorite=True, date=NOW - timedelta(days=400)
    )
    assert screenshot_sweep([kept], older_than_months=12, now=NOW).records == ()


def test_screen_recordings_sweep_with_screenshots():
    rec = make_photo_record(
        is_screen_recording=True, is_video=True, date=NOW - timedelta(days=400)
    )
    group = screenshot_sweep([rec], older_than_months=12, now=NOW)
    assert [r.uuid for r in group.records] == [rec.uuid]


def test_a_screenshot_with_no_date_is_never_swept():
    """A missing date cannot be shown to be older than the cutoff, and the safe
    direction for anything heading toward a manual delete is to leave it alone."""
    undated = make_photo_record(is_screenshot=True, date=None)
    assert screenshot_sweep([undated], older_than_months=12, now=NOW).records == ()


def test_short_videos_under_the_bound_are_swept():
    short = make_photo_record(is_video=True, duration_seconds=3.0)
    long = make_photo_record(is_video=True, duration_seconds=42.0)
    group = short_video_sweep([short, long], max_seconds=5.0)
    assert [r.uuid for r in group.records] == [short.uuid]


def test_the_short_video_bound_is_exclusive():
    exact = make_photo_record(is_video=True, duration_seconds=5.0)
    assert short_video_sweep([exact], max_seconds=5.0).records == ()


def test_a_video_with_no_duration_is_never_swept():
    unknown = make_photo_record(is_video=True, duration_seconds=None)
    assert short_video_sweep([unknown], max_seconds=5.0).records == ()


def test_a_favorited_short_video_is_never_swept():
    kept = make_photo_record(is_video=True, duration_seconds=1.0, is_favorite=True)
    assert short_video_sweep([kept], max_seconds=5.0).records == ()


def test_a_still_is_never_swept_as_a_short_video():
    """`duration_seconds` is None for stills today, but a still that somehow carries
    one must not be swept by the video selector."""
    still = make_photo_record(is_video=False, duration_seconds=1.0)
    assert short_video_sweep([still], max_seconds=5.0).records == ()


def test_exact_duplicates_keep_the_earliest_and_sweep_the_rest():
    first = make_photo_record(fingerprint="abc", date=datetime(2020, 1, 1))
    second = make_photo_record(fingerprint="abc", date=datetime(2021, 1, 1))
    third = make_photo_record(fingerprint="abc", date=datetime(2022, 1, 1))
    group = exact_duplicate_sweep([first, second, third])
    assert [r.uuid for r in group.records] == [second.uuid, third.uuid]


def test_a_lone_fingerprint_sweeps_nothing():
    assert exact_duplicate_sweep([make_photo_record(fingerprint="abc")]).records == ()


def test_items_without_a_fingerprint_never_group():
    """896 real items carry no fingerprint. Two Nones are not a duplicate pair."""
    a = make_photo_record(fingerprint=None)
    b = make_photo_record(fingerprint=None)
    assert exact_duplicate_sweep([a, b]).records == ()


def test_a_favorited_duplicate_is_never_swept():
    first = make_photo_record(fingerprint="abc", date=datetime(2020, 1, 1))
    second = make_photo_record(
        fingerprint="abc", date=datetime(2021, 1, 1), is_favorite=True
    )
    assert exact_duplicate_sweep([first, second]).records == ()


def test_nothing_shared_or_hidden_is_ever_swept():
    """Photos' AppleScript surface cannot act on shared assets at all, and a hidden
    photo was hidden on purpose."""
    shared = make_photo_record(is_screenshot=True, is_shared=True, date=datetime(2020, 1, 1))
    hidden = make_photo_record(is_screenshot=True, is_hidden=True, date=datetime(2020, 1, 1))
    group = screenshot_sweep([shared, hidden], older_than_months=12, now=NOW)
    assert group.records == ()


def test_a_photo_appears_in_at_most_one_group():
    """A screen recording under 5s matches two selectors; it must be staged once."""
    both = make_photo_record(
        is_screen_recording=True,
        is_video=True,
        duration_seconds=2.0,
        date=NOW - timedelta(days=400),
    )
    groups = build_sweep([both], config=SweepConfig(), now=NOW)
    staged = [r.uuid for g in groups for r in g.records]
    assert staged == [both.uuid]


def test_build_sweep_returns_every_category_even_when_empty():
    """The CLI table reports 0 for a category rather than omitting the row."""
    groups = build_sweep([], config=SweepConfig(), now=NOW)
    assert [g.category for g in groups] == [
        SWEEP_SCREENSHOTS,
        SWEEP_SHORT_VIDEOS,
        SWEEP_EXACT_DUPLICATES,
    ]
    assert all(g.records == () for g in groups)


def test_groups_are_ordered_by_uuid_within_a_category():
    """Deterministic across processes: `PhotosDB.photos()` ordering is not
    (DECISIONS.md `within-burst-distance-band-corrected`)."""
    a = make_photo_record(uuid="zzz", is_screenshot=True, date=datetime(2020, 1, 1))
    b = make_photo_record(uuid="aaa", is_screenshot=True, date=datetime(2020, 1, 1))
    group = screenshot_sweep([a, b], older_than_months=12, now=NOW)
    assert [r.uuid for r in group.records] == ["aaa", "zzz"]


@pytest.mark.parametrize("months", [0, 3, 12, 36])
def test_the_cutoff_is_configurable(months):
    at_cutoff = make_photo_record(
        is_screenshot=True, date=NOW - timedelta(days=30.44 * months + 1)
    )
    group = screenshot_sweep([at_cutoff], older_than_months=months, now=NOW)
    assert [r.uuid for r in group.records] == [at_cutoff.uuid]
