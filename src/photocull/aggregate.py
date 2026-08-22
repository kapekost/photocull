"""Pure aggregation functions for the Phase 0 audit -- every function here takes an
Iterable[PhotoRecord] and returns plain counts/dicts. No osxphotos import, no I/O."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from .models import CountSize, PhotoRecord

# Exactly the 9 categories the spec lists under "Counts by type". Order here is the
# order they'll render in the table.
TYPE_FLAGS: dict[str, Callable[[PhotoRecord], bool]] = {
    "screenshot": lambda r: r.is_screenshot,
    "screen_recording": lambda r: r.is_screen_recording,
    "selfie": lambda r: r.is_selfie,
    "burst": lambda r: r.is_burst,
    "live_photo": lambda r: r.is_live_photo,
    "slow_mo": lambda r: r.is_slow_mo,
    "time_lapse": lambda r: r.is_time_lapse,
    "panorama": lambda r: r.is_panorama,
    "raw": lambda r: r.is_raw,
}

DURATION_BUCKETS = ("<5s", "5-30s", "30s-2m", ">2m")


def by_year(records: Iterable[PhotoRecord]) -> dict[int, CountSize]:
    result: dict[int, CountSize] = {}
    for r in records:
        if r.date is None:
            continue
        result.setdefault(r.date.year, CountSize()).add(r.filesize)
    return result


def by_type(records: Iterable[PhotoRecord]) -> dict[str, CountSize]:
    result: dict[str, CountSize] = {name: CountSize() for name in TYPE_FLAGS}
    for r in records:
        for name, flag in TYPE_FLAGS.items():
            if flag(r):
                result[name].add(r.filesize)
    return result


def duration_bucket(seconds: float) -> str:
    if seconds < 5:
        return "<5s"
    if seconds < 30:
        return "5-30s"
    if seconds < 120:
        return "30s-2m"
    return ">2m"


def videos_by_duration(records: Iterable[PhotoRecord]) -> dict[str, CountSize]:
    result: dict[str, CountSize] = {name: CountSize() for name in DURATION_BUCKETS}
    for r in records:
        if not r.is_video or r.duration_seconds is None:
            continue
        result[duration_bucket(r.duration_seconds)].add(r.filesize)
    return result


def favorites_count(records: Iterable[PhotoRecord]) -> int:
    return sum(1 for r in records if r.is_favorite)


def no_album_count(records: Iterable[PhotoRecord]) -> int:
    return sum(1 for r in records if r.album_count == 0)


def hidden_count(records: Iterable[PhotoRecord]) -> int:
    return sum(1 for r in records if r.is_hidden)
