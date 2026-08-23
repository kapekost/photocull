"""Bulk categories that need no per-photo thought — the whole surface of the sweep.

Pure over `PhotoRecord`: no pixels, no osxphotos, no I/O. That is not a style
preference. Every signal here is metadata the audit command already collects, which is
exactly why the sweep is cheap; the moment a selector needs to look at an image it
belongs in the clustering pipeline instead. That is why the spec's fourth bullet
("singles with sharpness below a threshold") is deliberately absent here -- sharpness
needs pixels, so it lives in the pipeline's scoring, not in this metadata-only sweep.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

from .models import PhotoRecord

SWEEP_SCREENSHOTS = "screenshots"
SWEEP_SHORT_VIDEOS = "short-videos"
SWEEP_EXACT_DUPLICATES = "exact-duplicates"

#: Average days per month, so "older than N months" means the same thing every month.
_DAYS_PER_MONTH = 30.44


@dataclass(frozen=True)
class SweepConfig:
    """Tunables for the sweep. Deliberately separate from `ClusterConfig`: nothing here
    affects clustering, and mixing them would make a sweep setting invalidate every
    recorded review decision through `config_digest`."""

    screenshot_age_months: float = 12.0
    short_video_seconds: float = 5.0
    sweep_screenshots: bool = True
    sweep_short_videos: bool = True
    sweep_exact_duplicates: bool = True


@dataclass(frozen=True)
class SweepGroup:
    category: str
    album: str
    reason: str
    records: tuple[PhotoRecord, ...] = ()

    def __len__(self) -> int:
        return len(self.records)


def _eligible(record: PhotoRecord) -> bool:
    """The three exclusions that apply to every category.

    `is_shared` because Photos' AppleScript surface cannot favourite, album or keyword
    a shared asset — staging one is labour that ends in a silent no-op (the same
    reasoning as `pipeline._candidate_buckets`). `is_favorite` because the owner has
    already said this one stays. `is_hidden` because hiding was a deliberate act and
    the app must not quietly stage it for the owner's manual review."""
    return not (record.is_shared or record.is_favorite or record.is_hidden)


def _ordered(records: Iterable[PhotoRecord]) -> tuple[PhotoRecord, ...]:
    return tuple(sorted(records, key=lambda r: r.uuid))


def _aware(when: datetime | None) -> datetime | None:
    if when is None:
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def screenshot_sweep(
    records: Iterable[PhotoRecord], *, older_than_months: float, now: datetime
) -> SweepGroup:
    cutoff = _aware(now) - timedelta(days=_DAYS_PER_MONTH * older_than_months)
    picked = [
        r
        for r in records
        if (r.is_screenshot or r.is_screen_recording)
        and _eligible(r)
        and (taken := _aware(r.date)) is not None
        and taken < cutoff
    ]
    return SweepGroup(
        category=SWEEP_SCREENSHOTS,
        album="Screenshots",
        reason=f"screenshot or screen recording older than {older_than_months:g} months",
        records=_ordered(picked),
    )


def short_video_sweep(
    records: Iterable[PhotoRecord], *, max_seconds: float
) -> SweepGroup:
    picked = [
        r
        for r in records
        if r.is_video
        and _eligible(r)
        and r.duration_seconds is not None
        and r.duration_seconds < max_seconds
    ]
    return SweepGroup(
        category=SWEEP_SHORT_VIDEOS,
        album="Short Videos",
        reason=f"video shorter than {max_seconds:g}s",
        records=_ordered(picked),
    )


def exact_duplicate_sweep(records: Iterable[PhotoRecord]) -> SweepGroup:
    """Keep the earliest of each byte-identical group, sweep the rest.

    A group is abandoned wholesale if any member is ineligible: with a favourite or a
    shared copy in the group there is no longer an obvious survivor, and picking one
    anyway would be the app making a judgement it has no basis for."""
    groups: dict[str, list[PhotoRecord]] = defaultdict(list)
    for record in records:
        if record.fingerprint:
            groups[record.fingerprint].append(record)

    picked: list[PhotoRecord] = []
    for members in groups.values():
        if len(members) < 2 or not all(_eligible(m) for m in members):
            continue
        # `(date is None, date, uuid)` so undated members sort last deterministically
        # rather than raising on a None comparison.
        ordered = sorted(
            members, key=lambda r: (r.date is None, r.date or datetime.min, r.uuid)
        )
        picked.extend(ordered[1:])

    return SweepGroup(
        category=SWEEP_EXACT_DUPLICATES,
        album="Exact Duplicates",
        reason="byte-identical to an earlier copy (same Photos fingerprint)",
        records=_ordered(picked),
    )


def build_sweep(
    records: Sequence[PhotoRecord], *, config: SweepConfig, now: datetime
) -> tuple[SweepGroup, ...]:
    """Every enabled category, in a fixed order, with each photo in at most one.

    First match wins, and the order is the order of confidence: a screen recording of
    two seconds is a screenshot before it is a short video. Disabled categories still
    appear, empty, so the CLI table reports a 0 rather than dropping a row."""
    records = list(records)
    built = [
        screenshot_sweep(
            records if config.sweep_screenshots else [],
            older_than_months=config.screenshot_age_months,
            now=now,
        ),
        short_video_sweep(
            records if config.sweep_short_videos else [],
            max_seconds=config.short_video_seconds,
        ),
        exact_duplicate_sweep(records if config.sweep_exact_duplicates else []),
    ]

    seen: set[str] = set()
    deduped: list[SweepGroup] = []
    for group in built:
        kept = tuple(r for r in group.records if r.uuid not in seen)
        seen.update(r.uuid for r in kept)
        deduped.append(
            SweepGroup(group.category, group.album, group.reason, kept)
        )
    return tuple(deduped)
