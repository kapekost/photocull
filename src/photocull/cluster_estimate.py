"""Phase 0's near-duplicate cluster estimate: a fast time-proximity pass only, no
image analysis. Deliberately the same sweep-line chaining Phase 1's bucketing stage
will use (minus burst/live-photo pre-join and GPS/same-device reinforcement, which
are Phase-1-only signals) -- this number previews what Phase 1 will actually cluster."""

from __future__ import annotations

from collections.abc import Iterable

from .models import PhotoRecord


def count_near_duplicate_clusters(
    records: Iterable[PhotoRecord], *, gap_seconds: float = 90.0
) -> int:
    """Count groups of >=2 items whose capture times chain together with consecutive
    gaps strictly under `gap_seconds`. Records with no date are ignored. A cluster of
    exactly 1 (a singleton) is not counted -- it isn't a near-duplicate of anything."""
    dates = sorted(r.date for r in records if r.date is not None)
    if len(dates) < 2:
        return 0

    clusters = 0
    run_length = 1
    for previous, current in zip(dates, dates[1:]):
        gap = (current - previous).total_seconds()
        if gap < gap_seconds:
            run_length += 1
        else:
            if run_length >= 2:
                clusters += 1
            run_length = 1
    if run_length >= 2:
        clusters += 1
    return clusters
