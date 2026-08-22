"""Composes aggregate.py + cluster_estimate.py into one AuditSummary. This is the
single public entry point report.py and cli.py call."""

from __future__ import annotations

from collections.abc import Iterable

from .aggregate import (
    by_type,
    by_year,
    favorites_count,
    hidden_count,
    no_album_count,
    videos_by_duration,
)
from .cluster_estimate import count_near_duplicate_clusters
from .models import AuditSummary, PhotoRecord

DEFAULT_GAP_SECONDS = 90.0


def build_summary(
    records: Iterable[PhotoRecord],
    *,
    gap_seconds: float = DEFAULT_GAP_SECONDS,
    library_path: str | None = None,
) -> AuditSummary:
    records = list(records)  # consumed multiple times below

    total_size = sum(r.filesize for r in records)
    total_videos = sum(1 for r in records if r.is_video)

    return AuditSummary(
        total_items=len(records),
        total_photos=len(records) - total_videos,
        total_videos=total_videos,
        total_size_bytes=total_size,
        by_year=by_year(records),
        by_type=by_type(records),
        videos_by_duration=videos_by_duration(records),
        favorites_count=favorites_count(records),
        no_album_count=no_album_count(records),
        hidden_count=hidden_count(records),
        estimated_near_dup_clusters=count_near_duplicate_clusters(
            records, gap_seconds=gap_seconds
        ),
        gap_seconds=gap_seconds,
        library_path=library_path,
    )
