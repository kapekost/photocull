"""Renders an AuditSummary as a compact console table (rich) and as JSON."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .aggregate import DURATION_BUCKETS, TYPE_FLAGS
from .models import AuditSummary, CountSize


def _size_to_dict(c: CountSize) -> dict:
    return {"count": c.count, "size_bytes": c.size_bytes}


def summary_to_json_dict(summary: AuditSummary) -> dict:
    return {
        "total_items": summary.total_items,
        "total_photos": summary.total_photos,
        "total_videos": summary.total_videos,
        "total_size_bytes": summary.total_size_bytes,
        "by_year": {
            str(year): _size_to_dict(cs) for year, cs in sorted(summary.by_year.items())
        },
        "by_type": {name: _size_to_dict(cs) for name, cs in summary.by_type.items()},
        "videos_by_duration": {
            name: _size_to_dict(cs) for name, cs in summary.videos_by_duration.items()
        },
        "favorites_count": summary.favorites_count,
        "no_album_count": summary.no_album_count,
        "hidden_count": summary.hidden_count,
        "estimated_near_dup_clusters": summary.estimated_near_dup_clusters,
        "gap_seconds": summary.gap_seconds,
        "library_path": summary.library_path,
        "generated_at": summary.generated_at.isoformat(),
    }


def write_json(summary: AuditSummary, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(summary_to_json_dict(summary), f, indent=2)


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def render_table(summary: AuditSummary, console: Console | None = None) -> None:
    console = console or Console()

    overview = Table(title="Photo Cull -- Audit Overview", show_header=False)
    overview.add_row("Total items", str(summary.total_items))
    overview.add_row("Photos", str(summary.total_photos))
    overview.add_row("Videos", str(summary.total_videos))
    overview.add_row("Total size", _human_size(summary.total_size_bytes))
    overview.add_row("Favorites", str(summary.favorites_count))
    overview.add_row("Items in no album", str(summary.no_album_count))
    overview.add_row("Hidden", str(summary.hidden_count))
    overview.add_row(
        f"Est. near-duplicate clusters (gap<{summary.gap_seconds:g}s)",
        str(summary.estimated_near_dup_clusters),
    )
    console.print(overview)

    year_table = Table(title="By Year")
    year_table.add_column("Year")
    year_table.add_column("Count", justify="right")
    year_table.add_column("Size", justify="right")
    for year in sorted(summary.by_year):
        cs = summary.by_year[year]
        year_table.add_row(str(year), str(cs.count), _human_size(cs.size_bytes))
    console.print(year_table)

    type_table = Table(title="By Type")
    type_table.add_column("Type")
    type_table.add_column("Count", justify="right")
    type_table.add_column("Size", justify="right")
    for name in TYPE_FLAGS:
        cs = summary.by_type[name]
        type_table.add_row(name, str(cs.count), _human_size(cs.size_bytes))
    console.print(type_table)

    duration_table = Table(title="Videos by Duration")
    duration_table.add_column("Bucket")
    duration_table.add_column("Count", justify="right")
    duration_table.add_column("Size", justify="right")
    for bucket in DURATION_BUCKETS:
        cs = summary.videos_by_duration[bucket]
        duration_table.add_row(bucket, str(cs.count), _human_size(cs.size_bytes))
    console.print(duration_table)
