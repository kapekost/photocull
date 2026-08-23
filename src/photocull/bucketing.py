"""Stage 1 of clustering: the cheap pass. Groups by capture-time proximity using
the same sweep-line the audit command's near-duplicate estimate uses, then
force-joins burst siblings, which are one moment by definition no matter what their
timestamps say.

Unlike the audit estimate, this keeps singletons -- stage 2 needs every record, and
a one-item bucket simply produces no cluster.

Output is fully deterministic: records sort by `(date, uuid)`, never by arrival
order. This is load-bearing, not tidiness -- `PhotosDB.photos()` ordering varies
across processes, and a library can easily have many records sharing a timestamp
with another record, so sorting on date alone would reshuffle the calibration export
between runs."""

from __future__ import annotations

from collections.abc import Iterable

from .config import ClusterConfig
from .models import PhotoRecord


def _sort_key(r: PhotoRecord) -> tuple:
    return (r.date, r.uuid)


def bucket_records(
    records: Iterable[PhotoRecord], config: ClusterConfig | None = None
) -> list[list[PhotoRecord]]:
    """Group records into time+burst buckets.

    Every record with a date appears in exactly one bucket; records without a date
    are dropped (nothing can be said about when they were taken). Buckets are
    ordered by their earliest record, and records within a bucket by `(date, uuid)`.
    """
    cfg = config or ClusterConfig()
    dated = sorted((r for r in records if r.date is not None), key=_sort_key)
    if not dated:
        return []

    buckets: list[list[PhotoRecord]] = [[dated[0]]]
    for prev, cur in zip(dated, dated[1:]):
        gap = (cur.date - prev.date).total_seconds()
        same_burst = cur.burst_key is not None and cur.burst_key == prev.burst_key
        if gap < cfg.gap_seconds or same_burst:
            buckets[-1].append(cur)
        else:
            buckets.append([cur])

    return _join_split_bursts(buckets)


def _join_split_bursts(buckets: list[list[PhotoRecord]]) -> list[list[PhotoRecord]]:
    """Merge buckets that share a burst_key but ended up separated by a large time
    gap (non-adjacent in the sorted sweep).

    Union-find rather than a first-seen map, because merging is transitive: a bucket
    holding two burst keys joins *both* of their groups to each other, not just each
    to itself. A pass that rewrites only the bucket it is looking at leaves the
    second group stranded -- verified failing on the four-record case in
    `tests/test_bucketing.py` before this was rewritten.
    """
    parent = list(range(len(buckets)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Keep the earlier bucket as the root so ordering by earliest date survives.
            parent[max(ra, rb)] = min(ra, rb)

    first_bucket_for_key: dict[str, int] = {}
    for index, bucket in enumerate(buckets):
        keys = {r.burst_key for r in bucket if r.burst_key is not None}
        for key in sorted(keys):  # sorted: set order varies across processes
            if key in first_bucket_for_key:
                union(first_bucket_for_key[key], index)
            else:
                first_bucket_for_key[key] = index

    grouped: dict[int, list[PhotoRecord]] = {}
    for index, bucket in enumerate(buckets):
        grouped.setdefault(find(index), []).extend(bucket)

    return [sorted(grouped[root], key=_sort_key) for root in sorted(grouped)]
