"""Clustering end to end: records in, ranked near-duplicate clusters out.

    bucket (time + burst)
      -> feature print every photo in a bucket that could still form a pair
      -> cluster within the bucket
      -> analyse only the photos that landed in a cluster
      -> score, re-total on common criteria, rank
      -> record what the cluster knows about its own trustworthiness

**Analyzers are keyed by derivative path, never by uuid.** An earlier design used a
`printer` that meant "uuid" in tests and "path" in production, which is this project's
recurring silent-wrongness shape wearing a seam's clothing: the same callable would be
correct in both places for different reasons and wrong in neither obviously. Records
carry `derivative_path` (stamped at scan time by `iter_photo_records`), so tests inject
synthetic paths and exercise the production code path exactly.

**Two stages, two populations, and the gap between them is the cost of the whole
thing.** Feature prints must be computed for every photo sharing a time bucket with
another, because that is what decides the grouping. Faces, sharpness and horizon are
only needed for photos actually being ranked against a sibling, which on a real
library is a minority of everything bucketed -- and per-photo, a feature print is much
cheaper than the three pixel analyzers together. Analysing everything bucketed instead
of everything clustered would meaningfully slow a cold run for no extra answer.

Everything expensive is cached under `(uuid, mod_date, derivative_path)`, so an
interrupted run resumes and a re-run is a table scan."""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .bucketing import bucket_records
from .cache import AnalysisCache
from .config import ClusterConfig
from .imaging import PixelStats
from .models import Cluster, PhotoRecord, PhotoScore
from .scoring import (
    cluster_sharpness,
    exposure_score,
    on_common_criteria,
    rank_cluster,
    score_photo,
)
from .similarity import cluster_bucket
from .vision_backend import FaceObservation, l2_distance

#: stage name, photos done, photos in that stage
ProgressFn = Callable[[str, int, int], None]


@dataclass(frozen=True)
class Analyzers:
    """The four expensive reads, injected so nothing in the pipeline's own logic
    requires Vision, Quartz or a Photos library to test."""

    printer: Callable[[str], list[float] | None]
    stats: Callable[[str], PixelStats | None]
    faces: Callable[[str], list[FaceObservation]]
    horizon: Callable[[str], float | None]


def default_analyzers() -> Analyzers:
    """The real Vision/Quartz backends. Imported lazily so importing this module --
    which the CLI does at startup -- never drags in pyobjc."""
    from .imaging import pixel_stats
    from .vision_backend import detect_faces, feature_print, horizon_angle

    return Analyzers(
        printer=feature_print,
        stats=pixel_stats,
        faces=detect_faces,
        horizon=horizon_angle,
    )


# --- cache payload -------------------------------------------------------------


def _encode(
    stats: PixelStats | None, faces: Sequence[FaceObservation], horizon: float | None
) -> dict[str, Any]:
    return {
        "stats": asdict(stats) if stats is not None else None,
        "faces": [asdict(f) for f in faces],
        "horizon": horizon,
    }


def _decode(
    payload: dict[str, Any],
) -> tuple[PixelStats | None, list[FaceObservation], float | None]:
    raw = payload.get("stats")
    stats = PixelStats(**raw) if raw is not None else None
    faces = []
    for entry in payload.get("faces") or []:
        fields = dict(entry)
        box = fields.get("bounding_box")
        # JSON has no tuples. `framing_score` unpacks this field, and a list unpacks
        # just as happily -- so without the re-tuple a warm run would compute the
        # right answer from an object that no longer equals the cold one. Exactly the
        # kind of difference that stays invisible until something compares them.
        fields["bounding_box"] = tuple(box) if box else (0.0, 0.0, 0.0, 0.0)
        faces.append(FaceObservation(**fields))
    return stats, faces, payload.get("horizon")


# --- stages --------------------------------------------------------------------


def _candidate_buckets(
    records: Iterable[PhotoRecord], cfg: ClusterConfig
) -> list[list[PhotoRecord]]:
    """Buckets reduced to photos that can actually be analysed, keeping only those
    that could still produce a pair. A photo with no local derivative is dropped
    rather than escalated to its original.

    Shared-album assets are dropped here too unless `cfg.include_shared`. They are
    consistently overrepresented in the set staged for culling relative to their share
    of the library, because a shared album is precisely where several takes of one
    moment pile up, and because it holds separate copies of photos that are often
    already in the library. Photos' AppleScript surface cannot favourite, album or
    keyword them either, so every one of them is review labour that ends in a silent
    no-op.

    Dropped at the *bucketing* seam rather than at scan time on purpose: the audit
    still counts them, and `PhotoRecord.is_shared` still travels, so turning the flag
    on is a config edit rather than a re-scan."""
    eligible = (
        r for r in records if cfg.include_shared or not r.is_shared
    )
    usable = (
        [r for r in bucket if r.derivative_path] for bucket in bucket_records(eligible, cfg)
    )
    return [bucket for bucket in usable if len(bucket) >= 2]


def _feature_prints(
    buckets: Sequence[Sequence[PhotoRecord]],
    cache: AnalysisCache,
    analyzers: Analyzers,
    report: ProgressFn,
) -> dict[str, list[float]]:
    total = sum(len(b) for b in buckets)
    vectors: dict[str, list[float]] = {}
    done = 0
    for bucket in buckets:
        for record in bucket:
            key = record.derivative_path
            assert key is not None  # guaranteed by _candidate_buckets
            vector = cache.get_feature_print(record.uuid, record.mod_date, key)
            if vector is None:
                vector = analyzers.printer(key)
                if vector is not None:
                    cache.put_feature_print(record.uuid, record.mod_date, key, vector)
            if vector is not None:
                vectors[record.uuid] = vector
            done += 1
            report("feature-print", done, total)
    return vectors


def _analyse(
    record: PhotoRecord, cache: AnalysisCache, analyzers: Analyzers
) -> tuple[PixelStats | None, list[FaceObservation], float | None]:
    key = record.derivative_path
    assert key is not None
    payload = cache.get_scores(record.uuid, record.mod_date, key)
    if payload is not None:
        return _decode(payload)
    stats = analyzers.stats(key)
    faces = list(analyzers.faces(key))
    horizon = analyzers.horizon(key)
    cache.put_scores(record.uuid, record.mod_date, key, _encode(stats, faces, horizon))
    return stats, faces, horizon


def _annotate(
    cluster: Cluster,
    records: Sequence[PhotoRecord],
    scores: Sequence[PhotoScore],
    vectors: dict[str, list[float]],
    cfg: ClusterConfig,
) -> None:
    """Fill in what the cluster knows about its own trustworthiness. None of this can
    be reconstructed downstream: the vectors are not kept, and `on_common_criteria`
    leaves no record of what it excluded."""
    distances = [
        l2_distance(vectors[a.uuid], vectors[b.uuid])
        for i, a in enumerate(records)
        for b in records[i + 1 :]
    ]
    cluster.diameter = max(distances) if distances else None
    # statistics.median, matching the percentile convention used elsewhere.
    cluster.median_distance = statistics.median(distances) if distances else None

    winner = cluster.winner_uuid
    cluster.distances_to_winner = (
        {}
        if winner is None
        else {
            r.uuid: l2_distance(vectors[r.uuid], vectors[winner])
            for r in records
            if r.uuid != winner
        }
    )

    cluster.sharpness_available = any(
        s.sub_scores.get("sharpness") is not None for s in scores
    )
    # Mirrors on_common_criteria's own rule, minus the zero-weighted signals: `facing`
    # is None on every photo by design, so listing it would annotate every cluster.
    cluster.dropped_criteria = sorted(
        name
        for name, weight in cfg.weights.items()
        if weight > 0 and not all(s.sub_scores.get(name) is not None for s in scores)
    )


def run_pipeline(
    records: Iterable[PhotoRecord],
    *,
    cache: AnalysisCache,
    analyzers: Analyzers | None = None,
    config: ClusterConfig | None = None,
    progress: ProgressFn | None = None,
) -> list[Cluster]:
    """Group a library's records into ranked near-duplicate clusters.

    Ordering is inherited, never re-imposed: bucketing sorts by `(date, uuid)` and
    `cluster_bucket` orders groups by lowest member, so clusters come out in capture
    order. Do not sort the result."""
    cfg = config or ClusterConfig()
    backend = analyzers or default_analyzers()
    report = progress or (lambda stage, done, total: None)

    buckets = _candidate_buckets(records, cfg)
    vectors = _feature_prints(buckets, cache, backend, report)

    groups = [group for bucket in buckets for group in cluster_bucket(bucket, vectors, cfg)]

    clusters: list[Cluster] = []
    total = sum(len(g) for g in groups)
    done = 0
    for group in groups:
        analysis = []
        for record in group:
            analysis.append(_analyse(record, cache, backend))
            done += 1
            report("analyze", done, total)

        sharpness = cluster_sharpness([a[0] for a in analysis], cfg)
        scores = [
            score_photo(
                record.uuid,
                faces=faces,
                horizon=horizon,
                sharpness=sharpness[i],
                exposure=exposure_score(stats),
                config=cfg,
            )
            for i, (record, (stats, faces, horizon)) in enumerate(zip(group, analysis))
        ]
        cluster = rank_cluster(group, on_common_criteria(scores, cfg), cfg)
        _annotate(cluster, group, scores, vectors, cfg)
        clusters.append(cluster)

    return clusters
