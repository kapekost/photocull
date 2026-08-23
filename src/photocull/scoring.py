"""Best-shot scoring: which take of a near-duplicate cluster is the keeper.

Emits per-criterion sub-scores, never just a total, so the review UI can show *why* a
take won (spec requirement). A sub-score of None means "this criterion does not apply
to this image" -- no faces, no detectable horizon, or a measurement Vision declined to
make. Those are dropped and the remaining weights renormalised, so a landscape is not
punished for having no faces.

This module is deliberately PURE: it never opens an image. Face observations, horizon
angle and the pixel-derived sharpness/exposure numbers are passed in by the caller.
That keeps every rule here testable with plain floats, and it keeps image access
inside the one chokepoint that is allowed to do it (`derivatives.py`).

The two pixel criteria are computed here but the pixels are not read here: the caller
passes `imaging.pixel_stats` output in, and `exposure_score`/`cluster_sharpness` turn
it into 0-1 numbers. They are deliberately different kinds of signal.

- `exposure` is ABSOLUTE and per-photo. Exposure statistics are scale-invariant
  (verified across a 2.1x scale change: mean luma moves a median of 0.06/255), so it
  needs no comparability gate and survives clusters whose derivatives differ in size.
- `sharpness` is CLUSTER-RELATIVE and SIZE-GATED. Laplacian variance is not comparable
  between rasters of different size and resampling does not make it so, so a cluster
  whose members' rasters differ by more than `sharpness_size_tolerance` gets no
  sharpness signal at all rather than a misleading one."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .config import ClusterConfig
from .imaging import PixelStats
from .models import Cluster, PhotoRecord, PhotoScore
from .vision_backend import FaceObservation, framing_score, horizon_penalty


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def weighted_total(sub_scores: dict[str, float | None], config: ClusterConfig) -> float:
    """Weighted mean over the criteria that apply, renormalised to the weights used.

    Keys absent from `config.weights` are ignored entirely, which is what lets
    `capture_quality` ride along as a diagnostic without moving anyone's total."""
    applicable = {
        k: v for k, v in sub_scores.items() if v is not None and k in config.weights
    }
    total_weight = sum(config.weights[k] for k in applicable)
    if total_weight <= 0:
        return 0.0
    return sum(config.weights[k] * v for k, v in applicable.items()) / total_weight


def faces_score(faces: Sequence[FaceObservation]) -> float | None:
    """Eye-openness, driven by the WORST eye of the worst face in frame.

    The owner's requirement is that everyone is looking at the camera with their eyes
    open, so one blinker must sink the take -- an average would let a crowd of good
    faces hide it, and averaging a person's two eyes would hide a half-blink. Closed
    eyes are the strongest reject-this-take signal in the spec.

    Deliberately ignores `smiling`: that score is not calibrated (see
    `ClusterConfig.weights`), and folding it in here would give it weight through the
    back door where no config value could disable it.

    None when there are no faces, and also when Vision measured no eye values at all --
    "not measured" must not read as "eyes shut"."""
    if not faces:
        return None
    per_face = []
    for f in faces:
        eyes = [v for v in (f.left_eye_open, f.right_eye_open) if v is not None]
        if eyes:
            per_face.append(_clamp(min(eyes)))
    if not per_face:
        return None
    return min(per_face)


def smile_score(faces: Sequence[FaceObservation]) -> float | None:
    """Worst smile in frame, mapped from `smile_curvature`'s signed value into 0..1.

    Computed and reported so the owner can inspect it against their own photos, but it
    carries zero weight until it's actually calibrated. None when there are no faces
    or nothing was measured."""
    if not faces:
        return None
    values = [f.smiling for f in faces if f.smiling is not None]
    if not values:
        return None
    return _clamp(min(values) / 2.0 + 0.5)


def capture_quality_score(faces: Sequence[FaceObservation]) -> float | None:
    """Apple's own face-quality score for the worst face, carried as a diagnostic.

    Measured as continuous across a real range of faces, so it is comparable between
    takes -- unlike yaw/roll. Not weighted; see `ClusterConfig.weights`."""
    if not faces:
        return None
    values = [f.capture_quality for f in faces if f.capture_quality is not None]
    if not values:
        return None
    return _clamp(min(values))


#: Where the tone curve peaks, and how fast it falls. The peak is set from the
#: measured median of real photos rather than a notional 0.5 "correct" exposure, and
#: the width is chosen so a real library's typical dim-to-bright range still scores
#: reasonably well instead of being crushed. Uncalibrated against taste -- that's the
#: owner's call, checked against their own clusters.
_LUMA_PEAK = 0.46
_LUMA_WIDTH = 0.18
#: Clipping is a defect at any level, but a few clipped pixels are normal, so the
#: penalty is gentle and saturates.
_CLIP_WEIGHT = 3.0


def exposure_score(stats: PixelStats | None) -> float | None:
    """How well-exposed a frame is, on 0-1. None when the photo could not be read.

    Absolute rather than cluster-relative, and deliberately so: exposure statistics are
    scale-invariant (verified across a real scale change -- mean luma barely moved),
    unlike sharpness. So exposure stays comparable even in the minority of buckets
    whose members' derivatives differ in scale."""
    if stats is None:
        return None
    tone = math.exp(-((stats.mean_luma - _LUMA_PEAK) ** 2) / (2 * _LUMA_WIDTH**2))
    clipped = stats.shadow_clipped + stats.highlight_clipped
    penalty = min(1.0, _CLIP_WEIGHT * clipped)
    return _clamp(tone * (1.0 - penalty))


def cluster_sharpness(
    stats: Sequence[PixelStats | None], config: ClusterConfig | None = None
) -> list[float | None]:
    """Sharpness for every take in one cluster, as a ratio to the sharpest of them.

    Two decisions worth not undoing:

    **Relative, not absolute.** Laplacian variance has no meaningful absolute scale, so
    there is no honest way to map one photo's value onto 0-1 alone. Within a cluster
    there is: the best take is 1.0 and the others are what fraction of it they reach.
    That is also all the app needs -- takes are only ever compared with their own
    cluster.

    **Ratio, not min-max.** Real take-to-take spread within a burst is often small.
    Min-max normalisation would stretch a small difference across the full 0-1 range
    and declare a confident winner where there is none; ratio-to-best reports the true,
    modest gap instead. Even a burst with much wider spread than usual still lands with
    a real difference dominating the ratio, so genuinely soft takes are still caught.

    Returns all-None when the cluster's rasters are not comparable -- see the module
    note on why that is a drop rather than a discount."""
    cfg = config or ClusterConfig()
    if not stats:
        return []
    if any(s is None for s in stats):
        return [None] * len(stats)
    # Long side, not (width, height): a portrait and a landscape take of the same pixel
    # class are comparable, and PixelStats reports post-EXIF-transform dimensions.
    long_sides = [max(s.width, s.height) for s in stats]  # type: ignore[union-attr]
    if min(long_sides) <= 0:
        return [None] * len(stats)
    if max(long_sides) / min(long_sides) > cfg.sharpness_size_tolerance:
        return [None] * len(stats)
    variances = [s.laplacian_variance for s in stats]  # type: ignore[union-attr]
    best = max(variances)
    if best <= 0:
        # A cluster of featureless frames. Ranking them on sharpness would be division
        # by noise; drop the criterion and let on_common_criteria renormalise.
        return [None] * len(stats)
    return [_clamp(v / best) for v in variances]


def score_photo(
    uuid: str,
    *,
    faces: Sequence[FaceObservation] | None = None,
    horizon: float | None = None,
    sharpness: float | None = None,
    exposure: float | None = None,
    config: ClusterConfig | None = None,
) -> PhotoScore:
    """Assemble one photo's sub-scores and its weighted total.

    `sharpness` and `exposure` are passed in rather than computed: they need pixels,
    and this module never opens an image. Passing None for either simply drops that
    criterion from the total."""
    cfg = config or ClusterConfig()
    faces = list(faces or [])
    sub_scores: dict[str, float | None] = {
        "sharpness": sharpness,
        "exposure": exposure,
        "faces": faces_score(faces),
        "framing": framing_score(faces),
        "horizon": horizon_penalty(horizon),
        "facing": None,  # see ClusterConfig.weights -- owner's call whether to enable it
        "capture_quality": capture_quality_score(faces),
        "smiling": smile_score(faces),
    }
    return PhotoScore(uuid=uuid, sub_scores=sub_scores, total=weighted_total(sub_scores, cfg))


def on_common_criteria(
    scores: Sequence[PhotoScore], config: ClusterConfig | None = None
) -> list[PhotoScore]:
    """Re-total a cluster's takes over only the criteria measured for ALL of them.

    Without this, a criterion can win a cluster by merely being *present*. `score_photo`
    renormalises over whatever applied to that one photo, which is correct in isolation
    but not across takes: dropping a criterion that scores high raises the renormalised
    mean, and dropping one that scores low lowers it.

    Seen on a real burst, which is what prompted this function: Vision detected a
    horizon in most but not all of a set of near-identical frames, and a horizon score
    is typically much higher than an eye-openness score, so the frames that happened to
    get a horizon detection totalled noticeably higher than the ones that didn't. The
    winner was decided by detection luck, not by anything visible in the photos.

    `sub_scores` are passed through untouched, so the UI still shows every measurement;
    only `total` changes. Note this makes a total meaningful *within* its cluster and
    not across clusters -- which is all the app ever needs, since takes are only ever
    compared with their own cluster."""
    cfg = config or ClusterConfig()
    if not scores:
        return []
    common = {
        k for k in cfg.weights
        if all(s.sub_scores.get(k) is not None for s in scores)
    }
    return [
        PhotoScore(
            uuid=s.uuid,
            sub_scores=s.sub_scores,
            total=weighted_total({k: s.sub_scores[k] for k in common}, cfg),
        )
        for s in scores
    ]


def rank_cluster(
    records: Sequence[PhotoRecord],
    scores: Sequence[PhotoScore],
    config: ClusterConfig | None = None,
) -> Cluster:
    """Rank a cluster's takes and decide whether the pick is trustworthy.

    Sorted by descending total with the uuid as tiebreak, so the ordering is a total
    order that does not depend on the order the scores arrived in. That matters: this
    project has shipped ordering defects before, from output order tracking input
    order. An exact tie is always inside any sane ambiguity margin anyway, so it gets
    flagged for a manual pick rather than settled by the tiebreak.

    `is_ambiguous` means the top two are within `ambiguity_margin` -- the scorer cannot
    tell them apart, so the UI must not auto-pick.

    Ranks on the totals it is given. Feed it through `on_common_criteria` first unless
    you have a reason not to -- see that function for what goes wrong otherwise."""
    cfg = config or ClusterConfig()
    ordered = sorted(scores, key=lambda s: (-s.total, s.uuid))
    ambiguous = (
        len(ordered) >= 2 and (ordered[0].total - ordered[1].total) < cfg.ambiguity_margin
    )
    return Cluster(
        records=list(records),
        scores=ordered,
        winner_uuid=ordered[0].uuid if ordered else None,
        is_ambiguous=ambiguous,
    )
