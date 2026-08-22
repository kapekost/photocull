"""Every tunable threshold in one place -- no magic numbers scattered through the
clustering and scoring logic.

**`similarity_threshold` is a CUT HEIGHT, not a pair-admission threshold** (Task 8c).
`cluster_bucket` cuts a complete-linkage dendrogram at it, so every cluster that comes
out is guaranteed to have **diameter under it** -- no two photos in a cluster are
further apart than this, and therefore no photo is ever staged for culling as a
duplicate of something further away than this. Under the single linkage it replaced,
the value bounded the pairs going IN and a cluster could chain arbitrarily wide: on
the real library at 0.40, 27 photos of diameter 0.7997. Raising it is the recall dial.
See DECISIONS.md `complete-linkage-replaces-single-linkage-and-keeper-radius`.

Defaults are deliberately precision-first. 0.40 is measured against the real library
under SMALLEST-DERIVATIVE selection (Task 7c). The quantity it now bounds is group
diameter, so the numbers to judge it against are the 8 Apple burst groups -- Apple's
own ground-truth take-sets -- whose diameters run 0.1535 / 0.1643 / 0.1919 / 0.2164 /
0.2372 / 0.2934 / 0.2998 / 0.9259: seven of eight fit under a 0.40 cut with 33% margin,
and the outlier is a panning burst rather than one take-set. For context on pairs, the
360 within-burst pairs (all 8 groups, 0 of which now mix raster scale) run min 0.0658 /
p25 0.1178 / median 0.1494 / p75 0.2114 / p90 0.4195 / max 0.9259, while an unrelated
pair measured 1.0398 against a planning band of 0.83-1.26.

Percentiles here are `sorted(d)[int(p/100*n)]` and the median is `statistics.median`;
stated because the two conventions disagree materially on this sample (linear
interpolation puts p90 at 0.3780, not 0.4195) and a band nobody can reproduce is
worse than no band.

This docstring has now been corrected twice, which is the point of saying so. It first
cited a 0.21-0.42 band from a SINGLE pair measured during planning; Task 3 replaced
that with the full distribution (min 0.066 / median 0.167 / p90 0.496 / max 0.926);
Task 7c then re-measured it because the old figures were computed while photos were
being read from whichever derivative `path_derivatives[0]` happened to be -- which
moved measured similarity more than a real difference between two takes does. Do not
re-narrow this claim without re-measuring. **Bursts are pre-joined by `burst_key` at
BUCKETING only** -- `cluster_bucket` has never had any burst awareness, so this value
does gate them, and the 0.9259-diameter panning group is split by design (building a
burst-may-not-split constraint would force that group whole, which is the least safe
option available). Setting the value for real is the owner's call at Task 9; loosen
only against the calibration sample. See DECISIONS.md `cluster-precision-over-recall`,
`within-burst-distance-band-corrected` and `derivative-selection-is-smallest-class`."""

from __future__ import annotations

from dataclasses import dataclass, field


def _default_weights() -> dict[str, float]:
    """Best-shot criteria and their weights. Must sum to 1.0.

    Three criteria ship at 0.0 on purpose. They are listed rather than omitted so the
    owner can enable one by editing a value in `photocull.toml` -- the config loader
    rejects weight keys it does not already know, so a missing key would mean a code
    change instead.

    - `facing` -- the owner's `straightness-three-subscores` sub-score (b), "subject
      facing the camera". It was specified against Vision's `yaw`/`roll`, which Task 6
      measured to be quantized to 45/30 degree steps: two takes of one burst both read
      0.0, so the criterion is constant across the very comparison it exists to make.
      Do NOT rebuild it on those fields. DECISIONS.md `vision-pose-angles-are-quantized`.
    - `capture_quality` -- Apple's own face-quality score, the continuous candidate to
      replace (b). Carried on every face and reported as a diagnostic sub-score, but
      substituting it for the owner's stated criterion is their call at Task 9.
    - `smiling` -- derived from lip landmarks and NOT calibrated: it read negative for
      9 of 12 real selfies, which is not plausibly nine people frowning.

    `facing`'s original 0.10 went to `sharpness` and `faces`, the two criteria whose
    signal is measured and continuous."""
    return {
        "sharpness": 0.35,
        "faces": 0.35,
        "exposure": 0.15,
        "horizon": 0.10,
        "framing": 0.05,
        "facing": 0.00,
        "capture_quality": 0.00,
        "smiling": 0.00,
    }


@dataclass(frozen=True)
class ClusterConfig:
    gap_seconds: float = 90.0
    similarity_threshold: float = 0.40
    ambiguity_margin: float = 0.05
    gps_reinforce_metres: float = 50.0
    #: How close two photos' rasters must be, as a long-side ratio, before their
    #: sharpness may be compared. 1.0 means exact. Laplacian variance is not
    #: comparable across scales and resampling does not make it so (the same photo
    #: through two derivatives disagrees 1.19-1.89x even at an identical 384px), so a
    #: cluster whose members exceed this simply gets no sharpness signal and is
    #: annotated in review instead. Measured after Task 7c's selection change: 77.8%
    #: of multi-item buckets are already at identical scale, 79.8% within 1.05.
    sharpness_size_tolerance: float = 1.0
    #: Whether shared-album assets take part in clustering. Off by default: they are
    #: 27.1% of the staged-for-cull set against 6.2% of the library, and write-back
    #: cannot touch them, so reviewing them is labour that produces nothing.
    include_shared: bool = False
    weights: dict[str, float] = field(default_factory=_default_weights)
