"""Every tunable threshold in one place -- no magic numbers scattered through the
clustering and scoring logic.

**`similarity_threshold` is a CUT HEIGHT, not a pair-admission threshold.**
`cluster_bucket` cuts a complete-linkage dendrogram at it, so every cluster that comes
out is guaranteed to have **diameter under it** -- no two photos in a cluster are
further apart than this, and therefore no photo is ever staged for culling as a
duplicate of something further away than this. Under the single linkage it replaced,
the value bounded the pairs going IN and a cluster could chain arbitrarily wide,
producing large, high-diameter clusters that grouped photos which didn't actually
resemble each other. Raising it is the recall dial.

Defaults are deliberately precision-first, calibrated against a real library under
SMALLEST-DERIVATIVE selection. The quantity it bounds is group diameter, so the right
way to judge it is against real burst groups -- Apple's own ground-truth take-sets:
most fit comfortably under the default cut, with a healthy margin, and the rare
outlier tends to be a panning burst rather than a genuine take-set. Same-burst pairs
cluster at much smaller distances than the cut height, which is what leaves headroom
for the cut to sit where it does.

Percentiles here use `sorted(d)[int(p/100*n)]`, and the median uses
`statistics.median`; stated because the two conventions can disagree materially on the
same sample, and a band nobody can reproduce is worse than no band.

This default has been revisited more than once, which is the point of saying so: an
early version was picked from a single measured pair, then replaced with the full
distribution once more data was available, then re-measured again because the earlier
figures were computed while photos were being read from whichever derivative
`path_derivatives[0]` happened to be -- which moved measured similarity more than a
real difference between two takes does. Do not re-narrow this claim without
re-measuring. **Bursts are pre-joined by `burst_key` at BUCKETING only** --
`cluster_bucket` has never had any burst awareness, so this value does gate them, and
a burst that's a wide panning shot is allowed to split by design (forcing it whole
would be the least safe option available). Setting the value for a specific library is
the owner's call, tuned against their own calibration sample -- the shipped default is
deliberately generic, not tuned to any one person's photos."""

from __future__ import annotations

from dataclasses import dataclass, field


def _default_weights() -> dict[str, float]:
    """Best-shot criteria and their weights. Must sum to 1.0.

    Three criteria ship at 0.0 on purpose. They are listed rather than omitted so the
    owner can enable one by editing a value in `photocull.toml` -- the config loader
    rejects weight keys it does not already know, so a missing key would mean a code
    change instead.

    - `facing` -- one of the three straightness sub-scores, "subject facing the
      camera". It was specified against Vision's `yaw`/`roll`, which turned out to be
      quantized to 45/30 degree steps: two takes of the same burst can both read 0.0,
      so the criterion is constant across the very comparison it exists to make.
      Do NOT rebuild it on those fields.
    - `capture_quality` -- Apple's own face-quality score, the continuous candidate to
      replace `facing`. Carried on every face and reported as a diagnostic sub-score,
      but substituting it for the owner's stated criterion is their call to make.
    - `smiling` -- derived from lip landmarks and NOT calibrated: it reads negative far
      more often than real photos are actually frowning, so it isn't trustworthy yet.

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
    #: annotated in review instead. Most multi-item buckets are already at identical
    #: scale under smallest-derivative selection, so this rarely costs a signal.
    sharpness_size_tolerance: float = 1.0
    #: Whether shared-album assets take part in clustering. Off by default: they are
    #: overrepresented in the staged-for-cull set relative to the library as a whole,
    #: and write-back cannot touch them, so reviewing them is labour that produces
    #: nothing.
    include_shared: bool = False
    weights: dict[str, float] = field(default_factory=_default_weights)
