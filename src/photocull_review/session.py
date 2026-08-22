"""The review session document: `list[Cluster]` in, the thing the browser reads out.

Every later Phase 1b task binds to this shape, so two properties are structural rather
than incidental:

**It is built by reusing `calibrate.sample_to_dict`.** That function already carries a
fix that cost a tick to find — `Cluster.records` and `Cluster.scores` are *not*
index-aligned, because `rank_cluster` sorts the scores while the records keep bucket
order, so joining them by position gives every photo another photo's sub-scores. It
reads as entirely plausible and is wrong throughout. `build_session` enriches
`sample_to_dict`'s entries and never re-derives a member.

**It carries no filesystem paths.** The calibration sample embeds `derivative_path`
because its contact sheet reads files off disk directly; this document is served over
HTTP to a browser, and the review UI fetches images by uuid (Task 7). Stripping happens
here, in the builder every consumer goes through, rather than at serialisation time in a
later task where omitting it would be silent.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from photocull.calibrate import sample_to_dict
from photocull.config import ClusterConfig
from photocull.config_io import describe_config
from photocull.imaging import image_dimensions
from photocull.models import Cluster, PhotoRecord

from .explain import explain, ranking_basis

#: Bumped when the document's shape changes in a way a stored session cannot be read
#: back under. Task 5's resume check reads it.
SESSION_VERSION = 1

#: Length of the hex digests below. 16 hex chars = 64 bits, which is far more than
#: enough to keep a few thousand cluster keys distinct and short enough to read in a
#: URL and in a failure message.
DIGEST_CHARS = 16

#: A cluster whose largest local raster is no wider than this cannot be examined in any
#: more detail than the calibration contact sheet already showed. Measured on the real
#: library at the live 0.48 cut: 25.3% of clusters (457 of 1,807). The remedy is Task
#: 12's explicit full-resolution fetch, not magnification, so the owner needs to know
#: which clusters they are before spending time squinting at one.
LOW_DETAIL_MAX_LONG_SIDE = 640

#: Two takes whose display rasters differ by more than this factor are being compared at
#: different scales, so detail visible in one may be absent from the other rather than
#: absent from the photograph. Same tolerance the analysis side uses for sharpness.
#: Measured: 7.7% of real clusters.
MIXED_RESOLUTION_RATIO = 1.35

#: Measured: 9 clusters (0.5%) at the live cut, largest 30 photos. Rare enough to be
#: worth flagging and consequential enough to want flagged — one accepted proposal here
#: stages 29 photos.
LARGE_CLUSTER_SIZE = 10

#: Prose, never an icon (`low-confidence-clusters-are-annotated-in-review`). Each says
#: what is weak *and* what to do about it, because "low confidence" on its own gives the
#: owner nowhere to go.
ANNOTATION_TEXT: dict[str, str] = {
    "ambiguous": (
        "A close call — the top two takes scored within the confidence margin of each "
        # Deliberately says no direction. Task 10 moved this note under the keeper's
        # score table (it had been between the image and the numbers, which knocked the
        # two panes' score rows out of line on the 40.1% of clusters that are close
        # calls), and "below" then pointed at the wrong thing. Prose that names a
        # position is prose that a layout change silently falsifies.
        "other, so treat this proposal as a coin flip and compare them yourself."
    ),
    "no-sharpness": (
        "No sharpness signal for this cluster — the takes could not be compared on "
        "focus, so the ranking rests on the other criteria alone."
    ),
    "low-detail": (
        "No take in this cluster has a local copy above "
        f"{LOW_DETAIL_MAX_LONG_SIDE}px, so this is as much detail as there is to see "
        "without fetching the full-resolution original."
    ),
    "mixed-resolution": (
        "These takes have local copies at different sizes, so one may look softer "
        "than the other because of its copy rather than because of the photograph."
    ),
    "large-cluster": (
        "A large cluster: accepting the proposal as it stands stages every other take "
        "here, so it is worth stepping through them."
    ),
    "no-display": (
        "One take has no local copy at all and cannot be shown — decide on the others, "
        "or fetch its original."
    ),
}


@dataclass(frozen=True)
class DisplayInfo:
    """Where a photo's largest local raster is, and how big it actually is.

    `local_path` is deliberately not spelled the obvious way: GUARDRAILS' standing
    iCloud grep matches the attribute osxphotos uses for the original file, so a field
    named that here would report a false positive on every future run — and a safety
    check that cries wolf stops being read (CLAUDE.md hard rule #2).

    `long_side` is the longest edge and nothing finer, because `image_dimensions` reads
    ImageIO's pre-orientation header: for a portrait photo the width and height can come
    back transposed, which leaves `max()` correct and a "1024x768" label wrong. The
    browser knows the true post-orientation size from the decoded image, so the document
    only carries the quantity every decision here actually rests on."""

    uuid: str
    local_path: str | None
    long_side: int | None

    @property
    def available(self) -> bool:
        return self.long_side is not None


#: How `build_session` turns a record into its display info. Injectable so tests can
#: exercise the annotation *policy* without writing rasters for every case.
Resolver = Callable[[PhotoRecord], DisplayInfo]


def cluster_key(cluster: Cluster) -> str:
    """A stable identity for a cluster, derived from its membership alone.

    Order-independent on purpose. Bucket order is deterministic today
    (`bucketing-is-deterministic-and-burst-closed`), but this key is what a decision
    recorded on Monday is looked up by on Thursday, and the only thing that should
    invalidate it is the cluster genuinely containing different photos."""
    joined = "\n".join(sorted(record.uuid for record in cluster.records))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:DIGEST_CHARS]


def session_settings(cfg: ClusterConfig, source: str | Path | None = None) -> dict[str, Any]:
    """The settings a clustering was produced under — `source` excluded.

    `describe_config` stamps where the config was loaded from, which is a path on one
    machine; two runs of identical settings must not read as a settings change.

    Split out from `config_digest` because Task 5 has to *name the field that moved*, and
    a digest is a one-way function. The decision log stores this dict beside the digest,
    on the same reasoning that makes `sub_scores` a snapshot rather than a join: the
    record of what the owner was deciding under has to survive the settings changing."""
    described = describe_config(cfg, source)
    described.pop("source", None)
    return described


def digest_settings(settings: Mapping[str, Any]) -> str:
    """Digest an already-described settings dict.

    Goes through `json.dumps(..., sort_keys=True)`, which recurses into `weights`,
    because a re-tuning under `review-decisions-train-the-scorer` changes only that
    nested dict."""
    payload = json.dumps(dict(settings), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:DIGEST_CHARS]


def config_digest(cfg: ClusterConfig, source: str | Path | None = None) -> str:
    """A digest of the settings that produced a clustering — `source` excluded.

    Task 5 refuses to resume a session when this moves, so it has to move for exactly the
    right reasons. Defined as the digest *of* `session_settings` rather than alongside
    it, so the stored snapshot and the stored digest cannot describe different configs."""
    return digest_settings(session_settings(cfg, source))


def resolve_display(record: PhotoRecord) -> DisplayInfo:
    """Measure the raster the review UI will show for `record`.

    Reads only `display_path`, stamped at scan time by `derivative_path`'s sibling
    selector, so nothing here can reach the original and trigger an iCloud fetch. A path
    that no longer resolves degrades to "cannot show this": derivatives are a cache
    Photos evicts and regenerates, so a stale one is an ordinary Tuesday, not an error."""
    local_path = record.display_path
    if not local_path:
        return DisplayInfo(record.uuid, None, None)
    dims = image_dimensions(local_path)
    if dims is None:
        return DisplayInfo(record.uuid, local_path, None)
    return DisplayInfo(record.uuid, local_path, max(dims))


def annotations(cluster: Cluster, displays: dict[str, DisplayInfo]) -> list[dict[str, str]]:
    """Why this cluster deserves more attention than the last one, in words.

    **The policy is a measurement.** Seven candidate signals were measured over all
    1,807 clusters of the real library at the live 0.48 cut, and taking all of them
    would annotate **96.0%** of clusters — a badge on nineteen clusters in twenty says
    nothing at all, which is the lesson `unvalidated-signals-ship-at-zero-weight` already
    taught once. Two candidates were therefore demoted to the surface where they are
    actionable, exactly as the Phase 1b plan's Step 5 requires, and neither is discarded:

    - `dropped_criteria` is non-empty for **90.3%** of clusters (`horizon` alone 80.9%,
      `faces`/`framing` 53.1%). That is the background condition of this library, not a
      warning about one cluster. It stays a field on every entry and becomes Task 11's
      evidence panel — "ranked on sharpness and exposure" is useful prose; a warning
      triangle on nine clusters in ten is not.
    `is_ambiguous` (**40.1%**) was demoted the same way and has since been **promoted
    back**, because the surface it was demoted *to* no longer exists. The reasoning was
    "`proposed_marks` already leaves every take unset, so the UI shows a pick-one screen
    instead of a confirm screen" — true until `every-cluster-opens-with-a-suggestion`
    made every cluster a confirm screen. An ambiguous cluster is now byte-identical in
    shape to a confident one, so without this note a coin flip would present exactly as
    a considered recommendation. That is the silent-plausible-wrongness shape this
    project has now hit seven times, and it is why the badge is not optional here.

    What is left annotates **34.6%** of clusters on the attention signals, plus the
    40.1% that are close calls — each for a reason the owner can act on.
    """
    notes: list[dict[str, str]] = []
    sides = [displays[record.uuid].long_side for record in cluster.records]
    known = [side for side in sides if side]

    if cluster.is_ambiguous:
        notes.append({"code": "ambiguous", "text": ANNOTATION_TEXT["ambiguous"]})
    if any(side is None for side in sides):
        notes.append({"code": "no-display", "text": ANNOTATION_TEXT["no-display"]})
    if not cluster.sharpness_available:
        notes.append({"code": "no-sharpness", "text": ANNOTATION_TEXT["no-sharpness"]})
    if known and max(known) <= LOW_DETAIL_MAX_LONG_SIDE:
        notes.append({"code": "low-detail", "text": ANNOTATION_TEXT["low-detail"]})
    if known and max(known) > min(known) * MIXED_RESOLUTION_RATIO:
        notes.append({"code": "mixed-resolution", "text": ANNOTATION_TEXT["mixed-resolution"]})
    if len(cluster.records) >= LARGE_CLUSTER_SIZE:
        notes.append({"code": "large-cluster", "text": ANNOTATION_TEXT["large-cluster"]})
    return notes


def proposed_marks(cluster: Cluster) -> dict[str, str]:
    """The scorer's opening position: `keep` / `cull` / `unset` per photo.

    A proposal, never a verdict — `cull-selection-is-propose-and-adjust` lets the owner
    flip any photo either way and end with several keepers.

    **Every cluster gets a suggestion, ambiguous ones included**
    (`every-cluster-opens-with-a-suggestion`), which supersedes the no-pre-mark half of
    `ambiguous-clusters-go-to-manual-pick`. The owner's instruction was explicit: the
    algorithm recommends, the UI decides. Withholding a suggestion from the 40.1% of
    clusters that are near-ties made the tool least helpful exactly where the owner has
    the most work to do, and expressed low confidence as an *absence* — something the
    owner has to already understand in order to read. The near-tie is still surfaced,
    but as prose in `annotations`, where it can say what it means.

    The only cluster that proposes nothing is one with no winner at all, which is an
    absent measurement rather than a close call."""
    if cluster.winner_uuid is None:
        return {record.uuid: "unset" for record in cluster.records}
    return {
        record.uuid: "keep" if record.uuid == cluster.winner_uuid else "cull"
        for record in cluster.records
    }


def _public_member(
    member: dict[str, Any], display: DisplayInfo, evidence: list[str]
) -> dict[str, Any]:
    """One member as the browser sees it: paths out, display size and evidence in."""
    public = {key: value for key, value in member.items() if not key.endswith("_path")}
    public["display"] = {"available": display.available, "long_side": display.long_side}
    # Computed here rather than in the browser because the phrasing rules are about what
    # the *scorer* did — which criteria it ranked on, which weights are zero — and those
    # live in `ClusterConfig`, which the session document deliberately does not ship in
    # a form JavaScript could re-derive them from.
    public["evidence"] = evidence
    return public


def build_session(
    clusters: Sequence[Cluster],
    *,
    config: ClusterConfig | None = None,
    source: str | Path | None = None,
    resolver: Resolver = resolve_display,
) -> dict[str, Any]:
    """The whole review session as one plain dict, ready for `json.dumps`.

    `limit=len(clusters)` because `sample_to_dict` exists to *subsample* for calibration
    and the review covers everything. That is not a hack: `photocull cluster --json-out`
    already calls it the same way, so the full-population path is the one in production
    use today.

    Cost, measured over the real library: resolving 4,498 display rasters costs 6.1s
    (1.35 ms each), paid once when a session opens and then read for days."""
    clusters = list(clusters)
    cfg = config or ClusterConfig()
    doc = sample_to_dict(clusters, config=cfg, source=source, limit=len(clusters))

    entries = []
    for cluster, entry in zip(clusters, doc["clusters"], strict=True):
        displays = {record.uuid: resolver(record) for record in cluster.records}
        entries.append(
            entry
            | {
                "cluster_key": cluster_key(cluster),
                "annotations": annotations(cluster, displays),
                "proposed": proposed_marks(cluster),
                # One sentence for the whole cluster, not one per take: see
                # `explain.ranking_basis`.
                "ranking_basis": ranking_basis(cluster, config=cfg),
                "members": [
                    _public_member(
                        member,
                        displays[member["uuid"]],
                        explain(cluster, member["uuid"], config=cfg),
                    )
                    for member in entry["members"]
                ],
            }
        )

    return doc | {
        "session_version": SESSION_VERSION,
        "config_digest": config_digest(cfg, source),
        "clusters": entries,
    }
