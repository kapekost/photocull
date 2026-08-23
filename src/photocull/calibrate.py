"""The calibration sample the owner reviews before trusting a threshold.

That review asks the owner for two judgements — is the cut height right, and is the
ambiguity margin right — by looking at real clusters. This module decides *which*
clusters they see, which is the whole substance of the task: a fixed-size sample out
of a much larger cluster population means most clusters go unreviewed, and a naive
sample can look equally plausible while covering almost none of the actual risk.

**The sampling rule is stratified on cluster size x cluster diameter, and it
deliberately over-samples the tail.** Both axes were checked against real library
data before being chosen:

- They are nearly independent — knowing a cluster is a pair says almost nothing about
  how close its grouping came to the cut. One axis could not stand in for the other.
- The risk is concentrated where the clusters are not. Large clusters are a small
  fraction of all clusters but a disproportionate share of culled photos: a 20-photo
  cluster stages 19 photos toward deletion on a single grouping decision, and it has
  190 internal pairs for a false grouping to hide in.

Checked against the two obvious alternatives, on real cluster populations — the column
that matters is how much of the actual cull risk the owner gets to see: taking the
first N clusters is the tempting one-liner and the worst of the three, because it
hands the owner only the library's earliest photos and almost none of the large
clusters that carry the risk. An even spread across all clusters does better but still
under-samples the large-cluster tail. The stratified rule used here covers every
large-cluster stratum and a much larger share of the actual cull risk than either
alternative, at the same sample size.

**No RNG anywhere.** Strata are filled by even allocation with the shortfall from
small strata redistributed, and members are picked at evenly-spaced indices in capture
order (both endpoints included, so the sample spans the library's full date range
rather than clustering at one end). A seeded RNG would be reproducible too, but this
is reproducible *and* explainable — the owner can re-derive why any given cluster is
in the file, which matters because the sample is evidence for a threshold decision."""

from __future__ import annotations

import json
import shutil
import statistics
from collections.abc import Iterable, Sequence
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .config import ClusterConfig
from .config_io import describe_config
from .models import Cluster

#: The calibration sample's default size: enough clusters to judge the threshold by,
#: small enough to actually look through.
DEFAULT_SAMPLE_SIZE = 200

#: Upper bound (inclusive) of each size band, and its label. The bands are cut where
#: the real distribution tends to fall, not on round numbers: most clusters are pairs,
#: and clusters above 4 photos are a small fraction of the population.
_SIZE_BANDS: tuple[tuple[int, str], ...] = ((2, "2"), (4, "3-4"), (9, "5-9"))
_LARGEST_SIZE_BAND = "10+"

#: Diameter is banded by quartile of the population rather than at fixed heights,
#: because the cut height is exactly what the owner is about to tune — fixed bands
#: would need re-picking every time the owner turns the dial.
_DIAMETER_BANDS = 4


def size_band(size: int) -> str:
    """Which size band a cluster of `size` photos belongs to."""
    for upper, label in _SIZE_BANDS:
        if size <= upper:
            return label
    return _LARGEST_SIZE_BAND


def _quantile(values: Sequence[float], pct: float) -> float:
    """`sorted(v)[int(p/100*n)]`. Four defensible answers to "p75" exist and they can
    disagree materially on the same data, so the project uses exactly one convention
    everywhere rather than mixing them."""
    ordered = sorted(values)
    return ordered[int(pct / 100 * len(ordered))]


def _diameter_cuts(clusters: Sequence[Cluster]) -> list[float]:
    measured = [c.diameter for c in clusters if c.diameter is not None]
    if not measured:
        return []
    return [_quantile(measured, 100 * i / _DIAMETER_BANDS) for i in range(1, _DIAMETER_BANDS)]


def _diameter_band(diameter: float | None, cuts: Sequence[float]) -> str:
    # A cluster with no measurable pair sorts to the bottom band rather than being
    # dropped: it is still a grouping decision the owner may want to see.
    if diameter is None:
        return "d1"
    return f"d{sum(1 for cut in cuts if diameter >= cut) + 1}"


def _allocate(pools: dict[str, int], limit: int) -> dict[str, int]:
    """Spread `limit` picks over strata as evenly as their sizes allow.

    Even allocation is the point, not proportional allocation: proportional would give
    the 14 clusters of 10+ photos two slots, and those are the clusters where a false
    grouping costs the most. Strata smaller than their even share contribute
    everything they have and the shortfall is redistributed over the rest, so the
    sample stays at exactly `limit` without any stratum being over-drawn."""
    alloc = {key: 0 for key in pools}
    remaining = min(limit, sum(pools.values()))
    open_keys = {key for key, size in pools.items() if size > 0}

    while remaining > 0 and open_keys:
        share, extra = divmod(remaining, len(open_keys))
        if share == 0:
            # Fewer picks left than strata: hand them out in a fixed key order so the
            # result never depends on dict iteration order.
            for key in sorted(open_keys)[:extra]:
                alloc[key] += 1
                remaining -= 1
            break
        for key in sorted(open_keys):
            take = min(share, pools[key] - alloc[key])
            alloc[key] += take
            remaining -= take
        open_keys = {key for key in open_keys if alloc[key] < pools[key]}
    return alloc


def _evenly_spaced(count: int, take: int) -> list[int]:
    """`take` indices spread over `count` items, including both endpoints.

    Endpoint-inclusive is not cosmetic: with plain `i * n // k` the real sample stops
    at 2026-04-19 against a population running to 2026-08-08, because that form never
    selects the last element of any stratum."""
    if take >= count:
        return list(range(count))
    if take <= 0:
        return []
    if take == 1:
        return [0]
    return [round(i * (count - 1) / (take - 1)) for i in range(take)]


def _stratify(clusters: Sequence[Cluster]) -> dict[str, list[int]]:
    """Population indices grouped by stratum, each list in capture order."""
    cuts = _diameter_cuts(clusters)
    strata: dict[str, list[int]] = {}
    for index, cluster in enumerate(clusters):
        key = f"{size_band(len(cluster.records))}/{_diameter_band(cluster.diameter, cuts)}"
        strata.setdefault(key, []).append(index)
    return strata


def sample_allocation(
    clusters: Sequence[Cluster], *, limit: int = DEFAULT_SAMPLE_SIZE
) -> dict[str, int]:
    """How many clusters each stratum contributes. Exported so the sample file can
    carry it and the owner can audit the rule instead of reverse-engineering it."""
    strata = _stratify(clusters)
    return _allocate({key: len(idx) for key, idx in strata.items()}, limit)


def stratified_sample(
    clusters: Sequence[Cluster], *, limit: int = DEFAULT_SAMPLE_SIZE
) -> list[Cluster]:
    """Pick `limit` clusters spanning size, diameter and capture time.

    Returned in population order — which is capture order, inherited from bucketing and
    never re-imposed — so the owner walks the sample chronologically like every other
    surface in this project, rather than in stratum order, which would group all the
    scary clusters together and skew the impression the sample gives."""
    clusters = list(clusters)
    strata = _stratify(clusters)
    alloc = _allocate({key: len(idx) for key, idx in strata.items()}, limit)

    chosen: list[int] = []
    for key in sorted(strata):
        members = strata[key]
        chosen.extend(members[i] for i in _evenly_spaced(len(members), alloc[key]))
    return [clusters[i] for i in sorted(chosen)]


# --- serialisation -------------------------------------------------------------


def _members(cluster: Cluster) -> list[dict[str, Any]]:
    """One entry per photo, joined to its score **by uuid**.

    Never by index. `rank_cluster` sorts `scores` by descending total while `records`
    keep bucket order, so the two lists routinely disagree — an index join produces a
    file in which every photo carries a different photo's sub-scores, which reads as
    entirely plausible and is wrong throughout."""
    by_uuid = {s.uuid: s for s in cluster.scores}
    entries = []
    for record in cluster.records:
        score = by_uuid.get(record.uuid)
        is_winner = record.uuid == cluster.winner_uuid
        entries.append(
            {
                "uuid": record.uuid,
                "date": record.date.isoformat() if record.date else None,
                "derivative_path": record.derivative_path,
                "is_winner": is_winner,
                # None for the keeper itself: the distance that matters is always
                # "how far is this photo from the one being kept".
                "distance_to_winner": None
                if is_winner
                else cluster.distances_to_winner.get(record.uuid),
                "total": score.total if score else None,
                "sub_scores": dict(score.sub_scores) if score else {},
            }
        )
    return entries


def _cluster_to_dict(cluster: Cluster) -> dict[str, Any]:
    return {
        "size": len(cluster.records),
        "winner_uuid": cluster.winner_uuid,
        "is_ambiguous": cluster.is_ambiguous,
        "diameter": cluster.diameter,
        "median_distance": cluster.median_distance,
        "sharpness_available": cluster.sharpness_available,
        "dropped_criteria": list(cluster.dropped_criteria),
        "members": _members(cluster),
    }


def sample_to_dict(
    clusters: Sequence[Cluster],
    *,
    config: ClusterConfig | None = None,
    source: str | Path | None = None,
    limit: int = DEFAULT_SAMPLE_SIZE,
) -> dict[str, Any]:
    """The calibration sample as a plain dict, ready for `json.dumps`."""
    clusters = list(clusters)
    sample = stratified_sample(clusters, limit=limit)
    return {
        "config": describe_config(config or ClusterConfig(), source),
        # A file read weeks later must say what it is a sample *of*: "200 clusters" and
        # "200 of 1,496, covering 28% of the culls" are very different claims.
        "population": {
            "clusters": len(clusters),
            "sampled": len(sample),
            "photos_in_clusters": sum(len(c.records) for c in clusters),
            "culled_photos": sum(len(c.records) - 1 for c in clusters),
            "culled_photos_sampled": sum(len(c.records) - 1 for c in sample),
        },
        "strata": sample_allocation(clusters, limit=limit),
        "clusters": [_cluster_to_dict(c) for c in sample],
    }


def write_calibration_sample(
    clusters: Sequence[Cluster],
    path: str | Path,
    *,
    config: ClusterConfig | None = None,
    source: str | Path | None = None,
    limit: int = DEFAULT_SAMPLE_SIZE,
) -> dict[str, Any]:
    """Write the sample as JSON and return it."""
    doc = sample_to_dict(clusters, config=config, source=source, limit=limit)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


# --- contact sheet -------------------------------------------------------------

_STYLE = """
body { font: 14px -apple-system, system-ui, sans-serif; margin: 2rem; background: #111; color: #eee; }
h1 { font-size: 1.2rem; }
section { border-top: 1px solid #333; padding: 1rem 0; }
.takes { display: flex; flex-wrap: wrap; gap: 1rem; }
figure { margin: 0; max-width: 22rem; }
img { width: 100%; height: auto; background: #222; }
figcaption { font-size: 12px; line-height: 1.5; padding-top: .25rem; }
.keep { color: #7fdc7f; font-weight: 600; }
.cull { color: #d88; }
.flags { color: #e0b050; font-size: 12px; }
.meta { color: #999; font-size: 12px; }
.num { color: #888; }
.legend { background: #1a1a1a; padding: 1rem; border-radius: 6px; font-size: 13px;
          line-height: 1.7; max-width: 46rem; }
.legend dt { font-weight: 600; color: #ddd; float: left; clear: left; width: 11rem; }
.legend dd { margin: 0 0 .3rem 11rem; color: #aaa; }
"""

#: Hover text for each sub-score. The owner's first reaction to the sheet was "I don't
#: understand what the values are ... I'm not sure what I'm looking at", which is a
#: defect in the output, not in the reader.
_CRITERIA = {
    "sharpness": "How sharp, relative to the sharpest take in this same group. 1.00 is the sharpest one.",
    "exposure": "How well exposed, on its own. Penalises blown highlights and crushed shadows.",
    "faces": "How open the eyes are, averaged over detected faces.",
    "framing": "Whether the subject is well placed in the frame and not cut off at an edge.",
    "horizon": "How level the horizon is. Absent when no horizon line was detected.",
    "capture_quality": "Apple's own blended face-quality score. Measured but unweighted.",
    "smiling": "Derived smile curvature. Measured but unweighted - not yet calibrated.",
    "facing": "Subject facing the camera. Measured but unweighted - Vision's angles are unusable.",
}

_LEGEND = """
<div class="legend"><dl>
<dt>KEEP / CULL</dt><dd>The scorer's <em>proposal</em>, not a decision. Nothing has been
written to your library and nothing is deleted, ever - this is a read-only report.</dd>
<dt>"nearly identical"</dt><dd>How different a photo is from the one being kept. This is the
only number a cull decision actually rests on. Raw scale on hover: 0 = identical,
~0.15 = frames of one burst, ~1.0 = unrelated photos.</dd>
<dt>diameter</dt><dd>How far apart the two <em>most different</em> photos in the group are.
Every group is guaranteed tighter than the cut height above.</dd>
<dt>score</dt><dd>The weighted best-shot total, 0-1. Only meaningful <em>inside</em> its own
group - never compare it across groups.</dd>
<dt>amber line</dt><dd>Why the scorer thinks this group is weak evidence. Hover any
sub-score for what it measures; hover a photo for its file path.</dd>
</dl>
<p><b>What to look for:</b> photos grouped together that are <em>not</em> the same shot
(the expensive error - it risks a one-of-a-kind photo), and obvious near-duplicates left in
separate groups (the cheap error). Images here are ~480px, which is plenty to judge whether
a grouping is right, but <em>not</em> enough to judge which take is sharper.</p></div>
"""


def _similarity_words(distance: float) -> str:
    """Plain language for a distance, because the raw number means nothing on its own.

    Calibrated against this project's own measurements: frames within a real Apple
    burst run a median 0.1494 apart, while two unrelated photos measured 1.0398."""
    if distance < 0.15:
        return "nearly identical"
    if distance < 0.25:
        return "very similar"
    if distance < 0.33:
        return "similar"
    return "loosest match in this group"


def _figure(member: dict[str, Any]) -> str:
    path = member.get("derivative_path") or ""
    # `src` is a RELATIVE path to a copy beside the HTML. It cannot be a file:// URL
    # into the library: those files live inside Photos Library.photoslibrary, which
    # macOS protects with TCC, and browsers do not have Full Disk Access -- so every
    # image renders as a broken link even though the file is mode 644 and this
    # process reads it fine. Measured, not guessed.
    src = f"img/{quote(Path(path).name)}" if path else ""
    verdict = (
        '<span class="keep">KEEP — the take the scorer proposes</span>'
        if member["is_winner"]
        else '<span class="cull">CULL</span>'
    )
    distance = member.get("distance_to_winner")
    distance_text = (
        ""
        if distance is None
        else f' &middot; <b>{_similarity_words(distance)}</b> to the keeper '
        f'<span class="num" title="0 = identical, ~0.15 = same burst, ~1.0 = unrelated">'
        f"({distance:.3f})</span>"
    )
    total = member.get("total")
    total_text = "" if total is None else f"{total:.3f}"
    subs = " ".join(
        f'<span title="{escape(_CRITERIA.get(name, name))}">{escape(name)} {value:.2f}</span>'
        for name, value in sorted((member.get("sub_scores") or {}).items())
        if value is not None
    )
    return (
        f'<figure><img loading="lazy" src="{escape(src)}" alt="{escape(member["uuid"])}" '
        f'title="{escape(path)}">'
        f"<figcaption>{verdict}{distance_text}<br>"
        f'<span class="meta" title="{escape(path)}">score {total_text} '
        f"&mdash; only comparable inside this group<br>{subs}</span>"
        f"</figcaption></figure>"
    )


def _section(index: int, entry: dict[str, Any]) -> str:
    flags = []
    if entry["is_ambiguous"]:
        flags.append("ambiguous — scorer cannot separate the top two")
    if not entry["sharpness_available"]:
        flags.append("no sharpness signal — compare at 100%")
    if entry["dropped_criteria"]:
        flags.append("not compared on: " + ", ".join(entry["dropped_criteria"]))
    flag_html = (
        f'<div class="flags">{escape(" | ".join(flags))}</div>' if flags else ""
    )
    diameter = entry["diameter"]
    diameter_text = "n/a" if diameter is None else f"{diameter:.4f}"
    return (
        f"<section><h2>#{index} &middot; {entry['size']} takes &middot; "
        f"diameter {diameter_text}</h2>{flag_html}"
        f'<div class="takes">{"".join(_figure(m) for m in entry["members"])}</div>'
        "</section>"
    )


def _copy_images(sample: dict[str, Any], into: Path) -> int:
    """Copy every referenced derivative next to the HTML, and return how many landed.

    This is what makes the sheet render at all. The derivatives live inside
    `Photos Library.photoslibrary`, which macOS protects with TCC; this process holds
    Full Disk Access and browsers do not, so a `file://` link into the bundle is
    denied by the OS and shows as a broken image even though the file is mode 644.
    Copying is a read of the library and a write into the report directory -- the
    library itself is never touched, moved or modified."""
    into.mkdir(parents=True, exist_ok=True)
    copied = 0
    for entry in sample.get("clusters", []):
        for member in entry["members"]:
            source = member.get("derivative_path")
            if not source:
                continue
            destination = into / Path(source).name
            if destination.exists():
                copied += 1
                continue
            try:
                shutil.copy2(source, destination)
                copied += 1
            except OSError:
                # A derivative Photos has since evicted. The figure still renders with
                # its caption and stays reviewable; skipping beats aborting the sheet.
                continue
    return copied


def write_contact_sheet(sample: dict[str, Any], path: str | Path) -> None:
    """Render the sample as a static HTML page of side-by-side takes.

    The task here is "look at a couple hundred clusters and judge whether the groupings
    are right", and a JSON file of raw filesystem paths does not make that possible in
    practice. This is deliberately static — no JS, no interactivity, no write-back. The
    real review UI is a separate, interactive tool; this only has to make the gate that
    unblocks trusting a threshold actually walkable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    copied = _copy_images(sample, path.parent / "img")

    pop = sample.get("population", {})
    cfg = sample.get("config", {})
    header = (
        f"<h1>Calibration sample — {pop.get('sampled', 0)} of "
        f"{pop.get('clusters', 0)} groups, "
        f"{pop.get('culled_photos_sampled', 0)} of {pop.get('culled_photos', 0)} "
        "photos proposed for culling</h1>"
        f'<p class="meta">cut height (similarity_threshold): '
        f"{cfg.get('similarity_threshold')} &middot; "
        f"ambiguity margin: {cfg.get('ambiguity_margin')} &middot; "
        f"{copied} images copied to ./img/</p>"
        f"{_LEGEND}"
    )
    body = "".join(
        _section(i, entry) for i, entry in enumerate(sample.get("clusters", []), start=1)
    )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>photocull calibration sample</title>"
        f"<style>{_STYLE}</style></head><body>{header}{body}</body></html>"
    )
    path.write_text(html, encoding="utf-8")
