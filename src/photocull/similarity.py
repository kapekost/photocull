"""Stage 2 of clustering: the expensive pass, run only inside a bucket.

Takes feature-print vectors as a plain dict so the logic is fully unit-testable
without Vision or a Photos library."""

from __future__ import annotations

from collections.abc import Sequence

from .config import ClusterConfig
from .models import PhotoRecord
from .vision_backend import l2_distance


def _pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def cluster_bucket(
    records: Sequence[PhotoRecord],
    vectors: dict[str, list[float]],
    config: ClusterConfig | None = None,
) -> list[list[PhotoRecord]]:
    """Refine one bucket into near-duplicate clusters by **agglomerative complete
    linkage**, cutting the dendrogram at `ClusterConfig.similarity_threshold`.
    Records without a feature print (no local derivative, or Vision failed) are
    excluded rather than being guessed at. Returns only groups of size >= 2.

    The threshold is a **cut height, not a pair-admission rule**, so every cluster
    that comes out is guaranteed to have diameter under it. Single linkage bounded
    the pairs going in instead, which let a cluster grow arbitrarily wide by
    chaining: on the real library it produced a 27-photo cluster of diameter 0.7997
    holding a photo 0.6760 from the take that would be kept, at a threshold of 0.40.
    Each of those is a photo staged for culling as a duplicate of something it does
    not resemble -- the exact failure `cluster-precision-over-recall` exists to
    prevent. See DECISIONS.md `complete-linkage-replaces-single-linkage-and-keeper-radius`.

    Three determinism rules, all load-bearing and all pinned by tests:

    * **`<`, not `<=`** -- a merge landing exactly on the cut is refused. Matches
      Phase 0's strictly-under-the-gap rule, and keeps this grouping a strict subset
      of what single linkage produced, since that admitted pairs on `<` too.
    * **Ties break on the merging clusters' lowest uuids**, as a sorted pair, so the
      answer never depends on index order, dict order or which side is "left".
    * **Groups come out ordered by lowest member**, with each group's records in
      bucket order (`clusters-ordered-by-earliest-member`).

    Naive O(n^3) is deliberate: the largest real bucket is 86 photos and every
    dendrogram in the library computes in ~0.1s total, against a Vision pass that
    dominates by orders of magnitude. No numpy -- distances are hand-rolled pure
    Python by decision (`vision-distance-is-hand-rolled-apple-crosscheck`)."""
    cfg = config or ClusterConfig()
    usable = [r for r in records if r.uuid in vectors]
    n = len(usable)
    if n < 2:
        return []

    # Complete-linkage distance between two live clusters, keyed by their ids. A
    # cluster's id is the lowest record index it contains, since a merge always
    # folds the higher id into the lower one.
    height: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            height[(i, j)] = l2_distance(vectors[usable[i].uuid], vectors[usable[j].uuid])

    members: dict[int, list[int]] = {i: [i] for i in range(n)}

    while len(members) > 1:
        best: tuple[tuple[float, str, str], int, int] | None = None
        for (a, b), h in height.items():
            if not h < cfg.similarity_threshold:
                continue
            lo, hi = sorted((usable[members[a][0]].uuid, usable[members[b][0]].uuid))
            key = (h, lo, hi)
            if best is None or key < best[0]:
                best = (key, a, b)
        if best is None:
            break

        _, a, b = best
        if b < a:
            a, b = b, a
        for c in members:
            if c in (a, b):
                continue
            # Lance-Williams for complete linkage: d(ab, c) = max(d(a, c), d(b, c)).
            height[_pair(a, c)] = max(height[_pair(a, c)], height.pop(_pair(b, c)))
        height.pop((a, b))
        members[a] = sorted(members[a] + members.pop(b))

    return [
        [usable[i] for i in group]
        for group in sorted(members.values(), key=lambda g: g[0])
        if len(group) >= 2
    ]
