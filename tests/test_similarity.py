from datetime import datetime, timedelta

from photocull.config import ClusterConfig
from photocull.similarity import cluster_bucket
from photocull.vision_backend import l2_distance
from tests.fixtures import make_photo_record

BASE = datetime(2025, 6, 1, 12, 0, 0)


def rec(i, uuid=None):
    return make_photo_record(uuid=uuid or f"u{i}", date=BASE + timedelta(seconds=i))


def diameter(cluster, vectors):
    return max(
        l2_distance(vectors[a.uuid], vectors[b.uuid])
        for i, a in enumerate(cluster)
        for b in cluster[i + 1 :]
    )


def test_cluster_bucket_groups_similar_vectors():
    records = [rec(0), rec(1)]
    vectors = {"u0": [0.0, 0.0], "u1": [0.1, 0.0]}  # distance 0.1 < 0.40
    out = cluster_bucket(records, vectors)
    assert [[r.uuid for r in c] for c in out] == [["u0", "u1"]]


def test_cluster_bucket_separates_dissimilar_vectors():
    records = [rec(0), rec(1)]
    vectors = {"u0": [0.0, 0.0], "u1": [1.0, 0.0]}  # distance 1.0 > 0.40
    assert cluster_bucket(records, vectors) == []


def test_cluster_bucket_drops_singletons():
    """A cluster of one is not a cluster -- nothing to choose between."""
    records = [rec(0), rec(1), rec(2)]
    vectors = {"u0": [0.0, 0.0], "u1": [0.05, 0.0], "u2": [5.0, 0.0]}
    out = cluster_bucket(records, vectors)
    assert [[r.uuid for r in c] for c in out] == [["u0", "u1"]]


def test_threshold_is_configurable():
    records = [rec(0), rec(1)]
    vectors = {"u0": [0.0, 0.0], "u1": [0.5, 0.0]}
    assert cluster_bucket(records, vectors) == []
    loose = ClusterConfig(similarity_threshold=0.9)
    assert len(cluster_bucket(records, vectors, config=loose)) == 1


def test_boundary_distance_does_not_cluster():
    """Threshold is strict less-than, matching Phase 0's gap semantics."""
    records = [rec(0), rec(1)]
    vectors = {"u0": [0.0, 0.0], "u1": [0.40, 0.0]}
    assert cluster_bucket(records, vectors) == []


def test_records_missing_a_vector_are_skipped_not_crashed():
    """A photo with no local derivative has no feature print. It must not join a
    cluster and must not raise -- it just isn't a cull candidate."""
    records = [rec(0), rec(1), rec(2)]
    vectors = {"u0": [0.0, 0.0], "u1": [0.05, 0.0]}
    out = cluster_bucket(records, vectors)
    assert [[r.uuid for r in c] for c in out] == [["u0", "u1"]]


def test_realistic_burst_distances_cluster_together():
    """Guards the default threshold against the real measured data: a real burst
    group scored 0.21-0.42 apart, unrelated photos 0.83+."""
    records = [rec(0), rec(1)]
    vectors = {"u0": [0.0, 0.0], "u1": [0.35, 0.0]}
    assert len(cluster_bucket(records, vectors)) == 1


def test_clusters_come_out_in_capture_order():
    """Bucketing hands records over sorted by (date, uuid) and Task 8b exports a
    calibration sample from these clusters for the owner to review, so clusters
    must stay chronological (`clusters-ordered-by-earliest-member`).

    The risk under complete linkage is emitting groups in MERGE order: {u1,u4} is
    the tightest pair in this fixture (0.05) so it merges first, while {u0,u2,u3}
    -- which holds the earliest photo and must come first -- only completes on its
    third merge. Ordering by lowest member is what separates the two."""
    records = [rec(i) for i in range(5)]
    vectors = {
        "u0": [0.0],
        "u1": [10.0],
        "u2": [0.30],  # 0.30 from u0, 0.15 from u3 -- every pair under the cut
        "u3": [0.15],
        "u4": [10.05],  # 0.05 from u1: merges before anything in the other cluster
    }
    out = cluster_bucket(records, vectors)
    assert [[r.uuid for r in c] for c in out] == [["u0", "u2", "u3"], ["u1", "u4"]]


def test_records_within_a_cluster_stay_in_bucket_order():
    """Within a cluster the records keep the order bucketing gave them, so the
    earliest take is first regardless of which pair triggered the merge. Here u0
    and u2 merge first (0.05) and u1 joins afterwards, so a cluster assembled in
    merge order would read u0, u2, u1."""
    records = [rec(0), rec(1), rec(2)]
    vectors = {"u0": [0.0], "u1": [0.35], "u2": [0.05]}
    out = cluster_bucket(records, vectors)
    assert [[r.uuid for r in c] for c in out] == [["u0", "u1", "u2"]]


# --- complete linkage: the cut height bounds the RESULT --------------------------


def test_a_chain_does_not_merge_when_the_result_would_be_too_wide():
    """THE change, and the whole reason this task exists. u0-u1 is 0.35 and u1-u2 is
    0.35, both admissible pairs under a 0.40 cut, but u0-u2 is 0.70. Single linkage
    groups all three on the strength of a chain -- and would then stage two of them
    for culling as duplicates of a photo they are 0.70 from. Complete linkage refuses
    the second merge, because the cut bounds the width of what comes OUT, not the
    width of the pairs going in."""
    records = [rec(0), rec(1), rec(2)]
    vectors = {"u0": [0.0], "u1": [0.35], "u2": [0.70]}
    out = cluster_bucket(records, vectors)
    assert [[r.uuid for r in c] for c in out] == [["u0", "u1"]]


def test_every_emitted_cluster_has_diameter_within_the_cut():
    """The invariant the whole task exists to establish, asserted directly rather
    than inferred from a grouping. Five photos evenly spaced 0.15 apart form one
    single-linkage chain of diameter 0.60; complete linkage must split them so no
    emitted cluster is wider than the cut."""
    records = [rec(i) for i in range(5)]
    vectors = {f"u{i}": [0.15 * i] for i in range(5)}
    out = cluster_bucket(records, vectors)
    assert len(out) > 1, "the fixture must actually be a chain, or this proves nothing"
    cfg = ClusterConfig()
    for cluster in out:
        assert diameter(cluster, vectors) <= cfg.similarity_threshold


def test_a_merge_landing_exactly_on_the_cut_is_refused():
    """Strict less-than, at merge level as well as pair level. u0-u1 is 0.10 and
    merges; the resulting cluster then sits EXACTLY 0.40 from u2, and is refused.

    Pinned rather than left to the reader, and pinned as `<` rather than `<=` for
    three reasons: it matches `test_boundary_distance_does_not_cluster` and Phase
    0's identical strictly-under-the-gap rule (`near-dup-cluster-definition`);
    refusing is the tighter side of `cluster-precision-over-recall`; and decisively,
    single linkage admits pairs on `<`, so a `<=` merge could produce a cluster that
    is NOT a subset of any single-linkage cluster -- which would put a hole in the
    strict-refinement argument that justifies making this change without the owner."""
    records = [rec(0), rec(1), rec(2)]
    vectors = {"u0": [0.0], "u1": [0.10], "u2": [0.40]}
    out = cluster_bucket(records, vectors)
    assert [[r.uuid for r in c] for c in out] == [["u0", "u1"]]


def test_identical_vectors_group():
    """Distance 0.0 is the most-similar case there is, not an edge to fall through."""
    records = [rec(0), rec(1)]
    vectors = {"u0": [0.2, 0.3], "u1": [0.2, 0.3]}
    out = cluster_bucket(records, vectors)
    assert [[r.uuid for r in c] for c in out] == [["u0", "u1"]]


def test_equal_height_merges_break_ties_on_uuid_not_input_order():
    """Three candidate merges sit at exactly 0.25 and the choice between them
    changes the answer, so the tiebreak is load-bearing rather than cosmetic.

    Records arrive in bucket order u9, u3, u1, u5 -- deliberately not uuid order.
    Breaking on input order takes the first pair (u9,u3), leaving [[u9,u3],[u1,u5]].
    Breaking on the CANONICAL (sorted) uuid pair picks (u1,u3), and that cluster
    then sits 0.50 from both neighbours, so it is the only one emitted.

    The grid is 0.25 rather than a rounder-looking 0.30 because the tie has to be
    exact to exist at all: on a 0.30 grid `0.90 - 0.60` is 0.30000000000000004, so
    the third candidate silently drops out of the tie and the surviving two-way tie
    is one both rules agree on -- a test that passes without testing anything.
    Verified by mutation: the 0.30 version survived dropping the tiebreak."""
    records = [rec(0, "u9"), rec(1, "u3"), rec(2, "u1"), rec(3, "u5")]
    vectors = {"u9": [0.0], "u3": [0.25], "u1": [0.50], "u5": [0.75]}
    assert len({l2_distance(vectors["u9"], vectors["u3"]),
                l2_distance(vectors["u3"], vectors["u1"]),
                l2_distance(vectors["u1"], vectors["u5"])}) == 1, "must be an exact tie"
    out = cluster_bucket(records, vectors)
    assert [[r.uuid for r in c] for c in out] == [["u3", "u1"]]


def test_a_tie_uses_the_cluster_s_lowest_member_not_its_latest():
    """A cluster's uuid for tiebreak purposes is its LOWEST member's, which only
    becomes observable once a cluster has more than one member -- while everything
    is still a singleton the two rules cannot disagree.

    u5 and u1 merge first at 0.125. That cluster is then exactly 0.375 from u3, tying
    with the (u3,u4) pair. Keyed on the cluster's lowest member (u5) the tie reads
    ("u3","u5") and loses to ("u3","u4"), so u3 and u4 pair off. Keyed on its latest
    member (u1) it reads ("u1","u3"), wins, and swallows u3 into a three-photo
    cluster instead. Distances are on a 1/8 grid so every one of them is exact."""
    records = [rec(0, "u5"), rec(1, "u1"), rec(2, "u3"), rec(3, "u4")]
    vectors = {"u5": [0.0], "u1": [0.125], "u3": [0.375], "u4": [0.75]}
    out = cluster_bucket(records, vectors)
    assert [[r.uuid for r in c] for c in out] == [["u5", "u1"], ["u3", "u4"]]


def test_grouping_does_not_depend_on_vector_dict_order():
    """Determinism across processes: `PhotosDB.photos()` ordering is not stable run
    to run (`within-burst-distance-band-corrected`), so the same photos must group
    identically however the vectors dict was assembled."""
    records = [rec(i) for i in range(5)]
    forward = {f"u{i}": [0.15 * i] for i in range(5)}
    reverse = {f"u{i}": [0.15 * i] for i in reversed(range(5))}
    assert [[r.uuid for r in c] for c in cluster_bucket(records, forward)] == [
        [r.uuid for r in c] for c in cluster_bucket(records, reverse)
    ]
