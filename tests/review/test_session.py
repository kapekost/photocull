"""Task 3: the review session document — the contract every later Phase 1b task binds to.

Three things are load-bearing here and each has cost a tick elsewhere in this project:

1. **The document is built by reusing `calibrate.sample_to_dict`, not by re-deriving
   members.** `calibrate._members` joins records to scores *by uuid* because
   `rank_cluster` sorts `scores` while `records` keep bucket order; an index join gives
   every photo another photo's sub-scores and reads as entirely plausible. The reuse is
   pinned by comparing against a direct `sample_to_dict` call, so a future rewrite that
   "simplifies" the enrichment into its own loop goes red.
2. **The document is what the browser receives**, so it carries no filesystem paths.
3. **Annotation policy is a measurement, not a preference** — see the coverage figures
   in `annotations`' own docstring.
"""

from __future__ import annotations

import json

from photocull.calibrate import sample_to_dict
from photocull.config import ClusterConfig
from photocull_review.demo import write_png
from photocull_review.session import (
    LARGE_CLUSTER_SIZE,
    LOW_DETAIL_MAX_LONG_SIDE,
    MIXED_RESOLUTION_RATIO,
    DisplayInfo,
    annotations,
    build_session,
    cluster_key,
    config_digest,
    proposed_marks,
    resolve_display,
)

from ..fixtures import make_cluster


def fixed_displays(sizes):
    """A resolver stub mapping uuid -> long side (None means no display derivative)."""

    def resolver(record):
        long_side = sizes.get(record.uuid, 1024)
        return DisplayInfo(
            uuid=record.uuid,
            local_path=record.display_path if long_side is not None else None,
            long_side=long_side,
        )

    return resolver


# --- cluster_key ----------------------------------------------------------------


def test_cluster_key_is_a_literal_hex_digest():
    """Asserted against a value computed by `shasum`, never by the implementation.

    A test that recomputes the expectation with the same code it is testing passes
    under any hash function, including a broken one."""
    assert cluster_key(make_cluster(["a", "b"])) == "7e18f737311b2dc3"
    assert cluster_key(make_cluster(["p1", "p2", "p3"])) == "5a43d43f9e2837fe"


def test_cluster_key_ignores_member_order():
    """Bucket order is stable today, but the key identifies a *set* of photos across
    re-scans — that is the whole point of it surviving into the decisions log."""
    assert cluster_key(make_cluster(["b", "a"])) == cluster_key(make_cluster(["a", "b"]))


def test_cluster_key_changes_when_membership_changes():
    assert cluster_key(make_cluster(["a", "b", "c"])) != cluster_key(make_cluster(["a", "b"]))
    assert cluster_key(make_cluster(["a", "z"])) != cluster_key(make_cluster(["a", "b"]))


# --- config_digest --------------------------------------------------------------


def test_config_digest_ignores_where_the_config_came_from():
    """`describe_config` stamps `source`, which is a path on this machine. Two people
    running identical settings from different files must not read as a config change —
    Task 5 refuses to resume when this digest moves."""
    cfg = ClusterConfig()
    assert config_digest(cfg, source="photocull.toml") == config_digest(cfg, source=None)
    assert config_digest(cfg, source="/somewhere/else.toml") == config_digest(cfg)


def test_config_digest_changes_with_the_similarity_threshold():
    a = config_digest(ClusterConfig(similarity_threshold=0.40))
    b = config_digest(ClusterConfig(similarity_threshold=0.48))
    assert a != b


def test_config_digest_changes_when_a_weight_changes():
    """The weights live one level down in a nested dict. A digest over the top-level
    fields only would miss every re-tuning `review-decisions-train-the-scorer` exists to
    enable, which is the exact case Task 5 must refuse to resume across."""
    base = ClusterConfig()
    tweaked = dict(base.weights)
    tweaked["sharpness"] = round(tweaked["sharpness"] + 0.01, 4)
    tweaked["exposure"] = round(tweaked["exposure"] - 0.01, 4)
    assert config_digest(base) != config_digest(ClusterConfig(weights=tweaked))


# --- proposed marks -------------------------------------------------------------


def test_the_winner_is_proposed_keep_and_the_rest_cull():
    cluster = make_cluster(["a", "b", "c"])
    marks = proposed_marks(cluster)
    assert marks[cluster.winner_uuid] == "keep"
    assert [marks[u] for u in ("a", "b", "c") if u != cluster.winner_uuid] == ["cull", "cull"]


def test_an_ambiguous_cluster_still_proposes_its_winner():
    """`every-cluster-opens-with-a-suggestion` supersedes the no-pre-mark half of
    `ambiguous-clusters-go-to-manual-pick`. The owner asked for a suggestion in every
    group; a near-tie now proposes its winner like any other cluster and is flagged
    ambiguous in words instead of by an absence the owner has to interpret."""
    cluster = make_cluster(["a", "b", "c"], ambiguous=True)
    marks = proposed_marks(cluster)
    assert marks[cluster.winner_uuid] == "keep"
    assert sorted(marks.values()) == ["cull", "cull", "keep"]
    assert set(marks) == {"a", "b", "c"}


def test_an_ambiguous_cluster_says_so_in_words():
    """The blank pre-selection used to be how ambiguity was visible, so removing it
    without adding the badge would make a near-tie silently indistinguishable from a
    confident call — the exact silent-plausible-wrongness shape this project keeps
    hitting."""
    cluster = make_cluster(["a", "b"], ambiguous=True)
    displays = {u: DisplayInfo(u, f"/tmp/{u}.jpg", 1024) for u in ("a", "b")}
    codes = [note["code"] for note in annotations(cluster, displays)]
    assert "ambiguous" in codes
    confident = make_cluster(["c", "d"])
    confident_displays = {u: DisplayInfo(u, f"/tmp/{u}.jpg", 1024) for u in ("c", "d")}
    assert "ambiguous" not in [n["code"] for n in annotations(confident, confident_displays)]


def test_a_cluster_with_no_winner_proposes_nothing():
    cluster = make_cluster(["a", "b"])
    cluster.winner_uuid = None
    assert set(proposed_marks(cluster).values()) == {"unset"}


# --- display resolution ---------------------------------------------------------


def test_resolve_display_reads_the_real_pixel_size(tmp_path):
    path = tmp_path / "d.png"
    write_png(path, width=1024, height=768, rgb=(200, 60, 60), stripe=8)
    cluster = make_cluster(["a"], display_paths={"a": str(path)})
    info = resolve_display(cluster.records[0])
    assert info.long_side == 1024
    assert info.local_path == str(path)


def test_resolve_display_reports_a_photo_with_no_local_display_raster():
    cluster = make_cluster(["a"], display_paths={"a": None})
    info = resolve_display(cluster.records[0])
    assert info.long_side is None
    assert info.local_path is None


def test_resolve_display_reports_a_path_photos_has_since_evicted(tmp_path):
    """Derivatives are a cache Photos regenerates and evicts. A stamped path that no
    longer resolves must degrade to "cannot show this", never raise mid-session."""
    cluster = make_cluster(["a"], display_paths={"a": str(tmp_path / "gone.png")})
    assert resolve_display(cluster.records[0]).long_side is None


# --- annotations ----------------------------------------------------------------


def test_a_cluster_the_scorer_is_confident_about_gets_no_annotation():
    cluster = make_cluster(["a", "b"])
    assert annotations(cluster, {u: DisplayInfo(u, "/p", 1024) for u in ("a", "b")}) == []


def test_sharpness_dropped_for_the_whole_cluster_is_annotated():
    cluster = make_cluster(["a", "b"], sharpness_available=False)
    codes = [a["code"] for a in annotations(cluster, {u: DisplayInfo(u, "/p", 1024) for u in "ab"})]
    assert codes == ["no-sharpness"]


def test_a_cluster_with_no_detail_anywhere_locally_is_annotated():
    """24.1% of real clusters have no member above 640px, so no amount of UI can show
    more than the calibration contact sheet already did — only Task 12's explicit
    full-resolution fetch can settle those, and the owner has to be told which they are."""
    small = LOW_DETAIL_MAX_LONG_SIDE
    cluster = make_cluster(["a", "b"])
    codes = [a["code"] for a in annotations(cluster, {u: DisplayInfo(u, "/p", small) for u in "ab"})]
    assert codes == ["low-detail"]
    bigger = {u: DisplayInfo(u, "/p", small + 1) for u in "ab"}
    assert [a["code"] for a in annotations(cluster, bigger)] == []


def test_takes_shown_at_different_scales_are_annotated():
    cluster = make_cluster(["a", "b"])
    displays = {
        "a": DisplayInfo("a", "/p", 1024),
        "b": DisplayInfo("b", "/p", int(1024 / MIXED_RESOLUTION_RATIO) - 1),
    }
    assert [a["code"] for a in annotations(cluster, displays)] == ["mixed-resolution"]


def test_a_large_cluster_is_annotated():
    uuids = [f"u{i:02d}" for i in range(LARGE_CLUSTER_SIZE)]
    cluster = make_cluster(uuids)
    displays = {u: DisplayInfo(u, "/p", 1024) for u in uuids}
    assert [a["code"] for a in annotations(cluster, displays)] == ["large-cluster"]
    smaller = make_cluster(uuids[:-1])
    assert annotations(smaller, displays) == []


def test_a_photo_with_no_display_derivative_is_annotated_not_dropped():
    cluster = make_cluster(["a", "b"], display_paths={"a": "/display/a.jpeg", "b": None})
    displays = {"a": DisplayInfo("a", "/p", 1024), "b": DisplayInfo("b", None, None)}
    assert [a["code"] for a in annotations(cluster, displays)] == ["no-display"]

    doc = build_session([cluster], resolver=fixed_displays({"b": None}))
    assert [m["uuid"] for m in doc["clusters"][0]["members"]] == ["a", "b"]


def test_every_annotation_carries_prose_not_just_a_code():
    """`low-confidence-clusters-are-annotated-in-review`: "no sharpness signal — compare
    at 100%", never a bare warning icon."""
    cluster = make_cluster(["a", "b"], sharpness_available=False)
    for note in annotations(cluster, {u: DisplayInfo(u, "/p", 400) for u in "ab"}):
        assert len(note["text"].split()) >= 5


def test_dropped_criteria_is_carried_as_data_but_never_annotated():
    """Measured on the real library at the live 0.48 cut: `dropped_criteria` is
    non-empty for **90.3%** of clusters (horizon alone 80.9%). A badge on nine clusters
    in ten is not a warning, it is the background — so it is demoted to the evidence
    panel's raw material rather than deleted. Same lesson as
    `unvalidated-signals-ship-at-zero-weight`."""
    cluster = make_cluster(["a", "b"], dropped=["faces", "framing", "horizon"])
    assert annotations(cluster, {u: DisplayInfo(u, "/p", 1024) for u in "ab"}) == []

    entry = build_session([cluster])["clusters"][0]
    assert entry["dropped_criteria"] == ["faces", "framing", "horizon"]


def test_ambiguity_is_carried_as_state_and_as_prose():
    """40.1% of clusters are ambiguous at the live cut. It used to be visible as an
    absence — nothing pre-marked — but `every-cluster-opens-with-a-suggestion` fills
    that in, so ambiguity has to say itself in words or it says nothing."""
    cluster = make_cluster(["a", "b"], ambiguous=True)
    codes = [n["code"] for n in annotations(cluster, {u: DisplayInfo(u, "/p", 1024) for u in "ab"})]
    assert codes == ["ambiguous"]

    entry = build_session([cluster])["clusters"][0]
    assert entry["is_ambiguous"] is True
    assert entry["proposed"][cluster.winner_uuid] == "keep"


# --- build_session --------------------------------------------------------------


def test_sub_scores_are_byte_identical_to_a_direct_sample_to_dict_call():
    """The reuse contract, pinned. `build_session` enriches `sample_to_dict`'s entries;
    it must never re-derive members, because re-deriving is exactly how the
    records/scores index-misalignment trap gets reintroduced."""
    clusters = [make_cluster(["a", "b", "c"], day=0), make_cluster(["d", "e"], day=1)]
    direct = sample_to_dict(clusters, limit=len(clusters))
    doc = build_session(clusters)

    for mine, theirs in zip(doc["clusters"], direct["clusters"]):
        for m, t in zip(mine["members"], theirs["members"]):
            assert m["uuid"] == t["uuid"]
            assert m["sub_scores"] == t["sub_scores"]
            assert m["total"] == t["total"]
            assert m["distance_to_winner"] == t["distance_to_winner"]


def test_the_session_covers_every_cluster_not_a_sample():
    """`sample_to_dict` exists to *subsample* for calibration. The review session is the
    whole library, so the limit is pinned at the population size — asserting the count
    catches a future default creeping back in."""
    clusters = [make_cluster([f"c{i}a", f"c{i}b"], day=i) for i in range(220)]
    doc = build_session(clusters)
    assert len(doc["clusters"]) == 220
    assert doc["population"]["clusters"] == 220


def test_clusters_stay_in_capture_order():
    clusters = [make_cluster([f"c{i}a", f"c{i}b"], day=i) for i in range(5)]
    doc = build_session(clusters)
    assert [c["members"][0]["uuid"] for c in doc["clusters"]] == [f"c{i}a" for i in range(5)]


def test_every_cluster_carries_its_key_and_the_keys_are_unique():
    clusters = [make_cluster([f"c{i}a", f"c{i}b"], day=i) for i in range(5)]
    doc = build_session(clusters)
    keys = [c["cluster_key"] for c in doc["clusters"]]
    assert keys == [cluster_key(c) for c in clusters]
    assert len(set(keys)) == 5


def test_the_document_carries_no_filesystem_paths():
    """This document is served to a browser. `sample_to_dict` embeds `derivative_path`
    for the calibration contact sheet, which reads files off disk directly; the review
    UI fetches images by uuid (Task 7), so the paths are stripped here rather than at
    serialisation time in a later task where forgetting them would be silent."""
    cluster = make_cluster(["a", "b"], display_paths={"a": "/Users/someone/x.jpeg", "b": None})
    blob = json.dumps(build_session([cluster], resolver=fixed_displays({"b": None})))
    assert "/Users/" not in blob
    assert "derivative_path" not in blob
    assert "display_path" not in blob
    assert ".jpeg" not in blob


def test_each_member_reports_the_display_size_the_ui_is_bounded_by():
    """`zoom-is-bounded-by-the-derivative`: the UI may not magnify past one source pixel
    per device pixel, so it needs the real long side, not the original's dimensions."""
    cluster = make_cluster(["a", "b"])
    doc = build_session([cluster], resolver=fixed_displays({"a": 1024, "b": 480}))
    members = {m["uuid"]: m for m in doc["clusters"][0]["members"]}
    assert members["a"]["display"] == {"available": True, "long_side": 1024}
    assert members["b"]["display"] == {"available": True, "long_side": 480}


def test_every_member_carries_its_own_evidence_joined_by_uuid():
    """The same trap as `sub_scores`, one layer up: `make_cluster` emits records in
    bucket order and scores by descending total, so a member enriched by position gets
    another photograph's explanation — and "The scorer's pick" under the take that lost
    is a sentence a reader has no way to doubt."""
    cluster = make_cluster(
        ["a", "b", "c"],
        totals={"a": 0.4, "b": 0.9, "c": 0.6},
        sub_scores={
            "a": {"sharpness": 0.40, "exposure": 0.5},
            "b": {"sharpness": 1.00, "exposure": 0.5},
            "c": {"sharpness": 0.60, "exposure": 0.5},
        },
    )
    members = {m["uuid"]: m for m in build_session([cluster])["clusters"][0]["members"]}
    assert any("scorer's pick" in s for s in members["b"]["evidence"])
    for loser in ("a", "c"):
        assert any("Behind the proposed keeper" in s for s in members[loser]["evidence"])
        assert not any("scorer's pick" in s for s in members[loser]["evidence"])
    # And the evidence describes the member's own numbers, not a neighbour's.
    assert any("Sharpest of the 3 takes" in s for s in members["b"]["evidence"])
    assert any("0.40× its detail" in s for s in members["a"]["evidence"])


def test_the_ranking_basis_is_carried_once_for_the_cluster():
    """Not per member. It is the same sentence for every take, and rendered per pane it
    would say one thing twice on the 90.3% of real clusters that drop a criterion."""
    entry = build_session([make_cluster(["a", "b"])])["clusters"][0]
    assert "could not be measured" in entry["ranking_basis"]
    for member in entry["members"]:
        assert not any("could not be measured" in s for s in member["evidence"])


def test_the_evidence_honours_the_config_the_session_was_built_under():
    """Weights are the owner's to change in `photocull.toml`, and they decide both which
    criterion is decisive and which ones may be cited at all."""
    cluster = make_cluster(
        ["a", "b"],
        sub_scores={
            "a": {"sharpness": 1.00, "exposure": 0.90},
            "b": {"sharpness": 0.50, "exposure": 0.40},
        },
    )
    exposure_only = ClusterConfig(
        weights={"sharpness": 0.0, "exposure": 1.0, "faces": 0.0,
                 "framing": 0.0, "horizon": 0.0, "facing": 0.0,
                 "capture_quality": 0.0, "smiling": 0.0}
    )
    members = {
        m["uuid"]: m
        for m in build_session([cluster], config=exposure_only)["clusters"][0]["members"]
    }
    assert any("exposed" in s for s in members["a"]["evidence"])
    assert not any("harp" in s for s in members["a"]["evidence"])


def test_the_session_stamps_the_config_it_was_built_under():
    cfg = ClusterConfig(similarity_threshold=0.48)
    doc = build_session([make_cluster(["a", "b"])], config=cfg, source="photocull.toml")
    assert doc["config_digest"] == config_digest(cfg)
    assert doc["config"]["similarity_threshold"] == 0.48
    assert doc["config"]["source"] == "photocull.toml"


def test_the_session_is_json_serialisable():
    doc = build_session([make_cluster(["a", "b"], day=i) for i in range(3)])
    assert json.loads(json.dumps(doc))["clusters"][0]["size"] == 2
