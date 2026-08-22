"""Task 5: resume reconciliation — what a relaunch may safely carry over.

This is the module that decides whether a week of the owner's judgement still applies to
the library in front of it. Getting that wrong in either direction is expensive: applying
a stale decision writes back a verdict about photos the owner never saw grouped that way,
while discarding a live one throws away the only artifact this project cannot recompute.

Three things here are structural rather than incidental, and each was measured before it
was coded:

1. **Naming the field that moved needs the settings, not the digest.** A digest is a
   one-way function; "config changed" is useless the day after Task 9 writes
   `photocull.toml`. The decision log therefore stores a settings *snapshot* beside the
   digest, on the same reasoning that made `sub_scores` a snapshot rather than a join.
2. **Rule 3 lives inside rule 2.** `cluster_key` is the digest of its sorted member
   uuids, so a photo leaving the library necessarily changes the key of the cluster it
   was in. A voided photo therefore always arrives on an orphaned decision — it is a
   per-photo classification, not a sibling rule.
3. **The `None`-vs-`''` `mod_date` sentinel is load-bearing.** Most photos are never
   edited, so `mod_date` is usually `None` and is stored as `''`. A comparison that
   mixed the two conventions would report every unedited photo as changed and requeue
   the entire library — which reads exactly like "the tool is being careful".
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from photocull.config import ClusterConfig
from photocull_review.decisions import DecisionLog, Mark
from photocull_review.resume import Resume, diff_settings, reconcile
from photocull_review.session import cluster_key, config_digest, session_settings

from ..fixtures import make_cluster

EDITED = datetime(2026, 3, 1, 9, 30, 0)
EDITED_LATER = datetime(2026, 4, 2, 18, 5, 0)


@pytest.fixture
def log(tmp_path):
    with DecisionLog(tmp_path / "decisions.db") as opened:
        yield opened


def decide(log, cluster, *, cfg, session_id="session-1", mark="keep"):
    """Record a decision covering every take in `cluster`, as the UI would."""
    key = cluster_key(cluster)
    log.record(
        session_id=session_id,
        cluster_key=key,
        config_digest=config_digest(cfg),
        settings=session_settings(cfg),
        marks=[
            Mark(photo_uuid=record.uuid, mark=mark, mod_date=record.mod_date)
            for record in cluster.records
        ],
    )
    return key


def uuids_of(*clusters):
    return {record.uuid for cluster in clusters for record in cluster.records}


# --- rule 1: the config moved ---------------------------------------------------


def test_an_unchanged_config_resumes(log):
    cfg = ClusterConfig(similarity_threshold=0.48)
    clusters = [make_cluster(["a", "b"], day=0), make_cluster(["c", "d"], day=1)]
    decide(log, clusters[0], cfg=cfg)

    result = reconcile(clusters, log=log, config=cfg, library_uuids=uuids_of(*clusters))

    assert result.can_resume
    assert result.blocked_reason is None
    assert set(result.applied) == {cluster_key(clusters[0])}


def test_a_changed_threshold_refuses_to_resume_and_names_the_field(log):
    """`blocked_reason` must say *which* setting moved and to what.

    The day after Task 9 writes a calibrated `photocull.toml`, "the configuration
    changed" tells the owner nothing they can act on — they need to see
    `similarity_threshold 0.48 -> 0.52` to decide whether to revert it or start over."""
    old = ClusterConfig(similarity_threshold=0.48)
    new = ClusterConfig(similarity_threshold=0.52)
    clusters = [make_cluster(["a", "b"])]
    decide(log, clusters[0], cfg=old)

    result = reconcile(clusters, log=log, config=new, library_uuids=uuids_of(*clusters))

    assert not result.can_resume
    assert result.applied == {}
    assert result.resume_index is None
    assert "similarity_threshold" in result.blocked_reason
    assert "0.48" in result.blocked_reason and "0.52" in result.blocked_reason
    assert [change.field for change in result.config_changes] == ["similarity_threshold"]


def test_a_retuned_weight_is_named_by_its_nested_key(log):
    """Re-tuning the weights is what `review-decisions-train-the-scorer` produces, and it
    only ever moves a nested dict. A diff over top-level fields would report
    `weights` changed and leave the owner to spot which one."""
    old = ClusterConfig()
    tweaked = dict(old.weights)
    tweaked["sharpness"] = round(tweaked["sharpness"] + 0.01, 4)
    tweaked["exposure"] = round(tweaked["exposure"] - 0.01, 4)
    new = ClusterConfig(weights=tweaked)
    clusters = [make_cluster(["a", "b"])]
    decide(log, clusters[0], cfg=old)

    result = reconcile(clusters, log=log, config=new, library_uuids=uuids_of(*clusters))

    assert not result.can_resume
    assert [change.field for change in result.config_changes] == [
        "weights.exposure",
        "weights.sharpness",
    ]
    assert "weights.sharpness" in result.blocked_reason


def test_the_reason_diffs_against_the_most_recent_settings_not_the_oldest(log):
    """With several superseded sessions on disk, the owner needs the diff against the one
    they were last in. Diffing against the oldest would name a value they moved away from
    two calibrations ago — and a single-batch fixture cannot tell the two apart, because
    there the oldest batch *is* the newest."""
    first = ClusterConfig(similarity_threshold=0.40)
    second = ClusterConfig(similarity_threshold=0.48)
    current = ClusterConfig(similarity_threshold=0.52)
    clusters = [make_cluster(["a", "b"], day=0), make_cluster(["c", "d"], day=1)]
    decide(log, clusters[0], cfg=first, session_id="session-1")
    decide(log, clusters[1], cfg=second, session_id="session-2")

    result = reconcile(clusters, log=log, config=current, library_uuids=uuids_of(*clusters))

    assert not result.can_resume
    assert [(c.field, c.was, c.now) for c in result.config_changes] == [
        ("similarity_threshold", 0.48, 0.52)
    ]
    assert "0.4," not in result.blocked_reason


def test_where_the_config_was_loaded_from_is_not_a_config_change(log):
    """`describe_config` stamps `source`, a path on one machine."""
    cfg = ClusterConfig()
    clusters = [make_cluster(["a", "b"])]
    log.record(
        session_id="s",
        cluster_key=cluster_key(clusters[0]),
        config_digest=config_digest(cfg, source="photocull.toml"),
        settings=session_settings(cfg, source="photocull.toml"),
        marks=[Mark(photo_uuid="a", mark="keep")],
    )

    result = reconcile(
        clusters,
        log=log,
        config=cfg,
        source="/somewhere/else.toml",
        library_uuids=uuids_of(*clusters),
    )

    assert result.can_resume, result.blocked_reason


def test_an_empty_log_is_not_a_config_change(log):
    """A first launch has nothing to compare against and must not read as a conflict."""
    clusters = [make_cluster(["a", "b"])]

    result = reconcile(clusters, log=log, config=ClusterConfig())

    assert result.can_resume
    assert result.resume_index == 0
    assert result.config_changes == ()


# --- force_new_session ----------------------------------------------------------


def test_forcing_a_new_session_starts_fresh_without_destroying_anything(log):
    """Refusing to resume must not be a one-way door 800 clusters in.

    The decision log is append-only, so "start fresh" can only ever mean *this layer
    stops applying* the old decisions — never that they leave the disk."""
    old = ClusterConfig(similarity_threshold=0.48)
    new = ClusterConfig(similarity_threshold=0.52)
    clusters = [make_cluster(["a", "b"], day=0), make_cluster(["c", "d"], day=1)]
    key = decide(log, clusters[0], cfg=old)

    result = reconcile(
        clusters,
        log=log,
        config=new,
        library_uuids=uuids_of(*clusters),
        force_new_session=True,
    )

    assert result.can_resume
    assert result.applied == {}
    assert result.resume_index == 0
    assert key in result.superseded
    # The evidence is still on disk, in full, with its sub-scores.
    assert log.state(key) is not None
    assert len(log.history(key)) == 1


def test_forcing_a_new_session_holds_back_decisions_that_would_otherwise_apply(log):
    """Starting over is not only for a config change: the owner may simply want to redo
    the review. That is the case that actually pins the behaviour — the test above cannot,
    because when the settings have moved there is nothing to apply either way, so it
    passes whether or not `force_new_session` withholds anything."""
    cfg = ClusterConfig()
    clusters = [make_cluster(["a", "b"], day=0), make_cluster(["c", "d"], day=1)]
    key = decide(log, clusters[0], cfg=cfg)

    carried = reconcile(clusters, log=log, config=cfg, library_uuids=uuids_of(*clusters))
    assert set(carried.applied) == {key}, "without forcing, this decision applies"

    result = reconcile(
        clusters,
        log=log,
        config=cfg,
        library_uuids=uuids_of(*clusters),
        force_new_session=True,
    )

    assert result.applied == {}
    assert result.resume_index == 0
    assert log.state(key) is not None


def test_a_forced_session_can_itself_be_resumed_next_launch(log):
    """The counterpart one-way door: having forced past a config change once, the next
    launch under those same settings must resume normally rather than block forever."""
    old = ClusterConfig(similarity_threshold=0.48)
    new = ClusterConfig(similarity_threshold=0.52)
    clusters = [make_cluster(["a", "b"], day=0), make_cluster(["c", "d"], day=1)]
    decide(log, clusters[0], cfg=old)
    # ... the owner forces past it and decides a cluster under the new settings.
    decide(log, clusters[1], cfg=new, session_id="session-2")

    result = reconcile(clusters, log=log, config=new, library_uuids=uuids_of(*clusters))

    assert result.can_resume, result.blocked_reason
    assert set(result.applied) == {cluster_key(clusters[1])}
    assert cluster_key(clusters[0]) in result.superseded


def test_reverting_the_config_returns_to_the_earlier_session(log):
    """Symmetry check: settings are not a ratchet. Putting the old threshold back must
    bring back the decisions made under it rather than stranding them."""
    old = ClusterConfig(similarity_threshold=0.48)
    new = ClusterConfig(similarity_threshold=0.52)
    clusters = [make_cluster(["a", "b"], day=0), make_cluster(["c", "d"], day=1)]
    decide(log, clusters[0], cfg=old)
    decide(log, clusters[1], cfg=new, session_id="session-2")

    result = reconcile(clusters, log=log, config=old, library_uuids=uuids_of(*clusters))

    assert result.can_resume
    assert set(result.applied) == {cluster_key(clusters[0])}


# --- rule 2: the cluster is gone ------------------------------------------------


def test_a_decision_on_a_vanished_cluster_is_kept_for_audit_but_not_applied(log):
    cfg = ClusterConfig()
    decided = make_cluster(["a", "b"], day=0)
    key = decide(log, decided, cfg=cfg)
    # The rescan produces different groupings; the decided cluster no longer exists.
    fresh = [make_cluster(["a", "b", "z"], day=0), make_cluster(["c", "d"], day=1)]

    result = reconcile(fresh, log=log, config=cfg, library_uuids=uuids_of(*fresh))

    assert result.orphaned == (key,)
    assert key not in result.applied
    assert result.resume_index == 0
    assert log.state(key) is not None, "an orphaned decision stays on disk for audit"


# --- rule 3: the photo is gone --------------------------------------------------


def test_a_photo_gone_from_the_library_is_voided_and_reported(log):
    cfg = ClusterConfig()
    decided = make_cluster(["a", "b"], day=0)
    key = decide(log, decided, cfg=cfg)
    # `b` was deleted in Photos.app, so the rescan sees only `a`, unclustered.
    fresh = [make_cluster(["c", "d"], day=1)]

    result = reconcile(fresh, log=log, config=cfg, library_uuids={"a", "c", "d"})

    assert [(void.cluster_key, void.photo_uuid) for void in result.voided] == [(key, "b")]
    assert result.voided[0].mark == "keep"


def test_a_voided_photo_never_reaches_an_applied_decision(log):
    """The property that matters for write-back, stated directly: nothing in `applied`
    may name a photo that is no longer in the library. This holds structurally — losing a
    member changes the cluster's key — but it is the invariant Task 14 depends on, so it
    is pinned here rather than left to be re-derived."""
    cfg = ClusterConfig()
    decided = make_cluster(["a", "b"], day=0)
    decide(log, decided, cfg=cfg)
    fresh = [make_cluster(["a", "b"], day=0)]

    result = reconcile(fresh, log=log, config=cfg, library_uuids={"a"})

    applied_uuids = {
        mark.photo_uuid for decision in result.applied.values() for mark in decision.marks
    }
    assert "b" not in applied_uuids


def test_without_a_library_listing_no_photo_is_declared_gone(log):
    """`library_uuids=None` means "not checked", never "the library is empty".

    Only 4,498 of this library's 14,235 photos reach a cluster, so inferring the library
    from the clusters would declare two thirds of it deleted."""
    cfg = ClusterConfig()
    decided = make_cluster(["a", "b"], day=0)
    decide(log, decided, cfg=cfg)

    result = reconcile([make_cluster(["c", "d"], day=1)], log=log, config=cfg)

    assert result.voided == ()


# --- rule 4: a decided photo was edited -----------------------------------------


def test_an_edited_photo_requeues_its_whole_cluster(log):
    """Not just the photo. The edit changes that take's derivative, which changes its
    feature print, its sharpness and therefore the ranking of every take beside it — the
    comparison the owner made no longer exists."""
    cfg = ClusterConfig()
    before = make_cluster(["a", "b", "c"], day=0, mod_dates={"b": EDITED})
    key = decide(log, before, cfg=cfg)
    after = make_cluster(["a", "b", "c"], day=0, mod_dates={"b": EDITED_LATER})

    result = reconcile([after], log=log, config=cfg, library_uuids=uuids_of(after))

    assert result.requeued == (key,)
    assert key not in result.applied
    assert result.resume_index == 0, "a requeued cluster is undecided again"


def test_an_unedited_library_requeues_nothing(log):
    """The sentinel test. Every one of these photos has `mod_date=None`, stored as `''`;
    a comparison that mixed the conventions would requeue all of them."""
    cfg = ClusterConfig()
    clusters = [make_cluster(["a", "b"], day=0), make_cluster(["c", "d"], day=1)]
    for cluster in clusters:
        decide(log, cluster, cfg=cfg)

    result = reconcile(clusters, log=log, config=cfg, library_uuids=uuids_of(*clusters))

    assert result.requeued == ()
    assert len(result.applied) == 2
    assert result.resume_index is None


def test_the_same_instant_in_a_different_timezone_still_requeues(log):
    """`mod_key` compares isoformat strings, not instants, and that is deliberate rather
    than incidental: `AnalysisCache` keys on the very same string, so a `mod_date` whose
    representation moves is a cache miss and the photo gets re-analysed — new derivative
    measurement, new sub-scores, new ranking. Comparing the two `datetime`s directly
    would call this unchanged (they are `==`) and carry over a decision made against
    scores that no longer exist."""
    cfg = ClusterConfig()
    utc = datetime(2026, 3, 1, 9, 30, tzinfo=timezone.utc)
    same_instant_elsewhere = utc.astimezone(timezone(timedelta(hours=2)))
    assert utc == same_instant_elsewhere and utc.isoformat() != same_instant_elsewhere.isoformat()

    before = make_cluster(["a", "b"], day=0, mod_dates={"a": utc})
    key = decide(log, before, cfg=cfg)
    after = make_cluster(["a", "b"], day=0, mod_dates={"a": same_instant_elsewhere})

    result = reconcile([after], log=log, config=cfg, library_uuids=uuids_of(after))

    assert result.requeued == (key,)


def test_a_photo_edited_since_being_decided_from_never_edited(log):
    """The `None` -> a real timestamp transition, which is what a first edit looks like."""
    cfg = ClusterConfig()
    before = make_cluster(["a", "b"], day=0)
    key = decide(log, before, cfg=cfg)
    after = make_cluster(["a", "b"], day=0, mod_dates={"a": EDITED})

    result = reconcile([after], log=log, config=cfg, library_uuids=uuids_of(after))

    assert result.requeued == (key,)


# --- rule 5: where to pick up ---------------------------------------------------


def test_resume_lands_on_the_first_undecided_cluster_in_capture_order(log):
    cfg = ClusterConfig()
    clusters = [
        make_cluster(["a", "b"], day=0),
        make_cluster(["c", "d"], day=1),
        make_cluster(["e", "f"], day=2),
    ]
    decide(log, clusters[0], cfg=cfg)
    decide(log, clusters[2], cfg=cfg)

    result = reconcile(clusters, log=log, config=cfg, library_uuids=uuids_of(*clusters))

    assert result.resume_index == 1
    assert result.resume_cluster_key == cluster_key(clusters[1])
    assert result.decided_count == 2


def test_a_fully_decided_library_has_nowhere_to_resume(log):
    cfg = ClusterConfig()
    clusters = [make_cluster(["a", "b"], day=0), make_cluster(["c", "d"], day=1)]
    for cluster in clusters:
        decide(log, cluster, cfg=cfg)

    result = reconcile(clusters, log=log, config=cfg, library_uuids=uuids_of(*clusters))

    assert result.resume_index is None
    assert result.resume_cluster_key is None
    assert result.decided_count == 2


def test_an_undone_decision_reopens_its_cluster(log):
    cfg = ClusterConfig()
    clusters = [make_cluster(["a", "b"], day=0), make_cluster(["c", "d"], day=1)]
    key = decide(log, clusters[0], cfg=cfg)
    decide(log, clusters[1], cfg=cfg)
    log.undo(key)

    result = reconcile(clusters, log=log, config=cfg, library_uuids=uuids_of(*clusters))

    assert result.resume_index == 0
    assert key not in result.applied


# --- the settings snapshot ------------------------------------------------------


def test_the_stored_settings_redigest_to_the_stored_digest(log):
    """The two are written by the caller as separate arguments, so they *can* disagree —
    the shape of silent-wrong-answer bug this project has hit five times. `reconcile`
    checks rather than trusts, and says so loudly."""
    cfg = ClusterConfig(similarity_threshold=0.48)
    clusters = [make_cluster(["a", "b"])]
    log.record(
        session_id="s",
        cluster_key=cluster_key(clusters[0]),
        config_digest=config_digest(ClusterConfig(similarity_threshold=0.40)),
        settings=session_settings(cfg),
        marks=[Mark(photo_uuid="a", mark="keep")],
    )

    with pytest.raises(ValueError, match="digest"):
        reconcile(clusters, log=log, config=cfg, library_uuids=uuids_of(*clusters))


def test_a_batch_recorded_without_settings_still_blocks_but_says_so(log):
    """A file written before the snapshot table existed carries a digest and nothing
    else. It must still refuse to resume across a config change — just without being able
    to name the field, which the reason has to admit rather than imply."""
    old = ClusterConfig(similarity_threshold=0.48)
    new = ClusterConfig(similarity_threshold=0.52)
    clusters = [make_cluster(["a", "b"])]
    log.record(
        session_id="s",
        cluster_key=cluster_key(clusters[0]),
        config_digest=config_digest(old),
        marks=[Mark(photo_uuid="a", mark="keep")],
    )

    result = reconcile(clusters, log=log, config=new, library_uuids=uuids_of(*clusters))

    assert not result.can_resume
    assert result.config_changes == ()
    assert "not recorded" in result.blocked_reason


# --- diff_settings --------------------------------------------------------------


def test_diff_settings_reports_added_and_removed_fields():
    """A new tunable is a config change too, and reads as neither side's value."""
    changes = diff_settings({"a": 1, "gone": 2}, {"a": 1, "added": 3})
    assert [(c.field, c.was, c.now) for c in changes] == [
        ("added", None, 3),
        ("gone", 2, None),
    ]


def test_diff_settings_is_empty_for_identical_settings():
    settings = session_settings(ClusterConfig())
    assert diff_settings(settings, dict(settings)) == ()


# --- the reported shape ---------------------------------------------------------


def test_the_result_is_json_serialisable_for_the_api(log):
    """Task 8 hands this to the browser, so it has to survive `json.dumps` without a
    custom encoder for anything but the decisions themselves."""
    import json

    cfg = ClusterConfig()
    clusters = [make_cluster(["a", "b"], day=0)]
    decide(log, clusters[0], cfg=cfg)
    result = reconcile(clusters, log=log, config=cfg, library_uuids=uuids_of(*clusters))

    assert isinstance(result, Resume)
    payload = json.dumps(result.report())
    assert "resume_index" in payload
