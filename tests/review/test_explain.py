"""Task 11: the evidence panel — the sentences that say *why* a take was ranked.

The owner's report was "I don't understand what the values are... I'm not sure what I'm
looking at". Task 10 put ten numbered rows under each photograph, which is the same
information and the same problem: `sharpness 0.912` against `sharpness 0.865` is correct
and does not tell anyone that the left photograph is the sharper one.

Every rule tested here is a rule about *not misleading*, which is why they are tests and
not taste:

- A criterion `on_common_criteria` excluded from the comparison cannot be a reason,
  because the scorer did not use it. Reporting it would describe a ranking that did not
  happen — and `horizon` is dropped in 77.6% of real clusters, so this is the common
  case rather than an edge.
- A zero-weighted signal moved nothing (`unvalidated-signals-ship-at-zero-weight`), so a
  sentence about it is a claim the total does not support.
- A difference the score table renders as identical cannot be offered as the reason one
  take beat the other.
- An ambiguous cluster is 40.1% of this library. Prose that reads as a verdict there is
  exactly the "silent, plausible wrongness" this project keeps finding.
"""

from __future__ import annotations

import dataclasses

import pytest

from photocull.config import ClusterConfig
from photocull.models import Cluster, PhotoScore
from photocull_review.explain import (
    MAX_REASONS,
    MIN_VISIBLE_DELTA,
    explain,
    ranking_basis,
)

from ..fixtures import make_cluster

ALL_CRITERIA = {
    "sharpness": 0.5,
    "exposure": 0.5,
    "faces": 0.5,
    "framing": 0.5,
    "horizon": 0.5,
}


def cluster_with(scores, *, ambiguous=False, dropped=()):
    """A two-or-more-take cluster whose sub-scores are stated per uuid.

    `totals` are derived from the weighted mean so the sentences and the ranking cannot
    disagree — a fixture that let them drift would be testing prose against nothing."""
    from photocull.scoring import weighted_total

    cfg = ClusterConfig()
    totals = {u: weighted_total(sub, cfg) for u, sub in scores.items()}
    return make_cluster(
        list(scores),
        totals=totals,
        sub_scores=scores,
        ambiguous=ambiguous,
        dropped=dropped,
    )


def joined(sentences):
    return " ".join(sentences).lower()


def winner_of(cluster: Cluster) -> str:
    return cluster.winner_uuid


# --- the decisive criterion leads --------------------------------------------------


def test_the_decisive_criterion_is_named_first():
    """Weight times difference, not difference alone: `exposure` at 0.15 moving 0.40 is
    worth more than `sharpness` at 0.35 moving 0.01, and the sentence order has to say
    which one actually decided it."""
    by_sharpness = cluster_with(
        {
            "a": ALL_CRITERIA | {"sharpness": 1.00, "exposure": 0.70},
            "b": ALL_CRITERIA | {"sharpness": 0.70, "exposure": 0.50},
        }
    )
    first = explain(by_sharpness, "a")[0].lower()
    assert "sharp" in first, first

    by_exposure = cluster_with(
        {
            "a": ALL_CRITERIA | {"sharpness": 1.00, "exposure": 0.90},
            "b": ALL_CRITERIA | {"sharpness": 0.99, "exposure": 0.50},
        }
    )
    sentences = explain(by_exposure, "a")
    assert "exposed" in sentences[0].lower(), sentences
    # Still reported, just not as the reason — 0.01 clears the visibility floor.
    assert any("sharp" in s.lower() for s in sentences[1:]), sentences


def test_the_order_is_weight_times_movement_not_movement_alone():
    """The distinguishing case, and the reason the test above cannot stand alone: there,
    the heavier criterion also moved further, so dropping the weight from the sort left
    the order unchanged and the mutation survived.

    Here they disagree. `framing` moves 0.50 at weight 0.05 (worth 0.025) against
    `sharpness` moving 0.10 at weight 0.35 (worth 0.035): sorting on movement puts
    framing first, and framing decided almost nothing."""
    cluster = cluster_with(
        {
            "a": ALL_CRITERIA | {"sharpness": 1.00, "framing": 0.90},
            "b": ALL_CRITERIA | {"sharpness": 0.90, "framing": 0.40},
        }
    )
    sentences = explain(cluster, "a")
    assert "sharp" in sentences[0].lower(), sentences
    assert "frame" in sentences[1].lower(), sentences


def test_a_reason_says_which_way_it_went():
    """The template pair is the whole content of a criterion sentence. Swapping better
    for worse leaves every sentence present, well-formed, in the right order — and
    exactly backwards, under a photograph the reader is being asked to trust it about."""
    cluster = cluster_with(
        {
            "a": ALL_CRITERIA | {"exposure": 0.90, "faces": 0.90, "horizon": 0.90, "framing": 0.90},
            "b": ALL_CRITERIA | {"exposure": 0.30, "faces": 0.30, "horizon": 0.30, "framing": 0.30},
        }
    )
    better = joined(explain(cluster, "a"))
    worse = joined(explain(cluster, "b"))
    assert "better exposed" in better and "less well exposed" in worse
    assert "eyes are more open" in better and "eyes are less open" in worse
    # Capped at three reasons, so which four of the criteria appear is not fixed — but
    # no sentence may ever run the wrong way.
    assert "more level horizon" not in worse and "more tilted horizon" not in better
    assert "sits better in the frame" not in worse


def test_the_same_ordering_holds_for_the_take_that_lost():
    cluster = cluster_with(
        {
            "a": ALL_CRITERIA | {"sharpness": 1.00, "exposure": 0.70},
            "b": ALL_CRITERIA | {"sharpness": 0.70, "exposure": 0.50},
        }
    )
    assert "sharp" in explain(cluster, "b")[0].lower()


# --- what may never be cited -------------------------------------------------------


def test_a_criterion_dropped_from_the_comparison_is_never_cited():
    """`on_common_criteria` re-totals over the criteria measured for ALL takes, so a
    criterion missing from one take moved nothing. Citing it would name a reason the
    ranking never used."""
    cluster = cluster_with(
        {
            "a": ALL_CRITERIA | {"horizon": 0.95},
            "b": {k: v for k, v in ALL_CRITERIA.items() if k != "horizon"} | {"horizon": None},
        }
    )
    for uuid in ("a", "b"):
        assert "horizon" not in joined(explain(cluster, uuid)), explain(cluster, uuid)
    # It is named exactly once, in the cluster's basis line, as something absent.
    assert "horizon" in ranking_basis(cluster)


def test_only_the_top_reasons_are_shown():
    """Five reasons plus a verdict is an audit, and it costs the frames the height it
    takes. The score table above the panel still shows every criterion, so the cap hides
    nothing — it only stops the panel claiming that all five mattered equally."""
    cluster = cluster_with(
        {
            "a": {"sharpness": 1.00, "exposure": 0.62, "faces": 0.81, "framing": 0.74, "horizon": 0.90},
            "b": {"sharpness": 0.78, "exposure": 0.70, "faces": 0.12, "framing": 0.30, "horizon": 0.55},
        }
    )
    sentences = explain(cluster, "a")
    assert ranking_basis(cluster) is None, "this fixture measures everything"
    # Every sentence but the standing line is a reason.
    assert len(sentences) == MAX_REASONS + 1, sentences
    assert "pick" in sentences[-1].lower()


def test_a_zero_weighted_signal_is_never_cited():
    """`facing`, `capture_quality` and `smiling` ship at weight 0.0 and are computed as
    diagnostics only. They move no total, so no sentence may attribute anything to them
    — the review UI already shows their numbers, and a *reason* is a stronger claim."""
    cluster = cluster_with(
        {
            "a": ALL_CRITERIA | {"capture_quality": 0.90, "smiling": 0.90, "facing": 0.90},
            "b": ALL_CRITERIA | {"capture_quality": 0.10, "smiling": 0.10, "facing": 0.10},
        }
    )
    text = joined(explain(cluster, "a"))
    assert "quality" not in text and "smil" not in text and "facing" not in text, text


def test_a_difference_the_score_table_cannot_show_is_not_cited():
    """`renderScores` prints three decimals. A 0.001 gap renders as two identical rows,
    so offering it as the reason one take won is emphasis with nothing behind it."""
    cluster = cluster_with(
        {
            "a": ALL_CRITERIA | {"exposure": 0.5000},
            "b": ALL_CRITERIA | {"exposure": 0.5 - MIN_VISIBLE_DELTA / 2},
        }
    )
    reasons = [s for s in explain(cluster, "a") if "exposed" in s.lower()]
    assert reasons == []
    assert any("nothing" in s.lower() for s in explain(cluster, "a"))


def test_the_visibility_floor_is_inclusive():
    """Exactly at the floor counts as visible; the score table does render it.

    The delta is built by subtracting from zero on purpose. The first version of this
    test used `0.5 + MIN_VISIBLE_DELTA` against `0.5`, whose difference is
    0.0050000000000000044 — *above* the floor, so it never tested the boundary and a
    mutation from `>=` to `>` survived it. Same shape as `0.90 - 0.60` giving
    0.30000000000000004 in `complete-linkage-shipped-as-measured`, and the same lesson:
    a boundary fixture has to be checked, not assumed."""
    cluster = cluster_with(
        {
            "a": ALL_CRITERIA | {"exposure": MIN_VISIBLE_DELTA},
            "b": ALL_CRITERIA | {"exposure": 0.0},
        }
    )
    assert MIN_VISIBLE_DELTA - 0.0 == MIN_VISIBLE_DELTA, "the fixture is not on the floor"
    assert any("exposed" in s.lower() for s in explain(cluster, "a"))


# --- ratios are ratios --------------------------------------------------------------


def test_sharpness_is_stated_as_a_ratio_and_never_as_a_variance():
    """`cluster_sharpness` already emits a ratio to the sharpest take, and that is the
    only honest form: laplacian variance has no absolute scale, so "sharpness 1153"
    means nothing to anyone and is not comparable between clusters."""
    cluster = cluster_with(
        {
            "a": ALL_CRITERIA | {"sharpness": 1.00},
            "b": ALL_CRITERIA | {"sharpness": 0.75},
            "c": ALL_CRITERIA | {"sharpness": 0.75},
        }
    )
    best = [s for s in explain(cluster, "a") if "sharp" in s.lower()][0]
    assert "1.33×" in best, best
    assert "3 takes" in best, best

    softer = [s for s in explain(cluster, "b") if "sharp" in s.lower()][0]
    assert "0.75×" in softer, softer


def test_a_barely_sharper_take_says_so():
    """Measured take-to-take spread inside real burst groups is 1.04-1.27x, so "sharpest
    by 4%" is the *common* case here, not an edge one. Reporting it the same way as a
    9.55x difference would make the sentence useless exactly where it is most used."""
    cluster = cluster_with(
        {
            "a": ALL_CRITERIA | {"sharpness": 1.00},
            "b": ALL_CRITERIA | {"sharpness": 0.96},
        }
    )
    best = [s for s in explain(cluster, "a") if "sharp" in s.lower()][0]
    assert "only just" in best.lower(), best


def test_sharpness_survives_a_cluster_whose_softest_take_measured_zero():
    """`cluster_sharpness` clamps to 0-1 and only drops the criterion when the *best*
    take is featureless, so a 0.0 member is reachable and must not divide by zero."""
    cluster = cluster_with(
        {
            "a": ALL_CRITERIA | {"sharpness": 1.00},
            "b": ALL_CRITERIA | {"sharpness": 0.00},
        }
    )
    best = [s for s in explain(cluster, "a") if "sharp" in s.lower()][0]
    assert "sharpest" in best.lower()
    assert "×" not in best, best


# --- standing: the verdict sentence -------------------------------------------------


def test_a_confident_cluster_names_the_pick_and_its_margin():
    cluster = cluster_with(
        {
            "a": ALL_CRITERIA | {"sharpness": 1.00},
            "b": ALL_CRITERIA | {"sharpness": 0.20},
        }
    )
    standing = [s for s in explain(cluster, "a") if "pick" in s.lower()]
    assert len(standing) == 1, explain(cluster, "a")
    assert "ahead" in standing[0].lower()


def test_an_ambiguous_cluster_says_so_rather_than_asserting_a_winner():
    """`ambiguous-clusters-go-to-manual-pick`: the scorer cannot tell these apart. The
    close-call annotation already sits beside the proposal, but the evidence panel must
    not contradict it by announcing a pick two lines below."""
    cluster = cluster_with(
        {
            "a": ALL_CRITERIA | {"sharpness": 1.00},
            "b": ALL_CRITERIA | {"sharpness": 0.98},
        },
        ambiguous=True,
    )
    for uuid in ("a", "b"):
        sentences = explain(cluster, uuid)
        assert any("too close to call" in s.lower() for s in sentences), sentences
        assert not any("the scorer's pick" in s.lower() for s in sentences), sentences

    # And it is about *this* take, not about the cluster. Found by looking at the
    # screen: the first version repeated one cluster-level sentence verbatim in both
    # panes, directly under a close-call annotation that already said the same thing —
    # three copies of "coin flip" on the 40.1% of clusters that are close calls.
    keeper_line = [s for s in explain(cluster, "a") if "too close" in s.lower()][0]
    other_line = [s for s in explain(cluster, "b") if "too close" in s.lower()][0]
    assert keeper_line != other_line
    assert "ahead" in keeper_line and "behind" in other_line
    # The margin is what the annotation cannot say, so it must be here.
    assert "0.007" in keeper_line, keeper_line


def test_an_also_ran_in_an_ambiguous_cluster_is_told_its_real_distance():
    """The close call is about the top two. A take 0.3 behind is not part of it, and
    telling it "too close to call" would be false about that photograph."""
    cluster = cluster_with(
        {
            "a": ALL_CRITERIA | {"sharpness": 1.00},
            "b": ALL_CRITERIA | {"sharpness": 0.98},
            "c": ALL_CRITERIA | {"sharpness": 0.10},
        },
        ambiguous=True,
    )
    sentences = explain(cluster, "c")
    assert any("behind the proposed keeper" in s.lower() for s in sentences), sentences
    assert not any("too close to call" in s.lower() for s in sentences), sentences


def test_the_take_that_lost_is_told_how_far_behind_it_is():
    cluster = cluster_with(
        {
            "a": ALL_CRITERIA | {"sharpness": 1.00},
            "b": ALL_CRITERIA | {"sharpness": 0.20},
        }
    )
    behind = [s for s in explain(cluster, "b") if "behind" in s.lower()]
    assert len(behind) == 1, explain(cluster, "b")


# --- the basis sentence -------------------------------------------------------------


def test_the_basis_names_what_was_measured_and_what_was_not():
    """`annotations` demoted `dropped_criteria` to this panel precisely because it is the
    library's background condition (90.3% of clusters) rather than a per-cluster warning
    — "ranked on sharpness and exposure" is useful prose; a warning triangle on nine
    clusters in ten is not."""
    cluster = cluster_with(
        {
            "a": {"sharpness": 1.0, "exposure": 0.6, "faces": None, "framing": None, "horizon": None},
            "b": {"sharpness": 0.5, "exposure": 0.6, "faces": 0.4, "framing": 0.4, "horizon": 0.4},
        }
    )
    basis = ranking_basis(cluster)
    assert "sharpness" in basis and "exposure" in basis
    for name in ("faces", "framing", "horizon"):
        assert name in basis, basis


def test_the_basis_is_a_cluster_fact_and_never_repeated_per_take():
    """It is the same sentence for every member — which is exactly why it is not in
    `explain`: returned per take it renders twice on screen, once under each pane."""
    cluster = cluster_with(
        {
            "a": ALL_CRITERIA | {"horizon": None},
            "b": ALL_CRITERIA | {"horizon": None, "sharpness": 0.4},
        }
    )
    for uuid in ("a", "b"):
        assert not any("could not be measured" in s for s in explain(cluster, uuid))


def test_no_basis_when_every_criterion_applied():
    cluster = cluster_with(
        {"a": ALL_CRITERIA | {"sharpness": 1.0}, "b": ALL_CRITERIA | {"sharpness": 0.5}}
    )
    assert ranking_basis(cluster) is None


def test_a_cluster_with_nothing_measurable_says_that_outright():
    """Every weighted criterion absent means `weighted_total` returned 0.0 for every
    take and the ranking is the uuid tiebreak. That has to be legible, because the
    proposal still looks exactly like a considered one."""
    cluster = cluster_with(
        {
            "a": {"sharpness": None, "exposure": None, "faces": None, "framing": None, "horizon": None},
            "b": {"sharpness": None, "exposure": None, "faces": None, "framing": None, "horizon": None},
        }
    )
    assert "arbitrary" in ranking_basis(cluster)


def test_the_basis_of_an_empty_cluster_is_nothing():
    assert ranking_basis(Cluster()) is None


def test_the_common_criteria_match_the_pipelines_own_dropped_list():
    """`pipeline._annotate` stamps `dropped_criteria` with the same rule this module
    re-derives from the sub-scores. They are computed in two places, so a test holds
    them together rather than a comment asking nicely."""
    cluster = cluster_with(
        {
            "a": ALL_CRITERIA | {"horizon": None, "faces": None},
            "b": ALL_CRITERIA,
        }
    )
    cfg = ClusterConfig()
    dropped_by_pipeline = sorted(
        name
        for name, weight in cfg.weights.items()
        if weight > 0 and not all(s.sub_scores.get(name) is not None for s in cluster.scores)
    )
    basis = ranking_basis(cluster)
    for name in dropped_by_pipeline:
        assert name in basis, (name, basis)
    assert dropped_by_pipeline == ["faces", "horizon"]


# --- configurability ----------------------------------------------------------------


def test_a_criterion_the_owner_enables_gets_an_honest_sentence():
    """`unvalidated-signals-ship-at-zero-weight` exists so the owner can enable one by
    editing a value in `photocull.toml`, with no code change. A phrasing table that only
    knows today's five criteria must therefore degrade to something true, not vanish."""
    cfg = dataclasses.replace(
        ClusterConfig(),
        weights={
            "sharpness": 0.0,
            "faces": 0.0,
            "exposure": 0.0,
            "horizon": 0.0,
            "framing": 0.0,
            "facing": 0.0,
            "capture_quality": 1.0,
            "smiling": 0.0,
        },
    )
    cluster = make_cluster(
        ["a", "b"],
        totals={"a": 0.9, "b": 0.1},
        sub_scores={"a": {"capture_quality": 0.9}, "b": {"capture_quality": 0.1}},
    )
    sentences = explain(cluster, "a", config=cfg)
    assert any("capture_quality" in s for s in sentences), sentences


# --- degenerate shapes ---------------------------------------------------------------


def test_a_single_take_cluster_explains_nothing():
    """There is no comparison to report, and the compare view already says the cluster
    holds one take."""
    assert explain(make_cluster(["only"]), "only") == []


def test_an_unknown_uuid_explains_nothing():
    assert explain(make_cluster(["a", "b"]), "not-a-member") == []


def test_an_empty_cluster_explains_nothing():
    assert explain(Cluster(), "a") == []


def test_the_proposed_keeper_is_the_clusters_own_winner_not_the_top_total():
    """Two modules derive "the proposed keeper" and they must not disagree.

    The compare view pins its left pane to `entry.winner_uuid`, so that is who the words
    "the proposed keeper" refer to on screen. `rank_cluster` always makes it the
    top-total take, which is why re-deriving it from the totals passes every other test
    here — but then the pane and the sentence under it would be naming different
    photographs the moment they ever parted, and nothing would say so."""
    cluster = Cluster(
        records=[],
        scores=[
            PhotoScore(uuid="a", sub_scores={"sharpness": 1.0, "exposure": 0.9}, total=0.9),
            PhotoScore(uuid="b", sub_scores={"sharpness": 0.4, "exposure": 0.2}, total=0.4),
        ],
        winner_uuid="b",
    )
    assert any("scorer's pick" in s for s in explain(cluster, "b")), explain(cluster, "b")
    assert any("Behind the proposed keeper" in s for s in explain(cluster, "a"))


def test_a_cluster_with_no_winner_still_explains_its_takes():
    """`proposed_marks` leaves a winnerless cluster entirely unset; the panel must not
    raise on the way past."""
    cluster = Cluster(
        records=[],
        scores=[
            PhotoScore(uuid="a", sub_scores={"sharpness": 1.0}, total=0.9),
            PhotoScore(uuid="b", sub_scores={"sharpness": 0.4}, total=0.4),
        ],
        winner_uuid=None,
    )
    assert explain(cluster, "b") != []


@pytest.mark.parametrize("uuid", ["a", "b", "c"])
def test_no_sentence_is_ever_empty_or_unterminated(uuid):
    cluster = cluster_with(
        {
            "a": ALL_CRITERIA | {"sharpness": 1.0, "horizon": None},
            "b": ALL_CRITERIA | {"sharpness": 0.6, "horizon": None},
            "c": ALL_CRITERIA | {"sharpness": 0.3, "horizon": None},
        }
    )
    for sentence in explain(cluster, uuid):
        assert sentence.strip() == sentence and sentence
        assert sentence[0].isupper(), sentence
        assert sentence.endswith("."), sentence
