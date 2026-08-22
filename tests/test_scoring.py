"""Best-shot scoring.

Several tests here exist to pin owner decisions and Task 6 measurements rather than
mere behaviour -- specifically that `smiling` and `facing` carry NO weight until the
owner rules at Task 9. If a later change gives either of them weight, these fail
loudly, which is the point. See DECISIONS.md `vision-pose-angles-are-quantized`.
"""

import pytest

from photocull.config import ClusterConfig
from photocull.imaging import PixelStats
from photocull.models import PhotoScore
from photocull.scoring import (
    cluster_sharpness,
    exposure_score,
    faces_score,
    on_common_criteria,
    rank_cluster,
    score_photo,
    smile_score,
    weighted_total,
)
from photocull.vision_backend import FaceObservation
from tests.fixtures import make_photo_record


def face(**kw):
    base = dict(bounding_box=(0.1, 0.1, 0.2, 0.2), left_eye_open=0.9,
                right_eye_open=0.9, smiling=0.5)
    base.update(kw)
    return FaceObservation(**base)


# --- weighted total + renormalisation -----------------------------------------


def test_weighted_total_combines_sub_scores():
    cfg = ClusterConfig(weights={"sharpness": 0.5, "exposure": 0.5})
    assert weighted_total({"sharpness": 1.0, "exposure": 0.0}, cfg) == pytest.approx(0.5)


def test_inapplicable_criteria_are_dropped_and_weights_renormalised():
    """A landscape has no faces. It must not be punished for that -- the faces
    weight is removed and the rest renormalised."""
    cfg = ClusterConfig(weights={"sharpness": 0.5, "faces": 0.5})
    assert weighted_total({"sharpness": 0.8, "faces": None}, cfg) == pytest.approx(0.8)


def test_all_criteria_inapplicable_scores_zero_rather_than_dividing_by_zero():
    cfg = ClusterConfig(weights={"faces": 1.0})
    assert weighted_total({"faces": None}, cfg) == pytest.approx(0.0)


def test_sub_scores_absent_from_the_weights_are_ignored():
    """capture_quality rides along for the owner to inspect at Task 9. Carrying a
    diagnostic must not silently move the total."""
    cfg = ClusterConfig(weights={"sharpness": 1.0})
    assert weighted_total({"sharpness": 0.6, "capture_quality": 0.99}, cfg) == pytest.approx(0.6)


def test_zero_weighted_criteria_do_not_move_the_total():
    cfg = ClusterConfig(weights={"sharpness": 1.0, "smiling": 0.0})
    assert weighted_total({"sharpness": 0.4, "smiling": 1.0}, cfg) == pytest.approx(0.4)


# --- the two decisions this task must not quietly redefine ---------------------


def test_smiling_carries_no_weight_by_default():
    """Task 6 measured `smiling` negative on 9 of 12 real selfies -- the mapping is
    not validated, so it must not influence a pick until the owner says so."""
    assert ClusterConfig().weights["smiling"] == 0.0


def test_facing_carries_no_weight_by_default():
    """Vision's yaw/roll are quantized to 45/30 degree steps, so sub-score (b) cannot
    rank two takes of one burst. Owner decides its replacement at Task 9."""
    assert ClusterConfig().weights["facing"] == 0.0


def test_default_weights_sum_to_one():
    """Task 7b's loader rejects weight tables that do not sum to 1.0; the shipped
    defaults must satisfy their own rule."""
    assert sum(ClusterConfig().weights.values()) == pytest.approx(1.0)


# --- faces --------------------------------------------------------------------


def test_faces_score_sinks_a_take_where_someone_is_blinking():
    """With several people the take where EVERYONE has their eyes open must beat the
    one with a single great face next to a blinker."""
    all_good = [face(), face(bounding_box=(0.5, 0.1, 0.2, 0.2))]
    one_blinking = [
        face(),
        face(bounding_box=(0.5, 0.1, 0.2, 0.2), left_eye_open=0.02, right_eye_open=0.02),
    ]
    assert faces_score(all_good) > faces_score(one_blinking)


def test_a_single_closed_eye_dominates_the_face_score():
    """Spec: closed eyes is the single strongest reject-this-take signal, and it must
    win even when the blinking face is the one that is smiling."""
    open_face = [face(left_eye_open=0.9, right_eye_open=0.9, smiling=0.1)]
    shut_face = [face(left_eye_open=0.01, right_eye_open=0.01, smiling=0.9)]
    assert faces_score(open_face) > faces_score(shut_face)


def test_faces_score_uses_the_worst_face_not_the_average():
    """Written after a mutation test showed the blinker case above passes even for an
    AVERAGING implementation. These numbers are chosen so min and mean disagree: the
    pair averages higher (0.75 vs 0.60) but contains the worse face (0.50 vs 0.60)."""
    evenly_ok = [face(left_eye_open=0.6, right_eye_open=0.6)]
    one_great_one_poor = [
        face(left_eye_open=1.0, right_eye_open=1.0),
        face(bounding_box=(0.5, 0.1, 0.2, 0.2), left_eye_open=0.5, right_eye_open=0.5),
    ]
    assert faces_score(evenly_ok) > faces_score(one_great_one_poor)


def test_faces_score_uses_the_worst_eye_not_the_average():
    """One eye shut is a blink; averaging the two would hide it."""
    both_open = [face(left_eye_open=0.8, right_eye_open=0.8)]
    one_shut = [face(left_eye_open=1.0, right_eye_open=0.05)]
    assert faces_score(both_open) > faces_score(one_shut)


def test_faces_score_with_no_faces_is_inapplicable():
    assert faces_score([]) is None


def test_faces_score_ignores_smiling_entirely():
    """Guards the decision above at the function level, not just in config."""
    grinning = [face(smiling=1.0)]
    glum = [face(smiling=-1.0)]
    assert faces_score(grinning) == faces_score(glum)


def test_smile_score_is_still_computed_for_the_owner_to_inspect():
    assert smile_score([face(smiling=0.8)]) > smile_score([face(smiling=-0.8)])
    assert smile_score([]) is None


def test_faces_score_when_vision_reported_no_eye_values():
    """None means "not measured", which must not read as "eyes shut"."""
    assert faces_score([face(left_eye_open=None, right_eye_open=None)]) is None


# --- score_photo --------------------------------------------------------------


def test_score_photo_assembles_sub_scores_and_a_total():
    s = score_photo("a", faces=[face()], horizon=0.0, sharpness=0.8, exposure=0.7)
    assert s.uuid == "a"
    assert s.sub_scores["sharpness"] == 0.8
    assert s.sub_scores["horizon"] == pytest.approx(1.0)
    assert 0.0 < s.total <= 1.0


def test_score_photo_marks_missing_criteria_none_rather_than_zero():
    s = score_photo("a", faces=[], horizon=None, sharpness=0.8, exposure=None)
    assert s.sub_scores["faces"] is None
    assert s.sub_scores["horizon"] is None
    assert s.sub_scores["framing"] is None
    assert s.total == pytest.approx(0.8)


def test_score_photo_carries_capture_quality_as_a_diagnostic():
    s = score_photo("a", faces=[face(capture_quality=0.42)], sharpness=0.5)
    assert s.sub_scores["capture_quality"] == pytest.approx(0.42)


# --- common-criteria correction -----------------------------------------------


def test_a_criterion_missing_from_one_take_cannot_decide_the_cluster():
    """The real defect this exists for, in miniature: two near-identical takes where
    Vision found a horizon in only one. Horizon scores far higher than eye-openness, so
    renormalising per-photo hands the win to whichever frame got the detection.

    Measured on a real 9-frame burst before the fix: 6 frames with a horizon totalled
    ~0.30, the 3 without ~0.17, with nothing else distinguishing them."""
    lucky = PhotoScore(uuid="lucky", sub_scores={"faces": 0.11, "horizon": 0.87})
    unlucky = PhotoScore(uuid="unlucky", sub_scores={"faces": 0.12, "horizon": None})
    naive = rank_cluster([], [
        PhotoScore(uuid=s.uuid, sub_scores=s.sub_scores,
                   total=weighted_total(s.sub_scores, ClusterConfig()))
        for s in (lucky, unlucky)
    ])
    assert naive.winner_uuid == "lucky"  # the artifact, reproduced

    fixed = rank_cluster([], on_common_criteria([lucky, unlucky]))
    assert fixed.winner_uuid == "unlucky"  # decided by the eyes, which is the signal


def test_on_common_criteria_keeps_a_criterion_all_takes_share():
    a = PhotoScore(uuid="a", sub_scores={"faces": 0.2, "horizon": 0.9})
    b = PhotoScore(uuid="b", sub_scores={"faces": 0.2, "horizon": 0.1})
    out = {s.uuid: s.total for s in on_common_criteria([a, b])}
    assert out["a"] > out["b"]


def test_on_common_criteria_leaves_sub_scores_untouched_for_the_ui():
    a = PhotoScore(uuid="a", sub_scores={"faces": 0.2, "horizon": 0.9})
    b = PhotoScore(uuid="b", sub_scores={"faces": 0.2, "horizon": None})
    assert on_common_criteria([a, b])[0].sub_scores["horizon"] == 0.9


def test_on_common_criteria_handles_an_empty_cluster():
    assert on_common_criteria([]) == []


# --- cluster ranking ----------------------------------------------------------


def test_rank_cluster_picks_the_highest_total_as_winner():
    records = [make_photo_record(uuid="a"), make_photo_record(uuid="b")]
    scores = [PhotoScore(uuid="a", sub_scores={}, total=0.9),
              PhotoScore(uuid="b", sub_scores={}, total=0.2)]
    cluster = rank_cluster(records, scores)
    assert cluster.winner_uuid == "a"
    assert cluster.is_ambiguous is False


def test_rank_cluster_flags_a_near_tie_as_ambiguous():
    """Never auto-pick a coin flip -- DECISIONS.md ambiguous-clusters-go-to-manual-pick."""
    records = [make_photo_record(uuid="a"), make_photo_record(uuid="b")]
    scores = [PhotoScore(uuid="a", sub_scores={}, total=0.80),
              PhotoScore(uuid="b", sub_scores={}, total=0.79)]
    assert rank_cluster(records, scores).is_ambiguous is True


def test_ambiguity_margin_is_configurable():
    records = [make_photo_record(uuid="a"), make_photo_record(uuid="b")]
    scores = [PhotoScore(uuid="a", sub_scores={}, total=0.80),
              PhotoScore(uuid="b", sub_scores={}, total=0.70)]
    assert rank_cluster(records, scores).is_ambiguous is False
    strict = ClusterConfig(ambiguity_margin=0.2)
    assert rank_cluster(records, scores, config=strict).is_ambiguous is True


def test_a_gap_exactly_equal_to_the_margin_is_not_ambiguous():
    """The boundary itself, which a mutation test found untested. Values are exact in
    binary (1.0 - 0.5 == 0.5 with no float slop) so this really pins < against <=."""
    records = [make_photo_record(uuid="a"), make_photo_record(uuid="b")]
    scores = [PhotoScore(uuid="a", sub_scores={}, total=1.0),
              PhotoScore(uuid="b", sub_scores={}, total=0.5)]
    cfg = ClusterConfig(ambiguity_margin=0.5)
    assert rank_cluster(records, scores, config=cfg).is_ambiguous is False


def test_an_ambiguous_cluster_still_reports_its_best_guess_winner():
    """The UI shows a grid with no default, but downstream code still needs a
    deterministic ordering."""
    records = [make_photo_record(uuid="a"), make_photo_record(uuid="b")]
    scores = [PhotoScore(uuid="a", sub_scores={}, total=0.80),
              PhotoScore(uuid="b", sub_scores={}, total=0.79)]
    assert rank_cluster(records, scores).winner_uuid == "a"


def test_rank_cluster_is_deterministic_when_totals_tie():
    """Ordering bugs have shipped twice in this project (Ticks 7 and 8), both times
    because output order depended on input order. An exact tie must not."""
    records = [make_photo_record(uuid="a"), make_photo_record(uuid="b")]
    tie_a = [PhotoScore(uuid="a", sub_scores={}, total=0.5),
             PhotoScore(uuid="b", sub_scores={}, total=0.5)]
    tie_b = [PhotoScore(uuid="b", sub_scores={}, total=0.5),
             PhotoScore(uuid="a", sub_scores={}, total=0.5)]
    first = rank_cluster(records, tie_a)
    second = rank_cluster(records, tie_b)
    assert first.winner_uuid == second.winner_uuid
    assert [s.uuid for s in first.scores] == [s.uuid for s in second.scores]


def test_rank_cluster_keeps_every_record_and_score():
    records = [make_photo_record(uuid="a"), make_photo_record(uuid="b")]
    scores = [PhotoScore(uuid="a", sub_scores={}, total=0.9),
              PhotoScore(uuid="b", sub_scores={}, total=0.2)]
    cluster = rank_cluster(records, scores)
    assert len(cluster.records) == 2
    assert {s.uuid for s in cluster.scores} == {"a", "b"}


def test_a_single_photo_cluster_is_never_ambiguous():
    records = [make_photo_record(uuid="a")]
    scores = [PhotoScore(uuid="a", sub_scores={}, total=0.5)]
    cluster = rank_cluster(records, scores)
    assert cluster.winner_uuid == "a"
    assert cluster.is_ambiguous is False


def test_an_empty_cluster_has_no_winner():
    cluster = rank_cluster([], [])
    assert cluster.winner_uuid is None
    assert cluster.is_ambiguous is False


def _stats(variance=1000.0, width=480, height=360, mean=0.46, shadow=0.0, highlight=0.0):
    return PixelStats(
        width=width,
        height=height,
        laplacian_variance=variance,
        mean_luma=mean,
        shadow_clipped=shadow,
        highlight_clipped=highlight,
    )


def test_exposure_peaks_at_the_librarys_median_luma():
    assert exposure_score(_stats(mean=0.46)) == pytest.approx(1.0, abs=1e-3)


def test_exposure_falls_off_for_a_very_dark_frame():
    assert exposure_score(_stats(mean=0.08)) < 0.2


def test_clipping_is_penalised():
    clean = exposure_score(_stats(mean=0.46))
    blown = exposure_score(_stats(mean=0.46, highlight=0.10))
    assert blown < clean


def test_exposure_is_none_when_the_photo_could_not_be_read():
    assert exposure_score(None) is None


def test_sharpness_is_relative_to_the_best_take_in_the_cluster():
    scores = cluster_sharpness([_stats(variance=1000.0), _stats(variance=500.0)])
    assert scores == pytest.approx([1.0, 0.5])


def test_sharpness_preserves_the_size_of_a_small_real_difference():
    """Within real burst groups laplacian variance spreads only 1.04-1.27x. A min-max
    normalisation would map that onto the full 0-1 range and invent a confident winner;
    ratio-to-best keeps a 4% difference looking like a 4% difference."""
    scores = cluster_sharpness([_stats(variance=1040.0), _stats(variance=1000.0)])
    assert scores[1] == pytest.approx(0.9615, abs=1e-3)


def test_sharpness_is_dropped_entirely_when_rasters_differ_in_scale():
    """Not "scored lower" -- dropped. A 480px and a 1024px raster produce laplacian
    variances that differ by more than any real difference between two takes, so any
    comparison between them is noise wearing a number."""
    scores = cluster_sharpness([_stats(width=480, height=360), _stats(width=1024, height=768)])
    assert scores == [None, None]


def test_sharpness_tolerance_can_admit_near_matches():
    cfg = ClusterConfig(sharpness_size_tolerance=1.05)
    scores = cluster_sharpness([_stats(width=480, height=360), _stats(width=478, height=360)], cfg)
    assert scores[0] is not None and scores[1] is not None


def test_an_unreadable_member_drops_sharpness_for_the_whole_cluster():
    """on_common_criteria would drop it anyway; doing it here keeps the reason visible
    and stops a two-photo cluster ranking on a criterion only one of them has."""
    assert cluster_sharpness([_stats(), None]) == [None, None]


def test_sharpness_of_a_single_photo_cluster_is_defined():
    assert cluster_sharpness([_stats(variance=1000.0)]) == [1.0]


def test_sharpness_of_an_all_black_cluster_does_not_divide_by_zero():
    assert cluster_sharpness([_stats(variance=0.0), _stats(variance=0.0)]) == [None, None]


def test_a_transposed_raster_still_compares_equal():
    """The size gate compares LONG SIDES, so a portrait take and a landscape take of
    the same pixel class are comparable. image_dimensions reports pre-EXIF-transform
    dimensions while PixelStats reports post-transform ones, so a gate written on
    (width, height) rather than the long side would drop sharpness for whole clusters
    on nothing but orientation."""
    scores = cluster_sharpness([_stats(width=480, height=360), _stats(width=360, height=480)])
    assert scores == pytest.approx([1.0, 1.0])
