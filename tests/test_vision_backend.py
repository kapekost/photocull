"""Tests for the Vision backend's pure parts.

`vision_backend` is the live macOS layer: feature-print extraction itself needs the
Vision framework and a real image, so it cannot be unit-tested here (same pattern as
Phase 0's `photos_source.open_library`). What IS testable is covered: the distance
function, the `VisionUnavailable` contract, and the `FeaturePrinter` protocol that
later tasks fake against. The live path is verified by the tick's smoke step.
"""

import pytest

from photocull.vision_backend import (
    FaceObservation,
    FeaturePrinter,
    VisionUnavailable,
    eye_aspect_ratio,
    framing_score,
    horizon_penalty,
    l2_distance,
    smile_curvature,
)


def test_l2_distance_of_identical_vectors_is_zero():
    assert l2_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_l2_distance_is_euclidean():
    assert l2_distance([0.0, 0.0], [3.0, 4.0]) == pytest.approx(5.0)


def test_l2_distance_is_symmetric():
    a, b = [0.1, 0.9, 0.3], [0.4, 0.2, 0.8]
    assert l2_distance(a, b) == pytest.approx(l2_distance(b, a))


def test_l2_distance_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        l2_distance([1.0, 2.0], [1.0])


def test_l2_distance_is_euclidean_in_three_dimensions():
    """Regression guard for Euclidean semantics across more than two axes -- a
    sum-of-absolute-differences bug passes the 2D case above but fails here.

    This does NOT compare against Apple's own computeDistance:, which needs the
    framework and a real image; that cross-check lives in `apple_distance()` and is
    exercised by the tick's live smoke step (measured agreement: 8e-09)."""
    assert l2_distance([0.0, 0.3, 0.4], [0.0, 0.0, 0.0]) == pytest.approx(0.5)


def test_l2_distance_of_empty_vectors_is_zero():
    assert l2_distance([], []) == pytest.approx(0.0)


def test_l2_distance_handles_a_full_length_feature_print():
    """Exercise the real payload width (768) rather than a 3-element toy, so a
    length assumption baked into the implementation would show up here."""
    a = [0.0] * 768
    b = [0.0] * 767 + [2.0]
    assert l2_distance(a, b) == pytest.approx(2.0)


def test_vision_unavailable_is_a_runtime_error():
    """Callers that don't care why Vision is missing can catch RuntimeError."""
    assert issubclass(VisionUnavailable, RuntimeError)


def test_a_callable_satisfies_the_feature_printer_protocol():
    """Later tasks accept a fake feature printer instead of the live one."""

    def fake_printer(path: str):
        return [0.0] * 768

    assert isinstance(fake_printer, FeaturePrinter)


def test_a_non_callable_does_not_satisfy_the_feature_printer_protocol():
    assert not isinstance(object(), FeaturePrinter)


# --- Task 6: faces and horizon -------------------------------------------------
#
# Same split as above: `detect_faces`/`horizon_angle` need the live framework and are
# covered by the tick's smoke step. The geometry they feed is pure, and is what these
# tests pin -- including the two corrections the Task 6 spikes forced (see DECISIONS.md
# `vision-pose-angles-are-quantized` and `face-capture-quality-is-the-real-pose-signal`).


def test_eye_aspect_ratio_of_an_open_eye_is_higher_than_a_closed_one():
    open_eye = [(0.0, 0.0), (0.5, 0.2), (1.0, 0.0), (0.5, -0.2)]
    shut_eye = [(0.0, 0.0), (0.5, 0.02), (1.0, 0.0), (0.5, -0.02)]
    assert eye_aspect_ratio(open_eye) > eye_aspect_ratio(shut_eye)


def test_eye_aspect_ratio_handles_degenerate_input():
    assert eye_aspect_ratio([]) == 0.0
    assert eye_aspect_ratio([(0.0, 0.0)]) == 0.0


def test_eye_aspect_ratio_of_zero_width_points_is_zero():
    """A vertical line of points has no horizontal extent -- must not divide by zero."""
    assert eye_aspect_ratio([(0.5, 0.0), (0.5, 0.3)]) == 0.0


def test_eye_aspect_ratio_applies_the_box_aspect_correction():
    """Vision's landmark points are normalised to the FACE BOX, not to the image, so
    x and y are in different units until the box's own aspect is applied. Without this
    the same eye scores differently just because the box is wider or taller -- which
    would make the value incomparable between two takes of the same person.

    Measured on 25 real faces: raw median 0.3512 (max 1.4672), corrected median 0.3188
    (max 1.1004) -- the correction is what pulls the impossible >1.0 ratios back."""
    eye = [(0.0, 0.0), (0.5, 0.2), (1.0, 0.0), (0.5, -0.2)]
    assert eye_aspect_ratio(eye, box_aspect=0.5) == pytest.approx(
        eye_aspect_ratio(eye) * 0.5
    )


def test_smile_curvature_is_positive_when_the_corners_lift():
    """Vision ships no smile classifier -- there is no smile-like property on a real
    VNFaceObservation -- so this is derived from the outerLips landmarks."""
    smiling = [(0.0, 0.1), (0.5, 0.0), (1.0, 0.1)]
    assert smile_curvature(smiling) > 0


def test_smile_curvature_is_negative_when_the_corners_turn_down():
    frowning = [(0.0, 0.0), (0.5, 0.1), (1.0, 0.0)]
    assert smile_curvature(frowning) < 0


def test_smile_curvature_handles_degenerate_input():
    assert smile_curvature([]) == 0.0
    assert smile_curvature([(0.5, 0.0), (0.5, 0.2), (0.5, 0.4)]) == 0.0


def test_horizon_penalty_is_one_for_a_level_frame():
    assert horizon_penalty(0.0) == pytest.approx(1.0)


def test_horizon_penalty_falls_off_as_tilt_grows():
    assert horizon_penalty(0.05) > horizon_penalty(0.30)


def test_horizon_penalty_is_symmetric_about_level():
    """Real angles are signed -- measured range on this library is -0.131 to +0.170 rad.
    A tilt left must cost exactly what the same tilt right costs."""
    assert horizon_penalty(-0.12) == pytest.approx(horizon_penalty(0.12))


def test_horizon_penalty_of_none_is_neutral():
    """Vision fires a horizon on only ~38% of photos (23/60 measured). A miss means
    the criterion does not apply and must be dropped from the weighted total by the
    caller -- scoring it as 0.0 would punish every photo without a visible horizon."""
    assert horizon_penalty(None) is None


def test_framing_score_rewards_a_centred_subject():
    centred = FaceObservation(bounding_box=(0.4, 0.4, 0.2, 0.2))
    edged = FaceObservation(bounding_box=(0.0, 0.4, 0.2, 0.2))
    assert framing_score([centred]) > framing_score([edged])


def test_framing_score_penalises_a_subject_cut_off_by_the_frame():
    cut = FaceObservation(bounding_box=(-0.05, 0.4, 0.2, 0.2))
    whole = FaceObservation(bounding_box=(0.4, 0.4, 0.2, 0.2))
    assert framing_score([cut]) < framing_score([whole])


def test_framing_score_penalises_a_subject_cut_off_at_the_far_edges():
    """The first real photo sampled had bounding_box.x = -0.0229, so cut-off is not a
    hypothetical. Guard the right/top edges too -- a `min(x, y)`-only implementation
    ignores them.

    The cut-off faces here are deliberately MORE centred than the intact one, so the
    centre-distance term alone would score them higher. Only an implementation that
    actually checks all four edges can make these assertions hold; the obvious version
    of this test (an off-centre cut face vs a centred whole one) passes even when the
    cut-off branch is deleted, because centring dominates the result."""
    intact_but_off_centre = FaceObservation(bounding_box=(0.05, 0.4, 0.2, 0.2))
    cut_right = FaceObservation(bounding_box=(0.45, 0.4, 0.6, 0.2))
    cut_top = FaceObservation(bounding_box=(0.4, 0.45, 0.2, 0.6))
    assert framing_score([cut_right]) < framing_score([intact_but_off_centre])
    assert framing_score([cut_top]) < framing_score([intact_but_off_centre])


def test_framing_score_judges_the_largest_face():
    """Half the sampled selfies had 2+ faces, so which face is judged is load-bearing.
    A tiny bystander at the frame edge must not drag down a well-framed main subject.

    The bystander is listed FIRST: with the main face first, a `faces[0]`
    implementation passes this by coincidence."""
    main = FaceObservation(bounding_box=(0.4, 0.4, 0.3, 0.3))
    bystander = FaceObservation(bounding_box=(0.0, 0.0, 0.02, 0.02))
    assert framing_score([bystander, main]) == pytest.approx(framing_score([main]))


def test_framing_score_with_no_faces_is_neutral():
    assert framing_score([]) is None


def test_face_observation_defaults_are_all_unknown():
    """Every Vision-derived field defaults to None so that "not measured" is
    distinguishable from "measured as zero" -- scoring must drop the former, not
    average it in."""
    f = FaceObservation()
    assert f.yaw is None
    assert f.roll is None
    assert f.left_eye_open is None
    assert f.right_eye_open is None
    assert f.smiling is None
    assert f.capture_quality is None
