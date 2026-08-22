"""The live Apple Vision layer. Everything that needs macOS frameworks lives here
and nowhere else, so the rest of Phase 1 stays unit-testable with plain fakes.

Reads pixels only from paths handed to it by `derivatives.derivative_path` --
never an original (see that module's docstring for why).

Distances in production come from `l2_distance`, not from Apple's own
`computeDistance:`. The two agree to ~8e-09 on a real burst pair (verified in this
module's tick), and the hand-rolled version needs no framework, so clustering stays
testable with plain lists of floats. `apple_distance` exists to re-verify that
agreement against the live framework; it is the only caller of the pyobjc metadata
workaround below."""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

#: Vision returns VNElementTypeFloat (float32) feature prints -- confirmed on the
#: real library: elementType 1, elementCount 768, exactly 4.0 bytes per element.
_BYTES_PER_ELEMENT = 4


class VisionUnavailable(RuntimeError):
    """Raised when the Vision/pyobjc frameworks can't be loaded at all, or when
    Vision returns a feature print in a layout this module can't safely decode."""


@runtime_checkable
class FeaturePrinter(Protocol):
    """What later tasks depend on instead of the live backend, so they can be
    tested with a fake. Runtime-checkable, so `isinstance` confirms only that a
    candidate is callable -- the vector width is the caller's contract, not the
    protocol's."""

    def __call__(self, path: str) -> list[float] | None: ...


def l2_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two feature-print vectors.

    Verified equivalent to Apple's own computeDistance: on a real burst pair --
    0.42237750490829395 hand-rolled vs 0.42237749695777893 from Vision, an absolute
    difference of 7.95e-09. Hand-rolled because numpy is not a dependency and a
    768-element loop is not the bottleneck (feature print extraction is 6.7ms; this
    is microseconds)."""
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


_METADATA_REGISTERED = False


def _ensure_metadata() -> None:
    """pyobjc does not know computeDistance:'s first argument is a float* OUT
    parameter, so without this it returns (True, None) and every distance is
    lost -- silently, with no exception. Reproduced on the real library before this
    module was written; only `apple_distance` needs it."""
    global _METADATA_REGISTERED
    if _METADATA_REGISTERED:
        return
    import objc

    objc.registerMetaDataForSelector(
        b"VNFeaturePrintObservation",
        b"computeDistance:toFeaturePrintObservation:error:",
        {"arguments": {2: {"type": b"^f", "type_modifier": b"o"}}},
    )
    _METADATA_REGISTERED = True


def _frameworks() -> tuple[Any, Any]:
    try:
        import Vision
        from Foundation import NSURL
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise VisionUnavailable(str(exc)) from exc
    return Vision, NSURL


def _handler(path: str) -> Any:
    Vision, NSURL = _frameworks()
    url = NSURL.fileURLWithPath_(path)
    return Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})


def _observation(path: str) -> Any | None:
    Vision, _NSURL = _frameworks()
    handler = _handler(path)
    request = Vision.VNGenerateImageFeaturePrintRequest.alloc().init()
    ok, _err = handler.performRequests_error_([request], None)
    if not ok:
        return None
    results = request.results()
    return results[0] if results else None


def _vector(obs: Any) -> list[float]:
    count = obs.elementCount()
    raw = bytes(obs.data())
    if count and len(raw) // count != _BYTES_PER_ELEMENT:
        # Guard against a silently-wrong answer: unpacking float64 data as float32
        # yields a plausible-looking vector of garbage rather than an error.
        raise VisionUnavailable(
            f"unexpected feature print layout: {len(raw)} bytes for {count} elements"
        )
    return list(struct.unpack(f"<{count}f", raw[: count * _BYTES_PER_ELEMENT]))


def feature_print(path: str) -> list[float] | None:
    """Return the 768-element feature print for an image, or None if Vision
    could not process it (corrupt/unsupported derivative)."""
    obs = _observation(path)
    if obs is None:
        return None
    return _vector(obs)


def apple_distance(path_a: str, path_b: str) -> float | None:
    """Vision's own distance between two images, for cross-checking `l2_distance`.

    Not used by the clustering pipeline -- it needs the live framework, and the
    hand-rolled version agrees to ~8e-09. Kept so that agreement can be re-measured
    on demand rather than trusted from a planning note, and because it is what makes
    the `_ensure_metadata` workaround above reachable and therefore maintained."""
    _ensure_metadata()
    obs_a = _observation(path_a)
    obs_b = _observation(path_b)
    if obs_a is None or obs_b is None:
        return None
    _ok, distance = obs_a.computeDistance_toFeaturePrintObservation_error_(
        None, obs_b, None
    )
    return None if distance is None else float(distance)


# --- Faces and horizon ---------------------------------------------------------


@dataclass
class FaceObservation:
    """Vision's face result, reduced to what scoring needs.

    `bounding_box` is Vision's normalised (x, y, width, height) with origin at
    BOTTOM-left. Every measured field defaults to None so "Vision did not report
    this" stays distinguishable from "Vision reported zero" -- scoring must drop the
    former from its weighted total rather than average a 0.0 into it.

    `capture_quality` is Apple's own face-quality score and is the most useful field
    here; `yaw`/`roll` are quantized and nearly useless for comparing two takes. See
    DECISIONS.md `vision-pose-angles-are-quantized`."""

    bounding_box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    yaw: float | None = None
    roll: float | None = None
    left_eye_open: float | None = None
    right_eye_open: float | None = None
    smiling: float | None = None
    capture_quality: float | None = None


def eye_aspect_ratio(
    points: Sequence[tuple[float, float]], box_aspect: float = 1.0
) -> float:
    """Vertical extent over horizontal extent of the eye landmark points. A closed eye
    collapses vertically, so this drops toward 0.

    `box_aspect` is the face box's height/width. Vision normalises landmark points to
    the face BOX, so x and y arrive in different units; without the correction the
    same eye scores differently purely because the box is wider or taller. Measured
    over 25 real faces: raw median 0.3512 with a max of 1.4672 (an eye taller than it
    is wide, which is geometrically impossible), corrected median 0.3188, max 1.1004.

    Derived geometrically because Vision has no usable eyes-open signal: the
    `isBlinking`/`blinkScore` properties exist on VNFaceObservation but returned
    False/0.0 for every face sampled, at every supported revision (1-3)."""
    if len(points) < 2:
        return 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    width = max(xs) - min(xs)
    if width <= 0:
        return 0.0
    return ((max(ys) - min(ys)) / width) * box_aspect


def smile_curvature(
    points: Sequence[tuple[float, float]], box_aspect: float = 1.0
) -> float:
    """How far the mouth corners sit above the lip centre, over mouth width.

    Positive means the corners are lifted (a smile), negative means they turn down.
    Vision ships no smile classifier at all -- `dir()` on a real VNFaceObservation has
    no smile-like property -- so this is derived from the outerLips landmarks."""
    if len(points) < 3:
        return 0.0
    xs = [p[0] for p in points]
    width = max(xs) - min(xs)
    if width <= 0:
        return 0.0
    left = min(points, key=lambda p: p[0])
    right = max(points, key=lambda p: p[0])
    corner_y = (left[1] + right[1]) / 2.0
    centre_y = sum(p[1] for p in points) / len(points)
    return ((corner_y - centre_y) / width) * box_aspect


def horizon_penalty(angle: float | None) -> float | None:
    """1.0 for a perfectly level frame, decaying as tilt grows.

    None in means None out: Vision found no horizon, so the criterion does not apply
    and must be dropped from the weighted total rather than scored as zero. That is
    the common case, not an edge case -- the request fired on only 23 of 60 real
    photos (38%)."""
    if angle is None:
        return None
    return math.exp(-abs(angle) * 6.0)


def framing_score(faces: Sequence[FaceObservation]) -> float | None:
    """Reward subjects that are centred and fully inside the frame.

    Judges the LARGEST face: half the sampled photos had two or more, and a small
    bystander at the edge must not drag down a well-framed main subject. None when
    there are no faces -- the criterion does not apply."""
    if not faces:
        return None
    best = max(faces, key=lambda f: f.bounding_box[2] * f.bounding_box[3])
    x, y, w, h = best.bounding_box
    cut_off = min(x, y, 1.0 - (x + w), 1.0 - (y + h))
    centre_distance = math.hypot((x + w / 2) - 0.5, (y + h / 2) - 0.5)
    score = math.exp(-centre_distance * 2.0)
    if cut_off < 0:
        score *= max(0.0, 1.0 + cut_off * 4.0)
    return score


def _points(region: Any) -> list[tuple[float, float]]:
    """Landmark region -> plain (x, y) tuples, so the geometry above never touches a
    CoreFoundation type and stays testable with literals."""
    if region is None:
        return []
    count = region.pointCount()
    raw = region.normalizedPoints()
    return [(float(raw[i].x), float(raw[i].y)) for i in range(count)]


def detect_faces(path: str) -> list[FaceObservation]:
    """Faces in an image, with pose, framing, derived eye-openness and smile, and
    Apple's own capture-quality score.

    Runs the landmarks request first, then feeds ITS observations into the
    capture-quality request via `setInputFaceObservations_`. That chaining is load
    bearing: issuing both requests in a single `performRequests_` call makes the
    quality request re-detect independently, and the observations it returns come back
    with `landmarks() is None` -- silently losing every geometric signal below."""
    Vision, _NSURL = _frameworks()

    handler = _handler(path)
    landmarks_request = Vision.VNDetectFaceLandmarksRequest.alloc().init()
    ok, _err = handler.performRequests_error_([landmarks_request], None)
    if not ok:
        return []
    faces = landmarks_request.results() or []
    if not faces:
        return []

    quality_request = Vision.VNDetectFaceCaptureQualityRequest.alloc().init()
    quality_request.setInputFaceObservations_(faces)
    ok, _err = handler.performRequests_error_([quality_request], None)
    scored = (quality_request.results() or []) if ok else []
    # Fall back to the landmark observations if quality could not be computed; the
    # geometric sub-scores are still worth having without it.
    observations = scored if len(scored) == len(faces) else faces

    result: list[FaceObservation] = []
    for face in observations:
        box = face.boundingBox()
        width = float(box.size.width)
        height = float(box.size.height)
        box_aspect = height / width if width else 1.0

        landmarks = face.landmarks()
        left = right = lips = []
        if landmarks is not None:
            left = _points(landmarks.leftEye())
            right = _points(landmarks.rightEye())
            lips = _points(landmarks.outerLips())

        result.append(
            FaceObservation(
                bounding_box=(float(box.origin.x), float(box.origin.y), width, height),
                yaw=None if face.yaw() is None else float(face.yaw()),
                roll=None if face.roll() is None else float(face.roll()),
                left_eye_open=eye_aspect_ratio(left, box_aspect) if left else None,
                right_eye_open=eye_aspect_ratio(right, box_aspect) if right else None,
                smiling=smile_curvature(lips, box_aspect) if lips else None,
                capture_quality=(
                    None
                    if face.faceCaptureQuality() is None
                    else float(face.faceCaptureQuality())
                ),
            )
        )
    return result


def horizon_angle(path: str) -> float | None:
    """Signed tilt of the detected horizon in radians, or None if Vision found no
    horizon (the majority of photos -- see `horizon_penalty`)."""
    Vision, _NSURL = _frameworks()
    handler = _handler(path)
    request = Vision.VNDetectHorizonRequest.alloc().init()
    ok, _err = handler.performRequests_error_([request], None)
    if not ok:
        return None
    results = request.results() or []
    return float(results[0].angle()) if results else None
