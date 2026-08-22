"""The Quartz/ImageIO pixel layer. Everything that decodes or measures an image file
lives here, so the rest of Phase 1 stays unit-testable with plain bytes.

Separate from `vision_backend` on purpose: that module is the *Vision* framework
(feature prints, faces, horizon), this one is CoreGraphics (dimensions, rasters). They
fail independently and are mocked independently.

Only ever receives paths from `derivatives.derivative_path` -- never `.path`, never
`.export()`. See CLAUDE.md hard rule #2."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

#: Pixel values at or below/above these count as clipped. Measured over 250 real
#: photos: shadow clipping runs median 0.0064 / p90 0.0458, highlight clipping median
#: 0.0005 / p90 0.0073 -- so these thresholds fire on real photos without firing on
#: most of them.
_SHADOW_CUT = 3
_HIGHLIGHT_CUT = 252


class ImagingUnavailable(RuntimeError):
    """Quartz could not be loaded at all."""


@dataclass(frozen=True)
class GrayBitmap:
    """An 8-bit grayscale raster. `stride` is bytes per row, which is NOT always equal
    to `width` -- CoreGraphics is free to pad. Every consumer must step by stride."""

    data: bytes
    width: int
    height: int
    stride: int


@dataclass(frozen=True)
class PixelStats:
    """Raw pixel measurements. Turning these into 0-1 sub-scores is `scoring`'s job,
    the same split `vision_backend.FaceObservation` already uses.

    `width`/`height` are the DECODED raster's, after the EXIF orientation transform --
    unlike `image_dimensions`. They are what the sharpness comparability gate compares,
    so the two must not be mixed."""

    width: int
    height: int
    laplacian_variance: float
    mean_luma: float
    shadow_clipped: float
    highlight_clipped: float


def _quartz() -> tuple[Any, Any]:
    try:
        import Quartz
        from Foundation import NSURL
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImagingUnavailable(str(exc)) from exc
    return Quartz, NSURL


def _source(path: str) -> Any | None:
    Quartz, NSURL = _quartz()
    url = NSURL.fileURLWithPath_(path)
    return Quartz.CGImageSourceCreateWithURL(url, None)


def image_dimensions(path: str) -> tuple[int, int] | None:
    """Pixel dimensions from the file header -- no decode, ~1.8 ms per call.

    These are ImageIO's `PixelWidth`/`PixelHeight`, which are reported BEFORE the EXIF
    orientation transform, so they can be the transpose of the raster a decoder
    produces. That is fine for the one thing this is used for -- choosing the smallest
    derivative, where area is invariant under a transpose -- and is exactly why it must
    NOT be used to decide whether two rasters are the same shape (see Task 7d)."""
    Quartz, _NSURL = _quartz()
    src = _source(path)
    if src is None:
        return None
    props = Quartz.CGImageSourceCopyPropertiesAtIndex(src, 0, None)
    if props is None:
        return None
    width = props.get("PixelWidth")
    height = props.get("PixelHeight")
    if not width or not height:
        return None
    return int(width), int(height)


def gray_bitmap(path: str, max_pixels: int | None = None) -> GrayBitmap | None:
    """Decode an image to 8-bit grayscale at its native size.

    Native size on purpose: laplacian variance is strongly scale-dependent (2779 at
    192px vs 1153 at native for one image) and resampling does NOT make two rasters
    comparable -- the same photo through its two derivatives still disagrees 1.19-1.89x
    at an identical 384px. Resampling to a canonical size would therefore hide the
    problem rather than fix it. The comparability gate in `scoring.cluster_sharpness`
    is the actual answer."""
    Quartz, _NSURL = _quartz()
    src = _source(path)
    if src is None:
        return None
    options = {
        Quartz.kCGImageSourceCreateThumbnailFromImageAlways: True,
        Quartz.kCGImageSourceCreateThumbnailWithTransform: True,
    }
    if max_pixels is not None:
        options[Quartz.kCGImageSourceThumbnailMaxPixelSize] = max_pixels
    else:
        dims = image_dimensions(path)
        if dims is None:
            return None
        options[Quartz.kCGImageSourceThumbnailMaxPixelSize] = max(dims)
    image = Quartz.CGImageSourceCreateThumbnailAtIndex(src, 0, options)
    if image is None:
        return None
    width = int(Quartz.CGImageGetWidth(image))
    height = int(Quartz.CGImageGetHeight(image))
    if width <= 0 or height <= 0:
        return None
    space = Quartz.CGColorSpaceCreateDeviceGray()
    ctx = Quartz.CGBitmapContextCreate(
        None, width, height, 8, width, space, Quartz.kCGImageAlphaNone
    )
    if ctx is None:
        return None
    Quartz.CGContextDrawImage(ctx, Quartz.CGRectMake(0, 0, width, height), image)
    # Read the stride back rather than assuming the requested one was honoured. It was
    # on all 200 files sampled, but assuming it is exactly the kind of silently-wrong
    # answer this project keeps meeting -- a mismatch shears every row.
    stride = int(Quartz.CGBitmapContextGetBytesPerRow(ctx))
    data = Quartz.CGBitmapContextGetData(ctx)
    if data is None:
        return None
    return GrayBitmap(
        data=bytes(data.as_buffer(stride * height)),
        width=width,
        height=height,
        stride=stride,
    )


def laplacian_variance(bitmap: GrayBitmap) -> float:
    """Variance of the 4-neighbour laplacian -- the standard focus measure.

    Row slices rather than per-pixel indexing keeps the inner work in C; measured at
    21.2 ms for a 480x360 raster, ~7.3 min for the whole library, once, then cached."""
    w, h, s, buf = bitmap.width, bitmap.height, bitmap.stride, bitmap.data
    if w < 3 or h < 3:
        return 0.0
    total = 0.0
    total_sq = 0.0
    n = 0
    for y in range(1, h - 1):
        row = buf[y * s : y * s + w]
        up = buf[(y - 1) * s : (y - 1) * s + w]
        down = buf[(y + 1) * s : (y + 1) * s + w]
        for c, left, right, u, d in zip(
            row[1 : w - 1], row[0 : w - 2], row[2:w], up[1 : w - 1], down[1 : w - 1]
        ):
            v = 4 * c - left - right - u - d
            total += v
            total_sq += v * v
            n += 1
    if n == 0:
        return 0.0
    mean = total / n
    return total_sq / n - mean * mean


def exposure_stats(bitmap: GrayBitmap) -> tuple[float, float, float]:
    """(mean luma 0-1, shadow-clipped fraction, highlight-clipped fraction).

    Unlike sharpness these are scale-invariant, verified on 30 real photos across a
    2.1x scale change: mean luma moved by a median of 0.06/255 (max 0.64) and the
    clipping fractions by ~0.001. So exposure needs no comparability gate.

    `Counter.update` per row, benchmarked against the two obvious alternatives on 20
    real derivatives (identical results from all three): 3.19 ms here, 5.10 ms for a
    per-pixel `counts[b] += 1` loop, and 19.59 ms for 256 `bytes.count` scans per row.
    The last one looks like the clever option and is 6x the slowest."""
    w, h, s, buf = bitmap.width, bitmap.height, bitmap.stride, bitmap.data
    tally: Counter[int] = Counter()
    for y in range(h):
        tally.update(buf[y * s : y * s + w])
    n = w * h
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(value * count for value, count in tally.items()) / n / 255.0
    shadow = sum(c for v, c in tally.items() if v <= _SHADOW_CUT) / n
    highlight = sum(c for v, c in tally.items() if v >= _HIGHLIGHT_CUT) / n
    return mean, shadow, highlight


def pixel_stats(path: str, max_pixels: int | None = None) -> PixelStats | None:
    """One decode, both measurements. None when the file cannot be read."""
    bitmap = gray_bitmap(path, max_pixels)
    if bitmap is None:
        return None
    mean, shadow, highlight = exposure_stats(bitmap)
    return PixelStats(
        width=bitmap.width,
        height=bitmap.height,
        laplacian_variance=laplacian_variance(bitmap),
        mean_luma=mean,
        shadow_clipped=shadow,
        highlight_clipped=highlight,
    )
