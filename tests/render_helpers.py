"""Builds real, tiny JPEG fixtures with Quartz -- render.py is tested against real files
it actually decodes, not mocks, because the whole point of this module is verifying pixel
geometry, which a mock cannot do. Kept out of tests/fixtures.py because it imports Quartz,
which tests/fixtures.py (imported by every synthetic-fixture test in the suite) must not
require."""

from __future__ import annotations

from pathlib import Path

import Quartz
from CoreFoundation import CFURLCreateWithFileSystemPath, kCFURLPOSIXPathStyle


def _sample_pixel(path, x, y):
    """Read one pixel's RGB via a fresh bitmap context. `y` is distance from the visual
    BOTTOM of the image (matching how callers describe sample points).

    For a context that has had a *decoded* `CGImage` drawn into it via
    `CGContextDrawImage`, buffer row 0 is empirically the visual TOP row, not the bottom
    -- opposite of the drawing coordinate system's own bottom-left-origin convention.
    Verified directly against a known-orientation fixture (`make_test_jpeg`'s own
    bottom_rgb/top_rgb halves, read with no `render_jpeg` involved) before trusting it,
    and confirmed `render_jpeg` itself introduces no additional flip on top of this
    (Tick 50's own diagnostic trail). Moved here from `tests/test_render.py` so both that
    file and `tests/test_album_export.py` share one implementation (Task 10 Step 5)."""
    url = CFURLCreateWithFileSystemPath(None, str(path), kCFURLPOSIXPathStyle, False)
    src = Quartz.CGImageSourceCreateWithURL(url, None)
    img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    w, h = Quartz.CGImageGetWidth(img), Quartz.CGImageGetHeight(img)
    cs = Quartz.CGColorSpaceCreateDeviceRGB()
    ctx = Quartz.CGBitmapContextCreate(None, w, h, 8, 0, cs, Quartz.kCGImageAlphaPremultipliedLast)
    Quartz.CGContextDrawImage(ctx, Quartz.CGRectMake(0, 0, w, h), img)
    data = Quartz.CGBitmapContextGetData(ctx)
    buf = data.as_buffer(4 * w * h) if hasattr(data, "as_buffer") else bytes(data)
    row_from_top = h - 1 - y
    offset = (row_from_top * w + x) * 4
    return tuple(buf[offset : offset + 3])


def make_test_jpeg(
    path: Path, width: int, height: int, *, top_rgb=(0, 1, 0), bottom_rgb=(0, 0, 1), orientation=None
) -> None:
    """A two-color JPEG: `top_rgb` fills the visual top half, `bottom_rgb` the bottom
    half -- lets a test assert which half survived a crop without needing a full pixel
    decoder, just a handful of sample points.

    `orientation`, when given, is written as the standard EXIF `kCGImagePropertyOrientation`
    tag (1-8) on top of the raw buffer drawn above -- i.e. `top_rgb`/`bottom_rgb` describe
    the *stored* pixels, not the corrected display image a orientation-aware reader would
    show. Lets a test build a "sideways real photo" fixture without needing one on disk."""
    cs = Quartz.CGColorSpaceCreateDeviceRGB()
    ctx = Quartz.CGBitmapContextCreate(None, width, height, 8, 0, cs, Quartz.kCGImageAlphaPremultipliedLast)
    Quartz.CGContextSetRGBFillColor(ctx, *bottom_rgb, 1.0)
    Quartz.CGContextFillRect(ctx, Quartz.CGRectMake(0, 0, width, height))
    Quartz.CGContextSetRGBFillColor(ctx, *top_rgb, 1.0)
    Quartz.CGContextFillRect(ctx, Quartz.CGRectMake(0, height / 2, width, height / 2))
    img = Quartz.CGBitmapContextCreateImage(ctx)

    url = CFURLCreateWithFileSystemPath(None, str(path), kCFURLPOSIXPathStyle, False)
    dest = Quartz.CGImageDestinationCreateWithURL(url, "public.jpeg", 1, None)
    props = {"kCGImageDestinationLossyCompressionQuality": 0.95}
    if orientation is not None:
        props[Quartz.kCGImagePropertyOrientation] = orientation
    Quartz.CGImageDestinationAddImage(dest, img, props)
    ok = Quartz.CGImageDestinationFinalize(dest)
    assert ok, f"failed to write test fixture JPEG to {path}"
