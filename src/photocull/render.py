"""Quartz/CoreGraphics rendering: crop+resize to a target pixel size and write JPEG, and
assemble one or more images into a print-ready PDF with optional bleed margin and crop
marks. Same framework `imaging.py` already uses for the clustering pipeline's pixel
stats -- no new dependency.

`CropRect` (from `printlayout`) is in image-space, top-left origin -- passed straight to
`CGImageCreateWithImageInRect` with no y-flip, verified empirically against Quartz's own
behavior. `PageLayout` is in page-space, bottom-left origin -- PDF/CoreGraphics' own
convention, used as-is for `CGContextDrawImage`/`CGContextStrokePath`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import Quartz
from CoreFoundation import CFURLCreateWithFileSystemPath, kCFURLPOSIXPathStyle

from .printlayout import CropRect, PageLayout, crop_mark_segments


class RenderError(RuntimeError):
    pass


def _url(path: Path):
    return CFURLCreateWithFileSystemPath(None, str(path), kCFURLPOSIXPathStyle, False)


#: Passed to `CGImageSourceCreateThumbnailAtIndex` as the cap on the long edge of the
#: orientation-corrected image it hands back. Large enough that no real photo this app
#: has ever measured (4032px on the long edge) gets downsampled -- this is a ceiling, not
#: a target size; the thumbnail API only shrinks, never enlarges, when the source is
#: smaller.
_MAX_DECODE_PIXELS = 20000


def render_jpeg(source_path: Path, dest_path: Path, crop: CropRect, *, target_w: int, target_h: int) -> None:
    """Crop `source_path` to `crop` (image-space) and scale to exactly `target_w` x
    `target_h`, writing a JPEG to `dest_path`."""
    src = Quartz.CGImageSourceCreateWithURL(_url(source_path), None)
    if src is None:
        raise RenderError(f"could not open image source: {source_path}")
    # `CGImageSourceCreateImageAtIndex` hands back the RAW stored pixel buffer, ignoring
    # `kCGImagePropertyOrientation` -- for a real phone photo (EXIF orientation 6/8, the
    # sensor's native landscape buffer plus a "display rotated" tag) that means every
    # crop/PDF this app produces comes out sideways, while every other viewer (Photos,
    # Preview, a browser <img>) shows it upright because they DO honor the tag. Found by
    # testing against real exported photos, which came back rotated 90 degrees.
    # `CreateThumbnailAtIndex` with `WithTransform` bakes the rotation in during decode,
    # verified against a real EXIF-rotated photo end to end.
    img = Quartz.CGImageSourceCreateThumbnailAtIndex(
        src,
        0,
        {
            Quartz.kCGImageSourceCreateThumbnailWithTransform: True,
            Quartz.kCGImageSourceCreateThumbnailFromImageAlways: True,
            Quartz.kCGImageSourceThumbnailMaxPixelSize: _MAX_DECODE_PIXELS,
        },
    )
    if img is None:
        raise RenderError(f"could not decode image: {source_path}")

    cropped = Quartz.CGImageCreateWithImageInRect(
        img, Quartz.CGRectMake(crop.x, crop.y, crop.width, crop.height)
    )

    cs = Quartz.CGColorSpaceCreateDeviceRGB()
    ctx = Quartz.CGBitmapContextCreate(
        None, target_w, target_h, 8, 0, cs, Quartz.kCGImageAlphaPremultipliedLast
    )
    Quartz.CGContextDrawImage(ctx, Quartz.CGRectMake(0, 0, target_w, target_h), cropped)
    out_img = Quartz.CGBitmapContextCreateImage(ctx)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest = Quartz.CGImageDestinationCreateWithURL(_url(dest_path), "public.jpeg", 1, None)
    if dest is None:
        raise RenderError(f"could not open destination for writing: {dest_path}")
    Quartz.CGImageDestinationAddImage(dest, out_img, {"kCGImageDestinationLossyCompressionQuality": 0.95})
    if not Quartz.CGImageDestinationFinalize(dest):
        raise RenderError(f"failed to write JPEG: {dest_path}")


@dataclass(frozen=True)
class PdfPage:
    image_path: Path
    layout: PageLayout


def build_pdf(pages: list[PdfPage], dest_path: Path) -> None:
    """One page per `PdfPage`, each already rendered to its `layout.image_w` x
    `layout.image_h` size by `render_jpeg` beforehand -- this function only places it and
    draws crop marks, it never resizes."""
    if not pages:
        raise RenderError("build_pdf needs at least one page")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    # A PDF's overall MediaBox is set at creation; per-page sizing after that comes from
    # each CGPDFContextBeginPage's own page-info dict, which is how a photocull album
    # mixing 4x6 and 5x7 prints stays possible even though this task always passes a
    # uniform layout.
    first = pages[0].layout
    pdf_ctx = Quartz.CGPDFContextCreateWithURL(
        _url(dest_path), Quartz.CGRectMake(0, 0, first.page_w, first.page_h), None
    )
    if pdf_ctx is None:
        raise RenderError(f"could not open PDF for writing: {dest_path}")

    for page in pages:
        layout = page.layout
        page_info = {Quartz.kCGPDFContextMediaBox: Quartz.CGRectMake(0, 0, layout.page_w, layout.page_h)}
        Quartz.CGPDFContextBeginPage(pdf_ctx, page_info)

        src = Quartz.CGImageSourceCreateWithURL(_url(page.image_path), None)
        img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
        Quartz.CGContextDrawImage(
            pdf_ctx, Quartz.CGRectMake(layout.image_x, layout.image_y, layout.image_w, layout.image_h), img
        )

        if layout.crop_marks:
            Quartz.CGContextSetRGBStrokeColor(pdf_ctx, 0, 0, 0, 1)
            Quartz.CGContextSetLineWidth(pdf_ctx, 1.0)
            for (x1, y1), (x2, y2) in crop_mark_segments(layout):
                Quartz.CGContextMoveToPoint(pdf_ctx, x1, y1)
                Quartz.CGContextAddLineToPoint(pdf_ctx, x2, y2)
            Quartz.CGContextStrokePath(pdf_ctx)

        Quartz.CGPDFContextEndPage(pdf_ctx)

    Quartz.CGPDFContextClose(pdf_ctx)
