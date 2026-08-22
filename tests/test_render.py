from __future__ import annotations

import Quartz
from CoreFoundation import CFURLCreateWithFileSystemPath, kCFURLPOSIXPathStyle

from photocull.printlayout import CropRect, PageLayout, compute_crop_rect, page_layout
from photocull.render import PdfPage, build_pdf, render_jpeg
from tests.render_helpers import _sample_pixel, make_test_jpeg


def _read_size(path):
    url = CFURLCreateWithFileSystemPath(None, str(path), kCFURLPOSIXPathStyle, False)
    src = Quartz.CGImageSourceCreateWithURL(url, None)
    img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    return Quartz.CGImageGetWidth(img), Quartz.CGImageGetHeight(img)


def test_render_jpeg_produces_the_exact_target_pixel_size(tmp_path):
    src = tmp_path / "src.jpeg"
    make_test_jpeg(src, 2000, 1500)
    crop = compute_crop_rect(2000, 1500, 1800, 1200)
    dest = tmp_path / "out.jpeg"
    render_jpeg(src, dest, crop, target_w=1800, target_h=1200)
    assert dest.exists()
    assert _read_size(dest) == (1800, 1200)


def test_render_jpeg_full_frame_crop_keeps_both_colors(tmp_path):
    src = tmp_path / "src.jpeg"
    make_test_jpeg(src, 1800, 1200, top_rgb=(0, 1, 0), bottom_rgb=(0, 0, 1))
    crop = compute_crop_rect(1800, 1200, 1800, 1200)  # matching ratio -> full frame
    dest = tmp_path / "out.jpeg"
    render_jpeg(src, dest, crop, target_w=1800, target_h=1200)
    top_px = _sample_pixel(dest, 900, 1100)     # near the visual top
    bottom_px = _sample_pixel(dest, 900, 100)   # near the visual bottom
    assert top_px[1] > top_px[2]      # greener than blue near the top
    assert bottom_px[2] > bottom_px[1]  # bluer than green near the bottom


def test_render_jpeg_honors_exif_orientation_so_a_sideways_photo_comes_out_upright(tmp_path):
    """Regression: `CGImageSourceCreateImageAtIndex` ignores `kCGImagePropertyOrientation`,
    so a real phone photo (raw sensor buffer plus a "rotate for display" tag -- EXIF
    orientation 6/8 are the common cases) rendered sideways in every crop/PDF this app
    produced, while every other viewer (Photos, Preview, this app's own browser UI, which
    all decode through APIs that honor the tag) showed it upright. Found live at Task 11:
    all three photos in the first-ever real album export came out rotated 90 degrees.

    Orientation 3 (180 degrees) is used here rather than 6/8 because it needs no
    width/height reasoning -- render_jpeg always outputs exactly target_w x target_h
    regardless of decode orientation, so a dimension check can't tell a correct decode
    from an incorrect one. Only the *pixel content* can, and a 180-degree tag swaps
    which half is on top without touching either dimension."""
    src = tmp_path / "sideways.jpeg"
    # Raw stored buffer: green top half, blue bottom half -- but tagged orientation=3,
    # so the correctly-oriented (displayed) image is the *rotated* version: blue top,
    # green bottom.
    make_test_jpeg(src, 1800, 1200, top_rgb=(0, 1, 0), bottom_rgb=(0, 0, 1), orientation=3)
    crop = compute_crop_rect(1800, 1200, 1800, 1200)  # matching ratio -> full frame
    dest = tmp_path / "out.jpeg"
    render_jpeg(src, dest, crop, target_w=1800, target_h=1200)
    top_px = _sample_pixel(dest, 900, 1100)     # near the visual top
    bottom_px = _sample_pixel(dest, 900, 100)   # near the visual bottom
    assert top_px[2] > top_px[1]        # bluer than green near the top (rotated)
    assert bottom_px[1] > bottom_px[2]  # greener than blue near the bottom (rotated)


def test_build_pdf_produces_one_page_per_input_and_the_expected_page_size(tmp_path):
    src = tmp_path / "src.jpeg"
    make_test_jpeg(src, 1800, 1200)
    layout = page_layout("4x6", bleed_mm=0.0, crop_marks=False)
    pages = [PdfPage(image_path=src, layout=layout), PdfPage(image_path=src, layout=layout)]
    dest = tmp_path / "out.pdf"
    build_pdf(pages, dest)
    assert dest.exists()

    url = CFURLCreateWithFileSystemPath(None, str(dest), kCFURLPOSIXPathStyle, False)
    doc = Quartz.CGPDFDocumentCreateWithURL(url)
    assert Quartz.CGPDFDocumentGetNumberOfPages(doc) == 2
    page1 = Quartz.CGPDFDocumentGetPage(doc, 1)
    box = Quartz.CGPDFPageGetBoxRect(page1, Quartz.kCGPDFMediaBox)
    assert (box.size.width, box.size.height) == (1800.0, 1200.0)


def test_build_pdf_with_bleed_and_marks_uses_the_layouts_own_page_size(tmp_path):
    src = tmp_path / "src.jpeg"
    make_test_jpeg(src, 1870, 1270)  # already bled-size, as render_jpeg would produce
    layout = page_layout("4x6", bleed_mm=3.0, crop_marks=True)
    dest = tmp_path / "out.pdf"
    build_pdf([PdfPage(image_path=src, layout=layout)], dest)

    url = CFURLCreateWithFileSystemPath(None, str(dest), kCFURLPOSIXPathStyle, False)
    doc = Quartz.CGPDFDocumentCreateWithURL(url)
    page1 = Quartz.CGPDFDocumentGetPage(doc, 1)
    box = Quartz.CGPDFPageGetBoxRect(page1, Quartz.kCGPDFMediaBox)
    assert (box.size.width, box.size.height) == (float(layout.page_w), float(layout.page_h))
