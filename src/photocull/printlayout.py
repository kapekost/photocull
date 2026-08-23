"""Pure geometry for print export: DPI/pixel targets, aspect-fill crop math, bleed and
crop-mark placement. No pixels touched here -- `render.py` is the only consumer that
opens an actual image, using this module's output as its instructions.

Every coordinate in this module is PDF/CoreGraphics page space: origin at the bottom-left,
y increasing upward. `CropRect`, by contrast, is in IMAGE space: origin at the top-left,
y increasing downward -- the convention `CGImageCreateWithImageInRect` actually uses,
verified empirically this session (not assumed from documentation; see this plan's "What
was measured" section). Do not mix the two without translating."""

from __future__ import annotations

from dataclasses import dataclass

DPI = 300

#: mm -> px helpers share one rounding rule so a page's parts always sum consistently.
def _mm_to_px(mm: float, dpi: int) -> int:
    return round(mm / 25.4 * dpi)


_MARK_LENGTH_MM = 4.0
_MARK_GAP_MM = 2.0


@dataclass(frozen=True)
class PrintSize:
    width_px: int
    height_px: int


PRINT_SIZES: dict[str, PrintSize] = {
    "4x6": PrintSize(width_px=1800, height_px=1200),
    "5x7": PrintSize(width_px=2100, height_px=1500),
}


@dataclass(frozen=True)
class CropRect:
    """Image-space (top-left origin) crop rectangle, ready for
    `CGImageCreateWithImageInRect`."""

    x: int
    y: int
    width: int
    height: int


def compute_crop_rect(
    source_w: int, source_h: int, target_w: int, target_h: int,
    offset_x: float = 0.5, offset_y: float = 0.5,
) -> CropRect:
    """Aspect-fill (cover) crop: the largest target-ratio rectangle that fits inside the
    source. When the ratios already match, this is a no-op full-frame "crop" -- there is
    no separate "does it match" branch, because the formula below degrades to the
    identity case on its own (verified by
    `test_compute_crop_rect_no_crop_needed_when_ratio_matches`). `offset_x`/`offset_y`
    (0..1, default centered) is the UI's adjustable pan within the leftover margin;
    values outside 0..1 are clamped rather than rejected, so a UI slider can't crash an
    export by rounding a hair past an edge."""
    target_ratio = target_w / target_h
    source_ratio = source_w / source_h

    if source_ratio > target_ratio:
        crop_h = source_h
        crop_w = round(crop_h * target_ratio)
    else:
        crop_w = source_w
        crop_h = round(crop_w / target_ratio)

    crop_w = min(crop_w, source_w)
    crop_h = min(crop_h, source_h)
    max_x = source_w - crop_w
    max_y = source_h - crop_h
    ox = min(1.0, max(0.0, offset_x))
    oy = min(1.0, max(0.0, offset_y))
    return CropRect(x=round(max_x * ox), y=round(max_y * oy), width=crop_w, height=crop_h)


def bleed_px(mm: float, dpi: int = DPI) -> int:
    return _mm_to_px(mm, dpi)


@dataclass(frozen=True)
class PageLayout:
    """Where everything sits on one PDF page, in page space (bottom-left origin).
    `image_*` is the bled photo's box (== the standalone JPEG's own pixel size);
    `trim_*` is the actual cut line, `bleed_px` inside the image box on every side.
    `mark_gap_px`/`mark_len_px` are populated (even when `crop_marks` is False) so
    `crop_mark_segments` never needs a second config lookup."""

    page_w: int
    page_h: int
    image_x: int
    image_y: int
    image_w: int
    image_h: int
    trim_x: int
    trim_y: int
    trim_w: int
    trim_h: int
    bleed_px: int
    mark_gap_px: int
    mark_len_px: int
    crop_marks: bool


def page_layout(size_name: str, *, bleed_mm: float = 0.0, crop_marks: bool = False, dpi: int = DPI) -> PageLayout:
    size = PRINT_SIZES[size_name]
    bleed = bleed_px(bleed_mm, dpi)
    image_w = size.width_px + 2 * bleed
    image_h = size.height_px + 2 * bleed
    mark_gap = _mm_to_px(_MARK_GAP_MM, dpi)
    mark_len = _mm_to_px(_MARK_LENGTH_MM, dpi)
    margin = (mark_gap + mark_len) if crop_marks else 0
    return PageLayout(
        page_w=image_w + 2 * margin,
        page_h=image_h + 2 * margin,
        image_x=margin,
        image_y=margin,
        image_w=image_w,
        image_h=image_h,
        trim_x=margin + bleed,
        trim_y=margin + bleed,
        trim_w=size.width_px,
        trim_h=size.height_px,
        bleed_px=bleed,
        mark_gap_px=mark_gap,
        mark_len_px=mark_len,
        crop_marks=crop_marks,
    )


Segment = tuple[tuple[float, float], tuple[float, float]]


def crop_mark_segments(layout: PageLayout) -> list[Segment]:
    """Two perpendicular segments per trim corner (8 total), each aligned to the trim
    line on one axis and offset into the margin (outside the bleed box, never inside it)
    on the other -- standard print-shop crop-mark convention. Empty when
    `layout.crop_marks` is False."""
    if not layout.crop_marks:
        return []

    gap, length = layout.mark_gap_px, layout.mark_len_px
    left, right = layout.trim_x, layout.trim_x + layout.trim_w
    bottom, top = layout.trim_y, layout.trim_y + layout.trim_h
    img_left, img_right = layout.image_x, layout.image_x + layout.image_w
    img_bottom, img_top = layout.image_y, layout.image_y + layout.image_h

    segments: list[Segment] = []
    for trim_x, outward_x in ((left, -1), (right, 1)):
        for trim_y, outward_y in ((bottom, -1), (top, 1)):
            img_edge_x = img_right if outward_x > 0 else img_left
            img_edge_y = img_top if outward_y > 0 else img_bottom
            # vertical mark: fixed x = trim_x, spans the gap+length band beyond the image's y edge
            y_start = img_edge_y + outward_y * gap
            y_end = img_edge_y + outward_y * (gap + length)
            segments.append(((trim_x, y_start), (trim_x, y_end)))
            # horizontal mark: fixed y = trim_y, spans the gap+length band beyond the image's x edge
            x_start = img_edge_x + outward_x * gap
            x_end = img_edge_x + outward_x * (gap + length)
            segments.append(((x_start, trim_y), (x_end, trim_y)))
    return segments
