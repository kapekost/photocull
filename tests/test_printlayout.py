from __future__ import annotations

from photocull.printlayout import (
    PRINT_SIZES,
    bleed_px,
    compute_crop_rect,
    crop_mark_segments,
    page_layout,
)


def test_print_sizes_match_the_spec():
    assert PRINT_SIZES["4x6"].width_px == 1800
    assert PRINT_SIZES["4x6"].height_px == 1200
    assert PRINT_SIZES["5x7"].width_px == 2100
    assert PRINT_SIZES["5x7"].height_px == 1500


def test_compute_crop_rect_no_crop_needed_when_ratio_matches():
    # 1800x1200 source, 1800x1200 target -- same ratio, full frame.
    rect = compute_crop_rect(1800, 1200, 1800, 1200)
    assert (rect.x, rect.y, rect.width, rect.height) == (0, 0, 1800, 1200)


def test_compute_crop_rect_crops_height_for_a_4x3_source_into_4x6():
    # 4:3 landscape (2000x1500, ratio 1.333) into 4x6 target (ratio 1.5, wider) --
    # source is narrower than target, so width is fully kept and height is cropped.
    rect = compute_crop_rect(2000, 1500, 1800, 1200)
    assert rect.width == 2000
    assert rect.height == round(2000 / 1.5) == 1333
    assert rect.x == 0
    assert rect.y == round((1500 - 1333) / 2)  # centered by default


def test_compute_crop_rect_crops_width_for_a_16x9_source_into_4x6():
    # 16:9 (1920x1080, ratio 1.778) into 4x6 (ratio 1.5, narrower) -- source is WIDER
    # than target, so height is fully kept and width is cropped.
    rect = compute_crop_rect(1920, 1080, 1800, 1200)
    assert rect.height == 1080
    assert rect.width == round(1080 * 1.5) == 1620
    assert rect.y == 0
    assert rect.x == round((1920 - 1620) / 2)


def test_compute_crop_rect_offset_shifts_within_the_safe_range():
    rect_centered = compute_crop_rect(1920, 1080, 1800, 1200, offset_x=0.5, offset_y=0.5)
    rect_left = compute_crop_rect(1920, 1080, 1800, 1200, offset_x=0.0, offset_y=0.5)
    rect_right = compute_crop_rect(1920, 1080, 1800, 1200, offset_x=1.0, offset_y=0.5)
    assert rect_left.x == 0
    assert rect_right.x == 1920 - rect_right.width
    assert rect_left.x < rect_centered.x < rect_right.x


def test_compute_crop_rect_never_exceeds_source_bounds():
    rect = compute_crop_rect(100, 100, 1800, 1200, offset_x=1.5, offset_y=-1.0)
    assert 0 <= rect.x <= 100 - rect.width
    assert 0 <= rect.y <= 100 - rect.height


def test_bleed_px_at_300_dpi_matches_hand_computed_value():
    # 3mm at 300dpi = 3/25.4*300 = 35.433... -> rounds to 35.
    assert bleed_px(3.0, dpi=300) == 35


def test_bleed_px_zero_when_no_bleed_requested():
    assert bleed_px(0.0, dpi=300) == 0


def test_page_layout_without_bleed_or_marks_is_just_the_target_size():
    layout = page_layout("4x6", bleed_mm=0.0, crop_marks=False)
    assert (layout.page_w, layout.page_h) == (1800, 1200)
    assert (layout.image_x, layout.image_y) == (0, 0)
    assert (layout.image_w, layout.image_h) == (1800, 1200)
    assert (layout.trim_x, layout.trim_y) == (0, 0)


def test_page_layout_with_bleed_grows_the_image_and_keeps_page_equal_to_image():
    layout = page_layout("4x6", bleed_mm=3.0, crop_marks=False)
    assert layout.image_w == 1800 + 2 * 35
    assert layout.image_h == 1200 + 2 * 35
    assert (layout.page_w, layout.page_h) == (layout.image_w, layout.image_h)
    assert (layout.trim_x, layout.trim_y) == (35, 35)


def test_page_layout_with_marks_adds_margin_beyond_the_bleed():
    layout = page_layout("4x6", bleed_mm=3.0, crop_marks=True)
    margin = layout.mark_gap_px + layout.mark_len_px
    assert layout.page_w == layout.image_w + 2 * margin
    assert layout.image_x == margin
    assert layout.trim_x == margin + 35


def test_crop_mark_segments_returns_eight_segments():
    layout = page_layout("4x6", bleed_mm=3.0, crop_marks=True)
    segments = crop_mark_segments(layout)
    assert len(segments) == 8


def test_crop_mark_segments_align_to_the_trim_position():
    layout = page_layout("4x6", bleed_mm=3.0, crop_marks=True)
    segments = crop_mark_segments(layout)
    # the bottom-left corner's vertical mark is a vertical line (same x on both ends)
    # at x == trim_x, positioned entirely below the bled image (y <= image_y). The
    # top-left corner's vertical mark shares the same x (both are on the left edge),
    # so the y <= trim_y filter is needed to pick out the bottom one specifically.
    verticals_at_trim_x = [
        s for s in segments
        if s[0][0] == s[1][0] == layout.trim_x and max(s[0][1], s[1][1]) <= layout.trim_y
    ]
    assert len(verticals_at_trim_x) >= 1
    for (x1, y1), (x2, y2) in verticals_at_trim_x:
        assert max(y1, y2) <= layout.image_y


def test_crop_mark_segments_stay_outside_the_bleed_box():
    layout = page_layout("4x6", bleed_mm=3.0, crop_marks=True)
    for (x1, y1), (x2, y2) in crop_mark_segments(layout):
        inside_x = layout.image_x < x1 < layout.image_x + layout.image_w and \
                   layout.image_x < x2 < layout.image_x + layout.image_w
        inside_y = layout.image_y < y1 < layout.image_y + layout.image_h and \
                   layout.image_y < y2 < layout.image_y + layout.image_h
        assert not (inside_x and inside_y), "a crop mark must never cross into the image"
