import struct
import zlib

import pytest

from photocull.imaging import (
    GrayBitmap,
    exposure_stats,
    image_dimensions,
    laplacian_variance,
    pixel_stats,
)


def _png(path, width, height, rows=None):
    """Smallest valid PNG with the requested dimensions, so the test exercises a real
    ImageIO header read rather than a mock. 8-bit grayscale, no filtering.

    `rows` is an optional list of `height` lists of `width` ints; omitted means all
    black, which is fine for the header-read tests but NOT for anything that reads
    pixels -- see `test_pixel_stats_decodes_real_pixels`."""

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    if rows is None:
        rows = [[0] * width for _ in range(height)]
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    raw = b"".join(b"\x00" + bytes(r) for r in rows)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return str(path)


def _gray(values, width):
    """A GrayBitmap from a flat list of 0-255 ints, stride == width."""
    height = len(values) // width
    return GrayBitmap(data=bytes(values), width=width, height=height, stride=width)


def test_reads_dimensions_from_a_real_file(tmp_path):
    path = _png(tmp_path / "a.png", 7, 3)
    assert image_dimensions(path) == (7, 3)


def test_returns_none_for_a_file_that_is_not_an_image(tmp_path):
    bad = tmp_path / "not-an-image.jpg"
    bad.write_bytes(b"definitely not a jpeg")
    assert image_dimensions(str(bad)) is None


def test_returns_none_for_a_missing_file(tmp_path):
    assert image_dimensions(str(tmp_path / "gone.jpg")) is None


def test_a_flat_image_has_zero_sharpness():
    assert laplacian_variance(_gray([128] * 25, 5)) == pytest.approx(0.0)


def test_an_edge_scores_higher_than_a_gradient():
    edge = _gray([0, 0, 255, 255, 255] * 5, 5)
    gradient = _gray([0, 60, 120, 180, 240] * 5, 5)
    assert laplacian_variance(edge) > laplacian_variance(gradient)


def test_sharpness_ignores_the_border_pixels():
    """The 4-neighbour laplacian is undefined on the border; including it would read
    out of bounds or wrap a row onto the next one."""
    inner_flat = _gray([255, 255, 255, 255, 255,
                        255, 128, 128, 128, 255,
                        255, 128, 128, 128, 255,
                        255, 128, 128, 128, 255,
                        255, 255, 255, 255, 255], 5)
    assert laplacian_variance(inner_flat) > 0
    assert laplacian_variance(_gray([128] * 25, 5)) == pytest.approx(0.0)


def test_sharpness_respects_stride_padding():
    """CGBitmapContext may pad each row; reading width*height bytes contiguously would
    shear the image and silently change every measurement.

    The rows differ from one another on purpose. An earlier version of this test used
    identical rows, and a PARTIAL stride bug -- mis-striding `row` while leaving `up`
    and `down` correct -- passed it. Verified: with varying rows, mutating any one of
    the three slices fails."""
    rows = [[(y * 37) % 256, 255, 0, (y * 11) % 256] for y in range(6)]
    padded = GrayBitmap(
        data=bytes(b for r in rows for b in (r + [99, 99, 99])),
        width=4, height=6, stride=7,
    )
    unpadded = GrayBitmap(
        data=bytes(b for r in rows for b in r), width=4, height=6, stride=4
    )
    assert laplacian_variance(padded) == pytest.approx(laplacian_variance(unpadded))
    assert laplacian_variance(unpadded) > 0


def test_exposure_stats_report_mean_and_clipping():
    stats = exposure_stats(_gray([0] * 5 + [128] * 15 + [255] * 5, 5))
    assert stats[0] == pytest.approx((0 * 5 + 128 * 15 + 255 * 5) / 25 / 255, abs=1e-6)
    assert stats[1] == pytest.approx(5 / 25)
    assert stats[2] == pytest.approx(5 / 25)


def test_exposure_stats_respect_stride_padding():
    """Same trap as sharpness: counting the padding bytes would fold them into the
    histogram and shift both the mean and the clipping fractions."""
    rows = [[0, 128, 255, 128] for _ in range(4)]
    padded = GrayBitmap(
        data=bytes(b for r in rows for b in (r + [255, 255, 255])),
        width=4, height=4, stride=7,
    )
    assert exposure_stats(padded) == pytest.approx(
        ((0 + 128 + 255 + 128) / 4 / 255.0, 1 / 4, 1 / 4)
    )


def test_pixel_stats_returns_none_for_an_unreadable_file(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    assert pixel_stats(str(bad)) is None


def test_pixel_stats_decodes_real_pixels(tmp_path):
    """Deliberately NOT an all-black image.

    A CGBitmapContext is zero-filled when created, so on an all-black source both
    `mean_luma == 0.0` and `laplacian_variance == 0.0` hold whether the decode
    succeeded or `CGContextDrawImage` never ran at all -- the test could not fail.
    Verified: a never-drawn context of the same size gives byte-identical results.

    This raster is a vertical black/white split, whose expected values are hand-traced
    rather than copied from a run: mean luma 0.5 exactly, and a laplacian variance of
    21675.0 (each interior row contributes -255 at the last black column and +255 at
    the first white one, so 12 non-zero values of 255**2 over 36 interior pixels)."""
    rows = [[0, 0, 0, 0, 255, 255, 255, 255] for _ in range(8)]
    stats = pixel_stats(_png(tmp_path / "split.png", 8, 8, rows))
    assert (stats.width, stats.height) == (8, 8)
    assert stats.mean_luma == pytest.approx(0.5)
    assert stats.laplacian_variance == pytest.approx(21675.0)
    assert stats.shadow_clipped == pytest.approx(0.5)
    assert stats.highlight_clipped == pytest.approx(0.5)
