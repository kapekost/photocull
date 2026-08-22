from __future__ import annotations

from pathlib import Path

from photocull.album_export import export_album
from photocull.album_store import AlbumRecord
from photocull.originals import ExportOutcome
from tests.render_helpers import _sample_pixel, make_test_jpeg


class _FakePhoto:
    def __init__(self, uuid, width, height):
        self.uuid = uuid
        self.hasadjustments = False
        self._width = width
        self._height = height

    def export(self, dest, filename=None, edited=False, use_photos_export=False):
        path = Path(dest) / f"{self.uuid}.jpeg"
        make_test_jpeg(path, self._width, self._height)
        return [str(path)]


def _album(tmp_path, uuids):
    from photocull.album_store import AlbumStore
    store = AlbumStore(tmp_path / "albums.db")
    return store.create("Test Trip", "4x6", uuids)


def test_export_album_writes_one_jpeg_per_photo_and_one_pdf(tmp_path):
    album = _album(tmp_path, ["a", "b"])
    photos = {"a": _FakePhoto("a", 1800, 1200), "b": _FakePhoto("b", 2000, 1500)}
    out_dir = tmp_path / "out"
    report = export_album(album, photos, out_dir, bleed_mm=0.0, crop_marks=False)

    assert (out_dir / "a.jpeg").exists()
    assert (out_dir / "b.jpeg").exists()
    assert (out_dir / "Test Trip.pdf").exists()
    assert report.exported == 2
    assert report.failed == []


def test_export_album_preserves_sequence_order_in_the_pdf(tmp_path):
    album = _album(tmp_path, ["b", "a"])  # sequence deliberately not alphabetical
    photos = {"a": _FakePhoto("a", 1800, 1200), "b": _FakePhoto("b", 1800, 1200)}
    report = export_album(album, photos, tmp_path / "out", bleed_mm=0.0, crop_marks=False)
    assert report.jpeg_order == ["b", "a"]


def test_export_album_reports_a_missing_photo_without_failing_the_rest(tmp_path):
    album = _album(tmp_path, ["a", "missing"])
    photos = {"a": _FakePhoto("a", 1800, 1200)}  # "missing" not in the dict
    report = export_album(album, photos, tmp_path / "out", bleed_mm=0.0, crop_marks=False)
    assert report.exported == 1
    assert report.failed == ["missing"]


def test_export_album_reports_an_unopenable_source_without_crashing_the_rest(tmp_path):
    """Defense in depth for `keeper_pool`'s video exclusion (Task albums.py fix): if any
    ExportablePhoto's own export() ever hands back a path Quartz can't decode as an image --
    a video, a corrupt file, anything -- one bad photo must not lose the whole album. Confirmed
    live (Tick 56): a real .MOV keeper's export() succeeds (returns a real file), so the crash
    was one step later in `_image_size`, which called `CGImageGetWidth` on a `None` CGImage with
    no guard at all."""
    class _UnopenablePhoto(_FakePhoto):
        def export(self, dest, filename=None, edited=False, use_photos_export=False):
            path = Path(dest) / f"{self.uuid}.mov"
            path.write_bytes(b"not an image")
            return [str(path)]

    album = _album(tmp_path, ["a", "bad"])
    photos = {"a": _FakePhoto("a", 1800, 1200), "bad": _UnopenablePhoto("bad", 0, 0)}
    report = export_album(album, photos, tmp_path / "out", bleed_mm=0.0, crop_marks=False)
    assert report.exported == 1
    assert report.failed == ["bad"]


def test_export_album_flags_crop_not_applied_photos(tmp_path):
    class _AdjustedPhoto(_FakePhoto):
        def __init__(self, uuid, width, height):
            super().__init__(uuid, width, height)
            self.hasadjustments = True

        def export(self, dest, filename=None, edited=False, use_photos_export=False):
            if edited:
                return []  # simulate no local edited render available
            return super().export(dest, filename, edited=False)

    album = _album(tmp_path, ["a"])
    photos = {"a": _AdjustedPhoto("a", 1800, 1200)}
    report = export_album(album, photos, tmp_path / "out", bleed_mm=0.0, crop_marks=False)
    assert report.crop_not_applied == ["a"]


def test_export_album_jpeg_matches_the_bled_layout_size_when_bleed_is_set(tmp_path):
    from photocull.printlayout import page_layout

    album = _album(tmp_path, ["a"])
    photos = {"a": _FakePhoto("a", 1800, 1200)}
    out_dir = tmp_path / "out"
    export_album(album, photos, out_dir, bleed_mm=3.0, crop_marks=False)

    layout = page_layout(album.print_size, bleed_mm=3.0, crop_marks=False)
    assert _image_size(out_dir / "a.jpeg") == (layout.image_w, layout.image_h)


def test_export_album_respects_a_custom_crop_offset(tmp_path):
    """A tall (portrait) source against a 4x6 (landscape) target needs a vertical crop --
    `compute_crop_rect` keeps the full width and crops height, so `offset_y` picks which
    vertical band survives. `_FakePhoto`'s export writes `make_test_jpeg`'s default
    two-color split (green top half, blue bottom half); at 1800x3600 the crop window
    (1800x1200) sits entirely inside one color band at either extreme offset (see this
    task's own evidence block for the by-hand geometry), so the sampled pixel is pure
    green at offset_y=0.0 and pure blue at offset_y=1.0 -- proof the offset actually
    reaches `compute_crop_rect`, not just that *some* crop happened."""
    album = _album(tmp_path, ["a"])
    out_dir = tmp_path / "out"

    photos = {"a": _FakePhoto("a", 1800, 3600)}
    export_album(album, photos, out_dir, crop_offsets={"a": (0.5, 0.0)})
    top_anchored = _sample_pixel(out_dir / "a.jpeg", 900, 600)

    photos = {"a": _FakePhoto("a", 1800, 3600)}
    export_album(album, photos, out_dir, crop_offsets={"a": (0.5, 1.0)})
    bottom_anchored = _sample_pixel(out_dir / "a.jpeg", 900, 600)

    assert top_anchored[1] > top_anchored[2]        # greener -- kept the visual top half
    assert bottom_anchored[2] > bottom_anchored[1]  # bluer -- kept the visual bottom half


def _image_size(path: Path) -> tuple[int, int]:
    import Quartz
    from CoreFoundation import CFURLCreateWithFileSystemPath, kCFURLPOSIXPathStyle

    url = CFURLCreateWithFileSystemPath(None, str(path), kCFURLPOSIXPathStyle, False)
    src = Quartz.CGImageSourceCreateWithURL(url, None)
    img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    return Quartz.CGImageGetWidth(img), Quartz.CGImageGetHeight(img)
