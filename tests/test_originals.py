from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from photocull.originals import export_original


class _FakePhoto:
    def __init__(self, uuid, hasadjustments=False, edited_paths=None, original_paths=None,
                 raises_on_edited=False):
        self.uuid = uuid
        self.hasadjustments = hasadjustments
        self._edited_paths = edited_paths if edited_paths is not None else []
        self._original_paths = original_paths if original_paths is not None else ["/out/orig.jpeg"]
        self._raises_on_edited = raises_on_edited

    def export(self, dest, filename=None, edited=False, use_photos_export=False, **kwargs):
        self.last_use_photos_export = use_photos_export
        self.last_kwargs = kwargs
        if edited:
            if self._raises_on_edited:
                raise RuntimeError("simulated Photos export failure")
            return self._edited_paths
        return self._original_paths


def test_export_original_uses_edited_when_photo_has_adjustments(tmp_path):
    photo = _FakePhoto("a", hasadjustments=True, edited_paths=["/out/edited.jpeg"])
    outcome = export_original(photo, tmp_path, use_edited=True)
    assert outcome.path == "/out/edited.jpeg"
    assert outcome.used_edited is True
    assert outcome.crop_not_applied is False


def test_export_original_falls_back_when_edited_export_returns_nothing(tmp_path):
    photo = _FakePhoto("a", hasadjustments=True, edited_paths=[])
    outcome = export_original(photo, tmp_path, use_edited=True)
    assert outcome.path == "/out/orig.jpeg"
    assert outcome.used_edited is False
    assert outcome.crop_not_applied is True


def test_export_original_falls_back_when_edited_export_raises(tmp_path):
    photo = _FakePhoto("a", hasadjustments=True, raises_on_edited=True)
    outcome = export_original(photo, tmp_path, use_edited=True)
    assert outcome.path == "/out/orig.jpeg"
    assert outcome.used_edited is False
    assert outcome.crop_not_applied is True


def test_export_original_skips_edited_call_when_photo_has_no_adjustments(tmp_path):
    photo = _FakePhoto("a", hasadjustments=False, raises_on_edited=True)  # would raise if called
    outcome = export_original(photo, tmp_path, use_edited=True)
    assert outcome.path == "/out/orig.jpeg"
    assert outcome.used_edited is False
    assert outcome.crop_not_applied is False  # never had a crop to apply in the first place


def test_export_original_reports_failure_when_nothing_exports(tmp_path):
    photo = _FakePhoto("a", hasadjustments=False, original_paths=[])
    outcome = export_original(photo, tmp_path, use_edited=True)
    assert outcome.path is None
    assert outcome.failed is True


def test_export_original_passes_use_photos_export_on_the_unedited_call(tmp_path):
    """Regression: plain osxphotos `.export()` reads `.path`, which is None for any
    cloud-only ("ismissing") photo -- ~98% of this library -- and silently returns []
    with no exception and no Photos.app launch. use_photos_export=True routes through
    Photos' own automation instead, which actually fetches from iCloud. Found live at
    Task 11: a real trip's first-ever export came back 0/3, empty out/, no Photos.app
    launch, until this flag was added."""
    photo = _FakePhoto("a", hasadjustments=False)
    export_original(photo, tmp_path, use_edited=True)
    assert photo.last_use_photos_export is True


def test_export_original_passes_use_photos_export_on_the_edited_call(tmp_path):
    photo = _FakePhoto("a", hasadjustments=True, edited_paths=["/out/edited.jpeg"])
    export_original(photo, tmp_path, use_edited=True)
    assert photo.last_use_photos_export is True


def test_export_original_never_requests_the_live_photo_movie_component(tmp_path):
    """Phase 4 gap-sweep (Tick 59): osxphotos' `PhotoInfo.export()` takes a
    `live_photo` kwarg (default False) that, if True, exports the paired .mov
    alongside the still and would make `paths[0]` no longer reliably be the
    still image. `export_original` must never pass `live_photo=True` -- verified
    against the real library, where 62 Live Photos are already in the Phase 3
    keeper pool (favorited or in Cull/Keepers) right now, so this is an
    already-reachable path, not a hypothetical one."""
    photo = _FakePhoto("a", hasadjustments=False)
    export_original(photo, tmp_path, use_edited=True)
    assert photo.last_kwargs.get("live_photo", False) is False
