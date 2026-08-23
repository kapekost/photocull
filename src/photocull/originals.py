"""The ONE place this app calls `PhotoInfo.export()` against an original (not a
derivative). This is the one sanctioned exception to the project's rule against ever
touching an original file -- lives here and nowhere else, enforced by
`tests/test_guardrails.py::test_no_file_outside_originals_calls_export`, the same shape
of chokepoint enforcement `derivatives.py` uses for locally-cached reads.

Every call here can trigger an iCloud download and block for as long as Photos needs to
fetch the asset. Callers decide WHEN to call this (only for photos in an album actually
being exported, never for a whole pool) -- this module only decides HOW.

**`use_photos_export=True` is not optional.** osxphotos' plain `PhotoInfo.export()`
reads straight from `.path`, which is `None` whenever a photo is `ismissing` (cloud-only,
not locally cached) -- and on a real optimized library, that's most of it. Without this
flag, `.export()` on such a photo silently returns `[]` with no exception and never
launches Photos.app; `use_photos_export=True` routes the call through Photos' own
automation instead, which *does* fetch the original from iCloud. Found by testing
against a real album export, which came back `failed` for every cloud-only photo with an
empty `out/` and Photos.app never launched, until this flag was added."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ExportablePhoto(Protocol):
    uuid: str
    hasadjustments: bool

    def export(
        self,
        dest: str,
        filename: str | None = None,
        edited: bool = False,
        use_photos_export: bool = False,
    ) -> list[str]: ...


@dataclass(frozen=True)
class ExportOutcome:
    path: str | None
    used_edited: bool
    crop_not_applied: bool
    failed: bool = False


def export_original(
    photo: ExportablePhoto, dest_dir: Path, *, use_edited: bool
) -> ExportOutcome:
    """Try the edited (already-cropped) render first when the photo has Photos-applied
    adjustments and the caller asked for it; fall back to the unedited original on any
    failure rather than losing the whole photo over a render Photos can't currently
    produce. `crop_not_applied` tells the caller to flag this photo in the export
    report rather than silently shipping an uncropped frame."""
    if use_edited and photo.hasadjustments:
        try:
            paths = photo.export(str(dest_dir), edited=True, use_photos_export=True)
        except Exception:
            paths = []
        if paths:
            return ExportOutcome(path=paths[0], used_edited=True, crop_not_applied=False)
        crop_not_applied = True
    else:
        crop_not_applied = False

    paths = photo.export(str(dest_dir), use_photos_export=True)
    if not paths:
        return ExportOutcome(path=None, used_edited=False, crop_not_applied=crop_not_applied, failed=True)
    return ExportOutcome(path=paths[0], used_edited=False, crop_not_applied=crop_not_applied)
