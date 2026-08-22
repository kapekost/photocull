"""Wires Tasks 4-7 together: for a saved album's sequence, export each original
(Task 4), compute its crop and render it to the target size (Tasks 5-6), and assemble
the print-ready PDF in sequence order. This is the only place all four modules meet --
each of them stays independently testable because this module's own tests use fakes for
`ExportablePhoto`, not the real ones those modules already covered."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import Quartz
from CoreFoundation import CFURLCreateWithFileSystemPath, kCFURLPOSIXPathStyle

from .album_store import AlbumRecord
from .originals import export_original
from .printlayout import compute_crop_rect, page_layout
from .render import PdfPage, build_pdf, render_jpeg


@dataclass
class ExportReport:
    exported: int = 0
    failed: list[str] = field(default_factory=list)
    crop_not_applied: list[str] = field(default_factory=list)
    jpeg_order: list[str] = field(default_factory=list)


def _image_size(path: Path) -> tuple[int, int] | None:
    """None if `path` isn't decodable as an image -- e.g. a video's export ever reaches here
    (see `videos-excluded-from-phase-3-keeper-pool`; this is the defense-in-depth backstop for
    any other non-image asset shape, not the primary fix)."""
    url = CFURLCreateWithFileSystemPath(None, str(path), kCFURLPOSIXPathStyle, False)
    src = Quartz.CGImageSourceCreateWithURL(url, None)
    if src is None:
        return None
    img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    if img is None:
        return None
    return Quartz.CGImageGetWidth(img), Quartz.CGImageGetHeight(img)


def export_album(
    album: AlbumRecord,
    photos_by_uuid: dict[str, Any],
    out_dir: Path,
    *,
    bleed_mm: float = 0.0,
    crop_marks: bool = False,
    crop_offsets: dict[str, tuple[float, float]] | None = None,
) -> ExportReport:
    """`photos_by_uuid` maps uuid -> an `ExportablePhoto`-shaped object (the real
    osxphotos `PhotoInfo`, or a test fake). Every uuid in `album.sequence` not present in
    the map is reported in `failed` rather than raising -- one missing photo (e.g.
    deleted from Photos since the album was sequenced) must not lose the whole export.
    `crop_offsets` (uuid -> (offset_x, offset_y)) carries the UI's pan adjustment from
    `AlbumStore.get_crop_offset` (Task 10 Step 4); a uuid missing from it gets the
    centered default, same as `compute_crop_rect`'s own default."""
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_offsets = crop_offsets or {}
    layout = page_layout(album.print_size, bleed_mm=bleed_mm, crop_marks=crop_marks)

    report = ExportReport()
    pages: list[PdfPage] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for uid in album.sequence:
            photo = photos_by_uuid.get(uid)
            if photo is None:
                report.failed.append(uid)
                continue

            outcome = export_original(photo, tmp_dir, use_edited=True)
            if outcome.failed or outcome.path is None:
                report.failed.append(uid)
                continue
            if outcome.crop_not_applied:
                report.crop_not_applied.append(uid)

            source_path = Path(outcome.path)
            size = _image_size(source_path)
            if size is None:
                report.failed.append(uid)
                continue
            source_w, source_h = size
            offset_x, offset_y = crop_offsets.get(uid, (0.5, 0.5))
            crop = compute_crop_rect(source_w, source_h, layout.image_w, layout.image_h, offset_x, offset_y)

            dest_path = out_dir / f"{uid}.jpeg"
            render_jpeg(source_path, dest_path, crop, target_w=layout.image_w, target_h=layout.image_h)

            report.exported += 1
            report.jpeg_order.append(uid)
            pages.append(PdfPage(image_path=dest_path, layout=layout))

    if pages:
        build_pdf(pages, out_dir / f"{album.title}.pdf")

    return report
