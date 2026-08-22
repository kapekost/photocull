"""The seam to osxphotos. `record_from_photoinfo` is duck-typed against whatever
object it's given (real osxphotos.PhotoInfo, or a test stand-in with the same
attributes) so its field-mapping logic is unit-testable without Photos access.
`iter_photo_records` / `open_library` are the thin, live parts that DO need a real
Photos library and Full Disk Access -- exercised only by a manual smoke test, not
by the automated suite. Metadata only: nothing here touches `.path` or `.export()`,
so a scan never triggers an iCloud download (see CLAUDE.md hard rule #2)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

from .derivatives import derivative_paths as select_derivatives
from .models import PhotoRecord


class PhotosDBLike(Protocol):
    def photos(self) -> list[Any]: ...


class AlbumInfoLike(Protocol):
    title: str
    folder_names: list[str]
    photos: list[Any]


class PhotosDBWithAlbums(Protocol):
    album_info: list[AlbumInfoLike]


def _duration_seconds(p: Any) -> float | None:
    if not p.ismovie:
        return None
    exif = p.exif_info
    if exif is None:
        return None
    return exif.duration


def _device(p: Any) -> str | None:
    exif = p.exif_info
    return getattr(exif, "camera_model", None) if exif is not None else None


def record_from_photoinfo(
    p: Any, *, burst_group: str | None = None, with_derivative: bool = False
) -> PhotoRecord:
    """Map one osxphotos.PhotoInfo-shaped object to a PhotoRecord. `date` uses
    PhotoInfo.date (the asset's current/displayed creation date, matching what
    Photos itself sorts the timeline by), not date_original (which freezes the
    EXIF import-time date even after the user edits it in Photos) -- see
    docs/orchestration/DECISIONS.md for anything that changes this.

    `burst_group` is supplied by iter_photo_records, which is the only code that
    knows a burst group's full membership.

    `with_derivative` is opt-in because resolving it costs ImageIO header reads --
    measured 2.74 ms each, 23,109 of them across the real library -- and the audit
    does not need them: Phase 0 stays pure metadata, and only the clustering pipeline
    and the review UI read pixels. Both picks come from ONE `derivative_paths` call,
    not two selector calls; asking separately would measure every file twice and add
    ~63 s to every scan for an answer already in hand."""
    analysis_path, display_path = (
        select_derivatives(p) if with_derivative else (None, None)
    )
    return PhotoRecord(
        uuid=p.uuid,
        date=p.date,
        is_video=p.ismovie,
        filesize=p.original_filesize,
        duration_seconds=_duration_seconds(p),
        is_screenshot=p.screenshot,
        is_screen_recording=p.screen_recording,
        is_selfie=p.selfie,
        is_burst=p.burst,
        is_live_photo=p.live_photo,
        is_slow_mo=p.slow_mo,
        is_time_lapse=p.time_lapse,
        is_panorama=p.panorama,
        is_raw=p.israw,
        is_favorite=p.favorite,
        is_hidden=p.hidden,
        is_shared=bool(getattr(p, "shared", False)),
        album_count=len(p.albums),
        latitude=p.latitude,
        longitude=p.longitude,
        device=_device(p),
        burst_key=burst_group,
        mod_date=p.date_modified,
        width=p.width,
        height=p.height,
        derivative_path=analysis_path,
        display_path=display_path,
        fingerprint=getattr(p, "fingerprint", None),
    )


def iter_photo_records(
    db: PhotosDBLike, *, with_derivatives: bool = False
) -> Iterator[PhotoRecord]:
    """Yield one PhotoRecord per library item, burst siblings included.

    `PhotosDB.photos()` returns only the key/selected image of each burst group and
    silently omits the rest -- against the real library that hid 72 items behind 9
    key images (14156 returned vs 14228 non-trashed rows in Photos.sqlite). Those
    siblings are precisely the near-duplicates this app exists to collapse, so we
    expand each burst group back in via `PhotoInfo.burst_photos` (a database-level
    property -- no `.path`, no iCloud download; see CLAUDE.md hard rule #2).
    Deduplicated by uuid, since a sibling may also surface as a top-level result.

    Pass `with_derivatives=True` on the clustering path. This loop is the only place
    that ever holds a sibling's PhotoInfo, so resolving each photo's derivative here
    is what keeps those 72 analyzable at all -- doing it afterwards from a uuid
    lookup against `photos()` would silently drop every one of them."""
    seen: set[str] = set()
    for p in db.photos():
        # osxphotos exposes no usable public burst-GROUP identifier: `burst_key` is
        # a bool ("is this the key image") and `burst_albums` is empty on this
        # library. Since this loop already knows the group's membership, it stamps
        # the key photo's uuid onto every member -- verified to match Photos' own
        # private burstUUID grouping. See DECISIONS.md burst-group-key-is-synthetic.
        group = p.uuid if p.burst else None
        if p.uuid not in seen:
            seen.add(p.uuid)
            yield record_from_photoinfo(
                p, burst_group=group, with_derivative=with_derivatives
            )
        if not p.burst:
            continue
        for sibling in p.burst_photos:
            # burst_photos can include siblings the user has since trashed; photos()
            # already excludes trashed items, so honour that here too.
            if sibling.uuid in seen or getattr(sibling, "intrash", False):
                continue
            seen.add(sibling.uuid)
            yield record_from_photoinfo(
                sibling, burst_group=group, with_derivative=with_derivatives
            )


def keeper_uuids(db: PhotosDBWithAlbums) -> set[str]:
    """uuids of every photo in the `Cull/Keepers` album, the same album Task 15's
    write-back populated (`writeback-shipped-as-measured`). Read-only: this never opens a
    writer, never imports `photoscript`, and costs one scan of `album_info` -- osxphotos
    already loads it as part of opening the library, so this adds no extra query."""
    for album in db.album_info:
        if album.title == "Keepers" and album.folder_names == ["Cull"]:
            return {p.uuid for p in album.photos}
    return set()


def open_library(library_path: str | None = None):
    """Open the Photos library (default: the system's last-opened library if
    library_path is None). Needs Full Disk Access -- see CLAUDE.md hard rule #3.
    Imports osxphotos lazily so importing this module never requires it."""
    import osxphotos

    return osxphotos.PhotosDB(library_path) if library_path else osxphotos.PhotosDB()
