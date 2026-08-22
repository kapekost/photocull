"""Pure selection and grouping logic for Phase 3. No pixels, no osxphotos, no I/O -- the
seam to the real library (which photos are in `Cull/Keepers`) lives in `photos_source.py`
(Task 2), exactly the same split Phase 0/1/2 already use between pure logic and the thin
live parts that need a real `PhotosDB`.

`keeper_pool` resolves the spec's own contradiction (`docs/SPEC.md` says "filter to
Favorites", but `keepers-are-favourited-opt-in` means Favorites is nearly empty on this
library -- see the plan's own measurements). The pool is Keepers union Favorites; "filter
to Favorites" survives as a narrowing toggle applied to a chosen group, not as the pool
itself."""

from __future__ import annotations

import math
import struct
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from .models import PhotoRecord

_EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class AlbumConfig:
    """Tunables for trip detection. Plain dataclass with CLI-flag overrides, the
    `SweepConfig` shape -- not `ClusterConfig`'s TOML system. Nothing here interacts with
    `config_digest` or invalidates a stored decision, because Phase 3 has no decision log."""

    trip_gap_hours: float = 48.0
    trip_min_size: int = 3
    gps_spread_flag_km: float = 150.0


@dataclass(frozen=True)
class TripGroup:
    title: str
    records: tuple[PhotoRecord, ...]
    wide_spread: bool = False

    def __len__(self) -> int:
        return len(self.records)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km. Pure math, no external geocoding -- see
    `trip-naming-has-no-geocoding` for why this app never resolves a place name."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def keeper_pool(
    records: Iterable[PhotoRecord], keeper_uuids: set[str]
) -> list[PhotoRecord]:
    """Every keeper-album member or favorite, deduped, input order preserved. Videos are
    excluded regardless of keeper/favorite status (`videos-excluded-from-phase-3-keeper-pool`)
    -- a video can never be part of a printed photo album, and Phase 3 is exclusively that."""
    seen: set[str] = set()
    pool = []
    for r in records:
        if r.uuid in seen or r.is_video:
            continue
        if r.uuid in keeper_uuids or r.is_favorite:
            seen.add(r.uuid)
            pool.append(r)
    return pool


def _title_for_range(start: datetime, end: datetime) -> str:
    if start.date() == end.date():
        return start.strftime("%b %-d, %Y")
    if start.year == end.year and start.month == end.month:
        return f"{start.strftime('%b %-d')} - {end.strftime('%-d, %Y')}"
    if start.year == end.year:
        return f"{start.strftime('%b %-d')} - {end.strftime('%b %-d, %Y')}"
    return f"{start.strftime('%b %-d, %Y')} - {end.strftime('%b %-d, %Y')}"


def _gps_spread_km(records: Sequence[PhotoRecord]) -> float | None:
    located = [
        (r.latitude, r.longitude) for r in records if r.latitude is not None and r.longitude is not None
    ]
    if len(located) < 2:
        return None
    max_km = 0.0
    for i in range(len(located)):
        for j in range(i + 1, len(located)):
            d = haversine_km(*located[i], *located[j])
            max_km = max(max_km, d)
    return max_km


def trip_sweep(records: Iterable[PhotoRecord], config: AlbumConfig) -> list[TripGroup]:
    """Chain consecutive-by-date records into groups whenever the gap to the previous
    member is under `config.trip_gap_hours`, transitively -- the same sweep-line shape
    Phase 0/1's `near-dup-cluster-definition` and `bucket_records` use, at a day scale
    instead of a seconds scale. Groups below `trip_min_size` are dropped (a lone photo is
    a date-range pick, not a "trip"). Default title is the formatted date range
    (`trip-naming-has-no-geocoding`); `wide_spread` flags -- never splits -- a group whose
    GPS spread exceeds the configured threshold, and is `False` when fewer than 2 members
    carry GPS (can't measure a spread from 0 or 1 points)."""
    ordered = sorted(records, key=lambda r: (r.date or datetime.min, r.uuid))
    if not ordered:
        return []

    gap = config.trip_gap_hours * 3600.0
    chains: list[list[PhotoRecord]] = [[ordered[0]]]
    for r in ordered[1:]:
        prev = chains[-1][-1]
        delta = (r.date - prev.date).total_seconds() if r.date and prev.date else gap + 1
        if delta <= gap:
            chains[-1].append(r)
        else:
            chains.append([r])

    groups = []
    for chain in chains:
        if len(chain) < config.trip_min_size:
            continue
        spread = _gps_spread_km(chain)
        groups.append(
            TripGroup(
                title=_title_for_range(chain[0].date, chain[-1].date),
                records=tuple(chain),
                wide_spread=spread is not None and spread > config.gps_spread_flag_km,
            )
        )
    return groups


def move_before(sequence: list[str], dragged: str, target: str) -> list[str]:
    """The reorder rule `app.js`'s drag-and-drop mirrors: pull `dragged` out and reinsert
    it immediately before `target`. A no-op if either id is missing or they're equal."""
    if dragged == target or dragged not in sequence or target not in sequence:
        return list(sequence)
    result = [u for u in sequence if u != dragged]
    insert_at = result.index(target)
    result.insert(insert_at, dragged)
    return result


# --- Demo data for `photocull albums serve --demo` ---------------------------------
#
# Task 10's own plan text asks for `photocull albums serve --demo` (mirroring
# `photocull review --demo`) but no step in Tasks 1-10 builds a synthetic-data path for
# `photocull_album` -- `photocull_review/demo.py`'s 24-cluster generator exists for a
# different package and importing it here would violate
# `phase-3-gets-its-own-package-not-photocull-review`. This is a small, independent
# generator instead: real PNGs on disk (stdlib `zlib`/`struct` only, no Pillow/Quartz),
# real `PhotoRecord`s, spread across two trips so the trip picker has something to
# show, with one photo whose aspect ratio exactly matches a `PRINT_SIZES` entry (no
# crop needed) and one non-square photo whose ratio matches none of them (crop
# needed) -- so Task 10 Step 9's crop-overlay check is meaningful either way.

_DEMO_BASE_DATE = datetime(2026, 3, 1, 9, 0, 0)

#: (uuid suffix, trip index, hour offset within the trip, width, height, rgb).
#: Trip 0 is 3 photos a few hours apart; trip 1 is 4 photos a week later -- both well
#: past `AlbumConfig`'s default `trip_min_size=3`, with a gap between them (7 days)
#: far past the default `trip_gap_hours=48`, so `trip_sweep` always returns two groups
#: from these defaults without the caller having to tune anything.
#: The first entry (1800x1200) is `PRINT_SIZES["4x6"]`'s exact ratio (1.5) -- no crop
#: needed. The second (1200x900, 4:3) is non-square and matches neither shipped print
#: size's ratio (1.5, 1.4) -- crop needed. The rest are ordinary rectangular photos.
_DEMO_SPEC: tuple[tuple[int, int, int, int, tuple[int, int, int]], ...] = (
    (0, 0, 1800, 1200, (208, 74, 62)),
    (0, 2, 1200, 900, (72, 178, 168)),
    (0, 4, 1600, 1200, (232, 158, 54)),
    (1, 0, 1600, 1200, (78, 142, 220)),
    (1, 2, 1600, 1200, (128, 108, 214)),
    (1, 4, 1600, 1200, (206, 96, 178)),
    (1, 6, 1600, 1200, (96, 128, 96)),
)


def _crc_chunk(tag: bytes, payload: bytes) -> bytes:
    body = tag + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))


def _write_solid_png(path: Path, *, width: int, height: int, rgb: tuple[int, int, int]) -> None:
    """A tiny, real PNG file: one solid color, one row repeated. Stdlib only -- no
    Pillow, no Quartz -- so the demo costs nothing beyond what `photocull.cli` already
    imports. Real bytes on disk are what let `images.py`'s endpoint serve something a
    browser can actually decode, rather than a stand-in the review UI would 415 on."""
    row = b"\x00" + bytes(rgb) * width  # filter byte 0 (None) + one pixel repeated
    raw = row * height
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    blob = (
        b"\x89PNG\r\n\x1a\n"
        + _crc_chunk(b"IHDR", header)
        + _crc_chunk(b"IDAT", zlib.compress(raw, 6))
        + _crc_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)


def demo_pool(root: str | Path) -> list[PhotoRecord]:
    """A small synthetic keeper pool for `photocull albums serve --demo`: real PNGs
    under `root`, spread across two trips, with both a ratio-matching and a
    crop-needing photo. No Photos library, no Full Disk Access, no Vision -- the same
    promise `photocull_review`'s own `--demo` makes for the review UI."""
    root = Path(root)
    records: list[PhotoRecord] = []
    for position, (trip, hour, width, height, rgb) in enumerate(_DEMO_SPEC):
        uuid = f"demo-album-{trip}-{position:02d}"
        path = root / f"{uuid}.png"
        _write_solid_png(path, width=width, height=height, rgb=rgb)
        records.append(
            PhotoRecord(
                uuid=uuid,
                date=_DEMO_BASE_DATE + timedelta(days=trip * 7, hours=hour),
                width=width,
                height=height,
                is_favorite=True,
                derivative_path=str(path),
                display_path=str(path),
            )
        )
    return records
