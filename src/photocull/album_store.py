"""Mutable draft state for albums being sequenced -- see this plan's Task 7 note on why
this is a plain CRUD table, not the append-only pattern `decisions.py`/`writeback.py`
use. Lives at `~/.local/state/photocull/albums.db` by default, alongside (but separate
from) `decisions.db` and `writeback.db` -- three files, three different questions, same
convention `the-writeback-ledger-is-its-own-file` already established."""

from __future__ import annotations

import json
import sqlite3
import uuid as uuidlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .printlayout import PRINT_SIZES

_SCHEMA = """
CREATE TABLE IF NOT EXISTS albums (
    album_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    print_size TEXT NOT NULL,
    sequence_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    crop_offsets_json TEXT NOT NULL DEFAULT '{}'
);
"""


@dataclass(frozen=True)
class AlbumRecord:
    album_id: str
    title: str
    print_size: str
    sequence: tuple[str, ...]
    created_at: datetime
    updated_at: datetime


def _row_to_record(row) -> AlbumRecord:
    album_id, title, print_size, sequence_json, created_at, updated_at = row
    return AlbumRecord(
        album_id=album_id,
        title=title,
        print_size=print_size,
        sequence=tuple(json.loads(sequence_json)),
        created_at=datetime.fromisoformat(created_at),
        updated_at=datetime.fromisoformat(updated_at),
    )


class AlbumStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: Task 9's web API opens one AlbumStore at launch time
        # (the main/launcher thread) and then calls it from whichever thread FastAPI
        # dispatches a sync route handler on -- a portal thread under TestClient,
        # uvicorn's own thread pool in production. Same fix, same reason,
        # `photocull_review/decisions.py` and `writeback.py` already apply.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(_SCHEMA)
        # Migration: a pre-existing albums.db from Tasks 7/9's own testing predates
        # crop_offsets_json -- CREATE TABLE IF NOT EXISTS is a no-op against it, so add
        # the column by hand, guarded so it's also a no-op against an already-migrated
        # or freshly-created table.
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(albums)")}
        if "crop_offsets_json" not in columns:
            self._conn.execute(
                "ALTER TABLE albums ADD COLUMN crop_offsets_json TEXT NOT NULL DEFAULT '{}'"
            )
        self._conn.commit()

    def create(self, title: str, print_size: str, uuids: list[str]) -> AlbumRecord:
        if print_size not in PRINT_SIZES:
            raise ValueError(f"unknown print size {print_size!r}, must be one of {sorted(PRINT_SIZES)}")
        now = datetime.now(timezone.utc).isoformat()
        album_id = uuidlib.uuid4().hex
        self._conn.execute(
            "INSERT INTO albums (album_id, title, print_size, sequence_json, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (album_id, title, print_size, json.dumps(uuids), now, now),
        )
        self._conn.commit()
        return self.get(album_id)

    def get(self, album_id: str) -> AlbumRecord | None:
        row = self._conn.execute(
            "SELECT album_id, title, print_size, sequence_json, created_at, updated_at"
            " FROM albums WHERE album_id = ?",
            (album_id,),
        ).fetchone()
        return _row_to_record(row) if row else None

    def save_sequence(self, album_id: str, uuids: list[str]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            "UPDATE albums SET sequence_json = ?, updated_at = ? WHERE album_id = ?",
            (json.dumps(uuids), now, album_id),
        )
        self._conn.commit()
        if cursor.rowcount == 0:
            raise KeyError(album_id)

    def list(self) -> list[AlbumRecord]:
        rows = self._conn.execute(
            "SELECT album_id, title, print_size, sequence_json, created_at, updated_at"
            " FROM albums ORDER BY created_at DESC"
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def get_crop_offset(self, album_id: str, uid: str) -> tuple[float, float]:
        row = self._conn.execute(
            "SELECT crop_offsets_json FROM albums WHERE album_id = ?", (album_id,)
        ).fetchone()
        if row is None:
            raise KeyError(album_id)
        offset_x, offset_y = json.loads(row[0]).get(uid, (0.5, 0.5))
        return (offset_x, offset_y)

    def set_crop_offset(self, album_id: str, uid: str, offset_x: float, offset_y: float) -> None:
        row = self._conn.execute(
            "SELECT crop_offsets_json FROM albums WHERE album_id = ?", (album_id,)
        ).fetchone()
        if row is None:
            raise KeyError(album_id)
        offsets = json.loads(row[0])
        offsets[uid] = [offset_x, offset_y]
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE albums SET crop_offsets_json = ?, updated_at = ? WHERE album_id = ?",
            (json.dumps(offsets), now, album_id),
        )
        self._conn.commit()
