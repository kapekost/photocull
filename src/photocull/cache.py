"""SQLite cache for expensive per-image analysis, keyed by
(uuid, mod_date, analysis_key).

`mod_date` means editing a photo in Photos.app automatically invalidates its cached
results. `analysis_key` is the *derivative the analysis was computed from*, and it is
there because that is not a detail: the same photo read through its large rather than
its small derivative produces a meaningfully different feature print, and under the
old two-part key that stale vector was served with no error and nothing downstream
could tell.

A full Vision pass over a large library is thousands of feature prints, each cheap on
its own but adding up. Caching turns a re-run into a table scan instead of a redo,
which is what makes the pipeline resumable after an interrupted run.

Vectors are stored as raw little-endian float32 rather than JSON: 768 floats per
image across the library, so the compactness matters and the round-trip is exact.

Note for the guardrail grep: this module owns a local cache file only. Nothing here
can reach the Photos library, which this app never mutates destructively under any
circumstances."""

from __future__ import annotations

import json
import sqlite3
import struct
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

#: Bump whenever anything that FEEDS this cache changes meaning -- the derivative
#: selection rule, a scoring formula, the Vision request configuration. On open, a file
#: at a different version is rebuilt from scratch rather than read. Cheap insurance
#: against a wrong answer served silently, which is worse than a slow rebuild.
#: v2: entries gained `analysis_key`; every v1 row was computed under the old
#: largest-derivative selection.
ANALYSIS_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feature_prints (
    uuid TEXT NOT NULL,
    mod_date TEXT NOT NULL,
    analysis_key TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (uuid, mod_date, analysis_key)
);
CREATE TABLE IF NOT EXISTS scores (
    uuid TEXT NOT NULL,
    mod_date TEXT NOT NULL,
    analysis_key TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (uuid, mod_date, analysis_key)
);
"""


def _key(mod_date: datetime | None) -> str:
    """Most photos are never edited, so `mod_date` is usually None; the empty string
    is its stable stand-in, and a real timestamp appearing later is exactly the
    signal that the cached analysis is stale."""
    return mod_date.isoformat() if mod_date is not None else ""


class AnalysisCache:
    def __init__(self, db_path: str | Path):
        self._conn = sqlite3.connect(str(db_path))
        self._migrate()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _migrate(self) -> None:
        """Rebuild the cache file if it was written by a different analysis version.

        A cache is disposable by definition -- the worst case is one slow run. Reading a
        stale row is the case that is NOT recoverable, because it is invisible: a
        feature print computed from a photo's large derivative measures meaningfully
        far from the one its small derivative produces, and nothing downstream can tell.

        (Guardrail note: the DROPs below discard local SQLite tables in a disposable
        cache file. Nothing in this module can reach the Photos library, which this app
        never deletes from under any circumstances.)"""
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version != ANALYSIS_VERSION:
            self._conn.executescript(
                "DROP TABLE IF EXISTS feature_prints; DROP TABLE IF EXISTS scores;"
            )
            self._conn.execute(f"PRAGMA user_version = {ANALYSIS_VERSION}")
            self._conn.commit()

    def get_feature_print(
        self, uuid: str, mod_date: datetime | None, analysis_key: str
    ) -> list[float] | None:
        row = self._conn.execute(
            "SELECT vector FROM feature_prints "
            "WHERE uuid = ? AND mod_date = ? AND analysis_key = ?",
            (uuid, _key(mod_date), analysis_key),
        ).fetchone()
        if row is None:
            return None
        blob = row[0]
        return list(struct.unpack(f"<{len(blob) // 4}f", blob))

    def put_feature_print(
        self,
        uuid: str,
        mod_date: datetime | None,
        analysis_key: str,
        vector: Sequence[float],
    ) -> None:
        blob = struct.pack(f"<{len(vector)}f", *vector)
        self._conn.execute(
            "INSERT OR REPLACE INTO feature_prints "
            "(uuid, mod_date, analysis_key, vector) VALUES (?, ?, ?, ?)",
            (uuid, _key(mod_date), analysis_key, blob),
        )
        self._conn.commit()

    def get_scores(
        self, uuid: str, mod_date: datetime | None, analysis_key: str
    ) -> dict | None:
        row = self._conn.execute(
            "SELECT payload FROM scores "
            "WHERE uuid = ? AND mod_date = ? AND analysis_key = ?",
            (uuid, _key(mod_date), analysis_key),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put_scores(
        self, uuid: str, mod_date: datetime | None, analysis_key: str, scores: dict
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO scores "
            "(uuid, mod_date, analysis_key, payload) VALUES (?, ?, ?, ?)",
            (uuid, _key(mod_date), analysis_key, json.dumps(scores)),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "AnalysisCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
