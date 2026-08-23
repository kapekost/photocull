"""The append-only decision log: every review decision, with the sub-scores behind it.

This is the only artifact this project produces that cannot be recomputed. Photos,
clusters, scores and contact sheets all fall out of the library plus a config file; a
week of the owner's judgement does not. The design follows from that one fact.

**Every row is written once and never rewritten.** Correcting a cluster appends a new
batch; an undo appends a row of its own. `state()` resolves a cluster to the newest
batch that has not been undone, so the current answer is a *query*, never a stored
mutable field. That matters for retraining the scorer from review overrides later: an
override is only a useful training signal if the thing it overrode is still on disk
next to it.

**It is a second SQLite file, never the analysis cache.** `AnalysisCache._migrate`
discards its tables whenever `PRAGMA user_version` differs from `ANALYSIS_VERSION` —
right for a disposable cache, and the exact mechanism that would wipe days of review
work if decisions lived there. That version has already been bumped once elsewhere in
this project, so this is a demonstrated hazard rather than a hypothetical one. Opening
this file at an unfamiliar version therefore reports the version and touches nothing.

**Append-only is enforced by SQLite itself, not by this module's good manners.** The
connection installs an authorizer that permits only the five actions this module was
measured to need — reads, inserts and transactions — and refuses everything else, so a
later caller holding the handle cannot rewrite history with otherwise-valid SQL. One
hole in that guard was found by running it rather than by reading about it, and is
covered by a static gate in `tests/test_guardrails.py` instead; see `_ALLOWED_ACTIONS`.

**Sub-scores are a snapshot, not a foreign key.** Re-tuning the weights is the entire
point of collecting them; a join to live scores would silently rewrite what the owner
was looking at when they decided, destroying the evidence the tuning is fitted to.

Keys are `photo_uuid` and the content-addressed `cluster_key`, never a cluster index or
ordinal — the same library can produce a different cluster count after any config
change, so any ordinal is meaningless one config change later. No derivative path is
stored: Photos evicts and regenerates derivatives, so a stored path would go stale the
same way a cached analysis path would.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

#: Bumped when this schema changes shape. Unlike `ANALYSIS_VERSION`, a mismatch here
#: NEVER discards anything — see `_migrate`. A file stamped by a newer version is opened
#: read-as-is and its version reported, because refusing to open would make an upgrade a
#: one-way door standing between the owner and their own irreplaceable data.
DECISIONS_VERSION = 1

#: The three states a photo can be in within a decided cluster. `unset` is a real,
#: recordable answer, not a missing one: an ambiguous cluster proposes nothing, so the
#: owner deferring on one take while deciding its siblings is a normal outcome, not a
#: gap to fill in later.
MARKS = frozenset({"keep", "cull", "unset"})

#: The complete set of things this connection may do, measured by running the module's
#: own statements under a logging authorizer rather than guessed. Everything else is
#: refused, including operations that do not exist yet.
#:
#: **Default-deny, not a list of forbidden verbs**, for two reasons. It is stronger: a
#: deny-list only stops what its author thought of, while this refuses ATTACH, VACUUM
#: and anything a future SQLite adds, on a file whose whole value is that it is never
#: rewritten. And it is honest with the guardrail suite — a module that named the
#: destructive constants in order to forbid them would trip the very gate in
#: `tests/test_guardrails.py` that looks for those names in executable code, teaching a
#: future reader that gate can be argued with. It cannot.
#:
#: One limit, found by testing rather than by reading SQLite's docs:
#: SQLite resolves `INSERT OR REPLACE` *below* the authorizer, so a conflicting REPLACE
#: silently clobbers a row while every action code still reads as an ordinary insert.
#: The static SQL gate in `tests/test_guardrails.py` is what covers that, which is why
#: the append-only promise deliberately rests on two independent mechanisms.
_ALLOWED_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_TRANSACTION,
        sqlite3.SQLITE_FUNCTION,
    }
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decision_batches (
    batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    cluster_key TEXT NOT NULL,
    config_digest TEXT NOT NULL,
    decided_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decision_marks (
    batch_id INTEGER NOT NULL,
    photo_uuid TEXT NOT NULL,
    mod_date TEXT NOT NULL,
    mark TEXT NOT NULL,
    proposed TEXT NOT NULL,
    favorite INTEGER NOT NULL,
    total REAL,
    sub_scores TEXT NOT NULL,
    PRIMARY KEY (batch_id, photo_uuid)
);
-- The settings each batch was decided under, as JSON. A separate table rather than a
-- column on `decision_batches`, and that is measured rather than stylistic: against a
-- file written by an earlier version, `CREATE TABLE IF NOT EXISTS` adds the table, while
-- an added column is simply absent and every later read fails with `no such column`.
-- A batch predating this table has no row here, which reads as "not recorded".
CREATE TABLE IF NOT EXISTS decision_configs (
    batch_id INTEGER PRIMARY KEY,
    settings TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS undos (
    undo_id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    undone_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS decision_batches_by_cluster
    ON decision_batches (cluster_key, batch_id);
"""


class DecisionError(ValueError):
    """A batch was refused before anything was written.

    Loud rather than lenient: a decision quietly stored with a photo missing reads as a
    complete decision in which the absent takes were never shown."""


def mod_key(mod_date: datetime | None) -> str:
    """`None` -> `''`, matching `cache._key` exactly.

    Deliberately duplicated rather than imported, so this module has no dependency on a
    private name in the analysis cache — with a test pinning the two implementations
    together. The reconcile step compares a stored `mod_date` against a freshly scanned
    one to decide whether a cluster went stale; if the two modules ever disagreed about
    the sentinel, every unedited photo in the library would read as changed and the
    entire review would be requeued."""
    return mod_date.isoformat() if mod_date is not None else ""


def default_db_path() -> Path:
    """`~/.local/state/photocull/decisions.db`.

    Deliberately not under `out/`, which is git-ignored disposable output a user may
    reasonably clear out without a second thought, and not beside the analysis cache,
    which is rebuilt on a version bump. XDG's state directory is the one location on
    this machine that means "small, precious, survives a clean-up"."""
    return Path.home() / ".local" / "state" / "photocull" / "decisions.db"


@dataclass(frozen=True)
class Mark:
    """One photo's verdict inside one submission, with the evidence the owner saw.

    `proposed` is the scorer's opening position, kept alongside the owner's answer so an
    override is legible without re-deriving what the scorer would have said under
    today's weights — which is precisely what re-tuning changes."""

    photo_uuid: str
    mark: str
    mod_date: datetime | None = None
    proposed: str = "unset"
    favorite: bool = False
    total: float | None = None
    sub_scores: Mapping[str, float | None] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    """One submitted batch: a cluster, the marks in it, and when they were made."""

    batch_id: int
    session_id: str
    cluster_key: str
    config_digest: str
    decided_at: datetime
    marks: tuple[Mark, ...]
    undone: bool = False
    #: The settings this batch was decided under. `None` means "not recorded" — a batch
    #: written before this snapshot existed — which is distinct from "recorded as empty"
    #: and is what lets the reconcile step say it cannot name the field that moved
    #: instead of implying nothing moved.
    settings: Mapping[str, Any] | None = None

    def marks_by_uuid(self) -> dict[str, str]:
        """Just the verdicts, for callers that do not need the evidence."""
        return {mark.photo_uuid: mark.mark for mark in self.marks}


def _validate(session_id: str, cluster_key: str, marks: Sequence[Mark]) -> None:
    """Everything checked before the first INSERT, so a bad batch cannot half-land."""
    if not session_id:
        raise DecisionError("session_id must not be empty")
    if not cluster_key:
        raise DecisionError("cluster_key must not be empty")
    if not marks:
        raise DecisionError("a decision with no marks records nothing")

    seen: set[str] = set()
    for mark in marks:
        if not mark.photo_uuid:
            raise DecisionError("photo_uuid must not be empty")
        if mark.mark not in MARKS:
            raise DecisionError(
                f"unknown mark {mark.mark!r} for {mark.photo_uuid}; "
                f"expected one of {sorted(MARKS)}"
            )
        if mark.photo_uuid in seen:
            raise DecisionError(
                f"{mark.photo_uuid} is marked twice in one batch; a submission must "
                "carry exactly one verdict per photo"
            )
        seen.add(mark.photo_uuid)


class DecisionLog:
    """An append-only SQLite log of review decisions.

    Read `state(cluster_key)` for the current answer, `history(cluster_key)` for the
    whole trail including superseded and undone batches."""

    def __init__(self, db_path: str | Path | None = None):
        path = Path(db_path) if db_path is not None else default_db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # `check_same_thread=False` because the log is opened by the launcher and then
        # used from whichever thread the server runs its handlers on — uvicorn's event
        # loop thread in production, a portal thread under `TestClient`. Python's
        # default guard would raise `ProgrammingError` on the first request rather than
        # at open time, which is a failure that only appears once a browser is attached.
        # Safe here, and measured rather than assumed: `sqlite3.threadsafety` is 3 on
        # this interpreter (SQLITE_THREADSAFE=1, serialized), so SQLite itself
        # serialises access to the connection. Nothing in this app opens a second
        # writer — one process, one log — so the guard was protecting against a
        # situation that cannot arise while blocking the one that does.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self.file_version = self._migrate()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        # Installed last: creating the schema and stamping the version are the only
        # legitimate non-INSERT writes this file ever sees.
        self._conn.set_authorizer(self._authorizer)

    @staticmethod
    def _authorizer(action: int, arg1, arg2, db_name, trigger) -> int:
        return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY

    def _migrate(self) -> int:
        """Stamp a fresh file; leave an existing one exactly as it is.

        The deliberate opposite of `AnalysisCache._migrate`, which drops its tables on a
        version mismatch. Here a mismatch is *reported* — `CREATE TABLE IF NOT EXISTS` is
        a no-op against a file that already has these tables, and a file written by a
        newer version is never restamped downward, so a downgrade cannot make a
        newer-format log look like a current one."""
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            self._conn.execute(f"PRAGMA user_version = {DECISIONS_VERSION}")
            self._conn.commit()
            return DECISIONS_VERSION
        return version

    @property
    def connection(self) -> sqlite3.Connection:
        """The guarded handle. Exposed so tests can prove the authorizer is live."""
        return self._conn

    # --- writing ------------------------------------------------------------------

    def record(
        self,
        *,
        session_id: str,
        cluster_key: str,
        config_digest: str,
        marks: Sequence[Mark],
        settings: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> int:
        """Append one decision and return its `batch_id`.

        Superseding an earlier decision on the same cluster is an ordinary append: the
        earlier batch stays on disk in full.

        `settings` is the full config snapshot behind `config_digest`. Pass both from one
        place — `session.session_settings` and `session.config_digest` are defined in
        terms of each other for that reason, and `reconcile` re-digests the snapshot and
        refuses to proceed if the two disagree."""
        _validate(session_id, cluster_key, marks)
        decided_at = (now or datetime.now()).isoformat()

        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO decision_batches "
                "(session_id, cluster_key, config_digest, decided_at) VALUES (?, ?, ?, ?)",
                (session_id, cluster_key, config_digest, decided_at),
            )
            batch_id = cursor.lastrowid
            if settings is not None:
                self._conn.execute(
                    "INSERT INTO decision_configs (batch_id, settings) VALUES (?, ?)",
                    (batch_id, json.dumps(dict(settings), sort_keys=True, default=str)),
                )
            self._conn.executemany(
                "INSERT INTO decision_marks (batch_id, photo_uuid, mod_date, mark, "
                "proposed, favorite, total, sub_scores) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        batch_id,
                        mark.photo_uuid,
                        mod_key(mark.mod_date),
                        mark.mark,
                        mark.proposed,
                        int(mark.favorite),
                        mark.total,
                        json.dumps(dict(mark.sub_scores)),
                    )
                    for mark in marks
                ],
            )
        return int(batch_id)

    def undo(
        self,
        cluster_key: str,
        *,
        session_id: str | None = None,
        now: datetime | None = None,
    ) -> int | None:
        """Undo the newest live decision for `cluster_key`; return the batch it undid.

        The undone batch is untouched — the undo is a new row pointing at it, so undoing
        is itself part of the audit trail and can be reasoned about later. Returns `None`
        when the cluster is already undecided, without inventing an event for a decision
        that was never made."""
        current = self.state(cluster_key)
        if current is None:
            return None
        with self._conn:
            self._conn.execute(
                "INSERT INTO undos (batch_id, session_id, undone_at) VALUES (?, ?, ?)",
                (
                    current.batch_id,
                    session_id or current.session_id,
                    (now or datetime.now()).isoformat(),
                ),
            )
        return current.batch_id

    # --- reading ------------------------------------------------------------------

    def state(self, cluster_key: str) -> Decision | None:
        """The current decision for one cluster, or `None` if it is undecided.

        `None` is distinct from "decided, everything kept" — resuming a session picks up
        at the first undecided cluster, so conflating the two either re-asks a settled
        question or skips a live one.

        Newest is by `batch_id`, never by `decided_at`: two submissions can share a
        timestamp, and a clock can move backwards, whereas the id is monotonic by
        construction. Not scoped to a session either — a decision made on Monday must be
        visible to the session that resumes on Thursday, which is the whole reason this
        outlives the process that wrote it."""
        row = self._conn.execute(
            "SELECT batch_id FROM decision_batches WHERE cluster_key = ? "
            "AND batch_id NOT IN (SELECT batch_id FROM undos) "
            "ORDER BY batch_id DESC LIMIT 1",
            (cluster_key,),
        ).fetchone()
        return self._load(row[0]) if row else None

    def all_state(self) -> dict[str, Decision]:
        """Every cluster's current decision, in one pass.

        `reconcile` diffs the whole log against a fresh scan; doing that one `state()`
        call at a time is thousands of queries. Both paths share `_load`, so what they
        return cannot drift apart in shape."""
        rows = self._conn.execute(
            "SELECT cluster_key, MAX(batch_id) FROM decision_batches "
            "WHERE batch_id NOT IN (SELECT batch_id FROM undos) "
            "GROUP BY cluster_key"
        ).fetchall()
        return {cluster_key: self._load(batch_id) for cluster_key, batch_id in rows}

    def history(self, cluster_key: str) -> list[Decision]:
        """Every batch ever recorded for this cluster, newest first, undone ones flagged."""
        rows = self._conn.execute(
            "SELECT batch_id FROM decision_batches WHERE cluster_key = ? "
            "ORDER BY batch_id DESC",
            (cluster_key,),
        ).fetchall()
        return [self._load(row[0]) for row in rows]

    def _load(self, batch_id: int) -> Decision:
        batch = self._conn.execute(
            "SELECT session_id, cluster_key, config_digest, decided_at "
            "FROM decision_batches WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()
        undone = (
            self._conn.execute(
                "SELECT 1 FROM undos WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            is not None
        )
        marks = self._conn.execute(
            "SELECT photo_uuid, mod_date, mark, proposed, favorite, total, sub_scores "
            "FROM decision_marks WHERE batch_id = ? ORDER BY photo_uuid",
            (batch_id,),
        ).fetchall()
        stored_settings = self._conn.execute(
            "SELECT settings FROM decision_configs WHERE batch_id = ?", (batch_id,)
        ).fetchone()
        return Decision(
            batch_id=batch_id,
            session_id=batch[0],
            cluster_key=batch[1],
            config_digest=batch[2],
            decided_at=datetime.fromisoformat(batch[3]),
            marks=tuple(_row_to_mark(row) for row in marks),
            undone=undone,
            settings=json.loads(stored_settings[0]) if stored_settings else None,
        )

    # --- lifecycle ----------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DecisionLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _row_to_mark(row: Iterable) -> Mark:
    photo_uuid, mod_date, mark, proposed, favorite, total, sub_scores = row
    return Mark(
        photo_uuid=photo_uuid,
        mark=mark,
        # `json.loads` preserves `null` as `None`, which is the point: an absent
        # measurement (`horizon` is dropped in ~81% of this library's clusters) must not
        # come back as a measured 0.0.
        mod_date=datetime.fromisoformat(mod_date) if mod_date else None,
        proposed=proposed,
        favorite=bool(favorite),
        total=total,
        sub_scores=json.loads(sub_scores),
    )
