"""Task 4: the append-only decision log — the one artifact here that cannot be recomputed.

Everything else this project produces is derivable from the Photos library and a config
file. A week of the owner's review decisions is not. That asymmetry is why this file is
paranoid in three specific directions, each of which pins a failure that has already
happened somewhere in this codebase or in the API underneath it:

1. **Nothing is ever rewritten.** `state()` resolves to the newest live batch; a
   correction is a *new* batch and an undo is a *new* row. Tests assert on row counts,
   not only on `state()`, because an implementation that UPDATEs in place returns exactly
   the same `state()` and loses the history `review-decisions-train-the-scorer` is fitted
   to. That is the sleeping-test shape this project has now found in nine consecutive
   ticks.
2. **A version bump destroys nothing.** `AnalysisCache._migrate` drops its tables when
   `PRAGMA user_version` moves, which is right for a disposable cache and catastrophic
   here. `test_a_version_bump_destroys_the_cache_but_not_the_decisions` runs the same
   operation against both and asserts they behave *oppositely*, so the reason these are
   two separate files is pinned by a test rather than by a comment.
3. **`None` sub-scores stay `None`.** `horizon` is dropped in ~81% of real clusters. A
   coercion to `0.0` would silently label the training set with a measurement nobody took.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from photocull.cache import ANALYSIS_VERSION, AnalysisCache
from photocull_review.decisions import (
    DECISIONS_VERSION,
    DecisionError,
    DecisionLog,
    Mark,
    default_db_path,
    mod_key,
)

NOW = datetime(2026, 8, 15, 9, 0, 0)


def make_mark(uuid="p1", mark="keep", **overrides):
    """A mark with the shape the review UI will actually submit."""
    fields = {
        "photo_uuid": uuid,
        "mark": mark,
        "mod_date": None,
        "proposed": "keep",
        "favorite": False,
        "total": 0.42,
        "sub_scores": {"sharpness": 0.8, "exposure": 0.61, "horizon": None},
    }
    return Mark(**(fields | overrides))


@pytest.fixture
def log(tmp_path):
    with DecisionLog(tmp_path / "decisions.db") as decisions:
        yield decisions


#: `marks=[]` is a case under test, and `marks or [...]` would have swallowed it — the
#: empty-batch test would have submitted two marks and passed for the wrong reason. An
#: explicit sentinel is the difference between a default and a silent substitution.
_DEFAULT_MARKS = object()


def record(log, key="c1", marks=_DEFAULT_MARKS, session="s1", digest="cfg0", at=NOW):
    if marks is _DEFAULT_MARKS:
        marks = [make_mark("p1", "keep"), make_mark("p2", "cull")]
    return log.record(
        session_id=session,
        cluster_key=key,
        config_digest=digest,
        marks=marks,
        now=at,
    )


def row_counts(db_path):
    """Raw row counts, read on a fresh connection so no caching can flatter them."""
    conn = sqlite3.connect(str(db_path))
    try:
        return {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("decision_batches", "decision_marks", "undos")
        }
    finally:
        conn.close()


# --- undecided is not the same as decided -----------------------------------------


def test_an_unseen_cluster_reads_as_none(log):
    """`None` means "never decided". "Decided, everything kept" is a real decision and
    must not be indistinguishable from it — Task 5 resumes at the first undecided
    cluster, so conflating them either re-asks a settled question or skips a live one."""
    assert log.state("never-seen") is None


def test_a_decision_that_keeps_everything_is_not_none(log):
    record(log, marks=[make_mark("p1", "keep"), make_mark("p2", "keep")])
    decision = log.state("c1")
    assert decision is not None
    assert {mark.mark for mark in decision.marks} == {"keep"}


# --- the append-only property ------------------------------------------------------


def test_the_latest_batch_wins_without_anything_being_rewritten(log, tmp_path):
    """The load-bearing test of this module.

    An implementation that UPDATEs the existing rows passes any assertion made only on
    `state()`. So this asserts the row count is the *sum* of both batches: the earlier
    decision must still be on disk, in full, after being superseded."""
    record(log, marks=[make_mark("p1", "keep"), make_mark("p2", "cull")])
    record(log, marks=[make_mark("p1", "cull"), make_mark("p2", "keep")], at=NOW + timedelta(minutes=5))

    assert row_counts(tmp_path / "decisions.db") == {
        "decision_batches": 2,
        "decision_marks": 4,
        "undos": 0,
    }
    assert log.state("c1").marks_by_uuid() == {"p1": "cull", "p2": "keep"}


def test_the_newest_batch_wins_even_when_the_clock_does_not_move(log):
    """Ordering comes from the batch id, never from `decided_at`.

    Timestamps are not a safe ordering key: two submissions can land in the same
    microsecond, and a clock can go backwards (NTP, DST, a laptop waking). The batch id
    is monotonic by construction, which is the only reason `state()` is well-defined."""
    record(log, marks=[make_mark("p1", "keep")], at=NOW)
    record(log, marks=[make_mark("p1", "cull")], at=NOW)
    assert log.state("c1").marks_by_uuid() == {"p1": "cull"}

    record(log, marks=[make_mark("p1", "unset")], at=NOW - timedelta(hours=3))
    assert log.state("c1").marks_by_uuid() == {"p1": "unset"}


def test_the_connection_itself_refuses_to_rewrite_a_row(log):
    """Append-only is enforced by SQLite, not only by this module's own discipline.

    Measured during the Task 4 spike: a `sqlite3` authorizer refuses anything outside
    the five actions this module needs, so a future caller reaching for the handle
    cannot quietly rewrite history even with valid SQL."""
    record(log)
    for sql in (
        "UPDATE decision_marks SET mark = 'cull'",
        "DELETE FROM decision_batches",
        "DROP TABLE undos",
        "ALTER TABLE undos ADD COLUMN sneaky TEXT",
    ):
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            log.connection.execute(sql)

    assert log.state("c1") is not None


def test_the_guard_is_default_deny_not_a_list_of_known_bad_verbs(log):
    """The property a deny-list cannot have: it refuses what nobody thought to forbid.

    None of these is a "destructive verb", and every one of them is a way to end up with
    a decision log that is not the file the owner has been writing to."""
    for sql in (
        "CREATE TABLE shadow (x TEXT)",
        "ATTACH DATABASE ':memory:' AS other",
        "CREATE TRIGGER t AFTER INSERT ON undos BEGIN SELECT 1; END",
    ):
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            log.connection.execute(sql)


def test_inserts_and_reads_still_work_under_the_authorizer(log):
    """The guard above must not be so broad that it breaks the module it protects."""
    record(log, key="c1")
    record(log, key="c2")
    assert log.state("c1") is not None and log.state("c2") is not None


# --- undo ---------------------------------------------------------------------------


def test_undo_restores_the_previous_decision_and_is_itself_a_row(log, tmp_path):
    record(log, marks=[make_mark("p1", "keep")])
    record(log, marks=[make_mark("p1", "cull")])

    undone = log.undo("c1")

    assert undone is not None
    assert log.state("c1").marks_by_uuid() == {"p1": "keep"}
    counts = row_counts(tmp_path / "decisions.db")
    assert counts["undos"] == 1, "an undo must be recorded as an event"
    assert counts["decision_batches"] == 2, "undo must not delete the batch it undoes"


def test_undo_of_the_only_decision_returns_to_undecided(log):
    record(log, marks=[make_mark("p1", "keep")])
    log.undo("c1")
    assert log.state("c1") is None


def test_a_second_undo_walks_back_a_second_batch(log):
    record(log, marks=[make_mark("p1", "keep")])
    record(log, marks=[make_mark("p1", "cull")])
    record(log, marks=[make_mark("p1", "unset")])

    log.undo("c1")
    assert log.state("c1").marks_by_uuid() == {"p1": "cull"}
    log.undo("c1")
    assert log.state("c1").marks_by_uuid() == {"p1": "keep"}
    log.undo("c1")
    assert log.state("c1") is None


def test_undoing_an_undecided_cluster_is_a_no_op(log, tmp_path):
    """There is no batch to reference, so no event is invented for one."""
    assert log.undo("never-seen") is None
    assert row_counts(tmp_path / "decisions.db")["undos"] == 0


def test_undo_is_scoped_to_its_own_cluster(log):
    record(log, key="c1", marks=[make_mark("p1", "keep")])
    record(log, key="c2", marks=[make_mark("p9", "keep")])
    log.undo("c1")
    assert log.state("c1") is None
    assert log.state("c2") is not None


# --- the sub-score snapshot ---------------------------------------------------------


def test_none_sub_scores_round_trip_as_none(log):
    """`horizon` is dropped in ~81% of this library's clusters.

    Coercing an absent measurement to `0.0` would tell the re-tuning step that the
    scorer measured a horizon and found it terrible, on four clusters in five."""
    record(log, marks=[make_mark("p1", "keep", sub_scores={"horizon": None, "sharpness": 0.0})])
    stored = log.state("c1").marks[0].sub_scores

    assert stored["horizon"] is None
    assert stored["sharpness"] == 0.0
    assert stored["sharpness"] is not None, "0.0 and None are different measurements"


def test_the_snapshot_is_a_copy_not_a_join(log):
    """Sub-scores are frozen at decision time on purpose.

    `review-decisions-train-the-scorer` re-tunes the weights; a join to live scores would
    rewrite what the owner was looking at when they decided, destroying the only evidence
    the tuning is fitted to. Re-scoring the same photo later must not move a stored row."""
    record(log, key="c1", marks=[make_mark("p1", "keep", total=0.42)])
    record(log, key="c2", marks=[make_mark("p1", "cull", total=0.99)])

    assert log.state("c1").marks[0].total == 0.42
    assert log.state("c2").marks[0].total == 0.99


def test_a_total_may_be_absent(log):
    record(log, marks=[make_mark("p1", "unset", total=None)])
    assert log.state("c1").marks[0].total is None


# --- keys and identity --------------------------------------------------------------


def test_state_is_not_keyed_on_the_session(log):
    """A decision made yesterday must be visible to the session that resumes today —
    that is the entire point of the log surviving the process that wrote it."""
    record(log, session="monday")
    assert log.state("c1", ) is not None
    later = record(log, session="thursday", marks=[make_mark("p1", "cull")])
    assert log.state("c1").batch_id == later
    assert log.state("c1").session_id == "thursday"


def test_the_mod_date_sentinel_matches_the_analysis_cache(log):
    """One convention for "this photo has never been edited", pinned across two modules.

    `cache._key` stores `None` as `''`. Task 5 compares a stored `mod_date` against a
    freshly scanned one to decide whether a cluster is stale; if the two modules disagree
    about the sentinel, every unedited photo in the library reads as changed and the whole
    review is requeued."""
    from photocull.cache import _key as cache_key

    assert mod_key(None) == cache_key(None) == ""
    edited = datetime(2026, 3, 1, 12, 30)
    assert mod_key(edited) == cache_key(edited) == edited.isoformat()


def test_a_mod_date_round_trips(log):
    edited = datetime(2026, 3, 1, 12, 30)
    record(log, marks=[make_mark("p1", "keep", mod_date=edited), make_mark("p2", "cull")])
    by_uuid = {mark.photo_uuid: mark for mark in log.state("c1").marks}
    assert by_uuid["p1"].mod_date == edited
    assert by_uuid["p2"].mod_date is None


# --- favorite is recorded, never applied --------------------------------------------


def test_favorite_is_recorded_not_applied(log):
    """`F` marks a photo favourite *for write-back* (Task 1 amended the SPEC to say so).

    Task 14 is the only module that may touch the live library, so this one records the
    intent and imports nothing that could act on it."""
    record(log, marks=[make_mark("p1", "keep", favorite=True), make_mark("p2", "cull")])
    by_uuid = {mark.photo_uuid: mark for mark in log.state("c1").marks}
    assert by_uuid["p1"].favorite is True
    assert by_uuid["p2"].favorite is False


def test_the_decision_log_cannot_reach_the_photos_library():
    """Structural, not aspirational: this module imports no write-back driver."""
    import ast
    import photocull_review.decisions as module

    source = Path(module.__file__).read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert "photoscript" not in imported
    assert "osxphotos" not in imported


# --- durability ---------------------------------------------------------------------


def test_decisions_survive_reopening(tmp_path):
    path = tmp_path / "decisions.db"
    with DecisionLog(path) as first:
        record(first, marks=[make_mark("p1", "keep"), make_mark("p2", "cull")])
        record(first, key="c2", marks=[make_mark("p9", "keep")])
        first.undo("c2")

    with DecisionLog(path) as second:
        assert second.state("c1").marks_by_uuid() == {"p1": "keep", "p2": "cull"}
        assert second.state("c2") is None, "an undo must survive a restart too"


def test_a_version_bump_destroys_the_cache_but_not_the_decisions(tmp_path):
    """The reason these are two files, asserted as behaviour rather than described.

    `AnalysisCache._migrate` drops its tables whenever `PRAGMA user_version` differs from
    `ANALYSIS_VERSION` — correct for a cache, and the exact mechanism that would erase
    days of irreplaceable review work if decisions lived there. The bump has already
    happened once (Task 7c), so this is a demonstrated hazard, not a hypothetical."""
    cache_path = tmp_path / "cache.db"
    with AnalysisCache(cache_path) as cache:
        cache.put_feature_print("p1", None, "deriv", [0.1, 0.2, 0.3])
    _stamp_version(cache_path, ANALYSIS_VERSION + 1)
    with AnalysisCache(cache_path) as cache:
        assert cache.get_feature_print("p1", None, "deriv") is None, (
            "precondition: the analysis cache is expected to discard on a version bump"
        )

    log_path = tmp_path / "decisions.db"
    with DecisionLog(log_path) as decisions:
        record(decisions, marks=[make_mark("p1", "keep")])
    _stamp_version(log_path, DECISIONS_VERSION + 1)

    with DecisionLog(log_path) as decisions:
        assert decisions.state("c1").marks_by_uuid() == {"p1": "keep"}
        assert decisions.file_version == DECISIONS_VERSION + 1, (
            "a file written by a newer version must be reported, not silently restamped"
        )


def _stamp_version(path, version):
    conn = sqlite3.connect(str(path))
    conn.execute(f"PRAGMA user_version = {version}")
    conn.commit()
    conn.close()


# --- loud validation ----------------------------------------------------------------


def test_an_unknown_mark_is_refused(log):
    with pytest.raises(DecisionError, match="maybe"):
        record(log, marks=[make_mark("p1", "maybe")])


def test_a_rejected_batch_writes_nothing_at_all(tmp_path):
    """Validation happens before the first INSERT, so a bad batch cannot half-land.

    A batch missing some of its photos is worse than a rejected one: it reads as a
    complete decision in which the absent takes were never shown."""
    path = tmp_path / "decisions.db"
    with DecisionLog(path) as log:
        with pytest.raises(DecisionError):
            record(log, marks=[make_mark("p1", "keep"), make_mark("p2", "nonsense")])
        assert log.state("c1") is None
    assert row_counts(path) == {"decision_batches": 0, "decision_marks": 0, "undos": 0}


def test_an_empty_batch_is_refused(log):
    with pytest.raises(DecisionError, match="no marks"):
        record(log, marks=[])


def test_a_duplicate_photo_in_one_batch_is_refused(log):
    """Two marks for one photo in a single submission has no defined winner, and the
    primary key would surface it as an opaque IntegrityError at insert time."""
    with pytest.raises(DecisionError, match="p1"):
        record(log, marks=[make_mark("p1", "keep"), make_mark("p1", "cull")])


@pytest.mark.parametrize("field", ["cluster_key", "session_id"])
def test_an_empty_identifier_is_refused(log, field):
    kwargs = {"key": "c1", "session": "s1"} | {
        {"cluster_key": "key", "session_id": "session"}[field]: ""
    }
    with pytest.raises(DecisionError):
        record(log, **kwargs)


# --- audit + resume support ---------------------------------------------------------


def test_history_shows_superseded_and_undone_batches(log):
    """Audit is a stated purpose of the log, so the full trail must be readable."""
    first = record(log, marks=[make_mark("p1", "keep")])
    second = record(log, marks=[make_mark("p1", "cull")])
    log.undo("c1")

    trail = log.history("c1")
    assert [entry.batch_id for entry in trail] == [second, first]
    assert [entry.undone for entry in trail] == [True, False]


def test_all_state_returns_one_live_decision_per_cluster(log):
    record(log, key="c1", marks=[make_mark("p1", "keep")])
    record(log, key="c1", marks=[make_mark("p1", "cull")])
    record(log, key="c2", marks=[make_mark("p9", "keep")])
    record(log, key="c3", marks=[make_mark("p7", "keep")])
    log.undo("c3")

    live = log.all_state()
    assert set(live) == {"c1", "c2"}, "an undone cluster is undecided, not present"
    assert live["c1"].marks_by_uuid() == {"p1": "cull"}


def test_all_state_agrees_with_state_cluster_by_cluster(log):
    """The bulk read and the single read must not be able to drift apart."""
    for index in range(6):
        record(log, key=f"c{index}", marks=[make_mark(f"p{index}", "keep")])
    record(log, key="c3", marks=[make_mark("p3", "cull")])
    log.undo("c4")

    live = log.all_state()
    for index in range(6):
        key = f"c{index}"
        single = log.state(key)
        assert (live.get(key) is None) == (single is None)
        if single is not None:
            assert live[key].batch_id == single.batch_id


def test_the_config_digest_is_stored_per_batch(log):
    """Task 5 refuses to resume when the config moved; it reads the digest from here."""
    record(log, digest="digest-a")
    assert log.state("c1").config_digest == "digest-a"


# --- the default location -----------------------------------------------------------


def test_the_default_path_is_under_local_state_not_out(monkeypatch, tmp_path):
    """Deliberately not `out/`, which is git-ignored disposable output a user may
    reasonably `rm -rf`. This is the one file the project produces that cannot be
    recomputed from the library."""
    monkeypatch.setenv("HOME", str(tmp_path))
    path = default_db_path()
    assert path == tmp_path / ".local" / "state" / "photocull" / "decisions.db"
    assert "out" not in path.parts


def test_opening_creates_the_parent_directory(tmp_path):
    path = tmp_path / "deep" / "nested" / "decisions.db"
    with DecisionLog(path) as log:
        record(log)
        assert log.state("c1") is not None
    assert path.exists()


# --- the thread the log is used from ------------------------------------------------


def test_the_log_can_be_written_and_read_from_another_thread(log):
    """The launcher opens this file; the server writes to it from its own thread.

    Python's `sqlite3` guards a connection to its creating thread by default, so the
    obvious `connect(path)` raises `ProgrammingError` on the **first request** rather
    than at open time — a failure that only appears once a browser is attached, which is
    exactly the kind this project keeps catching late. Safe to relax here and measured
    rather than assumed: `sqlite3.threadsafety == 3` on this interpreter, so SQLite
    itself serialises the connection, and one process holds one log.
    """
    import threading

    assert sqlite3.threadsafety == 3, "serialized mode is what makes the shared handle safe"

    failures: list[BaseException] = []

    def write_from_elsewhere():
        try:
            record(log, key="threaded")
            assert log.state("threaded").marks_by_uuid() == {"p1": "keep", "p2": "cull"}
        except BaseException as exc:  # noqa: BLE001 — re-raised on the main thread below
            failures.append(exc)

    worker = threading.Thread(target=write_from_elsewhere)
    worker.start()
    worker.join(timeout=10)

    assert not failures, failures[0]
    assert log.state("threaded") is not None, "the main thread must see the same file"
