"""Task 14: the one module that mutates the owner's real Photos library.

Every test here is written against a *plausible wrong* implementation, because each of
the four traps below produces working-looking output right up to the moment it costs
something irreversible.

1. **`photoscript` always targets the last-opened library**, never the one that was
   analysed. It drives Photos.app over AppleScript, and Photos.app has exactly one
   library open. So `photocull review --library /elsewhere` would read library B, show
   the owner library B's photographs, and write the results into library A — silently,
   with no error anywhere, into a *different personal photo library*.
2. **`create_album("Cull/Candidates")` makes a flat album whose NAME contains a slash**,
   not a folder holding an album; and the second run makes "Cull/Candidates 2". The
   folder-aware call is `make_album_folders(name, ["Cull"])`, verified idempotent by
   reading its source: it returns the existing album rather than creating a second.
3. **`Photo.keywords` is a setter that REPLACES the whole list** — verified in the
   installed 0.4.0: the setter runs `photoSetKeywords`, which assigns. A bare
   `photo.keywords = ["cull-candidate"]` destroys every keyword the owner ever applied,
   which is the one data-loss-shaped bug available in a project whose first rule is that
   nothing is lost. The read-modify-write therefore lives HERE, in the module under test,
   and not inside each writer — a protocol method called `add_keyword` would put the
   whole property inside the fake, and the test would then be checking the fake.
4. **A partially-failed run must be re-runnable.** 4,500 photos and ~9,000 AppleScript
   round trips is a long time for nothing to go wrong in.

And one trap of the module's own: **a dry run must not construct `PhotosLibrary()`**,
because its `__init__` runs `photosLibraryWaitForPhotos` — it *launches Photos and blocks
for up to 300 seconds*. That is why the library check reads the last-opened path from a
plist instead of asking the writer, and why nothing here builds a writer unless a real
run was both confirmed and launch-enabled.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from photocull.config import ClusterConfig
from photocull_review import server
from photocull_review.api import ReviewSession
from photocull_review.decisions import DecisionLog, Mark
from photocull_review.writeback import (
    CANDIDATES_PATH,
    CULL_FOLDER,
    CULL_KEYWORD,
    KEEPERS_PATH,
    Action,
    WritebackLedger,
    WritebackSettings,
    WrongLibrary,
    _album_location,
    assert_same_library,
    default_ledger_path,
    plan_writeback,
    run_writeback,
)

from ..fixtures import make_cluster

TOKEN = "n5Qk_test-token-with-plenty-of-entropy_7Xa"
BASE_URL = "http://127.0.0.1:54321"
WRITE_HEADERS = {"Sec-Fetch-Site": "same-origin"}

LIBRARY = "/Users/someone/Pictures/Photos Library.photoslibrary"
OTHER_LIBRARY = "/Volumes/Backup/Old Library.photoslibrary"

EDITED = datetime(2026, 8, 1, 12, 0, 0)


# --- a Photos library that records what was done to it -----------------------------


class FakeAlbum:
    """What `make_album_folders` hands back: an album at a path inside a folder."""

    def __init__(self, folder_path, name):
        self.folder_path = tuple(folder_path)
        self.name = name
        self.uuids: list[str] = []

    @property
    def path(self) -> str:
        return "/".join([*self.folder_path, self.name])


class FakeWriter:
    """A stand-in for Photos.app that answers like photoscript and remembers everything.

    `calls` is every *mutation* in order, which is what the dry-run test asserts is
    empty — reads are recorded separately, since reading a photo's keywords changes
    nothing and a dry run is allowed to be curious.
    """

    def __init__(self, *, keywords=None, favorites=(), fails_on=(), interrupts_on=()):
        self.albums: dict[str, FakeAlbum] = {}
        self.album_calls: list[tuple[tuple[str, ...], str]] = []
        self.keyword_map: dict[str, list[str]] = {
            uuid: list(values) for uuid, values in (keywords or {}).items()
        }
        self.favorite_map: dict[str, bool] = dict.fromkeys(favorites, True)
        self.calls: list[tuple] = []
        self.reads: list[tuple] = []
        #: `{(uuid, action_key)}` that raise instead of landing.
        self.fails_on = set(fails_on)
        #: `{(uuid, action_key)}` that raise `KeyboardInterrupt` — Ctrl-C, or Photos
        #: quitting under a run. A `BaseException`, so it leaves the loop entirely.
        self.interrupts_on = set(interrupts_on)

    def _guard(self, uuid: str, key: str) -> None:
        if (uuid, key) in self.interrupts_on:
            raise KeyboardInterrupt
        if (uuid, key) in self.fails_on:
            raise RuntimeError(f"Photos refused {key} for {uuid}")

    # --- the protocol ---------------------------------------------------------------

    def ensure_album(self, folder_path, album_name):
        self.album_calls.append((tuple(folder_path), album_name))
        album = FakeAlbum(folder_path, album_name)
        return self.albums.setdefault(album.path, album)

    def add_to_album(self, album, photo_uuid):
        self._guard(photo_uuid, f"album:{album.path}")
        self.calls.append(("add_to_album", album.path, photo_uuid))
        album.uuids.append(photo_uuid)

    def keywords(self, photo_uuid):
        self.reads.append(("keywords", photo_uuid))
        return list(self.keyword_map.get(photo_uuid, []))

    def set_keywords(self, photo_uuid, keywords):
        self._guard(photo_uuid, f"keyword:{CULL_KEYWORD}")
        self.calls.append(("set_keywords", photo_uuid, tuple(keywords)))
        self.keyword_map[photo_uuid] = list(keywords)

    def set_favorite(self, photo_uuid, value):
        self._guard(photo_uuid, "favorite")
        self.calls.append(("set_favorite", photo_uuid, value))
        self.favorite_map[photo_uuid] = value


# --- fixtures ----------------------------------------------------------------------


@pytest.fixture
def clusters():
    return [
        make_cluster(["c1", "c2", "c3"], day=0),
        make_cluster(["p1", "p2"], day=1),
    ]


@pytest.fixture
def log(tmp_path):
    with DecisionLog(tmp_path / "decisions.db") as opened:
        yield opened


@pytest.fixture
def ledger(tmp_path):
    with WritebackLedger(tmp_path / "writeback.db") as opened:
        yield opened


def _display(record):
    from photocull_review.session import DisplayInfo

    return DisplayInfo(uuid=record.uuid, local_path=record.display_path, long_side=1024)


@pytest.fixture
def review(clusters, log, tmp_path):
    return ReviewSession(
        clusters,
        log=log,
        session_id="test-session",
        config=ClusterConfig(),
        resolver=_display,
        writeback=WritebackSettings(
            analysed_library=LIBRARY,
            ledger_path=tmp_path / "writeback.db",
            target_library=lambda: LIBRARY,
        ),
    )


def keys(review):
    return [entry["cluster_key"] for entry in review.document["clusters"]]


def decided(review, key, marks, favorites=()):
    """Record a decision through the session, as the browser would."""
    return review.record(key, marks, favorites=list(favorites))


def a_plan(review):
    """The write-back plan for whatever this session's log currently says."""
    return plan_writeback(review.live_decisions(), review.live_photos())


def run(plan, **kwargs):
    kwargs.setdefault("analysed_library", LIBRARY)
    kwargs.setdefault("targeted_library", LIBRARY)
    kwargs.setdefault("session_id", "test-session")
    return run_writeback(plan, **kwargs)


# --- trap 1: the library that gets written is not the one that was read -------------


def test_writing_to_a_library_that_is_not_the_one_analysed_is_refused(review, ledger):
    """The failure this catches is silent, irreversible and lands in the wrong library.

    `photoscript` speaks to whatever Photos.app currently has open. Nothing about a
    `--library` flag reaches it, so the mismatch has no symptom of its own."""
    decided(review, keys(review)[0], {"c1": "keep", "c2": "cull", "c3": "cull"})
    plan = a_plan(review)
    writer = FakeWriter()

    with pytest.raises(WrongLibrary) as refusal:
        run(plan, writer=writer, ledger=ledger, dry_run=False, targeted_library=OTHER_LIBRARY)

    assert LIBRARY in str(refusal.value) and OTHER_LIBRARY in str(refusal.value), (
        "a refusal that does not name both libraries leaves the owner to guess which "
        "one Photos has open"
    )
    assert writer.calls == [], "the refusal must land before the first mutation"
    assert ledger.applied() == set()


def test_the_same_library_spelled_differently_is_the_same_library():
    """A trailing slash is how `get_last_library_path` and a shell completion differ."""
    assert_same_library(LIBRARY, LIBRARY + "/")
    assert_same_library(LIBRARY + "/", LIBRARY)


def test_a_library_that_cannot_be_identified_is_refused_rather_than_assumed():
    """`get_last_library_path()` returns None when Photos has never been opened.

    Read as "no mismatch found" that would turn the one guard against writing to the
    wrong library into a no-op exactly when it cannot see."""
    for analysed, targeted in ((None, LIBRARY), (LIBRARY, None), (None, None)):
        with pytest.raises(WrongLibrary):
            assert_same_library(analysed, targeted)


# --- trap 2: an album inside a folder, not an album named "Cull/Candidates" ---------


def test_albums_are_made_as_a_folder_never_as_a_name_with_a_slash_in_it(review, ledger):
    decided(review, keys(review)[0], {"c1": "keep", "c2": "cull", "c3": "cull"})
    writer = FakeWriter()
    run(a_plan(review), writer=writer, ledger=ledger, dry_run=False)

    # A keeper (`c1`) gets no album at all (`keepers-get-no-album`) -- `Cull/Candidates`
    # is the only album a write-back with no favourites pressed ever touches.
    assert sorted(writer.album_calls) == [(CULL_FOLDER, "Candidates")]
    for _folder, name in writer.album_calls:
        assert "/" not in name, (
            "a slash in the album NAME is the flat-album bug: Photos makes one album "
            "literally called 'Cull/Candidates'"
        )
    assert sorted(writer.albums) == [CANDIDATES_PATH]
    assert writer.albums[CANDIDATES_PATH].uuids == ["c2", "c3"]


def test_a_second_run_creates_no_second_album_and_repeats_no_action(review, ledger):
    decided(review, keys(review)[0], {"c1": "keep", "c2": "cull", "c3": "cull"})
    first = FakeWriter()
    run(a_plan(review), writer=first, ledger=ledger, dry_run=False)

    # A fresh writer with the same ledger: this is the second run, not the same run.
    second = FakeWriter(keywords=dict(first.keyword_map), favorites=first.favorite_map)
    report = run(a_plan(review), writer=second, ledger=ledger, dry_run=False)

    assert second.album_calls == [], (
        "an album call on a run with nothing left to do is how 'Cull/Candidates 2' "
        "arrives"
    )
    assert second.calls == []
    assert report.applied == ()
    assert len(report.already_done) == len(first.calls)


# --- trap 3: the keyword setter replaces the list ----------------------------------


def test_adding_the_keyword_preserves_the_keywords_the_owner_already_had(review, ledger):
    """The data-loss bug. `photo.keywords = [CULL_KEYWORD]` passes every other test here."""
    decided(review, keys(review)[0], {"c1": "keep", "c2": "cull", "c3": "cull"})
    writer = FakeWriter(keywords={"c2": ["Greece 2019", "family"], "c3": []})
    run(a_plan(review), writer=writer, ledger=ledger, dry_run=False)

    assert writer.keyword_map["c2"] == ["Greece 2019", "family", CULL_KEYWORD]
    assert writer.keyword_map["c3"] == [CULL_KEYWORD]
    assert ("keywords", "c2") in writer.reads, (
        "the existing keywords must be READ before they are written; a write that "
        "never reads cannot preserve them"
    )


def test_the_keyword_is_not_added_twice_to_a_photo_that_already_carries_it(review, ledger):
    decided(review, keys(review)[0], {"c1": "keep", "c2": "cull", "c3": "cull"})
    writer = FakeWriter(keywords={"c2": ["family", CULL_KEYWORD]})
    report = run(a_plan(review), writer=writer, ledger=ledger, dry_run=False)

    assert writer.keyword_map["c2"] == ["family", CULL_KEYWORD]
    assert not any(call[0] == "set_keywords" and call[1] == "c2" for call in writer.calls)
    assert Action("c2", "keyword", CULL_KEYWORD) in report.applied, (
        "the photo is in the state the write-back wanted, so the ledger must record it "
        "as done rather than retry it on every future run"
    )


def test_only_keepers_are_favourited_and_only_the_ones_the_owner_asked_for(review, ledger):
    """Opt-in, per the recommendation in STATE.md: auto-favoriting every keeper would
    make Phase 3's 'filter to Favorites' select ~1,807 photos instead of the 12 the
    owner has curated in fourteen thousand."""
    decided(review, keys(review)[0], {"c1": "keep", "c2": "keep", "c3": "cull"}, favorites=["c1"])
    writer = FakeWriter()
    run(a_plan(review), writer=writer, ledger=ledger, dry_run=False)

    assert writer.favorite_map == {"c1": True}, "c2 is a keeper the owner did not favourite"


def test_a_staged_photo_is_never_favourited(review, ledger):
    """Enforced at the API (`a-favourite-may-not-be-staged`) and again here, because
    write-back is the module that would act on it: a favourited photo inside
    `Cull/Candidates` is one the owner's own manual delete would take."""
    key = keys(review)[0]
    review.log.record(
        session_id="hand-written",
        cluster_key=key,
        config_digest=review.config_digest,
        settings=review.settings,
        marks=[
            Mark(photo_uuid="c1", mark="keep"),
            Mark(photo_uuid="c2", mark="cull", favorite=True),
            Mark(photo_uuid="c3", mark="cull"),
        ],
    )
    plan = a_plan(review)
    assert [action for action in plan.actions if action.kind == "favorite"] == []


# --- re-verification: the library kept living since the decision -------------------


def test_a_photo_edited_since_the_decision_is_skipped_and_reported(review, ledger):
    """A decision may be days old. The stored `mod_date` is what the owner was looking
    at; if the photo has moved since, the judgement was made about a different image."""
    key = keys(review)[0]
    review.log.record(
        session_id="an-older-sitting",
        cluster_key=key,
        config_digest=review.config_digest,
        settings=review.settings,
        marks=[
            Mark(photo_uuid="c1", mark="keep"),
            Mark(photo_uuid="c2", mark="cull", mod_date=EDITED),
            Mark(photo_uuid="c3", mark="cull"),
        ],
    )
    plan = a_plan(review)

    assert [skip.photo_uuid for skip in plan.skipped] == ["c2"]
    assert plan.skipped[0].reason == "edited"
    assert "c2" not in [action.photo_uuid for action in plan.actions]

    writer = FakeWriter()
    report = run(plan, writer=writer, ledger=ledger, dry_run=False)
    assert not any("c2" in call for call in writer.calls), (
        "an edited photo is skipped by the plan; the run must not reach it either"
    )
    assert report.skipped == plan.skipped


def test_a_photo_that_is_gone_from_the_library_is_skipped_and_reported():
    """Unreachable through today's `ReviewSession` and deliberately enforced anyway.

    `cluster_key` digests its sorted members, so a photo leaving the library changes its
    cluster's key and orphans the decision before it ever reaches here. That is a
    property of a hash function two modules away — exactly the reasoning Task 5 rejected
    when it moved the same invariant local. What this module promises is that nothing is
    written for a photo it cannot see, and that promise is enforced where it is made."""
    from photocull_review.decisions import Decision

    decision = Decision(
        batch_id=1,
        session_id="s",
        cluster_key="k",
        config_digest="d",
        decided_at=datetime(2026, 8, 16, 9, 0, 0),
        marks=(Mark(photo_uuid="ghost", mark="cull"), Mark(photo_uuid="here", mark="keep")),
    )
    plan = plan_writeback({"k": decision}, {"here": None})

    assert [skip.photo_uuid for skip in plan.skipped] == ["ghost"]
    assert plan.skipped[0].reason == "gone"
    # "here" is a bare keep (no favourite pressed), which generates no action at all
    # (`keepers-get-no-album`) -- `keepers` is what still shows it was processed rather
    # than silently dropped alongside "ghost".
    assert plan.keepers == ("here",)
    assert plan.actions == ()


# --- trap 4: a partially-failed run is re-runnable ---------------------------------


def test_a_failure_part_way_through_leaves_a_ledger_that_makes_the_rerun_a_noop(
    review, ledger
):
    decided(review, keys(review)[0], {"c1": "keep", "c2": "cull", "c3": "cull"})
    plan = a_plan(review)

    broken = FakeWriter(fails_on=[("c2", f"keyword:{CULL_KEYWORD}")])
    first = run(plan, writer=broken, ledger=ledger, dry_run=False)

    assert [failure.action.photo_uuid for failure in first.failed] == ["c2"]
    assert "Photos refused" in first.failed[0].error
    assert not first.ok
    assert Action("c3", "keyword", CULL_KEYWORD) in first.applied, (
        "one photo failing must not strand the rest of the run half-written"
    )

    healthy = FakeWriter(keywords={uuid: list(kw) for uuid, kw in broken.keyword_map.items()})
    second = run(plan, writer=healthy, ledger=ledger, dry_run=False)

    assert [action.photo_uuid for action in second.applied] == ["c2"]
    assert [action.kind for action in second.applied] == ["keyword"]
    assert second.ok
    assert healthy.keyword_map["c2"] == [CULL_KEYWORD]
    assert len(second.already_done) == len(first.applied)


def test_the_ledger_is_insert_only_at_the_sqlite_layer(ledger):
    """Same two-mechanism argument as the decision log: this file records what was done
    to the owner's real library, and a rewritten row means a repeated mutation."""
    ledger.record("u1", "album:Cull/Keepers", session_id="s")
    for statement in (
        "DELETE FROM writeback_ledger",
        "UPDATE writeback_ledger SET action = 'x'",
        "DROP TABLE writeback_ledger",
    ):
        with pytest.raises(sqlite3.DatabaseError):
            ledger.connection.execute(statement)
    assert ledger.applied() == {("u1", "album:Cull/Keepers")}


def test_the_ledger_is_its_own_file_and_never_the_decision_log():
    """`--decisions` relocates the decision log, and the two answer different questions:
    the log is keyed to a *config*, the ledger to a *library*. A ledger that travelled
    with `--decisions` would let a re-run repeat every action against the real library."""
    assert default_ledger_path() != Path.home() / ".local/state/photocull/decisions.db"
    assert default_ledger_path().name == "writeback.db"


# --- the dry run -------------------------------------------------------------------


def test_a_dry_run_mutates_nothing_and_still_reports_the_whole_plan(review, ledger):
    decided(review, keys(review)[0], {"c1": "keep", "c2": "cull", "c3": "cull"})
    decided(review, keys(review)[1], {"p1": "keep", "p2": "cull"}, favorites=["p1"])
    writer = FakeWriter()

    report = run(a_plan(review), writer=writer, ledger=ledger, dry_run=True)

    assert writer.calls == [] and writer.album_calls == []
    assert ledger.applied() == set(), "a dry run records nothing; there is nothing to record"
    assert report.dry_run is True
    assert report.applied == ()
    # Keepers get no album (`keepers-get-no-album`): the only album this plan ever
    # touches is `Cull/Candidates`.
    assert report.albums == (CANDIDATES_PATH,)
    assert len(report.planned) == 7, (
        "1 favourite (p1 only), 3 staged into an album, 3 keywords"
    )


def test_a_dry_run_needs_no_writer_at_all(review):
    """`PhotosLibrary()` launches Photos.app and blocks for up to 300 s in its
    constructor, so a dry run that built one would open an application to tell the owner
    what it was NOT going to do."""
    decided(review, keys(review)[0], {"c1": "keep", "c2": "cull", "c3": "cull"})
    report = run(a_plan(review), dry_run=True)
    assert len(report.planned) == 4, "c1 (keep, no favourite) plans nothing"
    assert report.applied == ()


def test_a_real_run_without_a_writer_is_a_refusal_rather_than_a_silent_dry_run(review, ledger):
    decided(review, keys(review)[0], {"c1": "keep", "c2": "cull", "c3": "cull"})
    with pytest.raises(ValueError, match="writer"):
        run(a_plan(review), ledger=ledger, dry_run=False)


# --- what the plan covers ----------------------------------------------------------


def test_the_plan_is_exactly_what_the_final_check_showed(review, ledger):
    """The gate is a digest of the staged set; the plan must be built from the same
    decisions, or the owner confirmed one set and a different one got written."""
    decided(review, keys(review)[0], {"c1": "keep", "c2": "cull", "c3": "cull"})
    decided(review, keys(review)[1], {"p1": "cull", "p2": "cull"})

    staged_on_screen = {row["photo"]["uuid"] for row in review.final_check()["staged"]}
    plan = a_plan(review)
    staged_in_plan = {
        action.photo_uuid for action in plan.actions if action.target == CANDIDATES_PATH
    }
    assert staged_in_plan == staged_on_screen == {"c2", "c3", "p1", "p2"}


def test_an_undecided_cluster_contributes_nothing(review, ledger):
    decided(review, keys(review)[0], {"c1": "keep", "c2": "cull", "c3": "cull"})
    plan = a_plan(review)
    # c1 (keep, no favourite) generates no action (`keepers-get-no-album`); `keepers`
    # is what still shows it was processed.
    assert {action.photo_uuid for action in plan.actions} == {"c2", "c3"}
    assert plan.keepers == ("c1",)


def test_a_deferred_photo_is_neither_kept_nor_staged(review, ledger):
    decided(review, keys(review)[0], {"c1": "keep", "c2": "unset", "c3": "cull"})
    plan = a_plan(review)
    assert {action.photo_uuid for action in plan.actions} == {"c3"}
    assert plan.keepers == ("c1",)


# --- the module itself -------------------------------------------------------------


def test_nothing_in_the_write_back_module_can_delete():
    """The plan's Step 1 asks for this by name, and it is the module the whole gate
    exists for. `tests/test_guardrails.py` already scans all of `src/`; this is the same
    check aimed at the one file where a destructive photoscript call would be plausible
    rather than absurd."""
    from tests.test_guardrails import destructive_names

    source = Path("src/photocull_review/writeback.py").read_text()
    assert destructive_names(source) == set()
    for forbidden in ("delete_album", "delete_folder", "remove_by_id", ".remove("):
        assert forbidden not in source


def test_the_module_imports_photoscript_only_inside_the_writer_that_needs_it():
    """Importing this module must not require the `[review]` extra's heaviest piece,
    and must never be what opens a connection to Photos.app."""
    import ast

    source = Path("src/photocull_review/writeback.py").read_text()
    tree = ast.parse(source)
    top_level = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
    }
    assert "photoscript" not in top_level
    assert "osxphotos" not in top_level


# --- over HTTP ---------------------------------------------------------------------


@pytest.fixture
def client(review):
    app = server.build_app(token=TOKEN, review=review)
    opened = TestClient(app, base_url=BASE_URL)
    assert opened.get(f"/?t={TOKEN}").status_code == 200
    return opened


def confirm(client, review):
    return client.post(
        "/api/final-check/confirm",
        json={"staged_digest": review.staged_digest()},
        headers=WRITE_HEADERS,
    )


def write_back(client, **body):
    return client.post("/api/writeback", json=body, headers=WRITE_HEADERS)


def test_write_back_is_refused_until_the_final_check_is_confirmed(client, review):
    decided(review, keys(review)[0], {"c1": "keep", "c2": "cull", "c3": "cull"})
    response = write_back(client, dry_run=True)
    assert response.status_code == 409
    assert "final check" in response.json()["detail"].lower()


def test_a_confirmed_dry_run_reports_the_plan_over_http(client, review):
    decided(review, keys(review)[0], {"c1": "keep", "c2": "cull", "c3": "cull"})
    assert confirm(client, review).status_code == 200

    body = write_back(client, dry_run=True).json()
    assert body["dry_run"] is True
    assert body["counts"]["planned"] == 4, "c1 (keep, no favourite) plans nothing"
    assert body["counts"]["applied"] == 0
    assert body["albums"] == [CANDIDATES_PATH]
    assert body["keepers"] == 1 and body["staged"] == 2


def test_a_confirmation_that_has_lapsed_refuses_the_write_back(client, review):
    """`writeback_ready()` is a comparison, so a decision recorded after the
    confirmation takes the authorisation away without anything having to reset it."""
    first, second = keys(review)
    decided(review, first, {"c1": "keep", "c2": "cull", "c3": "cull"})
    assert confirm(client, review).status_code == 200
    decided(review, second, {"p1": "keep", "p2": "cull"})

    assert write_back(client, dry_run=True).status_code == 409


# --- Task 15 Step 4: scoping a run to one cluster of an already-confirmed set ------


def test_a_dry_run_can_be_scoped_to_one_confirmed_cluster(client, review):
    """Step 4 needs a real run against one cluster first, to verify the writer by hand
    before trusting it with the rest. `cluster_keys` is how a run narrows to that
    subset — the gate above still checks the digest of everything staged."""
    first, second = keys(review)
    decided(review, first, {"c1": "keep", "c2": "cull", "c3": "cull"})
    decided(review, second, {"p1": "keep", "p2": "cull"})
    assert confirm(client, review).status_code == 200

    scoped = write_back(client, dry_run=True, cluster_keys=[first]).json()
    assert scoped["counts"]["planned"] == 4, "c1 (keep, no favourite) plans nothing"
    assert scoped["keepers"] == 1
    assert scoped["staged"] == 2

    everything = write_back(client, dry_run=True).json()
    assert everything["counts"]["planned"] == 6, "unscoped stays the whole confirmed set"


def test_a_write_back_scoped_to_an_unconfirmed_cluster_key_is_refused(client, review):
    decided(review, keys(review)[0], {"c1": "keep", "c2": "cull", "c3": "cull"})
    assert confirm(client, review).status_code == 200

    response = write_back(client, dry_run=True, cluster_keys=["not-a-real-cluster"])
    assert response.status_code == 404
    assert "not-a-real-cluster" in response.json()["detail"]


def test_a_real_run_scoped_to_one_cluster_leaves_the_rest_for_a_later_call(
    clusters, log, tmp_path
):
    """The property Step 4 through 6 depend on: a scoped real run touches only the
    photos it named, and an unscoped run afterwards picks up exactly what is left."""
    writer = FakeWriter()
    review = ReviewSession(
        clusters,
        log=log,
        session_id="allowed",
        config=ClusterConfig(),
        resolver=_display,
        writeback=WritebackSettings(
            allow_write_back=True,
            analysed_library=LIBRARY,
            ledger_path=tmp_path / "writeback.db",
            target_library=lambda: LIBRARY,
            writer=lambda: writer,
        ),
    )
    first, second = keys(review)
    review.record(first, {"c1": "keep", "c2": "cull", "c3": "cull"})
    review.record(second, {"p1": "keep", "p2": "cull"})
    review.confirm_final_check(review.staged_digest())

    one_cluster = review.write_back(dry_run=False, cluster_keys=[first])
    assert one_cluster["counts"]["applied"] == 4, "c1 (keep, no favourite) applies nothing"
    assert KEEPERS_PATH not in writer.albums, "keepers-get-no-album"
    assert writer.albums[CANDIDATES_PATH].uuids == ["c2", "c3"]
    assert "p1" not in writer.keyword_map and "p2" not in writer.keyword_map

    rest = review.write_back(dry_run=False)
    assert rest["counts"]["applied"] == 2, "only the second cluster's 2 actions are new"
    assert rest["counts"]["already_done"] == 4, "the first cluster is a no-op the 2nd time"
    assert KEEPERS_PATH not in writer.albums, "keepers-get-no-album"
    assert writer.albums[CANDIDATES_PATH].uuids == ["c2", "c3", "p2"]


def test_a_real_run_is_refused_unless_the_launch_allowed_it(client, review):
    """Two independent gates, and this is the one the owner controls from the command
    line: confirming the final check authorises a *set*, `--allow-write-back`
    authorises the *session* to touch Photos at all."""
    decided(review, keys(review)[0], {"c1": "keep", "c2": "cull", "c3": "cull"})
    assert confirm(client, review).status_code == 200

    response = write_back(client, dry_run=False)
    assert response.status_code == 403
    assert "--allow-write-back" in response.json()["detail"]


def test_the_default_is_a_dry_run_when_the_body_says_nothing(client, review):
    decided(review, keys(review)[0], {"c1": "keep", "c2": "cull", "c3": "cull"})
    assert confirm(client, review).status_code == 200
    assert write_back(client).json()["dry_run"] is True


def test_a_real_run_needs_the_same_origin_header_like_every_other_write(client, review):
    decided(review, keys(review)[0], {"c1": "keep", "c2": "cull", "c3": "cull"})
    confirm(client, review)
    assert client.post("/api/writeback", json={"dry_run": False}).status_code == 403


def test_an_allowed_session_performs_the_write_back_over_http(clusters, log, tmp_path):
    """The whole path, with the launch flag on and a fake Photos library underneath."""
    writer = FakeWriter()
    review = ReviewSession(
        clusters,
        log=log,
        session_id="allowed",
        config=ClusterConfig(),
        resolver=_display,
        writeback=WritebackSettings(
            allow_write_back=True,
            analysed_library=LIBRARY,
            ledger_path=tmp_path / "writeback.db",
            target_library=lambda: LIBRARY,
            writer=lambda: writer,
        ),
    )
    app = server.build_app(token=TOKEN, review=review)
    client = TestClient(app, base_url=BASE_URL)
    client.get(f"/?t={TOKEN}")

    key = keys(review)[0]
    review.record(key, {"c1": "keep", "c2": "cull", "c3": "cull"}, favorites=["c1"])
    assert confirm(client, review).status_code == 200

    body = write_back(client, dry_run=False).json()
    assert body["dry_run"] is False
    assert body["counts"]["applied"] == 5, "keepers-get-no-album: c1 gets only the favourite"
    assert body["ok"] is True
    assert KEEPERS_PATH not in writer.albums
    assert writer.albums[CANDIDATES_PATH].uuids == ["c2", "c3"]
    assert writer.favorite_map == {"c1": True}
    assert writer.keyword_map == {"c2": [CULL_KEYWORD], "c3": [CULL_KEYWORD]}

    # And the ledger makes it re-runnable: a second call is a no-op.
    again = write_back(client, dry_run=False).json()
    assert again["counts"]["applied"] == 0
    assert again["counts"]["already_done"] == 5
    assert len(writer.calls) == 5


def test_the_review_session_defaults_to_no_write_back_at_all(clusters, log):
    """A session built without write-back settings must refuse a real run, not fall back
    to the owner's real library."""
    review = ReviewSession(clusters, log=log, config=ClusterConfig(), resolver=_display)
    review.record(keys(review)[0], {"c1": "keep", "c2": "cull", "c3": "cull"})
    review.confirm_final_check(review.staged_digest())

    with pytest.raises(Exception) as refusal:
        review.write_back(dry_run=False)
    assert "--allow-write-back" in str(refusal.value)


def test_a_dry_run_through_the_session_never_builds_a_writer(clusters, log, tmp_path):
    """The one place the "a dry run constructs nothing" rule can actually be broken.

    `run_writeback` takes a writer it is handed; `ReviewSession.write_back` is what
    *builds* one, and its default factory is `PhotoScriptWriter` — whose constructor runs
    `photosLibraryWaitForPhotos`, launching Photos.app and blocking for up to 300 seconds.
    A dry run that built one would open an application in order to report what it was not
    going to do. Caught by the mutation pass, not by inspection."""
    def never() -> object:
        raise AssertionError("a dry run must not construct a writer")

    review = ReviewSession(
        clusters,
        log=log,
        config=ClusterConfig(),
        resolver=_display,
        writeback=WritebackSettings(
            allow_write_back=True,
            analysed_library=LIBRARY,
            ledger_path=tmp_path / "writeback.db",
            target_library=lambda: LIBRARY,
            writer=never,
        ),
    )
    review.record(keys(review)[0], {"c1": "keep", "c2": "cull", "c3": "cull"})
    review.confirm_final_check(review.staged_digest())

    assert review.write_back(dry_run=True)["counts"]["planned"] == 4


def test_a_dry_run_still_checks_which_library_photos_has_open(clusters, log, tmp_path):
    """Task 15 Step 2 confirms the write target before anything else; a dry run that
    skipped the check would report a plan for a library it could never write to."""
    review = ReviewSession(
        clusters,
        log=log,
        config=ClusterConfig(),
        resolver=_display,
        writeback=WritebackSettings(
            analysed_library=LIBRARY,
            ledger_path=tmp_path / "writeback.db",
            target_library=lambda: OTHER_LIBRARY,
        ),
    )
    review.record(keys(review)[0], {"c1": "keep", "c2": "cull", "c3": "cull"})
    review.confirm_final_check(review.staged_digest())

    with pytest.raises(WrongLibrary):
        review.write_back(dry_run=True)


def test_the_launch_flag_reaches_the_session(tmp_path, monkeypatch):
    """`photocull review --allow-write-back` is the only way the flag can be set, so the
    wiring from argv to `WritebackSettings` is what makes the gate real."""
    from photocull.cli import build_parser

    args = build_parser().parse_args(["review", "--demo", "--allow-write-back"])
    assert args.allow_write_back is True
    assert build_parser().parse_args(["review", "--demo"]).allow_write_back is False


def test_a_partial_write_leaves_every_landed_action_in_the_ledger(review, tmp_path):
    """The ledger is written per action as it lands, not batched at the end — a batch
    written after the loop is exactly what a mid-run failure would lose."""
    decided(review, keys(review)[0], {"c1": "keep", "c2": "cull", "c3": "cull"})
    plan = a_plan(review)
    writer = FakeWriter(fails_on=[("c2", f"album:{CANDIDATES_PATH}")])

    with WritebackLedger(tmp_path / "partial.db") as opened:
        run(plan, writer=writer, ledger=opened, dry_run=False)
        landed = opened.applied()

    # c1 (keep, no favourite) generates no action at all (`keepers-get-no-album`), so it
    # never appears in the ledger, landed or otherwise.
    assert not any(uuid == "c1" for uuid, _action in landed)
    assert ("c2", f"album:{CANDIDATES_PATH}") not in landed
    assert ("c2", f"keyword:{CULL_KEYWORD}") in landed, (
        "the album add failed; the keyword did not, and each action is its own ledger row"
    )


def test_an_interrupted_run_keeps_every_action_that_had_already_landed(review, tmp_path):
    """The property that makes the ledger per-action rather than a batch at the end.

    Found by the mutation pass: moving the writes into one loop after the run passes
    every other test here, because a *failing* action is caught and the loop still
    finishes. Ctrl-C is not — `except Exception` does not catch `KeyboardInterrupt`, so
    it leaves the function — and on a real run of ~9,000 AppleScript round trips,
    stopping it by hand is an ordinary thing to do. A batched ledger would lose the
    record of everything already written to Photos, and the re-run would repeat all of
    it against the live library."""
    decided(review, keys(review)[0], {"c1": "keep", "c2": "cull", "c3": "cull"})
    plan = a_plan(review)
    writer = FakeWriter(interrupts_on=[("c3", f"album:{CANDIDATES_PATH}")])

    with WritebackLedger(tmp_path / "interrupted.db") as opened:
        with pytest.raises(KeyboardInterrupt):
            run(plan, writer=writer, ledger=opened, dry_run=False)
        landed = opened.applied()

    # c1 (keep, no favourite) generates no action at all (`keepers-get-no-album`).
    assert not any(uuid == "c1" for uuid, _action in landed)
    assert ("c2", f"album:{CANDIDATES_PATH}") in landed
    assert ("c2", f"keyword:{CULL_KEYWORD}") in landed
    assert ("c3", f"album:{CANDIDATES_PATH}") not in landed
    assert len(landed) == len(writer.calls), (
        "the ledger and the library must agree exactly at the moment the run stops"
    )


def test_the_ledger_survives_being_reopened(tmp_path):
    with WritebackLedger(tmp_path / "w.db") as first:
        first.record("u1", "favorite", session_id="s")
    with WritebackLedger(tmp_path / "w.db") as second:
        assert second.applied() == {("u1", "favorite")}
        second.record("u1", "favorite", session_id="s2")
        assert len(second.rows()) == 1, "recording the same action twice adds no row"


def test_the_report_names_what_would_be_skipped_over_http(clusters, log, tmp_path):
    """A skip is the exception, so it is the part of the report that is listed in full
    rather than counted — the owner needs to know *which* photo was left alone."""
    review = ReviewSession(
        clusters,
        log=log,
        session_id="s",
        config=ClusterConfig(),
        resolver=_display,
        writeback=WritebackSettings(
            analysed_library=LIBRARY,
            ledger_path=tmp_path / "writeback.db",
            target_library=lambda: LIBRARY,
        ),
    )
    key = keys(review)[0]
    log.record(
        session_id="older",
        cluster_key=key,
        config_digest=review.config_digest,
        settings=review.settings,
        marks=[
            Mark(photo_uuid="c1", mark="keep"),
            Mark(photo_uuid="c2", mark="cull", mod_date=EDITED),
            Mark(photo_uuid="c3", mark="cull"),
        ],
    )
    review.confirm_final_check(review.staged_digest())
    report = review.write_back(dry_run=True)

    assert report["counts"]["skipped"] == 1
    assert report["skipped"][0]["photo_uuid"] == "c2"
    assert report["skipped"][0]["reason"] == "edited"


def test_time_moves_only_forward_in_the_ledger(tmp_path):
    """`applied_at` is stamped, never read back as a key — the ledger's identity is
    `(photo_uuid, action)`, so two clocks disagreeing cannot make an action repeat."""
    when = datetime(2026, 8, 16, 10, 0, 0)
    with WritebackLedger(tmp_path / "w.db") as opened:
        opened.record("u1", "favorite", session_id="s", now=when)
        opened.record("u1", "favorite", session_id="s", now=when - timedelta(days=1))
        rows = opened.rows()
    assert len(rows) == 1
    assert rows[0]["applied_at"] == when.isoformat()


@pytest.mark.parametrize(
    "path", ["Cull/Keepers", "Cull/Candidates", "Cull/Screenshots", "Cull/Short Videos"]
)
def test_album_paths_one_level_under_cull_resolve(path):
    folder, name = _album_location(path)
    assert folder == ("Cull",)
    assert name == path.split("/", 1)[1]


@pytest.mark.parametrize(
    "path",
    [
        "Keepers",              # not under Cull at all
        "Other/Keepers",        # a different top-level folder
        "Cull/A/B",             # nested deeper than one level
        "Cull/",                # empty album name
        "Cull",                 # the folder itself, no album
        "",
    ],
)
def test_album_paths_outside_the_rule_are_refused(path):
    with pytest.raises(ValueError):
        _album_location(path)
