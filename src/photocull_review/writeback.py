"""Write-back: the one module in this project that mutates the Photos library.

Everything it may do is three verbs — add to an album, set a keyword, set favorite —
and this project never allows anything else anywhere in the tree (see
`tests/test_guardrails.py`). What follows is not defensive style; each piece is here
because the failure it prevents is silent, plausible and irreversible.

**The library that gets written is not the one that was read.** `photoscript` drives
Photos.app over AppleScript, and Photos.app has exactly one library open — whichever was
opened last. Nothing about `--library /elsewhere` reaches it. So an analysis of library B
would write its verdicts into library A with no error at any layer. `assert_same_library`
compares the analysed path against the last-opened path and hard-fails before the first
mutation. It refuses an unknown path on either side rather than reading it as "no
mismatch found", which would turn the guard into a no-op exactly when it cannot see.

**An album inside a folder is not an album with a slash in its name.**
`create_album("Cull/Candidates")` makes a single flat album literally called
"Cull/Candidates", and the next run makes "Cull/Candidates 2". `make_album_folders` is
the folder-aware call, and — read off the installed source rather than assumed — it
returns the existing album when one is already there.

**`Photo.keywords` is a setter that replaces the list.** `photo.keywords = [KEYWORD]`
destroys every keyword the owner ever applied, and it looks exactly like working code.
The read-modify-write therefore lives in `run_writeback`, above the writer protocol,
where one test covers every writer — a protocol method called `add_keyword` would push
the property down into each implementation and leave the fake testing itself.

**A run that stops half way must be re-runnable.** `WritebackLedger` records each action
as it lands, keyed `(photo_uuid, action)`, so a second run repeats nothing. It is its own
file: `--decisions` may relocate the decision log, and the two answer different questions
— the log is keyed to a *config*, the ledger to a *library*. A ledger that travelled with
the decision log would let a re-run repeat every action against the real Photos library.

**A dry run constructs nothing.** `PhotosLibrary.__init__` runs
`photosLibraryWaitForPhotos`: it launches Photos.app and blocks for up to 300 seconds.
A dry run that built a writer to ask which library was open would open an application in
order to report what it was not going to do. The library check reads a plist instead
(`osxphotos.utils.get_last_library_path`), and no writer is built unless a real run has
cleared both gates.

**Keepers are favourited opt-in, not automatically.** `docs/SPEC.md` says keepers "set
Favorite"; the Review UI section says `F` *marks* a photo for write-back. The tie is
broken by the album builder, which filters to favorites within a date range —
auto-favoriting every keeper would make that filter select every keeper in the
library instead of the much smaller set the owner actually curated as favorites,
destroying the meaning of the flag the album builder relies on.

**A keeper gets no album.** An earlier version of this app also added every keeper to a
`Cull/Keepers` album, independently of favourites, so the album builder could read the
keeper set from album membership instead of favourites. Dropped on request: unlike
`Cull/Candidates` — which exists specifically to be reviewed and then deleted from — a
keepers album has no action attached to it at all, just a folder that grows forever with
nothing to do about it. The trade-off is real and worth stating plainly: the album
builder's keeper pool now only grows from favourites the owner sets by hand, not from
every review decision.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .decisions import Decision, mod_key

#: The folder the two albums live in, as a path of folder names — never as a string with
#: a slash in it, which is the flat-album trap.
CULL_FOLDER = ("Cull",)
KEEPERS_ALBUM = "Keepers"
CANDIDATES_ALBUM = "Candidates"

#: How the same albums read to a human, and how they are keyed in the ledger.
KEEPERS_PATH = "Cull/Keepers"
CANDIDATES_PATH = "Cull/Candidates"

#: The one keyword this app ever writes. Added alongside whatever the owner already has.
CULL_KEYWORD = "cull-candidate"

KEEP = "keep"
CULL = "cull"

#: The complete set of things this connection may do — reads, inserts, transactions —
#: measured the same way `DecisionLog._ALLOWED_ACTIONS` was, by running this module's own
#: statements under a logging authorizer. Default-deny rather than a list of forbidden
#: verbs, for the reason recorded in `decisions.py`: naming the destructive constants in
#: order to forbid them would trip the guardrail gate that looks for exactly those names
#: in executable code, and teach a future reader that the gate is arguable. It is not.
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
CREATE TABLE IF NOT EXISTS writeback_ledger (
    photo_uuid TEXT NOT NULL,
    action TEXT NOT NULL,
    session_id TEXT NOT NULL,
    applied_at TEXT NOT NULL,
    PRIMARY KEY (photo_uuid, action)
);
"""


class WrongLibrary(RuntimeError):
    """The library Photos has open is not the library that was analysed.

    A hard failure with both paths in the message, because the owner is the only one who
    can fix it — by opening the right library in Photos.app — and a refusal that does not
    say which library is which just relocates the puzzle."""


def normalise_library(path: str | Path | None) -> str | None:
    """A library path in one spelling. `None` stays `None` — see `assert_same_library`."""
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    return str(Path(text).expanduser())


def assert_same_library(analysed: str | Path | None, targeted: str | Path | None) -> None:
    """Refuse unless the library about to be written is the one that was read."""
    here, there = normalise_library(analysed), normalise_library(targeted)
    if here is None or there is None:
        raise WrongLibrary(
            "Cannot confirm which Photos library would be written to "
            f"(analysed: {analysed!r}, Photos has open: {targeted!r}). Write-back "
            "targets whichever library Photos.app opened last, so an unidentified "
            "target is refused rather than assumed to be the right one."
        )
    if here != there:
        raise WrongLibrary(
            f"The analysed library is {here!r} but Photos.app last opened {there!r}. "
            "photoscript always writes to the last-opened library, so this run would "
            "put one library's verdicts into another. Open the analysed library in "
            "Photos and try again."
        )


def last_opened_library() -> str | None:
    """The library `photoscript` would write to, read from Photos' own preferences.

    A plist read — no AppleScript, no Photos.app, so it is safe on the dry-run path.
    `osxphotos` is imported lazily so importing this module needs neither it nor a
    Photos library."""
    from osxphotos.utils import get_last_library_path

    return get_last_library_path()


# --- what a write-back is made of ---------------------------------------------------


@dataclass(frozen=True)
class Action:
    """One mutation against one photo, and the ledger key that makes it idempotent."""

    photo_uuid: str
    kind: str  # "album" | "keyword" | "favorite"
    target: str = ""  # the album path, the keyword, or "" for favorite

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.target}" if self.target else self.kind

    def to_dict(self) -> dict[str, str]:
        return {"photo_uuid": self.photo_uuid, "kind": self.kind, "target": self.target}


@dataclass(frozen=True)
class Skipped:
    """A photo the write-back deliberately left alone, and why.

    Listed in full in every report rather than counted: a skip is the exception, and the
    owner needs to know *which* photograph their decision was not applied to."""

    photo_uuid: str
    cluster_key: str
    reason: str  # "gone" | "edited"
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "photo_uuid": self.photo_uuid,
            "cluster_key": self.cluster_key,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Failure:
    """An action Photos refused. Named, never swallowed."""

    action: Action
    error: str

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action.to_dict(), "error": self.error}


@dataclass(frozen=True)
class WritePlan:
    """Everything a run would do, decided before anything is done."""

    actions: tuple[Action, ...] = ()
    skipped: tuple[Skipped, ...] = ()
    keepers: tuple[str, ...] = ()
    staged: tuple[str, ...] = ()
    favorites: tuple[str, ...] = ()

    @property
    def albums(self) -> tuple[str, ...]:
        """The album paths this plan actually needs, in first-use order.

        Derived from the actions rather than fixed, so a review with nothing staged does
        not create an empty `Cull/Candidates` in the owner's library."""
        seen: list[str] = []
        for action in self.actions:
            if action.kind == "album" and action.target not in seen:
                seen.append(action.target)
        return tuple(seen)


@dataclass(frozen=True)
class WriteReport:
    """What a run did, or — for a dry run — what it would have done."""

    dry_run: bool
    planned: tuple[Action, ...] = ()
    applied: tuple[Action, ...] = ()
    already_done: tuple[Action, ...] = ()
    failed: tuple[Failure, ...] = ()
    skipped: tuple[Skipped, ...] = ()
    albums: tuple[str, ...] = ()
    keepers: int = 0
    staged: int = 0
    favorites: int = 0

    @property
    def ok(self) -> bool:
        return not self.failed

    def to_dict(self) -> dict[str, Any]:
        """The report as JSON, summarised where it would otherwise be the sheet again.

        The real staged set runs to thousands of photos, so a report listing every action
        would be `/api/final-check` a second time and megabytes over the wire. What is
        listed in full is what is *exceptional*: the skips and the failures."""
        return {
            "dry_run": self.dry_run,
            "ok": self.ok,
            "counts": {
                "planned": len(self.planned),
                "applied": len(self.applied),
                "already_done": len(self.already_done),
                "failed": len(self.failed),
                "skipped": len(self.skipped),
            },
            "albums": list(self.albums),
            "keepers": self.keepers,
            "staged": self.staged,
            "favorites": self.favorites,
            "keyword": CULL_KEYWORD,
            "skipped": [skip.to_dict() for skip in self.skipped],
            "failed": [failure.to_dict() for failure in self.failed],
        }


def plan_writeback(
    decisions: Mapping[str, Decision], live: Mapping[str, datetime | None]
) -> WritePlan:
    """Turn the decisions that still stand into the actions that would carry them out.

    `live` is the library as the current scan sees it: uuid -> last-modified date. Both
    re-verification rules below compare against it rather than against decision time,
    because a decision may be days old.

    A photo absent from `live` is **gone** and a photo whose `mod_date` has moved was
    **edited** since the owner judged it; either way nothing is written for it. The
    "gone" branch is unreachable through today's `ReviewSession` — `cluster_key` digests
    its sorted members, so a photo leaving the library changes its cluster's key and
    orphans the decision before it can reach here. It stays because that is a property of
    a hash function two modules away, and this module's promise is that nothing is
    written for a photo it cannot see. The reconcile step in `decisions.py` makes the
    same move for the same reason.
    """
    actions: list[Action] = []
    skipped: list[Skipped] = []
    keepers: list[str] = []
    staged: list[str] = []
    favorites: list[str] = []

    for cluster_key, decision in decisions.items():
        for mark in decision.marks:
            uuid = mark.photo_uuid
            if mark.mark not in (KEEP, CULL):
                continue  # `unset` is a real answer: it stages nothing and keeps nothing.
            if uuid not in live:
                skipped.append(
                    Skipped(uuid, cluster_key, "gone", "no longer in the library")
                )
                continue
            if mod_key(mark.mod_date) != mod_key(live[uuid]):
                skipped.append(
                    Skipped(
                        uuid,
                        cluster_key,
                        "edited",
                        "edited since the decision was made; decided against "
                        f"{mod_key(mark.mod_date) or 'an unedited photo'}, "
                        f"now {mod_key(live[uuid]) or 'unedited'}",
                    )
                )
                continue

            if mark.mark == KEEP:
                keepers.append(uuid)
                # No album action here: a keeper has nothing the owner needs to do
                # about it, so unlike `Cull/Candidates` it gets no cleanup step of its
                # own. `keepers` above still names the set for the report and for
                # `favorite`, below.
                #
                # Only a keeper is ever favourited. `_check_favorites` refuses the
                # combination at the API, and this is the module that would act on it:
                # a favourited photo inside `Cull/Candidates` is one the owner would
                # have to manually delete despite having favourited it.
                if mark.favorite:
                    favorites.append(uuid)
                    actions.append(Action(uuid, "favorite"))
            else:
                staged.append(uuid)
                actions.append(Action(uuid, "album", CANDIDATES_PATH))
                actions.append(Action(uuid, "keyword", CULL_KEYWORD))

    return WritePlan(
        actions=tuple(actions),
        skipped=tuple(skipped),
        keepers=tuple(keepers),
        staged=tuple(staged),
        favorites=tuple(favorites),
    )


# --- the seam to Photos --------------------------------------------------------------


class PhotosWriter(Protocol):
    """The complete set of mutations this project may perform on a Photos library.

    Deliberately primitive. `keywords`/`set_keywords` are a read and a *replacing* write
    — exactly what photoscript offers — so the read-modify-write that preserves the
    owner's existing keywords lives in `run_writeback` and is covered once, for every
    implementation, by one test."""

    def ensure_album(self, folder_path: Sequence[str], album_name: str) -> Any:
        """The album at `folder_path/album_name`, creating folder and album if needed."""

    def add_to_album(self, album: Any, photo_uuid: str) -> None: ...

    def keywords(self, photo_uuid: str) -> Sequence[str]: ...

    def set_keywords(self, photo_uuid: str, keywords: Sequence[str]) -> None: ...

    def set_favorite(self, photo_uuid: str, value: bool) -> None: ...


class PhotoScriptWriter:
    """The real one. Every call here drives Photos.app over AppleScript.

    Constructing it launches Photos and can block for up to 300 seconds
    (`photosLibraryWaitForPhotos`), so it is built only on a real run — never to answer a
    question, and never on the dry-run path.
    """

    def __init__(self, library: Any = None):
        import photoscript

        self._photoscript = photoscript
        self._library = library if library is not None else photoscript.PhotosLibrary()

    def _photo(self, photo_uuid: str) -> Any:
        # `Photo(uuid)` raises ValueError for a uuid Photos does not know, which is a
        # second, live check behind `plan_writeback`'s re-verification.
        return self._photoscript.Photo(photo_uuid)

    def ensure_album(self, folder_path: Sequence[str], album_name: str) -> Any:
        # `make_album_folders`, never `create_album`: the latter would make one flat
        # album whose name contains a slash, and a second one on the next run.
        return self._library.make_album_folders(album_name, list(folder_path))

    def add_to_album(self, album: Any, photo_uuid: str) -> None:
        album.add([self._photo(photo_uuid)])

    def keywords(self, photo_uuid: str) -> Sequence[str]:
        return list(self._photo(photo_uuid).keywords)

    def set_keywords(self, photo_uuid: str, keywords: Sequence[str]) -> None:
        self._photo(photo_uuid).keywords = list(keywords)

    def set_favorite(self, photo_uuid: str, value: bool) -> None:
        self._photo(photo_uuid).favorite = bool(value)


# --- the ledger ----------------------------------------------------------------------


def default_ledger_path() -> Path:
    """`~/.local/state/photocull/writeback.db`, beside the decision log but not in it."""
    return Path.home() / ".local" / "state" / "photocull" / "writeback.db"


class WritebackLedger:
    """What has already been done to the real library, keyed `(photo_uuid, action)`.

    Insert-only, and enforced by SQLite's own authorizer rather than by this module's
    good manners — the same two-mechanism argument the decision log makes, for the same
    reason: a rewritten row here means a repeated mutation there.
    """

    def __init__(self, db_path: str | Path | None = None):
        path = Path(db_path) if db_path is not None else default_ledger_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._conn.set_authorizer(self._authorizer)

    @staticmethod
    def _authorizer(action: int, arg1, arg2, db_name, trigger) -> int:
        return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY

    @property
    def connection(self) -> sqlite3.Connection:
        """The guarded handle. Exposed so tests can prove the authorizer is live."""
        return self._conn

    def applied(self) -> set[tuple[str, str]]:
        return {
            (row[0], row[1])
            for row in self._conn.execute("SELECT photo_uuid, action FROM writeback_ledger")
        }

    def record(
        self, photo_uuid: str, action: str, *, session_id: str, now: datetime | None = None
    ) -> None:
        """Note that this action has landed. Recording it twice adds no row.

        `INSERT OR IGNORE`, never `INSERT OR REPLACE`: SQLite resolves REPLACE below the
        authorizer, so it would silently rewrite the row that says when this photo was
        first written to (found by testing, which is why the static SQL gate exists
        alongside the authorizer)."""
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO writeback_ledger "
                "(photo_uuid, action, session_id, applied_at) VALUES (?, ?, ?, ?)",
                (photo_uuid, action, session_id, (now or datetime.now()).isoformat()),
            )

    def rows(self) -> list[dict[str, Any]]:
        return [
            {
                "photo_uuid": row[0],
                "action": row[1],
                "session_id": row[2],
                "applied_at": row[3],
            }
            for row in self._conn.execute(
                "SELECT photo_uuid, action, session_id, applied_at FROM writeback_ledger "
                "ORDER BY applied_at, photo_uuid, action"
            )
        ]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "WritebackLedger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --- running it ----------------------------------------------------------------------


@dataclass(frozen=True)
class WritebackSettings:
    """What a review sitting is allowed to do to Photos, decided at launch.

    Defaults to "nothing": a session built without this refuses a real run rather than
    falling back to the owner's live library.
    """

    #: `photocull review --allow-write-back`. Authorises the *session*; confirming the
    #: final check authorises the *set*. Both are required, and they are independent.
    allow_write_back: bool = False
    #: The library that was analysed, as `PhotosDB.library_path` reported it.
    analysed_library: str | None = None
    ledger_path: Path | None = None
    #: Deferred so the plist is read when the write-back runs, not at launch.
    target_library: Callable[[], str | None] = last_opened_library
    #: Built only for a real run — see `PhotoScriptWriter`.
    writer: Callable[[], PhotosWriter] | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


def run_writeback(
    plan: WritePlan,
    *,
    analysed_library: str | Path | None,
    targeted_library: str | Path | None,
    dry_run: bool = True,
    writer: PhotosWriter | None = None,
    ledger: WritebackLedger | None = None,
    session_id: str = "",
) -> WriteReport:
    """Perform the plan, or report it without touching anything.

    The library check runs in **both** modes. A dry run that skipped it would report a
    plan for a library it could never write to, and confirming the write target is the
    first thing this function does.

    A failing action does not abort the run. One photo Photos refuses must not strand the
    other four thousand half-written — every failure is named in the report, and the
    ledger records each action as it lands rather than in a batch at the end, so a re-run
    picks up exactly where this one stopped.
    """
    assert_same_library(analysed_library, targeted_library)

    done = ledger.applied() if ledger is not None else set()
    already = tuple(a for a in plan.actions if (a.photo_uuid, a.key) in done)
    pending = [a for a in plan.actions if (a.photo_uuid, a.key) not in done]

    report = WriteReport(
        dry_run=dry_run,
        planned=plan.actions,
        already_done=already,
        skipped=plan.skipped,
        albums=plan.albums,
        keepers=len(plan.keepers),
        staged=len(plan.staged),
        favorites=len(plan.favorites),
    )
    if dry_run:
        return report

    if writer is None:
        raise ValueError(
            "a real write-back needs a writer; refusing rather than quietly performing "
            "a dry run, which would report success and change nothing"
        )
    if ledger is None:
        raise ValueError(
            "a real write-back needs a ledger, or a re-run after any failure would "
            "repeat every action against the library"
        )

    albums: dict[str, Any] = {}
    applied: list[Action] = []
    failed: list[Failure] = []

    for action in pending:
        try:
            _perform(action, writer=writer, albums=albums)
        except Exception as error:  # noqa: BLE001 - reported, never swallowed
            failed.append(Failure(action, f"{type(error).__name__}: {error}"))
            continue
        ledger.record(action.photo_uuid, action.key, session_id=session_id)
        applied.append(action)

    from dataclasses import replace

    return replace(report, applied=tuple(applied), failed=tuple(failed))


def _perform(action: Action, *, writer: PhotosWriter, albums: dict[str, Any]) -> None:
    """One mutation. The only place in this project that changes a Photos library."""
    if action.kind == "album":
        album = albums.get(action.target)
        if album is None:
            # Resolved on first use, so a run with nothing left to do touches no album
            # at all — which is what keeps a second run from ever meeting the
            # "Cull/Candidates 2" path.
            folder, name = _album_location(action.target)
            album = writer.ensure_album(folder, name)
            albums[action.target] = album
        writer.add_to_album(album, action.photo_uuid)
    elif action.kind == "keyword":
        # Read-modify-write. `set_keywords` REPLACES, so anything else here destroys
        # every keyword the owner has on this photo.
        existing = list(writer.keywords(action.photo_uuid))
        if action.target not in existing:
            writer.set_keywords(action.photo_uuid, [*existing, action.target])
    elif action.kind == "favorite":
        writer.set_favorite(action.photo_uuid, True)
    else:
        raise ValueError(f"unknown write-back action {action.kind!r}")


def _album_location(album_path: str) -> tuple[tuple[str, ...], str]:
    """`"Cull/Keepers"` -> `(("Cull",), "Keepers")`.

    The split happens here and the two halves stay apart from then on: the name is what
    Photos calls the album, and a name carrying a slash is the flat-album bug.

    This was an allowlist of the two albums the review writes to. The bulk sweep adds
    one album per sweep category, so it is now the *rule* those two constants always
    satisfied:
    exactly one level under the `Cull` folder. Everything the allowlist actually
    protected is still enforced -- nothing outside `Cull/`, no nesting, no empty name --
    and it is enforced for the review's own albums too rather than waved through by
    identity."""
    folder, sep, name = album_path.partition("/")
    if (folder,) != CULL_FOLDER or not sep or not name or "/" in name:
        raise ValueError(f"unknown album {album_path!r}")
    return CULL_FOLDER, name
