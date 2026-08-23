"""The review API: the owner's judgement in, an append-only record out.

Eight routes, and one class that holds everything they share. The class exists rather
than a bag of handlers because every rule below has to hold whether it is reached over
HTTP or not — the same argument `images.ImageOutcome` makes, for the same reason: the
safety properties of this module must not be contingent on a test client, a route, or a
running server.

**The request body carries verdicts and nothing else.** `keep`/`cull`/`unset` per photo,
plus which takes to favourite. Everything the log stores *about* a photo — its
`total`, its `sub_scores`, what the scorer proposed, its `mod_date` — is read from this
session's own document. That is not defensive tidiness: every decision is training data
for a later re-tuning of the scorer, so a body-supplied `sub_scores` would let anything
holding the session cookie author the training set, and the resulting log would read as
entirely ordinary afterwards. The defence is structural, not a validation step — the
body's evidence fields are never looked at, so there is no check to get wrong.

**A cluster may be decided with no keeper at all.** This route refused that until the
owner overruled it: "none of these is worth keeping" is an answer, and a burst of five
frames of nothing is the backlog this app exists to cull. The refusal is gone rather
than narrowed, so both overview surfaces below now receive staged photos with an empty
`keepers` list. Deferring the whole cluster (`unset` throughout) is unchanged and still
stages nothing — that is the state an ambiguous cluster expects the owner to be in a
real share of the time, and it is a different answer from "cull them all".

**Both overview surfaces read decisions, never proposals.** Since every cluster arrives
with a proposed mark set, a final check built from proposals would show the whole
staged set before the owner has decided anything — the scorer's opinion presented as
theirs, on the one screen whose job is to confirm what they actually chose.

**Decisions recorded under other settings are invisible here.** `reconcile` holds those
back as `superseded` rather than applying them; if the dashboard counted them, it would
report a cluster as decided that resume is about to re-ask.

**The final check is a gate, and it names what it authorised.** Confirming it stores a
digest of the staged set rather than setting a flag, so `writeback_ready()` is a
comparison and not a flag read. That is the difference between "the owner has confirmed
*this* set" and "the owner confirmed something once": with a flag, confirming one staged
set and then reviewing many more clusters would hand write-back an authorisation for a
set nobody ever saw, and every reset would have to be remembered by hand at each of the
routes that can move the set. Here it lapses by construction and comes back on its own if
an undo restores exactly the set that was confirmed.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from photocull.config import ClusterConfig
from photocull.models import Cluster

from .decisions import MARKS, Decision, DecisionLog, Mark
from .session import (
    Resolver,
    build_session,
    config_digest,
    resolve_display,
    session_settings,
)
from .writeback import (
    PhotoScriptWriter,
    WritebackLedger,
    WritebackSettings,
    WrongLibrary,
    plan_writeback,
    run_writeback,
)

KEEP = "keep"
CULL = "cull"
UNSET = "unset"

#: How many staged clusters the dashboard names. The owner asked for "a quick overview
#: of what's to be deleted"; the full list is the final check, and a dashboard that
#: printed every staged row on a real library would be the same thing twice.
LARGEST_STAGED = 10

#: How many staged rows one page of the final check carries over HTTP. The sheet is the
#: only surface in this app whose size is the *library's* rather than one cluster's — on
#: a real library the staged set runs to thousands — so the route pages while the
#: in-process call does not (write-back builds its plan from the whole set instead).
FINAL_CHECK_PAGE = 200

#: An upper bound on `limit`, so one request cannot ask for the whole sheet by accident.
FINAL_CHECK_MAX_PAGE = 1000


class ApiRefusal(Exception):
    """A refusal the owner should read, decided without any HTTP involved.

    Carries the status because the distinction matters to the UI: 404 means "this
    session does not know that cluster" (a stale tab, a mistyped key) while 422 means
    "the submission itself is wrong" and names which part.
    """

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class Evidence:
    """What the log stores about one photo, sourced entirely from this session.

    Assembled once when the session opens, from the scan (`mod_date`) and the session
    document (everything else), so a decision handler has nothing left to look up and
    nothing to take on trust from its caller."""

    uuid: str
    mod_date: datetime | None
    proposed: str
    total: float | None
    sub_scores: Mapping[str, float | None]


def new_session_id() -> str:
    """A per-launch id, so the log can tell two review sittings apart."""
    return secrets.token_hex(8)


class ReviewSession:
    """One review sitting: the document the browser reads, and the log it writes to.

    Holds the clusters' evidence privately and the document publicly, the same split the
    session document itself draws for filesystem paths — `mod_date` and the scorer's
    proposal are needed to *write* a decision and are of no use to the browser, so they
    never leave this object.
    """

    def __init__(
        self,
        clusters: Sequence[Cluster],
        *,
        log: DecisionLog,
        session_id: str | None = None,
        config: ClusterConfig | None = None,
        source: str | Path | None = None,
        resolver: Resolver = resolve_display,
        writeback: WritebackSettings | None = None,
    ):
        cfg = config or ClusterConfig()
        self.log = log
        self.session_id = session_id or new_session_id()
        #: Defaults to "this session may not touch Photos at all" — see
        #: `WritebackSettings`, and `write_back` for the two gates.
        self.writeback = writeback or WritebackSettings()
        self.settings = session_settings(cfg, source)
        self.config_digest = config_digest(cfg, source)
        self.document = build_session(clusters, config=cfg, source=source, resolver=resolver)
        self.shutdown_requested = False
        #: The staged set the owner confirmed on the final check, as a digest, or `None`
        #: for "not confirmed". Session-scoped on purpose: it authorises a write-back in
        #: *this* sitting, and a confirmation surviving into a later launch would mean
        #: the owner had authorised a set they may never have seen in that session.
        self._confirmed: str | None = None

        self._entries: dict[str, dict[str, Any]] = {}
        self._evidence: dict[str, dict[str, Evidence]] = {}
        self._order: list[str] = []

        for cluster, entry in zip(clusters, self.document["clusters"], strict=True):
            key = entry["cluster_key"]
            if key in self._entries:
                # Two clusters cannot hold the same photos, so this means the key stopped
                # identifying a membership — and every decision ever recorded is looked up
                # by it. Fail at launch rather than silently merging two clusters' marks.
                raise ValueError(f"two clusters share the key {key!r}")
            members = {member["uuid"]: member for member in entry["members"]}
            self._entries[key] = entry
            self._order.append(key)
            self._evidence[key] = {
                record.uuid: Evidence(
                    uuid=record.uuid,
                    mod_date=record.mod_date,
                    proposed=entry["proposed"].get(record.uuid, UNSET),
                    total=members[record.uuid]["total"],
                    sub_scores=members[record.uuid]["sub_scores"],
                )
                for record in cluster.records
            }

    # --- reading ------------------------------------------------------------------

    def entry(self, cluster_key: str) -> dict[str, Any]:
        """One cluster as the browser sees it.

        `cluster_key` is a dictionary key and never anything else — never joined to a
        directory, never passed to `Path()` — exactly as `images.resolve_image` treats a
        uuid. A traversal string is not sanitised here; it simply misses the mapping."""
        try:
            return self._entries[cluster_key]
        except KeyError:
            raise ApiRefusal(404, "No such cluster in this review session.") from None

    def live_decisions(self) -> dict[str, Decision]:
        """Every decision that still stands *for this session's settings*.

        Read fresh from the log on every call rather than cached, so the API cannot
        drift from the file that outlives it. Measured on the real library: `all_state()`
        reads 1,807 clusters back in 0.04 s, which is affordable per request for a
        single-user local app and removes a whole class of staleness bug."""
        return {
            key: decision
            for key, decision in self.log.all_state().items()
            if key in self._entries and decision.config_digest == self.config_digest
        }

    def session_document(self) -> dict[str, Any]:
        return self.document | {
            "session_id": self.session_id,
            "decided": {
                key: {
                    "marks": decision.marks_by_uuid(),
                    # Carried here because a reload is ordinary — the session is designed
                    # to survive a closed browser — and the compare view seeds its state
                    # from this document. Without them, re-recording a cluster after a
                    # reload, or pulling one photo back to KEEP from the contact sheet,
                    # would drop every `F` press in that cluster silently.
                    "favorites": _favorites_of(decision),
                    "batch_id": decision.batch_id,
                    "decided_at": decision.decided_at.isoformat(),
                }
                for key, decision in self.live_decisions().items()
            },
        }

    def cluster_document(self, cluster_key: str) -> dict[str, Any]:
        entry = self.entry(cluster_key)
        decision = self.live_decisions().get(cluster_key)
        return entry | {
            "decision": None if decision is None else self._state(cluster_key, decision)
        }

    def _state(self, cluster_key: str, decision: Decision | None) -> dict[str, Any]:
        """A decision reduced to what the UI acts on: who is kept, who is staged."""
        marks = {} if decision is None else decision.marks_by_uuid()
        ordered = [uuid for uuid in self._evidence[cluster_key] if uuid in marks]
        return {
            "cluster_key": cluster_key,
            "batch_id": None if decision is None else decision.batch_id,
            "marks": marks,
            "kept": [uuid for uuid in ordered if marks[uuid] == KEEP],
            "staged": [uuid for uuid in ordered if marks[uuid] == CULL],
            "favorites": [] if decision is None else _favorites_of(decision),
        }

    # --- writing ------------------------------------------------------------------

    def record(
        self,
        cluster_key: str,
        marks: Mapping[str, str],
        *,
        favorites: Iterable[str] = (),
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Validate a submission, then append it. Nothing half-lands: every check runs
        before the first INSERT, mirroring `DecisionLog._validate`'s own contract."""
        self.entry(cluster_key)  # 404s an unknown key before anything else runs
        evidence = self._evidence[cluster_key]
        favorites = list(favorites)
        _check_marks(marks, evidence)
        _check_favorites(favorites, marks, evidence)

        batch_id = self.log.record(
            session_id=self.session_id,
            cluster_key=cluster_key,
            config_digest=self.config_digest,
            settings=self.settings,
            now=now,
            marks=[
                Mark(
                    photo_uuid=uuid,
                    mark=marks[uuid],
                    # Every field below comes from this session, never from the caller.
                    mod_date=item.mod_date,
                    proposed=item.proposed,
                    favorite=uuid in favorites,
                    total=item.total,
                    sub_scores=item.sub_scores,
                )
                for uuid, item in evidence.items()
            ],
        )
        return self._state(cluster_key, self.log.state(cluster_key)) | {"batch_id": batch_id}

    def undo(self, cluster_key: str) -> dict[str, Any]:
        """Undo the newest live decision, exposing whatever it superseded.

        Undoing a cluster that was never decided is not an error — pressing undo once too
        often is ordinary, and inventing an event for a decision nobody made would put a
        fiction into an append-only audit trail."""
        self.entry(cluster_key)
        undone = self.log.undo(cluster_key, session_id=self.session_id)
        return self._state(cluster_key, self.log.state(cluster_key)) | {"undone": undone}

    # --- the two overview surfaces ------------------------------------------------

    def dashboard(self) -> dict[str, Any]:
        """Reviewed vs remaining, running keep/stage counts, largest staged clusters.

        Counts come from the marks, never from `size - 1`: `keep 5 of 5` stages nothing,
        and the propose-and-adjust model makes several keepers per cluster an ordinary
        outcome rather than an edge case."""
        decisions = self.live_decisions()
        counts = dict.fromkeys((KEEP, CULL, UNSET), 0)
        staged_clusters = []

        for key in self._order:
            decision = decisions.get(key)
            if decision is None:
                continue
            marks = decision.marks_by_uuid()
            for mark in marks.values():
                counts[mark] = counts.get(mark, 0) + 1
            staged = sum(1 for mark in marks.values() if mark == CULL)
            if staged:
                staged_clusters.append(
                    {
                        "cluster_key": key,
                        "size": len(self._evidence[key]),
                        "staged": staged,
                        "kept": sum(1 for mark in marks.values() if mark == KEEP),
                    }
                )

        in_clusters = sum(len(members) for members in self._evidence.values())
        index = next((i for i, key in enumerate(self._order) if key not in decisions), None)
        return {
            "clusters": {
                "total": len(self._order),
                "decided": len(decisions),
                "remaining": len(self._order) - len(decisions),
            },
            "photos": {
                "in_clusters": in_clusters,
                "keep": counts[KEEP],
                "cull": counts[CULL],
                "unset": counts[UNSET],
                "undecided": in_clusters - sum(counts.values()),
            },
            # Sorted by how much one accepted proposal would stage; ties keep capture
            # order, which `sorted`'s stability gives for free off `self._order`.
            "largest_staged": sorted(
                staged_clusters, key=lambda row: -row["staged"]
            )[:LARGEST_STAGED],
            "next_undecided": None
            if index is None
            else {"index": index, "cluster_key": self._order[index]},
        }

    def final_check(self, *, offset: int = 0, limit: int | None = None) -> dict[str, Any]:
        """Every photo left marked `cull`, beside the keepers it lost to.

        Nothing reaches write-back without having appeared here first, so it is built
        from the log alone: a cluster the owner has not decided contributes nothing, no
        matter what the scorer proposed for it.

        `limit=None` — the default, and what write-back reads in-process — is the whole
        staged set. The HTTP route defaults to one page instead. Those are opposite
        answers to the same question on purpose: a write-back covering only the first 200
        photos and a sheet fetching several thousand rows at once are both wrong."""
        rows = list(self._staged_rows())
        window = rows[offset:] if limit is None else rows[offset : offset + limit]
        return {
            "total": len(rows),
            "offset": offset,
            "limit": limit,
            "staged": window,
            "staged_digest": _digest_rows(rows),
            "confirmed": self.writeback_ready(),
        }

    def _staged_rows(self) -> list[dict[str, Any]]:
        """The staged set in capture order — cluster by cluster, take by take.

        One list behind both the sheet and the digest, so what gets confirmed cannot
        drift from what was shown."""
        decisions = self.live_decisions()
        rows = []
        for key in self._order:
            decision = decisions.get(key)
            if decision is None:
                continue
            marks = decision.marks_by_uuid()
            members = {member["uuid"]: member for member in self._entries[key]["members"]}
            keepers = [
                members[uuid]
                for uuid in self._evidence[key]
                if marks.get(uuid) == KEEP and uuid in members
            ]
            rows.extend(
                {"cluster_key": key, "photo": members[uuid], "keepers": keepers}
                for uuid in self._evidence[key]
                if marks.get(uuid) == CULL and uuid in members
            )
        return rows

    # --- the write-back gate ---------------------------------------------------------

    def staged_digest(self) -> str:
        """A digest of exactly what would be written back, and of nothing else.

        Not of the decisions, their batch ids or when they were made: an undo that
        restores the staged set the owner already confirmed has to return the same
        value, or the confirmation would lapse for a reason invisible on screen."""
        return _digest_rows(self._staged_rows())

    def confirm_final_check(self, staged_digest: str) -> dict[str, Any]:
        """Record that the owner has seen *this* staged set, and authorised it.

        The digest comes from the caller rather than from `staged_digest()`, which is the
        difference between a confirmation and a rubber stamp: a tab drawn an hour ago, or
        a second tab that has recorded decisions since, would otherwise authorise photos
        that were never on the screen being confirmed."""
        current = self.staged_digest()
        if staged_digest != current:
            raise ApiRefusal(
                409,
                "What is staged has changed since this screen was drawn, so confirming "
                "it would authorise photos you have not seen. Reload the final check and "
                "confirm what it shows.",
            )
        self._confirmed = current
        return {"confirmed": True, "staged_digest": current, "total": len(self._staged_rows())}

    def writeback_ready(self) -> bool:
        """Whether write-back may run at all — the one question write-back asks first.

        A comparison rather than a flag read, so it lapses on its own the moment the
        staged set moves, and no route that can move it has to remember to reset it."""
        return self._confirmed is not None and self._confirmed == self.staged_digest()

    def live_photos(self) -> dict[str, datetime | None]:
        """Every clustered photo as *this session's scan* saw it: uuid -> `mod_date`.

        This is what write-back re-verifies a decision against. It is the freshest read
        of the library available without a second scan, which takes real time on a large
        library, and the plan's requirement is met by it precisely because a decision
        may be days old while this scan is minutes old: comparing the two is what
        detects a photograph edited since the owner judged it."""
        return {
            uuid: item.mod_date
            for members in self._evidence.values()
            for uuid, item in members.items()
        }

    def write_back(
        self, *, dry_run: bool = True, cluster_keys: Iterable[str] | None = None
    ) -> dict[str, Any]:
        """Apply the confirmed decisions to Photos, or report what that would do.

        Two independent gates, and neither substitutes for the other. The final-check
        confirmation authorises a **set** — it is a digest comparison, so it lapses the
        moment a decision moves the staged set. `--allow-write-back` authorises the
        **session**: a review launched without it can plan and report all day and can
        never touch the library. A dry run needs only the first, because it changes
        nothing.

        `cluster_keys`, when given, narrows the plan to that subset of the *already
        confirmed* set — it never widens it. The gate above still checks the digest of
        everything staged, because that is what the owner looked at on the final check;
        this only limits how much of what they confirmed gets acted on in one call. This
        is what lets the owner run write-back against one cluster first, to verify the
        writer against the live library by hand before trusting it with the rest."""
        if not self.writeback_ready():
            raise ApiRefusal(
                409,
                "Write-back needs the final check confirmed for exactly what is staged "
                "now. Open the final check, look at what is there, and confirm it.",
            )
        if not dry_run and not self.writeback.allow_write_back:
            raise ApiRefusal(
                403,
                "This review was not launched with --allow-write-back, so it cannot "
                "write to Photos. Relaunch with that flag to perform the write-back; "
                "a dry run needs no flag and reports exactly what would happen.",
            )

        decisions = self.live_decisions()
        if cluster_keys is not None:
            requested = list(cluster_keys)
            missing = [key for key in requested if key not in decisions]
            if missing:
                raise ApiRefusal(
                    404,
                    "No confirmed decision for cluster(s): "
                    + ", ".join(missing)
                    + ". Only clusters in the confirmed staged set can be written back.",
                )
            decisions = {key: decisions[key] for key in requested}

        plan = plan_writeback(decisions, self.live_photos())
        ledger = WritebackLedger(self.writeback.ledger_path)
        writer = None
        try:
            if not dry_run:
                # Built here and nowhere else: constructing it launches Photos.app and
                # can block for 300 s, so it must never be on the dry-run path.
                factory = self.writeback.writer or PhotoScriptWriter
                writer = factory()
            report = run_writeback(
                plan,
                analysed_library=self.writeback.analysed_library,
                targeted_library=self.writeback.target_library(),
                dry_run=dry_run,
                writer=writer,
                ledger=ledger,
                session_id=self.session_id,
            )
        finally:
            # Opened per call rather than held for the session: this handle is only ever
            # needed inside this method, and Tasks 8 and 9 both lost a tick to a
            # long-lived SQLite handle outliving or predeceasing its user.
            ledger.close()
        return report.to_dict()

    def request_shutdown(self) -> None:
        self.shutdown_requested = True


def _favorites_of(decision: Decision) -> list[str]:
    """The photos this decision asked to be favourited at write-back.

    Read off the marks rather than stored separately, because the log already keeps the
    flag per photo — a second list would be a second source of truth about the same
    fact, and the rule that a favourite may not be staged is enforced against the marks
    directly."""
    return [mark.photo_uuid for mark in decision.marks if mark.favorite]


def _digest_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    """Name a staged set in 16 hex characters.

    Truncated because of what it is compared against: the same session's own value,
    minutes apart, with no adversary in the loop — this identifies a set, it does not
    authenticate one. The token in `server.py` is the thing that authenticates, and it
    is full length.

    Digesting the uuid alone is **equivalent** and was measured to be — a photo belongs to
    exactly one cluster, so the pair and the uuid name the same set. The cluster key stays
    because that equivalence is a property of the clustering, enforced two packages away,
    while this string is what authorises a write-back. Written down rather than dropped,
    so nobody "simplifies" it back and has to re-derive why it was safe."""
    joined = "\n".join(f"{row['cluster_key']}\t{row['photo']['uuid']}" for row in rows)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _check_marks(marks: Mapping[str, str], evidence: Mapping[str, Evidence]) -> None:
    """Exactly one known verdict per photo in this cluster.

    Deliberately *not* checked: that something is kept. Culling every take is a real
    answer, and the refusal that used to live here is gone rather than relaxed."""
    unknown = sorted(set(marks) - set(evidence))
    if unknown:
        raise ApiRefusal(
            422,
            f"{', '.join(unknown)} is not in this cluster. A decision names only the "
            "photos it is about.",
        )
    missing = sorted(set(evidence) - set(marks))
    if missing:
        # A partial submission would freeze the unmentioned takes at whatever an earlier
        # batch said, which is indistinguishable from the owner having answered.
        raise ApiRefusal(
            422,
            f"No verdict for {', '.join(missing)}. A submission carries exactly one "
            "verdict per photo in the cluster.",
        )
    bad = sorted({mark for mark in marks.values() if mark not in MARKS})
    if bad:
        raise ApiRefusal(
            422, f"Unknown verdict {', '.join(repr(mark) for mark in bad)}; "
            f"expected one of {sorted(MARKS)}."
        )


def _check_favorites(
    favorites: Sequence[str], marks: Mapping[str, str], evidence: Mapping[str, Evidence]
) -> None:
    """Favourites name kept photos in this cluster, and nothing else.

    Favouriting a photo that is being staged is a contradiction with a real cost:
    write-back would set Favorite on a photo that is also in `Cull/Candidates`, and the
    owner's manual delete would then take a favourited photo with it."""
    outside = sorted(set(favorites) - set(evidence))
    if outside:
        raise ApiRefusal(422, f"{', '.join(outside)} is not in this cluster.")
    staged = sorted(uuid for uuid in favorites if marks.get(uuid) == CULL)
    if staged:
        raise ApiRefusal(
            422,
            f"{', '.join(staged)} is marked favourite and staged for culling. Keep it or "
            "drop the favourite — write-back would otherwise favourite a photo it is "
            "about to stage for deletion.",
        )


# --- HTTP ------------------------------------------------------------------------


class DecisionBody(BaseModel):
    """The whole vocabulary a client may use.

    `extra="ignore"` on purpose. A body carrying `sub_scores` is not refused, it is
    *unread* — the evidence fields are never consulted, so there is no check that could
    be got wrong later, and a future frontend sending a field this version does not know
    about does not break a review mid-session."""

    model_config = ConfigDict(extra="ignore")

    marks: dict[str, str]
    favorites: list[str] = Field(default_factory=list)


class UndoBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cluster_key: str


class WritebackBody(BaseModel):
    """`dry_run` defaults to True, so a body that says nothing performs nothing.

    That default is the one place in this app where forgetting a field has to be safe:
    every other route's mistakes are recorded in an append-only log, and this one's are
    in the owner's photo library."""

    model_config = ConfigDict(extra="ignore")

    dry_run: bool = True
    #: Narrows a run to this subset of the confirmed staged set — see
    #: `ReviewSession.write_back`. `None` (the default) means the whole set.
    cluster_keys: list[str] | None = None


class ConfirmBody(BaseModel):
    """What the owner is confirming, named rather than implied — see
    `ReviewSession.confirm_final_check`."""

    model_config = ConfigDict(extra="ignore")

    staged_digest: str


def add_review_routes(
    app: FastAPI,
    review: ReviewSession,
    *,
    on_shutdown: Callable[[], None] | None = None,
) -> None:
    """Mount the eight review routes onto an already-hardened app.

    Registered only when a session exists, matching the image endpoint's rule: a route
    that is absent is a stronger statement than one that always 404s."""

    @app.exception_handler(ApiRefusal)
    async def refused(_request: Request, exc: ApiRefusal) -> JSONResponse:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status)

    @app.exception_handler(WrongLibrary)
    async def wrong_library(_request: Request, exc: WrongLibrary) -> JSONResponse:
        # A refusal the owner can act on, not a traceback. In-process the specific
        # exception survives, because a script driving this from a tick should stop on
        # it rather than read a status code.
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.get("/api/session")
    async def get_session() -> dict[str, Any]:
        return review.session_document()

    @app.get("/api/clusters/{cluster_key}")
    async def get_cluster(cluster_key: str) -> dict[str, Any]:
        return review.cluster_document(cluster_key)

    @app.post("/api/clusters/{cluster_key}/decision")
    async def post_decision(cluster_key: str, body: DecisionBody) -> dict[str, Any]:
        return review.record(cluster_key, body.marks, favorites=body.favorites)

    @app.post("/api/undo")
    async def post_undo(body: UndoBody) -> dict[str, Any]:
        return review.undo(body.cluster_key)

    @app.get("/api/dashboard")
    async def get_dashboard() -> dict[str, Any]:
        return review.dashboard()

    @app.get("/api/final-check")
    async def get_final_check(
        offset: int = 0, limit: int = FINAL_CHECK_PAGE
    ) -> dict[str, Any]:
        # Refused rather than clamped, the same rule config validation applies to
        # settings: a request for page -1 is a bug in whatever built the URL, and
        # answering it with page 0 hides that bug behind plausible output.
        if offset < 0:
            raise ApiRefusal(422, "offset must not be negative.")
        if not 1 <= limit <= FINAL_CHECK_MAX_PAGE:
            raise ApiRefusal(
                422, f"limit must be between 1 and {FINAL_CHECK_MAX_PAGE}."
            )
        return review.final_check(offset=offset, limit=limit)

    @app.post("/api/final-check/confirm")
    async def post_final_check_confirm(body: ConfirmBody) -> dict[str, Any]:
        return review.confirm_final_check(body.staged_digest)

    @app.post("/api/writeback")
    async def post_writeback(body: WritebackBody) -> dict[str, Any]:
        return review.write_back(dry_run=body.dry_run, cluster_keys=body.cluster_keys)

    @app.post("/api/shutdown", status_code=202)
    async def post_shutdown() -> dict[str, bool]:
        review.request_shutdown()
        if on_shutdown is not None:
            on_shutdown()
        return {"stopping": True}
