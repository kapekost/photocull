"""Resume reconciliation: what a relaunch may safely carry over from an earlier session.

The review is a long job — 1,807 clusters at the live cut — spread over days, against a
library that keeps living in the meantime. Photos get edited, deleted, and reshuffled into
different clusters by a re-calibration. So every launch after the first has to answer one
question honestly: *does the judgement already on disk still describe what is in front of
me?* Both wrong answers are expensive. Applying a stale decision writes back a verdict
about a comparison the owner never actually saw; discarding a live one throws away the
only artifact in this project that cannot be recomputed.

The five rules, in order, and why each is where it is:

1. **The settings moved -> refuse to resume, naming the field.** Everything downstream is
   conditional on the config: change `similarity_threshold` and the clusters themselves
   are different objects. Refusing is the conservative act, but a bare "the configuration
   changed" is useless the day after Task 9 writes `photocull.toml`, so the reason names
   the field and both values. `force_new_session` is the way past it, and it destroys
   nothing — see below.
2. **The cluster is gone -> keep it for audit, do not apply it.** A decision names a
   `cluster_key`, which is content-addressed over the member uuids. If no cluster in the
   fresh scan has that key, the group the owner judged no longer exists and the verdict
   cannot be transferred to whatever replaced it.
3. **The photo is gone -> void the mark, report it, never write it back.** This one is a
   per-photo classification *inside* rule 2 rather than a sibling of it, and that is
   structural: because `cluster_key` digests its sorted members, a photo leaving the
   library necessarily changes the key of the cluster it belonged to, so a voided mark
   always arrives on an already-orphaned decision. Stated in the plan as a peer rule; it
   cannot be one.
4. **A decided photo was edited -> requeue its whole cluster, not just that photo.** An
   edit changes that take's derivative, hence its feature print, its sharpness and its
   rank against every take beside it. The comparison the owner made no longer exists, so
   the cluster goes back in the queue whole.
5. **Otherwise resume at the first undecided cluster in capture order.**

**On `force_new_session`.** Refusing to resume must not become a one-way door 800
clusters in, and neither must forcing past one. Because the log is append-only, "start
fresh" can only ever mean *this layer stops applying* those decisions — they stay on disk
in full. Decisions are therefore matched to the current settings by digest rather than by
recency: force past a change and the next launch resumes the new session normally, while
putting the old settings back brings the old session's decisions with it. Settings are
not a ratchet in either direction.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from photocull.config import ClusterConfig
from photocull.models import Cluster

from .decisions import Decision, DecisionLog, mod_key
from .session import cluster_key, config_digest, digest_settings, session_settings


@dataclass(frozen=True)
class ConfigChange:
    """One setting that moved between the stored session and this one.

    `field` is dotted for nested settings (`weights.sharpness`), because a re-tuning under
    `review-decisions-train-the-scorer` only ever moves the nested dict, and reporting
    `weights` changed would leave the owner to diff eight numbers by eye."""

    field: str
    was: Any
    now: Any

    def describe(self) -> str:
        return f"{self.field} {self.was!r} -> {self.now!r}"


@dataclass(frozen=True)
class VoidedMark:
    """A verdict about a photo that is no longer in the library.

    Reported so the owner knows their work on it was not lost silently, and carried
    separately from `applied` so it can never reach write-back."""

    cluster_key: str
    photo_uuid: str
    mark: str


@dataclass(frozen=True)
class Resume:
    """What this launch may carry over, and everything it deliberately did not."""

    can_resume: bool
    blocked_reason: str | None = None
    config_changes: tuple[ConfigChange, ...] = ()
    #: cluster_key -> the decision that still stands. The only thing write-back may read.
    applied: Mapping[str, Decision] = field(default_factory=dict)
    #: Rule 2: decided clusters absent from the fresh scan.
    orphaned: tuple[str, ...] = ()
    #: Rule 3: marks naming photos no longer in the library.
    voided: tuple[VoidedMark, ...] = ()
    #: Rule 4: decided clusters whose photos changed underneath the decision.
    requeued: tuple[str, ...] = ()
    #: Live decisions recorded under different settings, held back rather than applied.
    superseded: tuple[str, ...] = ()
    resume_index: int | None = 0
    resume_cluster_key: str | None = None

    @property
    def decided_count(self) -> int:
        return len(self.applied)

    def report(self) -> dict[str, Any]:
        """A plain-JSON summary for the API and the launcher's console line.

        Counts and keys, never whole `Decision` objects: the browser gets the decisions
        themselves through the session document, and duplicating them here would be a
        second copy to keep in step."""
        return {
            "can_resume": self.can_resume,
            "blocked_reason": self.blocked_reason,
            "config_changes": [
                {"field": change.field, "was": change.was, "now": change.now}
                for change in self.config_changes
            ],
            "decided_count": self.decided_count,
            "orphaned": list(self.orphaned),
            "voided": [
                {
                    "cluster_key": void.cluster_key,
                    "photo_uuid": void.photo_uuid,
                    "mark": void.mark,
                }
                for void in self.voided
            ],
            "requeued": list(self.requeued),
            "superseded": list(self.superseded),
            "resume_index": self.resume_index,
            "resume_cluster_key": self.resume_cluster_key,
        }


def diff_settings(was: Mapping[str, Any], now: Mapping[str, Any]) -> tuple[ConfigChange, ...]:
    """Every field that differs, sorted, with nested dicts reported by dotted key.

    A field present on one side only is a change too — adding a tunable changes what the
    config means — and is reported against `None` rather than skipped."""
    changes: list[ConfigChange] = []
    for key in sorted(set(was) | set(now)):
        before, after = was.get(key), now.get(key)
        if isinstance(before, Mapping) and isinstance(after, Mapping):
            changes.extend(
                ConfigChange(f"{key}.{nested.field}", nested.was, nested.now)
                for nested in diff_settings(before, after)
            )
        elif before != after:
            changes.append(ConfigChange(key, before, after))
    return tuple(changes)


def _blocked(newest: Decision, settings: Mapping[str, Any]) -> tuple[str, tuple[ConfigChange, ...]]:
    """Why this session will not resume, in terms the owner can act on."""
    if newest.settings is None:
        return (
            "The settings have changed since these decisions were recorded, and the "
            "earlier settings were not recorded alongside them, so the change cannot be "
            "named. Restore the configuration you reviewed under, or start a new "
            "session — the existing decisions are kept either way.",
            (),
        )
    changes = diff_settings(newest.settings, settings)
    listed = "; ".join(change.describe() for change in changes)
    return (
        f"The settings have changed since these decisions were recorded: {listed}. "
        "The clusters those decisions describe may no longer exist. Restore the previous "
        "configuration to continue that session, or start a new one — the existing "
        "decisions are kept either way.",
        changes,
    )


def reconcile(
    clusters: Sequence[Cluster],
    *,
    log: DecisionLog,
    config: ClusterConfig | None = None,
    source: str | Path | None = None,
    library_uuids: Collection[str] | None = None,
    force_new_session: bool = False,
) -> Resume:
    """Diff the stored decision log against a fresh scan and say exactly what changed.

    `clusters` is the fresh scan, in capture order (`clusters-ordered-by-earliest-member`).

    `library_uuids` is every uuid in the library, and `None` means "not checked" rather
    than "the library is empty" — the distinction matters because only 4,498 of this
    library's 14,235 photos reach a cluster at all, so inferring the library from
    `clusters` would declare two thirds of it deleted.
    """
    cfg = config or ClusterConfig()
    settings = session_settings(cfg, source)
    digest = config_digest(cfg, source)

    stored = log.all_state()
    for decision in stored.values():
        if decision.settings is not None and digest_settings(decision.settings) != decision.config_digest:
            raise ValueError(
                f"batch {decision.batch_id} stores settings that do not match its own "
                f"config digest {decision.config_digest!r}; the log was written by a "
                "caller that passed the two from different configs"
            )

    current = {key: decision for key, decision in stored.items() if decision.config_digest == digest}
    superseded = tuple(sorted(key for key in stored if key not in current))

    if stored and not current and not force_new_session:
        newest = max(stored.values(), key=lambda decision: decision.batch_id)
        reason, changes = _blocked(newest, settings)
        return Resume(
            can_resume=False,
            blocked_reason=reason,
            config_changes=changes,
            superseded=superseded,
            resume_index=None,
        )

    fresh_keys = [cluster_key(cluster) for cluster in clusters]
    fresh_by_key = dict(zip(fresh_keys, clusters, strict=True))

    if force_new_session:
        # Nothing is deleted and nothing is applied: the whole log is held back, and the
        # session starts at the top.
        return Resume(
            can_resume=True,
            superseded=tuple(sorted(stored)),
            resume_index=0 if clusters else None,
            resume_cluster_key=fresh_keys[0] if clusters else None,
        )

    orphaned: list[str] = []
    requeued: list[str] = []
    voided: list[VoidedMark] = []
    applied: dict[str, Decision] = {}

    for key in sorted(current):
        decision = current[key]
        cluster = fresh_by_key.get(key)
        gone = (
            ()
            if library_uuids is None
            else tuple(
                VoidedMark(key, mark.photo_uuid, mark.mark)
                for mark in decision.marks
                if mark.photo_uuid not in library_uuids
            )
        )
        voided.extend(gone)

        if cluster is None:
            orphaned.append(key)
            continue

        if gone:
            # Checked here too, not only on the orphaned branch. A photo leaving the
            # library *should* change its cluster's key and land the decision above —
            # `cluster_key` digests its sorted members — so reaching this line means that
            # structural guarantee did not hold. Write-back safety is not left resting on
            # a property of a hash function two modules away: the cost is one set lookup
            # per mark, and it fails in the safe direction by re-asking the owner.
            requeued.append(key)
            continue

        scanned = {record.uuid: record.mod_date for record in cluster.records}
        # Compared through `mod_key` on both sides rather than as datetimes: the log
        # stores `None` as `''`, matching `cache._key`, and a comparison that mixed the
        # two conventions would report every unedited photo as changed — which is most of
        # the library — and requeue the entire review while looking careful.
        if any(mod_key(mark.mod_date) != mod_key(scanned.get(mark.photo_uuid)) for mark in decision.marks):
            requeued.append(key)
            continue

        applied[key] = decision

    resume_index = next((i for i, key in enumerate(fresh_keys) if key not in applied), None)
    return Resume(
        can_resume=True,
        applied=applied,
        orphaned=tuple(orphaned),
        voided=tuple(voided),
        requeued=tuple(requeued),
        superseded=superseded,
        resume_index=resume_index,
        resume_cluster_key=fresh_keys[resume_index] if resume_index is not None else None,
    )
