"""The seam between the bulk sweep's selectors and the write-back that already exists.

Nothing in `writeback.py` is edited or subclassed. `plan_writeback` iterates decisions
and marks and treats a cluster key as an opaque label, so a synthetic key per sweep
group is all it needs. The one real difference — the sweep wants an album per category
rather than one `Cull/Candidates` — is applied here, on the plan it returns.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from typing import Mapping, Sequence

from photocull.sweep import SweepGroup

from .decisions import Decision, Mark
from .session import DIGEST_CHARS
from .writeback import CULL, CULL_FOLDER, WritePlan, plan_writeback

#: The Photos folder every sweep album lives in — the same folder the review writes to,
#: so everything this app proposes for deletion is under one node in the sidebar.
#: Taken from `writeback.CULL_FOLDER` rather than restated, because `_album_location`
#: validates against that tuple and two spellings of "Cull" would diverge silently.
SWEEP_FOLDER = CULL_FOLDER[0]


def sweep_key(group: SweepGroup) -> str:
    """A stable identity for a sweep group, from its category and membership alone.

    Order-independent, and category-sensitive, for the same reason `session.cluster_key`
    is membership-derived: this is what a decision recorded on Monday is looked up by on
    Thursday, and the only thing that should invalidate it is the group genuinely
    holding different photos."""
    joined = "\n".join([group.category, *sorted(r.uuid for r in group.records)])
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:DIGEST_CHARS]
    return f"sweep:{group.category}:{digest}"


def sweep_decisions(
    groups: Sequence[SweepGroup], *, decided_at: datetime | None = None
) -> dict[str, Decision]:
    """One `Decision` per non-empty group, every mark `cull`.

    These are synthesised for `plan_writeback` and never stored — the sweep derives its
    selection from metadata on every run, so there is nothing to remember. `batch_id 0`
    and the empty `session_id`/`config_digest` say exactly that: they are the fields a
    stored decision would carry, and nothing here reads them.

    Empty groups are dropped rather than recorded: `decisions._validate` refuses a batch
    with no marks, and "this category matched nothing" is not a judgement about any
    photo."""
    when = decided_at or datetime.now(timezone.utc)
    out: dict[str, Decision] = {}
    for group in groups:
        if not group.records:
            continue
        key = sweep_key(group)
        out[key] = Decision(
            batch_id=0,
            session_id="",
            cluster_key=key,
            config_digest="",
            decided_at=when,
            marks=tuple(
                Mark(photo_uuid=r.uuid, mark=CULL, favorite=False, mod_date=r.mod_date)
                for r in group.records
            ),
        )
    return out


def plan_sweep(
    groups: Sequence[SweepGroup], live: Mapping[str, datetime | None]
) -> WritePlan:
    """Plan the sweep through the review's own planner, then redirect the albums.

    `plan_writeback` sends every cull to `Cull/Candidates`. The sweep keeps one album
    per category so the owner can select-all the ones they trust and scroll the ones
    they do not. Rewriting the destination here rather than parameterising
    `plan_writeback` keeps the module that mutates Photos free of caller-specific
    branching."""
    plan = plan_writeback(sweep_decisions(groups), live)

    by_uuid = {
        r.uuid: f"{SWEEP_FOLDER}/{g.album}" for g in groups for r in g.records
    }
    actions = tuple(
        replace(action, target=by_uuid[action.photo_uuid])
        if action.kind == "album" and action.photo_uuid in by_uuid
        else action
        for action in plan.actions
    )
    return replace(plan, actions=actions)
