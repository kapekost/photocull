"""Why the scorer ranked a take the way it did, in sentences rather than numbers.

Task 10 put ten numbered rows under each photograph and the owner's report did not
change: *"I don't understand what the values are... I'm not sure what I'm looking at."*
`sharpness 0.912` beside `sharpness 0.865` is correct, and it does not say that the left
photograph is the sharper one, by how much, or whether that is what decided the pick.
This module says it. No new measurement is taken — everything here is already in
`Cluster`; the task is phrasing, which is why it is the cheapest decision aid in the
phase and the only one that does not depend on how many pixels are on screen.

Three rules are about *not misleading*, and each is enforced by a test rather than by
care:

**Only a criterion the ranking actually used may be a reason.** `on_common_criteria`
re-totals every take over the criteria measured for *all* of them, so a criterion missing
from one take moved nothing. Citing it would describe a ranking that never happened, and
`horizon` is absent from 77.6% of real clusters, so this is the ordinary case.

**Only a weighted criterion may be a reason.** `facing`, `capture_quality` and `smiling`
ship at 0.0 (`unvalidated-signals-ship-at-zero-weight`) and are carried as diagnostics.
Their numbers are on screen already; a *reason* is a stronger claim than a number, and
the total does not support it.

**Only a difference the screen can show may be a reason.** The score table renders three
decimals, so a 0.001 gap is two identical rows with a sentence beside them claiming one
beat the other. That is emphasis with nothing behind it — the exact failure this panel
exists to fix.

The comparison partner is the runner-up for the proposed keeper and the keeper for
everyone else: "why did this win" and "why did this lose" are the two questions the
screen asks, and both are about the photograph a take's fate actually hinges on.

**What was measured at all is a separate function, on purpose.** The Phase 1b plan put
"ranked on exposure alone; no sharpness signal in this group" inside `explain`'s list.
It is read on the screen instead as `ranking_basis`, once, spanning both columns,
because it is a fact about the *cluster* and not about either photograph: returned per
take it renders as the same sentence at the foot of both panes, on the 90.3% of real
clusters that drop a criterion, costing two wrapped lines of frame height to say one
thing twice. The sentence itself is unchanged and nothing is dropped.
"""

from __future__ import annotations

from collections.abc import Sequence

from photocull.config import ClusterConfig
from photocull.models import Cluster, PhotoScore

#: Below this, two takes are not separated by a criterion in any way the owner can
#: check: `renderScores` prints three decimals, so anything under half of the last
#: printed digit renders as two identical rows. Deliberately a property of the display
#: rather than a taste threshold — if the score table ever prints more precision, this
#: number is what should move with it.
MIN_VISIBLE_DELTA = 0.005

#: How many reasons a pane may carry. Not a claim that the fourth criterion contributed
#: nothing — it is a claim about the screen: this panel sits in a viewport-height column
#: beside two photographs, and every line it takes comes out of the frames. A cluster
#: with all five criteria measured emitted five reasons plus a verdict, which is an audit
#: rather than an explanation. Nothing is hidden by the cap: the full score table sits
#: directly above, showing every criterion for both takes.
MAX_REASONS = 3

#: A sharpness ratio under this is a real difference and a small one. Measured spread
#: between takes inside the 8 real burst groups is 1.04-1.27x (see `cluster_sharpness`),
#: so "sharpest by 4%" is the common case on this library, not an edge case, and
#: reporting it in the same words as the one genuinely soft group (9.55x) would make the
#: sentence useless exactly where it is used most.
BARELY_SHARPER = 1.05

#: (better, worse) sentence templates per criterion, `{partner}` filled with what this
#: take is being compared against. Sharpness is absent on purpose: it is already a ratio
#: to the sharpest take in the cluster, so it is stated against the cluster rather than
#: against one partner (see `_sharpness_sentence`).
_TEMPLATES: dict[str, tuple[str, str]] = {
    "faces": (
        "Eyes are more open than in {partner}.",
        "Eyes are less open than in {partner}.",
    ),
    "exposure": (
        "Better exposed than {partner}.",
        "Less well exposed than {partner}.",
    ),
    "horizon": (
        "A more level horizon than {partner}.",
        "A more tilted horizon than {partner}.",
    ),
    "framing": (
        "The subject sits better in the frame than in {partner}.",
        "The subject sits less well in the frame than in {partner}.",
    ),
}


def _common_criteria(scores: Sequence[PhotoScore], cfg: ClusterConfig) -> list[str]:
    """The criteria that actually ranked this cluster, heaviest first.

    Deliberately re-derived from the sub-scores rather than read off
    `Cluster.dropped_criteria`: this is the same rule `on_common_criteria` applies and
    `pipeline._annotate` mirrors, and deriving it here means `explain` is correct for any
    cluster, including the hand-built ones the demo and the tests use. A test holds the
    two derivations together."""
    return sorted(
        (
            name
            for name, weight in cfg.weights.items()
            if weight > 0 and all(s.sub_scores.get(name) is not None for s in scores)
        ),
        key=lambda name: (-cfg.weights[name], name),
    )


def _prose_list(names: Sequence[str]) -> str:
    names = list(names)
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _sharpness_sentence(scores: Sequence[PhotoScore], uuid: str) -> str:
    """Sharpness said against the whole cluster, and always as a ratio.

    Laplacian variance has no absolute scale, so "sharpness 1153" is meaningless to a
    reader and not comparable between clusters; `cluster_sharpness` therefore already
    emits a fraction of the sharpest take, and the only honest rendering of a fraction is
    a fraction."""
    values = {s.uuid: float(s.sub_scores["sharpness"]) for s in scores}  # type: ignore[arg-type]
    mine = values[uuid]
    best = max(values.values())
    softest = min(values.values())
    if mine < best:
        return f"Softer than the sharpest take — {mine / best:.2f}× its detail."
    count = len(values)
    if softest <= 0:
        # A take measured at zero detail. The ratio is unbounded, and "infinitely
        # sharper" is not a thing to tell someone about a photograph.
        return f"Sharpest of the {count} takes — the softest has no measurable detail."
    # `best`, not `mine`, though they are equal here — the early return above leaves only
    # `mine == max(values)` reachable, so a mutation swapping them survives the suite and
    # is genuinely equivalent rather than untested. Written as `best` because the sentence
    # is a claim about the cluster's range, and it stays correct if this branch is ever
    # reached another way.
    ratio = best / softest
    if ratio < BARELY_SHARPER:
        return (
            f"Sharpest of the {count} takes, though only just — "
            f"{ratio:.2f}× the detail of the softest."
        )
    return f"Sharpest of the {count} takes — {ratio:.2f}× the detail of the softest."


def explain(
    cluster: Cluster, uuid: str, *, config: ClusterConfig | None = None
) -> list[str]:
    """Plain sentences saying why `uuid` sits where it does in `cluster`.

    Ordered so the decisive criterion is read first and the take's standing last. What
    the cluster could be ranked on at all is `ranking_basis`, for the reason in the
    module docstring. Returns `[]` when there is nothing to compare — a single-take
    cluster, or a uuid that is not a member."""
    cfg = config or ClusterConfig()
    scores = list(cluster.scores)
    by_uuid = {s.uuid: s for s in scores}
    mine = by_uuid.get(uuid)
    if mine is None or len(scores) < 2:
        return []

    ranked = sorted(scores, key=lambda s: (-s.total, s.uuid))
    winner = by_uuid.get(cluster.winner_uuid or "") or ranked[0]
    runner_up = next(s for s in ranked if s.uuid != winner.uuid)
    is_winner = mine.uuid == winner.uuid
    partner = runner_up if is_winner else winner
    partner_label = "the next-best take" if is_winner else "the proposed keeper"

    common = _common_criteria(scores, cfg)
    sentences: list[str] = []

    # --- the reasons, decisive first ------------------------------------------------
    deltas = {
        name: float(mine.sub_scores[name]) - float(partner.sub_scores[name])  # type: ignore[arg-type]
        for name in common
    }
    cited = [name for name in common if abs(deltas[name]) >= MIN_VISIBLE_DELTA]
    # Weight times movement, not movement alone: `exposure` at 0.15 moving 0.40 decided
    # more than `sharpness` at 0.35 moving 0.01, and the reader is being told which one
    # actually settled it.
    cited.sort(key=lambda name: (-abs(cfg.weights[name] * deltas[name]), name))
    for name in cited[:MAX_REASONS]:
        if name == "sharpness":
            sentences.append(_sharpness_sentence(scores, uuid))
            continue
        template = _TEMPLATES.get(name)
        if template is None:
            # A criterion the owner enabled in `photocull.toml` that has no phrasing
            # here. `unvalidated-signals-ship-at-zero-weight` makes that a value edit
            # rather than a code change, so this path is reachable by design and has to
            # degrade to something true rather than to silence.
            direction = "higher" if deltas[name] > 0 else "lower"
            sentences.append(
                f"Scores {direction} on {name} than {partner_label} "
                f"({mine.sub_scores[name]:.3f} against {partner.sub_scores[name]:.3f})."
            )
            continue
        better, worse = template
        sentences.append((better if deltas[name] > 0 else worse).format(partner=partner_label))

    if common and not cited:
        sentences.append(
            f"Nothing measured here separates this take from {partner_label}."
        )

    # --- standing --------------------------------------------------------------------
    top_gap = winner.total - runner_up.total
    if cluster.is_ambiguous and mine.uuid in (winner.uuid, runner_up.uuid):
        # No winner is asserted. `ambiguous-clusters-go-to-manual-pick`: the scorer
        # cannot separate these, and prose that reads as a verdict here would contradict
        # the close-call note sitting two lines above it.
        #
        # It says what *this* take's margin is and stops there. The first version
        # appended "so this proposal is a coin flip rather than a verdict" — advice the
        # close-call annotation already gives in the pane directly above, and being a
        # cluster fact it came out identical in both panes. Read on screen, an ambiguous
        # cluster said "coin flip" three times, which is the whole library's most common
        # cluster shape at 40.1%. The margin itself is what the annotation does not carry.
        side = "ahead of the next take" if is_winner else "behind the proposed keeper"
        sentences.append(f"Too close to call — only {top_gap:.3f} {side}.")
    elif is_winner:
        sentences.append(f"The scorer's pick, {top_gap:.3f} ahead of the next take.")
    else:
        sentences.append(
            f"Behind the proposed keeper by {winner.total - mine.total:.3f}."
        )

    return sentences


def ranking_basis(cluster: Cluster, *, config: ClusterConfig | None = None) -> str | None:
    """What this cluster could be ranked on at all, or `None` when everything applied.

    `annotations` demoted `dropped_criteria` to this panel deliberately: it is non-empty
    for 90.3% of real clusters, which makes it the background condition of the library
    rather than a warning about one cluster. As prose it is useful — "ranked on sharpness
    and exposure alone" tells the owner why a pick rests on less than they expected — and
    as a warning triangle on nine clusters in ten it is worth nothing.

    Re-derived from the sub-scores rather than read off `Cluster.dropped_criteria`, for
    the reason in `_common_criteria`."""
    cfg = config or ClusterConfig()
    scores = list(cluster.scores)
    if not scores:
        return None
    common = _common_criteria(scores, cfg)
    dropped = sorted(
        name for name, weight in cfg.weights.items() if weight > 0 and name not in common
    )
    if not common:
        return (
            "Nothing measurable ranked these takes — no criterion could be measured for "
            "every one of them, so the order between them is arbitrary."
        )
    if not dropped:
        return None
    # No "alone": it is right when the ranking rests on one criterion and wrong when it
    # lists four of five, and both shapes are common here.
    return (
        f"Ranked on {_prose_list(common)} — "
        f"{_prose_list(dropped)} could not be measured for every take."
    )
