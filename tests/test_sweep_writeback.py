from datetime import datetime

from photocull.sweep import SweepGroup
from photocull_review.sweep_writeback import plan_sweep, sweep_decisions, sweep_key
from tests.fixtures import make_photo_record

MOD = datetime(2026, 1, 1, 9, 0, 0)


def group(*records, category="screenshots", album="Screenshots"):
    return SweepGroup(category, album, "because", tuple(records))


def test_every_swept_photo_is_marked_cull():
    a = make_photo_record(mod_date=MOD)
    decisions = sweep_decisions([group(a)])
    (decision,) = decisions.values()
    assert decision.marks_by_uuid() == {a.uuid: "cull"}


def test_nothing_is_ever_marked_keep_or_favorite():
    """The sweep proposes deletions only. A keeper is what it does NOT select."""
    a = make_photo_record(mod_date=MOD)
    (decision,) = sweep_decisions([group(a)]).values()
    assert all(m.mark == "cull" and not m.favorite for m in decision.marks)


def test_the_key_is_stable_for_the_same_membership():
    a, b = make_photo_record(), make_photo_record()
    assert sweep_key(group(a, b)) == sweep_key(group(b, a))


def test_the_key_changes_when_the_membership_changes():
    a, b = make_photo_record(), make_photo_record()
    assert sweep_key(group(a)) != sweep_key(group(a, b))


def test_the_key_changes_with_the_category():
    a = make_photo_record()
    assert sweep_key(group(a, category="screenshots")) != sweep_key(
        group(a, category="short-videos")
    )


def test_an_empty_group_records_no_decision():
    """`DecisionLog.record` refuses a batch with no marks; an empty category must not
    reach it."""
    assert sweep_decisions([group()]) == {}


def test_each_category_lands_in_its_own_album():
    a = make_photo_record(mod_date=MOD)
    b = make_photo_record(mod_date=MOD)
    groups = [
        group(a, category="screenshots", album="Screenshots"),
        group(b, category="short-videos", album="Short Videos"),
    ]
    plan = plan_sweep(groups, {a.uuid: MOD, b.uuid: MOD})
    albums = {act.photo_uuid: act.target for act in plan.actions if act.kind == "album"}
    assert albums == {a.uuid: "Cull/Screenshots", b.uuid: "Cull/Short Videos"}


def test_every_album_the_sweep_plans_can_actually_be_resolved():
    """The test that would have caught the `_album_location` allowlist. Planning an
    album `run_writeback` cannot resolve is green everywhere until the live run."""
    from photocull_review.writeback import _album_location

    a = make_photo_record(mod_date=MOD)
    plan = plan_sweep([group(a)], {a.uuid: MOD})
    for act in plan.actions:
        if act.kind == "album":
            folder, name = _album_location(act.target)
            assert folder == ("Cull",)
            assert "/" not in name


def test_every_swept_photo_gets_the_cull_keyword():
    a = make_photo_record(mod_date=MOD)
    plan = plan_sweep([group(a)], {a.uuid: MOD})
    keywords = [act.target for act in plan.actions if act.kind == "keyword"]
    assert keywords == ["cull-candidate"]


def test_no_favorite_action_is_ever_planned():
    a = make_photo_record(mod_date=MOD)
    plan = plan_sweep([group(a)], {a.uuid: MOD})
    assert [act for act in plan.actions if act.kind == "favorite"] == []
    assert plan.favorites == ()


def test_a_photo_edited_since_selection_is_skipped():
    """The same re-verification the review's write-back does: the sweep may be minutes
    old, but it may also be days old."""
    a = make_photo_record(mod_date=MOD)
    plan = plan_sweep([group(a)], {a.uuid: datetime(2026, 6, 1, 9, 0, 0)})
    assert plan.actions == ()
    assert [s.reason for s in plan.skipped] == ["edited"]


def test_a_photo_gone_from_the_library_is_skipped():
    a = make_photo_record(mod_date=MOD)
    plan = plan_sweep([group(a)], {})
    assert plan.actions == ()
    assert [s.reason for s in plan.skipped] == ["gone"]
