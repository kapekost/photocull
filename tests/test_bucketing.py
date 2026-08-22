"""Tests for clustering stage 1 -- the cheap capture-time + burst pass.

Determinism is a first-class requirement here, not a nicety: `PhotosDB.photos()`
ordering varies across processes (DECISIONS.md `within-burst-distance-band-corrected`)
and 798 records in the real library share a timestamp with another record, so any
bucketing that leans on input order produces different output run to run. Task 8
exports a calibration sample from these buckets for the owner to review, which is
worthless if it reshuffles on every run.
"""

from datetime import datetime, timedelta

from photocull.bucketing import bucket_records
from photocull.config import ClusterConfig
from tests.fixtures import make_photo_record

BASE = datetime(2025, 6, 1, 12, 0, 0)


def rec(offset_seconds, **kw):
    return make_photo_record(date=BASE + timedelta(seconds=offset_seconds), **kw)


def test_empty_input_gives_no_buckets():
    assert bucket_records([]) == []


def test_items_within_the_gap_share_a_bucket():
    out = bucket_records([rec(0, uuid="a"), rec(10, uuid="b")])
    assert [[r.uuid for r in b] for b in out] == [["a", "b"]]


def test_items_beyond_the_gap_split():
    out = bucket_records([rec(0, uuid="a"), rec(600, uuid="b")])
    assert [[r.uuid for r in b] for b in out] == [["a"], ["b"]]


def test_chaining_is_transitive():
    out = bucket_records([rec(0, uuid="a"), rec(60, uuid="b"), rec(120, uuid="c")])
    assert [[r.uuid for r in b] for b in out] == [["a", "b", "c"]]


def test_input_order_does_not_matter():
    out = bucket_records([rec(120, uuid="c"), rec(0, uuid="a"), rec(60, uuid="b")])
    assert [[r.uuid for r in b] for b in out] == [["a", "b", "c"]]


def test_burst_siblings_join_even_across_a_large_time_gap():
    """A burst is by definition one moment; if its timestamps somehow spread past
    the gap, the burst_key still wins."""
    out = bucket_records(
        [
            rec(0, uuid="a", burst_key="B1"),
            rec(9999, uuid="b", burst_key="B1"),
        ]
    )
    assert [sorted(r.uuid for r in b) for b in out] == [["a", "b"]]


def test_different_burst_keys_do_not_join():
    out = bucket_records(
        [
            rec(0, uuid="a", burst_key="B1"),
            rec(9999, uuid="b", burst_key="B2"),
        ]
    )
    assert len(out) == 2


def test_records_without_a_date_are_dropped():
    out = bucket_records([rec(0, uuid="a"), make_photo_record(uuid="b", date=None)])
    assert [[r.uuid for r in b] for b in out] == [["a"]]


def test_gap_is_configurable():
    cfg = ClusterConfig(gap_seconds=5.0)
    out = bucket_records([rec(0, uuid="a"), rec(10, uuid="b")], config=cfg)
    assert len(out) == 2


def test_every_record_appears_exactly_once():
    records = [rec(i * 30, uuid=f"u{i}") for i in range(10)]
    out = bucket_records(records)
    flat = [r.uuid for b in out for r in b]
    assert sorted(flat) == sorted(r.uuid for r in records)


def test_burst_keys_merge_transitively_through_a_shared_bucket():
    """Two burst groups that meet inside one time bucket must all end up together.

    `a` and `c` share B1; `b` and `d` share B2; `c` and `d` are close enough in time
    to bucket together, which transitively joins all four. A merge pass that stops
    after the first key of a bucket leaves `b` stranded in its own bucket while `d`
    -- its own burst sibling -- sits somewhere else."""
    out = bucket_records(
        [
            rec(0, uuid="a", burst_key="B1"),
            rec(1000, uuid="b", burst_key="B2"),
            rec(2000, uuid="c", burst_key="B1"),
            rec(2010, uuid="d", burst_key="B2"),
        ]
    )
    assert [sorted(r.uuid for r in b) for b in out] == [["a", "b", "c", "d"]]


def test_records_sharing_a_burst_key_are_never_split_across_buckets():
    """The invariant the previous test is a specific instance of."""
    out = bucket_records(
        [
            rec(0, uuid="a", burst_key="B1"),
            rec(1000, uuid="b", burst_key="B2"),
            rec(2000, uuid="c", burst_key="B1"),
            rec(2010, uuid="d", burst_key="B2"),
        ]
    )
    seen: dict[str, int] = {}
    for index, bucket in enumerate(out):
        for r in bucket:
            if r.burst_key is not None:
                assert seen.setdefault(r.burst_key, index) == index


def test_buckets_are_ordered_by_earliest_date():
    out = bucket_records([rec(5000, uuid="late"), rec(0, uuid="early")])
    assert [b[0].uuid for b in out] == ["early", "late"]


def test_records_within_a_bucket_are_ordered_by_date():
    out = bucket_records([rec(20, uuid="c"), rec(0, uuid="a"), rec(10, uuid="b")])
    assert [r.uuid for r in out[0]] == ["a", "b", "c"]


def test_identical_timestamps_are_ordered_deterministically():
    """798 real records share a timestamp with another record, and the upstream
    ordering from PhotosDB.photos() varies across processes, so a stable sort on
    date alone leaks that variation into the output."""
    forwards = bucket_records([rec(0, uuid="z"), rec(0, uuid="a")])
    backwards = bucket_records([rec(0, uuid="a"), rec(0, uuid="z")])
    assert [[r.uuid for r in b] for b in forwards] == [["a", "z"]]
    assert [[r.uuid for r in b] for b in backwards] == [["a", "z"]]
