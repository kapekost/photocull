from datetime import datetime, timedelta

from photocull.cluster_estimate import count_near_duplicate_clusters

from .fixtures import make_photo_record


def at(seconds_offset: float) -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0) + timedelta(seconds=seconds_offset)


def test_empty_list_has_no_clusters():
    assert count_near_duplicate_clusters([]) == 0


def test_single_record_is_not_a_cluster():
    records = [make_photo_record(date=at(0))]
    assert count_near_duplicate_clusters(records) == 0


def test_two_records_within_gap_form_one_cluster():
    records = [make_photo_record(date=at(0)), make_photo_record(date=at(10))]
    assert count_near_duplicate_clusters(records, gap_seconds=90) == 1


def test_two_records_at_exactly_the_gap_do_not_cluster():
    # spec says "gaps UNDER 90s" -- exactly 90s apart is not a near-duplicate
    records = [make_photo_record(date=at(0)), make_photo_record(date=at(90))]
    assert count_near_duplicate_clusters(records, gap_seconds=90) == 0


def test_two_records_beyond_gap_do_not_cluster():
    records = [make_photo_record(date=at(0)), make_photo_record(date=at(200))]
    assert count_near_duplicate_clusters(records, gap_seconds=90) == 0


def test_transitive_chain_forms_one_cluster_even_if_ends_are_far_apart():
    # A-B 40s apart, B-C 40s apart -> chained into one cluster of 3, even though A-C is 80s
    # which is itself still < 90s here, so also directly test a chain where the ends alone
    # would NOT qualify: A-B 60s, B-C 60s -> A-C is 120s (> 90s) but still one chained cluster.
    records = [
        make_photo_record(date=at(0)),
        make_photo_record(date=at(60)),
        make_photo_record(date=at(120)),
    ]
    assert count_near_duplicate_clusters(records, gap_seconds=90) == 1


def test_two_separate_clusters_are_counted_separately():
    records = [
        make_photo_record(date=at(0)),
        make_photo_record(date=at(10)),
        make_photo_record(date=at(1000)),
        make_photo_record(date=at(1010)),
    ]
    assert count_near_duplicate_clusters(records, gap_seconds=90) == 2


def test_singletons_between_clusters_are_not_counted():
    records = [
        make_photo_record(date=at(0)),
        make_photo_record(date=at(10)),  # cluster of 2 with the one above
        make_photo_record(date=at(5000)),  # lone singleton
        make_photo_record(date=at(9000)),
        make_photo_record(date=at(9010)),  # cluster of 2 with the one above
    ]
    assert count_near_duplicate_clusters(records, gap_seconds=90) == 2


def test_records_with_no_date_are_ignored():
    records = [
        make_photo_record(date=None),
        make_photo_record(date=at(0)),
        make_photo_record(date=at(10)),
        make_photo_record(date=None),
    ]
    assert count_near_duplicate_clusters(records, gap_seconds=90) == 1


def test_unsorted_input_is_handled():
    records = [
        make_photo_record(date=at(10)),
        make_photo_record(date=at(0)),
    ]
    assert count_near_duplicate_clusters(records, gap_seconds=90) == 1


def test_gap_seconds_is_configurable():
    records = [make_photo_record(date=at(0)), make_photo_record(date=at(50))]
    assert count_near_duplicate_clusters(records, gap_seconds=90) == 1
    assert count_near_duplicate_clusters(records, gap_seconds=30) == 0
