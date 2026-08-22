from datetime import datetime

import pytest

from photocull.cache import AnalysisCache

D1 = datetime(2025, 1, 1, 12, 0, 0)
D2 = datetime(2025, 6, 1, 12, 0, 0)

# The analysis key is the selected derivative path -- it identifies the exact raster a
# cached vector was computed from. These are the two real size classes.
K1 = "/derivatives/u1_480x360.jpeg"
K2 = "/derivatives/u1_1024x768.jpeg"


@pytest.fixture
def cache(tmp_path):
    c = AnalysisCache(tmp_path / "analysis.db")
    yield c
    c.close()


def test_miss_returns_none(cache):
    assert cache.get_feature_print("nope", D1, K1) is None


def test_round_trips_a_feature_print(cache):
    vec = [0.1, 0.2, 0.3]
    cache.put_feature_print("u1", D1, K1, vec)
    got = cache.get_feature_print("u1", D1, K1)
    assert got == pytest.approx(vec)


def test_a_changed_mod_date_invalidates_the_entry(cache):
    cache.put_feature_print("u1", D1, K1, [0.1, 0.2])
    assert cache.get_feature_print("u1", D2, K1) is None


def test_reput_overwrites_rather_than_duplicating(cache):
    cache.put_feature_print("u1", D1, K1, [0.1])
    cache.put_feature_print("u1", D1, K1, [0.9])
    assert cache.get_feature_print("u1", D1, K1) == pytest.approx([0.9])


def test_round_trips_scores(cache):
    scores = {"sharpness": 0.8, "faces": 0.5, "total": 0.7}
    cache.put_scores("u1", D1, K1, scores)
    assert cache.get_scores("u1", D1, K1) == pytest.approx(scores)


def test_survives_reopening_the_file(tmp_path):
    path = tmp_path / "analysis.db"
    c1 = AnalysisCache(path)
    c1.put_feature_print("u1", D1, K1, [0.5, 0.6])
    c1.close()
    c2 = AnalysisCache(path)
    assert c2.get_feature_print("u1", D1, K1) == pytest.approx([0.5, 0.6])
    c2.close()


def test_none_mod_date_is_usable_as_a_key(cache):
    cache.put_feature_print("u1", None, K1, [0.4])
    assert cache.get_feature_print("u1", None, K1) == pytest.approx([0.4])


def test_scores_miss_returns_none(cache):
    assert cache.get_scores("u1", D1, K1) is None


def test_works_as_a_context_manager(tmp_path):
    path = tmp_path / "analysis.db"
    with AnalysisCache(path) as c:
        c.put_scores("u1", D1, K1, {"total": 0.3})
    with AnalysisCache(path) as c:
        assert c.get_scores("u1", D1, K1) == pytest.approx({"total": 0.3})


def test_a_full_length_vision_vector_round_trips(cache):
    """768 float32s is the real payload size -- see DECISIONS.md
    `vision-verified-works-no-phash-fallback`."""
    vec = [i / 768 for i in range(768)]
    cache.put_feature_print("u1", D1, K1, vec)
    assert cache.get_feature_print("u1", D1, K1) == pytest.approx(vec)


def test_a_different_derivative_is_a_different_entry(cache):
    """The whole point. Verified live before this was written: a vector cached from a
    photo's LARGE derivative was served for the same photo under the new smallest-first
    selection, 0.4416 away from the correct answer, with no error anywhere."""
    cache.put_feature_print("u1", D1, K2, [0.1, 0.2])
    assert cache.get_feature_print("u1", D1, K1) is None


def test_both_derivatives_can_be_cached_at_once(cache):
    cache.put_feature_print("u1", D1, K1, [0.1, 0.2])
    cache.put_feature_print("u1", D1, K2, [0.9, 0.9])
    assert cache.get_feature_print("u1", D1, K1) == pytest.approx([0.1, 0.2])
    assert cache.get_feature_print("u1", D1, K2) == pytest.approx([0.9, 0.9])


def test_scores_are_keyed_by_derivative_too(cache):
    cache.put_scores("u1", D1, K2, {"sharpness": 0.5})
    assert cache.get_scores("u1", D1, K1) is None


def test_an_old_schema_file_is_discarded_rather_than_read(tmp_path):
    """A cache written before this task has no derivative column, so every row in it
    was computed under the largest-derivative selection. Reusing it is precisely the
    stale-hit failure above, so the file is rebuilt instead."""
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        "CREATE TABLE feature_prints (uuid TEXT, mod_date TEXT, vector BLOB);"
        "INSERT INTO feature_prints VALUES ('u1', '', x'0000');"
    )
    conn.commit()
    conn.close()

    with AnalysisCache(path) as c:
        assert c.get_feature_print("u1", D1, K1) is None
        c.put_feature_print("u1", D1, K1, [0.1, 0.2])
        assert c.get_feature_print("u1", D1, K1) == pytest.approx([0.1, 0.2])


def test_a_current_schema_file_survives_reopening(tmp_path):
    """The version check must not throw away a VALID cache -- that would silently turn
    every run into a cold 12-minute Vision pass."""
    path = tmp_path / "keep.db"
    with AnalysisCache(path) as c:
        c.put_feature_print("u1", D1, K1, [0.1, 0.2])
    with AnalysisCache(path) as c:
        assert c.get_feature_print("u1", D1, K1) == pytest.approx([0.1, 0.2])
