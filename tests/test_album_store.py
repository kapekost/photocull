from __future__ import annotations

import pytest

from photocull.album_store import AlbumStore


@pytest.fixture
def store(tmp_path):
    return AlbumStore(tmp_path / "albums.db")


def test_create_returns_a_record_with_the_given_sequence(store):
    record = store.create("Tahoe 2026", "4x6", ["a", "b", "c"])
    assert record.title == "Tahoe 2026"
    assert record.print_size == "4x6"
    assert record.sequence == ("a", "b", "c")
    assert record.album_id


def test_get_returns_none_for_an_unknown_id(store):
    assert store.get("does-not-exist") is None


def test_get_round_trips_a_created_album(store):
    created = store.create("Tahoe 2026", "4x6", ["a", "b"])
    fetched = store.get(created.album_id)
    assert fetched == created


def test_save_sequence_updates_the_order(store):
    created = store.create("Tahoe 2026", "4x6", ["a", "b", "c"])
    store.save_sequence(created.album_id, ["c", "a", "b"])
    fetched = store.get(created.album_id)
    assert fetched.sequence == ("c", "a", "b")


def test_save_sequence_on_an_unknown_id_raises(store):
    with pytest.raises(KeyError):
        store.save_sequence("does-not-exist", ["a"])


def test_list_returns_every_album_newest_first(store):
    first = store.create("First", "4x6", ["a"])
    second = store.create("Second", "5x7", ["b"])
    listed = store.list()
    assert [r.album_id for r in listed] == [second.album_id, first.album_id]


def test_create_rejects_an_unknown_print_size(store):
    with pytest.raises(ValueError):
        store.create("Bad", "8x10", ["a"])


# --- crop offsets (Task 10 Step 4) --------------------------------------------------


def test_get_crop_offset_defaults_to_centered_when_never_set(store):
    created = store.create("Tahoe 2026", "4x6", ["a"])
    assert store.get_crop_offset(created.album_id, "a") == (0.5, 0.5)


def test_set_then_get_crop_offset_round_trips(store):
    created = store.create("Tahoe 2026", "4x6", ["a"])
    store.set_crop_offset(created.album_id, "a", 0.2, 0.9)
    assert store.get_crop_offset(created.album_id, "a") == (0.2, 0.9)


def test_crop_offsets_are_independent_per_uuid(store):
    created = store.create("Tahoe 2026", "4x6", ["a", "b"])
    store.set_crop_offset(created.album_id, "a", 0.1, 0.1)
    assert store.get_crop_offset(created.album_id, "b") == (0.5, 0.5)


def test_crop_offsets_are_independent_per_album(store):
    first = store.create("First", "4x6", ["a"])
    second = store.create("Second", "4x6", ["a"])
    store.set_crop_offset(first.album_id, "a", 0.1, 0.1)
    assert store.get_crop_offset(second.album_id, "a") == (0.5, 0.5)


def test_set_crop_offset_on_an_unknown_album_raises(store):
    with pytest.raises(KeyError):
        store.set_crop_offset("does-not-exist", "a", 0.1, 0.1)


def test_a_preexisting_db_file_without_the_crop_column_is_migrated(tmp_path):
    """A real `~/.local/state/photocull/albums.db` may already exist from Tasks 7/9's
    testing, created before `crop_offsets_json` existed. Opening `AlbumStore` again
    must add the column rather than assume `CREATE TABLE IF NOT EXISTS` ran fresh."""
    import sqlite3

    db_path = tmp_path / "albums.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT NOT NULL, "
        "print_size TEXT NOT NULL, sequence_json TEXT NOT NULL, created_at TEXT NOT NULL, "
        "updated_at TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO albums VALUES (?, ?, ?, ?, ?, ?)",
        ("old-id", "Old Album", "4x6", '["a"]', "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    store = AlbumStore(db_path)
    assert store.get_crop_offset("old-id", "a") == (0.5, 0.5)
    store.set_crop_offset("old-id", "a", 0.3, 0.7)
    assert store.get_crop_offset("old-id", "a") == (0.3, 0.7)
    # The pre-existing row survived the migration untouched.
    assert store.get("old-id").title == "Old Album"
