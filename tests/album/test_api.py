"""Task 9: the sequencing API — trip listing, album creation, and reordering.

Every mutating endpoint validates uuids against the keeper pool this launch was built
with (see `api.py`'s own docstring): the same "never trust client input past a lookup"
posture `photocull_review`'s decisions endpoints use.

`_check_token` compares the presented bearer token with plain `!=` rather than
`secrets.compare_digest`, unlike `photocull_review.server.token_matches`. That is a
deliberate scope cut, not an oversight: this task's own instructions rule out porting
`photocull_review`'s heavier session apparatus (cookie handoff, idle timeout, constant-
time comparison) into a package built for a single short sequencing sitting rather than
a multi-day review. See docs/plans/2026-08-19-phase-3-album-builder.md Task 9.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from photocull.album_store import AlbumStore
from photocull.albums import TripGroup
from photocull_album.api import build_app
from tests.fixtures import make_photo_record


def _client(tmp_path, pool_records, groups=None):
    store = AlbumStore(tmp_path / "albums.db")
    app = build_app(
        pool_records=pool_records,
        trip_groups=groups or [],
        store=store,
        token="test-token",
    )
    client = TestClient(app)
    client.headers.update({"Authorization": "Bearer test-token"})
    return client, store


def test_list_trips_returns_the_supplied_groups(tmp_path):
    records = [make_photo_record(uuid="a"), make_photo_record(uuid="b")]
    group = TripGroup(title="Tahoe", records=tuple(records), wide_spread=False)
    client, _ = _client(tmp_path, records, groups=[group])
    resp = client.get("/api/trips")
    assert resp.status_code == 200
    assert resp.json() == [{"title": "Tahoe", "count": 2, "wide_spread": False, "uuids": ["a", "b"]}]


def test_missing_token_is_refused(tmp_path):
    records = [make_photo_record(uuid="a")]
    store = AlbumStore(tmp_path / "albums.db")
    app = build_app(pool_records=records, trip_groups=[], store=store, token="test-token")
    client = TestClient(app)
    resp = client.get("/api/trips")
    assert resp.status_code == 401


def test_a_wrong_token_is_refused(tmp_path):
    """Distinct from "no token at all": `_check_token` must reject a presented-but-
    wrong bearer credential, not just an absent one."""
    records = [make_photo_record(uuid="a")]
    store = AlbumStore(tmp_path / "albums.db")
    app = build_app(pool_records=records, trip_groups=[], store=store, token="test-token")
    client = TestClient(app)
    client.headers.update({"Authorization": "Bearer wrong-token"})
    resp = client.get("/api/trips")
    assert resp.status_code == 401


def test_create_album_then_reorder_round_trips(tmp_path):
    records = [make_photo_record(uuid="a"), make_photo_record(uuid="b")]
    client, store = _client(tmp_path, records)
    created = client.post("/api/albums", json={"title": "T", "print_size": "4x6", "uuids": ["a", "b"]})
    assert created.status_code == 200
    album_id = created.json()["album_id"]

    reordered = client.put(f"/api/albums/{album_id}/sequence", json={"uuids": ["b", "a"]})
    assert reordered.status_code == 200
    assert store.get(album_id).sequence == ("b", "a")


def test_reorder_unknown_album_is_404(tmp_path):
    client, _ = _client(tmp_path, [make_photo_record(uuid="a")])
    resp = client.put("/api/albums/does-not-exist/sequence", json={"uuids": ["a"]})
    assert resp.status_code == 404


def test_create_album_rejects_a_uuid_outside_the_pool(tmp_path):
    records = [make_photo_record(uuid="a")]
    client, _ = _client(tmp_path, records)
    resp = client.post("/api/albums", json={"title": "T", "print_size": "4x6", "uuids": ["a", "not-in-pool"]})
    assert resp.status_code == 400


def test_create_album_rejects_an_unknown_print_size(tmp_path):
    """`AlbumStore.create` raises `ValueError` for a print size outside `PRINT_SIZES`;
    the API must translate that into a 400, not a 500."""
    records = [make_photo_record(uuid="a")]
    client, _ = _client(tmp_path, records)
    resp = client.post("/api/albums", json={"title": "T", "print_size": "8x10", "uuids": ["a"]})
    assert resp.status_code == 400


def test_reorder_rejects_a_uuid_outside_the_pool(tmp_path):
    records = [make_photo_record(uuid="a"), make_photo_record(uuid="b")]
    client, _ = _client(tmp_path, records)
    created = client.post("/api/albums", json={"title": "T", "print_size": "4x6", "uuids": ["a", "b"]})
    album_id = created.json()["album_id"]

    resp = client.put(f"/api/albums/{album_id}/sequence", json={"uuids": ["a", "not-in-pool"]})
    assert resp.status_code == 400


# --- GET /api/albums/{id} (Task 10 Step 9's own read-endpoint need) -----------------


def test_get_album_returns_the_current_sequence(tmp_path):
    records = [make_photo_record(uuid="a"), make_photo_record(uuid="b")]
    client, store = _client(tmp_path, records)
    created = client.post("/api/albums", json={"title": "T", "print_size": "4x6", "uuids": ["a", "b"]})
    album_id = created.json()["album_id"]
    client.put(f"/api/albums/{album_id}/sequence", json={"uuids": ["b", "a"]})

    resp = client.get(f"/api/albums/{album_id}")
    assert resp.status_code == 200
    assert resp.json() == {"album_id": album_id, "title": "T", "print_size": "4x6", "sequence": ["b", "a"]}


def test_get_album_unknown_id_is_404(tmp_path):
    client, _ = _client(tmp_path, [make_photo_record(uuid="a")])
    resp = client.get("/api/albums/does-not-exist")
    assert resp.status_code == 404


# --- crop preview + crop offset persistence (Task 10 Step 4) ------------------------


def test_crop_preview_reports_no_crop_needed_when_ratio_matches(tmp_path):
    records = [make_photo_record(uuid="a", width=1800, height=1200)]
    client, _ = _client(tmp_path, records)
    resp = client.get("/api/crop-preview/a", params={"print_size": "4x6"})
    assert resp.status_code == 200
    assert resp.json() == {"needs_crop": False, "offset_x": 0.5, "offset_y": 0.5}


def test_crop_preview_reports_crop_needed_for_a_non_matching_ratio(tmp_path):
    records = [make_photo_record(uuid="a", width=1200, height=900)]
    client, _ = _client(tmp_path, records)
    resp = client.get("/api/crop-preview/a", params={"print_size": "4x6"})
    assert resp.status_code == 200
    assert resp.json()["needs_crop"] is True


def test_crop_preview_unknown_uuid_is_404(tmp_path):
    client, _ = _client(tmp_path, [make_photo_record(uuid="a", width=1800, height=1200)])
    resp = client.get("/api/crop-preview/not-in-pool", params={"print_size": "4x6"})
    assert resp.status_code == 404


def test_crop_preview_unknown_print_size_is_400(tmp_path):
    client, _ = _client(tmp_path, [make_photo_record(uuid="a", width=1800, height=1200)])
    resp = client.get("/api/crop-preview/a", params={"print_size": "8x10"})
    assert resp.status_code == 400


def test_set_crop_offset_round_trips_through_the_store(tmp_path):
    records = [make_photo_record(uuid="a", width=1200, height=900)]
    client, store = _client(tmp_path, records)
    created = client.post("/api/albums", json={"title": "T", "print_size": "4x6", "uuids": ["a"]})
    album_id = created.json()["album_id"]

    resp = client.put(f"/api/albums/{album_id}/crop/a", json={"offset_x": 0.2, "offset_y": 0.9})
    assert resp.status_code == 200
    assert store.get_crop_offset(album_id, "a") == (0.2, 0.9)


def test_set_crop_offset_unknown_album_is_404(tmp_path):
    client, _ = _client(tmp_path, [make_photo_record(uuid="a", width=1200, height=900)])
    resp = client.put("/api/albums/does-not-exist/crop/a", json={"offset_x": 0.1, "offset_y": 0.1})
    assert resp.status_code == 404


def test_set_crop_offset_rejects_a_uuid_outside_the_pool(tmp_path):
    records = [make_photo_record(uuid="a", width=1200, height=900)]
    client, _ = _client(tmp_path, records)
    created = client.post("/api/albums", json={"title": "T", "print_size": "4x6", "uuids": ["a"]})
    album_id = created.json()["album_id"]

    resp = client.put(f"/api/albums/{album_id}/crop/not-in-pool", json={"offset_x": 0.1, "offset_y": 0.1})
    assert resp.status_code == 400
