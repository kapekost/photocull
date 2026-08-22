"""Task 10: `photocull_album.server.build_app` -- the frontend shell, its static
assets, and the `/api/image/{uuid}` route `api.py` (Task 9) deliberately left unwired
(see `images.py`'s own docstring and `server.py`'s module docstring for why)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from photocull.album_store import AlbumStore
from photocull_album.server import build_app
from tests.fixtures import make_photo_record


def _client(tmp_path, pool_records, groups=None, token="test-token"):
    store = AlbumStore(tmp_path / "albums.db")
    app = build_app(pool_records=pool_records, trip_groups=groups or [], store=store, token=token)
    return TestClient(app), store


def test_shell_serves_the_index_page(tmp_path):
    client, _ = _client(tmp_path, [])
    resp = client.get("/")
    assert resp.status_code == 200
    assert "photocull album" in resp.text


def test_static_asset_serves_app_js(tmp_path):
    client, _ = _client(tmp_path, [])
    resp = client.get("/static/app.js")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/javascript")


def test_static_asset_serves_style_css(tmp_path):
    client, _ = _client(tmp_path, [])
    resp = client.get("/static/style.css")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/css")


def test_unknown_static_asset_is_404(tmp_path):
    client, _ = _client(tmp_path, [])
    resp = client.get("/static/does-not-exist.js")
    assert resp.status_code == 404


def test_traversal_through_static_name_is_404(tmp_path):
    """`{name}`, never `{name:path}` -- a slash-bearing spelling must miss the router
    (or the allowlist) rather than escape `STATIC_DIR`."""
    client, _ = _client(tmp_path, [])
    resp = client.get("/static/..%2f..%2fpyproject.toml")
    assert resp.status_code == 404


def test_image_route_serves_a_known_uuid_with_a_valid_token(tmp_path):
    image_path = tmp_path / "a.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nnot a real png but has the right magic")
    record = make_photo_record(uuid="a", display_path=str(image_path))
    client, _ = _client(tmp_path, [record], token="tok")
    resp = client.get("/api/image/a", params={"t": "tok"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"


def test_image_route_without_the_token_is_401(tmp_path):
    image_path = tmp_path / "a.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    record = make_photo_record(uuid="a", display_path=str(image_path))
    client, _ = _client(tmp_path, [record], token="tok")
    resp = client.get("/api/image/a")
    assert resp.status_code == 401


def test_image_route_with_the_wrong_token_is_401(tmp_path):
    image_path = tmp_path / "a.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    record = make_photo_record(uuid="a", display_path=str(image_path))
    client, _ = _client(tmp_path, [record], token="tok")
    resp = client.get("/api/image/a", params={"t": "wrong"})
    assert resp.status_code == 401


def test_image_route_unknown_uuid_is_404(tmp_path):
    client, _ = _client(tmp_path, [], token="tok")
    resp = client.get("/api/image/not-in-pool", params={"t": "tok"})
    assert resp.status_code == 404


def test_image_route_a_record_with_no_display_path_is_never_in_the_map(tmp_path):
    """A record whose `display_path` is `None` (no local derivative) must not appear
    in the image map at all -- `mapping_source` is built only from records that have
    one, same filter `photocull_review.launcher` already applies."""
    record = make_photo_record(uuid="a", display_path=None)
    client, _ = _client(tmp_path, [record], token="tok")
    resp = client.get("/api/image/a", params={"t": "tok"})
    assert resp.status_code == 404
