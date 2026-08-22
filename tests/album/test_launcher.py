"""Task 9 — the launcher: ephemeral port, launch URL, and building the app with no
Photos library.

Mirrors `tests/review/test_launcher.py`'s two load-bearing properties, trimmed to what
this package actually has:

* **The port and the URL are pinned on a real bound socket**, not a running app.
  Binding is the only part that can disagree with the operating system.
* **`launch(serve=False, records=..., keeper_uuids_set=...)` never opens a real
  library** when synthetic records are supplied -- the same seam
  `photocull.cli.run_albums_list`'s own tests already pin -- and the app it builds
  answers `/api/trips` with the expected groups.

No live-server test here: `photocull_review`'s equivalent needs one to prove its
`/api/shutdown` handshake and idle watchdog actually stop a running process, and Task 9
carries neither of those (see `launcher.py`'s own module docstring on why) -- there is
nothing live-only left to prove that `TestClient` against `launch(serve=False, ...)`'s
app does not already cover.
"""

from __future__ import annotations

import socket

import pytest
from fastapi.testclient import TestClient

from photocull.album_store import AlbumStore
from photocull_album import launcher
from tests.fixtures import make_photo_record


def _pool(count: int = 3):
    records = [make_photo_record(uuid=f"u{i}") for i in range(count)]
    return records, {r.uuid for r in records}


# --- the ephemeral port ---------------------------------------------------------


def test_port_zero_binds_and_reports_the_real_port(tmp_path):
    records, keepers = _pool()
    result = launcher.launch(
        records=records, keeper_uuids_set=keepers, serve=False, db_path=tmp_path / "albums.db"
    )
    try:
        assert result.port != 0
        assert 1 <= result.port <= 65535
        # The socket is bound and listening *before* the URL is handed out, so the
        # port in that URL is a fact rather than an intention.
        assert result.socket.getsockname() == ("127.0.0.1", result.port)
    finally:
        result.close()


def test_the_launch_url_carries_the_token_and_the_real_port(tmp_path):
    records, keepers = _pool()
    result = launcher.launch(
        records=records, keeper_uuids_set=keepers, serve=False, db_path=tmp_path / "albums.db"
    )
    try:
        assert result.url == f"http://127.0.0.1:{result.port}/?t={result.token}"
    finally:
        result.close()


def test_two_launches_share_no_secret(tmp_path):
    records, keepers = _pool()
    first = launcher.launch(
        records=records, keeper_uuids_set=keepers, serve=False, db_path=tmp_path / "a.db"
    )
    second = launcher.launch(
        records=records, keeper_uuids_set=keepers, serve=False, db_path=tmp_path / "b.db"
    )
    try:
        assert first.token != second.token
        assert first.port != second.port
    finally:
        first.close()
        second.close()


def test_the_bound_socket_is_refused_from_anywhere_but_loopback(tmp_path):
    records, keepers = _pool()
    result = launcher.launch(
        records=records, keeper_uuids_set=keepers, serve=False, db_path=tmp_path / "albums.db"
    )
    try:
        host, _port = result.socket.getsockname()
        assert host == "127.0.0.1", "binding 0.0.0.0 would expose the sequencer to the LAN"
    finally:
        result.close()


def test_closing_a_launch_releases_the_port(tmp_path):
    records, keepers = _pool()
    result = launcher.launch(
        records=records, keeper_uuids_set=keepers, serve=False, db_path=tmp_path / "albums.db"
    )
    result.close()

    with socket.socket() as probe:
        probe.settimeout(2)
        with pytest.raises(OSError):
            probe.connect(("127.0.0.1", result.port))


# --- no library is opened when records are supplied ------------------------------


def test_launch_never_opens_a_library_when_records_are_supplied(monkeypatch, tmp_path):
    def explode(*args, **kwargs):
        raise AssertionError("launch() opened the Photos library")

    monkeypatch.setattr("photocull_album.launcher.open_library", explode)
    records, keepers = _pool()

    result = launcher.launch(
        records=records, keeper_uuids_set=keepers, serve=False, db_path=tmp_path / "albums.db"
    )
    try:
        assert result.port != 0
    finally:
        result.close()


def test_launch_reads_the_real_library_with_derivatives_so_pixels_are_servable(
    monkeypatch, tmp_path
):
    """Regression: launch() once called `iter_photo_records(db, with_derivatives=False)`
    on the real-library branch (records=None), matching run_albums_list's pure-report
    seam it was copied from. That left every PhotoRecord's display_path unset, so
    /api/image/{uuid} 404'd on every real photo -- invisible to every other test here,
    which all supply `records=` directly and never exercise this branch. Found live
    against the real library at Task 11 (docs/plans/2026-08-19-phase-3-album-builder.md)."""
    captured = {}

    def fake_iter_photo_records(db, *, with_derivatives=False):
        captured["with_derivatives"] = with_derivatives
        records, _ = _pool()
        return records

    monkeypatch.setattr("photocull_album.launcher.open_library", lambda path: object())
    monkeypatch.setattr("photocull_album.launcher.iter_photo_records", fake_iter_photo_records)
    monkeypatch.setattr("photocull_album.launcher.keeper_uuids", lambda db: {"u0", "u1", "u2"})

    result = launcher.launch(serve=False, db_path=tmp_path / "albums.db")
    try:
        assert captured["with_derivatives"] is True
    finally:
        result.close()


def test_launch_builds_an_app_that_answers_trips(tmp_path):
    records, keepers = _pool()

    result = launcher.launch(
        records=records, keeper_uuids_set=keepers, serve=False, db_path=tmp_path / "albums.db"
    )
    try:
        client = TestClient(result.app)
        client.headers.update({"Authorization": f"Bearer {result.token}"})

        resp = client.get("/api/trips")

        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["count"] == 3
        assert set(body[0]["uuids"]) == keepers
    finally:
        result.close()


def test_launch_pool_excludes_a_record_that_is_neither_a_keeper_nor_a_favorite(tmp_path):
    keeper = make_photo_record(uuid="k1")
    other = make_photo_record(uuid="not-a-keeper", is_favorite=False)

    result = launcher.launch(
        records=[keeper, other],
        keeper_uuids_set={"k1"},
        serve=False,
        db_path=tmp_path / "albums.db",
    )
    try:
        assert [r.uuid for r in result.pool_records] == ["k1"]
    finally:
        result.close()


# --- the album store --------------------------------------------------------------


def test_launch_uses_a_supplied_store_directly(tmp_path):
    records, keepers = _pool()
    store = AlbumStore(tmp_path / "custom.db")

    result = launcher.launch(records=records, keeper_uuids_set=keepers, serve=False, store=store)
    try:
        assert result.store is store
    finally:
        result.close()


def test_default_album_db_path_matches_the_cli_helper():
    """The web UI and `photocull albums export` must share one store -- pinned by
    literal path equality, never by touching the real path (which lives under the
    owner's home directory and must not be created as a side effect of running the
    suite)."""
    from photocull.cli import _default_album_db_path as cli_default

    assert launcher._default_album_db_path() == cli_default()
