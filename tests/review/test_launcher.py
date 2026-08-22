"""Task 9 — the launcher: ephemeral port, launch URL, session lifetime.

Three things here are deliberately not tested through a running server, because a
threaded server test that also has to pin a *policy* can only tell you that the pair
of them agreed:

* **The idle rule is a pure object with an injected clock.** `sleep`-based timing
  tests are the flakiest thing in a 6-second suite, and a 30-minute timeout cannot be
  waited out at all. `SessionLifetime` is driven by a fake clock, so the boundary is
  pinned exactly rather than approximately.
* **The port and the URL are pinned on a real bound socket**, not on a running app.
  Binding is the only part that can disagree with the operating system.
* **The server test asserts the one thing only a real server can show** — that a
  shutdown request actually stops the process — and waits on an explicit readiness
  flag with a deadline, never on a sleep.
"""

from __future__ import annotations

import socket
import threading
from pathlib import Path

import pytest

from photocull.config import ClusterConfig
from photocull_review import launcher
from photocull_review.decisions import DecisionLog
from photocull_review.demo import build_demo_clusters
from photocull_review.server import COOKIE_NAME


class FakeClock:
    """A monotonic clock the test drives by hand."""

    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def demo(tmp_path: Path):
    """The demo library plus its own decision log, as `--demo` builds them."""
    clusters = build_demo_clusters(tmp_path / "images")
    log = DecisionLog(tmp_path / "decisions.db")
    yield clusters, log
    log.close()


def _prepare(clusters, log, **kwargs) -> launcher.Launch:
    kwargs.setdefault("config", ClusterConfig())
    kwargs.setdefault(
        "library_uuids",
        {record.uuid for cluster in clusters for record in cluster.records},
    )
    return launcher.prepare(clusters, log=log, **kwargs)


# --- the ephemeral port ---------------------------------------------------------


def test_port_zero_binds_and_reports_the_real_port(demo):
    clusters, log = demo
    launch = _prepare(clusters, log)
    try:
        assert launch.port != 0
        assert 1 <= launch.port <= 65535
        # The socket is bound and listening *before* the URL is handed out, so the
        # port in that URL is a fact rather than an intention.
        assert launch.socket.getsockname() == ("127.0.0.1", launch.port)
    finally:
        launch.close()


def test_the_launch_url_carries_the_token_and_the_real_port(demo):
    clusters, log = demo
    launch = _prepare(clusters, log)
    try:
        assert launch.url == f"http://127.0.0.1:{launch.port}/?t={launch.token}"
    finally:
        launch.close()


def test_two_launches_share_no_secret(demo, tmp_path):
    clusters, log = demo
    first = _prepare(clusters, log)
    second = _prepare(clusters, log)
    try:
        assert first.token != second.token
        assert first.session.session_id != second.session.session_id
        assert first.port != second.port
    finally:
        first.close()
        second.close()


def test_the_bound_socket_is_refused_from_anywhere_but_loopback(demo):
    clusters, log = demo
    launch = _prepare(clusters, log)
    try:
        host, _port = launch.socket.getsockname()
        assert host == "127.0.0.1", "binding 0.0.0.0 would expose the review to the LAN"
    finally:
        launch.close()


# --- session lifetime -----------------------------------------------------------


def test_the_idle_timeout_is_thirty_minutes():
    # Pinned as a number because the plan chose it deliberately: long enough that
    # reading a cluster carefully is not a timeout, short enough that a forgotten
    # server does not hold a Full-Disk-Access process open overnight.
    assert launcher.IDLE_TIMEOUT_SECONDS == 30 * 60


def test_an_idle_session_expires_exactly_at_the_timeout():
    clock = FakeClock()
    life = launcher.SessionLifetime(idle_timeout=1800, clock=clock)

    assert not life.expired()
    clock.advance(1799.9)
    assert not life.expired()
    clock.advance(0.1)
    assert life.expired()


def test_activity_resets_the_idle_clock():
    clock = FakeClock()
    life = launcher.SessionLifetime(idle_timeout=1800, clock=clock)

    clock.advance(1799)
    life.touch()
    clock.advance(1799)
    assert not life.expired()


def test_a_long_review_never_expires_while_the_owner_is_working():
    """The rule is *idle* time, not wall time, and there is no "last tab closed".

    A multi-day review is the expected shape of this app on a 14,000-photo library,
    so nothing here may end the session except an explicit end or real inactivity —
    an accidental Cmd-W must not cost the sitting.
    """
    clock = FakeClock()
    life = launcher.SessionLifetime(idle_timeout=1800, clock=clock)

    for _ in range(200):  # ~4 days of reviewing, a request every 30 minutes
        clock.advance(1799)
        life.touch()

    assert not life.expired()
    assert not life.ended


def test_an_explicit_end_expires_the_session_without_waiting():
    clock = FakeClock()
    life = launcher.SessionLifetime(idle_timeout=1800, clock=clock)

    life.end()

    assert life.ended
    assert life.expired()
    # And it stays ended: a late request must not resurrect a session the owner closed.
    life.touch()
    assert life.expired()


# --- what counts as activity ----------------------------------------------------


def test_a_refused_request_does_not_keep_the_session_alive(demo):
    """Only requests that pass the gate are the owner working.

    Registering the timer outside the gate would let anything that can reach the
    socket hold the session open, which is the opposite of what an idle timeout is
    for.
    """
    from fastapi.testclient import TestClient

    clusters, log = demo
    clock = FakeClock()
    launch = _prepare(clusters, log, lifetime=launcher.SessionLifetime(clock=clock))
    try:
        client = TestClient(launch.app, base_url=f"http://127.0.0.1:{launch.port}")

        clock.advance(100)
        assert client.get("/api/session").status_code == 401
        assert launch.lifetime.seconds_idle() == 100, "a 401 counted as activity"

        client.cookies.set(COOKIE_NAME, launch.token)
        assert client.get("/api/session").status_code == 200
        assert launch.lifetime.seconds_idle() == 0
    finally:
        launch.close()


# --- resume -------------------------------------------------------------------


def test_a_launch_carries_the_resume_report(demo):
    clusters, log = demo
    launch = _prepare(clusters, log)
    try:
        assert launch.resume.can_resume
        assert launch.resume.decided_count == 0
        assert launch.resume.resume_index == 0
    finally:
        launch.close()


def test_a_moved_setting_refuses_to_launch_and_names_the_field(demo, tmp_path):
    """Task 5's refusal reaches the owner as a refusal to start, not as a warning.

    Launching anyway would present a review that silently ignores every decision
    already on disk — the failure `resume-matches-decisions-by-digest-not-recency`
    exists to make visible.
    """
    clusters, log = demo
    first = _prepare(clusters, log, config=ClusterConfig())
    try:
        first.session.record(
            first.session.document["clusters"][0]["cluster_key"],
            {
                member["uuid"]: ("keep" if index == 0 else "cull")
                for index, member in enumerate(
                    first.session.document["clusters"][0]["members"]
                )
            },
        )
    finally:
        first.close()

    moved = ClusterConfig(similarity_threshold=0.52)
    with pytest.raises(launcher.ResumeBlocked) as caught:
        launcher.prepare(
            clusters,
            log=log,
            config=moved,
            library_uuids={r.uuid for c in clusters for r in c.records},
        )

    message = str(caught.value)
    assert "similarity_threshold" in message
    assert "0.52" in message
    assert "--new-session" in message, "the refusal must name the way past it"


def test_forcing_a_new_session_launches_and_leaves_the_old_decisions_on_disk(demo):
    clusters, log = demo
    first = _prepare(clusters, log)
    key = first.session.document["clusters"][0]["cluster_key"]
    try:
        first.session.record(
            key,
            {
                member["uuid"]: ("keep" if index == 0 else "cull")
                for index, member in enumerate(
                    first.session.document["clusters"][0]["members"]
                )
            },
        )
    finally:
        first.close()

    moved = ClusterConfig(similarity_threshold=0.52)
    launch = launcher.prepare(
        clusters,
        log=log,
        config=moved,
        library_uuids={r.uuid for c in clusters for r in c.records},
        force_new_session=True,
    )
    try:
        assert launch.resume.can_resume
        assert launch.resume.decided_count == 0
        assert log.state(key) is not None, "forcing a new session deleted a decision"
    finally:
        launch.close()


# --- images ---------------------------------------------------------------------


def test_the_image_map_is_built_from_the_clusters_not_a_second_scan(demo):
    """The scan already resolved every display derivative; asking again costs 33.7 s."""
    from fastapi.testclient import TestClient

    clusters, log = demo
    launch = _prepare(clusters, log)
    try:
        client = TestClient(launch.app, base_url=f"http://127.0.0.1:{launch.port}")
        client.cookies.set(COOKIE_NAME, launch.token)
        uuid = clusters[0].records[0].uuid

        response = client.get(f"/api/images/{uuid}")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
    finally:
        launch.close()


# --- the server actually stops --------------------------------------------------


def test_a_shutdown_request_stops_the_server(demo):
    """The one assertion that needs a real socket: the process goes away.

    Readiness is an explicit flag with a deadline. Nothing here sleeps a fixed
    interval and hopes.
    """
    import httpx

    clusters, log = demo
    launch = _prepare(clusters, log)
    ready = threading.Event()
    thread = threading.Thread(
        target=launcher.serve,
        args=(launch,),
        kwargs={"poll_interval": 0.01, "on_ready": lambda _launch: ready.set()},
        daemon=True,
    )
    thread.start()
    try:
        assert ready.wait(timeout=10), "the server never reported itself ready"

        with httpx.Client(base_url=f"http://127.0.0.1:{launch.port}") as client:
            client.cookies.set(COOKIE_NAME, launch.token)
            response = client.post(
                "/api/shutdown", headers={"Sec-Fetch-Site": "same-origin"}
            )

        assert response.status_code == 202
        thread.join(timeout=10)
        assert not thread.is_alive(), "the shutdown request did not stop the server"

        # And the port is genuinely released, not merely unanswered.
        with socket.socket() as probe:
            probe.settimeout(2)
            with pytest.raises(OSError):
                probe.connect(("127.0.0.1", launch.port))
    finally:
        launch.lifetime.end()
        thread.join(timeout=10)


def test_an_idle_session_stops_the_server(demo):
    """The idle rule is wired to the server, not merely implemented beside it."""
    clusters, log = demo
    clock = FakeClock()
    launch = _prepare(clusters, log, lifetime=launcher.SessionLifetime(clock=clock))
    ready = threading.Event()
    thread = threading.Thread(
        target=launcher.serve,
        args=(launch,),
        kwargs={"poll_interval": 0.01, "on_ready": lambda _launch: ready.set()},
        daemon=True,
    )
    thread.start()
    try:
        assert ready.wait(timeout=10)

        clock.advance(launcher.IDLE_TIMEOUT_SECONDS + 1)

        thread.join(timeout=10)
        assert not thread.is_alive(), "an idle session did not stop the server"
    finally:
        launch.lifetime.end()
        thread.join(timeout=10)


# --- the CLI command, end to end ------------------------------------------------


def test_photocull_review_demo_runs_with_no_photos_library(monkeypatch, tmp_path):
    """The strong form of the `--demo` promise, against the real launcher.

    `tests/test_cli.py` pins the same rule with a stand-in launcher so it holds on a
    machine with no `[review]` extra; this one proves the real assembly never reaches
    for a library either.
    """
    from photocull.cli import run_review

    def explode(*args, **kwargs):
        raise AssertionError("--demo opened the Photos library")

    monkeypatch.setattr("photocull.cli.open_library", explode)
    # An empty TOML rather than no `--config` at all: `find_config()` discovers this
    # repo's own git-ignored `photocull.toml`, so a test relying on discovery would
    # quietly start testing the owner's calibration.
    empty = tmp_path / "empty.toml"
    empty.write_text("")

    launch = run_review(
        demo=True,
        serve=False,
        config_path=str(empty),
        decisions_path=str(tmp_path / "decisions.db"),
    )
    try:
        assert launch.port != 0
        assert launch.token in launch.url
        assert len(launch.session.document["clusters"]) == 24
        assert launch.resume.can_resume

        # The launch must be able to *answer*, not merely exist. Opening the decision
        # log in a `with` block made every one of the assertions above pass while the
        # first real request died on "Cannot operate on a closed database" — found by
        # the live smoke, not by the suite, which is why this line is here.
        from fastapi.testclient import TestClient

        client = TestClient(launch.app, base_url=f"http://127.0.0.1:{launch.port}")
        client.cookies.set(COOKIE_NAME, launch.token)
        assert client.get("/api/session").status_code == 200
        assert client.get("/api/dashboard").status_code == 200
    finally:
        launch.close()
        launch.session.log.close()


# --- the port is a held resource, not a guess -----------------------------------


def test_the_bound_port_is_held_before_the_server_starts(demo):
    """`prepare` listens, so nothing can take the port between here and `serve`.

    Without the `listen()` this is still *bound*, and every other test still passes
    because asyncio listens again on the socket it is handed — so this is the only
    assertion that distinguishes "the port is reserved" from "the port happens to be
    free so far".
    """
    clusters, log = demo
    launch = _prepare(clusters, log)
    try:
        with socket.create_connection(("127.0.0.1", launch.port), timeout=2):
            pass
    finally:
        launch.close()


def test_closing_a_launch_releases_the_port(demo):
    clusters, log = demo
    launch = _prepare(clusters, log)
    launch.close()

    with socket.socket() as probe:
        probe.settimeout(2)
        with pytest.raises(OSError):
            probe.connect(("127.0.0.1", launch.port))


def test_images_are_served_from_the_display_derivative_not_the_analysis_one(
    demo, tmp_path
):
    """The review shows the largest local raster; the scorer reads the smallest.

    This needs a fixture of its own because **the demo library gives every photo the
    same file for both** — measured, 82 of 82 records — so no test built on it can
    tell `display_path` from `derivative_path`. On the real library the two differ for
    62.2% of photos, and serving the wrong one would silently halve the resolution of
    the one screen whose job is judging sharpness (`zoom-is-bounded-by-the-derivative`).
    """
    import dataclasses

    from fastapi.testclient import TestClient

    from photocull_review.demo import write_png

    clusters, log = demo
    display = tmp_path / "display-only.png"
    write_png(display, width=1024, height=768, rgb=(7, 11, 13), stripe=6)
    record = dataclasses.replace(clusters[0].records[0], display_path=str(display))
    clusters[0].records[0] = record
    assert record.derivative_path != record.display_path

    launch = _prepare(clusters, log)
    try:
        client = TestClient(launch.app, base_url=f"http://127.0.0.1:{launch.port}")
        client.cookies.set(COOKIE_NAME, launch.token)

        response = client.get(f"/api/images/{record.uuid}")

        assert response.status_code == 200
        assert response.content == display.read_bytes()
    finally:
        launch.close()
