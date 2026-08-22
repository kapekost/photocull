"""`photocull review`'s engine room: bind a port, build the app, decide when to stop.

Three decisions here are load-bearing enough to state rather than leave in the code.

**The port is bound before the URL exists, not after.** `build_app` is handed a token
but never a port, and `hostname_is_local` deliberately checks the hostname only
(`host-validation-is-by-name-not-by-port`) — so the launcher binds port 0 itself,
reads the real port off the bound socket, and hands that same socket to uvicorn. The
port in the URL is therefore a fact about a listening socket rather than an intention
that a later bind might not honour. It also closes the classic race: nothing can take
the port between "find a free one" and "listen on it", because those are one step.

**The session ends on an explicit end, on SIGINT, or on 30 minutes idle — never on
"the last tab closed".** A browser gives no reliable signal for the last of those
anyway, but the real reason is the shape of the job: this library produces 1,807
clusters, review is a multi-day sitting, and an accidental Cmd-W must not cost it. The
rule lives in `SessionLifetime`, a pure object driven by an injected clock, so the
30-minute boundary is pinned by a test in microseconds instead of waited out.

**Only requests that pass the gate count as activity.** `build_app` calls
`on_activity` from inside `_gated`, after all four checks — so a refused request
cannot hold the session open. Registering a timer as ordinary middleware would have
put it *outside* the gate (Starlette runs middleware in reverse registration order,
`the-request-gate-is-one-middleware-in-one-order`), which is the opposite of what an
idle timeout is for.

SIGINT needs no code here: uvicorn installs its own handlers when `Server.run` is
called on the main thread, which is exactly how `photocull review` runs it.
"""

from __future__ import annotations

import socket
import threading
import time
import webbrowser
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from photocull.config import ClusterConfig
from photocull.models import Cluster

from .api import ReviewSession
from .decisions import DecisionLog, default_db_path
from .demo import build_demo_clusters
from .images import mapping_source
from .resume import Resume, reconcile
from .server import build_app, new_token
from .sweep_writeback import plan_sweep
from .writeback import (
    PhotoScriptWriter,
    WritebackLedger,
    WritebackSettings,
    last_opened_library,
    run_writeback,
)

#: How long a review may sit untouched before the server stops. Long enough that
#: studying one cluster carefully is never a timeout; short enough that a forgotten
#: window does not hold a Full-Disk-Access process open indefinitely.
IDLE_TIMEOUT_SECONDS = 30 * 60

#: The only address this app ever binds. Not a default to be overridden — a
#: `0.0.0.0` bind would put a Full-Disk-Access process on the LAN, and the header
#: gate would not save it, since a non-browser client can send any `Host` it likes.
LOOPBACK = "127.0.0.1"

#: Re-exported so `photocull.cli` reaches the whole review package through this one
#: module, and therefore holds exactly one lazily-imported name.
demo_clusters = build_demo_clusters

__all__ = [
    "IDLE_TIMEOUT_SECONDS",
    "DecisionLog",
    "Launch",
    "PhotoScriptWriter",
    "ResumeBlocked",
    "SessionLifetime",
    "WritebackLedger",
    "WritebackSettings",
    "bind_socket",
    "default_db_path",
    "demo_clusters",
    "last_opened_library",
    "launch_url",
    "plan_sweep",
    "prepare",
    "run_writeback",
    "serve",
]


class ResumeBlocked(RuntimeError):
    """The stored decisions were recorded under different settings.

    Carries the `Resume` so a caller can report more than the message, and states both
    ways out in the message itself — a refusal that does not say what to do next just
    relocates the problem.
    """

    def __init__(self, resume: Resume):
        self.resume = resume
        super().__init__(
            f"{resume.blocked_reason}\n\n"
            "Decisions already recorded are matched to the settings they were made "
            "under, so nothing is lost either way:\n"
            "  * restore the previous settings and relaunch to carry on that session, or\n"
            "  * relaunch with --new-session to start a fresh one. The old decisions "
            "stay on disk and come back if you restore the settings."
        )


class SessionLifetime:
    """When this review sitting ends.

    Thread-safe because the watchdog thread reads it while the server thread writes
    it, and driven by `time.monotonic` so that a clock change or an NTP step cannot
    end a session early. On macOS `monotonic` also stops during system sleep, which is
    the behaviour worth having here: closing the lid overnight is not thirty minutes
    of the owner ignoring the review.
    """

    def __init__(
        self,
        *,
        idle_timeout: float = IDLE_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._clock = clock
        self._idle_timeout = idle_timeout
        self._lock = threading.Lock()
        self._last = clock()
        self._ended = False

    @property
    def ended(self) -> bool:
        with self._lock:
            return self._ended

    def touch(self) -> None:
        """Record activity.

        No guard against touching an ended session, because `expired()` checks
        `_ended` first and therefore cannot be talked out of it by a later touch. A
        guard here was written, then dropped: the mutation pass proved it changed no
        observable behaviour, and this project keeps only redundancy it can measure —
        the opposite conclusion to `append-only-is-enforced-by-two-mechanisms`, where
        the second mechanism caught a mutation the first did not.
        """
        with self._lock:
            self._last = self._clock()

    def end(self) -> None:
        with self._lock:
            self._ended = True

    def seconds_idle(self) -> float:
        with self._lock:
            return self._clock() - self._last

    def expired(self) -> bool:
        with self._lock:
            if self._ended:
                return True
            return (self._clock() - self._last) >= self._idle_timeout


def bind_socket(host: str = LOOPBACK, port: int = 0) -> socket.socket:
    """A bound, listening loopback socket. Port 0 means "let the OS choose"."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(128)
    return sock


def launch_url(host: str, port: int, token: str) -> str:
    """The one URL the owner opens.

    The token rides in the query string exactly once: `_handoff` trades it for an
    `HttpOnly` cookie and 303s to a clean path, so it never reaches the address bar's
    history or a `Referer` (`review-server-is-hardened-from-the-first-commit`).
    """
    return f"http://{host}:{port}/?t={token}"


@dataclass(frozen=True)
class Launch:
    """Everything one `photocull review` sitting needs, assembled but not yet serving."""

    url: str
    host: str
    port: int
    token: str
    app: Any
    session: ReviewSession
    resume: Resume
    lifetime: SessionLifetime
    socket: socket.socket

    def close(self) -> None:
        """Release the port without having served. Safe to call twice."""
        self.lifetime.end()
        self.socket.close()


def prepare(
    clusters: Sequence[Cluster],
    *,
    log: DecisionLog,
    config: ClusterConfig | None = None,
    source: str | Path | None = None,
    library_uuids: Collection[str] | None = None,
    force_new_session: bool = False,
    host: str = LOOPBACK,
    port: int = 0,
    lifetime: SessionLifetime | None = None,
    session_id: str | None = None,
    writeback: WritebackSettings | None = None,
) -> Launch:
    """Reconcile, build the session, bind the port. Serves nothing.

    Reconciliation runs **first** and refuses by raising, because a launch whose
    stored decisions cannot be applied is not a launch worth starting: the owner would
    review against a session that silently ignores everything already on disk.

    The image map is built from the clusters rather than from a fresh library scan.
    That is the difference between free and 33.7 s of startup — the scan that produced
    these clusters already resolved every display derivative onto the record
    (`pipeline-seam-is-the-derivative-path`), so asking the library again would be
    re-deriving what is already in hand.
    """
    resume = reconcile(
        clusters,
        log=log,
        config=config,
        source=source,
        library_uuids=library_uuids,
        force_new_session=force_new_session,
    )
    if not resume.can_resume:
        raise ResumeBlocked(resume)

    session = ReviewSession(
        clusters,
        log=log,
        config=config,
        source=source,
        session_id=session_id,
        writeback=writeback,
    )
    images = mapping_source(
        {
            record.uuid: Path(record.display_path)
            for cluster in clusters
            for record in cluster.records
            if record.display_path
        }
    )

    life = lifetime if lifetime is not None else SessionLifetime()
    token = new_token()
    sock = bind_socket(host, port)
    bound_host, bound_port = sock.getsockname()[:2]

    app = build_app(
        token=token,
        images=images,
        review=session,
        on_shutdown=life.end,
        on_activity=life.touch,
    )
    return Launch(
        url=launch_url(bound_host, bound_port, token),
        host=bound_host,
        port=bound_port,
        token=token,
        app=app,
        session=session,
        resume=resume,
        lifetime=life,
        socket=sock,
    )


def serve(
    launch: Launch,
    *,
    poll_interval: float = 0.25,
    on_ready: Callable[[Launch], None] | None = None,
    open_browser: bool = False,
) -> None:
    """Run the server until the session ends. Blocks.

    Call this on the main thread in production: uvicorn installs `SIGINT`/`SIGTERM`
    handlers only there, and that is how Ctrl-C becomes a graceful stop rather than a
    traceback through an open SQLite connection.

    The watchdog is what connects `SessionLifetime` to the socket. It also fires
    `on_ready` — including the browser open — once uvicorn reports itself started.

    That `server.started` gate is belt-and-braces rather than load-bearing, and the
    mutation pass is what showed it: dropping the check is the one mutant of this
    task's 21 that no test catches, because `prepare` already bound *and listened on*
    the socket, so a connection arriving before uvicorn accepts waits in the backlog
    and completes normally instead of being refused. Binding early is what makes the
    readiness race benign. The check stays because opening a browser 250 ms early
    buys nothing, not because removing it would break something.
    """
    import uvicorn

    config = uvicorn.Config(
        launch.app,
        log_level="warning",
        access_log=False,
        lifespan="off",
    )
    server = uvicorn.Server(config)
    stop = threading.Event()

    def watch() -> None:
        announced = False
        while not stop.wait(poll_interval):
            if not announced and server.started:
                announced = True
                if on_ready is not None:
                    on_ready(launch)
                if open_browser:
                    webbrowser.open(launch.url)
            if launch.lifetime.expired():
                server.should_exit = True
                return

    watcher = threading.Thread(
        target=watch, name="photocull-review-lifetime", daemon=True
    )
    watcher.start()
    try:
        server.run(sockets=[launch.socket])
    finally:
        stop.set()
        watcher.join(timeout=5)
        launch.socket.close()
