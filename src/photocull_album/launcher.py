"""photocull_album's engine room: build the keeper pool, sweep it into trips, open the
album store, bind a free loopback port, build the app.

Mirrors `photocull_review.launcher`'s load-bearing property -- **the port is bound
before the URL exists, not after** (`bind_socket` reads the real port off the socket it
just bound and listened on, so nothing can take it between "find a free one" and
"listen on it") -- but folds that package's `prepare()`/`serve()` two-step split into
one `launch()` call plus a `serve()` companion, since this package has no separate
CLI-wiring step at which to keep them apart. `launch(serve=False, ...)` is the test
seam: it builds everything and hands back a `Launch` without ever accepting a
connection, exactly like `photocull.cli.run_albums_list`'s `records=`/`keeper_uuids_set=`
seam lets tests supply synthetic input and assert no Photos library is opened.

Deliberately not ported from `photocull_review.launcher`: `SessionLifetime`, the idle
watchdog, and the `/api/shutdown` handshake. Those exist because a review sitting spans
days against a Full-Disk-Access process; a sequencing sitting is one browser tab
arranging a handful of trips, and `server.py`'s own docstring makes the same call for
the request-gate side of this same trade-off.
"""

from __future__ import annotations

import socket
import threading
import time
import webbrowser
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from photocull.album_store import AlbumStore
from photocull.albums import AlbumConfig, TripGroup, keeper_pool, trip_sweep
from photocull.models import PhotoRecord
from photocull.photos_source import iter_photo_records, keeper_uuids, open_library

from .server import build_app, new_token

#: The only address this app ever binds. A `0.0.0.0` bind would put this process --
#: which reads the keeper pool's uuids and serves its thumbnails -- on the LAN. Same
#: rule `photocull_review.launcher.LOOPBACK` states.
LOOPBACK = "127.0.0.1"


def _default_album_db_path() -> Path:
    """`~/.local/state/photocull/albums.db` -- the identical path
    `photocull.cli._default_album_db_path` returns, replicated here rather than
    imported so this package does not reach into the CLI module for a private helper,
    keeping `photocull_album` independent of `photocull.cli` the way it is already
    independent of `photocull_review`. Kept in sync by `tests/album/test_launcher.py`,
    which asserts the two literally agree."""
    return Path.home() / ".local" / "state" / "photocull" / "albums.db"


def bind_socket(host: str = LOOPBACK, port: int = 0) -> socket.socket:
    """A bound, listening loopback socket. Port 0 means "let the OS choose"."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(128)
    return sock


def launch_url(host: str, port: int, token: str) -> str:
    """The URL the frontend reads its bearer token from (`?t=`).

    Unlike `photocull_review`'s cookie handoff, this token is never traded away for a
    cookie: the frontend keeps it in page state and presents it as
    `Authorization: Bearer <token>` on every subsequent API call, per `api.py`'s
    per-request check.
    """
    return f"http://{host}:{port}/?t={token}"


@dataclass(frozen=True)
class Launch:
    """Everything one sequencing sitting needs, assembled and (once `serve()` has run)
    already offered to a browser."""

    url: str
    host: str
    port: int
    token: str
    app: Any
    store: AlbumStore
    pool_records: list[PhotoRecord]
    trip_groups: list[TripGroup]
    socket: socket.socket

    def close(self) -> None:
        """Release the port without having served. Safe to call twice."""
        self.socket.close()


def launch(
    *,
    records: Iterable[PhotoRecord] | None = None,
    keeper_uuids_set: set[str] | None = None,
    library_path: str | None = None,
    config: AlbumConfig | None = None,
    store: AlbumStore | None = None,
    db_path: Path | None = None,
    host: str = LOOPBACK,
    port: int = 0,
    open_browser: bool = True,
    serve: bool = True,
    console: Any | None = None,
) -> Launch:
    """Build the keeper pool, sweep it into trips, bind a port, build the app.

    `records`/`keeper_uuids_set` are the test seam: supply both and no Photos library
    is opened, mirroring `run_albums_list`'s own pattern. `store`/`db_path` are a
    second, independent seam: pass a store directly (e.g. one opened on `tmp_path`) or
    a path for one to be opened at; omitting both opens `_default_album_db_path()`, the
    same file `photocull albums export` reads.

    When `serve` is True (the production default) this also runs the blocking server
    before returning, closing the launch once it stops. When False it returns
    immediately with an app and a listening-but-unaccepted socket, which is what the
    test suite needs in order to drive the app with a `TestClient` and then release the
    port itself.
    """
    if records is None:
        db = open_library(library_path)
        # with_derivatives=True: unlike run_albums_list's pure report, this package
        # serves photo pixels (the crop overlay, /api/image/{uuid}) and needs
        # display_path populated -- without it every image route 404s.
        records = list(iter_photo_records(db, with_derivatives=True))
        keeper_uuids_set = keeper_uuids(db)

    pool = keeper_pool(records, keeper_uuids_set or set())
    groups = trip_sweep(pool, config or AlbumConfig())

    album_store = store if store is not None else AlbumStore(db_path or _default_album_db_path())
    token = new_token()
    sock = bind_socket(host, port)
    bound_host, bound_port = sock.getsockname()[:2]

    app = build_app(pool_records=pool, trip_groups=groups, store=album_store, token=token)

    result = Launch(
        url=launch_url(bound_host, bound_port, token),
        host=bound_host,
        port=bound_port,
        token=token,
        app=app,
        store=album_store,
        pool_records=pool,
        trip_groups=groups,
        socket=sock,
    )

    if console is not None:
        console.print(f"{len(pool)} keepers in the pool, {len(groups)} trips")
        console.print(f"open {result.url}")

    if serve:
        try:
            _serve_blocking(result, open_browser=open_browser)
        finally:
            result.close()

    return result


def _serve_blocking(session: Launch, *, open_browser: bool = False) -> None:
    """Run the server until interrupted. Blocks.

    Call on the main thread in production: uvicorn installs `SIGINT`/`SIGTERM`
    handlers only there, which is how Ctrl-C becomes a graceful stop rather than a
    traceback through an open SQLite connection.
    """
    import uvicorn

    config = uvicorn.Config(
        session.app, log_level="warning", access_log=False, lifespan="off"
    )
    server = uvicorn.Server(config)

    if open_browser:

        def _open_when_ready() -> None:
            while not server.started:
                time.sleep(0.05)
            webbrowser.open(session.url)

        threading.Thread(target=_open_when_ready, daemon=True).start()

    server.run(sockets=[session.socket])


#: Public alias, matching `photocull_review.launcher`'s `prepare()`/`serve()` naming
#: pair in shape: `launch()` builds and (optionally) runs; `serve()` names the part
#: that blocks, for a caller that already has a `Launch` and wants to run it directly.
serve = _serve_blocking
