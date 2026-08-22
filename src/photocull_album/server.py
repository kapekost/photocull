"""photocull_album's HTTP surface: token minting, plus (from Task 10) the frontend's
static assets and the one photo-pixel route `api.py` deliberately left unwired.

`new_token()` is the one piece of `photocull_review/server.py`'s hardened gate this
package borrows, unchanged. Everything else there (host validation, the cookie
handoff, `Sec-Fetch-Site` enforcement, session-lifetime tracking) is deliberately not
ported: an album-sequencing sitting is one browser tab, one launch, short by design,
and `api.py`'s per-request `Authorization: Bearer` check is what such a sitting
actually needs (docs/plans/2026-08-19-phase-3-album-builder.md Task 9's own scope
note).

**Why `/api/image/{uuid}` checks `?t=` instead of an `Authorization` header, unlike
every other route this package serves.** A plain `<img src>` cannot carry a custom
header, so the bearer-token convention every mutating/listing endpoint uses does not
reach this one route at all -- the browser issues that request itself, with whatever
the `src` attribute names and nothing else. The token therefore travels the same way
the page's own launch URL already carries it (`?t=`), which `app.js` mirrors onto
every `<img src>` it builds. Left unauthenticated entirely was the simpler option, but
this process holds Full Disk Access and this route serves the pixels of real photos
by uuid, so it gets the same "never trust reach without the token" posture as
everything else in this package, at the one-line cost query-param checking is here."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, PlainTextResponse

from photocull.album_store import AlbumStore
from photocull.albums import TripGroup

from . import api as api_module
from .images import mapping_source, resolve_image

#: Where the frontend lives. Plain files, no build step -- same convention
#: `photocull_review/server.py`'s `STATIC_DIR` already established.
STATIC_DIR = Path(__file__).parent / "static"

#: The shell, served at `/`.
SHELL_FILE = "index.html"

#: Asset name -> media type. An allowlist for the same structural reason
#: `photocull_review.server.STATIC_FILES` is one: the requested name is only ever a
#: dictionary key, never joined to `STATIC_DIR` before it has matched an entry here.
STATIC_FILES: dict[str, str] = {
    "app.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
}

#: What `/api/image/{uuid}` answers for each `resolve_image` outcome. Mirrors
#: `photocull_review.server.IMAGE_REFUSALS`'s wording, retargeted to this package's
#: own vocabulary ("this album's pool" rather than "this review session").
IMAGE_REFUSALS: dict[int, str] = {
    404: "No such photo in this album's pool.",
    410: "That photo's local copy is gone -- Photos evicts and regenerates derivatives.",
    415: "That file is not an image this server will serve.",
}


def new_token() -> str:
    """A per-launch bearer token: 256 bits, URL-safe, ASCII by construction."""
    return secrets.token_urlsafe(32)


def build_app(
    *,
    pool_records: list[Any],
    trip_groups: list[TripGroup],
    store: AlbumStore,
    token: str,
) -> FastAPI:
    """`api.py`'s `/api/*` routes (trips, albums, sequencing, crop offsets), plus the
    frontend shell/assets and the image route Task 9 left for this task to add."""
    app = api_module.build_app(
        pool_records=pool_records, trip_groups=trip_groups, store=store, token=token
    )
    images = mapping_source(
        {
            record.uuid: Path(record.display_path)
            for record in pool_records
            if record.display_path
        }
    )

    @app.get("/", response_class=HTMLResponse)
    async def shell() -> HTMLResponse:
        # Read per request, not cached at import: the file is a few kilobytes on
        # local disk, and this is one short sitting, not a long-running process
        # where that would matter -- same call `photocull_review.server.shell` makes.
        return HTMLResponse((STATIC_DIR / SHELL_FILE).read_text(encoding="utf-8"))

    @app.get("/static/{name}")
    async def static_asset(name: str) -> Response:
        """Serve one frontend file, by name, from the allowlist and nowhere else.

        `{name}`, never `{name:path}` -- that converter admits `/`, which is what
        keeps a traversal spelling 404ing at the router before this function runs.
        """
        media_type = STATIC_FILES.get(name)
        if media_type is None:
            return PlainTextResponse("No such asset.", status_code=404)
        return Response((STATIC_DIR / name).read_bytes(), media_type=media_type)

    @app.get("/api/image/{uuid}")
    async def image(uuid: str, t: str | None = None) -> Response:
        """Serve one photo's pixels. See this module's docstring for why the token
        arrives as `?t=` here instead of the `Authorization` header every other route
        checks -- a plain `<img src>` cannot set that header."""
        if t != token:
            return PlainTextResponse("invalid or missing token", status_code=401)
        outcome = resolve_image(images, uuid)
        if outcome.status != 200 or outcome.path is None:
            return PlainTextResponse(
                IMAGE_REFUSALS[outcome.status], status_code=outcome.status
            )
        return Response(outcome.path.read_bytes(), media_type=outcome.media_type)

    return app
