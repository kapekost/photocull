"""The review server's core: an app that is safe before it is useful.

This process holds Full Disk Access and, from Task 14, the ability to mutate a real
Photos library. Every hardening measure below is therefore in the first server commit
rather than in a later pass, per DECISIONS.md
`review-server-is-hardened-from-the-first-commit`.

The gate runs four checks, **in this order**, on every request:

1. **`Host` must be a loopback authority** → else `421 Misdirected Request`. This, not
   CORS, is what stops DNS rebinding: a page on any website can point its own hostname
   at `127.0.0.1` and reach a server here, and the browser considers the result
   same-origin, so no preflight ever happens and no CORS policy is consulted. The only
   thing that distinguishes such a request from a real one is the authority it names.
2. **`GET /?t=<token>`** sets an `HttpOnly`, `SameSite=Strict` cookie and **303**s to
   the same path without the token, so the secret leaves the address bar, the history
   and any screenshot the owner takes of their own review session.
3. **Every other request needs that cookie** → else `401`.
4. **Anything that is not a read needs `Sec-Fetch-Site: same-origin`** → else `403`.

Then every response — refusals included — carries `no-store`, `nosniff`,
`no-referrer` and a `default-src 'self'` CSP.

Two implementation notes that are load-bearing rather than stylistic:

- **The four checks live in one middleware, not four.** Starlette runs middleware in
  the *reverse* of registration order (measured: the first `@app.middleware("http")`
  registered is the innermost). Four stacked middlewares written in the order above
  would fire bottom-up, and the order is not cosmetic — it decides whether an
  unauthenticated request to a foreign authority answers 421 or 401. One function
  makes the order readable, and `test_the_host_check_runs_before_the_cookie_check`
  pins it by outcome.
- **Tokens are compared as bytes.** `secrets.compare_digest` raises `TypeError` on a
  non-ASCII `str` (measured), which would turn a mistyped URL into a 500 and a
  traceback.

Architecture: DECISIONS.md `review-ui-is-a-local-web-app`.
"""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response

from .api import ReviewSession, add_review_routes
from .images import ImageSource, resolve_image

#: What each refusal says. Prose, and deliberately free of the uuid that was asked for:
#: reflecting it would turn every refusal into a small echo surface, and the owner has
#: no use for a string they just typed.
IMAGE_REFUSALS: dict[int, str] = {
    404: "No such photo in this review session.",
    410: "That photo's local copy is gone — Photos evicts and regenerates derivatives.",
    415: "That file is not an image this server will serve.",
}

#: Session cookie holding the launch token. Not `__Host-` prefixed, because that
#: prefix requires `Secure`, and this app is plain HTTP on loopback by design.
COOKIE_NAME = "photocull_review"

#: Query parameter carrying the token on the launch URL, and nowhere else.
TOKEN_PARAM = "t"

#: The only authorities this server answers to. `localhost` is here because that is
#: what a human types; the socket itself is bound to `127.0.0.1` by the launcher, so
#: nothing off-machine can reach us regardless of the name it uses.
LOCAL_HOSTNAMES = frozenset({"127.0.0.1", "localhost"})

#: Methods that cannot change state, and so do not need `Sec-Fetch-Site`. Everything
#: not listed here is treated as a mutation — default-deny, so a method nobody
#: anticipated is gated rather than waved through.
READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

#: The one `Sec-Fetch-Site` value a mutation may carry. `same-site` is deliberately
#: absent: a different port on `localhost` is same-site but a different origin.
SAME_ORIGIN = "same-origin"

MISDIRECTED_REQUEST = 421

#: `frame-ancestors` and `base-uri` are not covered by `default-src` and have to be
#: stated separately; both are `'none'` because this page is never framed and never
#: rewrites its own base.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)

SECURITY_HEADERS: dict[str, str] = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
}

#: Where the frontend lives. Plain files, no build step — `default-src 'self'` makes
#: inline JS inert, so the frontend arrives as served ES modules and a stylesheet.
STATIC_DIR = Path(__file__).parent / "static"

#: The shell, served at `/`.
SHELL_FILE = "index.html"

#: Asset name -> media type. An allowlist for the same structural reason
#: `images.ALLOWED_MEDIA_TYPES` is one, and with the same consequence: the requested
#: name is only ever a **dictionary key**, never joined to `STATIC_DIR` before it has
#: matched an entry here. A traversal string is not sanitised, it simply misses the
#: mapping — so this process, which holds Full Disk Access, cannot be talked into
#: reading a file the package does not ship. The media type comes from this table and
#: never from the filename, so nothing here can be served as something it is not.
STATIC_FILES: dict[str, str] = {
    "app.js": "text/javascript; charset=utf-8",
    "keymap.js": "text/javascript; charset=utf-8",
    "magnify.js": "text/javascript; charset=utf-8",
    "overview.js": "text/javascript; charset=utf-8",
    "app.css": "text/css; charset=utf-8",
}


def new_token() -> str:
    """A per-launch bearer token: 256 bits, URL-safe, ASCII by construction."""
    return secrets.token_urlsafe(32)


def hostname_is_local(host_header: str | None) -> bool:
    """True when the `Host` header names a loopback authority on any port.

    Any port, rather than *our* port, and that is a deliberate narrowing of
    `review-server-is-hardened-from-the-first-commit`'s wording. A browser always
    sends the port it actually connected to, so an attacker cannot present a
    different one and still reach this socket; and the launcher binds port 0, so the
    real port is not known until after the app object exists. Checking the hostname
    is what closes rebinding; checking the port would only add a plumbing hazard.
    """
    if not host_header:
        return False
    hostname, separator, port = host_header.strip().lower().partition(":")
    if separator and not (port.isdigit() and 1 <= int(port) <= 65535):
        # Catches `127.0.0.1:54321@evil.example` and any IPv6 authority, both of
        # which survive a naive "split on colon and check the first half".
        return False
    return hostname in LOCAL_HOSTNAMES


def token_matches(presented: str | None, expected: str) -> bool:
    """Constant-time comparison that is total over any string a client can send.

    Encoding both sides sidesteps `compare_digest`'s refusal to compare non-ASCII
    `str`. `errors="replace"` cannot cause a false accept: `expected` is
    `token_urlsafe` output, so it holds none of the substitute characters.
    """
    if not presented:
        return False
    return secrets.compare_digest(
        presented.encode("utf-8", "replace"), expected.encode("utf-8", "replace")
    )


def is_mutation(method: str) -> bool:
    """Everything that is not a known-safe read counts as a mutation."""
    return method.upper() not in READ_METHODS


def is_same_origin(sec_fetch_site: str | None) -> bool:
    """Exact match. A prefix or substring test would accept `same-site`."""
    return sec_fetch_site == SAME_ORIGIN


def _handoff(request: Request, token: str) -> Response:
    """Trade the token in the URL for a cookie, then send the browser to a clean URL."""
    remaining = [
        (key, value)
        for key, value in request.query_params.multi_items()
        if key != TOKEN_PARAM
    ]
    target = request.url.path
    if remaining:
        target = f"{target}?{urlencode(remaining)}"

    # 303 rather than 302: it forces the follow-up to be a GET and leaves the query
    # string behind, which is the entire point of the handoff.
    response = RedirectResponse(target, status_code=303)
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        samesite="strict",
        path="/",
    )
    return response


async def _gated(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
    *,
    token: str,
    on_activity: Callable[[], None] | None = None,
) -> Response:
    """The four checks, in the order the module docstring states.

    `on_activity` fires only on the two paths that mean *the owner's browser is
    working*: a successful handoff, and a request that cleared every check. It is
    deliberately not called before the checks, and deliberately not implemented as
    another middleware — middleware registered around this one would run outside the
    gate, and a refused request would then keep the review session alive.
    """
    if not hostname_is_local(request.headers.get("host")):
        return PlainTextResponse(
            "This server answers only on 127.0.0.1.", status_code=MISDIRECTED_REQUEST
        )

    presented = request.query_params.get(TOKEN_PARAM)
    if request.method == "GET" and presented is not None:
        if not token_matches(presented, token):
            return PlainTextResponse("Invalid session token.", status_code=401)
        if on_activity is not None:
            on_activity()
        return _handoff(request, token)

    if not token_matches(request.cookies.get(COOKIE_NAME), token):
        return PlainTextResponse(
            "No session. Open the URL printed by `photocull review`.", status_code=401
        )

    if is_mutation(request.method) and not is_same_origin(
        request.headers.get("sec-fetch-site")
    ):
        return PlainTextResponse("Cross-origin write refused.", status_code=403)

    if on_activity is not None:
        on_activity()
    return await call_next(request)


def build_app(
    *,
    token: str,
    images: ImageSource | None = None,
    review: "ReviewSession | None" = None,
    on_shutdown: Callable[[], None] | None = None,
    on_activity: Callable[[], None] | None = None,
) -> FastAPI:
    """The hardened app: the placeholder shell, the image endpoint, the review API.

    The docs routes are off because FastAPI serves Swagger UI and ReDoc from
    `cdn.jsdelivr.net` — an outbound fetch this app must never make, and one the CSP
    would block anyway.

    `images` and `review` are both optional so Task 6's app object keeps working
    unchanged and so the security tests can build an app with no filesystem reach and no
    decision log at all. When one is omitted its routes are not registered, which is a
    stronger statement than a route that always 404s: there is nothing to reach.

    Every review route is behind the same gate as everything else — the middleware runs
    before routing, so a write arriving without `Sec-Fetch-Site: same-origin` is refused
    without the API being consulted.
    """
    if not token or not token.isascii():
        raise ValueError("the launch token must be a non-empty ASCII string")

    app = FastAPI(
        title="photocull review",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def enforce(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await _gated(
            request, call_next, token=token, on_activity=on_activity
        )
        for header, value in SECURITY_HEADERS.items():
            response.headers[header] = value
        return response

    @app.get("/", response_class=HTMLResponse)
    async def shell() -> HTMLResponse:
        # Read per request rather than cached at import: the file is a few kilobytes on
        # local disk, and a cached copy would mean editing the frontend requires
        # restarting a review sitting that took 32 s to scan.
        return HTMLResponse((STATIC_DIR / SHELL_FILE).read_text(encoding="utf-8"))

    @app.get("/static/{name}")
    async def static_asset(name: str) -> Response:
        """Serve one frontend file, by name, from the allowlist and nowhere else.

        `{name}`, never `{name:path}` — that converter admits `/`, which is what makes
        the traversal spellings 404 at the router before this function runs. The same
        rule the image endpoint is built on, for the same reason.
        """
        media_type = STATIC_FILES.get(name)
        if media_type is None:
            return PlainTextResponse("No such asset.", status_code=404)
        return Response(
            (STATIC_DIR / name).read_bytes(),
            media_type=media_type,
        )

    if images is not None:

        @app.get("/api/images/{uuid}")
        async def image(uuid: str) -> Response:
            """Serve one photo's pixels.

            The path parameter is `{uuid}`, **never** `{uuid:path}`. That converter
            excludes `/`, which is what makes `../../etc/passwd` and its percent-encoded
            spellings 404 at the router before this function runs — measured, all four
            shapes, in `tests/review/test_images.py`. Widening it would reopen every one
            of them, so the tests fail loudly if anyone does.
            """
            outcome = resolve_image(images, uuid)
            if outcome.status != 200 or outcome.path is None:
                return PlainTextResponse(
                    IMAGE_REFUSALS[outcome.status], status_code=outcome.status
                )
            # Read and hand over the bytes rather than returning `FileResponse`: the
            # media type must come from the allowlist, not from the filename.
            return Response(
                outcome.path.read_bytes(),
                media_type=outcome.media_type,
            )

    if review is not None:
        add_review_routes(app, review, on_shutdown=on_shutdown)

    return app
