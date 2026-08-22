"""The review server's gate, tested before it has anything worth guarding.

Every test here is written against a *plausible wrong* implementation rather than
against the happy path, because each of these checks has an obvious version that
passes the obvious cases:

- a `startswith` host check accepts `127.0.0.1.evil.example`;
- an `==` token compare is correct and not constant-time, so no behavioural test can
  tell it apart from `secrets.compare_digest` — that one needs a structural assertion,
  and it is the reason `test_the_token_is_never_compared_with_equality` reads source;
- accepting `same-site` instead of `same-origin` passes every same-origin test;
- five stacked middlewares run in the **reverse** of their registration order, so the
  order the checks fire in has to be pinned by outcome, not by reading the file.

`TestClient`'s default `base_url` is `http://testserver`, which the host gate refuses
with 421 — so every client here is built on a loopback base URL. That is not a
workaround: it means the gate is proven to be live on every single test in this file,
since forgetting it turns the whole module red.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient

from photocull_review import server

TOKEN = "n5Qk_test-token-with-plenty-of-entropy_7Xa"
BASE_URL = "http://127.0.0.1:54321"


@pytest.fixture
def app():
    """The real app, plus one probe route so an *allowed* mutation has somewhere to go.

    Refusals need no route at all — middleware runs before routing, so a POST to a
    path that does not exist is still refused by the gate (verified: it returns the
    gate's status, not 404).
    """
    application = server.build_app(token=TOKEN)

    @application.post("/api/probe")
    def probe() -> dict[str, bool]:
        return {"ok": True}

    return application


@pytest.fixture
def client(app):
    return TestClient(app, base_url=BASE_URL)


@pytest.fixture
def authed(client):
    """A client that has been through the launch URL and holds the cookie."""
    response = client.get(f"/?t={TOKEN}")
    assert response.status_code == 200, "the launch handoff itself failed"
    return client


# --------------------------------------------------------------------------------
# 1. Host validation — what actually stops DNS rebinding
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("127.0.0.1:54321", True),
        ("localhost", True),
        ("localhost:8000", True),
        ("LOCALHOST:8000", True),  # hostnames are case-insensitive
        ("  127.0.0.1:54321  ", True),
        # The whole point: a prefix match is not a host match.
        ("127.0.0.1.evil.example", False),
        ("localhost.evil.example", False),
        ("evil.example", False),
        ("evil.example:54321", False),
        # A userinfo-shaped tail after the port fools a "split on colon, check the
        # first half" implementation.
        ("127.0.0.1:54321@evil.example", False),
        ("127.0.0.1:notaport", False),
        ("127.0.0.1:", False),
        ("127.0.0.1:99999", False),
        # We bind IPv4 loopback only, so an IPv6 authority is not ours.
        ("[::1]:8000", False),
        ("", False),
        (None, False),
    ],
)
def test_only_a_loopback_authority_is_accepted(host, expected):
    assert server.hostname_is_local(host) is expected


def test_a_foreign_host_is_refused_even_with_a_valid_token(client):
    """A rebinding request carries a valid-looking everything except its authority."""
    response = client.get(
        f"/?t={TOKEN}", headers={"Host": "evil.example"}, follow_redirects=False
    )
    assert response.status_code == 421
    assert "set-cookie" not in response.headers


def test_a_foreign_host_is_refused_even_with_a_valid_cookie(authed):
    response = authed.get("/", headers={"Host": "127.0.0.1.evil.example"})
    assert response.status_code == 421


def test_the_host_check_runs_before_the_cookie_check(client):
    """Pins the order the middleware fires in, which registration order reverses.

    A foreign host with no cookie must read 421, not 401: the authority is refused
    before authentication is even considered.
    """
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 421
    assert client.get("/").status_code == 401


# --------------------------------------------------------------------------------
# 2. The launch token
# --------------------------------------------------------------------------------


def test_the_launch_url_sets_a_cookie_then_redirects_the_token_out_of_the_url(client):
    response = client.get(f"/?t={TOKEN}", follow_redirects=False)

    assert response.status_code == 303, "303 forces a GET and drops the query string"
    assert response.headers["location"] == "/"
    assert TOKEN not in response.headers["location"]

    cookie = response.headers["set-cookie"]
    assert f"{server.COOKIE_NAME}={TOKEN}" in cookie
    assert "HttpOnly" in cookie, "script-readable would defeat the CSP"
    assert "SameSite=strict" in cookie.replace("SameSite=Strict", "SameSite=strict")
    assert "Path=/" in cookie


def test_the_redirect_keeps_any_other_query_parameters(client):
    response = client.get(f"/?cluster=abc&t={TOKEN}", follow_redirects=False)
    assert response.headers["location"] == "/?cluster=abc"


def test_the_launch_handoff_is_a_read_only_affordance(client, authed):
    """A write may not buy itself a session on the way past the gate.

    Found by the mutation pass, not by inspection: honouring `?t=` on any method
    survived every other test in this file, because the handoff answers with a
    redirect rather than by executing anything. It is still a hole — the write would
    have skipped the `Sec-Fetch-Site` check to get there — so the token is a GET-only
    affordance and a write carrying one falls through to the ordinary gate.
    """
    fresh = TestClient(client.app, base_url=BASE_URL)
    assert fresh.post(f"/?t={TOKEN}", follow_redirects=False).status_code == 401
    assert authed.post(f"/?t={TOKEN}", follow_redirects=False).status_code == 403


def test_a_wrong_token_in_the_url_is_refused_and_sets_no_cookie(client):
    response = client.get("/?t=not-the-token", follow_redirects=False)
    assert response.status_code == 401
    assert "set-cookie" not in response.headers


def test_a_request_without_the_cookie_is_refused(client):
    assert client.get("/").status_code == 401
    assert client.get("/api/probe").status_code == 401


def test_a_wrong_cookie_is_refused(client):
    client.cookies.set(server.COOKIE_NAME, "not-the-token")
    assert client.get("/").status_code == 401


def test_a_non_ascii_token_is_refused_rather_than_crashing(client):
    """`secrets.compare_digest` raises TypeError on non-ASCII str — measured, not read.

    Left unhandled that is a 500 and a traceback on input anyone can send. The query
    string is the reachable route for it: percent-encoded bytes are decoded as UTF-8
    before the gate ever sees them.
    """
    assert client.get("/?t=töken", follow_redirects=False).status_code == 401


def test_a_non_ascii_cookie_is_refused_rather_than_crashing():
    """The same hazard on the cookie, pinned at the function because httpx cannot send it.

    Measured both ways: httpx refuses to encode a non-ASCII cookie header client-side
    (`UnicodeEncodeError`), while Starlette decodes raw header bytes as latin-1 — so a
    non-browser client on a socket *can* deliver one and this is not a hypothetical.
    """
    for presented in ("töken", f"{TOKEN}ö", "\udce4", "🔑" * 40):
        assert server.token_matches(presented, TOKEN) is False, presented


def test_a_prefix_of_the_real_token_is_refused(client):
    assert client.get(f"/?t={TOKEN[:-1]}", follow_redirects=False).status_code == 401
    assert client.get(f"/?t={TOKEN}x", follow_redirects=False).status_code == 401


def test_the_token_is_never_compared_with_equality():
    """The one property no behavioural test can reach: constant-time comparison.

    `==` returns the same answers as `compare_digest` for every input, so a mutation
    swapping them survives any request-level test. This reads the function instead.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(server.token_matches)))
    called = {
        ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    assert "secrets.compare_digest" in called

    comparisons = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops)
    ]
    assert comparisons == [], "an equality compare leaked back into the token check"


def test_a_launch_token_is_unguessable():
    tokens = {server.new_token() for _ in range(50)}
    assert len(tokens) == 50
    assert all(len(t) >= 32 and t.isascii() for t in tokens)


def test_the_app_refuses_a_token_it_could_not_compare_safely():
    """Loud validation at construction, per this repo's standing rule."""
    for bad in ("", "töken"):
        with pytest.raises(ValueError):
            server.build_app(token=bad)


# --------------------------------------------------------------------------------
# 3. Sec-Fetch-Site on mutations
# --------------------------------------------------------------------------------


def test_a_same_origin_mutation_is_allowed(authed):
    response = authed.post("/api/probe", headers={"Sec-Fetch-Site": "same-origin"})
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_a_mutation_without_sec_fetch_site_is_refused(authed):
    assert authed.post("/api/probe").status_code == 403


def test_a_same_site_mutation_is_refused(authed):
    """`same-site` is not `same-origin`; a substring or prefix check accepts it.

    The last three exist to kill implementations no browser would ever expose: a
    substring test passes `not-same-origin`, and a `startswith`/`strip` test passes a
    trailing space.
    """
    for value in (
        "same-site",
        "cross-site",
        "none",
        "Same-Origin ",
        "same-origin ",
        "not-same-origin",
        "same-origin, cross-site",
    ):
        response = authed.post("/api/probe", headers={"Sec-Fetch-Site": value})
        assert response.status_code == 403, f"{value!r} was let through"


def test_a_read_without_sec_fetch_site_is_allowed(authed):
    """Browsers omit it on plenty of ordinary reads, and a read changes nothing."""
    assert authed.get("/").status_code == 200


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_every_method_that_is_not_a_read_is_gated(authed, method):
    """Default-deny by method: the gate lists what is safe, not what is dangerous."""
    assert authed.request(method, "/api/probe").status_code == 403


def test_the_cookie_check_runs_before_the_sec_fetch_check(client):
    """An unauthenticated mutation is a 401, not a 403 — authentication comes first."""
    assert client.post("/api/probe").status_code == 401


def test_a_mutation_is_gated_before_routing(authed):
    """Refusals need no route, which is why the negative tests above invent none."""
    assert authed.post("/api/does-not-exist").status_code == 403
    assert authed.get("/api/does-not-exist").status_code == 404


# --------------------------------------------------------------------------------
# 4. Response headers
# --------------------------------------------------------------------------------


def _all_responses(client, authed):
    return {
        "200": authed.get("/"),
        "404": authed.get("/api/does-not-exist"),
        "401": client.get("/", headers={"Cookie": ""}),
        "421": client.get("/", headers={"Host": "evil.example"}),
        "303": client.get(f"/?t={TOKEN}", follow_redirects=False),
    }


@pytest.mark.parametrize(
    ("header", "value"),
    [
        ("cache-control", "no-store"),
        ("x-content-type-options", "nosniff"),
        ("referrer-policy", "no-referrer"),
    ],
)
def test_every_response_carries_the_security_headers(app, header, value):
    """Including refusals — a 421 body is as cacheable as a 200 unless it says so."""
    fresh = TestClient(app, base_url=BASE_URL)
    authed_client = TestClient(app, base_url=BASE_URL)
    authed_client.get(f"/?t={TOKEN}")

    for label, response in _all_responses(fresh, authed_client).items():
        assert response.headers.get(header) == value, f"{label} response lacked {header}"


def test_the_csp_confines_the_page_to_its_own_origin(authed):
    csp = authed.get("/").headers["content-security-policy"]
    assert "default-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp


def test_the_shell_has_no_inline_script(authed):
    """`default-src 'self'` makes inline JS inert, so shipping any would fail silently.

    Task 6 wrote this as "the shell contains no `<script` at all", which was a correct
    proxy only while the shell was a placeholder — its own docstring named the day it
    would stop being one. Task 10 serves ES modules as files, so the assertion is now
    the property the test is named for: every script tag carries a `src`, and none
    carries a body. A CSP violation is silent in the browser console, which is exactly
    why this is pinned in the suite rather than trusted to show up during review.
    """
    body = authed.get("/").text

    class Scripts(HTMLParser):
        def __init__(self):
            super().__init__()
            self.tags: list[dict[str, str | None]] = []
            self.inline = []
            self._open = False

        def handle_starttag(self, tag, attrs):
            if tag == "script":
                self.tags.append(dict(attrs))
                self._open = True

        def handle_endtag(self, tag):
            if tag == "script":
                self._open = False

        def handle_data(self, data):
            if self._open and data.strip():
                self.inline.append(data)

    parser = Scripts()
    parser.feed(body)

    assert parser.tags, "the shell must load the frontend"
    for attrs in parser.tags:
        assert attrs.get("src"), f"inline script in the shell: {attrs}"
    assert parser.inline == [], "a script tag carries an inline body"
    assert authed.get("/").headers["content-type"].startswith("text/html")


# --------------------------------------------------------------------------------
# 7. The static assets — an allowlist, for the same reason the images are
# --------------------------------------------------------------------------------


def test_every_declared_asset_is_served_with_its_declared_media_type(authed):
    """The type comes from the table, never from the filename."""
    for name, media_type in server.STATIC_FILES.items():
        response = authed.get(f"/static/{name}")
        assert response.status_code == 200, f"/static/{name} is not served"
        assert response.headers["content-type"] == media_type
        assert response.content, f"/static/{name} is empty"


def test_an_asset_that_is_not_on_the_allowlist_is_refused(authed):
    """Absence from the table is the whole check — no suffix rule, no directory walk."""
    assert authed.get("/static/index.html").status_code == 404
    assert authed.get("/static/nothing.js").status_code == 404


def test_the_static_route_cannot_be_walked_out_of(authed):
    """This process holds Full Disk Access, so the name is only ever a dictionary key.

    `{name}` excludes `/` at the router, and every spelling that survives that misses
    the allowlist. Nothing here is sanitised, because nothing here is ever joined to a
    path before it has matched an entry.
    """
    for attempt in (
        "../server.py",
        "..%2Fserver.py",
        "....//server.py",
        "%2e%2e%2fserver.py",
        "../../../../etc/passwd",
        "app.js%00",
    ):
        response = authed.get(f"/static/{attempt}")
        assert response.status_code == 404, f"{attempt} was not refused"
        assert "photocull" not in response.text.lower()


def test_the_assets_the_shell_asks_for_are_the_assets_that_exist(authed):
    """A typo in a `src=` is a blank page, and the CSP report is the only clue.

    Cheap to pin, and it closes the gap between the shell's own references and the
    allowlist that decides what can be served at all.
    """
    body = authed.get("/").text
    referenced = set(re.findall(r'(?:src|href)="/static/([^"]+)"', body))

    assert referenced, "the shell references no assets at all"
    assert referenced <= set(server.STATIC_FILES), (
        f"the shell asks for assets that are not served: "
        f"{sorted(referenced - set(server.STATIC_FILES))}"
    )
    for name in referenced:
        assert authed.get(f"/static/{name}").status_code == 200


def test_every_module_the_frontend_imports_is_on_the_allowlist(authed):
    """The shell is not the only thing that asks for an asset — modules import modules.

    Found at Task 12, and the reach is the whole app rather than one feature: an ES
    module import that 404s stops the importing module evaluating at all, so a missing
    allowlist entry is a **blank page**, not a missing button. Measured by removing
    `magnify.js` from `STATIC_FILES` and running the suite on a machine with no
    Chromium: **638 passed, 58 skipped** — completely green, with the review UI dead.
    The test above could not see it because it reads `src=`/`href=` attributes out of
    the shell, and `app.js` reaches its dependencies by `import`.
    """
    imports: set[str] = set()
    for name in server.STATIC_FILES:
        if not name.endswith(".js"):
            continue
        source = (server.STATIC_DIR / name).read_text()
        imports |= set(re.findall(r'(?:from|import)\s*["\']/static/([^"\']+)["\']', source))

    assert imports, "no module imports another; this gate is watching nothing"
    assert imports <= set(server.STATIC_FILES), (
        f"the frontend imports modules that cannot be served: "
        f"{sorted(imports - set(server.STATIC_FILES))}"
    )
    for name in imports:
        assert authed.get(f"/static/{name}").status_code == 200


def test_the_interactive_api_docs_are_off(client):
    """FastAPI's docs pages load Swagger UI and ReDoc from `cdn.jsdelivr.net`.

    Verified against the installed FastAPI, not recalled. A local-only app must not
    have a route whose whole job is to fetch a megabyte of JavaScript from a CDN, and
    the CSP would block it anyway.
    """
    for path in ("/docs", "/redoc", "/openapi.json"):
        client.cookies.set(server.COOKIE_NAME, TOKEN)
        assert client.get(path).status_code == 404, f"{path} is still routed"
