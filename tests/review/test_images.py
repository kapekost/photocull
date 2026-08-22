"""The image endpoint: uuid in, bytes out, never a path.

This process holds Full Disk Access, so the failure this file exists to prevent is not
"a broken image" — it is an FDA-privileged arbitrary-file-read server reachable by
anything holding the session cookie. Every test here is a traversal or a content-type
question first and a delivery question second.

**Three findings from the spike that shaped these tests**, all measured against the
installed FastAPI/Starlette rather than assumed:

1. **Slash-bearing traversals never reach the handler.** `../../etc/passwd`,
   `..%2F..%2Fetc%2Fpasswd`, `%2e%2e%2f...` and `/etc/passwd` are all 404'd by the
   router, because the default `{uuid}` converter excludes `/` and Starlette matches on
   the already-decoded path.

   **Corrected by the mutation pass:** an earlier version of this docstring said that
   safety was "entirely a property of the converter" and that these tests would fail if
   someone widened the route to `{uuid:path}`. Both claims were wrong, and the mutation
   pass proved it — widening the converter left every test green. The reason is the good
   news: the resolver is a **dictionary lookup**, so `../../etc/passwd` is simply a key
   that is not there, and traversal is impossible with or without the converter. The
   converter is a real second layer but it is not the load-bearing one, and no
   status-code test can tell the two apart. `test_a_slash_bearing_uuid_never_reaches_the_resolver`
   pins the converter specifically so the redundancy cannot erode unnoticed.
2. **A NUL byte does reach the handler.** `abc%00.jpg` matches and arrives as
   `'abc\\x00.jpg'`, and `Path.exists()` on a NUL-bearing path raises `ValueError`
   rather than returning False — so the obvious `if not path.exists(): 404` implementation
   answers a crafted uuid with a 500 and a traceback.
3. **`FileResponse` guesses the media type from the extension** — measured: `.plist`
   yields `application/octet-stream`, `.heic` yields `image/heic`. That is precisely the
   "trust the filename" behaviour the allowlist replaces.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from photocull_review.images import (
    ALLOWED_MEDIA_TYPES,
    ImageOutcome,
    mapping_source,
    resolve_image,
)
from photocull_review.server import COOKIE_NAME, build_app

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 32
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


@pytest.fixture
def library(tmp_path: Path) -> dict[str, Path]:
    """A stand-in for the launch-time uuid -> display-path map."""
    jpeg = tmp_path / "keeper.jpeg"
    jpeg.write_bytes(JPEG_BYTES)
    png = tmp_path / "demo.png"
    png.write_bytes(PNG_BYTES)
    plist = tmp_path / "secrets.plist"
    plist.write_bytes(b"<plist>whatever</plist>")
    missing = tmp_path / "evicted.jpeg"  # deliberately never created
    return {"jpeg-uuid": jpeg, "png-uuid": png, "plist-uuid": plist, "gone-uuid": missing}


@pytest.fixture
def client(library: dict[str, Path]) -> TestClient:
    app = build_app(token="tok", images=mapping_source(library))
    client = TestClient(app, base_url="http://127.0.0.1:8000")
    client.cookies.set(COOKIE_NAME, "tok")
    return client


# --- the pure resolver ----------------------------------------------------------


def test_an_unknown_uuid_resolves_to_404_without_touching_the_filesystem(monkeypatch, library):
    """The plan's phrasing, kept literally: *404 touching no filesystem*. An unknown
    uuid must be answered from the mapping alone, because anything that reaches the disk
    on an attacker-chosen key is a probe oracle."""

    def explode(*args, **kwargs):
        raise AssertionError("the filesystem was touched for an unknown uuid")

    monkeypatch.setattr(Path, "open", explode)
    monkeypatch.setattr(Path, "exists", explode)
    monkeypatch.setattr(Path, "stat", explode)

    outcome = resolve_image(mapping_source(library), "no-such-uuid")
    assert outcome == ImageOutcome(404)


def test_a_resolved_but_deleted_file_is_410_not_500(library):
    """Photos evicts and regenerates derivatives; a path that resolved at launch and is
    gone by the request is an ordinary Tuesday, and it has its own status so the UI can
    say "this take is no longer cached" instead of "the server broke"."""
    outcome = resolve_image(mapping_source(library), "gone-uuid")
    assert outcome.status == 410


def test_the_media_type_comes_from_an_allowlist_not_the_extension(library):
    """`.plist` is served by nothing. Measured: `FileResponse` would have handed it out
    as `application/octet-stream` — a download, from a process holding Full Disk
    Access."""
    assert resolve_image(mapping_source(library), "plist-uuid").status == 415


@pytest.mark.parametrize("uuid,media", [("jpeg-uuid", "image/jpeg"), ("png-uuid", "image/png")])
def test_both_real_formats_are_allowed(library, uuid, media):
    """Measured, and this pairing is load-bearing: **every one of the 14,233 derivatives
    in the real library is `.jpeg`**, while `demo.write_png` writes `.png`. An allowlist
    covering only the library would leave `--demo` — the fixture Tasks 9-13 are built
    and reviewed against — serving 415 for every photo."""
    outcome = resolve_image(mapping_source(library), uuid)
    assert outcome.status == 200
    assert outcome.media_type == media


def test_the_allowlist_maps_only_image_types():
    assert set(ALLOWED_MEDIA_TYPES) == {".jpeg", ".jpg", ".png"}
    assert all(media.startswith("image/") for media in ALLOWED_MEDIA_TYPES.values())


def test_a_nul_bearing_uuid_is_404_and_never_raises(library):
    """Measured in the spike: `abc%00.jpg` reaches the handler, and `Path.exists()` on a
    NUL-bearing path raises `ValueError` rather than returning False. A lookup-first
    resolver is immune by construction; a path-building one is not."""
    assert resolve_image(mapping_source(library), "abc\x00.jpg") == ImageOutcome(404)


def test_the_uuid_is_never_used_to_build_a_path(tmp_path):
    """The architectural guarantee behind every traversal test: the uuid is a dictionary
    **key**, never a path component. A resolver that joined it to a base directory would
    pass the routing tests above (the router blocks slashes) and still be wrong the day
    a `{uuid:path}` route, a symlink, or a non-slash traversal appears."""
    victim = tmp_path / "victim.jpeg"
    victim.write_bytes(JPEG_BYTES)
    source = mapping_source({"known": victim})
    for hostile in ("victim.jpeg", "./victim.jpeg", str(victim), "..", "."):
        assert resolve_image(source, hostile) == ImageOutcome(404), hostile


# --- over HTTP ------------------------------------------------------------------


def test_the_bytes_come_back_exactly(client):
    response = client.get("/api/images/jpeg-uuid")
    assert response.status_code == 200
    assert response.content == JPEG_BYTES
    assert response.headers["content-type"] == "image/jpeg"


def test_every_image_response_is_no_store(client):
    """Consistent with the whole app (`review-server-is-hardened-from-the-first-commit`):
    a photo of the owner's family must not sit in a shared disk cache."""
    response = client.get("/api/images/jpeg-uuid")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize(
    "hostile",
    [
        "../../etc/passwd",
        "..%2F..%2Fetc%2Fpasswd",
        "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "/etc/passwd",
        "....//etc/passwd",
        "abc%00.jpg",
    ],
)
def test_traversal_shapes_never_serve_a_file(client, hostile):
    """Four of these are stopped by the router's `{uuid}` converter and two by the
    resolver. The test does not care which — it cares that the answer is never 200 and
    never 5xx, so that changing the route to `{uuid:path}` turns four of them red."""
    response = client.get(f"/api/images/{hostile}")
    assert response.status_code == 404, response.text
    assert b"root:" not in response.content


def test_a_slash_bearing_uuid_never_reaches_the_resolver(library):
    """Pins the router's `{uuid}` converter as a distinct layer from the lookup.

    Added because the mutation pass showed that widening the route to `{uuid:path}`
    changed no status code anywhere — the dict lookup 404s traversals on its own, so the
    two defences are indistinguishable by result. This test watches the *seam* instead:
    with `{uuid}`, a slash-bearing request is rejected before any resolver call happens.
    Widen the converter and the spy records a call, which is the only way to notice that
    the outer layer has quietly stopped contributing.
    """
    seen: list[str] = []

    def spy(uuid: str) -> Path | None:
        seen.append(uuid)
        return library.get(uuid)

    app = build_app(token="tok", images=spy)
    client = TestClient(app, base_url="http://127.0.0.1:8000")
    client.cookies.set(COOKIE_NAME, "tok")

    assert client.get("/api/images/jpeg-uuid").status_code == 200
    assert seen == ["jpeg-uuid"]

    seen.clear()
    for hostile in ("../../etc/passwd", "..%2F..%2Fetc%2Fpasswd", "/etc/passwd"):
        assert client.get(f"/api/images/{hostile}").status_code == 404
    assert seen == [], f"the router let a slash-bearing uuid through: {seen}"


def test_a_refused_type_is_never_opened_or_stated(monkeypatch, library):
    """Pins the *order* inside `resolve_image`: allowlist before any filesystem call.

    A survivor of the mutation pass — inserting a stat before the suffix check left every
    other test green, because the unknown-uuid test returns before reaching it. A file
    this server would refuse to serve should never be touched at all.
    """

    def explode(*args, **kwargs):
        raise AssertionError("a refused file was touched")

    monkeypatch.setattr(Path, "open", explode)
    monkeypatch.setattr(Path, "is_file", explode)
    monkeypatch.setattr(Path, "stat", explode)

    assert resolve_image(mapping_source(library), "plist-uuid") == ImageOutcome(415)


def test_the_suffix_check_is_case_insensitive(tmp_path):
    """macOS filesystems are case-insensitive by default, so `IMG_0001.JPG` is an
    ordinary filename rather than an exotic one. Another mutation-pass survivor: dropping
    `.lower()` refused it with a 415 and no test noticed."""
    shouty = tmp_path / "IMG_0001.JPG"
    shouty.write_bytes(JPEG_BYTES)
    outcome = resolve_image(mapping_source({"shouty": shouty}), "shouty")
    assert outcome.status == 200
    assert outcome.media_type == "image/jpeg"


def test_an_unknown_uuid_is_404_over_http(client):
    assert client.get("/api/images/nope").status_code == 404


def test_an_evicted_derivative_is_410_over_http(client):
    assert client.get("/api/images/gone-uuid").status_code == 410


def test_a_disallowed_type_is_415_over_http(client):
    response = client.get("/api/images/plist-uuid")
    assert response.status_code == 415
    assert b"plist" not in response.content


def test_the_image_endpoint_is_still_behind_the_session_gate(library):
    """The gate is the app's, not the route's — but a route added after the middleware
    is exactly where that could stop being true."""
    app = build_app(token="tok", images=mapping_source(library))
    client = TestClient(app, base_url="http://127.0.0.1:8000")
    assert client.get("/api/images/jpeg-uuid").status_code == 401


def test_an_app_built_without_images_serves_no_image_route(library):
    """Task 6's app object must keep working unchanged — the endpoint is additive."""
    app = build_app(token="tok")
    client = TestClient(app, base_url="http://127.0.0.1:8000")
    client.cookies.set(COOKIE_NAME, "tok")
    assert client.get("/api/images/jpeg-uuid").status_code == 404
