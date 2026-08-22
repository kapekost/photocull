"""The pure image resolver: uuid in, bytes-source-decision out, never a path.

Ported from `tests/review/test_images.py` per this plan's Task 9 Step 3, retargeted to
`photocull_album.images`. Only the pure-resolver section is carried over here: the
"over HTTP" section of the review file exercises `photocull_review.server.build_app`'s
cookie-gated `/api/images/{uuid}` route, and `photocull_album` wires no image route in
this task -- `images.py` is duplicated now so Task 10's frontend has it ready, but the
route itself (`/api/image/{uuid}`) is added when `server.py` grows static-asset and
`/api` mounting in Task 10 (see the plan's own Task 10 file list). Porting the HTTP
tests here would either require inventing a route this task doesn't ship, or reaching
across into `photocull_review.server` for one -- exactly the coupling Task 9 exists to
avoid (`phase-3-gets-its-own-package-not-photocull-review`). The security properties
these tests protect are already fully covered at the pure-function level below, which
is what `resolve_image` is designed to make possible.

This process holds Full Disk Access, so the failure this file exists to prevent is not
"a broken image" -- it is an FDA-privileged arbitrary-file-read server reachable by
anything that later gets to call this resolver over HTTP. Every test here is a
traversal or a content-type question first and a delivery question second.

**Three findings from the spike that shaped these tests**, all measured against the
installed FastAPI/Starlette rather than assumed (carried over from the review file's
own investigation, since `resolve_image` here is a byte-for-byte copy):

1. **Slash-bearing traversals never reach a `{uuid}`-routed handler** -- the default
   converter excludes `/`. Not exercised over HTTP here (no route in this task), but
   `resolve_image` itself is immune regardless, because the resolver is a dictionary
   lookup: `../../etc/passwd` is simply a key that is not there.
2. **A NUL byte does reach a handler.** `abc%00.jpg` is a valid string and
   `Path.exists()` on a NUL-bearing path raises `ValueError` rather than returning
   False -- so the obvious `if not path.exists(): 404` implementation would answer a
   crafted uuid with a 500 and a traceback.
3. **`FileResponse` guesses the media type from the extension** -- measured: `.plist`
   yields `application/octet-stream`, `.heic` yields `image/heic`. That is precisely the
   "trust the filename" behaviour the allowlist replaces.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from photocull_album.images import (
    ALLOWED_MEDIA_TYPES,
    ImageOutcome,
    mapping_source,
    resolve_image,
)

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
    covering only the library would leave a synthetic-fixture-only test fixture serving
    415 for every photo."""
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
    pass a routing-level test (the router blocks slashes) and still be wrong the day a
    `{uuid:path}` route, a symlink, or a non-slash traversal appears."""
    victim = tmp_path / "victim.jpeg"
    victim.write_bytes(JPEG_BYTES)
    source = mapping_source({"known": victim})
    for hostile in ("victim.jpeg", "./victim.jpeg", str(victim), "..", "."):
        assert resolve_image(source, hostile) == ImageOutcome(404), hostile


def test_a_refused_type_is_never_opened_or_stated(monkeypatch, library):
    """Pins the *order* inside `resolve_image`: allowlist before any filesystem call.

    A survivor of the review file's mutation pass — inserting a stat before the suffix
    check left every other test green, because the unknown-uuid test returns before
    reaching it. A file this server would refuse to serve should never be touched at
    all.
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
