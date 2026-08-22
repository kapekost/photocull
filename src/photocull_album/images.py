"""Serving photo pixels to the browser: uuid in, bytes out, never a path.

Phase 3's own copy of `photocull_review/images.py`'s pattern -- duplicated, not
imported, per this plan's architecture decision `phase-3-gets-its-own-package-not-
photocull-review` (docs/plans/2026-08-19-phase-3-album-builder.md, Task 9).

**Why a uuid and never a path.** This process holds Full Disk Access — it can read the
owner's entire disk. A `?path=` parameter, or a uuid joined onto a base directory, would
turn the review server into an FDA-privileged arbitrary-file-read service reachable by
anything holding the session cookie. That is the highest-severity mistake this design
could make, so the defence is structural rather than validating: **the uuid is only ever
a dictionary key.** It is never joined to a directory, never normalised, never passed to
`Path()`. A traversal string is not sanitised here — it simply misses the mapping and
404s, exactly like any other unknown key.

That choice also disposes of a real crash, measured during the Task 7 spike: a
NUL-bearing uuid such as `abc%00.jpg` *does* reach the handler (unlike slash-bearing
ones, which the router rejects), and `Path.exists()` on a NUL-bearing path raises
`ValueError` rather than returning False. A lookup-first resolver never constructs that
path at all; a path-building one answers a crafted uuid with a 500.

**Why the media type comes from an allowlist.** `FileResponse` guesses from the
extension — measured on the installed Starlette, `.plist` yields
`application/octet-stream` and `.heic` yields `image/heic`. Guessing means the filename
decides what this server hands out. The allowlist inverts that: a suffix this module
does not recognise is refused with 415 and is never opened.

**Why the path is re-checked on every request and never persisted.** Photos evicts and
regenerates derivatives, so a path that resolved when the session opened can be gone, or
can point at a regenerated file of a different size, by the time the browser asks for it.
An eviction is an ordinary event and gets its own status (410) so the UI can say "this
take is no longer cached locally" rather than showing a server error.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

#: Suffix -> media type. Deliberately tiny, and both entries are measured rather than
#: guessed: **all 14,233 derivatives in the real library are `.jpeg`**, while
#: `demo.write_png` writes `.png`. Dropping the PNG entry would leave `--demo` — the
#: dataset Tasks 9-13 are built and reviewed against — serving 415 for every photo.
#: `.jpg` is carried because Photos is not the only thing that has ever written a
#: derivative, and it costs nothing.
ALLOWED_MEDIA_TYPES: dict[str, str] = {
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
}

#: Resolve a uuid to the file to serve, or `None` when the uuid is not ours. Kept a
#: plain callable rather than a class so the launcher can close over a scan result and
#: the tests can pass a dict, with no shared base class between them.
ImageSource = Callable[[str], Path | None]


@dataclass(frozen=True)
class ImageOutcome:
    """What the endpoint should answer, decided without any HTTP involved.

    Separating the decision from the response keeps every traversal and eviction case
    testable as a pure function — the security properties of this module are not
    contingent on a test client, a route, or a running server.
    """

    status: int
    path: Path | None = None
    media_type: str | None = None


def mapping_source(paths: Mapping[str, Path]) -> ImageSource:
    """The only source shape this app uses: a plain uuid -> path lookup.

    Built once at launch from the scan (which already resolved every display derivative)
    and held in memory. It is never written to the decisions database — a stored path is
    the stale-answer surface `derivative-selection-is-smallest-class` closed.
    """

    def resolve(uuid: str) -> Path | None:
        return paths.get(uuid)

    return resolve


def resolve_image(source: ImageSource, uuid: str) -> ImageOutcome:
    """Decide what to serve for `uuid`, in the one order that is safe.

    The order is not arbitrary. The mapping lookup comes first so an unknown uuid costs
    no filesystem access at all — anything that reaches the disk on an attacker-chosen
    key is a probe oracle for what exists. The suffix check comes next, before any stat,
    so a file this module would refuse to serve is never even opened. Existence is
    checked last, because it is the only question that requires touching the disk.
    """
    path = source(uuid)
    if path is None:
        return ImageOutcome(404)

    media_type = ALLOWED_MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        return ImageOutcome(415)

    if not path.is_file():
        return ImageOutcome(410)

    return ImageOutcome(200, path=path, media_type=media_type)
