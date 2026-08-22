"""Task 10: a package-local mirror of `tests/test_guardrails.py`'s two frontend-asset
gates (`test_the_frontend_references_no_external_origin`,
`test_every_non_python_source_file_is_a_frontend_asset`), scoped to
`photocull_album/static` specifically.

Those two guardrail tests already cover this directory too -- they were widened as
part of this same task to add `photocull_album/static` to their scan, after adding
this package's frontend tripped `test_every_non_python_source_file_is_a_frontend_asset`
the first time (that scanner predated this package and needed the same kind of
widening Ticks 51/52 each found in a different guardrail scanner). This file is a
narrower, package-scoped pin: it fails on its own if `photocull_album/static` regresses,
without needing every other package's frontend to also be present or correct."""

from __future__ import annotations

import re
from pathlib import Path

STATIC_DIR = Path(__file__).parent.parent.parent / "src" / "photocull_album" / "static"

EXPECTED_ASSETS = {"index.html", "app.js", "style.css"}

EXTERNAL_ORIGIN_PATTERNS = (r"https?://", r'(?:src|href|url\()\s*=?\s*["\']?//')


def test_every_expected_asset_exists():
    names = {p.name for p in STATIC_DIR.iterdir() if p.is_file()}
    assert EXPECTED_ASSETS <= names


def test_every_file_in_the_static_dir_is_an_expected_asset():
    """Same "no stray source" property `test_every_non_python_source_file_is_a_
    frontend_asset` checks repo-wide, pinned here at the single-directory level."""
    names = {p.name for p in STATIC_DIR.iterdir() if p.is_file()}
    assert names == EXPECTED_ASSETS


def test_no_frontend_file_references_an_external_origin():
    offenders = []
    for path in sorted(STATIC_DIR.iterdir()):
        if path.suffix not in {".html", ".css", ".js"}:
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            for pattern in EXTERNAL_ORIGIN_PATTERNS:
                if re.search(pattern, line):
                    offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert offenders == [], f"photocull_album's frontend reaches off-origin: {offenders}"
