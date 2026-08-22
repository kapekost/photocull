"""Collection gate for `tests/album/`.

Mirrors `tests/review/conftest.py`'s own gate: everything under this directory needs
`fastapi`, so importing it at conftest level skips the whole directory cleanly on a
machine where the `[album]` extra was never installed, rather than erroring partway
through collection. No Chromium gate here -- Task 9 ships no browser-driven test."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="needs the [album] extra: pip install -e '.[dev,album]'")
