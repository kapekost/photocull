"""Proves the Phase 1b test scaffold itself works, before anything is built on it.

Both gates in `conftest.py` are load-bearing for the plan's promise that every Phase
1b task leaves `pytest -q` green, and neither would be exercised by any test that
Tasks 2-15 add — the `[review]` skip only fires where the extra is absent, and the
`ui` skip only fires where Chromium is. A gate nobody has watched fail is a guess.
"""

from __future__ import annotations

import pytest


def test_the_review_directory_is_collected_when_the_extra_is_installed():
    """If this runs at all, `importorskip("fastapi")` let the directory through."""
    import fastapi

    import photocull_review

    assert fastapi.__version__
    assert photocull_review.__doc__ is not None


def test_the_core_package_is_importable_from_here_too():
    """The boundary runs one way: review may import photocull, never the reverse."""
    from photocull.config import ClusterConfig

    assert ClusterConfig().similarity_threshold > 0


@pytest.mark.ui
def test_chromium_launches(page):
    """The only browser test in Task 1, and its job is to be skipped correctly.

    On a machine with Chromium this genuinely drives it; on one without, `conftest`'s
    `ui` gate turns it into a skip rather than an error. Task 10's Playwright suite
    inherits whichever outcome this one gets.
    """
    page.set_content("<h1>photocull</h1>")
    assert page.locator("h1").inner_text() == "photocull"
