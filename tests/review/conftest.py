"""Collection gates for the review-UI tests.

Two independent skips, because the review suite has two independent prerequisites and
conflating them would hide which one is missing:

1. **The `[review]` extra.** Everything under this directory needs `fastapi`. Without
   it the whole directory is skipped, so `pip install -e ".[dev]"` still yields a
   fully green run. Verified against pytest 9.1.1 that `importorskip` at conftest
   level skips the directory cleanly rather than erroring.
2. **A Chromium binary.** `pytest-playwright` being installed does not mean the
   browser is — `playwright install chromium` is a separate download. Tests marked
   `ui` are skipped when it is absent.

This matters beyond convenience: the plan requires every Phase 1b task to leave
`pytest -q` green, and that promise is only meaningful if a machine without a browser
reports skips rather than errors.
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys
import threading

import pytest

pytest.importorskip("fastapi", reason="needs the [review] extra: pip install -e '.[dev,review]'")

from photocull_review import launcher  # noqa: E402
from photocull_review.decisions import DecisionLog  # noqa: E402
from photocull_review.demo import build_demo_clusters  # noqa: E402


_PROBE = """
from pathlib import Path
from playwright.sync_api import sync_playwright

with sync_playwright() as playwright:
    raise SystemExit(0 if Path(playwright.chromium.executable_path).exists() else 1)
"""


@functools.cache
def chromium_available() -> bool:
    """True when Playwright can actually launch Chromium, not merely import itself.

    Run in a subprocess rather than in-process, and not for tidiness. Resolving the
    executable path starts Playwright's Node driver over asyncio; when no browser has
    been downloaded, the failure surfaces *twice* — once as the exception this
    function handles, and again later as `Task was destroyed but it is pending` and an
    unretrieved-future `TargetClosedError` printed during garbage collection, well
    outside any `try` or `redirect_stderr` block. That second copy lands on every
    `pytest -q` run in the repo. A subprocess contains it by construction and cannot
    leave a half-torn-down event loop inside the test session.

    Costs one interpreter spawn per session, and only when `ui` tests were collected.
    """
    try:
        return (
            subprocess.run(
                [sys.executable, "-c", _PROBE],
                capture_output=True,
                timeout=60,
            ).returncode
            == 0
        )
    except (subprocess.SubprocessError, OSError):
        return False


def pytest_collection_modifyitems(config, items):
    ui_items = [item for item in items if "ui" in item.keywords]
    if not ui_items:
        # Nothing to gate. Do not pay for (or risk) starting the driver.
        return
    if os.environ.get("PHOTOCULL_REQUIRE_UI") or chromium_available():
        return
    skip = pytest.mark.skip(reason="Chromium unavailable; run `playwright install chromium`")
    for item in ui_items:
        item.add_marker(skip)


# --- the live app, for every `ui` module -------------------------------------------
#
# Both browser-driven modules need the same two things, so they live here rather than
# in whichever file happened to need them first. `test_ui_magnify.py` additionally
# needs a second page at a different device pixel ratio, which is only expressible
# against a fixture that hands back the server rather than an already-loaded page.


@pytest.fixture
def review_server(tmp_path):
    """A live `photocull review --demo` on a loopback port, torn down after the test.

    A real uvicorn socket rather than `TestClient`, for the reason Task 6 recorded:
    `TestClient` never opens one, and the security gate, the cookie handoff and the
    browser's own `Sec-Fetch-Site` header only exist on the real path."""
    clusters = build_demo_clusters(tmp_path / "images")
    log = DecisionLog(tmp_path / "decisions.db")
    launch = launcher.prepare(clusters, log=log)
    ready = threading.Event()
    thread = threading.Thread(
        target=launcher.serve,
        args=(launch,),
        kwargs={"poll_interval": 0.01, "on_ready": lambda _launch: ready.set()},
        daemon=True,
    )
    thread.start()
    assert ready.wait(timeout=20), "the review server never reported itself ready"
    try:
        yield launch
    finally:
        launch.lifetime.end()
        thread.join(timeout=10)
        log.close()


@pytest.fixture
def review(page, review_server):
    """The review UI, loaded and past the token handoff, sitting on the first cluster."""
    page.goto(review_server.url)
    page.wait_for_selector("#app:not([hidden])", timeout=15_000)
    return page
