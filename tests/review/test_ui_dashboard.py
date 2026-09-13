"""Task 13: the two overview surfaces, driven in a real browser.

`staged-set-gets-dashboard-and-final-check` asked for two things the per-cluster walk
cannot give: a running sense of the whole, and one screen where everything about to be
staged is seen in context before any library is touched. The second is a **gate** — "write-
back only runs on confirmation from (b)" — so the tests below are written against the
ways a gate silently stops being one:

- a dashboard that loses the owner's place, which makes it too expensive to open and so
  never opened;
- a contact sheet that shows the *scorer's* proposal rather than the owner's decisions;
- a confirmation that survives the staged set changing under it;
- a click that pulls a photo back to KEEP in the browser while the log still says `cull`;
- a restore that quietly drops the favourites recorded on that cluster an hour earlier;
- and a sheet that is correct at 24 clusters and unusable at the real staged count, which
  is the failure the plan predicted for this task specifically.

The last one has its own fixture rather than its own assertion, because 1,817 rows is not
a bigger version of 19 rows — it is the case where lazy images and a windowed DOM are the
difference between a screen and a hang.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from photocull.config import ClusterConfig
from photocull_review import launcher
from photocull_review.decisions import DecisionLog, Mark
from photocull_review.demo import build_demo_clusters
from photocull_review.session import cluster_key, config_digest, session_settings
from photocull_review.writeback import WritebackSettings

from ..fixtures import make_cluster
from .test_writeback import LIBRARY, FakeWriter

pytestmark = pytest.mark.ui

# Demo cluster indices, read off `demo._SPEC`.
BIG = 23  # 20 takes: one accepted proposal stages 19 photos
PAIR = 0  # the median cluster — two takes, no annotations


# --- helpers ------------------------------------------------------------------------


def open_dashboard(page):
    page.keyboard.press("d")
    page.wait_for_selector("#dashboard:not([hidden])", timeout=5000)


def open_sheet(page):
    """Open the final check and wait for its first page of rows to be on screen.

    `data-loading` rather than a timeout: the sheet shows its bar before the rows arrive
    (a round trip on the real staged set), so every assertion below would otherwise race
    the fetch that fills it."""
    page.keyboard.press("v")
    page.wait_for_selector("#final-check[data-loading='false']", timeout=15_000)
    page.wait_for_selector("#sheet-rows .staged", timeout=15_000)


def api(page, path):
    return page.evaluate(
        "(p) => fetch(p, {credentials: 'same-origin'}).then((r) => r.json())", path
    )


def decide(page, index, marks, favorites=()):
    """Record a decision the way the app does, without driving 20 keystrokes."""
    return page.evaluate(
        """async ({index, marks, favorites}) => {
            const key = window.__photocull.state.clusters[index].cluster_key;
            const response = await fetch(`/api/clusters/${key}/decision`, {
              method: "POST",
              credentials: "same-origin",
              headers: {"Content-Type": "application/json"},
              body: JSON.stringify({marks, favorites}),
            });
            const data = await response.json();
            // The page's own state has to learn about it too, exactly as `submit` does,
            // or the next render would draw a cluster the server considers decided.
            window.__photocull.state.decided.set(key, {
              marks: data.marks, favorites: data.favorites, batch_id: data.batch_id,
            });
            return data;
        }""",
        {"index": index, "marks": marks, "favorites": list(favorites)},
    )


def marks_for(page, index, keep=(), cull=()):
    """A full mark set for a demo cluster: every take gets exactly one verdict."""
    return page.evaluate(
        """({index, keep, cull}) => {
            const entry = window.__photocull.state.clusters[index];
            const marks = {};
            for (const member of entry.members) {
              marks[member.uuid] = keep.includes(member.uuid) ? "keep"
                : cull.includes(member.uuid) ? "cull"
                : (member.uuid === entry.winner_uuid ? "keep" : "cull");
            }
            return marks;
        }""",
        {"index": index, "keep": list(keep), "cull": list(cull)},
    )


# --- the dashboard --------------------------------------------------------------------


def test_the_dashboard_opens_and_closes_without_losing_the_owners_place(review):
    """The property that decides whether this screen gets used at all.

    Asserted on the challenger index as well as the cluster index, and that is the half
    that would actually break: a re-render on close runs `goTo`, which resets the
    challenger to 0. On a 20-take cluster that silently throws away nineteen steps of
    work, and the owner would learn not to open the dashboard."""
    review.evaluate("(i) => window.__photocull.goTo(i)", BIG)
    review.keyboard.press("ArrowDown")
    review.keyboard.press("ArrowDown")
    before = review.locator("#position").text_content()
    challenger_before = review.evaluate("() => window.__photocull.state.challenger")

    open_dashboard(review)
    review.keyboard.press("Escape")
    review.wait_for_selector("#dashboard", state="hidden", timeout=5000)

    assert review.locator("#position").text_content() == before
    assert review.evaluate("() => window.__photocull.state.challenger") == challenger_before
    assert challenger_before == 2, "the fixture never actually stepped the takes"


def test_the_dashboard_counts_agree_with_the_api(review):
    """Reviewed, remaining, keeping, staging — the four numbers the owner asked for."""
    decide(review, PAIR, marks_for(review, PAIR))
    decide(review, BIG, marks_for(review, BIG))
    open_dashboard(review)

    body = api(review, "/api/dashboard")
    text = review.locator("#dashboard").inner_text()

    assert f"{body['clusters']['decided']}" in text
    assert f"{body['clusters']['remaining']}" in text
    assert str(body["photos"]["keep"]) in text
    assert str(body["photos"]["cull"]) in text
    assert body["photos"]["cull"] == 20, "one pair plus the 20-take cluster stages 20"


def test_the_dashboard_is_re_read_on_every_open(review):
    """A dashboard rendered once and cached would show a cluster as undecided minutes
    after it was decided — and the owner's only way to notice is to already know the
    number, which is exactly what they opened it to find out."""
    open_dashboard(review)
    review.keyboard.press("Escape")

    decide(review, BIG, marks_for(review, BIG))
    open_dashboard(review)

    assert "19" in review.locator("#dashboard").inner_text()


def test_the_dashboard_flags_the_largest_staged_clusters_and_jumps_to_one(review):
    """The 20-take cluster is the shape this list exists for: one accepted proposal
    stages nineteen photos, and nothing in the per-cluster walk would ever say so."""
    decide(review, PAIR, marks_for(review, PAIR))
    decide(review, BIG, marks_for(review, BIG))
    review.evaluate("() => window.__photocull.goTo(1)")

    open_dashboard(review)
    first = review.locator("#dashboard .jump").first
    assert "19" in first.inner_text(), "the largest staged cluster is not listed first"
    first.click()

    review.wait_for_selector("#dashboard", state="hidden", timeout=5000)
    assert review.evaluate("() => window.__photocull.state.index") == BIG


# --- the final check ------------------------------------------------------------------


def test_the_sheet_shows_every_staged_photo_beside_its_keepers(review):
    decide(review, BIG, marks_for(review, BIG))
    open_sheet(review)

    body = api(review, "/api/final-check")
    shown = review.evaluate(
        "() => [...document.querySelectorAll('#sheet-rows .staged')]"
        ".map((node) => node.dataset.uuid)"
    )
    assert body["total"] == 19
    assert shown == [row["photo"]["uuid"] for row in body["staged"]]

    keepers = review.evaluate(
        "() => [...document.querySelectorAll('#sheet-rows .keeper')]"
        ".map((node) => node.dataset.uuid)"
    )
    assert len(set(keepers)) == 1, "the one keeper this cluster's takes lost to"


def test_the_sheet_reads_decisions_and_never_proposals(review):
    """Every cluster opens with a suggestion, so a sheet built from `proposed` would
    show 60-odd staged photos before the owner has decided anything."""
    open_sheet_empty(review)
    assert review.evaluate("() => document.querySelectorAll('#sheet-rows .staged').length") == 0
    assert "nothing" in review.locator("#final-check").inner_text().lower()


def open_sheet_empty(page):
    page.keyboard.press("v")
    page.wait_for_selector("#final-check[data-loading='false']", timeout=15_000)


def test_a_cluster_culled_entirely_renders_with_no_keeper_beside_it(review):
    """`keepers: []` is a real shape (`a-whole-cluster-may-be-culled`), and the sheet was
    specified as "every staged photo beside the keeper(s) it lost to" before that ruling
    existed. A row that renders nothing when there is no keeper would hide the photos
    that need the second look most."""
    decide(review, PAIR, marks_for(review, PAIR, cull=all_uuids(review, PAIR)))
    open_sheet(review)

    assert review.evaluate(
        "() => document.querySelectorAll('#sheet-rows .staged').length"
    ) == 2
    assert "no keeper" in review.locator("#sheet-rows").inner_text().lower()


def all_uuids(page, index):
    return page.evaluate(
        "(i) => window.__photocull.state.clusters[i].members.map((m) => m.uuid)", index
    )


def test_clicking_a_staged_photo_pulls_it_back_to_keep_and_the_api_agrees(review):
    decide(review, BIG, marks_for(review, BIG))
    open_sheet(review)

    uuid = review.evaluate(
        "() => document.querySelector('#sheet-rows .staged').dataset.uuid"
    )
    review.locator("#sheet-rows .staged").first.click()
    review.wait_for_selector("#sheet-rows .staged[data-restored='true']", timeout=5000)

    body = api(review, "/api/final-check")
    assert body["total"] == 18
    assert uuid not in [row["photo"]["uuid"] for row in body["staged"]]

    key = review.evaluate("(i) => window.__photocull.state.clusters[i].cluster_key", BIG)
    assert api(review, "/api/session")["decided"][key]["marks"][uuid] == "keep"


def test_a_restored_photo_can_be_staged_again_from_the_sheet(review):
    """A mis-click has to be undoable on the screen it happened on. The row therefore
    stays in place rather than vanishing — a row that disappears takes its own undo with
    it."""
    decide(review, BIG, marks_for(review, BIG))
    open_sheet(review)

    row = review.locator("#sheet-rows .staged").first
    row.click()
    review.wait_for_selector("#sheet-rows .staged[data-restored='true']", timeout=5000)
    row.click()
    review.wait_for_selector("#sheet-rows .staged:not([data-restored='true'])", timeout=5000)

    assert api(review, "/api/final-check")["total"] == 19


def test_a_cluster_split_across_two_pages_stays_one_section(review):
    """The sheet is grouped by cluster and fetched by page, and those two boundaries do
    not line up. With every demo cluster culled entirely there are 82 staged photos
    against a 60-row page, so the 9-take cluster straddles the boundary — and a sheet
    that opened a fresh section per page would show it twice, under two headings, with
    the owner counting the same cluster twice on the screen where counting is the point."""
    for index in range(24):
        decide(review, index, marks_for(review, index, cull=all_uuids(review, index)))
    open_sheet(review)

    review.evaluate("() => document.querySelector('#sheet-scroll').scrollTo(0, 10 ** 7)")
    review.wait_for_function(
        "() => document.querySelectorAll('#sheet-rows .staged').length === 82",
        timeout=15_000,
    )

    sections = review.evaluate(
        "() => [...document.querySelectorAll('.sheet-cluster')]"
        ".map((node) => node.dataset.clusterKey)"
    )
    assert len(sections) == len(set(sections)) == 24, sections
    sizes = review.evaluate(
        "() => [...document.querySelectorAll('.sheet-cluster')]"
        ".map((node) => node.querySelectorAll('.staged').length)"
    )
    assert 9 in sizes and 20 in sizes, sizes


def test_restoring_a_photo_keeps_the_favourites_recorded_on_that_cluster(review):
    """One click on the contact sheet re-submits the whole cluster. If the favourites
    are not carried through that submission, every `F` press in it is silently undone —
    and nothing on the screen would say so."""
    keeper = review.evaluate(
        "(i) => window.__photocull.state.clusters[i].winner_uuid", BIG
    )
    decide(review, BIG, marks_for(review, BIG), favorites=[keeper])
    open_sheet(review)

    review.locator("#sheet-rows .staged").first.click()
    review.wait_for_selector("#sheet-rows .staged[data-restored='true']", timeout=5000)

    key = review.evaluate("(i) => window.__photocull.state.clusters[i].cluster_key", BIG)
    assert api(review, "/api/session")["decided"][key]["favorites"] == [keeper]


def test_re_staging_a_photo_that_was_favourited_is_accepted(review):
    """A photo pulled back to KEEP can be favourited in the compare view and then staged
    again from the sheet. `a-favourite-may-not-be-staged` refuses a submission carrying
    both, so without dropping the favourite here that second click would 422 — which the
    owner would read as a click that does nothing."""
    decide(review, BIG, marks_for(review, BIG))
    open_sheet(review)
    row = review.locator("#sheet-rows .staged").first
    uuid = review.evaluate("() => document.querySelector('#sheet-rows .staged').dataset.uuid")
    row.click()
    review.wait_for_selector("#sheet-rows .staged[data-restored='true']", timeout=5000)

    # Favourite it now that it is kept, exactly as `F` would.
    key = review.evaluate("(i) => window.__photocull.state.clusters[i].cluster_key", BIG)
    marks = api(review, "/api/session")["decided"][key]["marks"]
    decide(review, BIG, marks, favorites=[uuid])

    row.click()
    review.wait_for_selector("#sheet-rows .staged:not([data-restored='true'])", timeout=5000)

    body = api(review, "/api/session")["decided"][key]
    assert body["marks"][uuid] == "cull"
    assert uuid not in body["favorites"]


def test_the_compare_view_agrees_with_what_the_sheet_recorded(review):
    """The sheet and the walk are two views of one decision. If a restore only updated
    the sheet, the owner would step back to that cluster, see `CULL` on a photo they just
    kept, and have no way to tell which surface is lying.

    **The cluster has to be visited first, or this test is asleep.** `marksFor` fills its
    cache on first call and falls back to the recorded decision, so a version that never
    writes the marks back passes — the assertion below then reads the decision it just
    posted rather than the state the screen renders from. Visiting the cluster is what
    puts a stale entry in that cache, and it is also what the owner does: this whole
    failure is only reachable for a cluster they have actually looked at."""
    decide(review, BIG, marks_for(review, BIG))
    review.evaluate("(i) => window.__photocull.goTo(i)", BIG)
    open_sheet(review)
    uuid = review.evaluate("() => document.querySelector('#sheet-rows .staged').dataset.uuid")
    review.locator("#sheet-rows .staged").first.click()
    review.wait_for_selector("#sheet-rows .staged[data-restored='true']", timeout=5000)
    review.keyboard.press("Escape")

    marks = review.evaluate(
        "(i) => ({...window.__photocull.marksFor(window.__photocull.state.clusters[i])})",
        BIG,
    )
    assert marks[uuid] == "keep"


def test_reopening_the_sheet_starts_at_the_first_staged_photo(review):
    """The sheet is paged, and a page cursor that survives a close reopens the screen at
    row 61 with the first sixty photos nowhere on it — while the heading still says how
    many there are. The owner would confirm a set whose beginning they never saw, which
    is the one thing this screen exists to make impossible."""
    for index in range(24):
        decide(review, index, marks_for(review, index, cull=all_uuids(review, index)))
    open_sheet(review)
    first = review.evaluate(
        "() => document.querySelector('#sheet-rows .staged').dataset.uuid"
    )
    shown = review.evaluate("() => document.querySelectorAll('#sheet-rows .staged').length")
    assert shown == 60, "the fixture no longer fills more than one page"

    review.keyboard.press("Escape")
    review.wait_for_selector("#final-check", state="hidden", timeout=5000)
    open_sheet(review)

    assert (
        review.evaluate("() => document.querySelector('#sheet-rows .staged').dataset.uuid")
        == first
    )


def test_favourites_survive_a_reload_into_the_compare_view(review):
    """A review is designed to survive a closed browser, so after a reload the favourites
    of a decided cluster exist only in the session document. Seeding them from there is
    what stops re-recording that cluster from silently un-favouriting everything in it."""
    keeper = review.evaluate("(i) => window.__photocull.state.clusters[i].winner_uuid", BIG)
    decide(review, BIG, marks_for(review, BIG), favorites=[keeper])

    review.reload()
    review.wait_for_selector("#app:not([hidden])", timeout=15_000)

    favourites = review.evaluate(
        "(i) => [...window.__photocull.favoritesFor(window.__photocull.state.clusters[i])]",
        BIG,
    )
    assert favourites == [keeper]


# --- the gate --------------------------------------------------------------------------


def test_write_back_stays_unavailable_until_the_final_check_is_confirmed(review):
    decide(review, BIG, marks_for(review, BIG))
    open_sheet(review)

    assert api(review, "/api/final-check")["confirmed"] is False
    assert review.locator("#sheet-lock").inner_text().lower().startswith("write-back")
    assert "not" in review.locator("#sheet-lock").inner_text().lower()

    review.locator("#sheet-confirm").click()
    review.wait_for_selector("#final-check[data-confirmed='true']", timeout=5000)

    assert api(review, "/api/final-check")["confirmed"] is True


def test_pulling_a_photo_back_after_confirming_locks_write_back_again(review):
    """The confirmation names a set; changing the set on the very screen that confirmed
    it must take the authorisation with it."""
    decide(review, BIG, marks_for(review, BIG))
    open_sheet(review)
    review.locator("#sheet-confirm").click()
    review.wait_for_selector("#final-check[data-confirmed='true']", timeout=5000)

    review.locator("#sheet-rows .staged").first.click()
    review.wait_for_selector("#final-check[data-confirmed='false']", timeout=5000)

    assert api(review, "/api/final-check")["confirmed"] is False


def test_deciding_another_cluster_after_confirming_lapses_the_confirmation(review):
    """The failure a boolean flag would allow: confirm 19 photos, review more clusters,
    write back a set nobody ever saw."""
    decide(review, BIG, marks_for(review, BIG))
    open_sheet(review)
    review.locator("#sheet-confirm").click()
    review.wait_for_selector("#final-check[data-confirmed='true']", timeout=5000)
    review.keyboard.press("Escape")

    decide(review, PAIR, marks_for(review, PAIR))

    assert api(review, "/api/final-check")["confirmed"] is False


# --- the write-back button ----------------------------------------------------------
#
# Confirming used to be a dead end: the screen would say write-back was "unlocked" and
# then offer nothing to click. These exercise the button that actually closes the loop,
# against a fake writer so no test ever reaches for a real Photos library.


@pytest.fixture
def writable_server(tmp_path):
    """A live review server launched with `--allow-write-back`, writer swapped for a
    fake so this file never risks a real Photos.app call."""
    clusters = build_demo_clusters(tmp_path / "images")
    log = DecisionLog(tmp_path / "decisions.db")
    written: list[FakeWriter] = []

    def factory():
        writer = FakeWriter()
        written.append(writer)
        return writer

    launch = launcher.prepare(
        clusters,
        log=log,
        writeback=WritebackSettings(
            allow_write_back=True,
            writer=factory,
            # Both fixed and matching: `write_back` compares `analysed_library` against
            # `target_library()` and refuses on a mismatch, and the default
            # `target_library` reads this *real* Mac's Photos preferences, which has no
            # place in a test that must never touch anything outside `tmp_path`.
            analysed_library=LIBRARY,
            target_library=lambda: LIBRARY,
            # Left at the default (the owner's real ledger) once, by mistake, while
            # this fixture was still being built — every demo cluster uses the same
            # fixed uuids, so that run's rows made every later run see "already done"
            # forever. Scoped to `tmp_path` for exactly the reason `test_writeback.py`
            # already scopes it everywhere else.
            ledger_path=tmp_path / "writeback.db",
        ),
    )
    ready = threading.Event()
    thread = threading.Thread(
        target=launcher.serve,
        args=(launch,),
        kwargs={"poll_interval": 0.01, "on_ready": lambda _launch: ready.set()},
        daemon=True,
    )
    thread.start()
    assert ready.wait(timeout=20), "the writable review server never reported itself ready"
    try:
        yield SimpleNamespace(url=launch.url, written=written)
    finally:
        launch.lifetime.end()
        thread.join(timeout=10)
        log.close()


@pytest.fixture
def writable_review(page, writable_server):
    page.goto(writable_server.url)
    page.wait_for_selector("#app:not([hidden])", timeout=15_000)
    return page


def test_writeback_button_is_hidden_until_confirmed(writable_review):
    decide(writable_review, BIG, marks_for(writable_review, BIG))
    open_sheet(writable_review)

    assert writable_review.locator("#sheet-writeback").is_hidden()

    writable_review.locator("#sheet-confirm").click()
    writable_review.wait_for_selector("#final-check[data-confirmed='true']", timeout=5000)

    assert writable_review.locator("#sheet-writeback").is_visible()


def test_writeback_button_writes_the_confirmed_set_to_photos(writable_review, writable_server):
    decide(writable_review, BIG, marks_for(writable_review, BIG))
    open_sheet(writable_review)
    writable_review.locator("#sheet-confirm").click()
    writable_review.wait_for_selector("#final-check[data-confirmed='true']", timeout=5000)

    writable_review.locator("#sheet-writeback").click()
    writable_review.wait_for_selector("#sheet-writeback", state="hidden", timeout=5000)

    assert "Wrote back" in writable_review.locator("#sheet-lock").inner_text()
    (writer,) = writable_server.written
    album_calls = [call for call in writer.calls if call[0] == "add_to_album"]
    # BIG is 20 takes, one accepted keeper: the other 19 are the staged candidates.
    # The keeper gets no album at all (`keepers-get-no-album`).
    assert len([c for c in album_calls if c[1] == "Cull/Keepers"]) == 0
    assert len([c for c in album_calls if c[1] == "Cull/Candidates"]) == 19
    keyword_calls = [call for call in writer.calls if call[0] == "set_keywords"]
    assert len(keyword_calls) == 19


def test_writeback_button_click_without_allow_write_back_surfaces_the_refusal(review):
    """The ordinary `review` fixture launches without `--allow-write-back`. Confirming
    still works -- a dry run needs only that -- so the button appears the same as it
    would on a writable session; the frontend has no way to know the session's launch
    flags. Clicking it must surface the server's refusal rather than pretend it worked."""
    decide(review, BIG, marks_for(review, BIG))
    open_sheet(review)
    review.locator("#sheet-confirm").click()
    review.wait_for_selector("#final-check[data-confirmed='true']", timeout=5000)

    review.locator("#sheet-writeback").click()
    review.wait_for_selector("#message:not([hidden])", timeout=5000)
    assert "allow-write-back" in review.locator("#message").inner_text()


def test_the_review_keys_do_nothing_while_an_overview_is_open(review):
    """The defect an overlay invites, and the one nobody would see happen.

    The compare view is still mounted underneath: without a guard, `Space` under the
    contact sheet flips a mark on a cluster the owner is not looking at and `Enter`
    records it. Checked on the dashboard as well as the sheet, and the help overlay had
    the identical hole from Task 10 — it was merely harder to reach, since nobody leaves
    a shortcut list open."""
    marks = review.evaluate("() => ({...window.__photocull.marksFor(window.__photocull.state.clusters[0])})")

    for opener in ("d", "v", "?"):
        review.keyboard.press(opener)
        review.wait_for_timeout(120)
        review.keyboard.press(" ")
        review.keyboard.press("c")
        review.wait_for_timeout(60)
        after = review.evaluate(
            "() => ({...window.__photocull.marksFor(window.__photocull.state.clusters[0])})"
        )
        assert after == marks, f"a review key acted through the overlay opened by {opener!r}"
        review.keyboard.press("Escape")
        review.wait_for_timeout(80)

    # And the keys work again once everything is closed — a guard that never lifts would
    # pass every assertion above.
    review.keyboard.press(" ")
    review.wait_for_timeout(60)
    assert review.evaluate(
        "() => ({...window.__photocull.marksFor(window.__photocull.state.clusters[0])})"
    ) != marks


# --- the vocabulary the owner calibrated against ---------------------------------------


def test_a_staged_photo_says_how_close_it_is_to_the_keeper_in_words(review):
    """`calibrate.write_contact_sheet` established this phrasing and the owner reviewed
    200 clusters in it. A raw 0.132 means nothing on its own — it is the only number a
    cull decision actually rests on, so it is spelled out."""
    decide(review, BIG, marks_for(review, BIG))
    open_sheet(review)

    text = review.locator("#sheet-rows .staged").first.inner_text().lower()
    assert any(
        phrase in text
        for phrase in ("nearly identical", "very similar", "similar", "loosest match")
    ), text


# --- scale -----------------------------------------------------------------------------


STAGED_CLUSTERS = 1817


@pytest.fixture
def crowded_server(tmp_path):
    """A session the size of the real one: 1,817 decided clusters, all staged.

    Built by pointing many records at the *demo's* rasters rather than by writing 3,634
    files — the sheet's problem is how many images it asks the network for, which is a
    property of how many URLs it renders, not of how many distinct files exist behind
    them. The decisions are written straight to the log because 1,817 HTTP round trips
    would be measuring the test harness."""
    demo = build_demo_clusters(tmp_path / "images")
    rasters = [
        record.display_path for cluster in demo for record in cluster.records
    ]
    clusters = [
        make_cluster(
            [f"big-{i:05d}-a", f"big-{i:05d}-b"],
            day=i % 900,
            display_paths={
                f"big-{i:05d}-a": rasters[(2 * i) % len(rasters)],
                f"big-{i:05d}-b": rasters[(2 * i + 1) % len(rasters)],
            },
        )
        for i in range(STAGED_CLUSTERS)
    ]

    log = DecisionLog(tmp_path / "decisions.db")
    digest = config_digest(ClusterConfig(), None)
    settings = session_settings(ClusterConfig(), None)
    for index, cluster in enumerate(clusters):
        log.record(
            session_id="crowded",
            cluster_key=cluster_key(cluster),
            config_digest=digest,
            settings=settings,
            marks=[
                Mark(photo_uuid=f"big-{index:05d}-a", mark="keep"),
                Mark(photo_uuid=f"big-{index:05d}-b", mark="cull"),
            ],
        )

    launch = launcher.prepare(clusters, log=log)
    ready = threading.Event()
    thread = threading.Thread(
        target=launcher.serve,
        args=(launch,),
        kwargs={"poll_interval": 0.01, "on_ready": lambda _launch: ready.set()},
        daemon=True,
    )
    thread.start()
    assert ready.wait(timeout=60), "the crowded review server never reported itself ready"
    try:
        yield launch
    finally:
        launch.lifetime.end()
        thread.join(timeout=10)
        log.close()


def test_the_sheet_stays_usable_at_the_real_staged_count(page, crowded_server):
    """1,817 staged photos — the count measured on the owner's own library at the 0.48
    cut. The naive sheet is correct and unusable: 1,817 rows of two `<img>` each is
    ~3,600 image requests against a server that reads a file per request.

    Asserted as *requests in flight*, not as a stopwatch, because a slow machine should
    not fail this and a sheet that asks for everything should fail it everywhere.

    **The resource-timing buffer has to be raised first, or this assertion is asleep.**
    Chromium keeps 250 entries by default and silently drops the rest, so a sheet that
    requested all 3,634 images would report ~250 and pass a bound of 200-odd on the
    strength of the buffer overflowing. Found by measuring: a run that scrolled five
    pages reported exactly 241 requests with lazy loading on *and* off, which is the
    buffer, not the browser."""
    page.add_init_script("performance.setResourceTimingBufferSize(20000);")
    page.goto(crowded_server.url)
    page.wait_for_selector("#app:not([hidden])", timeout=60_000)

    page.keyboard.press("v")
    page.wait_for_selector("#final-check[data-loading='false']", timeout=30_000)
    page.wait_for_selector("#sheet-rows .staged", timeout=30_000)

    header = page.locator("#sheet-head").inner_text()
    assert str(STAGED_CLUSTERS) in header, header

    rows = page.evaluate("() => document.querySelectorAll('#sheet-rows .staged').length")
    assert rows < 200, f"{rows} rows rendered to open a sheet of {STAGED_CLUSTERS}"

    page.wait_for_timeout(400)
    images = page.evaluate(
        """() => performance.getEntriesByType('resource')
             .filter((entry) => entry.name.includes('/api/images/')).length"""
    )
    # Tied to what is on screen rather than to a round number: the property is "the sheet
    # asks for what it drew", and the failure it rules out is asking for the library.
    assert images <= 2 * rows + 20, f"{images} image requests for {rows} rendered rows"


def test_a_narrow_window_asks_only_for_the_photographs_it_shows(page, crowded_server):
    """Lazy images, measured where they actually do something.

    At 1440px the sheet lays its clusters out as a card grid, so one page of 60 rows is
    about two and a half screens — inside Chromium's lazy threshold, where lazy and eager
    both request all 120 images and an assertion here would be inert. At one column the
    same page is ~11,000px tall, and the difference is **32 requests against 120**. Which
    is the useful statement of what each mechanism buys: windowing bounds the page, and
    lazy bounds what a *tall* page fetches before it is scrolled to."""
    page.add_init_script("performance.setResourceTimingBufferSize(20000);")
    page.set_viewport_size({"width": 520, "height": 900})
    page.goto(crowded_server.url)
    page.wait_for_selector("#app:not([hidden])", timeout=60_000)

    page.keyboard.press("v")
    page.wait_for_selector("#final-check[data-loading='false']", timeout=30_000)
    page.wait_for_selector("#sheet-rows .staged", timeout=30_000)
    page.wait_for_timeout(600)

    rendered = page.evaluate("() => document.querySelectorAll('#sheet-rows img').length")
    requested = page.evaluate(
        """() => performance.getEntriesByType('resource')
             .filter((entry) => entry.name.includes('/api/images/')).length"""
    )
    assert rendered >= 100, "the page is no longer big enough for this to mean anything"
    assert requested < rendered / 2, f"{requested} requests for {rendered} rendered images"


def test_scrolling_the_crowded_sheet_reaches_further_rows(page, crowded_server):
    """The other half of windowing: the rows that are not drawn have to arrive when the
    owner scrolls to them, or the sheet is not a view of the staged set at all — it is a
    view of its first page, and the confirmation below it would be a lie."""
    page.goto(crowded_server.url)
    page.wait_for_selector("#app:not([hidden])", timeout=60_000)
    page.keyboard.press("v")
    page.wait_for_selector("#final-check[data-loading='false']", timeout=30_000)
    page.wait_for_selector("#sheet-rows .staged", timeout=30_000)

    first = page.evaluate("() => document.querySelectorAll('#sheet-rows .staged').length")
    page.evaluate(
        "() => document.querySelector('#sheet-scroll').scrollTo(0, 10 ** 7)"
    )
    page.wait_for_function(
        "(n) => document.querySelectorAll('#sheet-rows .staged').length > n",
        arg=first,
        timeout=15_000,
    )
