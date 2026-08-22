"""Task 10: the compare view, driven in a real browser against the demo library.

Every test here runs against `--demo` — no Photos library, no Full Disk Access, no
Vision — which is the whole reason the demo dataset exists. It is also why these tests
can assert *which* photograph is in *which* pane: the demo's rasters are a known colour
and checker frequency per photo, and (since this task) a photo's analysis copy and its
display copy are different files, so "the wrong raster is on screen" is a visible
failure rather than an invisible one
(`the-demo-library-cannot-tell-the-two-derivatives-apart`).

The server is a real uvicorn socket rather than `TestClient`, for the reason Task 6
recorded: `TestClient` never opens one, and the security gate, the cookie handoff and
the browser's own `Sec-Fetch-Site` header only exist on the real path.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.ui

# `review_server` and `review` live in `conftest.py` — `test_ui_magnify.py` needs the
# same live app plus a second page at another device pixel ratio.


# --- helpers ------------------------------------------------------------------------


def pane(page, side):
    return page.locator(f"#pane-{side}")


def badge_text(page, side):
    return pane(page, side).locator("[data-badge]").inner_text()


def photo_uuid(page, side):
    """The uuid the pane is actually showing, read off the served image URL."""
    src = pane(page, side).locator("[data-photo]").get_attribute("src")
    return src.rsplit("/", 1)[-1]


def session_document(page):
    return page.evaluate("() => window.__photocull.state.doc")


# --- the layout promise ---------------------------------------------------------------


def test_the_keeper_and_the_challenger_render_at_equal_size(review):
    """`compare-side-by-side-with-sync-zoom`'s load-bearing requirement.

    Asserted as equal bounding boxes rather than "both are present", because the failure
    this rule exists to prevent is not a missing pane — it is a large hero beside a small
    alternate, which biases the eye toward the hero regardless of which photograph is
    better. Both are `<img>` elements, so this compares what the owner actually looks at
    and not merely the boxes around them.
    """
    keeper = pane(review, "keeper").locator("[data-photo]").bounding_box()
    challenger = pane(review, "challenger").locator("[data-photo]").bounding_box()

    assert keeper is not None and challenger is not None
    assert keeper["width"] == challenger["width"]
    assert keeper["height"] == challenger["height"]
    assert keeper["width"] > 100, "a pane collapsed to nothing would pass an equality test"


def test_the_panes_stay_equal_on_every_cluster_in_the_demo(review):
    """The invariant, swept rather than sampled.

    Equality on the opening cluster is easy; it is the clusters whose two panes hold
    *different amounts of text* that break it — a close call annotates only the keeper,
    an evicted take replaces one image with a sentence, and the keeper alone has no
    distance-to-keeper value. Each of those made one pane taller and its frame shorter
    until both panes were given structurally identical rows.
    """
    doc = session_document(review)
    unequal = []
    for index in range(len(doc["clusters"])):
        review.evaluate("(i) => window.__photocull.goTo(i)", index)
        review.wait_for_timeout(30)
        left = pane(review, "keeper").locator("[data-frame]").bounding_box()
        right = pane(review, "challenger").locator("[data-frame]").bounding_box()
        if (left["width"], left["height"]) != (right["width"], right["height"]):
            unequal.append((index, left, right))
    assert unequal == [], f"{len(unequal)} clusters render their panes at different sizes"


def test_no_cluster_pushes_the_close_call_below_the_fold(review):
    """Found on the real library, not the demo: the caveat scrolled off the screen.

    A cluster carrying two annotations, a nine-row score table and a close call ran past
    1440x900 — and the part below the fold was the sentence saying the pick is a coin
    flip. The frames are now sized from the space left over, so the whole cluster fits
    whatever else is on screen.

    The first version of this test measured `document.body.scrollHeight` and was asleep:
    `#app` carries `overflow: auto` as a small-window safety valve, so content overflows
    *inside* it and the body never grows. The mutation pass caught that — restoring the
    fixed-height frames passed the test — which is the whole reason this project runs
    one. It now measures the scrolling element itself, and checks that the note is
    actually on screen.
    """
    doc = session_document(review)
    height = review.evaluate("() => window.innerHeight")
    overflowing = []
    off_screen = []
    for index in range(len(doc["clusters"])):
        review.evaluate("(i) => window.__photocull.goTo(i)", index)
        review.wait_for_timeout(30)

        overflow = review.evaluate(
            "() => document.getElementById('app').scrollHeight"
            " - document.getElementById('app').clientHeight"
        )
        if overflow > 0:
            overflowing.append((index, overflow))

        note = pane(review, "keeper").locator("[data-close-call]")
        if note.is_visible():
            box = note.bounding_box()
            if box["y"] + box["height"] > height:
                off_screen.append(index)

    assert overflowing == [], f"clusters running past the fold: {overflowing}"
    assert off_screen == [], f"the close call is below the fold on clusters {off_screen}"


def test_the_two_panes_hold_different_photographs(review):
    """A comparison screen showing the same photo twice would satisfy every size test."""
    assert photo_uuid(review, "keeper") != photo_uuid(review, "challenger")


def test_the_pane_shows_the_display_raster_not_the_analysis_one(review):
    """The wrong-raster bug, now visible because the demo can tell the two apart.

    The display copy is 1024px and the analysis copy 480px, so serving the wrong file
    halves the resolution of the one screen whose entire job is judging which take is
    sharper. Read off `naturalWidth` — the decoded image, not the CSS box.
    """
    natural = review.evaluate(
        "() => document.querySelector('#pane-keeper [data-photo]').naturalWidth"
    )
    assert natural == 1024


# --- stepping through the takes ---------------------------------------------------------


def go_to_a_cluster_of_at_least(review, size):
    """Move to a cluster with at least `size` takes, and say which one.

    Not a convenience. 17 of the demo's 24 clusters are pairs — exactly the real
    library's shape — so a stepping test written against the opening cluster has one
    challenger and cannot step: it passes without ever exercising the thing it names.
    This suite hit that on its first run, which is the eighth distinct sleeping-test
    shape this project has found.
    """
    doc = session_document(review)
    index = next(i for i, c in enumerate(doc["clusters"]) if len(c["members"]) >= size)
    review.evaluate("(i) => window.__photocull.goTo(i)", index)
    review.wait_for_timeout(50)
    return doc["clusters"][index]


def go_to_a_pair(review):
    """Move to a two-take cluster — the library's median shape, ~71% of real clusters.

    The mirror of the helper above: a pane-selection test written against a four-take
    cluster would pass on a build where the arrows still stepped takes, because there
    the two actions both have somewhere to go.
    """
    doc = session_document(review)
    index = next(i for i, c in enumerate(doc["clusters"]) if len(c["members"]) == 2)
    review.evaluate("(i) => window.__photocull.goTo(i)", index)
    review.wait_for_timeout(50)
    return doc["clusters"][index]


def focused(page):
    """Which pane the keystrokes are aimed at, read off the class the border comes from."""
    sides = [
        side
        for side in ("keeper", "challenger")
        if "focused" in (pane(page, side).get_attribute("class") or "")
    ]
    assert len(sides) == 1, f"exactly one pane must look focused, not {sides}"
    return sides[0]


def test_the_arrows_select_the_pane(review):
    """The owner: *"i can't move the selection with the arrow and i need to click."*

    `←`/`→` used to step the challenger through the other takes — an action that does not
    exist on the ~71% of clusters that are pairs, so on nearly three clusters in four both
    arrow keys were dead while the one thing the owner wanted to do sat on `Tab`. Moving
    between the two photographs is possible on 100% of clusters, so that is what the
    arrows do.

    They select rather than toggle: `←` is the left pane every time, not "the other one".
    """
    go_to_a_pair(review)

    review.keyboard.press("ArrowLeft")
    assert focused(review) == "keeper"
    review.keyboard.press("ArrowLeft")
    assert focused(review) == "keeper", "the arrows select a side, they do not alternate"

    review.keyboard.press("ArrowRight")
    assert focused(review) == "challenger"
    review.keyboard.press("ArrowRight")
    assert focused(review) == "challenger"


def test_tab_still_switches_panes(review):
    """Kept as an alias, so a habit built over a session of reviewing still works."""
    go_to_a_pair(review)
    review.keyboard.press("ArrowLeft")

    review.keyboard.press("Tab")
    assert focused(review) == "challenger"
    review.keyboard.press("Tab")
    assert focused(review) == "keeper"


def test_up_and_down_step_the_challenger_through_the_other_takes(review):
    """The keeper stays pinned; only the challenger moves.

    Stepping only exists on the 29% of clusters with three or more takes, which is why it
    is on the second pair of keys rather than the first.
    """
    go_to_a_cluster_of_at_least(review, 4)

    keeper_before = photo_uuid(review, "keeper")
    first = photo_uuid(review, "challenger")

    review.keyboard.press("ArrowDown")
    second = photo_uuid(review, "challenger")

    assert first != second, "the challenger did not advance"
    assert photo_uuid(review, "keeper") == keeper_before, "the keeper pane must stay put"

    review.keyboard.press("ArrowUp")
    assert photo_uuid(review, "challenger") == first


def test_the_take_counter_names_the_position(review):
    """A cluster of 20 needs to say where in it you are.

    Read with `text_content`, not `inner_text`: the role label is upper-cased in CSS, so
    `inner_text` returns the rendered "TAKE 1 OF 4" and would pin the stylesheet rather
    than the string the app produces.
    """
    entry = go_to_a_cluster_of_at_least(review, 4)
    others = len(entry["members"]) - 1

    role = pane(review, "challenger").locator("[data-role]").text_content()
    assert role == f"Take 1 of {others}"

    review.keyboard.press("ArrowDown")
    assert pane(review, "challenger").locator("[data-role]").text_content() == (
        f"Take 2 of {others}"
    )


# --- marking ------------------------------------------------------------------------------


def test_space_flips_the_challenger_and_the_badge_follows(review):
    """The badge is the only feedback that a keystroke landed, so it is pinned."""
    assert badge_text(review, "challenger").strip() == "CULL"

    review.keyboard.press(" ")
    assert badge_text(review, "challenger").strip() == "KEEP"

    review.keyboard.press(" ")
    assert badge_text(review, "challenger").strip() == "CULL"


def test_an_ambiguous_cluster_opens_with_its_winner_marked_keep(review):
    """`every-cluster-opens-with-a-suggestion`, asserted where it is easiest to break.

    This assertion was **inverted on 2026-08-15 by owner instruction**: the plan
    originally required an ambiguous cluster to start with nothing marked keep. The
    owner's ruling is that the algorithm recommends and the UI decides, so a close call
    opens exactly like a confident cluster — and the closeness is said in prose instead
    of expressed as a withheld suggestion, which the owner would have had to already
    understand in order to read.
    """
    doc = session_document(review)
    index = next(i for i, c in enumerate(doc["clusters"]) if c["is_ambiguous"])
    review.evaluate("(i) => window.__photocull.goTo(i)", index)
    review.wait_for_timeout(50)

    assert badge_text(review, "keeper").strip() == "KEEP"
    assert badge_text(review, "challenger").strip() == "CULL"


def test_the_close_call_is_attached_to_the_proposal_not_to_the_warnings(review):
    """Where the note sits is the decision, not whether it exists.

    An ambiguous cluster is byte-identical in shape to a confident one now that every
    cluster opens with a suggestion, so a close call listed among the cluster's warnings
    would read as background noise while the proposal beside it reads as considered.
    """
    doc = session_document(review)
    index = next(i for i, c in enumerate(doc["clusters"]) if c["is_ambiguous"])
    review.evaluate("(i) => window.__photocull.goTo(i)", index)
    review.wait_for_timeout(50)

    close_call = pane(review, "keeper").locator("[data-close-call]")
    assert close_call.is_visible()
    text = close_call.inner_text()
    assert "close call" in text.lower()
    # Prose, not an icon: a sentence long enough to say what it means.
    assert len(text.split()) > 8
    assert review.locator("#notes .note[data-code='ambiguous']").count() == 0

    # The challenger reserves the same space so the frames stay equal, but the note is
    # the keeper's — a reader must not see it twice, and it must not read as a claim
    # about the challenger.
    assert not pane(review, "challenger").locator("[data-close-call]").is_visible()

    # The note says nothing about where it sits. It used to say "the suggestion below",
    # which pointed at the proposal until this task moved it under the scores.
    assert "below" not in text.lower() and "above" not in text.lower()


def test_the_two_score_tables_stay_aligned_when_the_keeper_carries_a_note(review):
    """Found by looking at the screen, not by a failing test.

    The close call renders inside the keeper pane, and while it sat between the image
    and the numbers it pushed that pane's score rows down — so on the **40.1%** of
    clusters that are ambiguous, `sharpness 0.690` and `sharpness 0.720` sat on
    different lines and had to be hunted for. That is the exact failure mode the plan
    warned this task has and no test catches: correct, tested, and unpleasant. The note
    now sits below the scores, so both comparison surfaces line up.
    """
    doc = session_document(review)
    index = next(i for i, c in enumerate(doc["clusters"]) if c["is_ambiguous"])
    review.evaluate("(i) => window.__photocull.goTo(i)", index)
    review.wait_for_timeout(80)

    assert pane(review, "keeper").locator("[data-close-call]").is_visible()

    keeper = pane(review, "keeper").locator("[data-scores]").bounding_box()
    challenger = pane(review, "challenger").locator("[data-scores]").bounding_box()
    assert keeper["y"] == challenger["y"], "the note knocked the score tables out of line"

    # The images have to stay aligned too — the fix must not have simply moved the
    # problem up one element.
    keeper_photo = pane(review, "keeper").locator("[data-photo]").bounding_box()
    other_photo = pane(review, "challenger").locator("[data-photo]").bounding_box()
    assert keeper_photo["y"] == other_photo["y"]


def test_the_photographs_grow_to_fill_a_taller_window(review):
    """The other half of the layout fix, and the half the mutation pass had to find.

    Reverting the frames to a fixed `min(58vh, 30rem)` did **not** bring back the
    clipping — a fixed-height flex item still shrinks, so the viewport-height column
    alone prevents that. What the fixed height loses is the opposite case: on a large
    display the frames stop at 480px and the rest of the window is empty, on the one
    screen whose job is looking closely at two photographs. So the property worth
    pinning is not "it never overflows" but "it uses what it is given", and it is only
    observable where there is spare space to use.
    """
    review.set_viewport_size({"width": 1280, "height": 720})
    review.wait_for_timeout(80)
    short = pane(review, "keeper").locator("[data-frame]").bounding_box()["height"]

    review.set_viewport_size({"width": 1280, "height": 1240})
    review.wait_for_timeout(80)
    tall = pane(review, "keeper").locator("[data-frame]").bounding_box()["height"]

    grew = tall - short
    # The window gained 520px; a frame capped at 30rem could only gain 63 of them.
    assert grew > 300, f"the frames gained only {grew:.0f}px of 520"

    # And the two are still equal at the new size.
    other = pane(review, "challenger").locator("[data-frame]").bounding_box()["height"]
    assert tall == other


def test_a_pair_says_the_other_take_rather_than_take_1_of_1(review):
    """The most-shown label in the app: 17 of 24 demo clusters are pairs.

    "Take 1 of 1" is accurate and reads as a counter that broke.
    """
    doc = session_document(review)
    index = next(i for i, c in enumerate(doc["clusters"]) if len(c["members"]) == 2)
    review.evaluate("(i) => window.__photocull.goTo(i)", index)
    review.wait_for_timeout(50)

    assert pane(review, "challenger").locator("[data-role]").text_content() == (
        "The other take"
    )


def test_the_header_tallies_this_clusters_marks(review):
    """A 20-take cluster is 19 challengers deep with no other sense of standing."""
    entry = go_to_a_cluster_of_at_least(review, 9)
    header = review.locator("#position").text_content()
    assert f"{len(entry['members'])} takes" in header
    assert "keeping 1, staging" in header

    review.keyboard.press(" ")  # keep this challenger too
    review.wait_for_timeout(50)
    assert "keeping 2, staging" in review.locator("#position").text_content()


def test_the_footer_hint_is_scannable_and_comes_from_the_keymap(review):
    """One line, and still no second source of truth for the keys."""
    hint = review.locator("#hint").text_content()
    keymap = review.evaluate("() => window.__photocull.KEYMAP")

    for row in keymap:
        if row["hint"] is None:
            continue
        assert row["hint"] in hint, f"{row['label']} is missing from the footer"

    # And the other direction, listed literally rather than derived from the same table
    # the footer renders from. The loop above cannot catch a hint that went *missing*: a
    # `null` hint is skipped, so a mutation dropping `↑ ↓ step` passed it while the key
    # disappeared from the only always-visible legend in the app. Discoverability is the
    # whole reason the arrows had to be remapped — a binding nobody can find is a binding
    # nobody uses.
    for expected in ("← →", "pane", "↑ ↓", "step", "keep/cull", "C", "cull all",
                     "F", "favourite", "Enter", "record", "undo", "keys",
                     "D", "overview", "V", "check"):
        assert expected in hint, f"the footer no longer offers {expected!r}: {hint!r}"
    # One rendered line. Measured as content height against line height rather than as
    # the element's box, which is mostly padding — the first version of this assertion
    # compared 66px of padded box to a 40px guess and failed on a perfectly good layout.
    #
    # Measured at two widths rather than one. This replaces a `len(hint) < 120` proxy
    # that stood beside the same measurement: adding `C cull all` took the footer to 127
    # characters and tripped the proxy while the footer still rendered on one line, so
    # the proxy was answering a question the line below answers properly. 1000px is the
    # width `the-evidence-is-its-own-grid-row` established as a real narrow case.
    for width in (1280, 1000):
        review.set_viewport_size({"width": width, "height": 900})
        review.wait_for_timeout(50)
        lines = review.evaluate(
            """() => {
                const node = document.getElementById('hint');
                const style = getComputedStyle(node);
                const content =
                    node.clientHeight
                    - parseFloat(style.paddingTop)
                    - parseFloat(style.paddingBottom);
                return Math.round(content / parseFloat(style.lineHeight));
            }"""
        )
        assert lines == 1, f"the footer wraps onto {lines} lines at {width}px"


def evidence_text(page, side):
    return page.locator(f"#evidence-{side}").inner_text()


def test_the_evidence_panel_says_why_under_each_photograph(review):
    """Task 11. The owner's report was "I don't understand what the values are", and the
    score table answered it with more numbers. This is the sentence that says which
    photograph `sharpness 0.912` makes the sharper one."""
    keeper = evidence_text(review, "keeper")
    challenger = evidence_text(review, "challenger")

    # A reason, not a restated number: prose long enough to say something.
    assert len(keeper.split()) > 8, keeper
    # And the two sides say different things about different photographs.
    assert keeper != challenger
    assert "scorer's pick" in keeper or "Too close to call" in keeper, keeper
    assert "Behind the proposed keeper" in challenger, challenger


def test_the_evidence_never_prints_a_raw_score_as_a_reason(review):
    """`sharpness` is a ratio to the sharpest take and a laplacian variance underneath;
    neither is a thing to show a person. The panel states ratios as ratios."""
    doc = session_document(review)
    with_sharpness = next(
        i for i, c in enumerate(doc["clusters"]) if c["sharpness_available"]
    )
    review.evaluate("(i) => window.__photocull.goTo(i)", with_sharpness)
    review.wait_for_timeout(50)
    text = evidence_text(review, "keeper") + evidence_text(review, "challenger")
    assert "×" in text, text


def test_the_evidence_follows_the_challenger_through_the_takes(review):
    """A panel that kept explaining take 1 while take 3 is on screen would be worse than
    no panel: it is wrong in a way the reader has no way to notice."""
    doc = session_document(review)
    index = next(i for i, c in enumerate(doc["clusters"]) if len(c["members"]) > 2)
    review.evaluate("(i) => window.__photocull.goTo(i)", index)
    review.wait_for_timeout(50)

    first = evidence_text(review, "challenger")
    keeper_before = evidence_text(review, "keeper")
    review.keyboard.press("ArrowDown")
    review.wait_for_timeout(50)

    assert evidence_text(review, "challenger") != first
    # The keeper is pinned for the whole cluster, so its explanation must not move.
    assert evidence_text(review, "keeper") == keeper_before


def test_the_ranking_basis_is_rendered_once_for_the_cluster(review):
    """Not once per pane. `dropped_criteria` is non-empty for 90.3% of real clusters, so
    a per-take rendering says the same sentence twice on nine screens in ten."""
    doc = session_document(review)
    index = next(i for i, c in enumerate(doc["clusters"]) if c["ranking_basis"])
    review.evaluate("(i) => window.__photocull.goTo(i)", index)
    review.wait_for_timeout(50)

    basis = review.locator("#basis")
    assert basis.is_visible()
    assert "could not be measured" in basis.inner_text()
    for side in ("keeper", "challenger"):
        assert "could not be measured" not in evidence_text(review, side)

    # And it is gone from the layout, not merely empty, on a cluster that measured
    # everything. `is_visible()` cannot tell those apart — an empty <p> reports invisible
    # while still holding a grid row and its 1rem gap, taking that height off the
    # photographs. A mutation forcing it always-shown survived an `is_visible` assertion;
    # a null bounding box is `display: none` and nothing else.
    clean = next(i for i, c in enumerate(doc["clusters"]) if not c["ranking_basis"])
    review.evaluate("(i) => window.__photocull.goTo(i)", clean)
    review.wait_for_timeout(50)
    assert basis.bounding_box() is None


def test_the_panes_stay_equal_in_a_narrower_window(review):
    """The evidence sweep the default viewport cannot do.

    Both panes always receive the *same number* of sentences — the criteria cited are
    symmetric and the standing line is one either way — so at 1280px and 1440px the two
    blocks wrap identically and any layout would pass. Measured across the demo, they
    first diverge when a column falls under about 500px: **10 of 24 clusters at 1000px
    and 14 of 24 at 820px**, by 21px, because "Sharpest of the 2 takes, though only just
    — 1.02x the detail of the softest" wraps to three lines where "Softer than the
    sharpest take — 0.98x its detail" wraps to two.

    That is why the evidence is its own grid row rather than a block inside each figure.
    Moving it into the figures passes every other test in this file — this is the one
    that fails.
    """
    review.set_viewport_size({"width": 1000, "height": 900})
    review.wait_for_timeout(80)
    doc = session_document(review)
    unequal = []
    for index in range(len(doc["clusters"])):
        review.evaluate("(i) => window.__photocull.goTo(i)", index)
        review.wait_for_timeout(30)
        left = pane(review, "keeper").locator("[data-frame]").bounding_box()
        right = pane(review, "challenger").locator("[data-frame]").bounding_box()
        if (left["width"], left["height"]) != (right["width"], right["height"]):
            unequal.append((index, left["height"], right["height"]))
    assert unequal == [], f"{len(unequal)} clusters render their panes at different sizes"


def test_no_ambiguous_cluster_announces_a_pick_in_its_evidence(review):
    """Swept, not sampled. The close-call note and the evidence panel sit two lines
    apart, so prose reading as a verdict here contradicts the caveat above it — and
    40.1% of the real library is this case."""
    doc = session_document(review)
    contradictions = []
    for index, cluster in enumerate(doc["clusters"]):
        if not cluster["is_ambiguous"]:
            continue
        review.evaluate("(i) => window.__photocull.goTo(i)", index)
        review.wait_for_timeout(30)
        text = evidence_text(review, "keeper")
        if "scorer's pick" in text or "Too close to call" not in text:
            contradictions.append((index, text))
    assert contradictions == [], contradictions


def test_every_cluster_in_the_demo_explains_both_of_its_panes(review):
    """A silent panel on some cluster shape is the failure that would never be noticed —
    it looks exactly like a cluster with nothing to say."""
    doc = session_document(review)
    silent = []
    for index, cluster in enumerate(doc["clusters"]):
        review.evaluate("(i) => window.__photocull.goTo(i)", index)
        review.wait_for_timeout(30)
        for side in ("keeper", "challenger"):
            if not evidence_text(review, side).strip():
                silent.append((index, side))
    assert silent == [], silent


def test_sub_scores_show_under_each_photo(review):
    """The numbers themselves; the sentences beside them are the evidence panel above."""
    for side in ("keeper", "challenger"):
        scores = pane(review, side).locator("[data-scores]")
        assert "total" in scores.inner_text()

    # An absent measurement is said in words rather than printed as 0.000 — 43.9% of
    # real sub-scores are absent, so this is the common case, not an edge one.
    doc = session_document(review)
    index = next(
        i
        for i, c in enumerate(doc["clusters"])
        if any(v is None for m in c["members"] for v in m["sub_scores"].values())
    )
    review.evaluate("(i) => window.__photocull.goTo(i)", index)
    review.wait_for_timeout(50)
    assert "not measured" in pane(review, "keeper").locator("[data-scores]").inner_text()


# --- deciding -------------------------------------------------------------------------------


def test_enter_advances_and_the_decision_survives_a_reload(review, review_server):
    """The one test that proves the decision reached SQLite and not just the DOM."""
    doc = session_document(review)
    first_key = doc["clusters"][0]["cluster_key"]

    review.keyboard.press("Enter")
    review.wait_for_function(
        "() => window.__photocull.state.index === 1", timeout=10_000
    )

    review.goto(review_server.url)
    review.wait_for_selector("#app:not([hidden])", timeout=15_000)

    reloaded = review.evaluate("() => Object.keys(window.__photocull.state.doc.decided)")
    assert first_key in reloaded, "the decision did not survive the reload"
    # And the session resumes where the owner left off, not at the top.
    assert review.evaluate("() => window.__photocull.state.index") == 1


def test_a_cluster_can_end_with_several_keepers(review, review_server):
    """`cull-selection-is-propose-and-adjust`, end to end.

    The one-winner model could not express "keep 2 of 5" at all. Here it has to survive
    the round trip to the decision log and back.
    """
    doc = session_document(review)
    index = next(i for i, c in enumerate(doc["clusters"]) if len(c["members"]) >= 4)
    key = doc["clusters"][index]["cluster_key"]
    review.evaluate("(i) => window.__photocull.goTo(i)", index)
    review.wait_for_timeout(50)

    review.keyboard.press(" ")  # keep the first challenger too
    review.keyboard.press("ArrowDown")
    review.keyboard.press(" ")  # and the second
    review.keyboard.press("Enter")
    review.wait_for_timeout(300)

    state = review.evaluate(
        "async (key) => (await (await fetch(`/api/clusters/${key}`,"
        " {credentials: 'same-origin'})).json()).decision",
        key,
    )
    assert len(state["kept"]) == 3, "three keepers should have survived the round trip"


def test_u_undoes_the_cluster_just_recorded(review):
    doc = session_document(review)
    first_key = doc["clusters"][0]["cluster_key"]

    review.keyboard.press("Enter")
    review.wait_for_function("() => window.__photocull.state.index === 1", timeout=10_000)

    review.keyboard.press("u")
    review.wait_for_function(
        "(key) => !window.__photocull.state.decided.has(key)",
        arg=first_key,
        timeout=10_000,
    )

    # Undo returns you to the cluster it undid, or you cannot see what you undid.
    assert review.evaluate("() => window.__photocull.state.index") == 0


def marks_now(page):
    """The marks the client holds for the cluster on screen."""
    return page.evaluate(
        "() => { const p = window.__photocull;"
        " return p.marksFor(p.state.clusters[p.state.index]); }"
    )


def test_c_stages_every_take_in_the_cluster(review):
    """`a-whole-cluster-may-be-culled`, and the reason it needs a key of its own.

    The owner hit this while reviewing: marking five bad frames `cull` by hand is five
    keystrokes, and it used to end at `Enter` in a 422 — the work happened, then the
    refusal. One key, and the decision is accepted.

    **Two takes are kept first, and that is what gives this test teeth.** The scorer
    proposes `cull` for every take but its winner, so a cull-all that reached only the
    two photographs on screen would leave the cluster all-`cull` anyway and pass — which
    is what the first version of this test did, caught by mutation rather than by
    reading it.
    """
    doc = session_document(review)
    index = next(i for i, c in enumerate(doc["clusters"]) if len(c["members"]) >= 4)
    key = doc["clusters"][index]["cluster_key"]
    review.evaluate("(i) => window.__photocull.goTo(i)", index)
    review.wait_for_timeout(50)

    size = len(doc["clusters"][index]["members"])
    review.keyboard.press(" ")  # keep this challenger
    review.keyboard.press("ArrowDown")  # and step away from it, so it is off screen
    review.wait_for_timeout(30)
    on_screen = {
        photo_uuid(review, "keeper"), photo_uuid(review, "challenger")
    }
    kept_off_screen = [
        uuid
        for uuid, mark in marks_now(review).items()
        if mark == "keep" and uuid not in on_screen
    ]
    assert kept_off_screen, "the fixture must leave a kept take out of both panes"

    review.keyboard.press("c")
    review.wait_for_timeout(50)
    assert set(marks_now(review).values()) == {"cull"}
    # And the tally says so on screen, which is the only feedback for the takes that are
    # not in one of the two panes.
    assert f"keeping 0, staging {size}" in review.locator("#position").inner_text()

    review.keyboard.press("Enter")
    review.wait_for_function(
        "(i) => window.__photocull.state.index === i + 1", arg=index, timeout=10_000
    )
    decision = review.evaluate(
        "async (key) => (await (await fetch(`/api/clusters/${key}`,"
        " {credentials: 'same-origin'})).json()).decision",
        key,
    )
    assert decision["kept"] == [], "the server should have accepted a keeperless cluster"
    assert len(decision["staged"]) >= 4


def test_c_pressed_twice_puts_back_what_was_there(review):
    """A mis-press costs one keystroke, not N.

    It restores the marks that were on screen rather than the scorer's proposal, because
    the owner may have already decided several takes by hand before pressing it — and
    silently discarding that work is the failure this exists to avoid.
    """
    doc = session_document(review)
    index = next(i for i, c in enumerate(doc["clusters"]) if len(c["members"]) >= 4)
    review.evaluate("(i) => window.__photocull.goTo(i)", index)
    review.wait_for_timeout(50)

    review.keyboard.press(" ")  # keep the first challenger, so the marks are not the proposal
    review.wait_for_timeout(30)
    before = marks_now(review)
    assert set(before.values()) != {"cull"}, "the fixture must not open already all-cull"

    review.keyboard.press("c")
    review.wait_for_timeout(30)
    review.keyboard.press("c")
    review.wait_for_timeout(30)

    assert marks_now(review) == before


def test_culling_the_whole_cluster_drops_a_favourite_it_would_contradict(review):
    """`a-favourite-may-not-be-staged` is unchanged, so cull-all has to honour it.

    Left alone, `F` then `c` builds a submission the server refuses at 422 — a favourite
    on a staged photo — which is the same "work first, refusal at Enter" shape the owner
    complained about, arriving by a different route.
    """
    # The pane the review opens on is the challenger, which the scorer proposes to cull;
    # favouriting it is already refused, so the favourite has to go on the keeper.
    review.keyboard.press("Tab")
    review.keyboard.press("f")
    review.wait_for_timeout(30)
    assert review.evaluate(
        "() => { const p = window.__photocull;"
        " return p.favoritesFor(p.state.clusters[p.state.index]).size; }"
    ) == 1

    review.keyboard.press("c")
    review.wait_for_timeout(30)
    review.keyboard.press("Enter")
    review.wait_for_function("() => window.__photocull.state.index === 1", timeout=10_000)

    assert review.locator("#message").is_hidden(), "the decision was refused, not recorded"


# --- favourites -------------------------------------------------------------------------------


def test_f_records_favourite_intent_without_writing_to_photos(review):
    """`F` marks *for write-back*. Nothing may reach the library during a review.

    The structural guarantee is stronger than this test: `photoscript` is not imported
    anywhere this process can reach it, which `tests/test_guardrails.py` enforces with
    an AST walk. What is asserted here is the other half — that the intent is carried on
    the submission rather than dropped.
    """
    with review.expect_request("**/decision") as request:
        review.keyboard.press(" ")  # keep the challenger, so it may be favourited
        review.keyboard.press("f")
        assert "favourite" in badge_text(review, "challenger").lower()
        review.keyboard.press("Enter")

    body = request.value.post_data_json
    assert len(body["favorites"]) == 1
    assert body["favorites"][0] in body["marks"]
    assert body["marks"][body["favorites"][0]] == "keep"


def test_favouriting_a_staged_take_is_refused_before_the_server_sees_it(review):
    """`a-favourite-may-not-be-staged`, caught in the UI so the submission stays valid."""
    review.keyboard.press("f")  # the challenger opens marked cull
    review.wait_for_selector("#message:not([hidden])", timeout=5_000)

    assert "staged" in review.locator("#message").inner_text().lower()
    assert "favourite" not in badge_text(review, "challenger").lower()


def test_culling_a_favourited_take_drops_the_favourite(review):
    """Otherwise the submission would 422 on a contradiction the owner never typed."""
    review.keyboard.press(" ")  # keep
    review.keyboard.press("f")  # favourite
    assert "favourite" in badge_text(review, "challenger").lower()

    review.keyboard.press(" ")  # back to cull
    assert "favourite" not in badge_text(review, "challenger").lower()


# --- the help overlay and preloading ----------------------------------------------------------


def test_the_help_overlay_renders_from_the_keymap(review):
    """`KEYMAP` is the single source, so the overlay cannot document a stale key."""
    review.keyboard.press("?")
    review.wait_for_selector("#help:not([hidden])", timeout=5_000)

    rows = review.locator("#help-keys dt").count()
    keymap = review.evaluate("() => window.__photocull.KEYMAP")
    assert rows == len(keymap)

    shown = review.locator("#help-keys").inner_text()
    for row in keymap:
        assert row["label"] in shown
        assert row["description"] in shown

    review.keyboard.press("Escape")
    assert review.locator("#help").is_hidden()


def test_the_next_two_clusters_images_are_preloaded(review):
    """A review is 1,807 clusters of waiting for a JPEG unless the next ones are warm."""
    doc = session_document(review)
    ahead = [
        member["uuid"]
        for entry in doc["clusters"][1:3]
        for member in entry["members"]
    ]
    behind = [member["uuid"] for member in doc["clusters"][4]["members"]]

    requested = review.evaluate(
        "() => window.__photocull.preloaded.map((image) => image.dataset.url)"
    )
    for uuid in ahead:
        assert any(url.endswith(uuid) for url in requested), f"{uuid} was not preloaded"
    # Bounded, or a 1,807-cluster session would queue every image in the library.
    for uuid in behind:
        assert not any(url.endswith(uuid) for url in requested)


def test_an_evicted_derivative_is_reported_and_keeps_the_layout(review):
    """Photos reclaims derivatives; the pane says so rather than showing a broken image."""
    doc = session_document(review)
    index, entry = next(
        (i, c)
        for i, c in enumerate(doc["clusters"])
        if any(not m["display"]["available"] for m in c["members"])
    )
    review.evaluate("(i) => window.__photocull.goTo(i)", index)
    review.wait_for_timeout(200)

    gone = [m for m in entry["members"] if not m["display"]["available"]]
    assert gone, "the demo must carry an evicted derivative"

    # It is *said*, not merely survived. Asserting only that the layout held would pass
    # against a pane showing a broken-image icon, which tells the owner nothing about
    # why the take cannot be judged.
    reported = [
        side
        for side in ("keeper", "challenger")
        if pane(review, side).locator("[data-gone]").is_visible()
    ]
    assert reported, "the evicted take is not reported anywhere on screen"
    for side in reported:
        assert "reclaimed" in pane(review, side).locator("[data-gone]").inner_text()
        assert pane(review, side).locator("[data-photo]").is_hidden(), (
            "a broken image is still on screen beside the explanation"
        )

    # Whichever pane holds it, the frames stay the same size — the fixed frame height is
    # what stops a missing image from collapsing one side of the comparison.
    keeper = pane(review, "keeper").locator("[data-frame]").bounding_box()
    challenger = pane(review, "challenger").locator("[data-frame]").bounding_box()
    assert keeper["width"] == challenger["width"]
    assert keeper["height"] == challenger["height"]
