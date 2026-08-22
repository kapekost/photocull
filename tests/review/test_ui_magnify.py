"""Task 12: magnification — compare two takes closely, nothing more.

**The cap is one source pixel per CSS pixel, and `devicePixelRatio` does not enter it.**
That is a departure from the plan, which specified one source pixel per *device* pixel,
and it was forced by a measurement rather than preferred. Fitted — the state this app has
shipped since Task 10, before any magnification exists — the 1024x768 display raster
already renders at **1.022** device pixels per source pixel at 1440x900 dpr 2, **1.587**
at 1728x1117, and **2.412** at 2560x1440. A device-pixel cap therefore does not bound the
zoom; it condemns the screen that is already on the owner's Retina Mac. Measured on the
real library at 1440x900 dpr 2: **24 of 98 sampled panes are already past it fitted**, so
those refuse outright, and across the sample it would offer a median **1.45x** against the
**2.71x** available. None of that is visible at Playwright's default device pixel ratio of
1 — this project's recurring sleeping-test shape, arriving inside a cap — so
`test_the_cap_ignores_the_device_pixel_ratio` runs at dpr 2 and is what would catch a
revert.

Magnifying a 1024px raster from 523 CSS px to 1024 CSS px on a dpr-2 screen adds no
detail and removes none — it is the same pixels, larger. Past one source pixel per CSS
pixel you are looking at interpolation in the coordinate system the layout is written in,
which is what every image viewer calls "actual size".
"""

from __future__ import annotations

import pytest

from photocull_review.session import MIXED_RESOLUTION_RATIO

pytestmark = pytest.mark.ui


# Demo cluster indices, read off `demo._SPEC`. Named rather than inlined because the
# whole point of each test is *which* raster case it exercises.
BOTH_LARGE = 0  # a pair at 1024x768, no annotations — the roomiest frame in the demo
MIXED = 4  # a pair, one 1024x768 and one 480x360
ALL_SMALL = 20  # four takes at 480x360, and no sharpness signal either
MIXED_MANY = 19  # four takes, the first at 480x360 and the rest at 1024x768
LARGE_CLUSTER = 23  # 20 takes at 1024x768
EVICTED = 16  # a pair, one member with no readable file at all


GEOMETRY = """
() => {
  const out = {};
  for (const side of ["keeper", "challenger"]) {
    const img = document.querySelector(`#pane-${side} [data-photo]`);
    const frame = document.querySelector(`#pane-${side} [data-frame]`);
    if (!img || img.hidden || !img.naturalWidth) { out[side] = null; continue; }
    const ib = img.getBoundingClientRect();
    const fb = frame.getBoundingClientRect();
    // The box the photograph is actually *drawn* in, which is not the element box:
    // fitted, `object-fit: contain` letterboxes inside a full-size element, so reading
    // the element box would report a 480x360 raster as 675px wide and turn every
    // measurement below into a claim about the frame. Computed the same way in both
    // states — magnified, the element is sized at the raster's own aspect, so `contain`
    // lands back on the element box exactly.
    const fit = Math.min(ib.width / img.naturalWidth, ib.height / img.naturalHeight);
    const drawn = [img.naturalWidth * fit, img.naturalHeight * fit];
    const left = ib.left - fb.left + (ib.width - drawn[0]) / 2;
    const top = ib.top - fb.top + (ib.height - drawn[1]) / 2;
    out[side] = {
      natural: [img.naturalWidth, img.naturalHeight],
      rendered: drawn,
      frame: [fb.width, fb.height],
      offset: [left, top],
      // The normalised point of the photograph sitting at the middle of the frame.
      center: [
        (fb.width / 2 - left) / drawn[0],
        (fb.height / 2 - top) / drawn[1],
      ],
      dpr: window.devicePixelRatio,
      label: document.querySelector(`#pane-${side} [data-zoom-label]`).textContent,
      labelHidden: document.querySelector(`#pane-${side} [data-zoom-label]`).hidden,
    };
  }
  return out;
}
"""


def geometry(page):
    return page.evaluate(GEOMETRY)


def go_to(page, index):
    page.evaluate(f"() => window.__photocull.goTo({index})")
    page.wait_for_timeout(120)


def magnified(page):
    return page.evaluate("() => window.__photocull.state.magnified")


def message(page):
    node = page.locator("#message")
    return "" if node.is_hidden() else node.text_content()


def source_pixels_per_css_pixel(view):
    """> 1 means the screen is showing detail the raster does not have."""
    return view["rendered"][0] / view["natural"][0]


# --- the cap, as arithmetic ----------------------------------------------------------


async_import = "async () => { const m = await import('/static/magnify.js'); return %s; }"


def test_max_scale_is_how_many_times_the_frame_fits_into_the_raster(review):
    """The cap as a pure function, exercised where a rendered page cannot reach.

    **Departure from the plan, recorded:** it asks for this to be "unit testable without
    a browser", which is not reachable without adding a JavaScript runtime to the test
    dependencies — and this repo's frontend deliberately has no build step and no node.
    What the plan actually wanted is preserved: the cap arithmetic is tested independently
    of layout, images and rendering, so a boundary case can be stated directly instead of
    conjured out of a viewport size."""
    cases = [
        # A 1024x768 raster fitted into a 675x392 frame is height-limited: it draws at
        # 523x392, so it may be magnified 1.96x before one source pixel is one CSS pixel.
        ({"naturalWidth": 1024, "naturalHeight": 768, "boxWidth": 675, "boxHeight": 392}, 1.959),
        # Width-limited instead: the same raster in a wide, short frame.
        ({"naturalWidth": 1024, "naturalHeight": 768, "boxWidth": 400, "boxHeight": 600}, 2.560),
        # A frame larger than the raster: `object-fit: contain` has already upscaled it,
        # so the cap is *below* 1 and there is nothing to magnify into.
        ({"naturalWidth": 480, "naturalHeight": 360, "boxWidth": 819, "boxHeight": 526}, 0.684),
        # Portrait against a landscape frame.
        ({"naturalWidth": 768, "naturalHeight": 1024, "boxWidth": 675, "boxHeight": 392}, 2.612),
    ]
    for arguments, expected in cases:
        got = review.evaluate(async_import % f"m.maxScale({arguments!r})".replace("'", '"'))
        assert got == pytest.approx(expected, abs=0.001), arguments


def test_max_scale_has_no_answer_when_a_size_is_unknown(review):
    """An image that has not decoded yet reports `naturalWidth === 0`.

    Answering `Infinity` there would magnify by whatever the first render happened to
    compute, which is the silent-and-plausible failure this project keeps meeting."""
    for arguments in (
        {"naturalWidth": 0, "naturalHeight": 0, "boxWidth": 675, "boxHeight": 392},
        {"naturalWidth": 1024, "naturalHeight": 768, "boxWidth": 0, "boxHeight": 0},
    ):
        got = review.evaluate(async_import % f"m.maxScale({arguments!r})".replace("'", '"'))
        assert got is None, arguments


def test_the_cluster_cap_is_the_lower_of_the_two_takes(review):
    """Both panes share one scale, so the shared scale must respect the smaller raster.

    Equal size is `compare-side-by-side-with-sync-zoom`'s load-bearing rule — a large
    hero beside a small alternate biases the eye regardless of which photograph is
    better — so the higher-resolution take gives up the detail it could have shown rather
    than the two panes drifting apart."""
    views = [
        {"naturalWidth": 1024, "naturalHeight": 768, "boxWidth": 675, "boxHeight": 392},
        {"naturalWidth": 480, "naturalHeight": 360, "boxWidth": 675, "boxHeight": 392},
    ]
    got = review.evaluate(async_import % f"m.clusterMaxScale({views!r})".replace("'", '"'))
    # 480/360 into 675x392 draws at 523x392 too, so its own cap is 0.918 — below 1.
    assert got == pytest.approx(0.918, abs=0.001)


def test_a_magnification_too_small_to_see_is_not_offered(review):
    """`MIN_USEFUL_SCALE`. A 2% zoom does not visibly change the photograph, and a key
    that appears to do nothing is indistinguishable from one that is broken — which is
    the "silently capped control" this task was told not to build.

    Stated either side of the floor rather than on it: `1 / (1000 / 1050)` is
    1.0500000000000003, so a fixture claiming to sit *on* 1.05 would not, which is the
    same float trap that left `MIN_VISIBLE_DELTA`'s boundary test asleep at Task 11."""
    def can(natural, box):
        views = [
            {
                "naturalWidth": natural,
                "naturalHeight": natural,
                "boxWidth": box,
                "boxHeight": box,
            }
        ]
        return review.evaluate(async_import % f"m.canMagnify({views!r})".replace("'", '"'))

    assert can(1020, 1000) is False  # 1.02x — below the floor
    assert can(1200, 1000) is True  # 1.20x — worth the keystroke


def test_the_crop_is_clamped_to_the_photograph(review):
    """Pure-function half of "a drag that runs off the edge banks no distance".

    Also the only place the `drawn <= box` branch is reachable: at maximum magnification
    the drawing is exactly the raster's own size, and every demo raster is larger than
    its frame in both axes, so a rendered test cannot get here."""
    view = {"naturalWidth": 1024, "naturalHeight": 768, "boxWidth": 675, "boxHeight": 392}

    def place(centerX, centerY, scale=1.959):
        arguments = view | {"scale": scale, "centerX": centerX, "centerY": centerY}
        return review.evaluate(async_import % f"m.placement({arguments!r})".replace("'", '"'))

    # Dragged well past the right edge: the photograph's right edge lands on the frame's,
    # and the centre that comes back is the clamped one rather than the 2.5 asked for.
    far = place(2.5, 0.5)
    assert far["left"] == pytest.approx(far["width"] * -1 + 675, abs=0.5)
    assert far["centerX"] < 1.0
    assert place(-3.0, 0.5)["left"] == pytest.approx(0.0, abs=0.5)

    # Fitted (scale 1) the drawing is smaller than the frame in the wide axis, so it is
    # centred and there is nothing to pan.
    fitted = place(0.9, 0.5, scale=1.0)
    assert fitted["left"] == pytest.approx((675 - fitted["width"]) / 2, abs=0.5)
    assert fitted["centerX"] == 0.5


def test_the_mixed_resolution_ratio_is_the_one_the_session_document_uses(review):
    """One number, two languages. A drift here would warn about a different population
    than the `mixed-resolution` annotation flags, on the same screen."""
    got = review.evaluate(async_import % "m.MIXED_RESOLUTION_RATIO")
    assert got == MIXED_RESOLUTION_RATIO


# --- what `Z` does -------------------------------------------------------------------


def test_z_magnifies_both_panes_to_the_same_scale_and_the_same_crop(review):
    review.set_viewport_size({"width": 1440, "height": 900})
    go_to(review, BOTH_LARGE)
    before = geometry(review)
    assert not magnified(review)

    review.keyboard.press("z")
    review.wait_for_timeout(120)

    assert magnified(review)
    after = geometry(review)
    for side in ("keeper", "challenger"):
        assert after[side]["rendered"][0] > before[side]["rendered"][0], side
    # The same scale, because both rasters are the same size here, and the same crop.
    assert after["keeper"]["rendered"] == pytest.approx(after["challenger"]["rendered"], abs=0.5)
    assert after["keeper"]["center"] == pytest.approx(after["challenger"]["center"], abs=0.005)


def test_magnification_never_exceeds_one_source_pixel_per_css_pixel(review):
    """The cap, measured on the rendered page rather than on the arithmetic.

    Only the panes that actually magnified are asserted on, and the counter at the end is
    what keeps that from quietly emptying the test: the *fitted* view legitimately draws a
    480x360 copy at 675px when the frame is that wide — `object-fit: contain` upscales,
    and it has done since Task 10. Task 12 promises that magnification never makes that
    worse, not that the fitted view was already innocent."""
    magnifications = 0
    for width, height in ((1440, 900), (1280, 800), (1100, 1000)):
        review.set_viewport_size({"width": width, "height": height})
        for index in (BOTH_LARGE, MIXED, ALL_SMALL, LARGE_CLUSTER):
            go_to(review, index)
            review.keyboard.press("z")
            review.wait_for_timeout(150)
            if not magnified(review):
                continue
            for side, view in geometry(review).items():
                if view is None:
                    continue
                magnifications += 1
                ratio = source_pixels_per_css_pixel(view)
                assert ratio <= 1.0 + 1e-6, (
                    f"{width}x{height} cluster {index} {side}: rendering "
                    f"{view['rendered']} from a {view['natural']} raster is a "
                    f"{ratio:.3f}x upscale"
                )
            review.keyboard.press("Escape")
            review.wait_for_timeout(60)
    assert magnifications >= 12, f"only {magnifications} panes magnified at all"


def test_the_cap_ignores_the_device_pixel_ratio(browser, review_server):
    """The test that would have caught the plan's cap, and the reason this file exists.

    At dpr 2 the *fitted* view already renders 1.022 device pixels per source pixel at
    this viewport, so a device-pixel cap reports "nothing to magnify" on a cluster that
    has 1.96x of genuine magnification available. Every other test here runs at dpr 1,
    where that mistake is invisible."""
    context = browser.new_context(
        viewport={"width": 1440, "height": 900}, device_scale_factor=2
    )
    page = context.new_page()
    try:
        page.goto(review_server.url)
        page.wait_for_selector("#app:not([hidden])", timeout=15_000)
        go_to(page, BOTH_LARGE)
        before = geometry(page)
        assert before["keeper"]["dpr"] == 2

        page.keyboard.press("z")
        page.wait_for_timeout(120)

        assert magnified(page), "magnification refused itself on a Retina display"
        after = geometry(page)
        assert after["keeper"]["rendered"][0] == pytest.approx(1024, abs=1)
        assert after["keeper"]["rendered"][0] > before["keeper"]["rendered"][0] * 1.5
    finally:
        context.close()


def test_the_label_states_the_real_pixel_size(review):
    review.set_viewport_size({"width": 1440, "height": 900})
    go_to(review, BOTH_LARGE)
    assert geometry(review)["keeper"]["labelHidden"], "the label shows before `Z` is pressed"

    review.keyboard.press("z")
    review.wait_for_timeout(120)

    for side, view in geometry(review).items():
        assert not view["labelHidden"], side
        # The size of the copy on disk, not the size on screen: the whole question the
        # owner is asking at this magnification is how much is really there.
        assert "1024 × 768 px" in view["label"], view["label"]
        assert "1.96×" in view["label"], view["label"]


def test_a_take_with_no_larger_local_copy_says_so_and_stays_fitted(review):
    """"Further zoom disabled", not a control that silently does nothing.

    At this viewport the 480x360 rasters are already drawn larger than they are, so the
    honest answer is that there is nothing to magnify into — which is the ordinary case
    for the **24.1% of real clusters with no take above 640px**, not an edge."""
    review.set_viewport_size({"width": 1440, "height": 1100})
    go_to(review, ALL_SMALL)
    before = geometry(review)

    review.keyboard.press("z")
    review.wait_for_timeout(120)

    assert not magnified(review)
    assert geometry(review)["keeper"]["rendered"] == pytest.approx(
        before["keeper"]["rendered"], abs=0.5
    )
    assert "as much detail as there is" in message(review)
    # And the same window magnifies a 1024px cluster perfectly well, so the refusal is
    # about this cluster's rasters rather than about the window being large.
    go_to(review, BOTH_LARGE)
    review.keyboard.press("z")
    review.wait_for_timeout(120)
    assert magnified(review)


def test_a_mixed_resolution_pair_is_capped_by_its_smaller_copy(review):
    """At 1440x900 the 480px copy in this pair is already at its own limit (cap 0.991),
    and the shared scale is the lower of the two — so the pair refuses outright rather
    than magnifying the 1024px take alone.

    That is the design decision, stated where a `Math.max` would fail it on the rendered
    page and not only in the arithmetic: letting the larger take set the scale would draw
    the smaller one past its raster, making it look softer than it is, on the one screen
    whose job is deciding which take is sharper."""
    review.set_viewport_size({"width": 1440, "height": 900})
    go_to(review, MIXED)
    review.keyboard.press("z")
    review.wait_for_timeout(150)
    assert not magnified(review)
    assert "as much detail as there is" in message(review)


def test_a_mixed_resolution_pair_warns_that_sharpness_is_not_comparable(review):
    # 1280x800, where the smaller copy still has headroom (cap 1.37) so the pair does
    # magnify and the warning has somewhere to appear.
    review.set_viewport_size({"width": 1280, "height": 800})
    go_to(review, MIXED)
    review.keyboard.press("z")
    review.wait_for_timeout(150)

    assert magnified(review)
    said = message(review)
    assert "sharp" in said.lower(), said
    assert "480" in said and "1024" in said, said

    views = geometry(review)
    # One scale for both panes, and it is the smaller raster's cap — so the 1024px take
    # is deliberately not shown at its own full detail.
    assert views["keeper"]["rendered"][1] == pytest.approx(
        views["challenger"]["rendered"][1], abs=0.5
    )
    small = min(views.values(), key=lambda view: view["natural"][0])
    assert source_pixels_per_css_pixel(small) == pytest.approx(1.0, abs=0.01)


def test_panning_one_pane_pans_the_other(review):
    review.set_viewport_size({"width": 1440, "height": 900})
    go_to(review, BOTH_LARGE)
    review.keyboard.press("z")
    review.wait_for_timeout(120)
    before = geometry(review)

    frame = review.locator("#pane-keeper [data-frame]").bounding_box()
    review.mouse.move(frame["x"] + frame["width"] / 2, frame["y"] + frame["height"] / 2)
    review.mouse.down()
    # Five steps, because a real drag is a stream of moves and a handler that measures
    # each one from the press rather than from the previous move pans five times too far
    # while a single-step test sees nothing wrong.
    review.mouse.move(
        frame["x"] + frame["width"] / 2 - 120, frame["y"] + frame["height"] / 2, steps=5
    )
    review.mouse.up()
    review.wait_for_timeout(120)

    after = geometry(review)
    # Dragging the photograph 120px to the left moves the crop 120px to the right, in the
    # photograph's own coordinates. Asserted as a quantity rather than a direction.
    moved = after["keeper"]["center"][0] - before["keeper"]["center"][0]
    assert moved == pytest.approx(120 / after["keeper"]["rendered"][0], rel=0.15)
    # The other pane moved with it, to the same normalised point of its own photograph.
    assert after["challenger"]["center"] == pytest.approx(after["keeper"]["center"], abs=0.005)


def test_the_crop_survives_stepping_to_the_next_take(review):
    """The reason this feature is worth building at all: hold a crop, step the takes.

    Comparing two faces at 2x means looking at the *same* corner of each photograph in
    turn; a crop that reset on every step would make that impossible on any cluster
    bigger than a pair."""
    review.set_viewport_size({"width": 1440, "height": 900})
    go_to(review, LARGE_CLUSTER)
    review.keyboard.press("z")
    review.wait_for_timeout(120)
    frame = review.locator("#pane-keeper [data-frame]").bounding_box()
    review.mouse.move(frame["x"] + frame["width"] / 2, frame["y"] + frame["height"] / 2)
    review.mouse.down()
    review.mouse.move(frame["x"] + frame["width"] / 2 - 100, frame["y"] + frame["height"] / 2 - 40)
    review.mouse.up()
    review.wait_for_timeout(120)
    before = geometry(review)

    review.keyboard.press("ArrowDown")
    review.wait_for_timeout(200)

    assert magnified(review), "stepping a take dropped out of magnification"
    after = geometry(review)
    assert after["challenger"]["center"] == pytest.approx(before["keeper"]["center"], abs=0.005)


def test_resizing_the_window_redraws_within_the_new_cap(review):
    """The frames are sized from what the window leaves over, so every resize moves the
    cap. A magnification computed once and left alone would sit at the old frame's crop,
    showing dead space beside the photograph."""
    review.set_viewport_size({"width": 1100, "height": 800})
    go_to(review, BOTH_LARGE)
    review.keyboard.press("z")
    review.wait_for_timeout(150)
    assert magnified(review)

    # Panned hard to one corner first, which is what makes the resize visible: at maximum
    # magnification the *drawing* is always exactly the raster's own size, so widening the
    # window cannot change it — only where it sits. A crop left against the old frame's
    # right edge leaves dead space beside the photograph in the new one.
    frame = review.locator("#pane-keeper [data-frame]").bounding_box()
    review.mouse.move(frame["x"] + frame["width"] / 2, frame["y"] + frame["height"] / 2)
    review.mouse.down()
    review.mouse.move(frame["x"] - 2000, frame["y"] - 2000, steps=3)
    review.mouse.up()
    review.wait_for_timeout(120)

    review.set_viewport_size({"width": 1440, "height": 1000})
    review.wait_for_timeout(250)

    for side, view in geometry(review).items():
        assert source_pixels_per_css_pixel(view) <= 1.0 + 1e-6, side
        # No gap between the photograph and the frame it is being cropped by.
        gap_right = view["frame"][0] - (view["offset"][0] + view["rendered"][0])
        gap_bottom = view["frame"][1] - (view["offset"][1] + view["rendered"][1])
        assert gap_right <= 0.5, f"{side}: {gap_right:.1f}px of dead space on the right"
        assert gap_bottom <= 0.5, f"{side}: {gap_bottom:.1f}px of dead space below"


def test_magnification_does_not_follow_you_to_the_next_cluster(review):
    """A crop means something across the takes of one cluster — same framing, seconds
    apart. Across clusters it means nothing, and arriving already zoomed into a corner
    would hide the photograph the owner is being asked to judge."""
    review.set_viewport_size({"width": 1440, "height": 900})
    go_to(review, BOTH_LARGE)
    review.keyboard.press("z")
    review.wait_for_timeout(150)
    assert magnified(review)

    review.keyboard.press("Enter")
    review.wait_for_timeout(300)

    assert not magnified(review)
    assert geometry(review)["keeper"]["labelHidden"]


def test_stepping_onto_a_take_of_another_size_replaces_it_at_its_own_scale(review):
    """The stale-raster trap, and the demo could not express it until this task.

    An `<img>` keeps reporting the *previous* raster's `naturalWidth` until the new one
    decodes — Chromium holds the old frame deliberately, to avoid a flash — so a
    magnification computed when the `src` changes is arithmetic on the take that just
    left. Every mixed-resolution cluster in the demo was a **pair**, where the challenger
    never changes and the bug cannot occur; the real library mixes the two display
    classes in 6.6% of clusters, at sizes up to 30."""
    review.set_viewport_size({"width": 1280, "height": 800})
    go_to(review, MIXED_MANY)
    review.keyboard.press("z")
    review.wait_for_timeout(150)
    assert magnified(review)

    # Asserted as **exactly** one source pixel per CSS pixel rather than "at most": the
    # shared scale is the lower of the two caps, and the challenger is always either the
    # binding take or tied with it, so it is drawn at precisely its own size. A stale
    # placement fails this in whichever direction it was stepped — over the cap coming
    # down from 1024 to 480, and silently *under*-magnified going the other way, which
    # `<= 1.0` would have waved straight through.
    seen = set()
    for _ in range(4):  # four, so the wrap from a 1024px take back onto the 480px one runs
        view = geometry(review)["challenger"]
        seen.add(tuple(view["natural"]))
        assert source_pixels_per_css_pixel(view) == pytest.approx(1.0, abs=0.01), (
            f"a {view['natural']} take was drawn at {view['rendered']} — "
            "placed with the dimensions of the take that just left"
        )
        review.keyboard.press("ArrowDown")
        review.wait_for_timeout(250)

    assert len(seen) == 2, f"the challengers never changed raster size: {seen}"


def test_a_drag_past_the_edge_banks_no_distance(review):
    """Pulled hard against the edge and then back, the photograph moves at once.

    A handler storing the centre it was *asked* for rather than the clamped one makes
    the first part of every drag-back do nothing, which reads as the pane being stuck."""
    review.set_viewport_size({"width": 1440, "height": 900})
    go_to(review, BOTH_LARGE)
    review.keyboard.press("z")
    review.wait_for_timeout(150)

    frame = review.locator("#pane-keeper [data-frame]").bounding_box()
    middle = (frame["x"] + frame["width"] / 2, frame["y"] + frame["height"] / 2)
    review.mouse.move(*middle)
    review.mouse.down()
    review.mouse.move(middle[0] - 3000, middle[1], steps=3)  # hard against the right edge
    review.wait_for_timeout(60)
    at_edge = geometry(review)["keeper"]["center"][0]
    review.mouse.move(middle[0] - 3000 + 80, middle[1], steps=3)  # and 80px back
    review.mouse.up()
    review.wait_for_timeout(120)

    after = geometry(review)["keeper"]
    assert at_edge == pytest.approx(1.0 - after["frame"][0] / 2 / after["rendered"][0], abs=0.01)
    moved_back = at_edge - after["center"][0]
    assert moved_back == pytest.approx(80 / after["rendered"][0], rel=0.2)


def test_clearing_a_message_re_places_the_magnified_photograph(review):
    """The message strip is part of the column, so showing or hiding it resizes every
    frame — and the crop is a position inside a frame. This is the only path that changes
    the frames' height without a resize event to notice it."""
    review.set_viewport_size({"width": 1280, "height": 800})
    go_to(review, MIXED)
    review.keyboard.press("z")
    review.wait_for_timeout(150)
    assert magnified(review) and message(review), "this cluster should warn on magnifying"

    review.keyboard.press(" ")  # any decision clears the message, and so grows the frames
    review.wait_for_timeout(150)
    assert not message(review)

    for side, view in geometry(review).items():
        # Never panned, so the crop is still the middle of the photograph. If the frames
        # grew underneath a placement nobody recomputed, it is no longer the middle.
        assert view["center"] == pytest.approx([0.5, 0.5], abs=0.01), side


def test_a_native_image_drag_cannot_start_on_a_magnified_pane(review):
    """Chromium drags an `<img>` by default, which swallows every pointer move after it
    and makes panning look simply broken.

    Asserted by dispatching `dragstart` rather than by dragging, and that is the point:
    Playwright's synthetic pointer stream never triggers the native drag at all, so a
    pan test cannot see this guard. Measured — with both the CSS and the JS guard
    removed, `test_panning_one_pane_pans_the_other` still passes."""
    go_to(review, BOTH_LARGE)
    review.keyboard.press("z")
    review.wait_for_timeout(120)
    delivered = review.evaluate(
        """() => {
            const img = document.querySelector('#pane-keeper [data-photo]');
            const event = new Event('dragstart', { bubbles: true, cancelable: true });
            return img.dispatchEvent(event);
        }"""
    )
    assert delivered is False, "a native image drag was allowed to start"


def test_esc_returns_to_the_fitted_view(review):
    review.set_viewport_size({"width": 1440, "height": 900})
    go_to(review, BOTH_LARGE)
    fitted = geometry(review)
    review.keyboard.press("z")
    review.wait_for_timeout(120)
    assert magnified(review)

    review.keyboard.press("Escape")
    review.wait_for_timeout(120)

    assert not magnified(review)
    after = geometry(review)
    assert after["keeper"]["rendered"] == pytest.approx(fitted["keeper"]["rendered"], abs=0.5)
    assert after["keeper"]["labelHidden"]


def test_magnifying_again_starts_from_the_middle(review):
    """`Esc` means "show me the whole photograph", so `Z` after it starts over rather
    than dropping the owner back into a corner they had already left.

    This is the contract `toggleMagnify` relies on when it does *not* reset the crop
    itself — every path out of magnification recentres, so entry is always centred."""
    review.set_viewport_size({"width": 1440, "height": 900})
    go_to(review, BOTH_LARGE)
    review.keyboard.press("z")
    review.wait_for_timeout(150)
    frame = review.locator("#pane-keeper [data-frame]").bounding_box()
    review.mouse.move(frame["x"] + frame["width"] / 2, frame["y"] + frame["height"] / 2)
    review.mouse.down()
    review.mouse.move(frame["x"] - 300, frame["y"] - 200, steps=3)
    review.mouse.up()
    review.wait_for_timeout(120)
    assert geometry(review)["keeper"]["center"][0] > 0.6

    review.keyboard.press("Escape")
    review.wait_for_timeout(80)
    review.keyboard.press("z")
    review.wait_for_timeout(150)

    assert geometry(review)["keeper"]["center"] == pytest.approx([0.5, 0.5], abs=0.01)


def test_esc_closes_the_help_overlay_before_it_leaves_magnification(review):
    """Two things answer to one key, so the order is pinned rather than left to reading.

    The overlay is on top; an `Esc` that dropped the magnification underneath it while
    the list stayed open would be the wrong one every time."""
    go_to(review, BOTH_LARGE)
    review.keyboard.press("z")
    review.keyboard.press("?")
    review.wait_for_timeout(120)
    assert review.locator("#help").is_visible()

    review.keyboard.press("Escape")
    review.wait_for_timeout(120)
    assert review.locator("#help").is_hidden()
    assert magnified(review), "Esc left magnification while the help overlay was open"

    review.keyboard.press("Escape")
    review.wait_for_timeout(120)
    assert not magnified(review)


def test_z_toggles_back_to_the_fitted_view(review):
    go_to(review, BOTH_LARGE)
    review.keyboard.press("z")
    review.wait_for_timeout(120)
    assert magnified(review)
    review.keyboard.press("z")
    review.wait_for_timeout(120)
    assert not magnified(review)


def test_an_evicted_take_does_not_block_magnifying_the_one_that_is_there(review):
    """One take having no local copy makes the *comparison* impossible, not the looking."""
    review.set_viewport_size({"width": 1440, "height": 900})
    go_to(review, EVICTED)
    views = geometry(review)
    present = [side for side, view in views.items() if view is not None]
    assert len(present) == 1, "the demo's evicted pair no longer has exactly one image"

    review.keyboard.press("z")
    review.wait_for_timeout(120)

    assert magnified(review)
    view = geometry(review)[present[0]]
    assert source_pixels_per_css_pixel(view) == pytest.approx(1.0, abs=0.01)


# --- the invariants this must not break ----------------------------------------------


def test_the_frames_stay_equal_while_magnified(review):
    """`the-two-panes-hold-identical-rows`, which magnification could break two ways:
    by adding a label that takes vertical space, or by letting an oversized image push
    its own pane wider."""
    review.set_viewport_size({"width": 1440, "height": 900})
    for index in (BOTH_LARGE, MIXED, LARGE_CLUSTER):
        go_to(review, index)
        review.keyboard.press("z")
        review.wait_for_timeout(120)
        views = geometry(review)
        boxes = [view["frame"] for view in views.values() if view is not None]
        if len(boxes) == 2:
            assert boxes[0] == pytest.approx(boxes[1], abs=0.5), index
        overflow = review.evaluate(
            "() => document.getElementById('app').scrollWidth"
            " - document.getElementById('app').clientWidth"
        )
        assert overflow <= 0, f"cluster {index} scrolls sideways while magnified"
        review.keyboard.press("Escape")
        review.wait_for_timeout(60)


def test_deciding_still_works_while_magnified(review):
    """Magnification is a way of looking, not a mode that captures the keyboard."""
    go_to(review, BOTH_LARGE)
    review.keyboard.press("z")
    review.wait_for_timeout(120)
    review.keyboard.press(" ")
    review.wait_for_timeout(120)
    assert "keeping 2" in review.locator("#position").text_content()
    review.keyboard.press("Enter")
    review.wait_for_timeout(250)
    assert "1 decided" in review.locator("#progress").text_content()


def test_the_footer_and_the_help_overlay_both_offer_z(review):
    """A binding nobody can find is a binding nobody uses — the lesson the arrow remap
    was fixing, and the one a new key is most likely to repeat."""
    hint = review.locator("#hint").text_content()
    assert "Z" in hint and "magnify" in hint, hint
    review.keyboard.press("?")
    review.wait_for_timeout(80)
    assert "magnif" in review.locator("#help-keys").text_content().lower()

    for width in (1280, 1000):
        review.set_viewport_size({"width": width, "height": 900})
        review.wait_for_timeout(60)
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
