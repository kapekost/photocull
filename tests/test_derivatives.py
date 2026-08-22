from types import SimpleNamespace

from photocull.derivatives import (
    derivative_path,
    derivative_paths,
    display_derivative_path,
)

# Real dimensions from the library: the dominant pair is a ~480px-long-side small
# derivative alongside a ~1024px large one, with the LARGE one first (8,851 of 8,853).
SIZES = {
    "/big.jpeg": (1024, 768),
    "/small.jpeg": (480, 360),
    "/tiny.jpeg": (342, 256),
}


def _p(**kw):
    return SimpleNamespace(**{"path_derivatives": [], "uuid": "u", **kw})


def _measure(path):
    return SIZES.get(path)


def test_selects_the_smallest_derivative_not_the_first():
    """osxphotos lists derivatives largest-first (measured: 8,851 of 8,853 items), so
    taking [0] systematically picked the LARGEST raster. Mixing rasters of different
    scale roughly triples the measured distance between photos that look identical
    (0.1188/0.0961 same-class vs 0.3092 mixed), so every photo is pinned to its
    smallest. DECISIONS.md derivative-selection-is-smallest-class."""
    p = _p(path_derivatives=["/big.jpeg", "/small.jpeg"])
    assert derivative_path(p, measure=_measure) == "/small.jpeg"


def test_selects_the_smallest_of_three():
    p = _p(path_derivatives=["/big.jpeg", "/small.jpeg", "/tiny.jpeg"])
    assert derivative_path(p, measure=_measure) == "/tiny.jpeg"


def test_selection_does_not_depend_on_list_order():
    """The only ordering guarantee this project trusts is the one it enforces itself."""
    a = _p(path_derivatives=["/big.jpeg", "/tiny.jpeg", "/small.jpeg"])
    b = _p(path_derivatives=["/small.jpeg", "/big.jpeg", "/tiny.jpeg"])
    assert derivative_path(a, measure=_measure) == derivative_path(b, measure=_measure)


def test_breaks_a_size_tie_deterministically_on_path():
    """29 real items carry two derivatives with IDENTICAL dimensions, so area alone
    does not pick a winner and list order must not be allowed to."""
    sizes = {"/b.jpeg": (480, 360), "/a.jpeg": (480, 360)}
    forward = _p(path_derivatives=["/b.jpeg", "/a.jpeg"])
    reverse = _p(path_derivatives=["/a.jpeg", "/b.jpeg"])
    assert derivative_path(forward, measure=sizes.get) == "/a.jpeg"
    assert derivative_path(reverse, measure=sizes.get) == "/a.jpeg"


def test_a_single_derivative_is_returned_without_measuring_it():
    """5,383 of 14,236 items have exactly one derivative; measuring costs ~1.8 ms each
    and cannot change the answer."""
    calls = []

    def counting(path):
        calls.append(path)
        return (480, 360)

    p = _p(path_derivatives=["/only.jpeg"])
    assert derivative_path(p, measure=counting) == "/only.jpeg"
    assert calls == []


def test_returns_none_when_no_derivative_available():
    assert derivative_path(_p(path_derivatives=[]), measure=_measure) is None


def test_returns_none_when_attribute_is_none():
    assert derivative_path(_p(path_derivatives=None), measure=_measure) is None


def test_returns_none_when_no_candidate_can_be_measured():
    """Degrade to None rather than silently falling back to [0] -- that fallback would
    quietly restore the largest-derivative behaviour this task exists to remove."""
    p = _p(path_derivatives=["/x.jpeg", "/y.jpeg"])
    assert derivative_path(p, measure=lambda _path: None) is None


def test_ignores_candidates_it_cannot_measure():
    """The unmeasurable entry is listed FIRST and a larger measurable one sits between
    it and the answer, so this can only pass if unmeasurable candidates are genuinely
    skipped AND the survivors are ranked. With `/small.jpeg` first (as this test was
    originally written) it passed against the old `derivatives[0]` code by coincidence,
    proving nothing -- the same sleeping-test shape Ticks 9 and 10 found."""
    p = _p(path_derivatives=["/unknown.jpeg", "/big.jpeg", "/small.jpeg"])
    assert derivative_path(p, measure=_measure) == "/small.jpeg"


def test_never_falls_back_to_the_original_path():
    """The whole point of this module. .path would trigger an iCloud download for
    ~98% of this library (measured: only 9/400 sampled photos have the original on
    disk), so a missing derivative must degrade to None, never to the original."""
    p = _p(path_derivatives=[], path="/originals/IMG_0001.HEIC")
    assert derivative_path(p, measure=_measure) is None


# --- the display derivative (Phase 1b Task 2) ----------------------------------
# Opposite rule, same file, deliberately adjacent. Analysis pins every photo to its
# SMALLEST raster so distances and cache keys are comparable; display takes the
# LARGEST because a human judging which take is sharper needs every pixel Photos has
# locally. Measured on the real library: the display pick has a median long side of
# 1024px against the analysis pick's 480px, and exceeds 640px for 62.0% of photos.


def test_display_selects_the_largest_derivative():
    p = _p(path_derivatives=["/small.jpeg", "/big.jpeg", "/tiny.jpeg"])
    assert display_derivative_path(p, measure=_measure) == "/big.jpeg"


def test_display_and_analysis_select_opposite_ends():
    """The two selectors must disagree on a photo that has more than one size --
    that disagreement IS the feature. 62.2% of real items have 2+ derivatives."""
    p = _p(path_derivatives=["/big.jpeg", "/small.jpeg"])
    assert derivative_path(p, measure=_measure) == "/small.jpeg"
    assert display_derivative_path(p, measure=_measure) == "/big.jpeg"


def test_display_selection_does_not_depend_on_list_order():
    a = _p(path_derivatives=["/big.jpeg", "/tiny.jpeg", "/small.jpeg"])
    b = _p(path_derivatives=["/small.jpeg", "/big.jpeg", "/tiny.jpeg"])
    assert display_derivative_path(a, measure=_measure) == display_derivative_path(
        b, measure=_measure
    )


def test_a_dimension_tie_at_the_top_resolves_to_the_SAME_file_as_analysis():
    """Measured on the real library: 30 items carry two derivatives with identical
    dimensions, and for 12 of them the tie is at the TOP, where it decides the
    display pick. A plain `max()` on `(area, dims, path)` takes the alphabetically
    LAST path while `min()` takes the first -- so those 12 photos would be displayed
    from a different file than they were analysed from, at the same pixel size, for
    no reason, and any "do these two agree?" check would score them as mixed. Both
    selectors therefore break a tie the same way: lowest path wins."""
    sizes = {"/b.jpeg": (480, 360), "/a.jpeg": (480, 360)}
    for order in (["/b.jpeg", "/a.jpeg"], ["/a.jpeg", "/b.jpeg"]):
        p = _p(path_derivatives=order)
        assert display_derivative_path(p, measure=sizes.get) == "/a.jpeg"
        assert derivative_path(p, measure=sizes.get) == "/a.jpeg"


def test_display_ignores_candidates_it_cannot_measure():
    """The unmeasurable entry is listed LAST, where an implementation that appended
    unmeasured candidates to the ranking would hand it the top slot."""
    p = _p(path_derivatives=["/small.jpeg", "/big.jpeg", "/unknown.jpeg"])
    assert display_derivative_path(p, measure=_measure) == "/big.jpeg"


def test_display_of_a_single_derivative_is_returned_without_measuring_it():
    """5,377 of 14,235 real items (37.8%) have exactly one derivative, and measuring
    it costs ~2.7 ms without being able to change the answer."""
    calls = []

    def counting(path):
        calls.append(path)
        return (1024, 768)

    p = _p(path_derivatives=["/only.jpeg"])
    assert display_derivative_path(p, measure=counting) == "/only.jpeg"
    assert calls == []


def test_display_returns_none_when_no_derivative_available():
    assert display_derivative_path(_p(path_derivatives=[]), measure=_measure) is None
    assert display_derivative_path(_p(path_derivatives=None), measure=_measure) is None


def test_display_returns_none_when_no_candidate_can_be_measured():
    p = _p(path_derivatives=["/x.jpeg", "/y.jpeg"])
    assert display_derivative_path(p, measure=lambda _path: None) is None


def test_display_never_falls_back_to_the_original_path():
    """Same rule as analysis, and it matters MORE here: this path is the one the
    review UI puts on screen, so `.path` would look like the obvious fix for a photo
    that renders small. It would mean an iCloud download per photo viewed."""
    p = _p(path_derivatives=[], path="/originals/IMG_0001.HEIC")
    assert display_derivative_path(p, measure=_measure) is None


def test_both_paths_come_from_one_measurement_pass():
    """The cost invariant, and the reason `derivative_paths` exists at all. Calling
    the two selectors independently measures every derivative twice: 23,109 header
    reads at 2.74 ms measured over the real library = ~63 s added to EVERY scan,
    warm or cold. One pass, two answers."""
    calls = []

    def counting(path):
        calls.append(path)
        return SIZES[path]

    p = _p(path_derivatives=["/big.jpeg", "/small.jpeg", "/tiny.jpeg"])
    assert derivative_paths(p, measure=counting) == ("/tiny.jpeg", "/big.jpeg")
    assert sorted(calls) == ["/big.jpeg", "/small.jpeg", "/tiny.jpeg"]


def test_the_pair_can_never_disagree_with_the_two_selectors():
    """Three ways to ask the same question; production asks the third and the tests
    above pin the first two. They must not be able to drift apart."""
    cases = [
        ["/big.jpeg", "/small.jpeg", "/tiny.jpeg"],
        ["/small.jpeg"],
        [],
        ["/unknown.jpeg", "/big.jpeg"],
        ["/unknown.jpeg"],
    ]
    for derivatives in cases:
        p = _p(path_derivatives=derivatives)
        assert derivative_paths(p, measure=_measure) == (
            derivative_path(p, measure=_measure),
            display_derivative_path(p, measure=_measure),
        ), derivatives


def test_the_pair_is_the_same_file_twice_when_only_one_size_exists():
    """Not a degenerate case: 37.8% of real items have exactly one derivative, and
    another 30 carry two of identical dimensions. The two selectors are opposite
    rules over one list, not two independent lookups."""
    p = _p(path_derivatives=["/only.jpeg"])
    analysis, display = derivative_paths(p, measure=_measure)
    assert analysis == display == "/only.jpeg"
