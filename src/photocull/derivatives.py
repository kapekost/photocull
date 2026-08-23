"""The single chokepoint for reaching image bytes.

Clustering needs pixels (Vision feature prints), which sits in tension with the
project's "never trigger an iCloud download" rule. This module is the resolution: it
reads ONLY `PhotoInfo.path_derivatives` -- locally-cached thumbnails Photos keeps
even for iCloud-optimised assets -- and never `.path` or `.export()`. In a sampled
check, nearly every photo had a local derivative while only a small fraction had the
original on disk, so falling back to `.path` would mean downloading almost the whole
library.

If you are tempted to add an original-file fallback here: don't. Degrade to None and
let the caller skip the photo.

A photo is pinned to its SMALLEST derivative, because which raster a photo is read
from moves its measured similarity more than a real difference between two takes
does: same-class near-duplicate pairs sit much closer together than mixed-class pairs
do, and the same photo through its own two derivatives can measure meaningfully far
from itself. Smallest wins over largest because a small derivative is near-universal
across a real library while a large one is not guaranteed to exist for every item.
Pinning to smallest measurably collapses false scale-mismatches within bursts and
buckets that the largest-derivative selection it replaced used to produce.

Do NOT "simplify" this back to `derivatives[0]`. That is not a neutral choice --
osxphotos lists derivatives largest-first for nearly every item, so `[0]` is the
systematically worst option.

This module holds TWO selectors with opposite rules, and both are right. Analysis
(`derivative_path`) takes the smallest for the reasons above; display
(`display_derivative_path`) takes the largest, because the review UI's whole job is
letting a human see which take is sharper and that needs every pixel Photos has
locally. Neither is the default -- a caller that does not know which it wants has a
bug -- and `derivative_paths` returns both from one measurement pass so they can
never describe different files by accident."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _measure(path: str) -> tuple[int, int] | None:
    # Imported lazily so importing this module never requires Quartz, matching how
    # photos_source defers osxphotos.
    from photocull.imaging import image_dimensions

    return image_dimensions(path)


Candidate = tuple[int, tuple[int, int], str]


def _candidates(
    derivatives: list[str], measure: Callable[[str], tuple[int, int] | None]
) -> list[Candidate]:
    """Measure every derivative once and rank them by `(pixel area, dimensions,
    path)`. Unmeasurable files are dropped, not ranked last -- an unknown size must
    never be able to win either end.

    The path tiebreak is not decoration: some real items carry two derivatives with
    identical dimensions, so area alone does not pick a winner and `path_derivatives`
    order must not be allowed to -- this project does not trust orderings it does not
    enforce."""
    ranked: list[Candidate] = []
    for path in derivatives:
        dims = measure(path)
        if dims is None:
            continue
        ranked.append((dims[0] * dims[1], (dims[0], dims[1]), path))
    ranked.sort()
    return ranked


def _smallest(ranked: list[Candidate]) -> str | None:
    # Deliberately NOT a `derivatives[0]` fallback when nothing measured: that would
    # silently restore the largest-derivative behaviour this selection exists to remove.
    return ranked[0][2] if ranked else None


def _largest(ranked: list[Candidate]) -> str | None:
    if not ranked:
        return None
    # `ranked[-1][2]` would be wrong: with the size tied at the top it takes the
    # HIGHEST path while `_smallest` takes the lowest, so the same photo at one pixel
    # size would be displayed from a different file than it was analysed from. Real
    # on this library -- 12 items have their tie at the top. Take the lowest path
    # among the winners instead, so one size class means one file.
    top = ranked[-1][:2]
    return next(c[2] for c in ranked if c[:2] == top)


def derivative_path(
    p: Any, *, measure: Callable[[str], tuple[int, int] | None] | None = None
) -> str | None:
    """Return the SMALLEST locally-cached derivative for a photo -- the one every
    pixel read on the analysis path goes through -- or None if it has none. Never
    escalates to the original file.

    Smallest, because distances and cache keys are only comparable at one raster
    scale. Its opposite number is `display_derivative_path`, which selects the
    largest for the same photo and is equally correct for its own purpose; a caller
    that does not know which of the two it wants has a bug. Use `derivative_paths`
    when you want both, so they measure the files once and cannot disagree."""
    derivatives = list(getattr(p, "path_derivatives", None) or [])
    if not derivatives:
        return None
    if len(derivatives) == 1:
        # A common case in practice. Measuring costs a few ms and cannot change the
        # answer when there is only one candidate.
        return derivatives[0]
    return _smallest(_candidates(derivatives, measure or _measure))


def display_derivative_path(
    p: Any, *, measure: Callable[[str], tuple[int, int] | None] | None = None
) -> str | None:
    """Return the LARGEST locally-cached derivative for a photo -- the one the review
    UI puts on screen -- or None if it has none. Never escalates to the original file.

    Largest, because a human deciding which of two near-identical takes is sharper
    needs every pixel Photos has locally: the display pick typically runs to a much
    larger long side than the analysis pick does. It is still bounded by what is
    cached, which is why the review UI is required to label the real pixel size
    rather than upscale past it.

    This is the exact opposite rule to `derivative_path`, deliberately, and both are
    correct for their own caller. Neither is a default. Do not "unify" them: pinning
    display to the smallest would hide the detail the review exists to check, and
    pinning analysis to the largest would corrupt every distance in the pipeline."""
    derivatives = list(getattr(p, "path_derivatives", None) or [])
    if not derivatives:
        return None
    if len(derivatives) == 1:
        return derivatives[0]
    return _largest(_candidates(derivatives, measure or _measure))


def derivative_paths(
    p: Any, *, measure: Callable[[str], tuple[int, int] | None] | None = None
) -> tuple[str | None, str | None]:
    """Return `(analysis, display)` for one photo from a SINGLE measurement pass.

    This is what the scan path calls. Asking the two selectors separately measures
    every derivative twice, and that is not a rounding error: on a large library the
    extra header reads add a real, measurable amount of time to every scan, warm or
    cold. It also makes the two answers structurally incapable of describing
    different files, which matters because they are compared to each other (a photo
    whose two picks differ is one whose display raster carries detail the analysis
    never saw)."""
    derivatives = list(getattr(p, "path_derivatives", None) or [])
    if not derivatives:
        return (None, None)
    if len(derivatives) == 1:
        return (derivatives[0], derivatives[0])
    ranked = _candidates(derivatives, measure or _measure)
    return (_smallest(ranked), _largest(ranked))
