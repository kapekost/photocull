"""Shared test fixture factories for building synthetic records and clusters tersely."""

from __future__ import annotations

from datetime import datetime, timedelta

from photocull.models import Cluster, PhotoRecord, PhotoScore

_counter = 0

#: Base capture time for `make_cluster`. Fixed rather than "now" so a fixture's dates
#: are reproducible in a failure message.
CLUSTER_BASE = datetime(2020, 1, 1, 12, 0, 0)


def make_photo_record(**overrides) -> PhotoRecord:
    """Build a PhotoRecord with sane defaults; override only the fields a test cares about.

    Each call gets a unique uuid and a unique date (1 hour apart) unless overridden, so
    tests that need distinct/ordered records don't have to specify date every time."""
    global _counter
    _counter += 1
    defaults = dict(
        uuid=f"uuid-{_counter}",
        date=datetime(2026, 1, 1, 12, 0, 0),
    )
    defaults.update(overrides)
    return PhotoRecord(**defaults)


def make_cluster(
    uuids,
    *,
    diameter=0.30,
    median=None,
    ambiguous=False,
    sharpness_available=True,
    dropped=(),
    day=0,
    totals=None,
    sub_scores=None,
    display_paths=None,
    mod_dates=None,
):
    """Build a Cluster shaped the way `rank_cluster` actually emits one.

    Deliberately mismatched on purpose: `records` stay in bucket order while `scores`
    come out sorted by descending total, because that is exactly what `rank_cluster`
    does. Any consumer that joins the two by index is wrong, and a fixture that kept
    them aligned would hide it.

    `display_paths` maps uuid -> path (or None) for the largest local derivative, which
    is what the review session document resolves its pixel sizes from. Unset means every
    member gets a plausible path; map a uuid to None to model a photo Photos has no
    local display raster for.

    `mod_dates` maps uuid -> the photo's last-edited time. Unset means `None` for every
    member, which is the library's own common case (most photos are never edited) and
    the sentinel Task 5's staleness check turns on."""
    records = [
        make_photo_record(
            uuid=u,
            date=CLUSTER_BASE + timedelta(days=day, seconds=i),
            derivative_path=f"/deriv/{u}.jpeg",
            display_path=(
                f"/display/{u}.jpeg" if display_paths is None else display_paths.get(u)
            ),
            mod_date=(mod_dates or {}).get(u),
        )
        for i, u in enumerate(uuids)
    ]
    totals = totals or {u: 1.0 - i * 0.2 for i, u in enumerate(uuids)}
    scores = sorted(
        (
            PhotoScore(
                uuid=u,
                sub_scores=(sub_scores or {}).get(u, {"sharpness": 0.5, "exposure": 0.5}),
                total=totals[u],
            )
            for u in uuids
        ),
        key=lambda s: (-s.total, s.uuid),
    )
    winner = scores[0].uuid
    return Cluster(
        records=records,
        scores=scores,
        winner_uuid=winner,
        is_ambiguous=ambiguous,
        distances_to_winner={u: 0.1 * (i + 1) for i, u in enumerate(uuids) if u != winner},
        diameter=diameter,
        median_distance=median
        if median is not None
        else (diameter / 2 if diameter is not None else None),
        sharpness_available=sharpness_available,
        dropped_criteria=list(dropped),
    )
