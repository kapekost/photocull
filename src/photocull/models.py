"""Data models for the Phase 0 audit. Pure dataclasses — no osxphotos import here,
so this module (and anything built on it) is testable without Photos library access."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class PhotoRecord:
    """A library item reduced to exactly the fields the Phase 0 audit needs.

    This is the seam between osxphotos (untestable here — needs a real library) and
    the aggregation/clustering logic (fully testable with synthetic instances)."""

    uuid: str
    date: datetime | None
    is_video: bool = False
    filesize: int = 0
    duration_seconds: float | None = None
    is_screenshot: bool = False
    is_screen_recording: bool = False
    is_selfie: bool = False
    is_burst: bool = False
    is_live_photo: bool = False
    is_slow_mo: bool = False
    is_time_lapse: bool = False
    is_panorama: bool = False
    is_raw: bool = False
    is_favorite: bool = False
    is_hidden: bool = False
    #: True when the asset lives in a shared album. Measured on the real library:
    #: 6.2% of assets but **27.1% of the photos this app stages for culling**, a
    #: 4.4x over-representation, because shared albums are exactly where duplicate
    #: takes accumulate. Photos' AppleScript surface cannot act on them either, so
    #: they are excluded from clustering unless `ClusterConfig.include_shared`.
    is_shared: bool = False
    album_count: int = 0

    # Phase 1 additions. `burst_key` identifies the burst *group* an item belongs
    # to (all members of one burst share it) -- note this is NOT osxphotos'
    # `PhotoInfo.burst_key`, which is a bool meaning "is this the group's key
    # image". See DECISIONS.md `burst-group-key-is-synthetic`.
    latitude: float | None = None
    longitude: float | None = None
    device: str | None = None
    burst_key: str | None = None
    mod_date: datetime | None = None
    width: int = 0
    height: int = 0

    # The locally-cached derivative every pixel read for this photo goes through, as
    # chosen by `derivatives.derivative_path`. Stamped at scan time by
    # `iter_photo_records(db, with_derivatives=True)` rather than looked up later,
    # and that is not a style choice: 72 burst siblings in the real library are
    # reachable only from `PhotoInfo.burst_photos` during that iteration and are
    # absent from `PhotosDB.photos()`, so a post-hoc uuid lookup silently drops
    # exactly the near-duplicates this app exists to collapse (measured; see
    # DECISIONS.md `burst-siblings-expanded-in-scan`). It doubles as the analysis
    # cache's `analysis_key`, so re-reading a photo from a different raster can never
    # serve stale numbers (`derivative-selection-is-smallest-class`).
    derivative_path: str | None = None

    #: Photos' own content signature, from `PhotoInfo.fingerprint`. Two items sharing
    #: one are byte-identical originals, which is a far stronger statement than the
    #: feature-print similarity Phase 1 clusters on. Measured on the real library:
    #: 10 groups, 11 redundant copies, and 896 items carry no fingerprint at all --
    #: so `None` must never group with `None`.
    fingerprint: str | None = None

    # The LARGEST locally-cached derivative, chosen by
    # `derivatives.display_derivative_path` in the same pass and stamped by the same
    # opt-in flag. This one is only ever put on screen -- it must never reach the
    # analysis path or the cache key, because the pipeline's distances are only
    # comparable at one raster scale (`derivative-selection-is-smallest-class`) and
    # this raster is a different one for 62.2% of the library (median 1024px long
    # side against the analysis pick's 480px). It exists because the review UI's job
    # is letting a human see which of two near-identical takes is sharper, and 480px
    # cannot settle that; `zoom-is-bounded-by-the-derivative` then requires the UI to
    # label the real pixel size rather than upscale past it.
    display_path: str | None = None


@dataclass
class CountSize:
    """Running count + total size (bytes) for one bucket of a breakdown."""

    count: int = 0
    size_bytes: int = 0

    def add(self, size_bytes: int) -> None:
        self.count += 1
        self.size_bytes += size_bytes


@dataclass
class AuditSummary:
    """The full Phase 0 audit result — one of these is rendered as a table and as JSON."""

    total_items: int
    total_photos: int
    total_videos: int
    total_size_bytes: int
    by_year: dict[int, CountSize]
    by_type: dict[str, CountSize]
    videos_by_duration: dict[str, CountSize]
    favorites_count: int
    no_album_count: int
    hidden_count: int
    estimated_near_dup_clusters: int
    gap_seconds: float
    library_path: str | None
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class PhotoScore:
    """One photo's best-shot verdict, kept as per-criterion sub-scores and not just a
    total, so the review UI can show *why* a take won (spec requirement).

    A sub-score of None means "this criterion does not apply here" -- no faces, no
    detectable horizon -- and is dropped from the weighted total rather than averaged
    in as a zero. Sub-scores not present in `ClusterConfig.weights` (currently
    `capture_quality`) ride along as diagnostics and never move the total."""

    uuid: str
    sub_scores: dict[str, float | None] = field(default_factory=dict)
    total: float = 0.0


@dataclass
class Cluster:
    """A group of near-duplicate takes with its scores, ranked.

    `winner_uuid` is the scorer's proposal, not a verdict: per DECISIONS.md
    `cull-selection-is-propose-and-adjust` the user may keep more than one photo, and
    when `is_ambiguous` is set the UI must present the takes for a manual pick rather
    than defaulting to the winner (`ambiguous-clusters-go-to-manual-pick`).

    The fields below the divider are what a cluster knows about its own
    trustworthiness. They exist because two decisions need them and neither can
    reconstruct them later: the calibration export the owner reviews at Task 9
    (`cull-safety-is-radius-to-keeper`), and the graded review annotation
    (`low-confidence-clusters-are-annotated-in-review`, which explicitly noted the
    scorer did not yet carry its caveats forward)."""

    records: list["PhotoRecord"] = field(default_factory=list)
    scores: list[PhotoScore] = field(default_factory=list)
    winner_uuid: str | None = None
    is_ambiguous: bool = False

    # --- self-assessment -------------------------------------------------------
    #: Distance from every other member to the photo actually being kept. This, not
    #: `diameter`, is the quantity a cull decision rests on: the irreversible claim is
    #: "X is redundant with keeper K", which is about d(X, K) alone. It is also the
    #: only number in the system a human can act on -- "this photo is 0.58 from the
    #: one you are keeping" is reviewable where "diameter 0.614" is not.
    distances_to_winner: dict[str, float] = field(default_factory=dict)
    #: Widest and typical internal pairwise distance. The cut-height curve the owner
    #: picks complete-linkage's parameter from at Task 9 is computed from these.
    diameter: float | None = None
    median_distance: float | None = None
    #: False when `cluster_sharpness` dropped the criterion for the whole cluster --
    #: mismatched rasters, an unreadable frame, or featureless takes. Measured at
    #: 7.7% of real multi-item buckets, dominated by panoramas. Such a cluster's
    #: winner was chosen on weaker evidence than elsewhere, and the UI says so
    #: ("no sharpness signal -- compare at 100%").
    sharpness_available: bool = True
    #: Weighted criteria that could not be measured for every take, so
    #: `on_common_criteria` excluded them from the comparison. Zero-weighted signals
    #: are never listed: `facing` is None on every photo by design, and reporting it
    #: on every cluster would make the annotation worthless.
    dropped_criteria: list[str] = field(default_factory=list)
