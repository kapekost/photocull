# Photo Cull — how it works

A local-only macOS app for culling a large Apple Photos library down to keepers, then
building print-ready 4x6 / 5x7 albums. Nothing leaves the machine. No cloud services,
no third-party photo APIs.

This document walks through the pipeline stage by stage — what each `photocull`
subcommand actually does and why it's built the way it is. For install and quick-start
instructions, see the top-level `README.md`.

## Design constraints

- Runs entirely on your Mac against your local Apple Photos library.
- **The app never deletes anything.** It writes tags and albums back into Photos; you
  do the actual deletion in Photos.app, where Recently Deleted gives a 30-day safety
  net.
- Your library may have "Optimize Mac Storage" enabled, so originals may not be present
  on disk. Analysis prefers Photos' own derivatives/thumbnails; it never triggers an
  iCloud download during a scan.
- Library size assumption: 20,000-50,000+ items. Everything is incremental and
  resumable.

## Stack

- Python 3.12+, plain venv.
- `osxphotos` for read-only library access (Python API, not just CLI).
- `photoscript` for write-back to Photos.app (create albums, add photos, set favorite,
  set keywords). Photos' AppleScript dictionary cannot delete photos — that's
  intentional and fine.
- `pyobjc-framework-Vision` + `pyobjc-framework-Quartz` for on-device image analysis.
- FastAPI + uvicorn backend, bound to `127.0.0.1` only.
- Front end: plain HTML/CSS/JS served by the backend. No build step, no npm.
- SQLite for the analysis cache.

## Audit (`photocull audit`)

A single command that prints a summary of the library, so you know what you're dealing
with before anything else runs:

- Total items, split photos vs videos, total size.
- Counts and size by year.
- Counts by type: screenshots, screen recordings, selfies, bursts, Live Photos,
  slo-mo, time-lapse, panoramas, RAW.
- Videos bucketed by duration (<5s, 5-30s, 30s-2m, >2m) with total size each.
- Favorites count, items in no album, hidden count.
- Estimated near-duplicate cluster count from a fast time-proximity pass only
  (no image analysis yet) — capture-time gaps under 90s.

Output is a compact table plus, with `--json-out`, a JSON file. This is read-only and
the safest way to try the app on your own library — it drives every threshold decision
downstream.

## Cluster & score (`photocull cluster`, `photocull calibrate`)

### Clustering

Two stages:

1. **Bucketing (cheap).** Group by capture-time proximity — default 90s gap, tunable.
   Same-device and near-identical GPS reinforce a bucket. Burst groups (`burst_key`)
   and Live Photo pairs are pre-joined into buckets automatically.
2. **Similarity (expensive, only within buckets).** Compute an Apple Vision feature
   print per image (`VNGenerateImageFeaturePrintRequest`), compare pairwise inside the
   bucket with `computeDistance`. Union-find on pairs under threshold forms the final
   clusters. The threshold is configurable and should be tuned against your own
   library's data — grouping is precision-first, since a missed cluster costs nothing
   but a wrong grouping can push a one-of-a-kind photo toward deletion.

Vision was chosen over a plain perceptual hash (`imagehash` pHash) because it holds up
better on crops, exposure shifts and minor subject movement — exactly the "five takes
of the same shot" case a photo library actually has.

Feature prints and scores are cached in SQLite keyed by photo UUID + modification date,
so re-runs are near-instant.

### Best-shot scoring

Per image, weighted and configurable:

- Sharpness — variance of Laplacian, center-weighted, and computed on the largest
  detected face region when faces are present.
- Exposure — histogram clipping at both ends.
- Faces — `VNDetectFaceLandmarksRequest`: count, size, eyes open, mouth/smile.
  Closed eyes is the single strongest "reject this take" signal. With several people
  in frame, the take where *everyone* is smiling with eyes open beats the one with the
  single best face.
- Straightness — three separately-weighted sub-scores: level horizon / frame rotation;
  subject facing the camera (face-landmark yaw/roll, faces only); well-framed subject
  (bounding box centered, not cut off at an edge).
- Existing signals — already-edited, higher resolution: small bonuses. Not
  already-favorited: favoriting habits vary too much between libraries, and favorites
  are rarely applied per-take, so the signal isn't reliable as a keeper indicator.

Every image gets per-criterion sub-scores, not just a total, so the review UI can show
*why* a take won.

**Ambiguity is a first-class outcome.** When the winner and runner-up fall within a
configurable confidence margin, the scorer doesn't pick — it flags the cluster
ambiguous and hands it to you in the review UI. A near-tie means the scorer can't tell,
and auto-picking there would be a coin flip that could send the better shot to
`Cull/Candidates`.

`photocull calibrate` exports a contact-sheet sample (`sample.html`) of proposed
keepers beside the takes they beat, so you can check the clustering/scoring thresholds
against your own judgment before trusting them in the review UI.

## Review UI (`photocull review`)

The governing requirement: you need to be able to **verify that the best version is the
one that stays**, not merely pick one. A thumbnail can't settle "is this one sharper" or
"are their eyes shut", so the comparison view carries real detail.

- **Propose-and-adjust.** A cluster can keep more than one photo — the scorer pre-marks
  its winner KEEP and every other take CULL, but you can flip any photo either way and
  end up with as many keepers as you want. Nothing is staged that you didn't leave
  marked CULL. Keeping 2 of 5 is a normal outcome here, not a workaround.
- **Side-by-side comparison, equal size, synchronized magnification.** Proposed keeper
  on the left, one challenger on the right, both the same size. `←`/`→` choose which
  pane you're on, `↑`/`↓` step through the other takes, `Z` magnifies both at the same
  crop so sharpness and eyes can be judged directly. Per-image sub-scores show under
  each, so you can see why a take won.

  Magnification is bounded by the pixels that actually exist. A locally-cached
  derivative's long side is a median 480px — zooming to "100%" against that would just
  upscale and show blur that isn't in the photograph, on the one screen whose entire
  job is verification. The view never shows more than one source pixel per device
  pixel, labels the real pixel size on screen, and offers an explicit, per-photo,
  user-triggered full-resolution fetch for when the local pixels genuinely can't settle
  it.
- Ambiguous clusters (winner and runner-up within `ambiguity_margin`) get flagged in
  prose, so no silent coin flip stages the better shot — but they still open with a
  suggestion like every other cluster, since withholding one would make the tool least
  helpful exactly where you have the most work to do. No cluster requires a keeper
  before advancing; "none of these is worth keeping" is a valid answer.
- Keyboard first: `←`/`→` to choose the pane, `↑`/`↓` to change which take is being
  compared, `C` to stage the whole cluster in one keystroke, space to toggle
  KEEP/CULL on the challenger, Enter to accept the cluster and advance, S to skip, X to
  keep the entire cluster, U to undo, `F` to **mark as favorite for write-back**, Z to
  magnify. `F` records an intent in the review log; it doesn't touch the live library
  at the moment it's pressed — every library mutation goes through the confirmed
  write-back step below, never mid-session.
- Preload the next few clusters' derivatives so advancing feels instant. Target under
  2 seconds per cluster of your time for the clusters you agree with.
- Persistent session state — closing the browser and coming back resumes exactly where
  you left off.

Run `photocull review --demo` to try the whole UI against a synthetic 24-cluster
library — no Photos access, no Full Disk Access needed.

### Overview of what's staged

Both a running dashboard and a final check, because the per-cluster walk alone gives no
view of the whole:

- **Dashboard, openable at any time mid-session:** clusters reviewed vs remaining,
  running counts of keeping vs staging, and the largest staged clusters flagged for a
  second look. Jump straight to any flagged cluster.
- **Final check before any write-back:** every staged photo as a contact sheet, grouped
  by cluster and shown beside the keeper(s) it lost to. Clicking a staged photo pulls it
  back to KEEP. Write-back only runs when you confirm from this screen — nothing
  reaches `Cull/Candidates` without having been seen in context at least twice.

### Write-back

On finishing a session, via `photoscript`, and only when you pass `--allow-write-back`
(otherwise it's a dry run):

- Keepers — **one or more per cluster**, whatever you left marked KEEP: add to album
  `Cull/Keepers`. Favorite is set **only on the takes you pressed `F` on**, not on every
  keeper.
- Staged — whatever you left marked CULL, after the final-check screen: add to album
  `Cull/Candidates` and set keyword `cull-candidate`.
- Photos that never clustered aren't touched at all — they're keepers by default and
  never appear in either album.
- Nothing else. Deletion is a manual step you perform in Photos.app: open
  `Cull/Candidates`, select all, delete.

## Sweep (`photocull sweep`)

Bulk pass for categories that don't need per-item thought:

- Screenshots and screen recordings older than N months, not favorited.
- Videos under 5s, not favorited.
- Singles (not in any cluster) with sharpness below a threshold.
- Duplicate-by-signature items that Photos' own Duplicates album already caught, so
  they're skipped here.

Everything is selected by default; you deselect keepers. Same write-back model as
review: `--dry-run` to preview, `--allow-write-back` to actually touch Photos, never a
delete.

## Album builder (`photocull albums`)

- Filter to favorites within a date range or trip grouping (`albums list`).
- Drag to sequence, with a contact-sheet overview (`albums serve`, or `--demo` for a
  synthetic two-trip pool with no Photos access at all).
- Export (`albums export`) at 300 dpi: 4x6 = 1800x1200, 5x7 = 2100x1500. Respects your
  existing Photos crop; offers aspect-fill with an adjustable crop box where the source
  ratio doesn't match.
- Output is a print-ready PDF plus the individual full-res JPEGs. Optional bleed
  (`--bleed-mm`) and crop marks (`--crop-marks`) for borderless printing.
- This is the one stage that needs originals, so it downloads from iCloud on demand —
  only for the specific photos in the album you're exporting.

## Performance targets

- Audit on 50k items: under 2 minutes.
- Full cluster/score analysis on 50k items: under 30 minutes, multiprocessed, using
  derivatives rather than originals.
- UI interactions: under 100ms.
