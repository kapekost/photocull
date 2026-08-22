# photocull

Cull a large Apple Photos library down to keepers, on your own Mac, without sending a
single byte anywhere.

It groups near-identical takes — the seven frames of the same shot you meant to thin
out five years ago — scores each one on sharpness, exposure, eye-openness, framing and
horizon, and proposes which to keep. You review the proposals side by side and decide.
Once you've culled, a bulk sweep clears out the obvious junk (old screenshots, short
videos, exact duplicates), and an album builder turns your favorites into print-ready
4x6/5x7 PDFs.

What it never does is delete anything.

**Status:** complete and exercised end to end against a real, multi-thousand-item
library — audit, clustering, the review UI with write-back, bulk sweep, and the print
album builder all work today. See `docs/SPEC.md` for how each stage works.

## What it will not do

**This app never deletes anything from your Photos library.** Not a photo, not an
album, not a keyword. The complete set of changes it can make is: set favorite, add to
an album, set or add a keyword. Photos it proposes culling are added to a
`Cull/Candidates` album — deleting them stays a deliberate act you perform yourself, in
Photos, after reviewing that album.

This is enforced, not just promised. `tests/test_guardrails.py` walks the syntax tree
of every source file and fails the build on any call or attribute named `delete`,
`remove`, `erase`, `unlink` or `rmtree`, with an empty allowlist, and requires every
occurrence of those words in a comment or string to carry a written justification.

It also will not download your photos from iCloud, except where the job genuinely
requires the original — album export, for the specific photos you're printing. Every
other stage reads only the thumbnails Photos has already cached locally, which on a
typical optimized library is the difference between reading 14,000 files and
downloading them. In the review UI you can also ask for a single full-resolution photo
on demand, when the local pixels genuinely can't settle a comparison.

## Requirements

- macOS, and a Mac whose Photos library is on it
- Python 3.12+ (developed on 3.14)
- **Full Disk Access granted to your terminal** — see below

## Install

This is a developer tool. There's no `.app`, no installer, no signed binary — you run
it from a checkout, in a terminal.

```bash
git clone <this repo> && cd photocull
python3 -m venv .venv && source .venv/bin/activate

pip install -e ".[dev]"                 # audit, clustering, scoring, sweep
pip install -e ".[dev,review]"          # + the review UI (browse clusters, write back)
pip install -e ".[dev,album]"           # + the print-album builder
pip install -e ".[dev,review,album]"    # everything
```

### Full Disk Access, and exactly what it covers

`osxphotos` reads your Photos library's SQLite database directly, and macOS gates that
behind Full Disk Access. You grant it to **the terminal application you run `photocull`
from** (Terminal, iTerm, your editor's integrated terminal — whichever it is), in
**System Settings → Privacy & Security → Full Disk Access**. The app may need
restarting afterwards.

Be clear-eyed about what you're granting: Full Disk Access applies to the *terminal*,
not to `photocull`, so it lets anything you subsequently run in that terminal read any
file on disk. That's a macOS design decision, not this app's. What `photocull` itself
does with it is read your Photos library's metadata database and its locally-cached
thumbnails, read-only. Nothing is uploaded, no network call is made, and no
third-party service is contacted at any point.

If you'd rather not grant it, this app can't work — there's no supported way to read a
Photos library without it, and routing around the permission is explicitly out of
scope.

## Quick start

Try the UIs with zero Photos access first — both ship a synthetic demo library, so you
can see how review and album-sequencing work before granting anything:

```bash
photocull review --demo          # cull a synthetic 24-cluster library
photocull albums serve --demo    # sequence a synthetic two-trip album pool
```

Once you've granted Full Disk Access, the real workflow:

```bash
source .venv/bin/activate

photocull audit                          # read-only: what's actually in your library
photocull audit --json-out out/audit.json

photocull cluster                        # group near-duplicates and score each take
photocull calibrate                      # export a review sample: out/calibration/sample.html
```

`photocull calibrate` writes a contact sheet showing proposed keepers beside the takes
they beat. Open it before trusting any threshold — it's how you find out whether the
grouping matches your own judgment. The settings it suggests go in a `photocull.toml`
next to the repo (copy `photocull.example.toml`; every value ships commented out, so an
untouched copy behaves exactly like the defaults).

```bash
photocull review                         # decide keepers, cluster by cluster (dry run by default)
photocull review --allow-write-back      # actually write favorite/album/keyword back to Photos

photocull sweep --dry-run                # preview screenshots, short videos, exact dupes
photocull sweep --allow-write-back       # write that back too

photocull albums list                    # see trip groupings in your keeper pool
photocull albums serve                   # sequence an album, adjust crops, in the browser
photocull albums export --album-id <id>  # render the print-ready PDF + full-res JPEGs
```

Every write-back command defaults to a dry run and needs an explicit flag to touch your
library — and even then, the only things it can touch are favorite/album/keyword.

Run the tests with `pytest -q`. They use synthetic fixtures throughout and need no
Photos access at all.

## How it works

`docs/SPEC.md` walks through the pipeline stage by stage — the clustering algorithm,
the scoring model, the review UI's design decisions, and the album export pipeline —
for anyone who wants the detail behind what each command above actually does.

## Your data

Everything stays on your machine. The analysis cache (`out/analysis.db`), the
calibration sample, and your `photocull.toml` are all git-ignored and never leave the
disk. There's no telemetry, no crash reporting, and no account.

## License

MIT — see `LICENSE`.
