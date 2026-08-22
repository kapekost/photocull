# AGENTS.md

Guidance for AI agents (Claude Code, Cursor, etc.) working on photocull, a
local-only macOS tool for culling an Apple Photos library. `README.md` covers
what the tool does; `docs/SPEC.md` covers how it works. This file covers what
you can and can't touch when changing the code.

## The rule that can't bend

**The only mutations this codebase is allowed to make to a Photos library,
anywhere, are: set favorite, add to an album, set or add a keyword.** Nothing
ever deletes a photo, an album, or a keyword.

This is enforced, not just documented. `tests/test_guardrails.py` walks the
syntax tree of every file under `src/` and fails the build on any call or
attribute named `delete`, `remove`, `erase`, `unlink`, `rmtree`, `trash`, or
`destroy` — zero of those calls are allowed, anywhere. The same test also
catches those words in comments and strings, and requires a reason logged in
the test file itself before it'll let one through. If a change needs one of
those calls, it doesn't belong in this codebase — find another way.

Run `pytest tests/test_guardrails.py` before considering any change to
write-back or Photos access done. Run the full suite, `pytest -q`, before
considering any change done at all.

## Where Photos access is allowed to happen

- Reads go through `osxphotos`, read-only, everywhere.
- Writes go through `photoscript`, and only from
  `src/photocull_review/writeback.py`. Don't call `photoscript` anywhere else.
- iCloud downloads happen in exactly one place: `src/photocull/originals.py`
  (album export, on demand, only for the photos being exported). Every other
  stage reads local derivatives via `src/photocull/derivatives.py`. Don't add
  a second path that pulls originals.

## Local-only

No cloud services, no third-party photo APIs, no network calls beyond what's
already there. Nothing about this tool phones home.

## Full Disk Access

Real-library work needs it — macOS gates the Photos SQLite database behind
it. Synthetic test fixtures don't. If you're adding tests, use the synthetic
fixtures; a test that needs a real library or Full Disk Access to pass
doesn't belong in the suite.

## Docs

Update `README.md` and `docs/SPEC.md` when a change alters behavior they
describe. They're the source of truth for how the tool works, not this file.
