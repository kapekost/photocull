"""The review UI: a local web app, and the only place write-back lives.

This package exists to make one boundary structural rather than aspirational: it is
the **only** package in this repo permitted to import `fastapi`, `starlette`,
`uvicorn` or `photoscript`. `src/photocull/` stays a pure CLI/library that can be
installed, imported and tested with none of them present.

Why the split is worth a whole package rather than a convention:

- **`photoscript` is the write-back driver.** Its API surface includes
  `Album.remove()` and `PhotosLibrary.delete_album()` — calls this project never
  allows. Keeping it out of the core package means the code that *reads* a Photos
  library structurally cannot be the code that mutates it, and the reviewable blast
  radius of that guarantee is one directory rather than the tree.
- **The CLI must keep working without a web server.** `photocull audit`,
  `photocull cluster` and `photocull calibrate` are useful on their own, and a
  FastAPI import on that path would make an ordinary install carry a server it never
  runs. The review dependencies live in the `[review]` extra for the same reason.
- **The boundary is a test, not a promise.** `tests/test_guardrails.py` walks the AST
  of every file under `src/photocull/` and additionally checks in a subprocess that
  importing `photocull.cli` loads none of these modules transitively.

Importing this package itself must stay cheap and dependency-free — submodules pull
in what they need. That is what lets the guardrail suite assert the package exists on
a machine where the `[review]` extra was never installed.
"""
