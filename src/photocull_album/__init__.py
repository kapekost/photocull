"""FastAPI + plain HTML/JS UI for sequencing an album and triggering its export.
Independent of photocull_review: needs fastapi/uvicorn but never photoscript, since
Phase 3 never writes to Photos (see docs/plans/2026-08-19-phase-3-album-builder.md,
`phase-3-gets-its-own-package-not-photocull-review`). Importable without fastapi --
only the submodules that use it require it, same convention `photocull_review/__init__.py`
already established (see `test_the_review_package_exists_and_is_import_light`)."""
