"""FastAPI + plain HTML/JS UI for sequencing an album and triggering its export.
Independent of photocull_review: needs fastapi/uvicorn but never photoscript, since
the album builder never writes to Photos. Importable without fastapi -- only the
submodules that use it require it, same convention `photocull_review/__init__.py`
already established (see `test_the_review_package_exists_and_is_import_light`)."""
