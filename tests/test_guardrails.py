"""Structural guardrails, enforced as tests rather than as tick-time greps.

Phase 1b is the first phase that writes to the Photos library at all, and the first
that introduces a second package. Both of the promises that protect it were, until
now, manual steps in `docs/orchestration/PLAYBOOK.md` that a human ran and eyeballed:

  1. `src/photocull/` never imports a review dependency (fastapi/starlette/uvicorn/
     photoscript), so the CLI stays installable and importable without them.
  2. No file under `src/` calls a destructive API against anything, which is the
     machine-checkable half of CLAUDE.md hard rule #1.

A manual gate is sufficient only while nothing in the tree can plausibly violate it.
`photoscript` — whose API surface includes `Album.remove()` and
`PhotosLibrary.delete_album()` — is about to be imported for the first time, so that
window has closed. These tests are the gate now.

Each checker below is paired with a test that feeds it a synthetic violation, because
a guardrail that has never been observed to fail is not evidence of anything. This
project has found a test pinned to a coincidence in seven consecutive ticks; see
DECISIONS.md `tests-must-fail-on-a-never-drawn-raster` and
`assertions-must-name-the-stage-they-pin`.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
CORE_PACKAGE = SRC / "photocull"

#: Packages that only `photocull_review` may import. `photoscript` is here because it
#: is the write-back driver: keeping it out of the core package means the library that
#: reads a Photos library cannot, structurally, be the one that mutates it.
REVIEW_ONLY_DEPENDENCIES = frozenset({"fastapi", "starlette", "uvicorn", "photoscript"})

#: Verbs that may not appear as a called function or an accessed attribute anywhere
#: under `src/`. Matched per name-token, so `delete_album` and `removeAll` both hit.
DESTRUCTIVE_VERBS = frozenset(
    {"delete", "remove", "erase", "unlink", "rmtree", "trash", "destroy"}
)

#: Deliberately empty. A destructive call is never justified in this codebase — the
#: complete set of allowed Photos-library mutations is favorite / album / keyword
#: (GUARDRAILS.md). This exists so that adding one requires writing down a reason
#: here, in a diff, rather than quietly widening a regex.
ALLOWED_DESTRUCTIVE_CALLS: dict[tuple[str, str], str] = {}

#: The prose lines that the PLAYBOOK's `delete|remove|erase` grep has been matching.
#: Ticks 13-18 eyeballed two of these every tick; the third arrived with commit
#: `e23c235` and was never logged, which is precisely the drift this pins.
JUSTIFIED_PROSE: dict[tuple[str, str], str] = {
    (
        "photocull/cache.py",
        'never deletes from under any circumstances.)"""',
    ): "Docstring describing what the analysis cache does NOT do. No code.",
    (
        "photocull/derivatives.py",
        "# silently restore the largest-derivative behaviour this selection exists "
        "to remove.",
    ): "Comment about superseding a derivative-selection rule. Nothing in the "
    "Photos library is involved. Re-wrapped onto one line in Phase 1b Task 2 when "
    "the fallback guard moved into `_smallest` so both selectors share it; the gate "
    "correctly refused to accept the reflowed text on the strength of the old "
    "entry, which is the point of pinning exact text rather than a pattern.",
    (
        "photocull/calibrate.py",
        "written to your library and nothing is deleted, ever - this is a "
        "read-only report.</dd>",
    ): "HTML shown to the owner in the calibration contact sheet, telling them "
    "the report is read-only. String literal, not code.",
    (
        "photocull_review/__init__.py",
        "`Album.remove()` and `PhotosLibrary.delete_album()` — calls this project "
        "never",
    ): "Docstring NAMING the photoscript calls this project forbids, to explain "
    "why the package boundary exists. Text updated when the docstring's CLAUDE.md "
    "cross-reference was generalized for the public repo; still the same "
    "justification — the gate goes red on this exact line the moment it names "
    "those calls.",
    (
        "photocull_review/resume.py",
        "library that keeps living in the meantime. Photos get edited, deleted, "
        "and reshuffled into",
    ): "Module docstring describing changes the OWNER makes in Photos.app between "
    "review sessions — the events resume reconciliation exists to detect. This app "
    "does none of them. No code.",
    (
        "photocull_review/resume.py",
        "`clusters` would declare two thirds of it deleted.",
    ): "Docstring explaining why `library_uuids=None` means 'not checked' rather "
    "than 'the library is empty': inferring the library from the clustered photos "
    "would misreport 9,737 of 14,235 real photos as gone. Describes a reporting "
    "error to avoid, not an action. No code.",
    (
        "photocull_review/resume.py",
        "# Nothing is deleted and nothing is applied: the whole log is held back, "
        "and the",
    ): "Comment on the `force_new_session` branch, stating the property that makes "
    "starting over safe — the append-only log keeps every earlier decision. The "
    "reassurance a reader of this branch most needs; rewording it to dodge the "
    "grep would cost exactly the sentence that matters.",
    (
        "photocull_review/api.py",
        "single-user local app and removes a whole class of staleness bug.\"\"\"",
    ): "Docstring on `live_decisions`, explaining why the log is re-read per request "
    "instead of cached. 'Removes' is about a category of bug, not about any photo, "
    "file or row. No code.",
    (
        "photocull_review/api.py",
        "#: of what's to be deleted\"; the full list is the final check, and a "
        "dashboard that",
    ): "Comment quoting the OWNER's own words for what the dashboard is for "
    "(`staged-set-gets-dashboard-and-final-check`). 'Deleted' names the manual step "
    "the owner takes later inside Photos.app — ⌘A + Delete in one album — which this "
    "app has no way to perform (`photos-applescript-genuinely-cannot-delete`). "
    "Quoting them exactly is the point; paraphrasing to dodge the grep would lose "
    "whose requirement it is.",
    (
        "photocull_review/api.py",
        "manual delete would then take a favourited photo with it.\"\"\"",
    ): "Docstring on `_check_favorites`, explaining why favouriting a photo that is "
    "staged for culling is refused. Describes a consequence of the owner's OWN "
    "manual deletion in Photos.app, and exists to prevent it costing them a photo "
    "they wanted. Nothing here deletes anything. No code.",
    (
        "photocull/cli.py",
        '"Without it, write-back can only be dry-run. Never deletes anything."',
    ): "The `--allow-write-back` help text — the sentence the owner reads at the "
    "moment they type the one flag in this app that mutates their library. Stating "
    "the boundary (favorite/album/keyword, never deletion) is the whole value of the "
    "line; rewording it to dodge this grep would cost exactly the reassurance that "
    "matters most, at the one place it matters. Same class as `calibrate.py`'s "
    "read-only notice. String literal, not code.",
    (
        "photocull/cli.py",
        '"Cull/ and sets the cull-candidate keyword. Never deletes anything."',
    ): "Phase 2 Task 4's `sweep --allow-write-back` help text — the identical sentence "
    "as the review's flag above, for the same reason at the same class of moment: the "
    "owner reads it right before typing the one flag that mutates their library via "
    "`photocull sweep`. String literal, not code.",
    (
        "photocull_review/writeback.py",
        "# have to manually delete despite having favourited it.",
    ): "Comment in `plan_writeback` on why only a keeper is ever favourited, naming "
    "the same consequence the `api.py` entry above does — the owner's OWN manual "
    "delete inside `Cull/Candidates`. Repeated here deliberately, because this is the "
    "module that would perform the mutation and the rule is invisible from the API "
    "two packages away. Comment, no code. Text reworded during the public-repo tone "
    "pass; same justification, updated pin.",
    (
        "photocull/album_export.py",
        "deleted from Photos since the album was sequenced) must not lose the whole export.",
    ): "Docstring on `export_album` (Phase 3 Task 8), explaining why a uuid missing "
    "from `photos_by_uuid` is reported in `failed` rather than raising -- e.g. the "
    "owner deleted that photo from Photos themselves, by hand, between sequencing "
    "the album and exporting it. Describes a scenario this function tolerates, not "
    "an action it performs. No code. Task 10 Step 5 added a further sentence to this "
    "same docstring (the `crop_offsets` parameter note), which pushed the closing "
    "`\"\"\"` onto its own line -- this entry's text updated to match, not re-justified, "
    "since it is the same sentence Tick 52 already justified.",
}

PROSE_GREP = re.compile(r"delete|remove|erase", re.IGNORECASE)

#: SQL that destroys or rewrites a stored row. Matched inside string literals, because
#: that is where SQL lives, and the AST scanner above cannot see into one: to
#: `destructive_names`, `conn.execute("DELETE FROM x")` is a call to `execute`.
#:
#: Added in Phase 1b Task 4, for a measured reason rather than a tidy one. The decision
#: log is the only artifact this project produces that cannot be recomputed, and its
#: append-only promise had **no** mechanical enforcement: `UPDATE` and `DROP` are absent
#: from `PROSE_GREP` entirely, and a `DELETE FROM` inside a string literal is classified
#: as *prose* by the gate below — so it fails, but as an unjustified comment, and could
#: be waved through by adding a `JUSTIFIED_PROSE` entry.
#:
#: `INSERT OR REPLACE` is here because of what the Task 4 spike found: SQLite resolves
#: REPLACE **below** the authorizer layer, so `DecisionLog`'s runtime guard — which does
#: deny UPDATE/DELETE/DROP/ALTER — silently lets a conflicting REPLACE clobber a row.
#: Verified by running it, not read in a doc. `cache.py` already uses that idiom twice,
#: which makes it exactly the line someone copies into the wrong module.
SQL_MUTATION = re.compile(
    r"\b(?:DELETE\s+FROM\s+\w+"
    r"|UPDATE\s+\w+\s+SET"
    r"|DROP\s+(?:TABLE|INDEX|VIEW)(?:\s+IF\s+EXISTS)?\s+\w+"
    r"|INSERT\s+OR\s+REPLACE\s+INTO\s+\w+"
    r"|TRUNCATE\s+\w+)",
    re.IGNORECASE,
)

#: Non-empty, unlike `ALLOWED_DESTRUCTIVE_CALLS` — a cache is *defined* by being
#: discardable. The point of listing them is that the analysis cache is the only place in
#: the tree where that is true.
ALLOWED_SQL_MUTATIONS: dict[tuple[str, str], str] = {
    (
        "photocull/cache.py",
        "DROP TABLE IF EXISTS feature_prints",
    ): "Rebuilds the disposable analysis cache on an ANALYSIS_VERSION bump. Local "
    "SQLite only; recomputable in one slow run.",
    (
        "photocull/cache.py",
        "DROP TABLE IF EXISTS scores",
    ): "Same rebuild, second table.",
    (
        "photocull/cache.py",
        "INSERT OR REPLACE INTO feature_prints",
    ): "Re-analysing a photo overwrites its own cached vector, keyed by "
    "(uuid, mod_date, analysis_key). Idempotent by construction.",
    (
        "photocull/cache.py",
        "INSERT OR REPLACE INTO scores",
    ): "Same, for the scores table.",
    (
        "photocull/album_store.py",
        "UPDATE albums SET",
    ): "An album's photo sequence is draft state the owner is actively rearranging "
    "before export, not an audit trail like decisions.db/writeback.db -- there is "
    "nothing to protect by keeping it append-only. See Task 7's own note in "
    "docs/plans/2026-08-19-phase-3-album-builder.md.",
}


def sql_statements(source: str) -> set[str]:
    """Destructive SQL found in any string literal, whitespace-normalised.

    String literals come from the AST, so implicitly concatenated SQL split across
    several source lines is seen as the one statement it actually is."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for match in SQL_MUTATION.finditer(node.value):
                found.add(" ".join(match.group(0).split()))
    return found


def source_files(root: Path) -> list[Path]:
    """Every tracked Python file under `root`, sorted so failures are reproducible."""
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _name_tokens(name: str) -> set[str]:
    """Split an identifier into lowercase words on `_` and camelCase boundaries."""
    parts: list[str] = []
    for chunk in name.split("_"):
        parts.extend(re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+", chunk))
    return {p.lower() for p in parts}


def imported_roots(source: str) -> set[str]:
    """Top-level package names imported by `source`, including inside functions.

    An AST walk rather than a grep, because `photos_source.open_library` already
    lazy-imports `osxphotos` inside a function body and a review dependency could
    hide the same way.
    """
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def destructive_names(source: str) -> set[str]:
    """Called functions and accessed attributes whose name contains a destructive verb."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        name: str | None = None
        if isinstance(node, ast.Attribute):
            name = node.attr
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
        if name and _name_tokens(name) & DESTRUCTIVE_VERBS:
            found.add(name)
    return found


def calls_dot_export(source: str) -> bool:
    """Whether `source` actually *calls* a `.export(...)` method, as opposed to merely
    mentioning `.export()` in a docstring or comment.

    An AST walk, not the plan's original `re.search(r"\\.export\\(", text)` — that regex
    matches string-literal content too, and three files in this project already carry
    prose like `never `.path` or `.export()``` explaining why they deliberately don't
    call it. A naive grep would flag those as violations of the very rule they document.
    Same reasoning as `destructive_names` using AST instead of a delete/remove/erase
    grep for structural (non-diff) checks.
    """
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "export"
        ):
            return True
    return False


def prose_line_numbers(source: str) -> set[int]:
    """Lines that are entirely comment, blank, or inside a string literal.

    Used to prove a `delete|remove|erase` match is prose rather than code. Comments
    are found by inspection (the AST discards them); string literals come from the
    AST, so an f-string or a docstring counts and a bare identifier never can.
    """
    lines = set()
    for number, text in enumerate(source.splitlines(), start=1):
        stripped = text.strip()
        if not stripped or stripped.startswith("#"):
            lines.add(number)
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return lines


# --------------------------------------------------------------------------------
# 1. The package boundary
# --------------------------------------------------------------------------------


def test_photocull_never_imports_a_review_dependency():
    """`src/photocull/` is a pure CLI/library. The review server is a second package."""
    offenders = {
        str(path.relative_to(SRC)): sorted(
            imported_roots(path.read_text()) & REVIEW_ONLY_DEPENDENCIES
        )
        for path in source_files(CORE_PACKAGE)
    }
    offenders = {path: names for path, names in offenders.items() if names}
    assert offenders == {}, (
        f"src/photocull/ imported a review-only dependency: {offenders}. "
        "It belongs in src/photocull_review/."
    )


def test_the_import_scanner_actually_catches_an_import():
    """The boundary test above must be able to fail. Proven, not assumed."""
    assert imported_roots("import fastapi") & REVIEW_ONLY_DEPENDENCIES == {"fastapi"}
    assert imported_roots("from photoscript import PhotosLibrary") & (
        REVIEW_ONLY_DEPENDENCIES
    ) == {"photoscript"}
    assert imported_roots(
        "def go():\n    import uvicorn\n    return uvicorn"
    ) & REVIEW_ONLY_DEPENDENCIES == {"uvicorn"}


def test_importing_the_cli_does_not_load_fastapi():
    """Transitive check the AST cannot make: nothing pulls a review dep in at runtime.

    A subprocess, because `fastapi` is installed in this environment and may already
    be in this process's `sys.modules` for unrelated reasons.
    """
    probe = (
        "import sys; import photocull.cli; "
        f"print(sorted(set(sys.modules) & {set(REVIEW_ONLY_DEPENDENCIES)!r}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", (
        f"importing photocull.cli loaded {result.stdout.strip()}"
    )


def test_review_dependencies_are_an_optional_extra_not_a_base_dependency():
    """Installing `photocull` must not drag in a web server.

    This is what keeps the boundary true for a fresh install rather than only for
    this checkout: if fastapi were a base dependency, importing it from the core
    package would work everywhere and nothing would ever break.
    """
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    base = {
        re.split(r"[<>=!\[ ]", spec, maxsplit=1)[0]
        for spec in pyproject["project"]["dependencies"]
    }
    extras = pyproject["project"]["optional-dependencies"]
    review = {
        re.split(r"[<>=!\[ ]", spec, maxsplit=1)[0] for spec in extras.get("review", [])
    }

    assert base & REVIEW_ONLY_DEPENDENCIES == set(), (
        f"{sorted(base & REVIEW_ONLY_DEPENDENCIES)} must move to the [review] extra"
    )
    assert REVIEW_ONLY_DEPENDENCIES - {"starlette"} <= review, (
        f"[review] extra is missing {sorted(REVIEW_ONLY_DEPENDENCIES - {'starlette'} - review)}"
    )


def test_the_review_package_exists_and_is_import_light():
    """`photocull_review` is importable without fastapi — only its submodules need it."""
    result = subprocess.run(
        [sys.executable, "-c", "import photocull_review; print('ok')"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


#: Only this file may call `.export(` against a real PhotoInfo, and only this file may
#: read the unmodified original's `.path`. Both trigger an iCloud download
#: (CLAUDE.md hard rule #2's Phase 3 exception) and must stay auditable at one seam, the
#: same reasoning `phase1-reads-local-derivatives-only` already established for
#: `derivative_path`.
ORIGINALS_CHOKEPOINT = "photocull/originals.py"


def test_no_file_outside_originals_calls_export():
    offenders = {}
    for path in source_files(SRC):
        rel = str(path.relative_to(SRC))
        if rel == ORIGINALS_CHOKEPOINT:
            continue
        if calls_dot_export(path.read_text()):
            offenders[rel] = "calls .export("
    assert offenders == {}, (
        f"iCloud-triggering .export() called outside {ORIGINALS_CHOKEPOINT}: {offenders}. "
        "Route it through originals.py or this violates CLAUDE.md hard rule #2."
    )


def test_the_export_scanner_actually_catches_a_call():
    """Proven, not assumed — same discipline as every other guardrail scanner here."""
    assert calls_dot_export("photo.export(dest)") is True


def test_the_export_scanner_does_not_fire_on_prose():
    """A docstring *mentioning* `.export()` is not a call to it.

    Real regression: `derivatives.py`, `photos_source.py` and `imaging.py` each carry
    prose explaining they deliberately never call `.export()` — the plan's original
    `re.search(r"\\.export\\(", text)` matched all three and would have permanently
    broken this gate for the very files documenting the rule it enforces.
    """
    source = '"""Never touches `.path` or `.export()`. See CLAUDE.md hard rule #2."""\n'
    assert calls_dot_export(source) is False


# --------------------------------------------------------------------------------
# 2. The app's own hard constraint
# --------------------------------------------------------------------------------


def test_no_source_file_calls_a_destructive_api():
    """CLAUDE.md hard rule #1, made structural.

    Covers attribute *access* as well as calls, so `getattr`-free indirection like
    `fn = album.remove` is caught at the point the name is taken.
    """
    findings = {
        (str(path.relative_to(SRC)), name)
        for path in source_files(SRC)
        for name in destructive_names(path.read_text())
    }
    unjustified = {key for key in findings if key not in ALLOWED_DESTRUCTIVE_CALLS}
    assert unjustified == set(), (
        f"destructive API calls found: {sorted(unjustified)}. The only Photos-library "
        "mutations this app may perform are: set favorite, add to album, set/add keyword."
    )


def test_the_destructive_scanner_actually_catches_a_call():
    """The rule above must be able to fail — including on photoscript's real API."""
    assert destructive_names("library.delete_album(a)") == {"delete_album"}
    assert destructive_names("album.remove(photos)") == {"remove"}
    assert destructive_names("p.unlink()") == {"unlink"}
    assert destructive_names("shutil.rmtree(d)") == {"rmtree"}
    assert destructive_names("fn = album.removeAll") == {"removeAll"}


def test_the_destructive_scanner_does_not_fire_on_ordinary_code():
    """`dict.pop` and friends are not destructive; similarity.py uses pop three times."""
    assert destructive_names("height.pop(key)") == set()
    assert destructive_names("cluster.dropped_criteria") == set()
    assert destructive_names("shutil.copy2(a, b)") == set()


@pytest.mark.parametrize("location", sorted(JUSTIFIED_PROSE))
def test_every_justified_prose_line_still_exists(location):
    """A justification that no longer describes anything must be deleted, not kept."""
    relative_path, text = location
    source = (SRC / relative_path).read_text()
    assert text in source, (
        f"{relative_path} no longer contains the justified line {text!r} — "
        "remove its entry from JUSTIFIED_PROSE."
    )


def test_the_delete_grep_gate_matches_only_justified_prose():
    """Pins what the PLAYBOOK's grep has been matching, and proves each match is prose.

    Two separate properties, and the order matters. (a) Every match must be a comment
    or a string literal — no allowlist entry can buy an exemption from that, because a
    matching *code* line means the app grew a delete path. (b) The set of prose matches
    must be exactly the justified set, so a fourth line fails until someone writes down
    why it is there.
    """
    matches: dict[tuple[str, str], int] = {}
    code_matches: list[str] = []
    for path in source_files(SRC):
        source = path.read_text()
        prose = prose_line_numbers(source)
        relative_path = str(path.relative_to(SRC))
        for number, text in enumerate(source.splitlines(), start=1):
            if not PROSE_GREP.search(text):
                continue
            if number in prose:
                matches[(relative_path, text.strip())] = number
            else:
                code_matches.append(f"{relative_path}:{number}: {text.strip()}")

    assert code_matches == [], (
        f"a delete/remove/erase match landed in executable code: {code_matches}"
    )
    assert set(matches) == set(JUSTIFIED_PROSE), (
        "unjustified prose matches: "
        f"{sorted(set(matches) - set(JUSTIFIED_PROSE))}; stale justifications: "
        f"{sorted(set(JUSTIFIED_PROSE) - set(matches))}"
    )


def test_the_prose_classifier_distinguishes_code_from_comment():
    """`prose_line_numbers` must not call everything prose, or the gate above is dead."""
    source = 'x = 1  # remove this later\ny = "nothing is deleted"\nz.delete()\n'
    assert prose_line_numbers(source) == {2}


# --------------------------------------------------------------------------------
# 3. SQL that rewrites or destroys a stored row
# --------------------------------------------------------------------------------


def test_no_source_file_runs_a_destructive_sql_statement():
    """Everything under `src/` writes SQL forward-only, except the disposable cache."""
    findings = {
        (str(path.relative_to(SRC)), statement)
        for path in source_files(SRC)
        for statement in sql_statements(path.read_text())
    }
    unjustified = {key for key in findings if key not in ALLOWED_SQL_MUTATIONS}
    assert unjustified == set(), (
        f"destructive SQL found: {sorted(unjustified)}. Add a justification to "
        "ALLOWED_SQL_MUTATIONS only if the table is genuinely recomputable."
    )
    stale = set(ALLOWED_SQL_MUTATIONS) - findings
    assert stale == set(), f"justifications describing SQL that no longer exists: {sorted(stale)}"


def test_the_decision_log_contains_no_destructive_sql_whatsoever():
    """The append-only promise, as a test rather than as a docstring.

    Separate from the tree-wide check above so it cannot be weakened by an allowlist
    entry: for this one file the answer must be the empty set, permanently. It is the
    only artifact this project produces that cannot be recomputed from the library."""
    source = (SRC / "photocull_review" / "decisions.py").read_text()
    assert sql_statements(source) == set()


def test_the_sql_scanner_catches_every_statement_form_it_claims_to():
    """Each alternation proven to fire, including across an implicit concatenation."""
    assert sql_statements('c.execute("DELETE FROM decision_marks")') == {
        "DELETE FROM decision_marks"
    }
    assert sql_statements('c.execute("UPDATE marks SET mark = ?")') == {"UPDATE marks SET"}
    assert sql_statements('c.execute("DROP TABLE undos")') == {"DROP TABLE undos"}
    assert sql_statements('c.execute("drop table if exists undos")') == {
        "drop table if exists undos"
    }
    assert sql_statements('c.execute("INSERT OR REPLACE INTO scores VALUES (?)")') == {
        "INSERT OR REPLACE INTO scores"
    }
    # The form the AST exists to catch: one statement, three source lines.
    assert sql_statements(
        'c.execute(\n    "DELETE   FROM "\n    "decision_batches WHERE x = 1"\n)'
    ) == {"DELETE FROM decision_batches"}


def test_the_sql_scanner_does_not_fire_on_forward_only_sql():
    """A gate that flagged ordinary inserts would be turned off within a week."""
    assert sql_statements('c.execute("INSERT INTO undos VALUES (?)")') == set()
    assert sql_statements('c.execute("SELECT batch_id FROM decision_batches")') == set()
    assert sql_statements('c.execute("CREATE TABLE IF NOT EXISTS undos (x TEXT)")') == set()
    assert sql_statements('x = "the owner can undo a decision"') == set()


# --------------------------------------------------------------------------------
# The frontend — what a browser asset can and cannot get wrong
# --------------------------------------------------------------------------------

#: Where each package's frontend lives. Plain files, no build step. A tuple, not a
#: single path, since Task 10 (Phase 3) gave `photocull_album` its own `static/`
#: independent of `photocull_review`'s -- this scanner predates that package and was
#: never widened when it landed, the same shape of gap Ticks 51/52 each found in a
#: different guardrail scanner (the SQL-mutation allowlist, the delete-grep prose
#: allowlist): a scanner's coverage was fixed to whatever existed before Phase 3, so a
#: new file doing something already-legitimate trips it once, by design.
STATIC_DIRS: tuple[Path, ...] = (
    SRC / "photocull_review" / "static",
    SRC / "photocull_album" / "static",
)

#: Source extensions this repo ships that the Python gates above do **not** read.
#: Listed rather than discovered, so adding a fifth kind of source file to `src/`
#: fails here and forces a decision about which gate should cover it.
NON_PYTHON_SOURCE_SUFFIXES = frozenset({".html", ".css", ".js"})


def test_every_non_python_source_file_is_a_frontend_asset():
    """Pins the delete-grep's scope as a decision rather than an accident.

    `source_files` globs `*.py`, so every gate above is blind to anything else in
    `src/`. That is correct for the frontend and only for the frontend: browser
    JavaScript holds no filesystem handle, no `photoscript` import and no Photos
    automation permission, so `Set.delete` on an in-memory favourites set is not the
    thing the delete gate exists to catch — and putting ordinary JS collection calls
    into `JUSTIFIED_PROSE` would fill the ledger with noise until nobody read it.

    What that reasoning depends on is that no *other* kind of source file exists here.
    A shell script or an AppleScript file under `src/` would inherit the same blindness
    while holding real capability, so this test fails the moment one appears.
    """
    def is_build_output(path: Path) -> bool:
        """Both are git-ignored build artefacts, not source anyone wrote."""
        return "__pycache__" in path.parts or any(
            part.endswith(".egg-info") for part in path.parts
        )

    stray = [
        str(path.relative_to(SRC))
        for path in sorted(SRC.rglob("*"))
        if path.is_file()
        and path.suffix != ".py"
        and not is_build_output(path)
        and not (path.suffix in NON_PYTHON_SOURCE_SUFFIXES and path.parent in STATIC_DIRS)
    ]
    assert stray == [], (
        f"non-Python source outside the frontend, uncovered by every gate above: {stray}"
    )


def test_the_frontend_references_no_external_origin():
    """`review-server-serves-nothing-from-a-cdn`, enforced on the assets themselves.

    Task 6 turned off FastAPI's docs routes because they fetch Swagger UI from
    `cdn.jsdelivr.net`. The same rule has to hold for the files this app actually
    serves, and it needs a gate rather than review: `default-src 'self'` blocks an
    external fetch *silently* — the page renders half-built and the only evidence is a
    line in a browser console nobody has open. A local-only app must also make no
    outbound request at all, which is a stronger claim than "it would be blocked".
    """
    offenders: list[str] = []
    for static_dir in STATIC_DIRS:
        for path in sorted(static_dir.iterdir()):
            if path.suffix not in NON_PYTHON_SOURCE_SUFFIXES:
                continue
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                # `//` alone is a comment in JS and CSS; a scheme or a protocol-relative
                # URL in an attribute or a stylesheet is what reaches the network.
                for pattern in (r"https?://", r'(?:src|href|url\()\s*=?\s*["\']?//'):
                    if re.search(pattern, line):
                        offenders.append(f"{static_dir.parent.name}/static/{path.name}:{number}: {line.strip()}")
    assert offenders == [], f"the frontend reaches off-origin: {offenders}"
