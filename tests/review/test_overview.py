"""Task 13, server side: the final check is a *gate*, not a report.

Three things here are load-bearing, and each is written against a plausible wrong
implementation rather than the happy path.

1. **The confirmation names what was confirmed.** A boolean flag would let the owner
   confirm 40 staged photos, review another 300 clusters, and hand Task 14 a write-back
   authorised for a set nobody ever saw — which is precisely what
   `staged-set-gets-dashboard-and-final-check` exists to prevent ("nothing reaches
   `Cull/Candidates` without having been seen in context"). The confirmation therefore
   carries a digest of the staged set, and `writeback_ready()` is a comparison rather
   than a flag read: it goes false again by construction the moment the set moves, and
   no code has to remember to reset it.
2. **A decision carries favourites, and every surface that reads a decision back has to
   carry them too.** Pulling a photo back to KEEP from the contact sheet re-submits its
   whole cluster, so if the round trip through `/api/session` drops the favourite flags
   the owner set an hour ago, one click silently un-favourites the keepers of that
   cluster — a data-loss shape in a project whose first rule is that nothing is lost.
3. **The sheet is the one surface whose size is the library's, not a cluster's.** At the
   0.48 cut the real staged set is ~1,800-2,700 photos, so the route pages.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from photocull.config import ClusterConfig
from photocull_review import server
from photocull_review.api import FINAL_CHECK_PAGE, ReviewSession
from photocull_review.decisions import DecisionLog

from ..fixtures import make_cluster

TOKEN = "n5Qk_test-token-with-plenty-of-entropy_7Xa"
BASE_URL = "http://127.0.0.1:54321"
WRITE_HEADERS = {"Sec-Fetch-Site": "same-origin"}


@pytest.fixture
def clusters():
    return [
        make_cluster(["c1", "c2", "c3", "c4", "c5"], day=0),
        make_cluster(["a1", "a2"], day=1, ambiguous=True),
        make_cluster(["p1", "p2"], day=2),
    ]


@pytest.fixture
def log(tmp_path):
    with DecisionLog(tmp_path / "decisions.db") as opened:
        yield opened


def _display(record):
    from photocull_review.session import DisplayInfo

    return DisplayInfo(uuid=record.uuid, local_path=record.display_path, long_side=1024)


@pytest.fixture
def review(clusters, log):
    return ReviewSession(
        clusters,
        log=log,
        session_id="test-session",
        config=ClusterConfig(),
        resolver=_display,
    )


@pytest.fixture
def client(review):
    app = server.build_app(token=TOKEN, review=review)
    opened = TestClient(app, base_url=BASE_URL)
    assert opened.get(f"/?t={TOKEN}").status_code == 200
    return opened


def keys(review):
    return [entry["cluster_key"] for entry in review.document["clusters"]]


def decide(client, key, marks, **body):
    return client.post(
        f"/api/clusters/{key}/decision",
        json={"marks": marks, **body},
        headers=WRITE_HEADERS,
    )


def confirm(client, digest):
    return client.post(
        "/api/final-check/confirm", json={"staged_digest": digest}, headers=WRITE_HEADERS
    )


# --- favourites survive the round trip --------------------------------------------


def test_a_recorded_decision_reports_which_photos_were_favourited(client, review):
    """Otherwise the only record of the owner's `F` presses is inside the log."""
    key = keys(review)[0]
    body = decide(
        client,
        key,
        {"c1": "keep", "c2": "keep", "c3": "cull", "c4": "cull", "c5": "cull"},
        favorites=["c2"],
    ).json()
    assert body["favorites"] == ["c2"]


def test_the_session_document_carries_favourites_so_a_reload_keeps_them(client, review):
    """A reload is ordinary — the session survives a closed browser by design — and the
    compare view seeds its state from this document. Without the favourites here, re-
    recording a cluster after a reload silently drops every `F` press in it."""
    key = keys(review)[1]
    decide(client, key, {"a1": "keep", "a2": "cull"}, favorites=["a1"])

    decided = client.get("/api/session").json()["decided"][key]
    assert decided["marks"] == {"a1": "keep", "a2": "cull"}
    assert decided["favorites"] == ["a1"]


def test_a_cluster_with_no_favourites_says_so_rather_than_omitting_the_field(
    client, review
):
    key = keys(review)[1]
    decide(client, key, {"a1": "keep", "a2": "cull"})
    assert client.get("/api/session").json()["decided"][key]["favorites"] == []


# --- the staged digest -------------------------------------------------------------


def test_the_digest_names_the_staged_set_and_moves_only_when_that_set_moves(
    client, review
):
    """Two properties in one test because they are two halves of one claim: the digest
    is a function of *what is staged*, and of nothing else."""
    first = keys(review)[1]
    empty = review.staged_digest()

    decide(client, first, {"a1": "keep", "a2": "cull"})
    staged_one = review.staged_digest()
    assert staged_one != empty

    # Deciding another cluster with nothing staged in it must not move the digest: the
    # owner has looked at more clusters, but the set to be written back is unchanged.
    second = keys(review)[2]
    decide(client, second, {"p1": "keep", "p2": "keep"})
    assert review.staged_digest() == staged_one

    # Staging one more photo does move it.
    decide(client, second, {"p1": "keep", "p2": "cull"})
    assert review.staged_digest() != staged_one


def test_the_digest_changes_when_the_same_cluster_stages_a_different_photo(client, review):
    """A digest over cluster keys alone would pass every other test here and still let
    the owner confirm one photo and write back another: flipping which take of a pair is
    kept leaves the count, the clusters and the batch count identical, and changes
    exactly the thing write-back acts on."""
    key = keys(review)[1]
    decide(client, key, {"a1": "keep", "a2": "cull"})
    first = review.staged_digest()

    decide(client, key, {"a1": "cull", "a2": "keep"})
    assert review.staged_digest() != first


def test_the_digest_returns_to_its_earlier_value_when_the_staged_set_does(client, review):
    """An undo that puts the staged set back exactly as it was leaves the confirmation
    honest — the owner has already seen this precise set. Pinning it rules out a digest
    built from batch ids or timestamps, which would look identical on every other test
    here while making the confirmation expire for reasons the owner cannot see."""
    key = keys(review)[1]
    decide(client, key, {"a1": "keep", "a2": "cull"})
    staged = review.staged_digest()

    decide(client, key, {"a1": "keep", "a2": "keep"})
    assert review.staged_digest() != staged

    decide(client, key, {"a1": "keep", "a2": "cull"})
    assert review.staged_digest() == staged


# --- the gate ----------------------------------------------------------------------


def test_write_back_is_unavailable_until_the_final_check_is_confirmed(client, review):
    key = keys(review)[1]
    decide(client, key, {"a1": "keep", "a2": "cull"})
    assert review.writeback_ready() is False

    assert confirm(client, review.staged_digest()).status_code == 200
    assert review.writeback_ready() is True


def test_confirming_a_digest_that_is_no_longer_current_is_refused(client, review):
    """The tab has been open for an hour; a decision landed since it was rendered.
    Confirming what is *now* staged rather than what was on screen would authorise a set
    the owner never saw, which is the whole failure this gate exists to prevent."""
    key = keys(review)[1]
    decide(client, key, {"a1": "keep", "a2": "cull"})
    stale = review.staged_digest()

    decide(client, keys(review)[0], {f"c{i}": "cull" for i in range(1, 6)})
    response = confirm(client, stale)

    assert response.status_code == 409
    assert "changed" in response.json()["detail"].lower()
    assert review.writeback_ready() is False


def test_a_confirmation_lapses_when_the_staged_set_grows_under_it(client, review):
    """The reason the confirmation is a digest and not a flag."""
    key = keys(review)[1]
    decide(client, key, {"a1": "keep", "a2": "cull"})
    confirm(client, review.staged_digest())
    assert review.writeback_ready() is True

    decide(client, keys(review)[2], {"p1": "keep", "p2": "cull"})
    assert review.writeback_ready() is False


def test_a_confirmation_stands_again_when_an_undo_restores_the_set_it_named(
    client, review
):
    key = keys(review)[1]
    decide(client, key, {"a1": "keep", "a2": "cull"})
    confirm(client, review.staged_digest())

    decide(client, keys(review)[2], {"p1": "keep", "p2": "cull"})
    assert review.writeback_ready() is False

    client.post("/api/undo", json={"cluster_key": keys(review)[2]}, headers=WRITE_HEADERS)
    assert review.writeback_ready() is True


def test_the_final_check_reports_whether_it_has_been_confirmed(client, review):
    key = keys(review)[1]
    decide(client, key, {"a1": "keep", "a2": "cull"})

    body = client.get("/api/final-check").json()
    assert body["confirmed"] is False
    assert body["staged_digest"] == review.staged_digest()

    confirm(client, body["staged_digest"])
    assert client.get("/api/final-check").json()["confirmed"] is True


def test_confirming_reads_the_digest_from_the_body_and_never_from_the_session(
    client, review
):
    """A confirm route that ignored its body and stamped `staged_digest()` would pass
    every test above. It is refused rather than treated as "confirm whatever is
    current"."""
    decide(client, keys(review)[1], {"a1": "keep", "a2": "cull"})
    assert confirm(client, "not-the-current-digest").status_code == 409
    assert review.writeback_ready() is False


# --- paging ------------------------------------------------------------------------


def test_the_final_check_pages_and_still_reports_the_whole_total(client, review):
    """`total` is the size of the staged set, never the size of the page — the sheet's
    heading is the one number the owner reads before confirming."""
    decide(client, keys(review)[0], {f"c{i}": "cull" for i in range(1, 6)})

    first = client.get("/api/final-check?offset=0&limit=2").json()
    assert first["total"] == 5
    assert first["offset"] == 0
    assert [row["photo"]["uuid"] for row in first["staged"]] == ["c1", "c2"]

    second = client.get("/api/final-check?offset=2&limit=2").json()
    assert second["total"] == 5
    assert second["offset"] == 2
    assert [row["photo"]["uuid"] for row in second["staged"]] == ["c3", "c4"]

    tail = client.get("/api/final-check?offset=4&limit=2").json()
    assert [row["photo"]["uuid"] for row in tail["staged"]] == ["c5"]
    assert client.get("/api/final-check?offset=99&limit=2").json()["staged"] == []


def test_the_route_pages_by_default_while_the_python_call_does_not(client, review):
    """Task 14 reads `final_check()` in-process to build the write-back plan and needs
    the whole set; the browser needs a page. Defaulting the HTTP route to everything is
    what would make the sheet unusable on the real library, and defaulting the method to
    a page is what would make a write-back silently cover the first 200 photos."""
    assert client.get("/api/final-check").json()["limit"] == FINAL_CHECK_PAGE
    assert review.final_check()["limit"] is None


@pytest.mark.parametrize("query", ["offset=-1", "limit=0", "limit=-3", "limit=5000"])
def test_a_nonsense_page_is_refused_rather_than_clamped(client, query):
    """Clamping would answer a question nobody asked. This is the same loud-validation
    rule `config-validation-covers-every-field` applies to settings."""
    assert client.get(f"/api/final-check?{query}").status_code == 422
