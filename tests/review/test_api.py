"""Task 8: the review API — the surface where the owner's judgement is recorded.

Four properties here are load-bearing, and each is written against a *plausible wrong*
implementation rather than the happy path:

1. **A decision snapshots its evidence from the session, never from the request body.**
   Every review decision is training data (`review-decisions-train-the-scorer`). If the
   client can post its own `total`/`sub_scores`/`proposed`, it can write the training set
   for the scorer that will later be tuned against it — and the resulting log would look
   entirely ordinary. The body carries verdicts and nothing else.
2. **A cluster may be decided with no keeper at all.** This file used to pin the opposite
   and the owner overruled it (`a-whole-cluster-may-be-culled`), so what is pinned now is
   that the decision lands *and* that both overview surfaces read it — they were written
   against a world where every staged photo had a keeper beside it. Deferring the whole
   cluster (`unset` throughout) is unchanged and still stages nothing.
3. **Final-check reads decisions, never proposals.** A final check built from
   `proposed_marks` would present the scorer's opinion as the owner's, on the one screen
   whose whole job is to confirm what the owner actually chose
   (`staged-set-gets-dashboard-and-final-check`).
4. **Keeping everything stages nothing.** `keep 5 of 5` must not fall through to staging
   the four non-winners — the counting has to come from the marks, not from
   `size - keepers` or `size - 1`.

`TestClient`'s default `base_url` is `http://testserver`, which the host gate refuses with
421, so every client here is built on a loopback base URL — which makes the Task 6 gate
provably live on every test in this file.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from photocull.config import ClusterConfig
from photocull_review import server
from photocull_review.api import ReviewSession
from photocull_review.decisions import DecisionLog

from ..fixtures import make_cluster

TOKEN = "n5Qk_test-token-with-plenty-of-entropy_7Xa"
BASE_URL = "http://127.0.0.1:54321"

#: What a browser sends on a same-origin `fetch`. Every mutating route needs it — the
#: Task 6 gate refuses anything else with 403, and that applies to these routes too.
WRITE_HEADERS = {"Sec-Fetch-Site": "same-origin"}


def sized(uuids, **kwargs):
    """A cluster whose members all have a display raster, so nothing is unshowable."""
    return make_cluster(uuids, **kwargs)


@pytest.fixture
def clusters():
    """Three clusters covering the shapes the API has to tell apart.

    `confident` has a clear winner, `close` is ambiguous, `pair` is the two-photo case
    that is the library's median cluster."""
    return [
        sized(["c1", "c2", "c3", "c4", "c5"], day=0),
        sized(["a1", "a2"], day=1, ambiguous=True),
        sized(["p1", "p2"], day=2),
    ]


@pytest.fixture
def log(tmp_path):
    with DecisionLog(tmp_path / "decisions.db") as opened:
        yield opened


@pytest.fixture
def review(clusters, log):
    return ReviewSession(
        clusters,
        log=log,
        session_id="test-session",
        config=ClusterConfig(),
        resolver=lambda record: _display(record),
    )


def _display(record):
    from photocull_review.session import DisplayInfo

    return DisplayInfo(uuid=record.uuid, local_path=record.display_path, long_side=1024)


@pytest.fixture
def client(review):
    app = server.build_app(token=TOKEN, review=review)
    opened = TestClient(app, base_url=BASE_URL)
    handoff = opened.get(f"/?t={TOKEN}")
    assert handoff.status_code == 200, "the launch handoff itself failed"
    return opened


def keys(review):
    """The three cluster keys, in capture order — the order the review walks them."""
    return [entry["cluster_key"] for entry in review.document["clusters"]]


def decide(client, key, marks, **body):
    return client.post(
        f"/api/clusters/{key}/decision",
        json={"marks": marks, **body},
        headers=WRITE_HEADERS,
    )


# --- the session document -------------------------------------------------------


def test_the_session_endpoint_serves_the_document_and_what_is_already_decided(client, review):
    response = client.get("/api/session")
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "test-session"
    assert len(body["clusters"]) == 3
    assert body["decided"] == {}, "nothing has been decided yet"

    first = keys(review)[0]
    decide(client, first, {u: "keep" for u in ["c1", "c2", "c3", "c4", "c5"]})
    assert set(client.get("/api/session").json()["decided"]) == {first}


def test_the_session_document_carries_no_filesystem_paths(client):
    """Task 3 strips them; this pins that the API does not put them back."""
    assert "/display/" not in client.get("/api/session").text
    assert "/deriv/" not in client.get("/api/session").text


def test_one_cluster_can_be_fetched_by_key(client, review):
    key = keys(review)[1]
    body = client.get(f"/api/clusters/{key}").json()
    assert body["cluster_key"] == key
    assert [member["uuid"] for member in body["members"]] == ["a1", "a2"]
    assert body["decision"] is None


def test_an_unknown_cluster_key_is_404_and_never_reaches_the_disk(client):
    """The key is a dictionary key, exactly as the image endpoint's uuid is.

    A traversal string is not sanitised — it simply misses the mapping. The
    slash-bearing spellings 404 at the router before the handler runs, and the rest
    miss the lookup; none of them may 500."""
    for candidate in ["nope", "..%2f..%2fetc%2fpasswd", "%00", "../../etc/passwd"]:
        response = client.get(f"/api/clusters/{candidate}")
        assert response.status_code == 404, candidate


# --- recording a decision -------------------------------------------------------


def test_keeping_everything_stages_nothing(client, review, log):
    """`keep 5 of 5` must not fall through to staging 4.

    The count has to come from the marks. Any implementation that derives staging from
    `size - 1` or `size - len(keepers) - 1` goes red here."""
    key = keys(review)[0]
    body = decide(client, key, {u: "keep" for u in ["c1", "c2", "c3", "c4", "c5"]}).json()

    assert body["staged"] == []
    assert sorted(body["kept"]) == ["c1", "c2", "c3", "c4", "c5"]
    assert log.state(key).marks_by_uuid() == {u: "keep" for u in ["c1", "c2", "c3", "c4", "c5"]}
    assert client.get("/api/final-check").json()["staged"] == []
    assert client.get("/api/dashboard").json()["photos"]["cull"] == 0


def test_a_whole_cluster_may_be_culled(client, review, log):
    """`a-whole-cluster-may-be-culled` — the owner's ruling, and the reverse of Task 8's.

    "None of these is worth keeping" is a first-class answer: a burst of five frames of
    nothing is exactly the backlog this app exists to cull, and it reaches the log like
    any other decision."""
    key = keys(review)[1]
    response = decide(client, key, {"a1": "cull", "a2": "cull"})

    assert response.status_code == 200
    body = response.json()
    assert sorted(body["staged"]) == ["a1", "a2"]
    assert body["kept"] == []
    assert log.state(key).marks_by_uuid() == {"a1": "cull", "a2": "cull"}


def test_a_confident_cluster_may_be_culled_entirely_too(client, review, log):
    """The ruling is not restricted to close calls, exactly as the refusal it replaces
    was not: `is_ambiguous` has nothing to do with whether a moment is worth keeping."""
    key = keys(review)[0]
    marks = {u: "cull" for u in ["c1", "c2", "c3", "c4", "c5"]}

    assert decide(client, key, marks).status_code == 200
    assert log.state(key).marks_by_uuid() == marks


def test_the_two_overviews_read_a_cluster_with_no_keeper(client, review):
    """The half of the ruling that is not a deletion: both surfaces have always assumed
    a staged photo has a keeper beside it, and neither had ever been handed a cluster
    without one.

    `keepers: []` is the shape Task 13's contact sheet must render — a staged photo with
    nothing to compare against, rather than a missing row."""
    key = keys(review)[1]
    decide(client, key, {"a1": "cull", "a2": "cull"})

    dashboard = client.get("/api/dashboard").json()
    assert dashboard["photos"] == {
        "in_clusters": 9,
        "keep": 0,
        "cull": 2,
        "unset": 0,
        "undecided": 7,
    }
    assert dashboard["largest_staged"][0] == {
        "cluster_key": key,
        "size": 2,
        "staged": 2,
        "kept": 0,
    }

    final = client.get("/api/final-check").json()
    assert final["total"] == 2
    assert [row["photo"]["uuid"] for row in final["staged"]] == ["a1", "a2"]
    assert [row["keepers"] for row in final["staged"]] == [[], []]


def test_deferring_a_whole_cluster_is_allowed_and_stages_nothing(client, review, log):
    """All-`unset` is a real answer, not an empty one: the owner looked and moved on.

    It stages nothing, so the no-keeper rule has nothing to protect against — refusing
    it would make "I am not sure yet" unrecordable."""
    key = keys(review)[1]
    body = decide(client, key, {"a1": "unset", "a2": "unset"}).json()

    assert body["staged"] == [] and body["kept"] == []
    assert log.state(key).marks_by_uuid() == {"a1": "unset", "a2": "unset"}


def test_a_uuid_that_is_not_in_the_cluster_is_refused(client, review, log):
    key = keys(review)[2]
    response = decide(client, key, {"p1": "keep", "p2": "cull", "c1": "cull"})

    assert response.status_code == 422
    assert "c1" in response.json()["detail"]
    assert log.state(key) is None


def test_a_missing_uuid_is_refused(client, review, log):
    """One verdict per photo, exactly — the same contract `DecisionLog._validate` holds.

    A partial submission would silently freeze the unmentioned takes at whatever the
    previous batch said, which is indistinguishable from the owner having answered."""
    key = keys(review)[2]
    response = decide(client, key, {"p1": "keep"})

    assert response.status_code == 422
    assert "p2" in response.json()["detail"]
    assert log.state(key) is None


def test_an_unknown_mark_is_refused(client, review, log):
    key = keys(review)[2]
    response = decide(client, key, {"p1": "keep", "p2": "maybe"})

    assert response.status_code == 422
    assert "maybe" in response.json()["detail"]
    assert log.state(key) is None


def test_a_decision_on_an_unknown_cluster_is_404(client):
    assert decide(client, "deadbeefdeadbeef", {"p1": "keep"}).status_code == 404


# --- the evidence is the session's, never the client's --------------------------


def test_a_decision_snapshots_sub_scores_from_the_session_not_the_request_body(
    client, review, log
):
    """The client must not be able to write its own training data.

    Every decision is logged with its sub-scores so the scorer can be tuned to this
    owner (`review-decisions-train-the-scorer`). Evidence that arrived from the browser
    would let a bug — or anything else holding the cookie — quietly author the training
    set, and the log would read as perfectly ordinary afterwards.

    **The owner overrides the scorer here**, and that is what makes the `proposed`
    assertions mean anything: `p1` is the proposed keeper and is being staged. A fixture
    in which the owner agreed would pass just as happily against an implementation that
    wrote the owner's own answer into the `proposed` column — which is exactly the column
    that exists to make an override legible under
    `review-decisions-train-the-scorer`. (Caught by the mutation pass, not by reading.)"""
    key = keys(review)[2]
    response = decide(
        client,
        key,
        {"p1": "cull", "p2": "keep"},
        sub_scores={"p1": {"sharpness": 99.0}, "p2": {"sharpness": -99.0}},
        totals={"p1": 99.0, "p2": -99.0},
        proposed={"p1": "cull", "p2": "keep"},
    )
    assert response.status_code == 200

    stored = {mark.photo_uuid: mark for mark in log.state(key).marks}
    assert stored["p1"].sub_scores == {"sharpness": 0.5, "exposure": 0.5}
    assert stored["p1"].total == 1.0
    assert stored["p2"].total == 0.8
    assert stored["p1"].mark == "cull" and stored["p1"].proposed == "keep"
    assert stored["p2"].mark == "keep" and stored["p2"].proposed == "cull"


def test_a_decision_records_the_photo_s_mod_date_from_the_scan(client, log, tmp_path):
    """Task 5 requeues a cluster whose photos changed underneath the decision, and it
    compares `mod_key` on both sides. A decision that stored no `mod_date` would make
    every edited photo look untouched."""
    edited = datetime(2024, 3, 1, 12, 0, 0)
    cluster = make_cluster(["m1", "m2"], mod_dates={"m1": edited})
    session = ReviewSession([cluster], log=log, session_id="s", resolver=_display)
    app = TestClient(server.build_app(token=TOKEN, review=session), base_url=BASE_URL)
    app.get(f"/?t={TOKEN}")

    key = session.document["clusters"][0]["cluster_key"]
    assert decide(app, key, {"m1": "keep", "m2": "cull"}).status_code == 200

    stored = {mark.photo_uuid: mark for mark in log.state(key).marks}
    assert stored["m1"].mod_date == edited
    assert stored["m2"].mod_date is None


def test_a_decision_records_the_session_s_settings_and_digest(client, review, log):
    """`reconcile` re-digests the stored snapshot and refuses if the two disagree, so
    both have to come from one place."""
    key = keys(review)[2]
    decide(client, key, {"p1": "keep", "p2": "cull"})

    stored = log.state(key)
    assert stored.config_digest == review.config_digest
    assert stored.settings == review.settings
    assert stored.session_id == "test-session"


# --- favourites -----------------------------------------------------------------


def test_a_favourite_mark_reaches_the_log(client, review, log):
    """`F` marks a photo as favourite *for write-back* rather than mutating the live
    library mid-session (SPEC, amended in Task 1)."""
    key = keys(review)[2]
    decide(client, key, {"p1": "keep", "p2": "cull"}, favorites=["p1"])

    stored = {mark.photo_uuid: mark for mark in log.state(key).marks}
    assert stored["p1"].favorite is True
    assert stored["p2"].favorite is False


def test_a_favourite_on_a_staged_photo_is_refused(client, review, log):
    """Favouriting a photo you are staging for culling is a contradiction with a real
    cost: Task 14 would set Favorite on a photo that is also in `Cull/Candidates`, and
    the owner's manual delete would then take a favourited photo with it."""
    key = keys(review)[2]
    response = decide(client, key, {"p1": "keep", "p2": "cull"}, favorites=["p2"])

    assert response.status_code == 422
    assert "p2" in response.json()["detail"]
    assert log.state(key) is None


def test_a_favourite_naming_a_photo_outside_the_cluster_is_refused(client, review):
    key = keys(review)[2]
    response = decide(client, key, {"p1": "keep", "p2": "cull"}, favorites=["c1"])
    assert response.status_code == 422


# --- undo -----------------------------------------------------------------------


def test_undo_restores_the_previous_state(client, review, log):
    """Superseding is an append, so undoing the newer batch has to expose the older one
    rather than leaving the cluster undecided."""
    key = keys(review)[2]
    decide(client, key, {"p1": "keep", "p2": "cull"})
    decide(client, key, {"p1": "cull", "p2": "keep"})
    assert log.state(key).marks_by_uuid() == {"p1": "cull", "p2": "keep"}

    body = client.post("/api/undo", json={"cluster_key": key}, headers=WRITE_HEADERS).json()
    assert body["undone"] is not None
    assert body["marks"] == {"p1": "keep", "p2": "cull"}
    assert log.state(key).marks_by_uuid() == {"p1": "keep", "p2": "cull"}

    client.post("/api/undo", json={"cluster_key": key}, headers=WRITE_HEADERS)
    assert log.state(key) is None


def test_undo_on_an_undecided_cluster_reports_that_nothing_was_undone(client, review):
    """Not an error: the owner pressing undo once too often is ordinary, and inventing
    an event for a decision that was never made would corrupt the audit trail."""
    key = keys(review)[2]
    body = client.post("/api/undo", json={"cluster_key": key}, headers=WRITE_HEADERS).json()
    assert body["undone"] is None
    assert body["marks"] == {}


def test_undo_on_an_unknown_cluster_is_404(client):
    response = client.post(
        "/api/undo", json={"cluster_key": "deadbeefdeadbeef"}, headers=WRITE_HEADERS
    )
    assert response.status_code == 404


# --- the two overview surfaces --------------------------------------------------


def test_dashboard_counts_match_the_log_after_a_decision_and_an_undo(client, review, log):
    key = keys(review)[0]
    before = client.get("/api/dashboard").json()
    assert before["clusters"] == {"total": 3, "decided": 0, "remaining": 3}
    assert before["photos"]["in_clusters"] == 9
    assert before["photos"]["undecided"] == 9

    decide(client, key, {"c1": "keep", "c2": "keep", "c3": "cull", "c4": "cull", "c5": "cull"})
    after = client.get("/api/dashboard").json()
    assert after["clusters"] == {"total": 3, "decided": 1, "remaining": 2}
    assert after["photos"]["keep"] == 2
    assert after["photos"]["cull"] == 3
    assert after["photos"]["undecided"] == 4
    assert after["largest_staged"][0] == {
        "cluster_key": key,
        "size": 5,
        "staged": 3,
        "kept": 2,
    }

    client.post("/api/undo", json={"cluster_key": key}, headers=WRITE_HEADERS)
    assert client.get("/api/dashboard").json() == before


def test_dashboard_staging_counts_only_photos_actually_marked_cull(client, review):
    """Staging is counted from the marks, never from `size - keepers`.

    A cluster decided with photos left `unset` is the case that separates the two, and
    it is not rare — deferring some takes while settling others is an ordinary outcome of
    `cull-selection-is-propose-and-adjust`. Counting the deferred ones as staged would
    overstate the cull on the one screen the owner checks before write-back."""
    key = keys(review)[0]
    decide(client, key, {"c1": "keep", "c2": "cull", "c3": "cull", "c4": "unset", "c5": "unset"})

    body = client.get("/api/dashboard").json()
    assert body["photos"] == {
        "in_clusters": 9,
        "keep": 1,
        "cull": 2,
        "unset": 2,
        "undecided": 4,
    }
    assert body["largest_staged"] == [
        {"cluster_key": key, "size": 5, "staged": 2, "kept": 1}
    ]


def test_largest_staged_names_the_biggest_cull_first(client, review):
    """The list exists to surface a systematic mistake before write-back, so the cluster
    that would stage the most has to be the one the owner sees first."""
    first, _second, third = keys(review)
    decide(client, third, {"p1": "keep", "p2": "cull"})
    decide(client, first, {"c1": "keep", "c2": "cull", "c3": "cull", "c4": "unset", "c5": "unset"})

    rows = client.get("/api/dashboard").json()["largest_staged"]
    assert [(row["cluster_key"], row["staged"]) for row in rows] == [(first, 2), (third, 1)]


def test_dashboard_points_at_the_first_undecided_cluster_in_capture_order(client, review):
    first, second, _third = keys(review)
    assert client.get("/api/dashboard").json()["next_undecided"] == {
        "index": 0,
        "cluster_key": first,
    }

    decide(client, first, {u: "keep" for u in ["c1", "c2", "c3", "c4", "c5"]})
    assert client.get("/api/dashboard").json()["next_undecided"] == {
        "index": 1,
        "cluster_key": second,
    }


def test_final_check_lists_exactly_the_photos_left_marked_cull_beside_their_keepers(
    client, review
):
    first, _second, third = keys(review)
    decide(client, first, {"c1": "keep", "c2": "keep", "c3": "cull", "c4": "cull", "c5": "unset"})
    decide(client, third, {"p1": "keep", "p2": "cull"})

    body = client.get("/api/final-check").json()
    assert body["total"] == 3
    assert [row["photo"]["uuid"] for row in body["staged"]] == ["c3", "c4", "p2"]
    assert [[k["uuid"] for k in row["keepers"]] for row in body["staged"]] == [
        ["c1", "c2"],
        ["c1", "c2"],
        ["p1"],
    ]
    assert all(row["cluster_key"] in {first, third} for row in body["staged"])
    assert "unset" not in {row["photo"]["uuid"] for row in body["staged"]}
    assert "c5" not in {row["photo"]["uuid"] for row in body["staged"]}


def test_final_check_reads_decisions_and_never_proposals(client, review):
    """Every cluster now opens with a suggestion (`every-cluster-opens-with-a-suggestion`),
    so a final check built from `proposed` would show 6 staged photos before the owner
    has decided anything at all — the scorer's opinion presented as theirs."""
    proposals = [entry["proposed"] for entry in review.document["clusters"]]
    assert sum(1 for marks in proposals for mark in marks.values() if mark == "cull") == 6

    body = client.get("/api/final-check").json()
    assert (body["total"], body["staged"]) == (0, [])


def test_final_check_carries_no_filesystem_paths(client, review):
    decide(client, keys(review)[2], {"p1": "keep", "p2": "cull"})
    assert "/display/" not in client.get("/api/final-check").text


# --- decisions made under other settings ----------------------------------------


def test_a_decision_recorded_under_other_settings_is_not_counted(client, review, log):
    """`reconcile` holds those back as `superseded` rather than applying them, and the
    two overview surfaces have to agree with it — otherwise the dashboard says a cluster
    is decided while resume re-asks it."""
    from photocull_review.decisions import Mark

    key = keys(review)[2]
    log.record(
        session_id="an-older-session",
        cluster_key=key,
        config_digest="0000000000000000",
        marks=[Mark(photo_uuid="p1", mark="keep"), Mark(photo_uuid="p2", mark="cull")],
    )

    assert client.get("/api/dashboard").json()["clusters"]["decided"] == 0
    assert client.get("/api/final-check").json()["total"] == 0
    assert client.get("/api/session").json()["decided"] == {}


# --- shutdown -------------------------------------------------------------------


def test_shutdown_asks_the_server_to_stop(review):
    stopped = []
    app = server.build_app(
        token=TOKEN, review=review, on_shutdown=lambda: stopped.append(True)
    )
    client = TestClient(app, base_url=BASE_URL)
    client.get(f"/?t={TOKEN}")

    response = client.post("/api/shutdown", headers=WRITE_HEADERS)
    assert response.status_code == 202
    assert stopped == [True]
    assert review.shutdown_requested is True


def test_shutdown_without_a_hook_still_records_the_request(client, review):
    assert client.post("/api/shutdown", headers=WRITE_HEADERS).status_code == 202
    assert review.shutdown_requested is True


# --- the gate still governs these routes ----------------------------------------


def test_the_api_routes_do_not_exist_without_a_session(client_without_review):
    """Stronger than a route that 404s on every key: there is nothing to reach."""
    for path in ["/api/session", "/api/dashboard", "/api/final-check"]:
        assert client_without_review.get(path).status_code == 404


@pytest.fixture
def client_without_review():
    opened = TestClient(server.build_app(token=TOKEN), base_url=BASE_URL)
    opened.get(f"/?t={TOKEN}")
    return opened


def test_a_decision_without_the_session_cookie_is_refused(review):
    client = TestClient(server.build_app(token=TOKEN, review=review), base_url=BASE_URL)
    response = client.post(
        "/api/clusters/whatever/decision", json={"marks": {}}, headers=WRITE_HEADERS
    )
    assert response.status_code == 401


def test_a_cross_origin_decision_is_refused(client, review):
    """The gate's `Sec-Fetch-Site` check covers these routes; `same-site` is not enough."""
    response = client.post(
        f"/api/clusters/{keys(review)[2]}/decision",
        json={"marks": {"p1": "keep", "p2": "cull"}},
        headers={"Sec-Fetch-Site": "same-site"},
    )
    assert response.status_code == 403


def test_every_api_response_carries_the_security_headers(client):
    response = client.get("/api/dashboard")
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
