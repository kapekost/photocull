"""Task 8b: the calibration sample the owner reviews at the Phase 1a Task 9 gate.

The sampling rule is the substance here, not the serialisation. `limit=200` against
1,496 real clusters means 87% go unreviewed, so *which* 200 decides what the owner can
possibly discover -- and the failure mode is silent, because any 200 clusters produce a
plausible-looking file."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from photocull.calibrate import (
    size_band,
    stratified_sample,
    write_calibration_sample,
    write_contact_sheet,
)
from photocull.config import ClusterConfig

from .fixtures import CLUSTER_BASE as BASE
from .fixtures import make_cluster


def population(n_pairs=100, n_big=6):
    """A population with the real one's shape: overwhelmingly pairs, a rare large tail."""
    clusters = []
    for i in range(n_pairs):
        clusters.append(
            make_cluster([f"p{i}a", f"p{i}b"], diameter=0.20 + (i % 20) * 0.01, day=i)
        )
    for j in range(n_big):
        size = 8 + j
        clusters.append(
            make_cluster(
                [f"b{j}_{k}" for k in range(size)],
                diameter=0.35 + j * 0.005,
                day=n_pairs + j,
            )
        )
    return clusters


# --- the sampling rule ---------------------------------------------------------


def test_size_band_boundaries():
    assert size_band(2) == "2"
    assert size_band(3) == "3-4"
    assert size_band(4) == "3-4"
    assert size_band(5) == "5-9"
    assert size_band(9) == "5-9"
    assert size_band(10) == "10+"
    assert size_band(20) == "10+"


def test_sample_takes_exactly_the_limit():
    assert len(stratified_sample(population(), limit=40)) == 40


def test_sample_returns_everything_when_population_is_smaller():
    pop = population(n_pairs=5, n_big=1)
    assert len(stratified_sample(pop, limit=200)) == len(pop)


def test_sample_never_repeats_a_cluster():
    sample = stratified_sample(population(), limit=40)
    uuids = [c.winner_uuid for c in sample]
    assert len(set(uuids)) == len(uuids)


def test_sample_is_deterministic():
    pop = population()
    first = [c.winner_uuid for c in stratified_sample(pop, limit=40)]
    second = [c.winner_uuid for c in stratified_sample(pop, limit=40)]
    assert first == second


def test_sample_covers_the_large_cluster_tail():
    """The whole point. A 20-photo cluster stages 19 photos on one grouping decision,
    so the rare tail carries most of the risk -- measured on the real library, clusters
    of 5+ are 4.1% of clusters but 18% of culled photos. Proportional sampling would
    hand the owner one or two of them."""
    pop = population(n_pairs=100, n_big=6)
    sample = stratified_sample(pop, limit=40)
    big_in_sample = [c for c in sample if len(c.records) >= 5]
    assert len(big_in_sample) == 6


def test_sample_is_not_merely_the_head_of_the_population():
    """`clusters[:200]` is the tempting implementation and it is measurably wrong: on
    the real library it spans 2019-2023 out of a 2019-2026 population and its largest
    cluster is 6 photos against a true max of 20."""
    pop = population()
    sample = stratified_sample(pop, limit=40)
    assert [c.winner_uuid for c in sample] != [c.winner_uuid for c in pop[:40]]
    # reaches well past the first `limit` clusters, i.e. into the later years
    assert max(pop.index(c) for c in sample) > 40


def test_sample_reaches_both_ends_of_a_stratum():
    """Plain `i * n // k` spacing never selects a stratum's last element. Measured on
    the real library that alone cut the sample short by four months -- it ended at
    2026-04-19 against a population running to 2026-08-08 -- while looking entirely
    reasonable, because the sample was still the right size and still spread out."""
    pop = [make_cluster([f"c{i}a", f"c{i}b"], diameter=0.30, day=i) for i in range(50)]
    sample = stratified_sample(pop, limit=10)
    assert pop[0] in sample
    assert pop[-1] in sample


def test_sample_comes_out_in_population_order():
    """Reviewed chronologically, not grouped by stratum -- the owner walks the sample
    in capture order like every other surface in this project."""
    pop = population()
    sample = stratified_sample(pop, limit=40)
    order = [pop.index(c) for c in sample]
    assert order == sorted(order)


def test_sample_covers_every_diameter_band():
    """Diameter is how close a grouping came to the cut height, and it is nearly
    independent of size on the real library (pairs spread 263/292/263/267 across the
    four diameter quartiles), so it has to be stratified separately. The guarantee is
    band coverage, not that the exact min and max happen to be drawn."""
    from photocull.calibrate import sample_to_dict

    doc = sample_to_dict(population(), limit=40)
    bands = {k.split("/")[1] for k in doc["strata"] if doc["strata"][k] > 0}
    assert bands == {"d1", "d2", "d3", "d4"}


def test_empty_population_samples_to_nothing():
    assert stratified_sample([], limit=200) == []


def test_clusters_with_no_diameter_are_still_sampled():
    """`diameter` is None only when a cluster has no measurable pair. It must not
    crash the banding, and it must not be silently dropped from the population."""
    pop = population(n_pairs=4, n_big=0)
    pop.append(make_cluster(["x1", "x2"], diameter=None, day=99))
    sample = stratified_sample(pop, limit=200)
    assert len(sample) == 5


# --- the export ----------------------------------------------------------------


def test_export_joins_members_on_uuid_not_index():
    """`rank_cluster` sorts `scores` by total while `records` keep bucket order, so the
    two lists are NOT index-aligned. Joining by index yields a file where every photo
    carries another photo's scores -- correct-looking, wrong throughout, and invisible
    without opening the photos. This test fails under an index join."""
    cluster = make_cluster(
        ["low", "high"],
        totals={"low": 0.1, "high": 0.9},
        sub_scores={"low": {"sharpness": 0.1}, "high": {"sharpness": 0.9}},
    )
    # records order is [low, high]; scores order is [high, low].
    assert [r.uuid for r in cluster.records] == ["low", "high"]
    assert [s.uuid for s in cluster.scores] == ["high", "low"]

    members = {m["uuid"]: m for m in _only_cluster(cluster)["members"]}
    assert members["low"]["sub_scores"]["sharpness"] == 0.1
    assert members["high"]["sub_scores"]["sharpness"] == 0.9
    assert members["low"]["total"] == 0.1
    assert members["high"]["total"] == 0.9


def _only_cluster(cluster, tmp=None, **kw):
    from photocull.calibrate import sample_to_dict

    return sample_to_dict([cluster], **kw)["clusters"][0]


def test_export_carries_the_four_self_assessment_fields(tmp_path):
    cluster = make_cluster(
        ["a", "b", "c"],
        diameter=0.37,
        median=0.21,
        ambiguous=True,
        sharpness_available=False,
        dropped=["horizon", "sharpness"],
    )
    path = tmp_path / "sample.json"
    write_calibration_sample([cluster], path)
    entry = json.loads(path.read_text())["clusters"][0]

    assert entry["diameter"] == 0.37
    assert entry["median_distance"] == 0.21
    assert entry["sharpness_available"] is False
    assert entry["dropped_criteria"] == ["horizon", "sharpness"]
    assert entry["is_ambiguous"] is True


def test_export_carries_distance_to_the_photo_being_kept():
    """The only reviewable number in the system: not "cluster diameter 0.40" but
    "this photo is 0.31 from the one you are keeping"."""
    cluster = make_cluster(["a", "b", "c"], totals={"a": 0.9, "b": 0.5, "c": 0.1})
    members = {m["uuid"]: m for m in _only_cluster(cluster)["members"]}

    assert members["a"]["is_winner"] is True
    assert members["a"]["distance_to_winner"] is None
    assert members["b"]["distance_to_winner"] == cluster.distances_to_winner["b"]
    assert members["c"]["distance_to_winner"] == cluster.distances_to_winner["c"]


def test_keeper_never_reports_a_distance_to_itself():
    """`_annotate` builds `distances_to_winner` excluding the winner, so on real data
    the keeper's entry is absent and reads as None either way. That makes the export's
    guard untestable from a well-formed fixture -- and an untested guard is one a later
    change deletes as redundant. Fed a cluster that *does* carry a self-distance, the
    export must still report the keeper as having none, so the contact sheet can never
    print "0.0000 from keeper" on the keeper's own tile."""
    cluster = make_cluster(["a", "b"], totals={"a": 0.9, "b": 0.2})
    cluster.distances_to_winner["a"] = 0.0

    members = {m["uuid"]: m for m in _only_cluster(cluster)["members"]}
    assert members["a"]["is_winner"] is True
    assert members["a"]["distance_to_winner"] is None


def test_export_carries_the_derivative_path_of_every_member():
    """Without it the owner cannot look at the photos, which is the entire activity
    Task 9 consists of."""
    cluster = make_cluster(["a", "b"])
    members = _only_cluster(cluster)["members"]
    assert [m["derivative_path"] for m in members] == ["/deriv/a.jpeg", "/deriv/b.jpeg"]


def test_export_embeds_the_config_that_produced_it():
    """`config-is-external-defaults-are-generic`: a sample shared or re-read later must
    be interpretable without the config file that made it."""
    from photocull.calibrate import sample_to_dict

    cfg = ClusterConfig(similarity_threshold=0.33)
    doc = sample_to_dict([make_cluster(["a", "b"])], config=cfg, source="photocull.toml")
    assert doc["config"]["similarity_threshold"] == 0.33
    assert doc["config"]["source"] == "photocull.toml"


def test_export_reports_the_population_it_sampled_from():
    """200 of 1,496 is a very different claim from 200 of 200, and the file is read
    long after the run that produced it."""
    pop = population(n_pairs=50, n_big=3)
    from photocull.calibrate import sample_to_dict

    doc = sample_to_dict(pop, limit=20)
    assert doc["population"]["clusters"] == 53
    assert doc["population"]["sampled"] == 20
    assert doc["population"]["culled_photos"] == sum(len(c.records) - 1 for c in pop)
    assert doc["population"]["culled_photos_sampled"] == sum(
        len(c.records) - 1 for c in stratified_sample(pop, limit=20)
    )


def test_export_records_the_strata_allocation():
    """So the sample is auditable: the owner can see the rule spent 30% of the sample
    on the 4% tail on purpose, rather than discovering it by counting."""
    from photocull.calibrate import sample_to_dict

    doc = sample_to_dict(population(), limit=40)
    assert sum(doc["strata"].values()) == 40
    assert any(k.startswith("5-9") or k.startswith("10+") for k in doc["strata"])


def test_export_round_trips_through_json(tmp_path):
    path = tmp_path / "sample.json"
    write_calibration_sample(population(n_pairs=10, n_big=2), path, limit=5)
    doc = json.loads(path.read_text())
    assert len(doc["clusters"]) == 5
    assert all("members" in c for c in doc["clusters"])


def test_export_respects_the_limit(tmp_path):
    path = tmp_path / "sample.json"
    doc = write_calibration_sample(population(), path, limit=7)
    assert len(doc["clusters"]) == 7


# --- the contact sheet ---------------------------------------------------------


def test_contact_sheet_shows_every_sampled_cluster(tmp_path):
    from photocull.calibrate import sample_to_dict

    doc = sample_to_dict(population(n_pairs=6, n_big=1), limit=4)
    path = tmp_path / "sample.html"
    write_contact_sheet(doc, path)
    html = path.read_text()
    assert html.count("<section") == 4


def test_contact_sheet_never_links_into_the_photos_library(tmp_path):
    """A `file://` URL into Photos Library.photoslibrary renders as a BROKEN IMAGE in
    every browser, because macOS TCC protects that bundle and browsers do not have
    Full Disk Access — while this process, which does, reads the same file happily. So
    the bug is invisible from Python and total for the reader. Images are copied out
    and referenced relatively instead."""
    from photocull.calibrate import sample_to_dict

    doc = sample_to_dict([make_cluster(["a", "b"])])
    path = tmp_path / "sample.html"
    write_contact_sheet(doc, path)
    html = path.read_text()

    assert "file://" not in html
    assert 'src="img/a.jpeg"' in html
    assert 'src="img/b.jpeg"' in html


def test_contact_sheet_copies_the_images_next_to_itself(tmp_path):
    """The copy is what makes the sheet render. It reads from the library and writes
    only into the report directory — the library is never touched."""
    from photocull.calibrate import sample_to_dict

    source = tmp_path / "src"
    source.mkdir()
    (source / "a.jpeg").write_bytes(b"\xff\xd8jpegbytes")
    cluster = make_cluster(["a", "b"])
    object.__setattr__(cluster.records[0], "derivative_path", str(source / "a.jpeg"))

    path = tmp_path / "out" / "sample.html"
    write_contact_sheet(sample_to_dict([cluster]), path)

    assert (path.parent / "img" / "a.jpeg").read_bytes() == b"\xff\xd8jpegbytes"


def test_contact_sheet_survives_a_derivative_photos_has_evicted(tmp_path):
    """Photos regenerates and evicts derivatives, so a path recorded on Monday can be
    gone on Wednesday. One missing file must not abort the whole sheet."""
    from photocull.calibrate import sample_to_dict

    doc = sample_to_dict([make_cluster(["a", "b"])])  # /deriv/*.jpeg do not exist
    path = tmp_path / "sample.html"
    write_contact_sheet(doc, path)
    assert path.read_text().count("<figure>") == 2


def test_contact_sheet_marks_the_keeper_and_the_distances(tmp_path):
    from photocull.calibrate import sample_to_dict

    cluster = make_cluster(["a", "b"], totals={"a": 0.9, "b": 0.2})
    doc = sample_to_dict([cluster])
    path = tmp_path / "sample.html"
    write_contact_sheet(doc, path)
    html = path.read_text()
    assert "KEEP" in html
    assert f"{cluster.distances_to_winner['b']:.3f}" in html
    # plain language beside the raw number -- the owner could not read the bare figure
    assert "nearly identical" in html or "very similar" in html


def test_contact_sheet_escapes_paths_rather_than_injecting_them(tmp_path):
    """Derivative paths come from the filesystem, not from a literal in this repo."""
    from photocull.calibrate import sample_to_dict

    cluster = make_cluster(["a", "b"])
    object.__setattr__(cluster.records[0], "derivative_path", '/x/<script>"&.jpeg')
    doc = sample_to_dict([cluster])
    path = tmp_path / "sample.html"
    write_contact_sheet(doc, path)
    html = path.read_text()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html or "%3Cscript%3E" in html
