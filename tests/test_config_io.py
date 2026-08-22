"""Config loading must fail loudly. A silently-ignored typo in a threshold would
corrupt a calibration run with no visible symptom -- see the Task 7b preamble."""

import re
import tomllib
from dataclasses import fields
from pathlib import Path

import pytest

from photocull.config import ClusterConfig
from photocull.config_io import (
    _FLAGS,
    _RANGES,
    ConfigError,
    describe_config,
    find_config,
    load_config,
)

EXAMPLE = Path(__file__).resolve().parent.parent / "photocull.example.toml"


@pytest.fixture(autouse=True)
def isolated_from_the_real_config(tmp_path, monkeypatch):
    """No test may read the developer's own photocull.toml. This task exists to tell
    users to create one, so without this every bare `load_config()` test would start
    reading real calibrated values the moment the owner follows its own advice."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))


def write(tmp_path, text):
    p = tmp_path / "photocull.toml"
    p.write_text(text)
    return p


def test_no_config_file_anywhere_gives_shipped_defaults():
    assert load_config() == ClusterConfig()


def test_file_overrides_shipped_defaults(tmp_path):
    p = write(tmp_path, "similarity_threshold = 0.25\n")
    assert load_config(p).similarity_threshold == 0.25


def test_untouched_keys_keep_their_defaults(tmp_path):
    p = write(tmp_path, "similarity_threshold = 0.25\n")
    cfg = load_config(p)
    assert cfg.gap_seconds == ClusterConfig().gap_seconds
    assert cfg.ambiguity_margin == ClusterConfig().ambiguity_margin


def test_cli_overrides_beat_the_file(tmp_path):
    p = write(tmp_path, "similarity_threshold = 0.25\n")
    cfg = load_config(p, overrides={"similarity_threshold": 0.31})
    assert cfg.similarity_threshold == 0.31


def test_none_overrides_are_ignored_not_applied(tmp_path):
    """argparse hands us None for a flag the user did not pass."""
    p = write(tmp_path, "similarity_threshold = 0.25\n")
    assert load_config(p, overrides={"similarity_threshold": None}).similarity_threshold == 0.25


def test_unknown_key_raises_rather_than_being_ignored(tmp_path):
    p = write(tmp_path, "similarity_treshold = 0.25\n")  # deliberate typo
    with pytest.raises(ConfigError, match="similarity_treshold"):
        load_config(p)


def test_wrong_type_raises(tmp_path):
    p = write(tmp_path, 'similarity_threshold = "tight"\n')
    with pytest.raises(ConfigError, match="must be a number"):
        load_config(p)


def test_true_is_not_a_number(tmp_path):
    """bool is an int subclass, so an unguarded isinstance check reads `true` as 1.0
    -- a similarity_threshold of 1.0 would cluster wildly unrelated photos together."""
    p = write(tmp_path, "similarity_threshold = true\n")
    with pytest.raises(ConfigError, match="must be a number"):
        load_config(p)


def test_out_of_range_value_raises(tmp_path):
    p = write(tmp_path, "similarity_threshold = 99.0\n")
    with pytest.raises(ConfigError, match="between"):
        load_config(p)


def test_malformed_toml_raises(tmp_path):
    p = write(tmp_path, "similarity_threshold = = 0.4\n")
    with pytest.raises(ConfigError, match="malformed"):
        load_config(p)


def test_explicitly_named_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_no_config_field_is_silently_ignored(tmp_path):
    """load_config copies values out of its own validation table, not out of the
    dataclass, so a field added to ClusterConfig later would pass the unknown-key
    check as a KNOWN key and then be dropped on the floor -- the user sets it, gets
    no error, and nothing changes. Fail here instead, at the moment it is added."""
    defaults = ClusterConfig()
    for f in fields(ClusterConfig):
        if f.name == "weights":
            continue
        current = getattr(defaults, f.name)
        if isinstance(current, bool):
            # Booleans have no range to probe; flipping the default is the equivalent
            # round-trip. Same lesson as `config-probes-must-respect-their-own-range`:
            # the probe encoded "every field is numeric", which `include_shared` is the
            # first to falsify. Fix the probe, never the field.
            assert f.name in _FLAGS, (
                f"{f.name} is a bool but has no _FLAGS entry, so load_config would "
                "accept it as a known key and then drop it"
            )
            probe = not current
            p = write(tmp_path, f"{f.name} = {str(probe).lower()}\n")
            assert getattr(load_config(p), f.name) is probe, (
                f"{f.name} was accepted from the config file and then ignored"
            )
            continue
        assert isinstance(current, (int, float)), (
            f"{f.name} is neither numeric nor boolean; load_config only knows how to "
            "validate numbers, flags and the weights table -- extend config_io first"
        )
        assert f.name in _RANGES, (
            f"{f.name} has no _RANGES entry, so load_config cannot accept it at all"
        )
        # The probe must differ from the default AND be inside the field's own valid
        # range. An earlier version simply halved the default, which assumed half of
        # every default is legal -- false as soon as a field's default sits at the
        # bottom of its range, which `sharpness_size_tolerance = 1.0` is the first to
        # do (a max/min ratio is never below 1.0, so 0.5 is rightly rejected). Using
        # the range midpoint keeps the assertion identical without encoding that
        # assumption.
        low, high = _RANGES[f.name]
        probe = (low + high) / 2
        if probe == current:
            probe = (current + high) / 2
        assert probe != current and low <= probe <= high
        p = write(tmp_path, f"{f.name} = {probe}\n")
        assert getattr(load_config(p), f.name) == probe, (
            f"{f.name} was accepted from the config file and then ignored"
        )


def test_partial_weights_merge_over_defaults(tmp_path):
    d = ClusterConfig().weights
    shifted = round(d["sharpness"] + 0.05, 10)
    faces = round(d["faces"] - 0.05, 10)
    p = write(tmp_path, f"[weights]\nsharpness = {shifted}\nfaces = {faces}\n")
    cfg = load_config(p)
    assert cfg.weights["sharpness"] == shifted
    assert cfg.weights["exposure"] == d["exposure"]  # untouched key survives


def test_unknown_weight_raises(tmp_path):
    p = write(tmp_path, "[weights]\nsharpnes = 0.3\n")
    with pytest.raises(ConfigError, match="unknown weight"):
        load_config(p)


def test_weights_that_do_not_sum_to_one_raise(tmp_path):
    """Weights that don't sum to 1 silently rescale every score, making totals
    incomparable between a calibration run and a real run."""
    p = write(tmp_path, "[weights]\nsharpness = 0.9\n")
    with pytest.raises(ConfigError, match="sum to 1"):
        load_config(p)


def test_negative_weight_raises(tmp_path):
    """A negative weight makes a criterion count against the photo that scores best
    on it, which no user ever means and which no sum check would catch on its own."""
    p = write(tmp_path, "[weights]\nsharpness = -0.65\nfaces = 1.35\n")
    with pytest.raises(ConfigError, match="negative"):
        load_config(p)


def test_weights_override_is_validated_like_the_file():
    """CLI overrides reach the same field the file does. Validating only the file
    would leave the override path free to install a table nothing ever checked."""
    with pytest.raises(ConfigError, match="unknown weight"):
        load_config(overrides={"weights": {"sharpnes": 1.0}})


def test_weights_override_must_still_sum_to_one():
    with pytest.raises(ConfigError, match="sum to 1"):
        load_config(overrides={"weights": {"sharpness": 0.9}})


def test_weights_override_merges_over_defaults():
    d = ClusterConfig().weights
    cfg = load_config(
        overrides={"weights": {"sharpness": d["sharpness"] + 0.05, "faces": d["faces"] - 0.05}}
    )
    assert cfg.weights["sharpness"] == d["sharpness"] + 0.05
    assert cfg.weights["exposure"] == d["exposure"]


def test_weights_override_merges_over_the_file_not_the_defaults(tmp_path):
    """Precedence is per key. An override adjusting one weight must not silently
    revert the others back to shipped defaults -- that would mean a --weight flag
    quietly discarded the calibrated table it was meant to tweak."""
    d = ClusterConfig().weights
    p = write(tmp_path, f"[weights]\nhorizon = {d['horizon'] + 0.05}\nframing = {d['framing'] - 0.05}\n")
    cfg = load_config(p, overrides={"weights": {"sharpness": d["sharpness"] + 0.05,
                                                "faces": d["faces"] - 0.05}})
    assert cfg.weights["horizon"] == d["horizon"] + 0.05  # the file's change survives
    assert cfg.weights["sharpness"] == d["sharpness"] + 0.05  # and the override applies


def test_unknown_override_key_raises():
    with pytest.raises(ConfigError, match="unknown override"):
        load_config(overrides={"similarity_treshold": 0.25})


def test_find_config_prefers_cwd(tmp_path):
    (tmp_path / "photocull.toml").write_text("")
    assert find_config() == Path("photocull.toml")


def test_find_config_returns_none_when_there_is_none():
    assert find_config() is None


def test_describe_config_makes_output_self_describing(tmp_path):
    """Any JSON this project writes must record the thresholds that produced it,
    so a shared result can be interpreted without the config that made it."""
    p = write(tmp_path, "similarity_threshold = 0.25\n")
    meta = describe_config(load_config(p), source=p)
    assert meta["similarity_threshold"] == 0.25
    assert meta["source"] == str(p)
    assert meta["weights"]["sharpness"] == ClusterConfig().weights["sharpness"]


def test_describe_config_says_defaults_when_no_file():
    assert describe_config(ClusterConfig())["source"] == "built-in defaults"


def test_describe_config_reports_every_field():
    meta = describe_config(ClusterConfig())
    for f in fields(ClusterConfig):
        assert f.name in meta, f"{f.name} missing from output provenance"


def test_describe_config_does_not_alias_the_config_weights():
    """ClusterConfig is frozen, but its weights dict is not. Handing callers the live
    dict lets an output writer mutate the config that produced the output."""
    cfg = ClusterConfig()
    describe_config(cfg)["weights"]["sharpness"] = 999.0
    assert cfg.weights["sharpness"] != 999.0


def test_example_file_is_exactly_the_shipped_defaults(tmp_path):
    """The example file is the only place a user reads the defaults from, and nothing
    checks prose. The version written into the Task 7b plan had drifted from config.py
    by a whole weight table -- and because it still summed to 1.0 it would have loaded
    cleanly while re-enabling `facing`, which DECISIONS.md `vision-pose-angles-are-
    quantized` says must stay at 0.0. Uncommenting the example must reproduce the
    shipped defaults exactly; prose lines are `##` so one strip leaves valid TOML."""
    uncommented = re.sub(r"(?m)^#\s?", "", EXAMPLE.read_text())
    tomllib.loads(uncommented)  # fails loudly if the file is not strip-to-valid-TOML
    p = tmp_path / "from-example.toml"
    p.write_text(uncommented)
    assert load_config(p) == ClusterConfig()


def test_sharpness_size_tolerance_loads_from_file(tmp_path):
    (tmp_path / "photocull.toml").write_text("sharpness_size_tolerance = 1.05\n")
    assert load_config().sharpness_size_tolerance == 1.05


def test_sharpness_size_tolerance_below_one_is_rejected(tmp_path):
    (tmp_path / "photocull.toml").write_text("sharpness_size_tolerance = 0.5\n")
    with pytest.raises(ConfigError, match="between 1.0 and 4.0"):
        load_config()


# --- boolean flags (added for `include_shared`) ---------------------------------


def test_boolean_flag_loads_from_the_file(tmp_path):
    path = tmp_path / "photocull.toml"
    path.write_text("include_shared = true\n")
    assert load_config(path).include_shared is True


def test_boolean_flag_defaults_to_excluding_shared():
    """Shared-album assets are 6.2% of the real library but 27.1% of the photos this
    app stages for culling -- a 4.4x over-representation, because shared albums are
    where duplicate takes pile up. Photos.app cannot act on them via AppleScript
    either, so reviewing them is labour that produces nothing."""
    assert ClusterConfig().include_shared is False


def test_boolean_flag_rejects_a_non_boolean(tmp_path):
    path = tmp_path / "photocull.toml"
    path.write_text("include_shared = 1\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_boolean_flag_can_be_overridden_from_the_cli(tmp_path):
    path = tmp_path / "photocull.toml"
    path.write_text("include_shared = false\n")
    assert load_config(path, {"include_shared": True}).include_shared is True
