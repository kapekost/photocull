"""Load ClusterConfig from a TOML file plus CLI overrides.

The shipped defaults in config.py are deliberately GENERIC. A threshold calibrated
against one person's library belongs in that person's own photocull.toml, never in
this repo -- see DECISIONS.md `config-is-external-defaults-are-generic`.

Everything here fails loudly. An unknown key is an error, not a shrug: a mistyped
threshold that silently changes nothing would corrupt a calibration run without a
single visible symptom."""

from __future__ import annotations

import tomllib
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from .config import ClusterConfig


class ConfigError(ValueError):
    """A config was supplied but cannot be trusted. Never silently ignored."""


# Sanity bounds, not taste. similarity_threshold's real working range is well under
# 1.5 (unrelated photos measured 1.0398), so 4.0 only catches nonsense.
_RANGES: dict[str, tuple[float, float]] = {
    "gap_seconds": (0.0, 86_400.0),
    "similarity_threshold": (0.0, 4.0),
    "ambiguity_margin": (0.0, 1.0),
    "gps_reinforce_metres": (0.0, 100_000.0),
    # Below 1.0 is not "stricter", it is unsatisfiable: the ratio is max/min over the
    # same list, so it can never be under 1.0 and every cluster would lose sharpness.
    "sharpness_size_tolerance": (1.0, 4.0),
}

# Every ClusterConfig field this module knows how to validate. Kept separate from the
# dataclass's own field list on purpose: a field added to ClusterConfig without a rule
# here would otherwise pass the unknown-key check and then be dropped on the floor.
#: Boolean fields. Separate from _RANGES because `_number` deliberately rejects bools
#: (`threshold = true` must not read as 1.0), so flags need their own strict check.
_FLAGS: frozenset[str] = frozenset({"include_shared"})

_HANDLED: frozenset[str] = frozenset(_RANGES) | _FLAGS | {"weights"}


def _flag(name: str, value: Any) -> bool:
    # Strict: `include_shared = 1` is a typo, not a truthy value. Silently coercing it
    # is how a config that reads as working ends up changing nothing.
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false, got {type(value).__name__}: {value!r}")
    return value


def _search_path() -> tuple[Path, ...]:
    # Resolved per call, not at import, so tests can move HOME and cwd.
    return (Path("photocull.toml"), Path.home() / ".config" / "photocull" / "config.toml")


def _number(name: str, value: Any) -> float:
    # bool is an int subclass; `threshold = true` must not read as 1.0.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number, got {type(value).__name__}: {value!r}")
    low, high = _RANGES[name]
    if not low <= value <= high:
        raise ConfigError(f"{name} must be between {low} and {high}, got {value}")
    return float(value)


def _weights(raw: Any, base: dict[str, float], known: dict[str, float]) -> dict[str, float]:
    """Merge a partial weight table over `base`, validating against `known`.

    `base` is what the lower-precedence layer produced, so a CLI override adjusts the
    file's table rather than silently reverting every key the file had set."""
    if not isinstance(raw, dict):
        raise ConfigError(f"weights must be a table, got {type(raw).__name__}")
    unknown = set(raw) - set(known)
    if unknown:
        raise ConfigError(f"unknown weight(s) {sorted(unknown)}; known weights are {sorted(known)}")
    merged = dict(base)
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"weights.{key} must be a number, got {value!r}")
        if value < 0:
            raise ConfigError(f"weights.{key} must not be negative, got {value}")
        merged[key] = float(value)
    total = sum(merged.values())
    if abs(total - 1.0) > 1e-6:
        raise ConfigError(
            f"weights must sum to 1.0, got {total:.6f}. Overrides merge over the "
            f"defaults, so lowering one weight means raising another. Merged: {merged}"
        )
    return merged


def _reject_unhandled(names: set[str], known: set[str], where: str, noun: str = "key") -> None:
    unknown = names - known
    if unknown:
        raise ConfigError(
            f"unknown {noun}(s) {sorted(unknown)} in {where}; known keys are {sorted(known)}"
        )
    unhandled = names - _HANDLED
    if unhandled:
        raise ConfigError(
            f"{sorted(unhandled)} is a ClusterConfig field but config_io has no validation "
            f"rule for it, so setting it in {where} would do nothing. Add one to _RANGES."
        )


def find_config(path: str | Path | None = None) -> Path | None:
    """Resolve which config file applies. An explicitly-named file must exist."""
    if path is not None:
        candidate = Path(path)
        if not candidate.is_file():
            raise ConfigError(f"config file not found: {candidate}")
        return candidate
    for candidate in _search_path():
        if candidate.is_file():
            return candidate
    return None


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ClusterConfig:
    """Shipped defaults <- config file <- overrides, in that order of precedence.
    `overrides` values of None are ignored, so argparse Namespaces can be passed
    straight through without the caller filtering unset flags."""
    defaults = ClusterConfig()
    source = find_config(path)

    data: dict[str, Any] = {}
    if source is not None:
        try:
            with open(source, "rb") as handle:
                data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"malformed TOML in {source}: {exc}") from exc
        except OSError as exc:
            raise ConfigError(f"cannot read config {source}: {exc}") from exc

    known = {f.name for f in fields(ClusterConfig)}
    _reject_unhandled(set(data), known, str(source))

    values: dict[str, Any] = {}
    for name in _RANGES:
        if name in data:
            values[name] = _number(name, data[name])
    for name in _FLAGS:
        if name in data:
            values[name] = _flag(name, data[name])
    if "weights" in data:
        values["weights"] = _weights(data["weights"], defaults.weights, defaults.weights)

    supplied = {name: value for name, value in (overrides or {}).items() if value is not None}
    _reject_unhandled(set(supplied), known, "overrides", noun="override")
    for name, value in supplied.items():
        if name == "weights":
            # Merge over what the file produced, so an override adjusts one weight
            # instead of quietly discarding every weight the file had set.
            values["weights"] = _weights(value, values.get("weights", defaults.weights), defaults.weights)
        elif name in _FLAGS:
            values[name] = _flag(name, value)
        else:
            values[name] = _number(name, value)

    return replace(defaults, **values)


def describe_config(cfg: ClusterConfig, source: str | Path | None = None) -> dict[str, Any]:
    """Provenance for output files: what settings produced this result, and where
    they came from. Every JSON this project writes should embed this, so a result
    shared with someone else can be interpreted without the config that made it."""
    described: dict[str, Any] = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    # Copy, don't alias: ClusterConfig is frozen but its weights dict is not, so
    # handing out the live dict lets an output writer mutate the config.
    described["weights"] = dict(cfg.weights)
    described["source"] = str(source) if source is not None else "built-in defaults"
    return described
