"""Utilities for experiment configuration management and hyperparameter sweeps."""
from __future__ import annotations

import copy
import itertools
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


class ConfigError(RuntimeError):
    """Raised when configuration files or overrides are invalid."""


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a configuration file in JSON or YAML format."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file '{path}' does not exist.")

    suffix = path.suffix.lower()
    data = path.read_text()

    if suffix in {".json"}:
        return json.loads(data)

    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ConfigError("PyYAML is required to load YAML configs but is not installed.")
        return yaml.safe_load(data)

    raise ConfigError(f"Unsupported config extension '{path.suffix}'. Use .json, .yaml, or .yml.")


def dump_config(config: Mapping[str, Any], path: str | Path) -> None:
    """Persist configuration to disk (infers JSON vs YAML from suffix)."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        serialized = json.dumps(config, indent=2, sort_keys=True)
    elif suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ConfigError("PyYAML is required to write YAML configs but is not installed.")
        serialized = yaml.safe_dump(dict(config), sort_keys=False)
    else:
        raise ConfigError(f"Unsupported config extension '{path.suffix}'. Use .json, .yaml, or .yml.")

    path.write_text(serialized + "\n")


def _ensure_mapping(config: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("Configuration must be a mapping.")
    return dict(config)


def _assign_path(config: Dict[str, Any], path: Sequence[str], value: Any) -> None:
    cursor = config
    for key in path[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[path[-1]] = value


def apply_overrides(
    base_config: Mapping[str, Any],
    overrides: Mapping[str, Any] | Iterable[Tuple[str, Any]],
    *,
    delimiter: str = ".",
) -> Dict[str, Any]:
    """Return a new config with dotted-path overrides applied."""
    result = copy.deepcopy(_ensure_mapping(base_config))

    if isinstance(overrides, Mapping):
        items = overrides.items()
    else:
        items = list(overrides)

    for dotted_key, value in items:
        if not isinstance(dotted_key, str):
            raise TypeError("Override keys must be strings.")
        path = [segment for segment in dotted_key.split(delimiter) if segment]
        if not path:
            raise ConfigError("Override keys cannot be empty.")
        _assign_path(result, path, value)

    return result


def grid_sweep(
    base_config: Mapping[str, Any],
    sweep: Mapping[str, Sequence[Any]],
    *,
    delimiter: str = ".",
) -> Iterator[Dict[str, Any]]:
    """Yield configurations for a Cartesian product sweep."""
    if not sweep:
        yield copy.deepcopy(_ensure_mapping(base_config))
        return

    keys = []
    values = []
    for key, candidates in sweep.items():
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise TypeError(f"Sweep values for '{key}' must be a sequence of candidates.")
        if len(candidates) == 0:
            raise ConfigError(f"Sweep list for '{key}' cannot be empty.")
        keys.append(key)
        values.append(list(candidates))

    for combo in itertools.product(*values):
        overrides = dict(zip(keys, combo))
        yield apply_overrides(base_config, overrides, delimiter=delimiter)


def merge_run_config(
    base_config: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Deep copy base config and update with extra top-level keys."""
    merged = copy.deepcopy(_ensure_mapping(base_config))
    if not extra:
        return merged
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_run_config(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged

