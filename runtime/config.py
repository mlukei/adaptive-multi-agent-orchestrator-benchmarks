"""Benchmark config loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def is_env_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("${") and value.endswith("}")


def resolve_env_placeholder(value: Any) -> Any:
    if not is_env_placeholder(value):
        return value
    env_var = str(value)[2:-1]
    return os.getenv(env_var, value)


def optional_config_value(value: Any, default: Any = None) -> Any:
    if value is None or is_env_placeholder(value):
        return default
    return value


class DictToObject:
    """Converts nested dict to object with dot notation access."""

    def __init__(self, data: dict[str, object]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            elif isinstance(value, list):
                setattr(self, key, [
                    DictToObject(item) if isinstance(item, dict) else item
                    for item in value
                ])
            else:
                setattr(self, key, resolve_env_placeholder(value))


def load_config(
    config_file: str | None = None,
) -> DictToObject:
    """Load a benchmark config from YAML.
    Args:
        config_file: Path to a config YAML file.

    Returns:
        Config object with dot notation access 
    """
    config_file = os.getenv("BENCHMARK_CONFIG", config_file)
    if not config_file:
        raise ValueError("A config path is required.")
    config_path = Path(config_file)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not config_path.is_file():
        raise IsADirectoryError(
            f"Config path must be a YAML file, got: {config_file}"
        )

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return DictToObject(data)
