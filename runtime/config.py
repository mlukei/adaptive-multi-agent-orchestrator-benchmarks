"""Benchmark yaml config loading."""

from __future__ import annotations

import os
from typing import Any

import yaml
from dotenv import find_dotenv, load_dotenv


def load_experiment_env() -> None:
    """Load the nearest .env file without overriding process variables."""
    dotenv_path = find_dotenv(usecwd=True)
    load_dotenv(dotenv_path or None, override=False)


def optional_config_value(value: Any, default: Any = None) -> Any:
    return default if value is None else value


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
                setattr(self, key, value)


def load_config(
    config_file: str | None = None,
) -> DictToObject:
    """Load a YAML config as an object with dot-notation access."""
    config_file = os.getenv("BENCHMARK_CONFIG", config_file)
    if not config_file:
        raise ValueError("A config path is required.")

    with open(config_file) as f:
        data = yaml.safe_load(f)

    return DictToObject(data)
