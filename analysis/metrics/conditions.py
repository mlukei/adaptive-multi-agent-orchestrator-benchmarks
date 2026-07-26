from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from loader import (
    REAL_AGENTS_OFFICEBENCH,
    REAL_GAIA_AGENTS,
    BenchmarkConfig,
    OUTPUT_DIR,
    ROOT,
    add_derived_metrics,
    load_runs,
)

_OB_DIR = OUTPUT_DIR / "officebench"
_OB_GOLD = ROOT / "annotations" / "officebench.json"

SYSTEM_ORDER = [
    "BL-Lower",
    "BL-Upper",
    "Blueprint",
    "Playbook",
    "Adaptive System",
]

CARD_PATHS: dict[str, dict[str, tuple[Path, ...]]] = {
    "rich": {
        "BL-Lower": tuple(
            _OB_DIR / "rich" / "baselines" / f"baseline_lower_fold_{fold}.csv"
            for fold in (1, 2, 3)
        ),
        "BL-Upper": tuple(
            _OB_DIR / "rich" / "baselines" / f"baseline_upper_fold_{fold}.csv"
            for fold in (1, 2, 3)
        ),
        "Blueprint": tuple(
            _OB_DIR / "rich" / "blueprints" / f"test_memory_ablation_blueprint_fold_{fold}.csv"
            for fold in (1, 2, 3)
        ),
        "Playbook": tuple(
            _OB_DIR / "rich" / "playbooks" / f"test_memory_ablation_playbook_fold_{fold}.csv"
            for fold in (1, 2, 3)
        ),
        "Adaptive System": tuple(
            _OB_DIR / "rich" / "adaptive" / f"test_memory_fold_{fold}.csv"
            for fold in (1, 2, 3)
        ),
    },
    "sparse": {
        "BL-Lower": tuple(
            _OB_DIR / "sparse" / "baselines" / f"sparse_baseline_lower_fold{fold}.csv"
            for fold in (1, 2, 3)
        ),
        "BL-Upper": tuple(
            _OB_DIR / "sparse" / "baselines" / f"sparse_baseline_upper_fold{fold}.csv"
            for fold in (1, 2, 3)
        ),
        "Blueprint": tuple(
            _OB_DIR / "sparse" / "blueprints" / f"sparse_test_blueprint_fold{fold}.csv"
            for fold in (1, 2, 3)
        ),
        "Playbook": tuple(
            _OB_DIR / "sparse" / "playbooks" / f"sparse_test_playbook_fold{fold}.csv"
            for fold in (1, 2, 3)
        ),
        "Adaptive System": tuple(
            _OB_DIR / "sparse" / "adaptive" / f"sparse_test_full_fold{fold}.csv"
            for fold in (1, 2, 3)
        ),
    },
}

OFFICEBENCH_CONFIG = BenchmarkConfig(
    name="officebench",
    real_agents=REAL_AGENTS_OFFICEBENCH,
    gold_path=_OB_GOLD,
    card_paths=CARD_PATHS,
    system_order=SYSTEM_ORDER,
)


_GAIA_DIR = OUTPUT_DIR / "gaia"
_GAIA_GOLD = ROOT / "annotations" / "gaia.json"

GAIA_SYSTEM_ORDER = [
    "BL-Lower",
    "BL-Upper",
    "Blueprint",
    "Playbook",
    "Adaptive System",
]

GAIA_CARD_PATHS: dict[str, dict[str, tuple[Path, ...]]] = {
    "rich": {
        "BL-Lower": tuple(
            _GAIA_DIR / "rich" / "baselines" / f"baseline_lower_rich_fold_{fold}.csv"
            for fold in (1, 2, 3)
        ),
        "BL-Upper": tuple(
            _GAIA_DIR / "rich" / "baselines" / f"baseline_upper_rich_fold_{fold}.csv"
            for fold in (1, 2, 3)
        ),
        "Blueprint":      tuple(
            _GAIA_DIR / "rich" / "blueprints" / f"adaptive_testing_rich_blueprint_fold_{fold}.csv"
            for fold in (1, 2, 3)
        ),        
        "Playbook": tuple(
            _GAIA_DIR / "rich" / "playbooks" / f"adaptive_testing_rich_playbook_fold_{fold}.csv"
            for fold in (1, 2, 3)
        ),        
        "Adaptive System": tuple(
            _GAIA_DIR / "rich" / "adaptive" / f"adaptive_testing_rich_fold_{fold}.csv"
            for fold in (1, 2, 3)
        ),        
    },
    "sparse": {
        "BL-Lower": tuple(
            _GAIA_DIR / "sparse" / "baselines" / f"baseline_lower_fold_{fold}.csv"
            for fold in (1, 2, 3)
        ),
        "BL-Upper": tuple(
            _GAIA_DIR / "sparse" / "baselines" / f"baseline_upper_fold_{fold}.csv"
            for fold in (1, 2, 3)
        ),
        "Blueprint": tuple(
            _GAIA_DIR / "sparse" / "blueprints" / f"blueprints_fold_{fold}.csv"
            for fold in (1, 2, 3)
        ),
        "Playbook": tuple(
            _GAIA_DIR / "sparse" / "playbooks" / f"playbook_fold_{fold}.csv"
            for fold in (1, 2, 3)
        ),
        "Adaptive System": tuple(
            _GAIA_DIR / "sparse" / "adaptive" / f"adaptive_testing_sparse_fold_{fold}.csv"
            for fold in (1, 2, 3)
        ),
    },
}

GAIA_CONFIG = BenchmarkConfig(
    name="gaia",
    real_agents=REAL_GAIA_AGENTS,
    gold_path=_GAIA_GOLD,
    card_paths=GAIA_CARD_PATHS,
    system_order=GAIA_SYSTEM_ORDER,
)

def load_card(card: str, config: BenchmarkConfig = OFFICEBENCH_CONFIG) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for condition in config.system_order:
        for fold, path in enumerate(config.card_paths[card][condition], start=1):
            runs = load_runs([path])
            runs["condition"] = condition
            runs["fold"] = fold
            frames.append(runs)

    df = add_derived_metrics(pd.concat(frames, ignore_index=True, sort=False))
    df["tier"] = df["task_id"].apply(tier_from_task_id)
    return df


def load_gold(config: BenchmarkConfig = OFFICEBENCH_CONFIG) -> dict[str, dict[str, Any]]:
    return json.loads(config.gold_path.read_text())


def tier_from_task_id(task_id: object) -> int | None:
    """Extract tier from task_id.

    OfficeBench: "1-task-name"   → parts[0] = "1"
    GAIA:        "gaia-1-uuid"   → parts[1] = "1"
    """
    parts = str(task_id).split("-")
    if parts[0].isdigit():
        return int(parts[0])
    if len(parts) > 1 and parts[1].isdigit():
        return int(parts[1])
    return None


def gold_agents(gold: dict[str, Any], config: BenchmarkConfig = OFFICEBENCH_CONFIG) -> frozenset[str]:
    return frozenset(str(agent) for agent in gold["gold_agents"]) & config.real_agents


def agent_sequence(value: object) -> list[str]:
    parsed = _json_value(value, [])
    if not isinstance(parsed, list):
        return []
    return [str(agent) for agent in parsed if agent]


def agent_set(value: object) -> frozenset[str]:
    return frozenset(agent_sequence(value))


def discovery_agents(value: object, config: BenchmarkConfig = OFFICEBENCH_CONFIG) -> frozenset[str]:
    return discovery_all(value) & config.real_agents


def discovery_all(value: object) -> frozenset[str]:
    parsed = _json_value(value, {})
    if isinstance(parsed, dict):
        names = parsed.keys()
    elif isinstance(parsed, list):
        names = parsed
    else:
        names = []
    return frozenset(str(name) for name in names if name)


def _json_value(value: object, default: object) -> object:
    if value is None or value is pd.NA:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    if isinstance(value, str):
        return json.loads(value) if value else default
    return value
