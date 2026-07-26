from __future__ import annotations

import json

import pandas as pd

from loader import BenchmarkConfig

from .conditions import OFFICEBENCH_CONFIG, discovery_all, gold_agents, load_card, load_gold


def retrieval_table(card: str, config: BenchmarkConfig = OFFICEBENCH_CONFIG) -> pd.DataFrame:
    """Aggregate episode-level retrieval coverage by experimental condition."""
    metrics = per_row_metrics(card, config)
    rows = []
    for condition in config.system_order:
        condition_metrics = metrics[metrics["Condition"] == condition]
        rows.append(
            {
                "Condition": condition,
                "Recall": condition_metrics["Recall"].mean(),
                "Precision": condition_metrics["Precision"].mean(),
                "All Gold (%)": condition_metrics["All Gold"].mean() * 100,
                "Pool Size": condition_metrics["Pool Size"].mean(),
                "Retrieval Calls": condition_metrics["Retrieval Calls"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("Condition").round(
        {
            "Recall": 3,
            "Precision": 3,
            "All Gold (%)": 1,
            "Pool Size": 2,
            "Retrieval Calls": 2,
        }
    )


def per_row_metrics(card: str, config: BenchmarkConfig = OFFICEBENCH_CONFIG) -> pd.DataFrame:
    """Measure coverage over the union of agents discovered during an episode."""
    runs = load_card(card, config)
    gold = load_gold(config)
    rows = []
    for _, row in runs.iterrows():
        task_key = row["task_key"]
        if task_key not in gold:
            continue
        gold_set = gold_agents(gold[task_key], config)
        if not gold_set:
            continue
        retrieved = discovery_all(row.get("agent_discovery_sources"))
        rows.append(
            {
                "Condition": row["condition"],
                "Fold": row["fold"],
                "Task Key": task_key,
                "Recall": _recall(retrieved, gold_set),
                "Precision": _precision(retrieved, gold_set),
                "All Gold": float(gold_set.issubset(retrieved)),
                "Pool Size": len(retrieved),
                "Retrieval Calls": _retrieval_calls(row.get("agent_retrieval_log")),
            }
        )
    return pd.DataFrame(rows)


def _recall(retrieved: frozenset[str], gold_set: frozenset[str]) -> float:
    if not gold_set:
        return 0.0
    return len(retrieved & gold_set) / len(gold_set)


def _precision(retrieved: frozenset[str], gold_set: frozenset[str]) -> float:
    if not retrieved:
        return 0.0
    return len(retrieved & gold_set) / len(retrieved)


def _retrieval_calls(value: object) -> int:
    if value is None or value is pd.NA:
        return 0
    if isinstance(value, float) and pd.isna(value):
        return 0
    if isinstance(value, str):
        if not value:
            return 0
        value = json.loads(value)
    return len(value) if isinstance(value, list) else 0
