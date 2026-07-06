from __future__ import annotations

import json

import pandas as pd

from loader import BenchmarkConfig

from .conditions import OFFICEBENCH_CONFIG, discovery_all, gold_agents, load_card, load_gold


def retrieval_table(card: str, config: BenchmarkConfig = OFFICEBENCH_CONFIG) -> pd.DataFrame:
    metrics = per_row_metrics(card, config)
    rows = []
    for condition in config.system_order:
        condition_metrics = metrics[metrics["Condition"] == condition]
        # Capability-list usage is aggregated at the retrieval-call level (a row
        # may issue several calls), so we sum the per-row counts rather than
        # averaging per-row means.
        calls = condition_metrics["Retrieval Calls"].sum()
        rows.append(
            {
                "Condition": condition,
                "Recall": condition_metrics["Recall"].mean(),
                "Precision": condition_metrics["Precision"].mean(),
                "All Gold (%)": condition_metrics["All Gold"].mean() * 100,
                "Caps/Call": (
                    condition_metrics["Capabilities Sent"].sum() / calls if calls else 0.0
                ),
                "Empty Caps (%)": (
                    condition_metrics["Empty Cap Calls"].sum() / calls * 100 if calls else 0.0
                ),
            }
        )
    return pd.DataFrame(rows).set_index("Condition").round(
        {
            "Recall": 3,
            "Precision": 3,
            "All Gold (%)": 1,
            "Caps/Call": 2,
            "Empty Caps (%)": 1,
        }
    )


def per_row_metrics(card: str, config: BenchmarkConfig = OFFICEBENCH_CONFIG) -> pd.DataFrame:
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
        n_calls, n_caps, n_empty = _capability_usage(row.get("agent_retrieval_log"))
        rows.append(
            {
                "Condition": row["condition"],
                "Fold": row["fold"],
                "Task Key": task_key,
                "Recall": _recall(retrieved, gold_set),
                "Precision": _precision(retrieved, gold_set),
                "All Gold": float(gold_set.issubset(retrieved)),
                "Retrieval Calls": n_calls,
                "Capabilities Sent": n_caps,
                "Empty Cap Calls": n_empty,
            }
        )
    return pd.DataFrame(rows)


def _capability_usage(value: object) -> tuple[int, int, int]:
    """Per-row capability-list usage from ``agent_retrieval_log``.

    Returns (number of retrieval calls, total capabilities sent across calls,
    number of calls that sent an empty capability list). A call with no
    capabilities falls back to blueprint/goal-only search, which returns fewer
    agents than an explicit capability list.
    """
    if not isinstance(value, str) or not value:
        return 0, 0, 0
    log = json.loads(value)
    n_calls = len(log)
    n_caps = sum(len(record.get("capabilities") or []) for record in log)
    n_empty = sum(1 for record in log if not record.get("capabilities"))
    return n_calls, n_caps, n_empty


def _recall(retrieved: frozenset[str], gold_set: frozenset[str]) -> float:
    if not gold_set:
        return 0.0
    return len(retrieved & gold_set) / len(gold_set)


def _precision(retrieved: frozenset[str], gold_set: frozenset[str]) -> float:
    if not retrieved:
        return 0.0
    return len(retrieved & gold_set) / len(retrieved)
