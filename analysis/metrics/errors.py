from __future__ import annotations

import pandas as pd

from loader import BenchmarkConfig

from .conditions import (
    OFFICEBENCH_CONFIG,
    agent_set,
    discovery_all,
    gold_agents,
    load_card,
    load_gold,
)
from .overall import per_split_metrics

ERROR_LABELS = [
    "success",
    "retrieval_miss",
    "selection_miss",
    "execution_fail",
]

ERROR_COLUMNS = {
    "success": "Success (%)",
    "retrieval_miss": "Retrieval Miss (%)",
    "selection_miss": "Selection Miss (%)",
    "execution_fail": "Execution Fail (%)",
}


def error_table(card: str, config: BenchmarkConfig = OFFICEBENCH_CONFIG) -> pd.DataFrame:
    rows = _classify_runs(load_card(card, config), config)
    table_rows = []
    for condition in config.system_order:
        split_metrics = per_split_metrics(
            rows[rows["Condition"] == condition],
            _error_rates,
            split_column="Split",
        )
        table_rows.append(
            {"Condition": condition}
            | split_metrics.mean().to_dict()
        )
    return pd.DataFrame(table_rows).set_index("Condition").round(1)


def _error_rates(rows: pd.DataFrame) -> dict[str, float]:
    shares = rows["Error"].value_counts(normalize=True)
    return {
        ERROR_COLUMNS[label]: shares.get(label, 0.0) * 100
        for label in ERROR_LABELS
    }


def classified_rows(card: str, config: BenchmarkConfig = OFFICEBENCH_CONFIG) -> pd.DataFrame:
    runs = load_card(card, config)
    return _classify_runs(runs, config)


def _classify_runs(runs: pd.DataFrame, config: BenchmarkConfig) -> pd.DataFrame:
    gold = load_gold(config)
    missing_gold = sorted(set(runs["task_key"]) - set(gold))
    if missing_gold:
        raise ValueError(
            f"{config.name} has {len(missing_gold)} task keys without gold annotations"
        )

    rows = []
    for _, row in runs.iterrows():
        task_key = row["task_key"]
        rows.append(
            {
                "Condition": row["condition"],
                "Split": row["split"],
                "Task Key": task_key,
                "Error": classify(row, gold[task_key], config),
            }
        )
    return pd.DataFrame(rows)


def classify(row: pd.Series, gold: dict[str, object], config: BenchmarkConfig) -> str:
    """Label success first; otherwise attribute failure to the earliest broken stage."""
    gold_set = gold_agents(gold, config)
    if row["success"] == 1:
        return "success"
    if not gold_set.issubset(discovery_all(row.get("agent_discovery_sources"))):
        return "retrieval_miss"
    if not gold_set.issubset(agent_set(row.get("executed_agents")) & config.real_agents):
        return "selection_miss"
    return "execution_fail"
