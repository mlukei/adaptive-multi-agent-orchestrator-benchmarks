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
    rows = classified_rows(card, config)
    table_rows = []
    for condition in config.system_order:
        counts = rows[rows["Condition"] == condition]["Error"].value_counts(normalize=True)
        table_rows.append(
            {"Condition": condition}
            | {ERROR_COLUMNS[label]: counts.get(label, 0.0) * 100 for label in ERROR_LABELS}
        )
    return pd.DataFrame(table_rows).set_index("Condition").round(1)


def classified_rows(card: str, config: BenchmarkConfig = OFFICEBENCH_CONFIG) -> pd.DataFrame:
    runs = load_card(card, config)
    gold = load_gold(config)
    rows = []
    for _, row in runs.iterrows():
        task_key = row["task_key"]
        if task_key not in gold:
            continue
        rows.append(
            {
                "Condition": row["condition"],
                "Fold": row["fold"],
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
