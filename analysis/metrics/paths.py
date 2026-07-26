"""Solution-path deviation metrics derived from discovery and delegation logs."""

from __future__ import annotations

import pandas as pd

from loader import BenchmarkConfig, percentage

from .conditions import (
    OFFICEBENCH_CONFIG,
    agent_set,
    discovery_all,
    iter_gold_task_rows,
    load_card,
    load_gold,
)


def per_row_paths(
    card: str,
    config: BenchmarkConfig = OFFICEBENCH_CONFIG,
) -> pd.DataFrame:
    """Return one path-coverage record per episode with a non-empty gold set."""
    runs = load_card(card, config)
    gold = load_gold(config)
    rows = []
    for row, task_key, gold_set in iter_gold_task_rows(runs, gold, config):
        discovered = discovery_all(row.get("agent_discovery_sources"))
        executed = agent_set(row.get("executed_agents"))
        rows.append(
            {
                "Benchmark": config.name,
                "Condition": row["condition"],
                "Fold": row["fold"],
                "Task Key": task_key,
                "Success": row["success"] == 1,
                "Gold Count": len(gold_set),
                "Retrieval All Gold": gold_set.issubset(discovered),
                "Usage All Gold": gold_set.issubset(executed),
                "Shell Used": "shell" in executed,
                "Shell Non-Gold": "shell" in executed and "shell" not in gold_set,
                "Shell Workaround Candidate": (
                    "shell" in executed
                    and "shell" not in gold_set
                    and not gold_set.issubset(executed)
                ),
            }
        )
    return pd.DataFrame(rows)



def shell_workaround_table(
    card: str = "rich",
    config: BenchmarkConfig = OFFICEBENCH_CONFIG,
    *,
    condition: str = "Adaptive System",
) -> pd.DataFrame:
    """Summarize episode-level Shell use and off-gold workaround candidates."""
    rows = per_row_paths(card, config)
    rows = rows[rows["Condition"] == condition]
    successes = rows[rows["Success"]]
    successful_shell = successes[successes["Shell Used"]]
    metrics = [
        ("Shell used", rows["Shell Used"], len(rows), "All gold-annotated episodes"),
        (
            "Shell used",
            successes["Shell Used"],
            len(successes),
            "Successful episodes",
        ),
        (
            "Non-gold Shell used",
            successes["Shell Non-Gold"],
            len(successes),
            "Successful episodes",
        ),
        (
            "Shell workaround candidate",
            successes["Shell Workaround Candidate"],
            len(successes),
            "Successful episodes",
        ),
        (
            "Shell workaround candidate",
            successful_shell["Shell Workaround Candidate"],
            len(successful_shell),
            "Successful episodes using Shell",
        ),
    ]
    result = []
    for metric, mask, denominator, population in metrics:
        count = int(mask.sum())
        result.append(
            {
                "Benchmark": _benchmark_label(config.name),
                "Cards": _card_label(card),
                "Metric": metric,
                "Population": population,
                "Count": count,
                "Denominator": denominator,
                "Rate (%)": percentage(count, denominator),
            }
        )
    return pd.DataFrame(result).round(1)



def _benchmark_label(name: str) -> str:
    return {"officebench": "OfficeBench", "gaia": "GAIA"}.get(name, name)


def _card_label(card: str) -> str:
    return card.capitalize()
