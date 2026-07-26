"""Solution-path deviation metrics derived from discovery and delegation logs."""

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
from .errors import classified_rows


def per_row_paths(
    card: str,
    config: BenchmarkConfig = OFFICEBENCH_CONFIG,
) -> pd.DataFrame:
    """Return one path-coverage record per episode with a non-empty gold set."""
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


def off_gold_success_table(
    card: str,
    config: BenchmarkConfig = OFFICEBENCH_CONFIG,
    *,
    condition: str = "Adaptive System",
) -> pd.DataFrame:
    """Summarize successful episodes without full retrieval or usage coverage."""
    rows = per_row_paths(card, config)
    rows = rows[rows["Condition"] == condition]
    successes = rows[rows["Success"]]
    result = []
    for metric, coverage_column in (
        ("Retrieval Off-Gold", "Retrieval All Gold"),
        ("Usage Off-Gold", "Usage All Gold"),
    ):
        count = int((~successes[coverage_column]).sum())
        result.append(
            {
                "Benchmark": _benchmark_label(config.name),
                "Cards": _card_label(card),
                "Metric": metric,
                "Off-Gold Successes": count,
                "Gold-Annotated Episodes": len(rows),
                "Successful Episodes": len(successes),
                "All Episodes (%)": _percentage(count, len(rows)),
                "Among Successes (%)": _percentage(count, len(successes)),
            }
        )
    return pd.DataFrame(result).round(1)


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
                "Rate (%)": _percentage(count, denominator),
            }
        )
    return pd.DataFrame(result).round(1)


def selection_miss_table(
    card: str,
    config: BenchmarkConfig = OFFICEBENCH_CONFIG,
    *,
    condition: str = "Adaptive System",
) -> pd.DataFrame:
    """Summarize failed episodes attributed to incomplete gold-agent selection."""
    rows = classified_rows(card, config)
    rows = rows[rows["Condition"] == condition]
    selection_misses = int((rows["Error"] == "selection_miss").sum())
    return pd.DataFrame(
        [
            {
                "Benchmark": _benchmark_label(config.name),
                "Cards": _card_label(card),
                "Selection Misses": selection_misses,
                "Episodes": len(rows),
                "Selection Miss (%)": _percentage(selection_misses, len(rows)),
            }
        ]
    ).round(1)


def _percentage(numerator: int, denominator: int) -> float:
    return numerator / denominator * 100 if denominator else 0.0


def _benchmark_label(name: str) -> str:
    return {"officebench": "OfficeBench", "gaia": "GAIA"}.get(name, name)


def _card_label(card: str) -> str:
    return card.capitalize()
