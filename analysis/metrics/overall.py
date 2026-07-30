from __future__ import annotations

from collections.abc import Callable, Mapping

import pandas as pd

from loader import BenchmarkConfig

from .conditions import OFFICEBENCH_CONFIG, distractor_share, load_card

TIER_COLUMNS = {
    1: "T1 SR (%)",
    2: "T2 SR (%)",
    3: "T3 SR (%)",
}


def per_split_metrics(
    runs: pd.DataFrame,
    summarize: Callable[[pd.DataFrame], Mapping[str, float | int]],
    *,
    split_column: str = "split",
) -> pd.DataFrame:
    """Compute one metric row per split so callers can average splits equally."""
    return pd.DataFrame(
        summarize(split_runs)
        for _, split_runs in runs.groupby(split_column, sort=True)
    )


def overall_table(card: str, config: BenchmarkConfig = OFFICEBENCH_CONFIG) -> pd.DataFrame:
    runs = load_card(card, config)
    expected = expected_counts_by_split(runs)
    rows = [
        _condition_summary(runs, condition, config, expected)
        for condition in config.system_order
    ]
    table = pd.DataFrame(rows).set_index("Condition")
    return table.round(
        {
            "T1 SR (%)": 1,
            "T2 SR (%)": 1,
            "T3 SR (%)": 1,
            "Overall SR (%)": 1,
            "SR Std. (%)": 1,
            "Delegations": 2,
            "Tokens (K)": 1,
            "Cost ($)": 3,
            "Distract (%)": 1,
        }
    )


def expected_counts_by_split(runs: pd.DataFrame) -> dict[tuple[int, int | None], int]:
    """
    Task counts per split, and per split/tier, unioned across all conditions.
    """
    counts: dict[tuple[int, int | None], int] = {}
    for split, split_runs in runs.groupby("split"):
        counts[(int(split), None)] = split_runs["task_key"].nunique()
        for tier, tier_runs in split_runs.groupby("tier"):
            counts[(int(split), int(tier))] = tier_runs["task_key"].nunique()
    return counts


def _condition_summary(
    runs: pd.DataFrame,
    condition: str,
    config: BenchmarkConfig,
    expected: dict[tuple[int, int | None], int],
) -> dict[str, float | str]:
    split_stats = per_split_metrics(
        runs[runs["condition"] == condition],
        lambda split_runs: _split_summary(split_runs, config, expected),
    )
    summary: dict[str, float | str] = {"Condition": condition}
    for column in TIER_COLUMNS.values():
        summary[column] = split_stats[column].mean()
    summary["Overall SR (%)"] = split_stats["Overall SR (%)"].mean()
    summary["SR Std. (%)"] = split_stats["Overall SR (%)"].std()
    summary["Delegations"] = split_stats["Delegations"].mean()
    summary["Tokens (K)"] = split_stats["Tokens (K)"].mean()
    summary["Cost ($)"] = split_stats["Cost ($)"].mean()
    summary["Distract (%)"] = split_stats["Distract (%)"].mean()
    return summary


def _split_summary(
    runs: pd.DataFrame,
    config: BenchmarkConfig,
    expected: dict[tuple[int, int | None], int],
) -> dict[str, float]:
    split = int(runs["split"].iloc[0])
    summary = {
        column: success_rate(runs[runs["tier"] == tier], expected.get((split, tier)))
        for tier, column in TIER_COLUMNS.items()
    }
    summary["Overall SR (%)"] = success_rate(runs, expected.get((split, None)))
    summary["Delegations"] = runs["total_delegations"].mean()
    summary["Tokens (K)"] = runs["tokens_total"].mean() / 1000
    summary["Cost ($)"] = runs["cost_total"].mean()
    summary["Distract (%)"] = distractor_share(runs, config.real_agents)
    return summary


def success_rate(runs: pd.DataFrame, expected_count: int | None = None) -> float:
    """Percentage of successful runs.
    """
    denominator = expected_count if expected_count else len(runs)
    if not denominator:
        return 0.0
    return runs["is_success"].sum() / denominator * 100
