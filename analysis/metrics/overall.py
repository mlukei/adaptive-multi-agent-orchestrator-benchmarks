from __future__ import annotations

import pandas as pd

from loader import BenchmarkConfig

from .conditions import OFFICEBENCH_CONFIG, distractor_share, load_card

TIER_COLUMNS = {
    1: "T1 SR (%)",
    2: "T2 SR (%)",
    3: "T3 SR (%)",
}


def overall_table(card: str, config: BenchmarkConfig = OFFICEBENCH_CONFIG) -> pd.DataFrame:
    runs = load_card(card, config)
    expected = _expected_counts_by_fold(runs)
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


def _expected_counts_by_fold(runs: pd.DataFrame) -> dict[tuple[int, int | None], int]:
    """
    Task counts per fold, and per fold/tier, unioned across all conditions.
    """
    counts: dict[tuple[int, int | None], int] = {}
    for fold, fold_runs in runs.groupby("fold"):
        counts[(int(fold), None)] = fold_runs["task_key"].nunique()
        for tier, tier_runs in fold_runs.groupby("tier"):
            counts[(int(fold), int(tier))] = tier_runs["task_key"].nunique()
    return counts


def _condition_summary(
    runs: pd.DataFrame,
    condition: str,
    config: BenchmarkConfig,
    expected: dict[tuple[int, int | None], int],
) -> dict[str, float | str]:
    fold_stats = pd.DataFrame(
        _fold_summary(fold, config, expected)
        for _, fold in runs[runs["condition"] == condition].groupby("fold")
    )
    summary: dict[str, float | str] = {"Condition": condition}
    for column in TIER_COLUMNS.values():
        summary[column] = fold_stats[column].mean()
    summary["Overall SR (%)"] = fold_stats["Overall SR (%)"].mean()
    summary["SR Std. (%)"] = fold_stats["Overall SR (%)"].std()
    summary["Delegations"] = fold_stats["Delegations"].mean()
    summary["Tokens (K)"] = fold_stats["Tokens (K)"].mean()
    summary["Cost ($)"] = fold_stats["Cost ($)"].mean()
    summary["Distract (%)"] = fold_stats["Distract (%)"].mean()
    return summary


def _fold_summary(
    runs: pd.DataFrame,
    config: BenchmarkConfig,
    expected: dict[tuple[int, int | None], int],
) -> dict[str, float]:
    fold = int(runs["fold"].iloc[0])
    summary = {
        column: success_rate(runs[runs["tier"] == tier], expected.get((fold, tier)))
        for tier, column in TIER_COLUMNS.items()
    }
    summary["Overall SR (%)"] = success_rate(runs, expected.get((fold, None)))
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
