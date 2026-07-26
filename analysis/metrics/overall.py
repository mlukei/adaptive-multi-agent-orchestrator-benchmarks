from __future__ import annotations

import pandas as pd

from loader import BenchmarkConfig

from .conditions import OFFICEBENCH_CONFIG, agent_sequence, load_card

TIER_COLUMNS = {
    1: "T1 SR (%)",
    2: "T2 SR (%)",
    3: "T3 SR (%)",
}


def overall_table(card: str, config: BenchmarkConfig = OFFICEBENCH_CONFIG) -> pd.DataFrame:
    runs = load_card(card, config)
    rows = [_condition_summary(runs, condition, config) for condition in config.system_order]
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


def _condition_summary(
    runs: pd.DataFrame, condition: str, config: BenchmarkConfig
) -> dict[str, float | str]:
    fold_stats = pd.DataFrame(
        _fold_summary(fold, config)
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


def _fold_summary(runs: pd.DataFrame, config: BenchmarkConfig) -> dict[str, float]:
    summary = {
        column: success_rate(runs[runs["tier"] == tier])
        for tier, column in TIER_COLUMNS.items()
    }
    summary["Overall SR (%)"] = success_rate(runs)
    summary["Delegations"] = runs["total_delegations"].mean()
    summary["Tokens (K)"] = runs["tokens_total"].mean() / 1000
    summary["Cost ($)"] = runs["cost_total"].mean()
    summary["Distract (%)"] = _distractor_share(runs, config.real_agents)
    return summary


def success_rate(runs: pd.DataFrame) -> float:
    return runs["is_success"].mean() * 100


def _distractor_share(runs: pd.DataFrame, real_agents: frozenset[str]) -> float:
    fractions = runs["executed_agents"].apply(lambda v: _distractor_fraction(v, real_agents))
    mean = fractions.mean()
    return 0.0 if pd.isna(mean) else mean * 100


def _distractor_fraction(value: object, real_agents: frozenset[str]) -> float:
    agents = agent_sequence(value)
    if not agents:
        return float("nan")
    distractors = sum(1 for agent in agents if agent not in real_agents)
    return distractors / len(agents)
