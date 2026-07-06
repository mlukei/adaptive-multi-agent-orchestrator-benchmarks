from __future__ import annotations

from pathlib import Path

import pandas as pd

from loader import add_derived_metrics, load_runs

from .conditions import OUTPUT_DIR, agent_sequence, tier_from_task_id

DEFAULT_PHASE1_PATH = OUTPUT_DIR / "officebench" / "dynamic_pool" / "train_memory_original_pool.csv"
DEFAULT_PHASE2_PATH = OUTPUT_DIR / "officebench" / "dynamic_pool" / "train_memory_new_agents.csv"
NEW_AGENTS = ("env_explorer", "file_transformer", "image_creator")

TIER_COLUMNS = {
    1: "T1 SR (%)",
    2: "T2 SR (%)",
    3: "T3 SR (%)",
}


def load_phase(path: str | Path) -> pd.DataFrame:
    runs = add_derived_metrics(load_runs([Path(path)]))
    runs["tier"] = runs["task_id"].apply(tier_from_task_id)
    return runs


def phase_table(
    phase1_path: str | Path = DEFAULT_PHASE1_PATH,
    phase2_path: str | Path = DEFAULT_PHASE2_PATH,
) -> pd.DataFrame:
    rows = [
        _phase_summary("Phase 1", load_phase(phase1_path)),
        _phase_summary("Phase 2", load_phase(phase2_path)),
    ]
    return pd.DataFrame(rows).set_index("Phase").round(
        {
            "T1 SR (%)": 1,
            "T2 SR (%)": 1,
            "T3 SR (%)": 1,
            "Overall SR (%)": 1,
            "Avg. Delegations": 2,
        }
    )


def adoption(phase2: pd.DataFrame) -> pd.DataFrame:
    ordered = phase2.sort_values(["created_at_utc", "task_id", "subtask_id"], kind="stable")
    counts = {agent: 0 for agent in NEW_AGENTS}
    rows = []
    for step, (_, row) in enumerate(ordered.iterrows(), start=1):
        for agent in agent_sequence(row.get("executed_agents")):
            if agent in counts:
                counts[agent] += 1
        rows.append({"Step": step, **counts})
    return pd.DataFrame(rows)


def _phase_summary(label: str, runs: pd.DataFrame) -> dict[str, float | int | str]:
    summary: dict[str, float | int | str] = {
        "Phase": label,
        "n_subtasks": len(runs),
        "n_tasks": runs["task_id"].nunique(),
    }
    for tier, column in TIER_COLUMNS.items():
        summary[column] = _success_rate(runs[runs["tier"] == tier])
    summary["Overall SR (%)"] = _success_rate(runs)
    summary["Avg. Delegations"] = runs["total_delegations"].mean()
    return summary


def _success_rate(runs: pd.DataFrame) -> float:
    return runs["is_success"].mean() * 100
