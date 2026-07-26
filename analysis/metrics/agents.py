"""New-agent adoption and playbook activity for the dynamic-pool runs.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from loader import (
    OUTPUT_DIR,
    REAL_AGENTS_OFFICEBENCH,
    REAL_GAIA_AGENTS,
    add_derived_metrics,
    load_runs,
)

from .conditions import agent_sequence, distractor_share


STABLE_POOL_PATHS = {
    "officebench": tuple(
        OUTPUT_DIR
        / "officebench"
        / "dynamic_pool"
        / f"stable_pool_fold_{fold}.csv"
        for fold in (1, 2, 3)
    ),
    "gaia": tuple(
        OUTPUT_DIR
        / "gaia"
        / "dynamic_pool"
        / f"stable_pool_fold_{fold}.csv"
        for fold in (1, 2, 3)
    ),
}

DYNAMIC_PATHS = {
    "officebench": tuple(
        OUTPUT_DIR / "officebench" / "dynamic_pool" / f"test_memory_new_agents_fold_{fold}.csv"
        for fold in (1, 2, 3)
    ),
    "gaia": tuple(
        OUTPUT_DIR / "gaia" / "dynamic_pool" / f"train_memory_new_agents_fold_{fold}.csv"
        for fold in (1, 2, 3)
    ),
}

PAIRED_ORIGINAL_PATHS = {
    "officebench": tuple(
        OUTPUT_DIR / "officebench" / "rich" / "adaptive" / f"test_memory_fold_{fold}.csv"
        for fold in (1, 2, 3)
    ),
    "gaia": tuple(
        OUTPUT_DIR / "gaia" / "rich" / "adaptive" / f"adaptive_testing_rich_fold_{fold}.csv"
        for fold in (1, 2, 3)
    ),
}

# Agents introduced partway through the dynamic-pool run.
NEW_AGENTS = {
    "officebench": ["env_explorer", "file_transformer", "image_creator"],
    "gaia": ["file_document_surfer", "file_table_surfer", "file_image_surfer", "file_audio_surfer"],
}

# Existing agents that also start accumulating playbooks in the dynamic run.
CAPABILITY_SHIFT_AGENTS = {
    "officebench": ["pdf", "word", "excel"],
    "gaia": ["file_surfer"],
}

BASE_AGENTS = {
    "officebench": REAL_AGENTS_OFFICEBENCH,
    "gaia": REAL_GAIA_AGENTS,
}

USAGE_SHIFT_AGENTS = {
    "officebench": ["shell", "word", "excel", "pdf", "env_explorer", "file_transformer", "image_creator"],
    "gaia": [
        "file_surfer",
        "file_document_surfer",
        "file_table_surfer",
        "file_image_surfer",
        "file_audio_surfer",
    ],
}

STABILITY_ROUNDING = {
    "Fold 1 SR (%)": 1,
    "Fold 2 SR (%)": 1,
    "Fold 3 SR (%)": 1,
    "Overall SR (%)": 1,
    "SR Std. (%)": 1,
    "Delegations": 2,
    "Tokens (K)": 1,
    "Cost ($)": 3,
    "Distract (%)": 1,
}


def stability_table(benchmark: str) -> pd.DataFrame:
    """Compare phase-2 stability with and without pool change."""
    base_agents = set(BASE_AGENTS[benchmark])
    changed_pool_agents = base_agents | set(NEW_AGENTS[benchmark])
    conditions = {
        "Original Pool (Control)": (STABLE_POOL_PATHS[benchmark], base_agents),
        "Restructured Pool": (DYNAMIC_PATHS[benchmark], changed_pool_agents),
    }
    return _stability_comparison(conditions)


def _stability_comparison(
    conditions: dict[str, tuple[Iterable[str | Path], set[str]]],
) -> pd.DataFrame:
    condition_runs = {
        name: [_load_runs(path) for path in paths]
        for name, (paths, _) in conditions.items()
    }
    expected_tasks = _expected_tasks_by_fold(condition_runs)

    rows = []
    for condition, (_, valid_agents) in conditions.items():
        fold_runs = condition_runs[condition]
        fold_stats = pd.DataFrame(
            _stability_fold_summary(runs, valid_agents, len(task_keys))
            for runs, task_keys in zip(fold_runs, expected_tasks, strict=True)
        )
        fold_rates = fold_stats["Success Rate (%)"]
        rows.append(
            {
                "Condition": condition,
                "Fold 1 SR (%)": fold_rates.iloc[0],
                "Fold 2 SR (%)": fold_rates.iloc[1],
                "Fold 3 SR (%)": fold_rates.iloc[2],
                "Overall SR (%)": fold_rates.mean(),
                "SR Std. (%)": fold_rates.std(),
                "Delegations": fold_stats["Delegations"].mean(),
                "Tokens (K)": fold_stats["Tokens (K)"].mean(),
                "Cost ($)": fold_stats["Cost ($)"].mean(),
                "Distract (%)": fold_stats["Distract (%)"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("Condition").round(STABILITY_ROUNDING)


def _expected_tasks_by_fold(
    condition_runs: dict[str, list[pd.DataFrame]],
) -> list[set[str]]:
    fold_count = len(next(iter(condition_runs.values())))
    return [
        set().union(*(set(runs[fold]["task_key"]) for runs in condition_runs.values()))
        for fold in range(fold_count)
    ]


def adoption_rate(benchmark: str) -> pd.DataFrame:
    """How often, and how early, each newly introduced agent is used per fold."""
    new_agents = NEW_AGENTS[benchmark]
    rows = []
    for fold, fold_runs in _load_folds(DYNAMIC_PATHS[benchmark]).groupby("fold", sort=True):
        usage = _agent_usage_by_step(fold_runs, new_agents)
        n = len(usage)
        for agent in new_agents:
            hits = usage[usage[agent]]
            rows.append(
                {
                    "fold": int(fold),
                    "agent": agent,
                    "used_tasks": len(hits),
                    "usage_rate (%)": (len(hits) / n * 100) if n else 0.0,
                    "first_step": int(hits["Step"].iloc[0]) if not hits.empty else pd.NA,
                    "first_task_key": hits["task_key"].iloc[0] if not hits.empty else pd.NA,
                }
            )
    return pd.DataFrame(rows).round({"usage_rate (%)": 1})


def adoption_by_fold(benchmark: str) -> pd.DataFrame:
    """Cumulative new-agent usage per step, with the step counter reset per fold."""
    new_agents = NEW_AGENTS[benchmark]
    frames = []
    for fold, fold_runs in _load_folds(DYNAMIC_PATHS[benchmark]).groupby("fold", sort=True):
        usage = _agent_usage_by_step(fold_runs, new_agents)
        cumulative = usage[["Step"]].copy()
        cumulative[new_agents] = usage[new_agents].cumsum()
        cumulative.insert(0, "fold", int(fold))
        frames.append(cumulative)
    return pd.concat(frames, ignore_index=True, sort=False)



def agent_usage_shift_table(
    benchmark: str,
    *,
    fold_average: bool = False,
) -> pd.DataFrame:
    """Compare old/new agent usage on paired original-pool and dynamic-pool runs."""
    original = _paired_condition_usage(
        benchmark,
        "Original Pool",
        by_fold=fold_average,
    )
    dynamic = _paired_condition_usage(
        benchmark,
        "Dynamic Pool",
        by_fold=fold_average,
    )
    merge_keys = ["Agent"]
    if fold_average:
        merge_keys.insert(0, "Fold")
    merged = original.merge(dynamic, on=merge_keys, suffixes=(" Original", " Dynamic"))
    merged["Delta task usage (pp)"] = (
        merged["Task usage (%) Dynamic"] - merged["Task usage (%) Original"]
    )
    merged["Delta agent-use share (pp)"] = (
        merged["Agent-use share (%) Dynamic"] - merged["Agent-use share (%) Original"]
    )

    if fold_average:
        merged = merged.groupby("Agent", sort=False, as_index=False).agg(
            {
                "Task usage (%) Original": "mean",
                "Task usage (%) Dynamic": "mean",
                "Delta task usage (pp)": "mean",
                "Agent-use share (%) Original": "mean",
                "Agent-use share (%) Dynamic": "mean",
                "Delta agent-use share (pp)": "mean",
            }
        )
    return merged[
        [
            "Agent",
            "Task usage (%) Original",
            "Task usage (%) Dynamic",
            "Delta task usage (pp)",
            "Agent-use share (%) Original",
            "Agent-use share (%) Dynamic",
            "Delta agent-use share (pp)",
        ]
    ].round(
        {
            "Task usage (%) Original": 1,
            "Task usage (%) Dynamic": 1,
            "Delta task usage (pp)": 1,
            "Agent-use share (%) Original": 1,
            "Agent-use share (%) Dynamic": 1,
            "Delta agent-use share (pp)": 1,
        }
    )



def playbook_activity(benchmark: str) -> pd.DataFrame:
    """Playbook bullets added or pruned for the tracked agents during the dynamic run."""
    tracked = NEW_AGENTS[benchmark] + CAPABILITY_SHIFT_AGENTS[benchmark]
    columns = ["event_type", "agent", "section", "Step", "task_key", "bullet_id", "rule"]
    frames = []
    for fold, fold_runs in _load_folds(DYNAMIC_PATHS[benchmark]).groupby("fold", sort=True):
        bullets = _bullet_table(fold_runs, tracked)
        if bullets.empty:
            continue
        bullets.insert(0, "fold", int(fold))
        frames.append(bullets[["fold", *columns]])
    if not frames:
        return pd.DataFrame(columns=["fold", *columns])
    return pd.concat(frames, ignore_index=True, sort=False).sort_values(
        ["event_type", "agent", "section", "Step", "bullet_id"], kind="stable"
    )


def _load_runs(path: str | Path) -> pd.DataFrame:
    """Load a single fold's CSV with derived metrics."""
    return add_derived_metrics(load_runs([Path(path)]))


def _load_folds(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load the per-fold CSVs into one frame, tagged with a ``fold`` column."""
    frames = []
    for fold, path in enumerate(paths, start=1):
        runs = load_runs([Path(path)])
        runs["fold"] = fold
        frames.append(runs)
    return add_derived_metrics(pd.concat(frames, ignore_index=True, sort=False))


def _stability_fold_summary(
    runs: pd.DataFrame,
    valid_agents: set[str],
    expected_task_count: int,
) -> dict[str, float]:
    """Fold-level success and efficiency metrics for the dynamic-pool table."""
    return {
        "Success Rate (%)": runs["is_success"].sum() / expected_task_count * 100,
        "Delegations": runs["total_delegations"].mean(),
        "Tokens (K)": runs["tokens_total"].mean() / 1000,
        "Cost ($)": runs["cost_total"].mean(),
        "Distract (%)": distractor_share(runs, valid_agents),
    }

def _paired_condition_usage(
    benchmark: str,
    condition: str,
    *,
    by_fold: bool,
) -> pd.DataFrame:
    paths = {
        "Original Pool": PAIRED_ORIGINAL_PATHS[benchmark],
        "Dynamic Pool": DYNAMIC_PATHS[benchmark],
    }[condition]
    frames = []
    for fold, path in enumerate(paths, start=1):
        runs = _load_runs(path)
        if by_fold:
            fold_summary = _agent_usage_summary(runs, USAGE_SHIFT_AGENTS[benchmark])
            fold_summary.insert(0, "Fold", fold)
            frames.append(fold_summary)
            continue
        frames.append(runs)

    if by_fold:
        return pd.concat(frames, ignore_index=True, sort=False)

    runs = pd.concat(frames, ignore_index=True, sort=False)
    return _agent_usage_summary(runs, USAGE_SHIFT_AGENTS[benchmark])


def _agent_usage_summary(runs: pd.DataFrame, agents: Iterable[str]) -> pd.DataFrame:
    """Count episode-agent uses from the deduplicated ``executed_agents`` field."""
    sequences = runs["executed_agents"].apply(agent_sequence)
    total_agent_uses = sum(len(sequence) for sequence in sequences)
    rows = []
    for agent in agents:
        task_used = int(sequences.apply(lambda sequence: agent in set(sequence)).sum())
        agent_uses = int(sum(sequence.count(agent) for sequence in sequences))
        rows.append(
            {
                "Agent": agent,
                "N": len(runs),
                "Task used": task_used,
                "Task usage (%)": task_used / len(runs) * 100 if len(runs) else 0.0,
                "Agent uses": agent_uses,
                "Agent-use share (%)": (
                    agent_uses / total_agent_uses * 100 if total_agent_uses else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _order_by_step(runs: pd.DataFrame) -> pd.DataFrame:
    """Chronological order of subtask executions (one step per row)."""
    return runs.sort_values(["created_at_utc", "task_id", "subtask_id"], kind="stable")


def _agent_usage_by_step(runs: pd.DataFrame, agents: Iterable[str]) -> pd.DataFrame:
    """Per-step boolean of whether each agent was executed."""
    rows = []
    for step, (_, row) in enumerate(_order_by_step(runs).iterrows(), start=1):
        used = set(agent_sequence(row.get("executed_agents")))
        rows.append(
            {
                "Step": step,
                "task_key": row.get("task_key"),
                **{agent: agent in used for agent in agents},
            }
        )
    return pd.DataFrame(rows)


def _bullet_table(runs: pd.DataFrame, agents: Iterable[str]) -> pd.DataFrame:
    """Flatten added/pruned playbook bullets for the given agents."""
    agent_filter = set(agents)
    rows = []
    for step, (_, row) in enumerate(_order_by_step(runs).iterrows(), start=1):
        for event in _playbook_evolution(row.get("playbook_evolution")):
            agent = str(event.get("agent", ""))
            if agent not in agent_filter:
                continue
            for event_type, key in (("added", "added_bullets"), ("pruned", "pruned_bullets")):
                for bullet in event.get(key) or []:
                    rows.append(
                        {
                            "Step": step,
                            "task_key": row.get("task_key"),
                            "agent": agent,
                            "event_type": event_type,
                            "bullet_id": bullet.get("bullet_id", ""),
                            "section": str(bullet.get("section", "")),
                            "rule": bullet.get("rule", ""),
                        }
                    )
    return pd.DataFrame(rows)


def _playbook_evolution(value: object) -> list[dict]:
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]
