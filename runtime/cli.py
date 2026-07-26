"""Shared CLI argument definitions and task-execution logic.

Used by both benchmark runners.
"""

from __future__ import annotations

import argparse
from typing import Any

from runtime.task_sampling import (
    discover_tasks,
    filter_indices_by_task_split,
    select_task_indices,
)


def add_task_selection_args(parser: argparse.ArgumentParser, cfg: Any) -> None:
    """Add task-slicing arguments to a benchmark parser."""
    parser.add_argument("--tasks-root", default=cfg.execution.tasks_root)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1, help="Run all tasks if -1")
    parser.add_argument(
        "--task-ids", type=int, nargs="+",
        help="Optional explicit task indices in discovered task list",
    )


def add_execution_args(parser: argparse.ArgumentParser, cfg: Any) -> None:
    """Add process-safety options shared by both benchmark runners."""
    execution = cfg.execution
    parser.add_argument(
        "--task-timeout-seconds",
        type=int,
        default=execution.task_timeout_seconds,
        help="Per-task hard timeout in seconds (OS-level kill). Set <=0 to disable.",
    )
    parser.add_argument(
        "--timeout-retries",
        type=int,
        default=getattr(execution, "timeout_retries", 0),
        help="Retries for timed-out tasks (0 = no retry).",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=int,
        default=getattr(execution, "retry_backoff_seconds", 0),
        help="Delay before retrying a timed-out task.",
    )
    parser.add_argument("--stop-on-error", action="store_true")


def validate_execution_args(args: argparse.Namespace) -> None:
    if args.timeout_retries < 0:
        raise ValueError("--timeout-retries must be >= 0")
    if args.retry_backoff_seconds < 0:
        raise ValueError("--retry-backoff-seconds must be >= 0")


def task_split_settings(cfg: Any) -> tuple[str, float, float, int]:
    """Read (split_name, train_fraction, test_fraction, seed) from execution.task_split."""
    split_cfg = getattr(cfg.execution, "task_split", None)
    return (
        str(getattr(split_cfg, "name", "all")),
        float(getattr(split_cfg, "train_fraction", 0.3)),
        float(getattr(split_cfg, "test_fraction", 0.7)),
        getattr(split_cfg, "seed", 42),
    )


def validate_split_fractions(train_fraction: float, test_fraction: float) -> None:
    """Validate that split fractions are in (0, 1) and sum to 1."""
    for name, value in (("train_fraction", train_fraction), ("test_fraction", test_fraction)):
        if not 0.0 < value < 1.0:
            raise ValueError(f"execution.task_split.{name} must be in the open interval (0, 1).")
    if abs(train_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("execution.task_split train_fraction and test_fraction must sum to 1.0.")


def resolve_task_indices(
    args: argparse.Namespace,
    cfg: Any,
) -> tuple[list[int], list[dict], set[str], set[str]]:
    """Discover tasks, apply selection and split filtering.

    Returns:
        (selected_indices, all_tasks, train_task_ids, test_task_ids)
    """
    all_tasks = discover_tasks(args.tasks_root)
    selected_indices = select_task_indices(args, len(all_tasks))
    split_name, train_fraction, test_fraction, seed = task_split_settings(cfg)
    selected_indices, train_ids, test_ids = filter_indices_by_task_split(
        selected_indices,
        all_tasks,
        split_name,
        train_fraction,
        test_fraction,
        seed,
    )

    return selected_indices, all_tasks, train_ids, test_ids
