"""Shared CLI argument definitions and task-execution logic.

Used by both benchmark runners.
"""

from __future__ import annotations

import argparse
import random
from typing import Any

from runtime.task_sampling import (
    discover_tasks,
    filter_indices_by_task_split,
    select_task_indices,
)


def add_task_selection_args(parser: argparse.ArgumentParser, cfg: Any) -> None:
    """Add shared task-selection and task-split arguments to parser.

    Args:
        parser: ArgumentParser to augment.
        cfg: Loaded YAML config object
    """
    split_cfg = getattr(cfg.execution, "task_split", None)

    # Task selection
    parser.add_argument("--tasks-root", default=cfg.execution.tasks_root)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1, help="Run all tasks if -1")
    parser.add_argument(
        "--task-ids", type=int, nargs="+",
        help="Optional explicit task indices in discovered task list",
    )
    parser.add_argument("--shuffle", action="store_true")

    # Task split
    default_split_name = str(getattr(split_cfg, "name", "all"))
    default_train_fraction = float(getattr(split_cfg, "train_fraction", 0.3))
    default_test_fraction = float(getattr(split_cfg, "test_fraction", 0.7))
    default_split_seed = getattr(split_cfg, "seed", 42)

    parser.add_argument(
        "--task-split",
        choices=["all", "train", "test"],
        default=default_split_name,
    )
    parser.add_argument("--train-fraction", type=float, default=default_train_fraction)
    parser.add_argument("--test-fraction", type=float, default=default_test_fraction)
    parser.add_argument("--split-seed", type=int, default=default_split_seed)


def validate_split_fractions(args: argparse.Namespace) -> None:
    """Validate that split fractions are in (0,1) and sum to 1.0.

    Raises ValueError on invalid input.
    """
    for flag_name, value in (
        ("--train-fraction", args.train_fraction),
        ("--test-fraction", args.test_fraction),
    ):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{flag_name} must be in the open interval (0, 1).")
    total = args.train_fraction + args.test_fraction
    if abs(total - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0.")


def resolve_task_indices(
    args: argparse.Namespace,
) -> tuple[list[int], list[dict], set[str], set[str]]:
    """Discover tasks, apply selection and split filtering.

    Returns:
        (selected_indices, all_tasks, train_task_ids, test_task_ids)
    """
    # Discover all tasks from disk.
    all_tasks = discover_tasks(args.tasks_root)

    # Apply index-based selection filters to get list of task ids (start/end/task-ids).
    selected_indices = select_task_indices(args, len(all_tasks))

    # get train/test split of task_ids, then filter selected_indices to those sets
    selected_indices, train_ids, test_ids = filter_indices_by_task_split(
        selected_indices,
        all_tasks,
        args.task_split,
        args.train_fraction,
        args.test_fraction,
        args.split_seed,
    )

    if args.shuffle:
        random.Random(args.split_seed).shuffle(selected_indices)

    return selected_indices, all_tasks, train_ids, test_ids
