import argparse
import json
from pathlib import Path
import random
import re
from typing import Any


def natural_sort_key(text: str) -> list[int | str]:
    """Sort key that orders mixed numeric strings.

    Example:
        sorted(["1-2", "1-10", "1-1"], key=natural_sort_key)
        # returns ["1-1", "1-2", "1-10"]
    """
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(text))]


def select_task_indices(
    args: argparse.Namespace,
    total_tasks: int,
) -> list[int]:
    """Resolve the list of task indices to run.

    Either an explicit ``--task-ids`` list or a ``--start-index``/``--end-index``
    range, clamped to the available tasks.
    """
    if args.task_ids:
        requested = list(args.task_ids)
    else:
        end_index = total_tasks if args.end_index == -1 else min(args.end_index, total_tasks)
        requested = list(range(args.start_index, end_index))

    selected = [i for i in requested if 0 <= i < total_tasks]
    if not selected:
        raise RuntimeError("No valid task indices selected.")
    return selected


def discover_tasks(tasks_root: str) -> list[dict[str, Any]]:
    """Discover all task JSON files and load metadata."""
    root = Path(tasks_root)
    if not root.exists():
        raise FileNotFoundError(f"Tasks root not found: {tasks_root}")

    discovered: list[dict[str, Any]] = []
    task_dirs = [p for p in root.iterdir() if p.is_dir()]
    task_dirs.sort(key=lambda p: natural_sort_key(p.name))

    for task_dir in task_dirs: # officebench runs with subtasks
        subtasks_dir = task_dir / "subtasks"
        if not subtasks_dir.exists():
            continue

        subtask_files = list(subtasks_dir.glob("*.json"))
        subtask_files.sort(key=lambda p: natural_sort_key(p.stem)) # sort by subtask id 
        for subtask_file in subtask_files:
            with subtask_file.open("r", encoding="utf-8") as f:
                cfg = json.load(f) # load the subtask config

            subtask_id = subtask_file.stem
            task_id = task_dir.name
            discovered.append(
                {
                    "task_id": task_id,
                    "subtask_id": subtask_id,
                    "task_key": f"{task_id}/{subtask_id}",
                    "task_dir": str(task_dir),
                    "config_file": str(subtask_file),
                    "config": cfg,
                }
            )

    return discovered


def task_tier_from_id(task_id: str) -> str:
    """Extract the difficulty-tier prefix used to stratify the split.

    OfficeBench:  '2-31'        -> '2'   (first segment is numeric)
    GAIA:         'gaia-1-0004' -> 'gaia-1' (first segment non-numeric,
                                             use first two segments)
    """
    parts = str(task_id).split("-")
    if parts[0].isdigit(): # OfficeBench style
        return parts[0]
    if len(parts) >= 2: # GAIA style
        return f"{parts[0]}-{parts[1]}"
    return parts[0]


def _allocate_split_counts(
    total: int,
    train_fraction: float,
    test_fraction: float,
) -> tuple[int, int]:
    """Allocate split counts for a stratum while preserving total."""
    raw_train = total * train_fraction
    raw_test = total * test_fraction

    train_count = int(raw_train)
    test_count = int(raw_test)

    remainder = total - (train_count + test_count)
    if remainder > 0:
        # Give remainder to whichever fraction has the larger fractional part
        if (raw_train - train_count) >= (raw_test - test_count):
            train_count += remainder
        else:
            test_count += remainder

    return train_count, test_count


def split_task_ids(
    all_tasks: list[dict[str, Any]],
    train_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """Split task_ids into deterministic stratified train/test sets.

    Stratification key is task_id prefix (difficulty tier): 1, 2, 3.
    """
    unique_task_ids = sorted(
        {str(task["task_id"]) for task in all_tasks},
        key=natural_sort_key,
    )
    if not unique_task_ids:
        return set(), set()

    rng = random.Random(seed)
    tier_to_task_ids: dict[str, list[str]] = {}

    # Group task_ids by tier prefix.
    for task_id in unique_task_ids:
        tier = task_tier_from_id(task_id)
        tier_to_task_ids.setdefault(tier, []).append(task_id)

    train_ids: set[str] = set()
    test_ids: set[str] = set()

    # For each tier, shuffle and split task_ids according to the fractions
    for tier in sorted(tier_to_task_ids.keys(), key=natural_sort_key):
        tier_task_ids = tier_to_task_ids[tier][:]
        rng.shuffle(tier_task_ids)
        train_count, test_count = _allocate_split_counts(
            len(tier_task_ids),
            train_fraction,
            test_fraction,
        )
        train_ids.update(tier_task_ids[:train_count])
        test_ids.update(tier_task_ids[train_count:train_count + test_count])

    return train_ids, test_ids


def filter_indices_by_task_split(
    selected_indices: list[int],
    all_tasks: list[dict[str, Any]],
    split: str,
    train_fraction: float,
    test_fraction: float,
    split_seed: int,
) -> tuple[list[int], set[str], set[str]]:
    """Apply the full-benchmark split to selected indices."""
    train_ids, test_ids = split_task_ids(
        all_tasks,
        train_fraction,
        test_fraction,
        split_seed,
    )

    if split == "all":
        return selected_indices, train_ids, test_ids

    allow_ids = {
        "train": train_ids,
        "test": test_ids,
    }[split]
    filtered = [
        idx
        for idx in selected_indices
        if str(all_tasks[idx]["task_id"]) in allow_ids
    ]
    return filtered, train_ids, test_ids
