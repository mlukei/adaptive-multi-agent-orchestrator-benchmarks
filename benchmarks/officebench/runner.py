"""OfficeBench experiment runner."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

from benchmarks.shared.runner import (
    add_retry_args,
    apply_retry_filters,
    make_retry_loaders,
    validate_retry_args,
)
from runtime.cli import (
    add_task_selection_args,
    resolve_task_indices,
    validate_split_fractions,
)
from runtime.config import load_config
from runtime.experiment_runtime import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_TIMEOUT,
    TaskExecutionConfig,
    execute_task_with_retries,
)


def load_experiment_env() -> None:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
    else:
        load_dotenv(override=False)


load_experiment_env()
_CFG = load_config()


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-")


def parse_args(cfg: Any = _CFG) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OfficeBench benchmark tasks")
    parser.add_argument("--docker-name", default=cfg.docker.image_name)
    parser.add_argument("--container-name", default=cfg.docker.container_name)
    parser.add_argument("--dockerfile-path", default=cfg.docker.dockerfile_path)
    parser.add_argument("--model-name", default=cfg.llm.model_name)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--mode", default=cfg.execution.mode)
    parser.add_argument("--experiment-name", default=cfg.orchestrator.variant)
    add_task_selection_args(parser, cfg)
    add_retry_args(parser, cfg)

    args = parser.parse_args()
    validate_retry_args(args)
    validate_split_fractions(args)
    return args


def build_task_execution_config(args: argparse.Namespace) -> TaskExecutionConfig:
    project_root = Path(__file__).resolve().parent.parent.parent
    return TaskExecutionConfig(
        python_executable=sys.executable,
        task_runner_script=str(project_root / "benchmarks" / "officebench" / "run_task.py"),
        docker_name=args.docker_name,
        dockerfile_path=args.dockerfile_path,
        model_name=args.model_name,
        mode=args.mode,
        container_name_prefix=args.container_name,
        task_timeout_seconds=args.task_timeout_seconds,
        timeout_retries=args.timeout_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
    )


def main() -> None:
    args = parse_args()
    experiment_name = args.experiment_name
    run_tag = args.tag or sanitize_filename(experiment_name)

    selected_indices, all_tasks, train_ids, test_ids = _select_tasks(args, experiment_name)
    if selected_indices is None:
        return

    print(f"Starting local experiment '{experiment_name}'")
    _log_split_summary(args, selected_indices, all_tasks, train_ids, test_ids)

    _run_tasks(args, selected_indices, all_tasks, run_tag)


def _select_tasks(
    args: argparse.Namespace,
    experiment_name: str,
) -> tuple[list[int] | None, list[dict[str, Any]], set[str], set[str]]:
    selected_indices, all_tasks, train_ids, test_ids = resolve_task_indices(args)
    selected_indices = apply_retry_filters(
        args,
        selected_indices,
        all_tasks,
        task_key=_task_key,
        loaders=make_retry_loaders(experiment_name, _row_key),
        log=print,
    )
    return selected_indices, all_tasks, train_ids, test_ids


def _run_tasks(
    args: argparse.Namespace,
    selected_indices: list[int],
    all_tasks: list[dict[str, Any]],
    run_tag: str,
) -> list[dict[str, Any]]:
    task_execution_config = build_task_execution_config(args)
    records: list[dict[str, Any]] = []

    for rank, idx in enumerate(selected_indices, 1):
        task = all_tasks[idx]
        print(f"[{rank}/{len(selected_indices)}] task[{idx}] {task['task_key']}")
        record = execute_task_with_retries(
            config=task_execution_config,
            task_key=task["task_key"],
            task_index=idx,
            task_dir=task["task_dir"],
            config_file=task["config_file"],
            tag=run_tag,
        )
        records.append(record)
        _handle_failed_record(record, stop_on_error=args.stop_on_error)

    return records


def _handle_failed_record(record: dict[str, Any], *, stop_on_error: bool) -> None:
    status = record["status"]
    if status == STATUS_OK:
        return
    if status == STATUS_TIMEOUT:
        print(f"Task timed out after {record['attempts']} attempt(s): {record['task_key']}")
        if stop_on_error:
            raise TimeoutError(str(record.get("error", "Task timed out.")))
    elif status == STATUS_ERROR:
        print(f"Task failed: {record['task_key']} -> {record.get('error', 'Unknown task error')}")
        if stop_on_error:
            raise RuntimeError(str(record.get("error", "Task failed.")))


def _log_split_summary(
    args: argparse.Namespace,
    selected_indices: list[int],
    all_tasks: list[dict[str, Any]],
    train_ids: set[str],
    test_ids: set[str],
) -> None:
    if args.task_split == "all":
        return

    selected_ids = {str(all_tasks[idx]["task_id"]) for idx in selected_indices}
    print(
        "Task split active: "
        f"split={args.task_split}, "
        f"train_fraction={args.train_fraction:.2f}, "
        f"test_fraction={args.test_fraction:.2f}, "
        f"split_seed={args.split_seed}, "
        f"train_task_ids={len(train_ids)}, "
        f"test_task_ids={len(test_ids)}, "
        f"selected_runs={len(selected_indices)}"
    )
    print(
        "Task split difficulty balance: "
        f"train[{_tier_counts(train_ids)}], "
        f"test[{_tier_counts(test_ids)}], "
        f"selected[{_tier_counts(selected_ids)}]"
    )


def _tier_counts(task_ids: set[str]) -> str:
    counts = Counter(str(task_id).split("-", 1)[0] for task_id in task_ids)
    return ", ".join(f"{tier}:{counts[tier]}" for tier in sorted(counts))




def _row_key(row: dict[str, str]) -> str:
    task_id = row.get("task_id", "")
    return f"{task_id}/{row.get('subtask_id', '0')}" if task_id else ""


def _task_key(task_entry: dict[str, Any]) -> str:
    return f"{task_entry['task_id']}/{task_entry['subtask_id']}"
