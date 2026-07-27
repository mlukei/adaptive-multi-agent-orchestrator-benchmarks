"""OfficeBench experiment runner."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from logger.run_logger import write_sentinel_summary
from runtime.cli import (
    add_execution_args,
    add_task_selection_args,
    resolve_task_indices,
    task_split_settings,
    validate_execution_args,
    validate_split_fractions,
)
from runtime.config import load_config, load_experiment_env
from runtime.experiment_runtime import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_TIMEOUT,
    TaskExecutionConfig,
    execute_task_with_retries,
)


load_experiment_env()
_CFG = load_config()


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-")


def parse_args(cfg: Any = _CFG) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OfficeBench benchmark tasks")
    parser.add_argument("--experiment-name", default=cfg.orchestrator.variant)
    add_task_selection_args(parser, cfg)
    add_execution_args(parser, cfg)

    args = parser.parse_args()
    args.mode = cfg.execution.mode
    validate_execution_args(args)
    validate_split_fractions(*task_split_settings(cfg)[1:3])
    return args


def build_task_execution_config(args: argparse.Namespace, cfg: Any) -> TaskExecutionConfig:
    project_root = Path(__file__).resolve().parent.parent.parent
    return TaskExecutionConfig(
        python_executable=sys.executable,
        worker_script=str(project_root / "benchmarks" / "officebench" / "worker.py"),
        docker_name=cfg.docker.image_name,
        dockerfile_path=cfg.docker.dockerfile_path,
        model_name=cfg.llm.model_name,
        mode=args.mode,
        container_name_prefix=cfg.docker.container_name,
        task_timeout_seconds=args.task_timeout_seconds,
        timeout_retries=args.timeout_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config()
    experiment_name = args.experiment_name
    run_tag = sanitize_filename(experiment_name)

    selected_indices, all_tasks, train_ids, test_ids = resolve_task_indices(args, cfg)

    print(f"Starting local experiment '{experiment_name}'")
    _log_split_summary(cfg, selected_indices, all_tasks, train_ids, test_ids)

    _run_tasks(args, cfg, selected_indices, all_tasks, run_tag)


def _run_tasks(
    args: argparse.Namespace,
    cfg: Any,
    selected_indices: list[int],
    all_tasks: list[dict[str, Any]],
    run_tag: str,
) -> list[dict[str, Any]]:
    task_execution_config = build_task_execution_config(args, cfg)
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
        _handle_failed_record(
            record,
            task_meta=task,
            variant=cfg.orchestrator.variant,
            tag=run_tag,
            stop_on_error=args.stop_on_error,
        )

    return records


def _handle_failed_record(
    record: dict[str, Any],
    *,
    task_meta: dict[str, Any],
    variant: str,
    tag: str,
    stop_on_error: bool,
) -> None:
    status = record["status"]
    if status == STATUS_OK:
        return

    if status == STATUS_TIMEOUT:
        print(f"Task timed out after {record['attempts']} attempt(s): {record['task_key']}")
        termination_reason, error_type = "timeout", "TimeoutError"
    elif status == STATUS_ERROR:
        print(f"Task failed: {record['task_key']} -> {record.get('error', 'Unknown task error')}")
        termination_reason, error_type = "error", "WorkerCrash"
    else:
        return
    write_sentinel_summary(
        path=f"results/{variant}.csv",
        task_id=task_meta["task_id"],
        subtask_id=task_meta.get("subtask_id", "0"),
        task_dir=task_meta["task_dir"],
        tag=tag,
        orchestrator_variant=variant,
        elapsed_seconds=record.get("duration_seconds") or 0.0,
        termination_reason=termination_reason,
        error_type=error_type,
    )

    if stop_on_error:
        if status == STATUS_TIMEOUT:
            raise TimeoutError(str(record.get("error", "Task timed out.")))
        raise RuntimeError(str(record.get("error", "Task failed.")))


def _log_split_summary(
    cfg: Any,
    selected_indices: list[int],
    all_tasks: list[dict[str, Any]],
    train_ids: set[str],
    test_ids: set[str],
) -> None:
    split_name, train_fraction, test_fraction, split_seed = task_split_settings(cfg)
    if split_name == "all":
        return

    selected_ids = {str(all_tasks[idx]["task_id"]) for idx in selected_indices}
    print(
        "Task split active: "
        f"split={split_name}, "
        f"train_fraction={train_fraction:.2f}, "
        f"test_fraction={test_fraction:.2f}, "
        f"split_seed={split_seed}, "
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
