"""GAIA benchmark runner."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from logger.run_logger import RunSummary, write_run_summary
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(cfg: Any = _CFG) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GAIA benchmark tasks")
    add_task_selection_args(parser, cfg)
    add_execution_args(parser, cfg)

    args = parser.parse_args()
    args.mode = cfg.execution.mode
    validate_split_fractions(*task_split_settings(cfg)[1:3])
    validate_execution_args(args)
    return args


def build_task_execution_config(args: argparse.Namespace, cfg: Any) -> TaskExecutionConfig:
    project_root = Path(__file__).resolve().parent.parent.parent
    return TaskExecutionConfig(
        python_executable=sys.executable,
        task_runner_script=str(project_root / "benchmarks" / "gaia" / "run_task.py"),
        model_name=cfg.llm.model_name,
        mode=args.mode,
        task_timeout_seconds=args.task_timeout_seconds,
        timeout_retries=args.timeout_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config()
    tag = cfg.orchestrator.variant
    split_name = task_split_settings(cfg)[0]

    logger.info(
        "GAIA Benchmark Runner | Model: %s | Tasks: %s | Split: %s | Variant: %s",
        cfg.llm.model_name,
        args.tasks_root,
        split_name,
        cfg.orchestrator.variant,
    )

    selected_indices, all_tasks, train_ids, test_ids = resolve_task_indices(args, cfg)

    logger.info(
        "Running %d tasks (split=%s, train=%d, test=%d)",
        len(selected_indices),
        split_name,
        len(train_ids),
        len(test_ids),
    )

    _run_tasks(args, cfg, selected_indices, all_tasks, tag)


def _run_tasks(
    args: argparse.Namespace,
    cfg: Any,
    selected_indices: list[int],
    all_tasks: list[dict[str, Any]],
    tag: str,
) -> list[dict[str, Any]]:
    task_execution_config = build_task_execution_config(args, cfg)
    records: list[dict[str, Any]] = []

    for rank, idx in enumerate(selected_indices, 1):
        task_meta = all_tasks[idx]
        logger.info("[%d/%d] Running task %s", rank, len(selected_indices), task_meta["task_id"])

        start = time.monotonic()
        record = execute_task_with_retries(
            config=task_execution_config,
            task_key=task_meta["task_key"],
            task_index=idx,
            task_dir=task_meta["task_dir"],
            config_file=task_meta["config_file"],
            tag=tag,
        )
        records.append(record)

        should_stop = _handle_worker_failure(
            record=record,
            task_meta=task_meta,
            variant=cfg.orchestrator.variant,
            tag=tag,
            elapsed=time.monotonic() - start,
            stop_on_error=args.stop_on_error,
        )
        if should_stop:
            break

    return records


def _handle_worker_failure(
    *,
    record: dict[str, Any],
    task_meta: dict[str, Any],
    variant: str,
    tag: str,
    elapsed: float,
    stop_on_error: bool,
) -> bool:
    status = record["status"]
    if status == STATUS_OK:
        return False

    if status == STATUS_TIMEOUT:
        logger.error(
            "Task %s timed out after %d attempt(s); writing sentinel row.",
            task_meta["task_id"],
            record.get("attempts", 1),
        )
        _write_sentinel(
            variant=variant,
            task_meta=task_meta,
            tag=tag,
            elapsed=elapsed,
            termination_reason="timeout",
            error_type="TimeoutError",
        )
    elif status == STATUS_ERROR:
        logger.error("Task %s failed: %s", task_meta["task_id"], record.get("error", "unknown"))
        _write_sentinel(
            variant=variant,
            task_meta=task_meta,
            tag=tag,
            elapsed=elapsed,
            termination_reason="error",
            error_type="WorkerCrash",
        )

    if stop_on_error:
        logger.warning("Stopping early due to --stop-on-error")
        return True
    return False


def _write_sentinel(
    *,
    variant: str,
    task_meta: dict[str, Any],
    tag: str,
    elapsed: float,
    termination_reason: str,
    error_type: str,
) -> None:
    summary = RunSummary(
        created_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        task_id=task_meta["task_id"],
        subtask_id=task_meta.get("subtask_id", "0"),
        task_dir=task_meta["task_dir"],
        tag=tag,
        orchestrator_variant=variant,
        success=-1,
        termination_reason=termination_reason,
        error_type=error_type,
        wall_clock_seconds=round(elapsed, 2),
    )
    write_run_summary(summary, f"results/{variant}.csv")
