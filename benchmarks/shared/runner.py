"""Shared runner for the OfficeBench and GAIA benchmark runners.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from typing import Any

from benchmarks.shared.result_filters import load_result_keys

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_retry_args(parser: argparse.ArgumentParser, cfg: Any) -> None:
    """Add the timeout/retry/--retry-* flags shared by both runners."""
    ex = cfg.execution
    parser.add_argument(
        "--task-timeout-seconds", type=int, default=ex.task_timeout_seconds,
        help="Per-task hard timeout in seconds (OS-level kill). Set <=0 to disable.",
    )
    parser.add_argument(
        "--timeout-retries", type=int, default=getattr(ex, "timeout_retries", 0),
        help="Retries for timed-out tasks (0 = no retry).",
    )
    parser.add_argument(
        "--retry-backoff-seconds", type=int, default=getattr(ex, "retry_backoff_seconds", 0),
        help="Delay before retrying a timed-out task.",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--retry-timeouts", action="store_true",
        help="Re-run only tasks that timed out in the last run of this variant "
             "(reads results/{variant}.csv; implies --mode force_new).",
    )
    parser.add_argument(
        "--retry-errors", action="store_true",
        help="Re-run only tasks that errored in the last run of this variant.",
    )
    parser.add_argument(
        "--retry-missing", action="store_true",
        help="Run only tasks with no row yet in results/{variant}.csv.",
    )


def validate_retry_args(args: argparse.Namespace) -> None:
    if args.timeout_retries < 0:
        raise ValueError("--timeout-retries must be >= 0")
    if args.retry_backoff_seconds < 0:
        raise ValueError("--retry-backoff-seconds must be >= 0")
    
# ---------------------------------------------------------------------------
# --retry-* task filtering
# ---------------------------------------------------------------------------

def make_retry_loaders(
    variant: str,
    row_key: Callable[[dict[str, str]], str],
    *,
    error_checks_success: bool = True,
) -> tuple[Callable[[], set[str]], Callable[[], set[str]], Callable[[], set[str]]]:
    """Return (load_existing, load_timeouts, load_errors) loaders for a variant.

    ``row_key`` maps a CSV row to the task key used for matching. The timeout
    filter always requires a sentinel ``success`` of -1; the error filter does
    too when *error_checks_success* (the default), matching OfficeBench, but
    GAIA disables it because its policy-level errors log ``success=0``.
    """
    def load_existing() -> set[str]:
        return load_result_keys(
            variant, key_of=row_key, keep=lambda row: True,
            label="--retry-missing", warn_if_missing=False,
        )

    def load_timeouts() -> set[str]:
        return load_result_keys(
            variant, key_of=row_key,
            keep=lambda row: row.get("termination_reason") == "timeout"
            and row.get("success") in ("-1", "-1.0"),
            label="--retry-timeouts", warn_if_missing=True,
        )

    def load_errors() -> set[str]:
        def keep(row: dict[str, str]) -> bool:
            if row.get("termination_reason") != "error":
                return False
            return not error_checks_success or row.get("success") in ("-1", "-1.0")
        return load_result_keys(
            variant, key_of=row_key, keep=keep,
            label="--retry-errors", warn_if_missing=True,
        )

    return load_existing, load_timeouts, load_errors


def apply_retry_filters(
    args: argparse.Namespace,
    selected_indices: list[int],
    all_tasks: list[dict[str, Any]],
    *,
    task_key: Callable[[dict[str, Any]], str],
    loaders: tuple[Callable[[], set[str]], Callable[[], set[str]], Callable[[], set[str]]],
    on_selected: Callable[[list[int], str], None] | None = None,
    log: Callable[[str], None] = logger.info,
) -> list[int] | None:
    """Apply the active --retry-* flag to *selected_indices*.

    Returns the filtered index list, or None to signal "nothing to run" (the
    caller should return). ``task_key`` maps a discovered task entry to the same
    key space as the loaders' CSV rows. ``on_selected(indices, label)`` is an
    optional hook (GAIA uses it to purge stale output dirs); each applied filter
    forces ``--mode force_new``.
    """
    load_existing, load_timeouts, load_errors = loaders

    def keep_indices(keys: set[str], *, include: bool) -> list[int]:
        return [
            idx for idx in selected_indices
            if (task_key(all_tasks[idx]) in keys) == include
        ]

    if args.retry_missing:
        before = len(selected_indices)
        selected_indices = keep_indices(load_existing(), include=False)
        log(f"--retry-missing: {before - len(selected_indices)} already done, "
            f"{len(selected_indices)} remaining to run")
        if not selected_indices:
            log("--retry-missing: all tasks complete — nothing to run.")
            return None
        if on_selected:
            on_selected(selected_indices, "--retry-missing")
        args.mode = "force_new"

    for active, load_keys, label in (
        (args.retry_timeouts, load_timeouts, "--retry-timeouts"),
        (args.retry_errors, load_errors, "--retry-errors"),
    ):
        if not active:
            continue
        keys = load_keys()
        if not keys:
            log(f"{label}: no matching tasks found — nothing to run.")
            return None
        selected_indices = keep_indices(keys, include=True)
        log(f"{label}: restricted to {len(selected_indices)} task(s) (mode forced to force_new)")
        if on_selected:
            on_selected(selected_indices, label)
        args.mode = "force_new"

    return selected_indices
