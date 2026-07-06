"""Policy run-context helpers used before writing run summaries."""

from __future__ import annotations

import logging
import time
from dataclasses import fields as dataclass_fields
from typing import Any

logger = logging.getLogger(__name__)


def get_episode_metrics(orchestrator: Any) -> dict[str, Any]:
    """Return public fields from the orchestrator's last metrics dataclass."""
    if orchestrator is None or orchestrator.last_metrics is None:
        return {}
    return {
        field.name: getattr(orchestrator.last_metrics, field.name)
        for field in dataclass_fields(orchestrator.last_metrics)
        if not field.name.startswith("_")
    }


def get_internal_tool_calls(
    *,
    orchestrator: Any,
    timeline: list[Any],
    thread_id: str,
) -> list[Any]:
    """Return internal tool calls from the saved timeline or call tracker."""
    timeline_tool_calls = [record for record in timeline if hasattr(record, "tool_name")]
    if timeline_tool_calls:
        return timeline_tool_calls

    if orchestrator is None:
        return []

    call_tracker = (
        getattr(orchestrator, "call_tracker", None)
        or getattr(orchestrator, "_call_tracker", None)
    )
    if call_tracker is None:
        return []
    return call_tracker.get_tool_calls(thread_id=thread_id)


def collect_run_context(
    *,
    orchestrator: Any,
    timeline: list[Any],
    thread_id: str,
    run_id: str,
    termination: str,
    error_type: str,
    start: float,
) -> dict[str, Any]:
    """Build the context consumed by RunLogger."""
    return {
        "internal_tool_calls": get_internal_tool_calls(
            orchestrator=orchestrator,
            timeline=timeline,
            thread_id=thread_id,
        ),
        "episode_metrics": get_episode_metrics(orchestrator),
        "run_id": run_id,
        "termination_reason": termination,
        "error_type": error_type,
        "duration": time.monotonic() - start,
    }


def log_last_run_summary(
    run_logger: Any,
    *,
    last_run_context: dict[str, Any] | None,
    eval_result: bool | None,
) -> None:
    """Write the last completed run summary, if a run context exists."""
    if last_run_context is None:
        logger.warning("No run context available; skipping summary.")
        return
    run_logger.log(ctx=last_run_context, eval_result=eval_result)
