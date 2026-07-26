"""CSV run summaries and policy logging helpers."""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runtime.usage_tracking import compute_cost_from_usage, compute_embedding_cost

logger = logging.getLogger(__name__)


@dataclass
class RunSummary:
    """Single benchmark run: identification, result, and metrics."""

    run_id: str = ""
    created_at_utc: str = ""
    task_id: str = ""
    subtask_id: str = ""
    task_dir: str = ""
    tag: str = ""
    orchestrator_variant: str = "baseline"

    success: int = -1
    termination_reason: str = ""
    error_type: str = ""
    judge_accepted: int = -1
    judge_force_accepted: int = -1
    wall_clock_seconds: float = 0.0

    plan_created: int = -1
    total_delegations: int = 0
    submission_attempts: int = 0
    judge_rejections: int = 0

    blueprint_matched: int = -1
    blueprint_id: str = ""
    blueprint_similarity: float = 0.0
    agent_curations_run: int = 0
    episode_curation_run: int = -1
    episode_curation_skipped: int = -1
    episode_familiarity: float = -1.0

    executed_agents: str = "[]"
    optimal_sequence: str = "[]"
    capability_boosted_agents: int = 0
    agent_discovery_sources: str = "{}"
    agent_retrieval_log: str = "[]"

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    embedding_tokens: int = 0
    cost_llm: float = 0.0
    cost_embedding: float = 0.0
    cost_total: float = 0.0
    cost_by_model: str = "{}"

    playbook_confirm_votes: int = 0
    playbook_contradict_votes: int = 0
    playbook_bullets_added: int = 0
    playbook_bullets_pruned: int = 0
    playbook_unconfirmed_prunes: int = 0
    playbook_harm_prunes: int = 0
    playbook_consolidation_merges: int = 0
    playbook_evolution: str = "[]"

    num_internal_tool_calls: int = 0
    tool_call_history: str = "[]"


def write_run_summary(summary: RunSummary, path: str | None) -> None:
    """Append a summary while preserving the schema of an existing CSV."""
    if not path:
        return

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [field.name for field in fields(RunSummary)]
    write_header = _resolve_header(output_path, header)
    needs_header = not output_path.exists() or output_path.stat().st_size == 0
    row = {
        key: _single_line(value)
        for key, value in asdict(summary).items()
        if key in write_header
    }

    with output_path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=write_header, quoting=csv.QUOTE_ALL)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def _resolve_header(output_path: Path, header: list[str]) -> list[str]:
    if not output_path.exists():
        return header

    with output_path.open(newline="", encoding="utf-8") as existing:
        existing_header = next(csv.reader(existing), [])
    if not existing_header or existing_header == header:
        return header

    unknown_fields = [name for name in existing_header if name not in header]
    if unknown_fields:
        raise RuntimeError(
            f"Existing CSV schema at {output_path} contains unexpected fields: "
            f"{unknown_fields}. Migrate or replace the file before logging."
        )

    missing_fields = [name for name in header if name not in existing_header]
    logger.warning(
        "Existing CSV schema at %s is missing %d current field(s): %s. "
        "Appending with the existing header.",
        output_path,
        len(missing_fields),
        ", ".join(missing_fields),
    )
    return existing_header


def _single_line(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\n", "\\n").replace("\r", "\\r")
    return value


def _optional_bool(value: bool | None) -> int:
    return {True: 1, False: 0}.get(value, -1)


def _json(value: Any, empty: list[Any] | dict[str, Any]) -> str:
    return json.dumps(value or empty, ensure_ascii=False)


class RunLogger:
    """Build and write one summary for a benchmark task."""

    def __init__(
        self,
        *,
        path: str | None,
        pricing_cfg: Any,
        task_config: dict[str, Any],
        tag: str | None,
        orchestrator_variant: str,
        embedder_model: str = "text-embedding-ada-002",
    ) -> None:
        self.path = path
        self.pricing_cfg = pricing_cfg
        self.task_config = task_config
        self.tag = tag
        self.orchestrator_variant = orchestrator_variant
        self.embedder_model = embedder_model

    def log_last(
        self,
        *,
        context: dict[str, Any] | None,
        eval_result: bool | None,
    ) -> None:
        if context is None:
            logger.warning("No run context available; skipping summary.")
            return
        self.log(context=context, eval_result=eval_result)

    def log(
        self,
        *,
        context: dict[str, Any],
        eval_result: bool | None,
    ) -> None:
        episode = context.get("episode_metrics") or {}
        task_dir = self.task_config.get("task_dir", "")
        internal_tool_calls = context.get("internal_tool_calls") or []
        tool_names = [
            getattr(tool_call, "tool_name", str(tool_call))
            for tool_call in internal_tool_calls
        ]

        llm_cost, cost_by_model = compute_cost_from_usage(
            episode.get("token_usage_by_model", {}),
            self.pricing_cfg,
        )
        embedding_tokens = episode.get("embedding_tokens", 0) or 0
        embedding_cost = compute_embedding_cost(
            embedding_tokens,
            self.embedder_model,
            self.pricing_cfg,
        )

        summary = RunSummary(
            run_id=context["run_id"],
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            task_id=Path(task_dir).name if task_dir else "",
            subtask_id=self.task_config.get("subtask_id", "0"),
            task_dir=task_dir,
            tag=self.tag or "",
            orchestrator_variant=self.orchestrator_variant,
            success=_optional_bool(eval_result),
            termination_reason=context.get("termination_reason", ""),
            error_type=context.get("error_type", ""),
            judge_accepted=_optional_bool(episode.get("judge_accepted")),
            judge_force_accepted=1 if episode.get("judge_force_accepted") else 0,
            wall_clock_seconds=round(context.get("duration", 0.0), 2),
            plan_created=_optional_bool(episode.get("plan_created")),
            total_delegations=episode.get("total_delegations", 0),
            submission_attempts=episode.get("submission_attempts", 0),
            judge_rejections=episode.get("judge_rejections", 0),
            blueprint_matched=_optional_bool(episode.get("blueprint_matched")),
            blueprint_id=episode.get("blueprint_id", ""),
            blueprint_similarity=float(episode.get("blueprint_similarity") or 0.0),
            agent_curations_run=episode.get("agent_curations_run", 0),
            episode_curation_run=_optional_bool(episode.get("episode_curation_run")),
            episode_curation_skipped=_optional_bool(
                episode.get("episode_curation_skipped")
            ),
            episode_familiarity=(
                float(episode["episode_familiarity"])
                if episode.get("episode_familiarity") is not None
                else -1.0
            ),
            executed_agents=_json(episode.get("executed_agents"), []),
            optimal_sequence=_json(episode.get("optimal_sequence"), []),
            capability_boosted_agents=episode.get("capability_boosted_agents", 0),
            agent_discovery_sources=_json(
                episode.get("agent_discovery_sources"), {}
            ),
            agent_retrieval_log=_json(episode.get("agent_retrieval_log"), []),
            total_input_tokens=episode.get("total_input_tokens", 0),
            total_output_tokens=episode.get("total_output_tokens", 0),
            total_tokens=episode.get("total_tokens", 0),
            embedding_tokens=embedding_tokens,
            cost_llm=llm_cost,
            cost_embedding=embedding_cost,
            cost_total=round(llm_cost + embedding_cost, 6),
            cost_by_model=_json(cost_by_model, {}),
            playbook_confirm_votes=episode.get("playbook_confirm_votes", 0),
            playbook_contradict_votes=episode.get("playbook_contradict_votes", 0),
            playbook_bullets_added=episode.get("playbook_bullets_added", 0),
            playbook_bullets_pruned=episode.get("playbook_bullets_pruned", 0),
            playbook_unconfirmed_prunes=episode.get(
                "playbook_unconfirmed_prunes", 0
            ),
            playbook_harm_prunes=episode.get("playbook_harm_prunes", 0),
            playbook_consolidation_merges=episode.get(
                "playbook_consolidation_merges", 0
            ),
            playbook_evolution=_json(episode.get("playbook_evolution"), []),
            num_internal_tool_calls=len(internal_tool_calls),
            tool_call_history=_json(tool_names, []),
        )
        write_run_summary(summary, self.path)

        logger.info(
            "Run logged: %s - success=%d, delegations=%d, familiarity=%.2f, %.1fs",
            summary.task_id,
            summary.success,
            summary.total_delegations,
            summary.episode_familiarity,
            summary.wall_clock_seconds,
        )


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
    """Collect the policy state needed for a run summary."""
    return {
        "internal_tool_calls": _internal_tool_calls(
            orchestrator=orchestrator,
            timeline=timeline,
            thread_id=thread_id,
        ),
        "episode_metrics": _episode_metrics(orchestrator),
        "run_id": run_id,
        "termination_reason": termination,
        "error_type": error_type,
        "duration": time.monotonic() - start,
    }


def _episode_metrics(orchestrator: Any) -> dict[str, Any]:
    metrics = getattr(orchestrator, "last_metrics", None)
    if metrics is None:
        return {}
    return {
        field.name: getattr(metrics, field.name)
        for field in fields(metrics)
        if not field.name.startswith("_")
    }


def _internal_tool_calls(
    *,
    orchestrator: Any,
    timeline: list[Any],
    thread_id: str,
) -> list[Any]:
    timeline_tool_calls = [
        record for record in timeline if hasattr(record, "tool_name")
    ]
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
