"""Unified run logger: writes one run summary per task to CSV."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from logger.local_logger import CsvLogger, RunSummary, serialize_optional_bool
from runtime.usage_tracking import compute_cost_from_usage, compute_embedding_cost

_logger = logging.getLogger(__name__)


class RunLogger:
    """Writes one run summary to CSV.

    Owns the CsvLogger so that benchmark policies only need to call
    ``log(ctx, eval_result)``.
    """

    def __init__(
        self,
        *,
        csv_logger: CsvLogger,
        pricing_cfg: Any,
        task_config: dict[str, Any],
        tag: str | None,
        orchestrator_variant: str,
        embedder_model: str = "text-embedding-ada-002",
    ) -> None:
        self.csv_logger = csv_logger
        self.pricing_cfg = pricing_cfg
        self.task_config = task_config
        self.tag = tag
        self.orchestrator_variant = orchestrator_variant
        self.embedder_model = embedder_model

    def log(
        self,
        *,
        ctx: dict[str, Any],
        eval_result: bool | None,
    ) -> None:
        """Build a RunSummary and write it to CSV.

        Args:
            ctx: Run context dict produced by a benchmark policy run –
                 keys: run_id, termination_reason, error_type, duration,
                       internal_tool_calls, episode_metrics.
            eval_result: True/False evaluation outcome, or None if unavailable.
        """
        ep = ctx.get("episode_metrics") or {}

        task_dir = self.task_config.get("task_dir", "")
        internal_tool_calls = ctx.get("internal_tool_calls") or []
        tool_names = [getattr(c, "tool_name", str(c)) for c in internal_tool_calls]

        cost, cost_by_model = compute_cost_from_usage(
            ep.get("token_usage_by_model", {}),
            self.pricing_cfg,
        )
        embedding_tokens = ep.get("embedding_tokens", 0) or 0
        embedding_cost = compute_embedding_cost(
            embedding_tokens,
            self.embedder_model,
            self.pricing_cfg,
        )

        summary = RunSummary(
            run_id=ctx["run_id"],
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            task_id=task_dir.split("/")[-1] if task_dir else "",
            subtask_id=self.task_config.get("subtask_id", "0"),
            task_dir=task_dir,
            tag=self.tag or "",
            orchestrator_variant=self.orchestrator_variant,
            success=serialize_optional_bool(eval_result),
            termination_reason=ctx.get("termination_reason", ""),
            error_type=ctx.get("error_type", ""),
            judge_accepted=serialize_optional_bool(ep.get("judge_accepted")),
            judge_force_accepted=1 if ep.get("judge_force_accepted") else 0,
            wall_clock_seconds=round(ctx.get("duration", 0.0), 2),
            plan_created=serialize_optional_bool(ep.get("plan_created")),
            total_delegations=ep.get("total_delegations", 0),
            submission_attempts=ep.get("submission_attempts", 0),
            judge_rejections=ep.get("judge_rejections", 0),
            blueprint_matched=serialize_optional_bool(ep.get("blueprint_matched")),
            blueprint_id=ep.get("blueprint_id", ""),
            blueprint_similarity=float(ep.get("blueprint_similarity") or 0.0),
            agent_curations_run=ep.get("agent_curations_run", 0),
            episode_curation_run=serialize_optional_bool(ep.get("episode_curation_run")),
            episode_curation_skipped=serialize_optional_bool(ep.get("episode_curation_skipped")),
            episode_familiarity=float(ep.get("episode_familiarity")) if ep.get("episode_familiarity") is not None else -1.0,
            executed_agents=json.dumps(ep.get("executed_agents") or [], ensure_ascii=False),
            optimal_sequence=json.dumps(ep.get("optimal_sequence") or [], ensure_ascii=False),
            capability_boosted_agents=ep.get("capability_boosted_agents", 0),
            agent_discovery_sources=json.dumps(ep.get("agent_discovery_sources") or {}, ensure_ascii=False),
            agent_retrieval_log=json.dumps(ep.get("agent_retrieval_log") or [], ensure_ascii=False),
            total_input_tokens=ep.get("total_input_tokens", 0),
            total_output_tokens=ep.get("total_output_tokens", 0),
            total_tokens=ep.get("total_tokens", 0),
            embedding_tokens=embedding_tokens,
            cost_llm=cost,
            cost_embedding=embedding_cost,
            cost_total=round(cost + embedding_cost, 6),
            cost_by_model=json.dumps(cost_by_model, ensure_ascii=False),
            playbook_confirm_votes=ep.get("playbook_confirm_votes", 0),
            playbook_contradict_votes=ep.get("playbook_contradict_votes", 0),
            playbook_bullets_added=ep.get("playbook_bullets_added", 0),
            playbook_bullets_pruned=ep.get("playbook_bullets_pruned", 0),
            playbook_unconfirmed_prunes=ep.get("playbook_unconfirmed_prunes", 0),
            playbook_harm_prunes=ep.get("playbook_harm_prunes", 0),
            playbook_consolidation_merges=ep.get("playbook_consolidation_merges", 0),
            playbook_evolution=json.dumps(ep.get("playbook_evolution") or [], ensure_ascii=False),
            num_internal_tool_calls=len(internal_tool_calls),
            tool_call_history=json.dumps(tool_names, ensure_ascii=False),
        )

        self.csv_logger.log(summary)

        _logger.info(
            "Run logged: %s - success=%d, delegations=%d, familiarity=%.2f, %.1fs",
            summary.task_id,
            summary.success,
            summary.total_delegations,
            summary.episode_familiarity,
            summary.wall_clock_seconds,
        )
