"""Factory helpers for constructing the shared AdaptiveOrchestrator."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from orchestrator import AdaptiveOrchestrator
from orchestrator.memory import create_pg_stores
from orchestrator.registration.registry import AgentRegistry
from orchestrator.shared.embedder import Embedder
from runtime.llm import create_llm

logger = logging.getLogger(__name__)


TimelineCallback = Callable[[list[Any]], None]
DB_CONNECT_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class OrchestratorFactoryConfig:
    """Configuration needed to build an AdaptiveOrchestrator instance."""

    db_conninfo: str
    schema: str
    embedder_model: str
    curator_model_name: str
    curator_temperature: float
    judge_model_name: str | None
    judge_temperature: float
    request_timeout: float | None
    enable_subagent_memory: bool
    enable_playbooks: bool
    enable_blueprints: bool
    enable_planning: bool
    enable_agent_filtering: bool
    memory_mode: str
    enable_judge: bool


def build_adaptive_orchestrator(
    *,
    llm: Any,
    config: OrchestratorFactoryConfig,
    on_timeline_saved: TimelineCallback | None = None,
) -> Any:
    """Build the AdaptiveOrchestrator and its persistence stores."""

    logger.info("Creating orchestrator (schema=%s)", config.schema)
    embedder = Embedder(model=config.embedder_model)
    db_conninfo = add_connect_timeout(
        config.db_conninfo,
        timeout_seconds=DB_CONNECT_TIMEOUT_SECONDS,
    )

    episode_store, blueprint_store, playbook_store, trajectory_store = create_pg_stores(
        db_conninfo,
        embedder,
        schema=config.schema,
    )
    trajectory_store = configure_trajectory_store(
        trajectory_store,
        on_timeline_saved=on_timeline_saved,
    )

    curator_llm = create_llm(
        model_name=config.curator_model_name,
        temperature=config.curator_temperature,
        request_timeout=config.request_timeout,
    )
    judge_llm = (
        create_llm(
            model_name=config.judge_model_name,
            temperature=config.judge_temperature,
            request_timeout=config.request_timeout,
        )
        if config.judge_model_name
        else None
    )

    return AdaptiveOrchestrator(
        llm=llm,
        registry=AgentRegistry(),
        episode_store=episode_store,
        blueprint_store=blueprint_store,
        playbook_store=playbook_store,
        enable_subagent_memory=config.enable_subagent_memory,
        trajectory_store=trajectory_store,
        enable_playbooks=config.enable_playbooks,
        enable_blueprints=config.enable_blueprints,
        enable_planning=config.enable_planning,
        enable_agent_filtering=config.enable_agent_filtering,
        memory_mode=config.memory_mode,
        enable_judge=config.enable_judge,
        curator_llm=curator_llm,
        judge_llm=judge_llm,
    )


def add_connect_timeout(conninfo: str, timeout_seconds: int = 10) -> str:
    """Append ``connect_timeout`` to a PostgreSQL DSN if not already set."""
    if "connect_timeout" in conninfo:
        return conninfo
    if conninfo.startswith("postgresql://") or conninfo.startswith("postgres://"):
        separator = "&" if "?" in conninfo else "?"
        return f"{conninfo}{separator}connect_timeout={timeout_seconds}"
    return f"{conninfo} connect_timeout={timeout_seconds}"


def configure_trajectory_store(
    trajectory_store: Any,
    *,
    on_timeline_saved: TimelineCallback | None = None,
) -> Any:
    """Capture timelines and keep failed trajectory saves non-fatal."""
    if trajectory_store is None:
        return None

    original_save = trajectory_store.save

    def safe_save(task, timeline, final_response, episode_id="", **kwargs):
        if on_timeline_saved is not None:
            on_timeline_saved(list(timeline))
        try:
            return original_save(
                task=task,
                timeline=timeline,
                final_response=final_response,
                episode_id=episode_id,
                **kwargs,
            )
        except Exception as exc:
            logger.warning("Trajectory save failed; skipping non-fatal side effect: %s", exc)
            return None

    trajectory_store.save = safe_save
    return trajectory_store
