"""Shared orchestrator setup and policy lifecycle."""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any

from orchestrator import AdaptiveOrchestrator
from orchestrator.memory import create_pg_stores
from orchestrator.registration.registry import AgentRegistry
from orchestrator.shared.embedder import Embedder

from benchmarks.shared.agent_registration import register_agents
from logger.run_logger import RunLogger, collect_run_context
from runtime.config import load_config
from runtime.llm import create_llm

logger = logging.getLogger(__name__)


class BaseOrchestratorPolicy(ABC):
    """Shared orchestrator policy base for OfficeBench and GAIA."""

    memory_mode: str = "test"  # default; overridden by config or subclass

    def __init__(
        self,
        model_name: str,
        env: Any,
        task_config: dict[str, Any],
        app_config: Any = None,
        tag: str | None = None,
    ) -> None:
        super().__init__()
        self.app_config = app_config or load_config()
        self.model_name = model_name
        self.env = env
        self.task_config = task_config
        self.tag = tag
        self.thread_id = uuid.uuid4().hex

        orch_cfg = self.app_config.orchestrator
        self.orchestrator_variant = orch_cfg.variant
        # memory_schema: optional override to read memory from a different variant's DB schema
        memory_schema_override = getattr(orch_cfg, "memory_schema", None)
        self.orchestrator_schema = (
            memory_schema_override.replace("/", "_")
            if memory_schema_override
            else self.orchestrator_variant.replace("/", "_")
        )

        # memory_mode: config > class attribute > "test"
        self.memory_mode = getattr(orch_cfg, "memory_mode", None) or self.__class__.memory_mode

        self.db_conninfo = getattr(
            getattr(orch_cfg, "db", None), "conninfo",
            "postgresql://orchestrator:orchestrator@localhost/orchestrator",
        )
        self.enable_playbooks = bool(getattr(orch_cfg, "enable_playbooks", True))
        self.enable_subagent_memory = bool(getattr(orch_cfg, "enable_subagent_memory", False))
        self.enable_agent_filtering = bool(getattr(orch_cfg, "enable_agent_filtering", True))
        self.profile_agents = bool(getattr(orch_cfg, "profile_agents", False))
        self.enable_blueprints = bool(getattr(orch_cfg, "enable_blueprints", True))
        self.enable_planning = bool(getattr(orch_cfg, "enable_planning", True))
        self.enable_judge = bool(getattr(orch_cfg, "enable_judge", False))

        csv_path = f"results/{self.orchestrator_variant}.csv"

        llm_cfg = self.app_config.llm
        default_model = self.model_name or getattr(llm_cfg, "model_name", None)
        orch_llm_cfg = getattr(llm_cfg, "orchestrator", None)
        agents_llm_cfg = getattr(llm_cfg, "agents", None)
        self.orchestrator_model_name = getattr(orch_llm_cfg, "model_name", None) or default_model
        self.orchestrator_temperature = getattr(
            orch_llm_cfg, "temperature", getattr(llm_cfg, "temperature", 0.0)
        )
        self.agents_model_name = getattr(agents_llm_cfg, "model_name", None) or default_model
        self.agents_temperature = getattr(
            agents_llm_cfg, "temperature", getattr(llm_cfg, "temperature", 0.0)
        )
        # Curator/judge roles can be separated for evaluation runs. When judge
        # is not configured, AdaptiveOrchestrator falls back to curator_llm.
        curator_llm_cfg = getattr(llm_cfg, "curator", None)
        judge_llm_cfg = getattr(llm_cfg, "judge", None)
        embedder_llm_cfg = getattr(llm_cfg, "embedder", None)
        self.curator_model_name = (
            getattr(curator_llm_cfg, "model_name", None) or self.orchestrator_model_name
        )
        self.curator_temperature = getattr(
            curator_llm_cfg, "temperature", self.orchestrator_temperature
        )
        self.judge_model_name = getattr(judge_llm_cfg, "model_name", None)
        self.judge_temperature = getattr(
            judge_llm_cfg, "temperature", self.curator_temperature
        )
        self.embedder_model = (
            getattr(embedder_llm_cfg, "model_name", None) or "text-embedding-ada-002"
        )
        self.request_timeout: float | None = float(
            getattr(llm_cfg, "request_timeout", 120.0) or 120.0
        )

        self.llm = create_llm(
            model_name=self.orchestrator_model_name,
            temperature=self.orchestrator_temperature,
            request_timeout=self.request_timeout,
        )

        self.run_logger = RunLogger(
            path=csv_path,
            pricing_cfg=getattr(llm_cfg, "pricing", None),
            task_config=task_config,
            tag=tag,
            orchestrator_variant=self.orchestrator_variant,
            embedder_model=self.embedder_model,
        )

        self.orchestrator = None
        self._last_run_context: dict[str, Any] | None = None
        self._last_timeline: list[Any] = []

    @abstractmethod
    def _get_real_agent_configs(self) -> dict[str, Any]:
        """Return a name → agent-config dict for all real (non-noise) agents."""

    def _build_orchestrator(self) -> None:
        """Lazily create the AdaptiveOrchestrator and register all agents."""
        logger.info("Creating orchestrator (schema=%s)", self.orchestrator_schema)
        embedder = Embedder(model=self.embedder_model)
        stores = create_pg_stores(
            _add_connect_timeout(self.db_conninfo),
            embedder,
            schema=self.orchestrator_schema,
        )
        episode_store, blueprint_store, playbook_store, trajectory_store = stores
        trajectory_store = self._configure_trajectory_store(trajectory_store)

        curator_llm = create_llm(
            model_name=self.curator_model_name,
            temperature=self.curator_temperature,
            request_timeout=self.request_timeout,
        )
        judge_llm = (
            create_llm(
                model_name=self.judge_model_name,
                temperature=self.judge_temperature,
                request_timeout=self.request_timeout,
            )
            if self.judge_model_name
            else None
        )

        self.orchestrator = AdaptiveOrchestrator(
            llm=self.llm,
            registry=AgentRegistry(),
            episode_store=episode_store,
            blueprint_store=blueprint_store,
            playbook_store=playbook_store,
            trajectory_store=trajectory_store,
            enable_subagent_memory=self.enable_subagent_memory,
            enable_playbooks=self.enable_playbooks,
            enable_blueprints=self.enable_blueprints,
            enable_planning=self.enable_planning,
            enable_agent_filtering=self.enable_agent_filtering,
            memory_mode=self.memory_mode,
            enable_judge=self.enable_judge,
            curator_llm=curator_llm,
            judge_llm=judge_llm,
        )
        register_agents(
            orchestrator=self.orchestrator,
            real_configs=self._get_real_agent_configs(),
            app_config=self.app_config,
            should_profile=self.profile_agents,
        )

    def _configure_trajectory_store(self, trajectory_store: Any) -> Any:
        """Capture timelines and keep failed trajectory saves non-fatal."""
        if trajectory_store is None:
            return None

        original_save = trajectory_store.save

        def safe_save(task, timeline, final_response, episode_id="", **kwargs):
            self._last_timeline = list(timeline)
            try:
                return original_save(
                    task=task,
                    timeline=timeline,
                    final_response=final_response,
                    episode_id=episode_id,
                    **kwargs,
                )
            except Exception as exc:
                logger.warning(
                    "Trajectory save failed; skipping non-fatal side effect: %s",
                    exc,
                )
                return None

        trajectory_store.save = safe_save
        return trajectory_store

    def _collect_run_context(
        self,
        *,
        run_id: str,
        termination: str,
        error_type: str,
        start: float,
    ) -> dict[str, Any]:
        return collect_run_context(
            orchestrator=self.orchestrator,
            timeline=self._last_timeline,
            thread_id=self.thread_id,
            run_id=run_id,
            termination=termination,
            error_type=error_type,
            start=start,
        )

    # Logging

    def log_last_run_summary(self, *, eval_result: bool | None) -> None:
        """Write the last run summary to CSV."""
        self.run_logger.log_last(
            context=self._last_run_context,
            eval_result=eval_result,
        )


def _add_connect_timeout(conninfo: str, timeout_seconds: int = 10) -> str:
    """Append ``connect_timeout`` to a PostgreSQL DSN if not already set."""
    if "connect_timeout" in conninfo:
        return conninfo
    if conninfo.startswith(("postgresql://", "postgres://")):
        separator = "&" if "?" in conninfo else "?"
        return f"{conninfo}{separator}connect_timeout={timeout_seconds}"
    return f"{conninfo} connect_timeout={timeout_seconds}"
