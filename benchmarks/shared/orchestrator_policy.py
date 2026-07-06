"""Shared base class for OfficeBench and GAIA orchestrator policies.

Both policies use the same AdaptiveOrchestrator, the same config layout,
and the same shared registration/factory/logging helpers.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any

from benchmarks.shared.agent_registration import register_agents
from benchmarks.shared.orchestrator_factory import (
    OrchestratorFactoryConfig,
    build_adaptive_orchestrator,
)
from logger import RunLogger
from logger.local_logger import CsvLogger
from logger.policy_logging import (
    collect_run_context,
    log_last_run_summary as write_last_run_summary,
)
from runtime.config import load_config
from runtime.llm import create_llm


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
            csv_logger=CsvLogger(path=csv_path),
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
        self.orchestrator = build_adaptive_orchestrator(
            llm=self.llm,
            config=OrchestratorFactoryConfig(
                db_conninfo=self.db_conninfo,
                schema=self.orchestrator_schema,
                embedder_model=self.embedder_model,
                curator_model_name=self.curator_model_name,
                curator_temperature=self.curator_temperature,
                judge_model_name=self.judge_model_name,
                judge_temperature=self.judge_temperature,
                request_timeout=self.request_timeout,
                enable_subagent_memory=self.enable_subagent_memory,
                enable_playbooks=self.enable_playbooks,
                enable_blueprints=self.enable_blueprints,
                enable_planning=self.enable_planning,
                enable_agent_filtering=self.enable_agent_filtering,
                memory_mode=self.memory_mode,
                enable_judge=self.enable_judge,
            ),
            on_timeline_saved=self._capture_timeline,
        )
        register_agents(
            orchestrator=self.orchestrator,
            real_configs=self._get_real_agent_configs(),
            app_config=self.app_config,
            should_profile=self.profile_agents,
        )

    def _capture_timeline(self, timeline: list[Any]) -> None:
        self._last_timeline = list(timeline)

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

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_last_run_summary(self, *, eval_result: bool | None) -> None:
        """Write the last run summary to CSV."""
        write_last_run_summary(
            self.run_logger,
            last_run_context=self._last_run_context,
            eval_result=eval_result,
        )
