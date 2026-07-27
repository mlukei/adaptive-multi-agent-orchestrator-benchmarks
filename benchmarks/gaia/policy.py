"""GAIA policy: orchestrator-facing execution logic."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from langgraph.errors import GraphRecursionError

from benchmarks.gaia.agents import GaiaAgents
from benchmarks.gaia.prompts import GAIA_FORMAT_SUFFIX
from benchmarks.shared.base_policy import BaseOrchestratorPolicy

logger = logging.getLogger(__name__)


class GaiaPolicy(BaseOrchestratorPolicy):
    """Multi-agent policy for GAIA tasks."""

    memory_mode = "train"

    def __init__(
        self,
        model_name: str,
        env: Any,
        task_config: dict[str, Any],
        app_config: Any = None,
        tag: str | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            env=env,
            task_config=task_config,
            app_config=app_config,
            tag=tag,
        )
        self.gaia_agents = GaiaAgents(
            app_config=self.app_config,
            env=env,
            agents_model_name=self.agents_model_name,
            agents_temperature=self.agents_temperature,
        )

    def _get_real_agent_configs(self) -> dict[str, Any]:
        return self.gaia_agents.build_configs()

    def run(self) -> str | None:
        """Execute one GAIA task through the adaptive orchestrator."""
        start = time.monotonic()
        termination = "planned"
        error_type = ""
        self.thread_id = uuid.uuid4().hex
        self._last_timeline = []

        try:
            if self.orchestrator is None:
                self._build_orchestrator()

            logger.info("Executing GAIA task: %s...", self.env.task[:100])
            return self.orchestrator.solve(
                task=self.env.task + GAIA_FORMAT_SUFFIX,
                thread_id=self.thread_id,
            )

        except GraphRecursionError:
            logger.warning("Orchestrator hit recursion limit")
            termination = "max_iterations"
            return None

        except Exception as exc:
            logger.exception("GAIA policy execution failed")
            termination = "error"
            error_type = type(exc).__name__
            return None

        finally:
            self._last_run_context = self._collect_run_context(
                run_id=self.thread_id,
                termination=termination,
                error_type=error_type,
                start=start,
            )
            logger.info(
                "GAIA task completed in %.1fs (termination=%s)",
                self._last_run_context["duration"],
                termination,
            )
