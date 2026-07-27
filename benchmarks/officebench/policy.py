"""Multi-Agent policy for OfficeBench benchmark execution."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from langgraph.errors import GraphRecursionError

from benchmarks.officebench.agents import OfficeBenchAgents
from benchmarks.shared.base_policy import BaseOrchestratorPolicy

logger = logging.getLogger(__name__)


class OfficeBenchPolicy(BaseOrchestratorPolicy):
    """Multi-Agent Policy: Orchestrator delegates to specialized OfficeBench sub-agents.

    Extends BaseOrchestratorPolicy with an OfficeBenchAgents-backed real-agents
    Tasks run in Docker via the OfficeBench environment.
    """

    memory_mode = "test"

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
        logger.info("Creating agents...")
        orch_cfg = getattr(self.app_config, "orchestrator", None)
        excluded_actions = getattr(orch_cfg, "disabled_actions", None)
        self.app_agents = OfficeBenchAgents(
            env=env,
            task_config=task_config,
            model_name=self.agents_model_name,
            temperature=self.agents_temperature,
            request_timeout=self.request_timeout,
            excluded_actions=excluded_actions,
            cards_path=getattr(orch_cfg, "agent_cards_path", None),
            extra_cards_path=getattr(orch_cfg, "extra_agent_cards_path", None),
        )

    # Agent configs
    def _get_real_agent_configs(self) -> dict[str, Any]:
        """Return the agent configs for actual OfficeBench agents."""
        return self.app_agents.get_all_agent_configs()

    # Execution
    def run(self) -> str | None:
        """Execute the multi-agent policy.

        Returns:
            Final response string, or None if recursion limit was reached.
        """
        start = time.monotonic()
        termination = "planned"
        error_type = ""
        self.thread_id = uuid.uuid4().hex
        self._last_timeline = []

        try:
            if self.orchestrator is None:
                self._build_orchestrator()

            answer = self.orchestrator.solve(
                task=self.env.task,
                thread_id=self.thread_id,
            )
            return self._handle_response(answer)

        except GraphRecursionError as e:
            logger.warning("Recursion limit reached: %s", e)
            termination = "max_iterations"
            return None

        except Exception as e:
            logger.exception("Error during execution")
            termination = "error"
            error_type = type(e).__name__
            raise

        finally:
            self._last_run_context = self._collect_run_context(
                run_id=self.thread_id,
                termination=termination,
                error_type=error_type,
                start=start,
            )

    def _handle_response(self, response: str) -> str:
        response_preview = response[:500] + ("..." if len(response) > 500 else "")
        logger.info("Response: %s", response_preview)
        self.env.step(
            str({
                "app": "system",
                "action": "finish_task",
                "answer": response,
            })
        )
        return response
