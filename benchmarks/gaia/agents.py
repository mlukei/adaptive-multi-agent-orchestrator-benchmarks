"""GAIA sub-agent construction for the adaptive orchestrator.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from benchmarks.shared.langchain_adapter import LangChainSubagentChain
from benchmarks.gaia.prompts import (
    LANGCHAIN_CODER_SYSTEM,
    LANGCHAIN_FILE_AUDIO_SYSTEM,
    LANGCHAIN_FILE_DOCUMENT_SYSTEM,
    LANGCHAIN_FILE_IMAGE_SYSTEM,
    LANGCHAIN_FILE_LOCATOR_SYSTEM,
    LANGCHAIN_FILE_SURFER_SYSTEM,
    LANGCHAIN_FILE_TABLE_SYSTEM,
    LANGCHAIN_WEB_SURFER_SYSTEM,
)
from benchmarks.gaia.tools import (
    CODER_TOOLS,
    FILE_AUDIO_SURFER_TOOLS,
    FILE_DOCUMENT_SURFER_TOOLS,
    FILE_IMAGE_SURFER_TOOLS,
    FILE_LOCATOR_TOOLS,
    FILE_SURFER_TOOLS,
    FILE_TABLE_SURFER_TOOLS,
    WEB_SURFER_TOOLS,
)
from benchmarks.shared.cards import build_agent_configs, load_cards
from runtime.config import optional_config_value
from runtime.llm import create_llm

logger = logging.getLogger(__name__)

# All known GAIA specialists: name -> (tools, system prompt template).
AGENT_SPECS: dict[str, tuple[list[Any], str]] = {
    "web_surfer": (WEB_SURFER_TOOLS, LANGCHAIN_WEB_SURFER_SYSTEM),
    "file_surfer": (FILE_SURFER_TOOLS, LANGCHAIN_FILE_SURFER_SYSTEM),
    "code_exec": (CODER_TOOLS, LANGCHAIN_CODER_SYSTEM),
    "file_document_surfer": (
        FILE_DOCUMENT_SURFER_TOOLS,
        LANGCHAIN_FILE_DOCUMENT_SYSTEM,
    ),
    "file_table_surfer": (FILE_TABLE_SURFER_TOOLS, LANGCHAIN_FILE_TABLE_SYSTEM),
    "file_image_surfer": (FILE_IMAGE_SURFER_TOOLS, LANGCHAIN_FILE_IMAGE_SYSTEM),
    "file_audio_surfer": (FILE_AUDIO_SURFER_TOOLS, LANGCHAIN_FILE_AUDIO_SYSTEM),
}

FILE_LOCATOR_TOOL_NAMES = frozenset(tool.name for tool in FILE_LOCATOR_TOOLS)


class GaiaAgents:
    """Build LangChain-backed GAIA sub-agents and registry configs."""

    def __init__(
        self,
        *,
        app_config: Any,
        env: Any,
        agents_model_name: str,
        agents_temperature: float,
    ) -> None:
        self.app_config = app_config
        self.env = env
        self.agents_model_name = agents_model_name
        self.agents_temperature = agents_temperature

    def build_configs(self) -> dict[str, Any]:
        """Build GAIA sub-agent configs using LangGraph ReAct agents."""
        workdir = str(self.env.workdir)
        orch_cfg = self.app_config.orchestrator
        use_memory = orch_cfg.enable_subagent_memory

        agent_cards = load_cards(
            orch_cfg.agent_cards_path,
            getattr(orch_cfg, "extra_agent_cards_path", None),
        )

        chains: dict[str, Any] = {}
        for agent_name, (tools, base_system_template) in AGENT_SPECS.items():
            if agent_name not in agent_cards:
                logger.info("Agent '%s' has no card; not registered.", agent_name)
                continue
            tools = self._agent_tools(agent_name, tools)
            tools = self._filter_disabled_tools(agent_name, tools)
            llm = self._build_langchain_llm(agent_name)
            system_template = self._system_template(
                agent_name,
                tools,
                base_system_template,
            )
            system_prompt = (
                system_template.format(workdir=workdir)
                if "{workdir}" in system_template
                else system_template
            )
            graph = create_agent(
                model=llm,
                tools=tools,
                system_prompt=system_prompt,
                checkpointer=InMemorySaver() if use_memory else None,
            )
            chains[agent_name] = LangChainSubagentChain(
                graph,
                agent_name=agent_name,
            )

        return build_agent_configs(
            agent_cards,
            chains,
            default_description=lambda name: f"GAIA {name} agent",
        )

    def _agent_tools(self, agent_name: str, tools: list[Any]) -> list[Any]:
        if agent_name != "file_surfer":
            return tools
        disabled_cfg = getattr(self.app_config.orchestrator, "disabled_tools", None)
        if not getattr(disabled_cfg, agent_name, None):
            return tools
        return [*FILE_LOCATOR_TOOLS, *tools]

    def _filter_disabled_tools(self, agent_name: str, tools: list[Any]) -> list[Any]:
        """Drop tools listed under ``orchestrator.disabled_tools.<agent>``."""
        disabled_cfg = getattr(self.app_config.orchestrator, "disabled_tools", None)
        disabled = set(getattr(disabled_cfg, agent_name, None) or [])
        if not disabled:
            return tools

        tool_names = {tool.name for tool in tools}
        unknown = sorted(disabled - tool_names)
        if unknown:
            logger.warning(
                "Agent '%s': disabled_tools lists unknown tool(s): %s",
                agent_name, unknown,
            )
        logger.info(
            "Agent '%s': disabled tools removed: %s",
            agent_name, sorted(disabled & tool_names),
        )
        return [tool for tool in tools if tool.name not in disabled]

    def _system_template(
        self,
        agent_name: str,
        tools: list[Any],
        base_system_template: str,
    ) -> str:
        if agent_name != "file_surfer":
            return base_system_template
        enabled_tool_names = {tool.name for tool in tools}
        if enabled_tool_names and enabled_tool_names.issubset(FILE_LOCATOR_TOOL_NAMES):
            return LANGCHAIN_FILE_LOCATOR_SYSTEM
        return base_system_template

    def _build_langchain_llm(self, agent_name: str) -> Any:
        """Create an AzureChatOpenAI for one sub-agent
        """
        agent_cfg = getattr(self.app_config.llm.agents, agent_name, None)

        model_name = optional_config_value(
            getattr(agent_cfg, "model_name", None),
            self.agents_model_name,
        )
        temperature = optional_config_value(
            getattr(agent_cfg, "temperature", None),
            self.agents_temperature,
        )
        return create_llm(model_name=model_name, temperature=temperature)
