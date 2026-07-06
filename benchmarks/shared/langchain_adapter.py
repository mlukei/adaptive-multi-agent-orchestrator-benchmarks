"""Shared LangGraph sub-agent adapter for benchmark orchestrators."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.errors import GraphRecursionError

DEFAULT_RECURSION_LIMIT = 80
LangGraphResult = dict[str, list[Any]]

logger = logging.getLogger(__name__)


class LangChainSubagentChain:
    """Expose a compiled LangGraph agent as a sync chain returning text."""

    def __init__(
        self,
        graph: Any,
        *,
        agent_name: str = "subagent",
        recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    ) -> None:
        self.graph = graph
        self.agent_name = agent_name
        self.recursion_limit = recursion_limit

    def invoke(self, input: dict[str, Any], config: Any | None = None) -> str:
        text, _ = invoke_langchain_graph(
            self.graph,
            input,
            config=config,
            agent_name=self.agent_name,
            recursion_limit=self.recursion_limit,
        )
        return text


def invoke_langchain_graph(
    graph: Any,
    input: dict[str, Any],
    *,
    config: Any | None = None,
    agent_name: str = "subagent",
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> tuple[str, LangGraphResult | None]:
    run_config = dict(config or {})
    run_config["recursion_limit"] = recursion_limit

    try:
        result = graph.invoke(input, config=run_config)
    except GraphRecursionError:
        logger.warning("%s hit recursion limit (%d)", agent_name, recursion_limit)
        return (
            f"ERROR: Agent {agent_name} exceeded its step limit "
            f"({recursion_limit}) before producing an answer.",
            None,
        )
    except Exception as exc:
        logger.exception("%s execution failed", agent_name)
        return f"ERROR: Agent {agent_name} failed: {type(exc).__name__}: {exc}", None

    text = extract_final_text(result)
    if text:
        return text, result
    return f"ERROR: Agent {agent_name} returned no answer.", result


def extract_final_text(result: LangGraphResult) -> str:
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return message_text(message).strip()
    return ""


def message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(
        part if isinstance(part, str) else part.get("text", "")
        for part in content
        if isinstance(part, str) or (isinstance(part, dict) and part.get("type") == "text")
    )
