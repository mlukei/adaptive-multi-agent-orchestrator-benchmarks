"""OfficeBench LangGraph adapter with task-local tool history maintained like the original benchmark."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from benchmarks.shared.langchain_adapter import (
    DEFAULT_RECURSION_LIMIT,
    LangGraphResult,
    invoke_langchain_graph,
    message_text,
)

TOOL_HISTORY_LIMIT = 6
TOOL_HISTORY_HEADING = "Previous tool interactions in this task:"


class LangChainSubagentAdapter:
    """Expose a LangChain agent graph through the orchestrator's chain interface."""

    def __init__(
        self,
        graph: Any,
        *,
        agent_name: str = "officebench",
        recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    ) -> None:
        self.graph = graph
        self.agent_name = agent_name
        self.recursion_limit = recursion_limit
        self._tool_history: list[str] = []

    def invoke(self, input: dict[str, Any], config: Any | None = None) -> str:
        text, result = invoke_langchain_graph(
            self.graph,
            _with_tool_history(input, self._tool_history),
            config=config,
            agent_name=self.agent_name,
            recursion_limit=self.recursion_limit,
        )
        self._remember_tool_interactions(result)
        return text

    def _remember_tool_interactions(self, result: LangGraphResult | None) -> None:
        if result is None:
            return

        pending_action = ""
        for message in result["messages"]:
            if isinstance(message, AIMessage) and message.tool_calls:
                pending_action = _format_tool_call(message.tool_calls[-1])
                continue
            if isinstance(message, ToolMessage):
                observation = message_text(message).strip()
                if observation:
                    action = pending_action or message.name or "tool"
                    self._tool_history.append(f"- {action} -> {observation[:600]}")
                    self._tool_history = self._tool_history[-TOOL_HISTORY_LIMIT:]
                pending_action = ""


def _with_tool_history(input: dict[str, Any], tool_history: list[str]) -> dict[str, Any]:
    if not tool_history:
        return input

    updated_messages = list(input["messages"])
    last_message = updated_messages[-1]
    history_block = TOOL_HISTORY_HEADING + "\n" + "\n".join(tool_history)
    updated_content = f"{history_block}\n\nCurrent instruction:\n{last_message['content']}"
    updated_messages[-1] = {**last_message, "content": updated_content}

    return {**input, "messages": updated_messages}


def _format_tool_call(tool_call: dict[str, Any]) -> str:
    name = tool_call.get("name", "tool")
    args = tool_call.get("args", {})
    return f"{name}({json.dumps(args, ensure_ascii=False)[:300]})"
