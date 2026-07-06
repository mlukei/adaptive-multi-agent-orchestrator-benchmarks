"""Noise agent support for benchmark agent-selection tests."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from benchmarks.shared.cards import build_agent_configs, load_cards
from benchmarks.shared.prompts import build_noise_persona_prompt
from runtime.llm import create_llm

logger = logging.getLogger(__name__)

DEFAULT_NOISE_TEMPERATURE = 0.7
DEFAULT_REQUEST_TIMEOUT = 120.0


class PersonaNoiseChain:
    """LLM-backed distractor that stays in character but has no real tools."""

    def __init__(self, name: str, card: dict[str, Any], llm: Any) -> None:
        self._name = name
        self._llm = llm
        self._system = SystemMessage(content=build_noise_persona_prompt(name, card))
        self._history: list[BaseMessage] = []

    def invoke(self, input: Any, config: Any | None = None) -> str:
        message = HumanMessage(content=input if isinstance(input, str) else str(input))
        response = self._llm.invoke(
            [self._system, *self._history, message],
            config=config,
        )
        self._history.extend([message, response])
        logger.info("Noise agent '%s' answered in character.", self._name)
        return _message_content_to_text(response)


def load_noise_agent_configs(
    noise_path: str,
    *,
    model_name: str | None = None,
    temperature: float = DEFAULT_NOISE_TEMPERATURE,
    request_timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
) -> dict[str, dict]:
    """Load noise agent descriptors from a JSON file and return registry-ready configs.

    Args:
        noise_path: Path to the noise agent cards JSON file.
        model_name: Optional Azure deployment override for noise personas.
        temperature: Sampling temperature for the shared noise-persona LLM.
        request_timeout: Per-request HTTP read timeout in seconds.

    Returns:
        Dict mapping agent name to config dict (name, description, chain, skills).
    """
    cards = load_cards(noise_path)
    if not cards:
        return {}
    llm = create_llm(
        model_name=model_name,
        temperature=temperature,
        request_timeout=request_timeout,
    )
    chains = {
        agent_name: PersonaNoiseChain(agent_name, card, llm)
        for agent_name, card in cards.items()
    }
    configs = build_agent_configs(
        cards,
        chains,
        default_description=lambda name: f"Specialist for {name}",
    )
    logger.info("Loaded %d noise agents from %s", len(configs), noise_path)
    return configs


def _message_content_to_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(
        part if isinstance(part, str) else part.get("text", "")
        for part in content
        if isinstance(part, str)
        or (isinstance(part, dict) and part.get("type") == "text")
    )
