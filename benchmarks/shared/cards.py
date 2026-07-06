"""Shared helpers for JSON-backed agent cards."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from a2a.types import AgentSkill

DescriptionFallback = str | Callable[[str], str]


def read_card_file(path: str | None) -> dict[str, dict[str, Any]]:
    """Read an agent-card JSON file."""
    if not path:
        return {}
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def load_cards(
    cards_path: str | None,
    extra_cards_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load base cards and optional extra cards into one name -> card map."""
    return {
        **read_card_file(cards_path),
        **read_card_file(extra_cards_path),
    }


def skills_from_card(card: dict[str, Any], agent_name: str) -> list[AgentSkill]:
    """Convert a JSON card's skills into A2A AgentSkill objects."""
    return [
        AgentSkill(
            id=skill["id"],
            name=skill["name"],
            description=skill["description"],
            tags=skill.get("tags", [agent_name]),
        )
        for skill in card.get("skills", [])
    ]


def build_agent_config(
    agent_name: str,
    chain: Any,
    card_data: dict[str, Any] | None,
    *,
    default_description: str = "",
) -> dict[str, Any]:
    """Build one registry config from one card and one runnable chain."""
    if card_data:
        description = card_data.get("description") or default_description
        skills = skills_from_card(card_data, agent_name)
    else:
        description = default_description
        skills = []

    config: dict[str, Any] = {
        "name": agent_name,
        "description": description,
        "chain": chain,
        "skills": skills,
    }
    if card_data and card_data.get("card_embedding"):
        config["card_embedding"] = card_data["card_embedding"]
    return config


def build_agent_configs(
    cards: Mapping[str, dict[str, Any]],
    chains: Mapping[str, Any],
    *,
    default_description: DescriptionFallback = "",
) -> dict[str, dict[str, Any]]:
    """Build registry configs for card-backed chains."""
    configs = {}
    for agent_name, chain in chains.items():
        card = cards[agent_name]
        fallback = (
            default_description(agent_name)
            if callable(default_description)
            else default_description
        )
        configs[agent_name] = build_agent_config(
            agent_name,
            chain,
            card,
            default_description=fallback,
        )
    return configs
