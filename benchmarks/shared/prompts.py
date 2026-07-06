"""Shared prompt templates for benchmark support agents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_noise_persona_prompt(name: str, card: Mapping[str, Any]) -> str:
    """Build the system prompt for a card-backed persona noise agent."""
    skills = _format_advertised_skills(card)
    description = card.get("description", "")
    return f"""You are the "{name}" agent. Advertised role: {description}
Your advertised skills:
{skills}

You are participating in a live multi-agent conversation and another agent has
delegated a request to you. Respond naturally and competently, in character.

IMPORTANT constraints:
- You have NO access to tools, files, or any real data in this environment.
- Therefore you must NEVER fabricate concrete results, file contents, answers,
  IDs, or claim a task is "done".
- Instead: engage with the request, describe what you would do, or ask for the
  inputs you would need. Stay plausible and on-topic.
- Stay in character; do not discuss your implementation, evaluation setup, or
  missing backend.
"""


def _format_advertised_skills(card: Mapping[str, Any]) -> str:
    skills = []
    for skill in card.get("skills", []):
        skill_name = skill.get("name") or skill.get("id") or "unnamed skill"
        description = skill.get("description", "")
        if description:
            skills.append(f"- {skill_name}: {description}")
        else:
            skills.append(f"- {skill_name}")
    return "\n".join(skills) if skills else "- None listed."
