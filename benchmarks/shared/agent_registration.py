"""Shared agent registration helpers for benchmark orchestrator policies."""

from __future__ import annotations

import json
import logging
from typing import Any

from benchmarks.shared.noise_agents import (
    DEFAULT_NOISE_TEMPERATURE,
    DEFAULT_REQUEST_TIMEOUT,
    load_noise_agent_configs,
)
from runtime.config import optional_config_value

logger = logging.getLogger(__name__)


AgentConfigs = dict[str, dict[str, Any]]


def load_profiled_bullets(app_config: Any, *, should_profile: bool) -> dict[str, list[str]]:
    """Load precomputed profiled bullets when profiling mode is enabled.

    When no ``profiled_bullets_path`` is configured (missing, null, or empty),
    return an empty mapping so agents are profiled at runtime instead of falling
    back to a hard-coded default file.
    """
    if not should_profile:
        return {}

    orch_cfg = getattr(app_config, "orchestrator", None)
    bullets_path = getattr(orch_cfg, "profiled_bullets_path", None)
    if not bullets_path:
        logger.info("No profiled_bullets_path configured; agents will be profiled at runtime.")
        return {}
    try:
        with open(bullets_path, encoding="utf-8") as file:
            bullets = json.load(file)
    except FileNotFoundError:
        logger.warning(
            "Profiled bullets file not found at '%s'; agents will be profiled at runtime.",
            bullets_path,
        )
        return {}

    logger.info(
        "Loaded pre-computed profiled bullets for %d agents from %s",
        len(bullets),
        bullets_path,
    )
    return bullets


def register_agents(
    *,
    orchestrator: Any,
    real_configs: AgentConfigs,
    app_config: Any,
    should_profile: bool,
) -> None:
    """Register real + noise agents with the orchestrator, then optionally profile."""
    profiled_bullets = load_profiled_bullets(app_config, should_profile=should_profile)
    _register_agent_group(
        orchestrator,
        real_configs,
        profiled_bullets,
        kind="real",
    )

    noise_path = getattr(
        getattr(app_config, "orchestrator", None),
        "noise_agents_path",
        None,
    )
    noise_configs = (
        load_noise_agent_configs(noise_path, **_noise_llm_kwargs(app_config))
        if noise_path
        else {}
    )
    _register_agent_group(
        orchestrator,
        noise_configs,
        profiled_bullets,
        kind="noise",
    )

    total_count = len(real_configs) + len(noise_configs)
    if should_profile and hasattr(orchestrator, "profile_agents"):
        logger.info("Profiling %d agents...", total_count)
        orchestrator.profile_agents()

    logger.info(
        "%d agents registered (%d real, %d noise)",
        total_count,
        len(real_configs),
        len(noise_configs),
    )


def _register_agent_group(
    orchestrator: Any,
    configs: AgentConfigs,
    profiled_bullets: dict[str, list[str]],
    *,
    kind: str,
) -> None:
    for name, cfg in configs.items():
        logger.info("Registering %s agent '%s'", kind, name)
        orchestrator.registry.register(
            name=cfg["name"],
            description=cfg.get("description", ""),
            chain=cfg["chain"],
            skills=cfg.get("skills"),
            card_embedding=cfg.get("card_embedding"),
            profiled_bullets=profiled_bullets.get(name),
        )


def _noise_llm_kwargs(app_config: Any) -> dict[str, Any]:
    orch_cfg = getattr(app_config, "orchestrator", None)
    llm_cfg = getattr(app_config, "llm", None)
    agents_llm_cfg = getattr(llm_cfg, "agents", None)

    model_name = optional_config_value(
        getattr(orch_cfg, "noise_model_name", None),
        optional_config_value(
            getattr(agents_llm_cfg, "model_name", None),
            optional_config_value(getattr(llm_cfg, "model_name", None)),
        ),
    )
    temperature = optional_config_value(
        getattr(orch_cfg, "noise_temperature", None),
        DEFAULT_NOISE_TEMPERATURE,
    )
    request_timeout = optional_config_value(
        getattr(llm_cfg, "request_timeout", None),
        DEFAULT_REQUEST_TIMEOUT,
    )

    return {
        "model_name": model_name,
        "temperature": float(temperature),
        "request_timeout": float(request_timeout) if request_timeout else None,
    }
