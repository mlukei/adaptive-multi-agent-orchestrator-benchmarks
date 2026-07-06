"""Cost computation from episode_metrics token usage."""

from __future__ import annotations

import logging
import re
from typing import Any

_logger = logging.getLogger(__name__)


def _resolve_pricing(pricing_cfg: Any, model: str, *, use_default: bool) -> Any | None:
    """Find the pricing entry for model.

    The API reports models with a snapshot-date suffix (e.g.
    gpt-5.4-nano-2026-03-17), but the config keys are the plain names
    (gpt-5.4-nano). So this function tries the exact name first, then the name with
    the trailing -YYYY-MM-DD stripped, and finally default in the config.
    """
    plain_name = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)
    candidates = [model, plain_name]
    if use_default:
        candidates.append("default")
    for name in candidates:
        pricing = getattr(pricing_cfg, name, None)
        if pricing is not None:
            return pricing
    return None


def compute_cost_from_usage(
    token_usage_by_model: dict[str, dict[str, int]],
    pricing_cfg: Any,
) -> tuple[float, dict[str, float]]:
    """
    Compute LLM cost from per-model token usage and config pricing.
    """
    if not token_usage_by_model or pricing_cfg is None:
        return 0.0, {}

    cost_by_model: dict[str, float] = {}
    for model, usage in token_usage_by_model.items():
        pricing = _resolve_pricing(pricing_cfg, model, use_default=True)
        if pricing is None:
            _logger.warning(
                "No pricing configured for model '%s' and no 'default' entry; "
                "treating its cost as $0.",
                model,
            )
            continue
        model_cost = (
            usage.get("input_tokens", 0) / 1_000_000 * float(getattr(pricing, "input", 0.0))
            + usage.get("output_tokens", 0) / 1_000_000 * float(getattr(pricing, "output", 0.0))
        )
        cost_by_model[model] = round(model_cost, 6)

    return round(sum(cost_by_model.values()), 6), cost_by_model


def compute_embedding_cost(
    embedding_tokens: int,
    embedding_model: str,
    pricing_cfg: Any,
) -> float:
    """
    Compute embedding cost from token count and config pricing.
    """
    if not embedding_tokens or pricing_cfg is None:
        return 0.0

    pricing = _resolve_pricing(pricing_cfg, embedding_model, use_default=False)
    if pricing is None:
        _logger.warning(
            "No pricing configured for embedding model '%s'; treating cost as $0.",
            embedding_model,
        )
        return 0.0

    price_per_million = float(getattr(pricing, "input", 0.0))
    return round(embedding_tokens / 1_000_000 * price_per_million, 6)

