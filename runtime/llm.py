"""Utilities for creating an Azure-backed LLM instance."""

from __future__ import annotations

import os

from langchain_openai import AzureChatOpenAI


def create_llm(
    model_name: str | None = None,
    temperature: float = 0.0,
    model_kwargs: dict | None = None,
    request_timeout: float | None = 120.0,
) -> AzureChatOpenAI:
    """Create an `AzureChatOpenAI` configured from environment variables.

    Args:
        model_name: Optional override for the Azure deployment name. If not
            provided, the function reads AZURE_OPENAI_DEPLOYMENT from the
            environment.
        temperature: Sampling temperature for the model.
        model_kwargs: Optional extra kwargs forwarded to the model API
        request_timeout: Per-request HTTP read timeout in seconds.

    Returns:
        Configured AzureChatOpenAI instance.
    """

    deployment = model_name or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

    max_retries = int(os.environ.get("AZURE_OPENAI_MAX_RETRIES", "8"))

    return AzureChatOpenAI(
        azure_deployment=deployment,
        api_version=api_version,
        temperature=temperature,
        model_kwargs=model_kwargs or {},
        timeout=request_timeout,
        max_retries=max_retries,
        seed=42,
    )
