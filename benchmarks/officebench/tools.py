"""The single LangChain tool OfficeBench sub-agents use to drive their app.

``interact_with_app`` validates the app/action pair, rewrites file paths so they
stay inside the container's ``/testbed`` sandbox, switches the environment to the
target app, and returns the resulting observation.
"""

from __future__ import annotations

import ast
import json
import logging
import posixpath
import re
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, model_validator

import apps

logger = logging.getLogger(__name__)

# Apps whose actions take file paths, and the parameter names carrying them.
FILE_PATH_APPS = {"word", "excel", "pdf", "ocr", "file_transformer"}
FILE_PATH_KEYS = {
    "file_path",
    "word_file_path",
    "pdf_file_path",
    "excel_file_path",
    "image_file_path",
    "source_path",
    "target_path",
}

# Absolute paths a shell command could use to reach outside the sandbox.
_OUTSIDE_PATH_RE = re.compile(r"/(?:home|root|var|tmp|mnt|opt|srv|usr)/[^\s;|&\"']+")



def _into_testbed(path: str) -> str:
    """Redirect a path pointing outside the sandbox to ``/testbed/data/<name>``."""
    tail = posixpath.basename(path)
    logger.warning("Path '%s' is outside /testbed; rewriting to /testbed/data/%s", path, tail)
    return f"/testbed/data/{tail}"


def _normalize_testbed_path(path: str) -> str:
    """Map an agent-supplied file path to a location inside ``/testbed``."""
    cleaned = path.strip()
    if not cleaned:
        return cleaned

    if cleaned.startswith("/"):
        return cleaned if cleaned.startswith("/testbed") else _into_testbed(cleaned)

    # Relative path: strip any leading "./" or "../" and anchor it under /testbed.
    cleaned = cleaned.lstrip("./")
    if cleaned.startswith("testbed/"):
        return f"/{cleaned}"
    if cleaned.startswith(("data/", "emails/", "calendar/")):
        return f"/testbed/{cleaned}"
    return f"/testbed/data/{cleaned}"


def _normalize_shell_command(command: str) -> str:
    """Redirect any outside-the-sandbox absolute path inside a shell command."""
    return _OUTSIDE_PATH_RE.sub(lambda match: _into_testbed(match.group(0)), command)


def _normalize_file_parameters(app_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Rewrite the paths in an action's parameters to stay inside ``/testbed``."""
    if app_name in FILE_PATH_APPS:
        return {
            key: _normalize_testbed_path(value)
            if key in FILE_PATH_KEYS and isinstance(value, str)
            else value
            for key, value in params.items()
        }

    if app_name == "shell":
        command = params.get("command")
        if isinstance(command, str) and command:
            return {**params, "command": _normalize_shell_command(command)}

    return params


def _parse_parameters(parameters: str) -> dict[str, Any]:
    """Parse the model-supplied parameter string into a dict.
    """
    if not parameters:
        return {}

    for parse, candidate in (
        (json.loads, parameters),
        (json.loads, parameters.replace("\\", "/")),
        (ast.literal_eval, parameters),
    ):
        try:
            parsed = parse(candidate)
        except (ValueError, SyntaxError):
            continue
        return parsed if isinstance(parsed, dict) else {}

    logger.warning("Could not parse tool parameters: %r; using empty dict.", parameters)
    return {}


# The tool
class ToolInput(BaseModel):
    """Arguments for ``interact_with_app``, validated against the app registry."""

    app_name: str = Field(
        ...,
        description="Target app name (e.g., 'calendar', 'excel', 'shell')",
    )
    action_name: str = Field(
        ...,
        description="Action to perform within the app (e.g., 'create_event')",
    )
    parameters: str = Field(
        default="{}",
        description="Action parameters as JSON string (e.g., '{\"user\": \"Bob\"}')",
    )

    @model_validator(mode="after")
    def validate_app_action(self) -> "ToolInput":
        """Reject unknown apps and actions, listing the valid options."""
        actions = apps.AVAILABLE_ACTIONS.get(self.app_name)
        if actions is None:
            raise ValueError(
                f"Unknown app: '{self.app_name}'. "
                f"Available apps: {', '.join(sorted(apps.AVAILABLE_ACTIONS))}"
            )
        if self.action_name not in actions:
            raise ValueError(
                f"Action '{self.action_name}' not available for app '{self.app_name}'. "
                f"Available actions: {', '.join(sorted(actions))}"
            )
        return self


def make_interact_tool(env: Any, excluded_actions: dict[str, set[str]] | None = None):
    """Build the ``interact_with_app`` tool bound to one environment.

    Args:
        env: The OfficeBench environment instance.
        excluded_actions: Per-app action names to refuse, for capability ablations.
    """
    excluded = excluded_actions or {}

    @tool("interact_with_app", args_schema=ToolInput)
    def interact_with_app(
        app_name: str,
        action_name: str,
        parameters: str = "{}",
    ) -> str:
        """Execute an action in an OfficeBench app and return its observation.

        Switches to the target app first if it is not already active.

        Args:
            app_name: Target app.
            action_name: Action to execute.
            parameters: Action parameters as JSON string (default "{}").
        """
        if action_name in excluded.get(app_name, set()):
            return (
                "OBSERVATION: This action is disabled by the benchmark config: "
                f"{app_name}.{action_name}. Use another available action or delegate "
                "to a different agent."
            )

        params = _parse_parameters(parameters)
        if app_name == "shell" and action_name == "command" and "command" not in params:
            return (
                "OBSERVATION: Missing required parameter 'command'. "
                "Usage: shell.command(command='your shell command here')"
            )

        params = _normalize_file_parameters(app_name, params)
        logger.info("[Tool] %s.%s(%s)", app_name, action_name, params)

        if getattr(env, "current_app", None) != app_name:
            env.step(str({"app": "system", "action": "switch_app", "target_app": app_name}))

        observation = str(env.step(str({"app": app_name, "action": action_name, **params}))[0])
        logger.info("[Tool] Observation: %s", observation[:200])
        return observation

    return interact_with_app
