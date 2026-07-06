"""LangChain tools for interacting with OfficeBench apps.

Used by specialized sub-agents to execute actions in the Docker environment.
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


_OUTSIDE_PATH_RE = re.compile(r"/(?:home|root|var|tmp|mnt|opt|srv|usr)/[^\s;|&\"']+")


def _quarantine_into_testbed(path: str) -> str:
    """Redirect a path that points outside the sandbox into /testbed/data/."""
    tail = posixpath.basename(path)
    logger.warning("Path '%s' is outside /testbed; rewriting to /testbed/data/%s", path, tail)
    return f"/testbed/data/{tail}"


def _normalize_testbed_path(path: str) -> str:
    """Map a file path the agent supplied to a location inside /testbed.
    """
    cleaned = path.strip()
    if not cleaned:
        return cleaned

    if cleaned.startswith("/"):
        if cleaned.startswith("/testbed"):
            return cleaned
        return _quarantine_into_testbed(cleaned)

    cleaned = cleaned.lstrip("./")
    if cleaned.startswith("testbed/"):
        return f"/{cleaned}"
    if cleaned.startswith(("data/", "emails/", "calendar/")):
        return f"/testbed/{cleaned}"
    return f"/testbed/data/{cleaned}"


def _normalize_shell_command(command: str) -> str:
    """Redirect absolute paths outside /testbed inside a shell command string.
    """
    def rewrite(match: re.Match) -> str:
        path = match.group(0)
        return path if path.startswith("/testbed") else _quarantine_into_testbed(path)

    return _OUTSIDE_PATH_RE.sub(rewrite, command)


def _normalize_file_parameters(app_name: str, params_dict: dict[str, Any]) -> dict[str, Any]:
    """Rewrite the file paths in an action's parameters to stay inside /testbed."""
    if app_name in FILE_PATH_APPS:
        return {
            key: _normalize_testbed_path(value)
            if key in FILE_PATH_KEYS and isinstance(value, str)
            else value
            for key, value in params_dict.items()
        }

    if app_name == "shell":
        cmd = params_dict.get("command")
        if isinstance(cmd, str) and cmd:
            return {**params_dict, "command": _normalize_shell_command(cmd)}

    return params_dict


def _parse_parameters(parameters: str) -> dict[str, Any]:
    """Parse the LLM-supplied JSON parameter string, tolerating invalid chars.
    """
    if not parameters:
        return {}
    for candidate in (parameters, parameters.replace("\\", "/")):
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            continue
    try:
        parsed = ast.literal_eval(parameters)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, SyntaxError) as e:
        logger.warning("Could not parse tool parameters: %r (%s); using empty dict.", parameters, e)
        return {}


class ToolInput(BaseModel):
    """Input schema for the `interact_with_app` tool.

    Agents use this schema to specify which app and action to invoke,
    plus any required parameters.
    """

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
        """Validate that the app and action combination is valid.

        Raises:
            ValueError: If app or action is unknown or not allowed.
        """
        if self.app_name not in apps.AVAILABLE_ACTIONS:
            available = ", ".join(sorted(apps.AVAILABLE_ACTIONS.keys()))
            raise ValueError(
                f"Unknown app: '{self.app_name}'. Available apps: {available}"
            )

        if self.action_name not in apps.AVAILABLE_ACTIONS[self.app_name]:
            available = ", ".join(sorted(apps.AVAILABLE_ACTIONS[self.app_name]))
            raise ValueError(
                f"Action '{self.action_name}' not available for app '{self.app_name}'. "
                f"Available actions: {available}"
            )

        return self


def make_interact_tool(env: Any, excluded_actions: dict[str, set[str]] | None = None):
    """Create a LangChain tool for direct app interaction (used by subagents).

    Args:
        env: The OfficeBench environment instance.
        excluded_actions: Dictionary of actions to exclude for each app.

    Returns:
        A LangChain `@tool` function for executing actions in OfficeBench apps.
    """
    @tool("interact_with_app", args_schema=ToolInput)
    def interact_with_app(
        app_name: str,
        action_name: str,
        parameters: str = "{}",
    ) -> str:
        """Execute an action in an OfficeBench app.

        Automatically switches to the target app and then
        executes the specified action with the provided parameters.

        Args:
            app_name: Target app.
            action_name: Action to execute.
            parameters: Action parameters as JSON string (default "{}").

        Returns:
            Observation string from the environment.
        """

        if action_name in (excluded_actions or {}).get(app_name, set()):
            return (
                "OBSERVATION: This action is disabled by the benchmark config: "
                f"{app_name}.{action_name}. Use another available action or delegate "
                "to a different agent."
            )

        params_dict = _parse_parameters(parameters)

        # Validate required keys before execution
        if app_name == "shell" and action_name == "command" and "command" not in params_dict:
            return (
                "OBSERVATION: Missing required parameter 'command'. "
                "Usage: shell.command(command='your shell command here')"
            )

        params_dict = _normalize_file_parameters(app_name, params_dict)

        logger.info(f"[Tool] {app_name}.{action_name}({params_dict})")

        # Auto-switch to target app if needed (skip 'system' app)
        current_app = getattr(env, "current_app", None)
        if current_app != app_name:
            switch_payload = {
                "app": "system",
                "action": "switch_app",
                "target_app": app_name,
            }
            env.step(str(switch_payload))

        # Execute action
        payload = {"app": app_name, "action": action_name, **params_dict}
        obs, _reward, _done, _info = env.step(str(payload))

        obs = str(obs)
        logger.info("[Tool] Observation: %s", obs[:200])
        return obs

    return interact_with_app
