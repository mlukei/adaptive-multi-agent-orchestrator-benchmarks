"""Task execution helpers shared by the OfficeBench and GAIA runners."""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

STATUS_OK = "ok"
STATUS_TIMEOUT = "timeout"
STATUS_ERROR = "error"


def make_run_tag() -> str:
    """Timestamp tag in German local time used for task output directories."""
    return datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y%m%d%H%M%S")


@dataclass(frozen=True)
class TaskExecutionConfig:
    """Process settings for one benchmark task."""

    python_executable: str
    task_runner_script: str
    model_name: str
    mode: str
    task_timeout_seconds: int
    timeout_retries: int
    retry_backoff_seconds: int
    docker_name: str | None = None
    dockerfile_path: str | None = None
    container_name_prefix: str | None = None


def build_isolated_container_name(
    base: str,
    task_key: str,
    attempt: int = 0,
) -> str:
    """Build a per-run Docker container name to prevent task state leakage."""
    safe_base = re.sub(r"[^a-z0-9_.-]+", "-", base.lower()).strip("-") or "officebench"
    safe_key = re.sub(r"[^a-z0-9_.-]+", "-", task_key.replace("/", "-").lower()).strip("-")
    base_name = f"{safe_base}-{safe_key}"
    suffix = f"-r{attempt}" if attempt > 0 else ""
    max_base_len = 120 - len(suffix)
    return f"{base_name[:max_base_len]}{suffix}"


def remove_container_if_present(container_name: str) -> None:
    """Force-remove a container if it exists."""
    result = subprocess.run(
        ["docker", "rm", "-f", container_name],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return

    error_text = (result.stderr or "").strip()
    if error_text and "No such container" not in error_text:
        print(f"Warning: failed to remove container '{container_name}': {error_text}")


def run_single_task_subprocess(
    *,
    config: TaskExecutionConfig,
    task_dir: str,
    config_file: str,
    tag: str,
    container_name: str | None,
    task_index: int = 0,
) -> dict[str, Any]:
    """
    Run one benchmark task in a subprocess with optional timeout.
    GAIA leaves them ``None`` and skips the Docker flags.
    """
    command = [
        config.python_executable,
        config.task_runner_script,
        "--model_name",
        config.model_name,
        "--task_dir",
        task_dir,
        "--config_file",
        config_file,
        "--tag",
        tag,
        "--mode",
        config.mode,
    ]
    if config.docker_name:
        command.extend(["--docker_name", config.docker_name])
    if container_name:
        command.extend(["--container_name", container_name])
    if config.dockerfile_path:
        command.extend(["--dockerfile_path", config.dockerfile_path])
    command.extend(["--task_index", str(task_index)])

    start = time.monotonic()
    timeout = None if config.task_timeout_seconds <= 0 else config.task_timeout_seconds
    try:
        subprocess.run(command, check=True, timeout=timeout)
        return {
            "status": STATUS_OK,
            "duration_seconds": round(time.monotonic() - start, 2),
        }
    except subprocess.TimeoutExpired:
        duration = round(time.monotonic() - start, 2)
        return {
            "status": STATUS_TIMEOUT,
            "duration_seconds": duration,
            "error": f"Task exceeded timeout ({config.task_timeout_seconds}s).",
        }
    except subprocess.CalledProcessError as exc:
        duration = round(time.monotonic() - start, 2)
        return {
            "status": STATUS_ERROR,
            "duration_seconds": duration,
            "error": f"{config.task_runner_script} exited with code {exc.returncode}",
        }


def execute_task_with_retries(
    *,
    config: TaskExecutionConfig,
    task_key: str,
    task_index: int,
    task_dir: str,
    config_file: str,
    tag: str,
) -> dict[str, Any]:
    """Execute one task with timeout retries and cleanup semantics."""
    max_attempts = config.timeout_retries + 1
    last_outcome: dict[str, Any] = {
        "status": STATUS_ERROR,
        "duration_seconds": 0.0,
        "error": "Task did not produce a result.",
    }
    active_container_name: str | None = None
    attempts_used = 0
    docker_mode = bool(config.container_name_prefix)

    for attempt in range(max_attempts):
        attempts_used = attempt + 1
        if docker_mode: # only for officebench, GAIA runs without Docker
            active_container_name = build_isolated_container_name(
                config.container_name_prefix,
                task_key,
                attempt=attempt,
            )
            print(
                f"  Attempt {attempt + 1}/{max_attempts} "
                f"(container={active_container_name})"
            )
        else:
            print(f"  Attempt {attempt + 1}/{max_attempts}")
        outcome = run_single_task_subprocess(
            config=config,
            task_dir=task_dir,
            config_file=config_file,
            tag=tag,
            container_name=active_container_name,
            task_index=task_index,
        )
        last_outcome = outcome
        status = outcome["status"]

        if status == STATUS_OK:
            break

        if docker_mode and active_container_name:
            remove_container_if_present(active_container_name)
        if status == STATUS_ERROR:
            break

        has_more_attempts = attempt + 1 < max_attempts
        if has_more_attempts and config.retry_backoff_seconds > 0:
            print(f"  Timeout. Retrying in {config.retry_backoff_seconds}s...")
            time.sleep(config.retry_backoff_seconds)

    record = {
        "task_index": task_index,
        "task_key": task_key,
        "status": last_outcome["status"],
        "attempts": attempts_used,
        "container_name": active_container_name or "",
        "duration_seconds": last_outcome.get("duration_seconds"),
    }
    if last_outcome["status"] != STATUS_OK:
        record["error"] = last_outcome.get("error", "Unknown task error")
    return record
