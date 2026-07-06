"""Internal subprocess worker for one GAIA benchmark task."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
from dotenv import load_dotenv

load_dotenv()

from benchmarks.gaia.environment import GaiaHostEnv
from benchmarks.gaia.evaluation import evaluate_answer_match
from benchmarks.gaia.policy import GaiaPolicy
from runtime.config import load_config
from runtime.experiment_runtime import make_run_tag

_CFG = load_config()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GaiaRunPaths:
    task_dir: Path
    config_file: Path
    output_dir: Path
    testbed_dir: Path
    task_id: str
    subtask_id: str


def main(
    model_name: str | None = None,
    task_dir: str = "",
    config_file: str = "",
    tag: str | None = None,
    mode: str | None = None,
    task_index: int = 0,
):
    """Execute one GAIA task, persist artifacts, evaluate, and log a CSV row."""
    if not task_dir or not config_file:
        raise ValueError("--task_dir and --config_file are required")

    cfg = _CFG
    model_name = model_name or cfg.llm.model_name
    mode = mode or cfg.execution.mode
    tag = tag or make_run_tag()

    task_config = _read_json(Path(config_file))
    paths = _build_paths(
        task_dir=Path(task_dir),
        config_file=Path(config_file),
        model_name=model_name,
        tag=tag,
    )
    if not _prepare_output_dir(paths.output_dir, mode):
        return
    _prepare_testbed(paths)
    _log_task_header(paths, task_config)

    env = GaiaHostEnv(
        task_dir=str(paths.task_dir),
        task=task_config["task"],
        output_dir=str(paths.output_dir),
    )
    env.setup()
    env.reset()

    policy = GaiaPolicy(
        model_name=model_name,
        env=env,
        task_config=_policy_task_config(paths, task_config, task_index),
        app_config=cfg,
        tag=tag,
    )

    answer, run_exception = _run_policy(policy)
    _persist_artifacts(paths, task_config, model_name, tag, answer)
    eval_result = _evaluate(task_config, paths.testbed_dir, crashed=run_exception is not None)
    _log_summary(policy, eval_result)

    if run_exception is not None:
        raise run_exception

    logger.info("Result: %s | Answer: %s", _status_label(eval_result), (answer or "")[:80])
    return answer


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _build_paths(
    *,
    task_dir: Path,
    config_file: Path,
    model_name: str,
    tag: str,
) -> GaiaRunPaths:
    subtask_id = config_file.stem
    output_dir = (
        task_dir
        / "outputs"
        / subtask_id
        / f"{model_name.replace('/', '_')}_{tag}"
    )
    return GaiaRunPaths(
        task_dir=task_dir,
        config_file=config_file,
        output_dir=output_dir,
        testbed_dir=output_dir / "testbed",
        task_id=task_dir.name,
        subtask_id=subtask_id,
    )


def _prepare_output_dir(output_dir: Path, mode: str) -> bool:
    if not output_dir.exists():
        return True
    if mode != "force_new":
        logger.info("Output already exists (mode=default), skipping: %s", output_dir)
        return False
    logger.info("Removing existing output: %s", output_dir)
    shutil.rmtree(output_dir)
    return True


def _prepare_testbed(paths: GaiaRunPaths) -> None:
    source_data = paths.task_dir / "testbed" / "data"
    if source_data.is_dir():
        shutil.copytree(source_data, paths.testbed_dir, dirs_exist_ok=True)
        return
    paths.testbed_dir.mkdir(parents=True, exist_ok=True)


def _log_task_header(paths: GaiaRunPaths, task_config: dict[str, Any]) -> None:
    logger.info("=" * 60)
    logger.info("GAIA task: %s (Level %d)", paths.task_id, int(task_config["gaia_level"]))
    logger.info("Question: %s", task_config["task"][:200])
    logger.info("=" * 60)


def _policy_task_config(
    paths: GaiaRunPaths,
    task_config: dict[str, Any],
    task_index: int,
) -> dict[str, Any]:
    return {
        "task_dir": str(paths.task_dir),
        "subtask_id": paths.subtask_id,
        "task_id": paths.task_id,
        "gaia_level": int(task_config["gaia_level"]),
        "task_index": int(task_index),
    }


def _run_policy(policy: GaiaPolicy) -> tuple[str | None, Exception | None]:
    try:
        return policy.run(), None
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        raise
    except Exception as exc:
        logger.exception("GAIA policy execution failed")
        return None, exc


def _persist_artifacts(
    paths: GaiaRunPaths,
    task_config: dict[str, Any],
    model_name: str,
    tag: str,
    answer: str | None,
) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    if answer:
        answer_path = paths.testbed_dir / "answer.txt"
        if not answer_path.is_file():
            answer_path.write_text(answer.strip(), encoding="utf-8")

    settings = task_config | {"model_name": model_name, "tag": tag}
    with (paths.output_dir / "settings.json").open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def _evaluate(
    task_config: dict[str, Any],
    testbed_dir: Path,
    *,
    crashed: bool,
) -> bool | None:
    if crashed:
        return None
    checks = task_config.get("evaluation", [])
    if not checks:
        return None
    try:
        return all(_run_eval_check(check, testbed_dir) for check in checks)
    except Exception:
        logger.exception("Task evaluation crashed")
        return False


def _run_eval_check(check: dict[str, Any], testbed_dir: Path) -> bool:
    if check["function"] != "evaluate_answer_match":
        raise ValueError(f"Unknown GAIA evaluation function: {check['function']}")
    return evaluate_answer_match(str(testbed_dir), **check.get("args", {}))


def _log_summary(policy: GaiaPolicy, eval_result: bool | None) -> None:
    try:
        policy.log_last_run_summary(eval_result=eval_result)
    except Exception:
        logger.exception("Failed to log run summary")


def _status_label(eval_result: bool | None) -> str:
    if eval_result is True:
        return "PASS"
    if eval_result is False:
        return "FAIL"
    return "NO_EVAL"


if __name__ == "__main__":
    fire.Fire(main)
