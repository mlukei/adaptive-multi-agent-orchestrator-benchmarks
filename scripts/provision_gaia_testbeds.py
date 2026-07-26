"""
Provision GAIA attachment files into the local task testbeds.
"""

from __future__ import annotations

import json
import shutil
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

REPO_ID = "gaia-benchmark/GAIA"
SPLIT = "2023/validation"
TASKS_ROOT = Path("tasks/gaia")


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set. Accept the terms at "
            f"https://huggingface.co/datasets/{REPO_ID} and set a read token."
        )

    snapshot = Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            token=token,
            allow_patterns=[f"{SPLIT}/*"],
        )
    )

    task_dirs = local_task_dirs()
    copied = 0
    for source in sorted((snapshot / SPLIT).iterdir()):
        task_dir = task_dirs.get(source.stem)
        if task_dir is None or source.suffix == ".parquet":
            continue  # metadata, or an attachment for a task we do not run
        destination = task_dir / "testbed" / "data" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1

    print(f"Provisioned {copied} attachments into {TASKS_ROOT}/*/testbed/data/")


def local_task_dirs() -> dict[str, Path]:
    task_dirs: dict[str, Path] = {}
    for config_path in sorted(TASKS_ROOT.glob("*/subtasks/*.json")):
        task_id = json.loads(config_path.read_text(encoding="utf-8")).get("gaia_task_id")
        if task_id:
            task_dirs[str(task_id)] = config_path.parent.parent
    return task_dirs


if __name__ == "__main__":
    main()
