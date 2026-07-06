"""GAIA host environment
"""

from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)


class GaiaHostEnv:
    """Host-based environment for GAIA benchmark tasks.
    """

    name = "gaia_host"

    def __init__(self, task_dir: str, task: str, output_dir: str | None = None, **kwargs):
        """Initialize the GAIA host environment.

        Args:
            task_dir: Path to the GAIA task directory (contains testbed/).
            task: The GAIA question text.
            output_dir: Per-run output directory. When provided, the agent
                operates on ``{output_dir}/testbed/`` instead of the shared
                ``{task_dir}/testbed/data/``.
        """
        self.task_dir = os.path.abspath(task_dir)
        self.task = task
        if output_dir:
            self.workdir = os.path.join(os.path.abspath(output_dir), "testbed")
        else:
            self.workdir = os.path.join(self.task_dir, "testbed", "data")

    def setup(self) -> None:
        """Prepare the working directory."""
        os.makedirs(self.workdir, exist_ok=True)
        # Set env var so GAIA apps know where to write
        os.environ["GAIA_WORKDIR"] = self.workdir
        logger.info("GAIA workdir: %s", self.workdir)

    def reset(self) -> None:
        """Reset environment state for a fresh run."""
        # Remove answer file if it exists from a previous run
        answer_path = os.path.join(self.workdir, "answer.txt")
        if os.path.exists(answer_path):
            os.remove(answer_path)
