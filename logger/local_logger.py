"""Run-level CSV logging shared by the OfficeBench and GAIA runners."""

from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RunSummary:
    """Single benchmark run: identification, config, result, and metrics."""

    # --- Identification ---
    run_id: str = ""
    created_at_utc: str = ""
    task_id: str = ""
    subtask_id: str = ""
    task_dir: str = ""
    tag: str = ""
    orchestrator_variant: str = "baseline"

    # --- Outcome ---
    success: int = -1                   # 1=pass, 0=fail, -1=no eval
    termination_reason: str = ""        # planned | max_iterations | error
    error_type: str = ""
    judge_accepted: int = -1            # 1=accepted, 0=rejected, -1=not run
    judge_force_accepted: int = -1      # 1=force-accepted by judge, 0=no, -1=not run

    # --- Timing ---
    wall_clock_seconds: float = 0.0

    # --- Orchestration ---
    plan_created: int = -1              # 1=yes, 0=no, -1=unknown
    total_delegations: int = 0
    submission_attempts: int = 0
    judge_rejections: int = 0

    # --- Memory ---
    blueprint_matched: int = -1          # 1/0/-1
    blueprint_id: str = ""               # DB UUID of matched blueprint (empty if none)
    blueprint_similarity: float = 0.0
    agent_curations_run: int = 0
    episode_curation_run: int = -1      # 1/0/-1
    episode_curation_skipped: int = -1  # 1/0/-1
    episode_familiarity: float = -1.0     # familiarity score of episode (-1.0 = not computed)

    # --- Agent selection ---
    executed_agents: str = "[]"         # JSON list
    optimal_sequence: str = "[]"        # JSON list — curator's minimal necessary agents
    capability_boosted_agents: int = 0  # agents whose score was boosted by capability bullets
    agent_discovery_sources: str = "{}" # JSON dict: agent → "embedding"|"capability"|…
    agent_retrieval_log: str = "[]"     # JSON list of per-context retrieval records

    # --- Tokens + Cost ---
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    embedding_tokens: int = 0
    cost_llm: float = 0.0
    cost_embedding: float = 0.0
    cost_total: float = 0.0
    cost_by_model: str = "{}"        # JSON dict: model -> USD cost

    # --- Playbook memory evolution ---
    playbook_confirm_votes: int = 0
    playbook_contradict_votes: int = 0
    playbook_bullets_added: int = 0
    playbook_bullets_pruned: int = 0
    playbook_unconfirmed_prunes: int = 0
    playbook_harm_prunes: int = 0
    playbook_consolidation_merges: int = 0
    playbook_evolution: str = "[]"      # JSON list of per-agent delta/consolidation events

    # --- Internal orchestrator tools ---
    num_internal_tool_calls: int = 0
    tool_call_history: str = "[]"       # JSON list of tool names


def serialize_optional_bool(value: bool | None) -> int:
    """Serialize True/False/None as 1/0/-1 for CSV storage.
    """
    return {True: 1, False: 0}.get(value, -1)


class CsvLogger:
    """Writes RunSummary rows to CSV incrementally, one row per run."""

    def __init__(self, path: str | None = "results/runs.csv") -> None:
        self.path = path
        self._write_header: list[str] | None = None

    def log(self, summary: RunSummary) -> None:
        if not self.path:
            return

        row = asdict(summary)
        header = [f.name for f in fields(RunSummary)]
        output_path = Path(self.path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Resolve the column order once per logger instance. On the first call we
        # reconcile the in-memory RunSummary schema with any header already on disk,
        # then reuse that decision for every subsequent row so values stay aligned.
        if self._write_header is None:
            self._write_header = self._resolve_header(output_path, header)

        write_header = self._write_header
        needs_header = not output_path.exists() or output_path.stat().st_size == 0

        with output_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=write_header, quoting=csv.QUOTE_ALL)
            if needs_header:
                writer.writeheader()
            # Escape embedded newlines in string fields so each run occupies exactly
            # one CSV line — prevents pandas from skipping multi-line rows on read.
            sanitized = {
                k: v.replace("\n", "\\n").replace("\r", "\\r") if isinstance(v, str) else v
                for k, v in row.items()
                if k in write_header
            }
            writer.writerow(sanitized)

    @staticmethod
    def _resolve_header(output_path: Path, header: list[str]) -> list[str]:
        """Return the column order to write, reconciling against an existing file.

        Handles QUOTE_ALL-style headers (fields wrapped in double-quotes). Raises
        if the on-disk file has columns unknown to the current RunSummary.
        """
        if not output_path.exists():
            return header
        with output_path.open("r", encoding="utf-8") as existing:
            first_line = existing.readline().strip()
        if not first_line:
            return header

        existing_header = next(csv.reader([first_line]))
        if existing_header == header:
            return header
        if all(field_name in header for field_name in existing_header):
            missing = [field_name for field_name in header if field_name not in existing_header]
            logger.warning(
                "Existing CSV schema at %s is missing %d current field(s): %s. "
                "Appending with the existing header.",
                output_path,
                len(missing),
                ", ".join(missing),
            )
            return existing_header

        extra = [field_name for field_name in existing_header if field_name not in header]
        raise RuntimeError(
            "Existing CSV schema does not match RunSummary for "
            f"{output_path}. Unexpected field(s): {extra}. "
            "Migrate or replace the file before logging."
        )