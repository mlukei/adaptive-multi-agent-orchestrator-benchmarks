"""
Load experiment CSVs for paper metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"


def require_output_file(path: Path) -> Path:
    """Return an existing file, provided it is inside the canonical output tree."""
    resolved_path = path.resolve()
    resolved_path.relative_to(OUTPUT_DIR.resolve())

    return resolved_path


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    real_agents: frozenset[str]
    gold_path: Path
    card_paths: dict[str, dict[str, tuple[Path, ...]]]
    system_order: list[str]

REAL_AGENTS_OFFICEBENCH: frozenset[str] = frozenset(
    ["calendar", "email", "excel", "llm", "ocr", "pdf", "shell", "word"]
)


REAL_GAIA_AGENTS: frozenset[str] = frozenset(
    [
        "web_surfer",
        "code_exec",
        "file_surfer",
    ]
)

NUMERIC_COLUMNS = [
    "success",
    "tokens_total",
    "cost_total",
    "total_delegations",
]

COLUMN_ALIASES: dict[str, str] = {
    "total_tokens": "tokens_total",
    "num_turns": "total_delegations",
}


def load_runs(paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate multiple CSVs of OfficeBench runs."""
    frames: list[pd.DataFrame] = []
    for path in paths:
        frames.append(pd.read_csv(require_output_file(path), engine="python"))
    runs = pd.concat(frames, ignore_index=True, sort=False)
    runs = _apply_aliases(runs)
    return _coerce_numeric(runs)


def add_derived_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    """Add metrics to the OfficeBench runs DataFrame."""
    df = runs.copy()
    df["task_key"] = df["task_id"].astype(str) + "/" + df["subtask_id"].astype(str)
    df["is_success"] = (df["success"] == 1).astype(int)
    return df


def _apply_aliases(runs: pd.DataFrame) -> pd.DataFrame:
    """Rename columns for display"""
    df = runs.copy()
    for alias, canonical in COLUMN_ALIASES.items():
        if alias not in df.columns:
            continue
        if canonical not in df.columns:
            df = df.rename(columns={alias: canonical})
        else:
            df[canonical] = df[canonical].fillna(df[alias])
    return df


def _coerce_numeric(runs: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns to numbers and errors to NaN."""
    df = runs.copy()
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def percentage(numerator: int, denominator: int) -> float:
    return numerator / denominator * 100 if denominator else 0.0
