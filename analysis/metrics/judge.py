"""Judge agreement metrics for adaptive-system training folds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from loader import OUTPUT_DIR, require_output_file


@dataclass(frozen=True)
class TrainingSource:
    benchmark: str
    cards: str
    paths: tuple[Path, ...]


TRAINING_SOURCES = (
    TrainingSource(
        "OfficeBench",
        "Rich",
        tuple(
            OUTPUT_DIR
            / "officebench"
            / "rich"
            / "adaptive"
            / f"train_memory_fold_{fold}.csv"
            for fold in (1, 2, 3)
        ),
    ),
    TrainingSource(
        "OfficeBench",
        "Sparse",
        tuple(
            OUTPUT_DIR
            / "officebench"
            / "sparse"
            / "adaptive"
            / f"train_memory_fold_{fold}.csv"
            for fold in (1, 2, 3)
        ),
    ),
    TrainingSource(
        "GAIA",
        "Rich",
        tuple(
            OUTPUT_DIR
            / "gaia"
            / "rich"
            / "adaptive"
            / f"adaptive_training_rich_fold_{fold}.csv"
            for fold in (1, 2, 3)
        ),
    ),
    TrainingSource(
        "GAIA",
        "Sparse",
        tuple(
            OUTPUT_DIR
            / "gaia"
            / "sparse"
            / "adaptive"
            / f"adaptive_training_sparse_fold_{fold}.csv"
            for fold in (1, 2, 3)
        ),
    ),
)

REQUIRED_COLUMNS = {"success", "judge_accepted", "tool_call_history"}


def load_training_runs() -> pd.DataFrame:
    """Load and validate all rich/sparse adaptive training folds."""
    frames = []
    for source in TRAINING_SOURCES:
        for fold, path in enumerate(source.paths, start=1):
            frames.append(_load_training_fold(path, source, fold))
    return pd.concat(frames, ignore_index=True, sort=False)


def agreement_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Return pooled judge-vs-benchmark agreement per benchmark and card type."""
    rows = [
        {"Benchmark": benchmark, "Cards": cards, **_agreement_metrics(group)}
        for (benchmark, cards), group in runs.groupby(["Benchmark", "Cards"], sort=False)
    ]
    return pd.DataFrame(rows).set_index(["Benchmark", "Cards"]).round(1)


def _load_training_fold(
    path: Path,
    source: TrainingSource,
    fold: int,
) -> pd.DataFrame:
    frame = pd.read_csv(require_output_file(path), engine="python")
    _validate_fold(frame, path)
    return pd.DataFrame(
        {
            "Benchmark": source.benchmark,
            "Cards": source.cards,
            "Fold": fold,
            "Benchmark Success": frame["success"].astype(int).astype(bool),
            "Judge Accept": frame["judge_accepted"].astype(int).astype(bool),
        }
    )


def _validate_fold(frame: pd.DataFrame, path: Path) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    for column in ("success", "judge_accepted"):
        invalid = set(frame[column].dropna().astype(int)) - {0, 1}
        if invalid:
            raise ValueError(f"{path} has invalid {column} labels: {sorted(invalid)}")
    judge_reviewed = frame["tool_call_history"].map(_has_judge_review)
    if not judge_reviewed.all():
        raise ValueError(f"{path} has {int((~judge_reviewed).sum())} rows without judge_review")


def _has_judge_review(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        history = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("tool_call_history contains invalid JSON") from error
    return isinstance(history, list) and "judge_review" in history


def _agreement_metrics(group: pd.DataFrame) -> dict[str, float | int]:
    truth = group["Benchmark Success"]
    accepted = group["Judge Accept"]
    total = len(group)
    true_positive = int((truth & accepted).sum())
    predicted_positive = int(accepted.sum())
    return {
        "N": total,
        "Benchmark Success (%)": _rate(int(truth.sum()), total),
        "Judge Accept (%)": _rate(predicted_positive, total),
        "Positive Label Precision (%)": _rate(true_positive, predicted_positive),
    }


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator * 100 if denominator else 0.0
