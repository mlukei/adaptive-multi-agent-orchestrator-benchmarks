"""Judge agreement metrics for adaptive-system training folds."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from loader import OUTPUT_DIR, percentage, require_output_file


TRAINING_SOURCES = {
    ("OfficeBench", "Rich"): tuple(
        OUTPUT_DIR
        / "officebench"
        / "rich"
        / "adaptive"
        / f"train_memory_fold_{fold}.csv"
        for fold in (1, 2, 3)
    ),
    ("OfficeBench", "Sparse"): tuple(
        OUTPUT_DIR
        / "officebench"
        / "sparse"
        / "adaptive"
        / f"train_memory_fold_{fold}.csv"
        for fold in (1, 2, 3)
    ),
    ("GAIA", "Rich"): tuple(
        OUTPUT_DIR
        / "gaia"
        / "rich"
        / "adaptive"
        / f"adaptive_training_rich_fold_{fold}.csv"
        for fold in (1, 2, 3)
    ),
    ("GAIA", "Sparse"): tuple(
        OUTPUT_DIR
        / "gaia"
        / "sparse"
        / "adaptive"
        / f"adaptive_training_sparse_fold_{fold}.csv"
        for fold in (1, 2, 3)
    ),
}

REQUIRED_COLUMNS = {"success", "judge_accepted", "tool_call_history"}


def load_training_runs() -> pd.DataFrame:
    """Load and validate all rich/sparse adaptive training folds."""
    frames = []
    for (benchmark, cards), paths in TRAINING_SOURCES.items():
        for fold, path in enumerate(paths, start=1):
            frames.append(_load_training_fold(path, benchmark, cards, fold))
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
    benchmark: str,
    cards: str,
    fold: int,
) -> pd.DataFrame:
    frame = pd.read_csv(require_output_file(path), engine="python")
    _validate_fold(frame, path)
    return pd.DataFrame(
        {
            "Benchmark": benchmark,
            "Cards": cards,
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


def _has_judge_review(value: str) -> bool:
    return "judge_review" in json.loads(value)


def _agreement_metrics(group: pd.DataFrame) -> dict[str, float | int]:
    truth = group["Benchmark Success"]
    accepted = group["Judge Accept"]
    total = len(group)
    true_positive = int((truth & accepted).sum())
    predicted_positive = int(accepted.sum())
    return {
        "N": total,
        "Benchmark Success (%)": percentage(int(truth.sum()), total),
        "Judge Accept (%)": percentage(predicted_positive, total),
        "Positive Label Precision (%)": percentage(true_positive, predicted_positive),
    }
