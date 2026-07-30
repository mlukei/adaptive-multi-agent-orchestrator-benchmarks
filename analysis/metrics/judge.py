"""Judge agreement metrics for adaptive-system training splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from loader import OUTPUT_DIR, percentage, require_output_file

from .overall import per_split_metrics


TRAINING_SOURCES = {
    ("OfficeBench", "Rich"): tuple(
        OUTPUT_DIR
        / "officebench"
        / "rich"
        / "adaptive"
        / f"train_memory_fold_{split}.csv"
        for split in (1, 2, 3)
    ),
    ("OfficeBench", "Sparse"): tuple(
        OUTPUT_DIR
        / "officebench"
        / "sparse"
        / "adaptive"
        / f"train_memory_fold_{split}.csv"
        for split in (1, 2, 3)
    ),
    ("GAIA", "Rich"): tuple(
        OUTPUT_DIR
        / "gaia"
        / "rich"
        / "adaptive"
        / f"adaptive_training_rich_fold_{split}.csv"
        for split in (1, 2, 3)
    ),
    ("GAIA", "Sparse"): tuple(
        OUTPUT_DIR
        / "gaia"
        / "sparse"
        / "adaptive"
        / f"adaptive_training_sparse_fold_{split}.csv"
        for split in (1, 2, 3)
    ),
}


DISPLAY_COLUMNS = [
    "Benchmark Success (%)",
    "Judge Accept (%)",
    "Positive Label Precision (%)",
]


def load_training_runs() -> pd.DataFrame:
    """Load and validate all rich/sparse adaptive training splits."""
    frames = []
    for (benchmark, cards), paths in TRAINING_SOURCES.items():
        for split, path in enumerate(paths, start=1):
            frames.append(_load_training_split(path, benchmark, cards, split))
    return pd.concat(frames, ignore_index=True, sort=False)


def agreement_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Return the three split-averaged judge-label metrics."""
    rows = []
    for (benchmark, cards), group in runs.groupby(
        ["Benchmark", "Cards"],
        sort=False,
    ):
        split_metrics = per_split_metrics(
            group,
            _agreement_metrics,
            split_column="Split",
        )
        metrics = split_metrics[DISPLAY_COLUMNS].mean().to_dict()
        rows.append({"Benchmark": benchmark, "Cards": cards, **metrics})

    table = pd.DataFrame(rows).set_index(["Benchmark", "Cards"])
    return table.round({column: 1 for column in DISPLAY_COLUMNS})


def _load_training_split(
    path: Path,
    benchmark: str,
    cards: str,
    split: int,
) -> pd.DataFrame:
    frame = pd.read_csv(require_output_file(path), engine="python")
    return pd.DataFrame(
        {
            "Benchmark": benchmark,
            "Cards": cards,
            "Split": split,
            "Benchmark Success": frame["success"].astype(int).astype(bool),
            "Judge Accept": frame["judge_accepted"].astype(int).astype(bool),
        }
    )

def _agreement_metrics(group: pd.DataFrame) -> dict[str, float]:
    truth = group["Benchmark Success"]
    accepted = group["Judge Accept"]
    total = len(group)
    true_positive = int((truth & accepted).sum())
    false_negative = int((truth & ~accepted).sum())
    false_positive = int((~truth & accepted).sum())
    return {
        "Benchmark Success (%)": percentage(true_positive + false_negative, total),
        "Judge Accept (%)": percentage(true_positive + false_positive, total),
        "Positive Label Precision (%)": percentage(
            true_positive,
            true_positive + false_positive,
        ),
    }
