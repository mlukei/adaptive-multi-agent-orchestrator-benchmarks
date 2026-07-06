"""Read prior result rows to drive the runners' ``--retry-*`` flags."""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


def load_result_keys(
    variant: str,
    *,
    key_of: Callable[[dict[str, str]], str],
    keep: Callable[[dict[str, str]], bool],
    label: str,
    warn_if_missing: bool,
) -> set[str]:
    """Collect task keys from ``results/<variant>.csv`` for rows where ``keep`` holds.

    ``key_of`` maps a row to its task key; rows whose key is empty are skipped.
    ``label`` prefixes the log lines (e.g. ``--retry-timeouts``).
    """
    csv_path = Path(f"results/{variant}.csv")
    if not csv_path.exists():
        if warn_if_missing:
            logger.warning("%s: CSV not found at %s", label, csv_path)
        return set()

    with csv_path.open(newline="", encoding="utf-8") as f:
        keys = {
            key for row in csv.DictReader(f) if keep(row) and (key := key_of(row))
        }
    logger.info("%s: found %d matching task(s) in %s", label, len(keys), csv_path)
    return keys
