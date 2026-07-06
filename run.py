"""Unified benchmark runner.

Usage:
    python run.py --benchmark officebench --config configs/officebench/<experiment>.yaml [flags...]
    python run.py --benchmark gaia --config configs/gaia/<experiment>.yaml [flags...]
"""

from __future__ import annotations

import os
import sys

BENCHMARKS = ("officebench", "gaia")


def main() -> None:
    # Extract dispatch-only flags before argparse; each benchmark owns the rest.
    benchmark = "officebench"
    config_file = None
    argv = sys.argv[1:]
    cleaned = []
    i = 0
    while i < len(argv):
        if argv[i] == "--benchmark" and i + 1 < len(argv):
            benchmark = argv[i + 1]
            i += 2
        elif argv[i].startswith("--benchmark="):
            benchmark = argv[i].split("=", 1)[1]
            i += 1
        elif argv[i] == "--config" and i + 1 < len(argv):
            config_file = argv[i + 1]
            i += 2
        elif argv[i].startswith("--config="):
            config_file = argv[i].split("=", 1)[1]
            i += 1
        else:
            cleaned.append(argv[i])
            i += 1
    sys.argv = [sys.argv[0]] + cleaned
    if config_file:
        os.environ["BENCHMARK_CONFIG"] = config_file

    if benchmark == "officebench":
        from benchmarks.officebench.runner import main as run_main
    elif benchmark == "gaia":
        from benchmarks.gaia.runner import main as run_main
    else:
        print(f"Unknown benchmark: {benchmark!r}. Choose from: {', '.join(BENCHMARKS)}")
        sys.exit(1)

    run_main()


if __name__ == "__main__":
    main()
