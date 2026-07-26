"""Benchmark runner.

Usage:
    python run.py --benchmark officebench --config configs/officebench/<experiment>.yaml
    python run.py --benchmark gaia --config configs/gaia/<experiment>.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

BENCHMARKS = ("officebench", "gaia")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="officebench")
    parser.add_argument("--config")
    dispatch, benchmark_args = parser.parse_known_args()

    sys.argv = [sys.argv[0], *benchmark_args]
    if dispatch.config:
        os.environ["BENCHMARK_CONFIG"] = dispatch.config

    if dispatch.benchmark == "officebench":
        from benchmarks.officebench.runner import main as run_main
    else:
        from benchmarks.gaia.runner import main as run_main

    run_main()


if __name__ == "__main__":
    main()
