"""Command line entry point: ``python -m kalman_research.experiments [names] [options]``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import REGISTRY


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Kalman filter experiments.")
    parser.add_argument("names", nargs="*", default=list(REGISTRY), help=f"subset of: {', '.join(REGISTRY)}")
    parser.add_argument("--out", type=Path, default=Path("results"), help="output directory")
    parser.add_argument("--runs", type=int, default=None, help="Monte Carlo runs (default: per experiment)")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        if name not in REGISTRY:
            parser.error(f"unknown experiment '{name}'")
        start = time.perf_counter()
        summary = REGISTRY[name].run(args.out, args.runs, args.seed)
        (args.out / f"{name}.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"{name}: done in {time.perf_counter() - start:.1f}s -> {args.out / (name + '.json')}")


if __name__ == "__main__":
    main()
