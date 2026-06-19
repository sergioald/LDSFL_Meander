"""Short regression/smoke runner for LDSFL-Meander.

This script runs a small deterministic simulation using the local project folder
by default. It is intended as a quick post-edit check for the source tree.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ldsfl.main import run_project


def main() -> None:
    project_root = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(description="Run a short LDSFL-Meander regression/smoke case.")
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=project_root,
        help="Folder containing Input/ and where Output/ will be created.",
    )
    ap.add_argument("--cases", type=str, default="1", help="Cases to run, e.g. '1' or '1,2'.")
    ap.add_argument("--max-steps", type=int, default=20, help="Short safety cap for regression runs.")
    ap.add_argument("--nprint", type=int, default=10, help="Output interval.")
    ap.add_argument("--ntstep", type=int, default=1, help="Timestep label used in filenames.")
    ap.add_argument("--max-cut", type=int, default=3, help="Maximum number of cutoffs.")
    ap.add_argument("--plots", action="store_true", help="Enable plots during the regression run.")
    args = ap.parse_args()

    cases = []
    for item in args.cases.split(','):
        item = item.strip()
        if item:
            cases.append(int(item))

    results = run_project(
        args.base_dir,
        cases=cases,
        Nprint=args.nprint,
        Ntstep=args.ntstep,
        Max_Cut=args.max_cut,
        max_steps=args.max_steps,
        do_plots=args.plots,
    )

    for result in results:
        print(result)


if __name__ == "__main__":
    main()
