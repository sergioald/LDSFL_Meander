"""Convenience launcher for an LDSFL-Meander run from the project folder."""
from __future__ import annotations

import argparse
from pathlib import Path

from ldsfl.main import run_project


def main() -> None:
    project_root = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(description="Run LDSFL-Meander from the local project folder.")
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=project_root,
        help="Folder containing Input/ and where Output/ will be created.",
    )
    ap.add_argument("--cases", type=str, default="1", help="Cases to run, e.g. '1' or '1,2'.")
    # numpy is the safe default: numba is an optional extra and may not be installed.
    ap.add_argument("--backend", choices=["numpy", "numba"], default="numpy")
    ap.add_argument("--max-steps", type=int, default=0, help="0 means unlimited.")
    ap.add_argument("--nprint", type=int, default=10)
    ap.add_argument("--ntstep", type=int, default=100000)
    ap.add_argument("--max-cut", type=int, default=4)
    ap.add_argument("--plots", action="store_true", help="Enable plots.")
    ap.add_argument("--numba-parallel", action="store_true")
    ap.add_argument("--numba-fastmath", action="store_true")
    args = ap.parse_args()

    cases = []
    for item in args.cases.split(','):
        item = item.strip()
        if item:
            cases.append(int(item))

    run_project(
        args.base_dir,
        cases=cases,
        flow_backend=args.backend,
        numba_parallel=args.numba_parallel,
        numba_fastmath=args.numba_fastmath,
        do_plots=args.plots,
        max_steps=None if args.max_steps == 0 else args.max_steps,
        Nprint=args.nprint,
        Ntstep=args.ntstep,
        Max_Cut=args.max_cut,
    )


if __name__ == "__main__":
    main()
