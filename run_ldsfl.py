#!/usr/bin/env python3
from __future__ import annotations

import os

# Defaults to avoid thread oversubscription. Users can override by exporting
# these variables before running.
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
from pathlib import Path

from ldsfl.main import run_project


def parse_cases(value: str) -> list[int]:
    """Parse comma-separated case numbers and inclusive ranges.

    Examples
    --------
    ``1`` -> ``[1]``
    ``1,3-5`` -> ``[1, 3, 4, 5]``
    ``1,2,1,2-3`` -> ``[1, 2, 3]``
    """
    if value is None or not value.strip():
        raise argparse.ArgumentTypeError("--cases cannot be empty; omit --cases to run all cases.")

    parsed: list[int] = []
    seen: set[int] = set()

    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            raise argparse.ArgumentTypeError("Empty item in --cases. Use values like '1,3-5'.")

        if "-" in part:
            bounds = part.split("-", 1)
            if len(bounds) != 2 or not bounds[0].strip() or not bounds[1].strip():
                raise argparse.ArgumentTypeError(f"Invalid case range: {part!r}.")
            try:
                start = int(bounds[0])
                end = int(bounds[1])
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"Invalid case range: {part!r}.") from exc
            if start < 1 or end < 1:
                raise argparse.ArgumentTypeError("Case numbers must be positive integers.")
            if end < start:
                raise argparse.ArgumentTypeError(f"Invalid decreasing case range: {part!r}.")
            values = range(start, end + 1)
        else:
            try:
                case = int(part)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"Invalid case number: {part!r}.") from exc
            if case < 1:
                raise argparse.ArgumentTypeError("Case numbers must be positive integers.")
            values = [case]

        for case in values:
            if case not in seen:
                parsed.append(case)
                seen.add(case)

    return parsed


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the LDSFL-Meander reduced meander model.")
    ap.add_argument("--base-dir", type=Path, default=Path("."), help="Folder containing Input/ and Output/")
    ap.add_argument("--cases", type=parse_cases, default=None, help="Cases to run, e.g. '1,3-5'. Omit to run all cases.")
    ap.add_argument("--nprint", type=int, default=10000)
    ap.add_argument("--ntstep", type=int, default=100000)
    ap.add_argument("--max-cut", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=0, help="Safety cap on iterations; 0=disabled")
    ap.add_argument("--max-sim-time", type=float, default=0.0, help="Maximum cumulative simulated time; 0=disabled")
    ap.add_argument("--stop-mode", choices=["first", "all"], default="first", help="How enabled stop criteria are combined.")
    ap.add_argument("--stop-on-steps", type=int, default=1, choices=[0, 1])
    ap.add_argument("--stop-on-time", type=int, default=0, choices=[0, 1])
    ap.add_argument("--stop-on-cutoffs", type=int, default=1, choices=[0, 1])
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--cstab", type=float, default=0.01, help="Timestep stability coefficient")
    ap.add_argument("--geometry-smoothing", type=int, default=1, choices=[0, 1], help="Enable geometry smoothing/filtering")
    ap.add_argument("--geometry-smoothing-factor", type=float, default=8.0)
    ap.add_argument("--neck-cutoff-interval", type=int, default=3)
    ap.add_argument("--resample-upper-factor", type=float, default=1.03)
    ap.add_argument("--resample-lower-factor", type=float, default=0.97)
    ap.add_argument(
        "--output-units",
        default="dimensionless",
        choices=["dimensionless", "dimensional"],
        help="Write selected outputs in dimensionless or dimensional form.",
    )
    ap.add_argument(
        "--output-length-scale",
        type=float,
        default=1.0,
        help="Length scale used when dimensional outputs are requested.",
    )
    ap.add_argument(
        "--output-velocity-scale",
        type=float,
        default=1.0,
        help="Velocity scale used when dimensional outputs are requested.",
    )
    ap.add_argument("--flow-bc", default="free", choices=["free", "periodic"])
    ap.add_argument("--flow-paral", type=int, default=0, choices=[0, 1])
    ap.add_argument("--flow-workers", type=int, default=0)  # 0 = auto
    ap.add_argument(
        "--backend",
        default="numpy",
        choices=["numpy", "numba"],
        help=(
            "Flow-field compute backend. 'numpy' is default. "
            "'numba' enables optional CPU JIT acceleration (requires numba)."
        ),
    )
    ap.add_argument(
        "--numba-parallel",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "(backend=numba) Use Numba parallel=True for inner kernels. "
            "Can speed up large N, but may oversubscribe if --flow-paral=1."
        ),
    )
    ap.add_argument(
        "--numba-fastmath",
        type=int,
        default=0,
        choices=[0, 1],
        help="(backend=numba) Use Numba fastmath=True (may slightly change results).",
    )
    args = ap.parse_args()

    max_steps = None if args.max_steps == 0 else args.max_steps
    max_sim_time = None if args.max_sim_time == 0 else args.max_sim_time
    results = run_project(
        args.base_dir,
        cases=args.cases,
        Nprint=args.nprint,
        Ntstep=args.ntstep,
        Max_Cut=args.max_cut,
        max_steps=max_steps,
        do_plots=not args.no_plots,
        max_sim_time=max_sim_time,
        stop_mode=args.stop_mode,
        stop_on_steps=bool(args.stop_on_steps),
        stop_on_time=bool(args.stop_on_time),
        stop_on_cutoffs=bool(args.stop_on_cutoffs),
        cstab=args.cstab,
        geometry_smoothing_enabled=bool(args.geometry_smoothing),
        geometry_smoothing_factor=args.geometry_smoothing_factor,
        neck_cutoff_interval=args.neck_cutoff_interval,
        resample_upper_factor=args.resample_upper_factor,
        resample_lower_factor=args.resample_lower_factor,
        output_units=args.output_units,
        output_length_scale=args.output_length_scale,
        output_velocity_scale=args.output_velocity_scale,
        flow_bc=args.flow_bc,
        flow_paral=args.flow_paral,
        flow_workers=args.flow_workers,
        flow_backend=args.backend,
        numba_parallel=bool(args.numba_parallel),
        numba_fastmath=bool(args.numba_fastmath),
    )
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
