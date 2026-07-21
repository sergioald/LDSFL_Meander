#!/usr/bin/env python3
"""Diagnose index-gradient versus arclength-scaled curvature.

This script is intentionally diagnostic-only. It does not import or change the
solver implementation. It compares the current curvature pattern used in the
code base, ``gradient(theta)``, with the arclength-aware alternative,
``gradient(theta, s)``, on manufactured constant-curvature arcs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CurvatureDiagnosticCase:
    """Summary of one manufactured constant-curvature diagnostic case."""

    name: str
    radius: float
    target_spacing: float
    mean_spacing: float
    n_points: int
    true_abs_curvature: float
    index_gradient_abs_median: float
    arclength_gradient_abs_median: float
    index_gradient_over_true: float
    arclength_gradient_over_true: float
    index_gradient_max_abs_error: float
    arclength_gradient_max_abs_error: float


def arclength(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return cumulative centreline arclength."""
    dx = np.diff(np.asarray(x, dtype=np.float64))
    dy = np.diff(np.asarray(y, dtype=np.float64))
    return np.concatenate(([0.0], np.cumsum(np.sqrt(dx * dx + dy * dy))))


def solver_style_tangent_angle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the same sign convention used by the solver geometry path."""
    dx = np.gradient(np.asarray(x, dtype=np.float64))
    dy = np.gradient(np.asarray(y, dtype=np.float64))
    return -1.0 * np.unwrap(np.arctan2(dy, dx))


def index_gradient_curvature(theta: np.ndarray) -> np.ndarray:
    """Current index-gradient curvature pattern: dtheta per point index."""
    return np.gradient(np.asarray(theta, dtype=np.float64))


def arclength_gradient_curvature(theta: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Arclength-scaled curvature pattern: dtheta/ds."""
    return np.gradient(np.asarray(theta, dtype=np.float64), np.asarray(s, dtype=np.float64))


def circular_arc(radius: float, target_spacing: float, arc_angle: float = np.pi / 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Build a constant-curvature circular arc with approximately target spacing."""
    arc_length = float(radius) * float(arc_angle)
    n_points = max(9, int(round(arc_length / float(target_spacing))) + 1)
    phi = np.linspace(0.0, float(arc_angle), n_points, dtype=np.float64)
    x = float(radius) * np.sin(phi)
    y = float(radius) * (1.0 - np.cos(phi))
    return x, y


def diagnose_case(name: str, *, radius: float, target_spacing: float) -> CurvatureDiagnosticCase:
    """Compare index-gradient and arclength-gradient curvature for one arc."""
    x, y = circular_arc(radius=radius, target_spacing=target_spacing)
    s = arclength(x, y)
    theta = solver_style_tangent_angle(x, y)
    c_index = index_gradient_curvature(theta)
    c_arclength = arclength_gradient_curvature(theta, s)

    # Avoid first/last two samples because endpoint finite differences are noisier.
    interior = slice(2, -2)
    true_abs = 1.0 / float(radius)
    index_abs = np.abs(c_index[interior])
    arclength_abs = np.abs(c_arclength[interior])
    mean_spacing = float(np.mean(np.diff(s)))

    return CurvatureDiagnosticCase(
        name=name,
        radius=float(radius),
        target_spacing=float(target_spacing),
        mean_spacing=mean_spacing,
        n_points=int(x.size),
        true_abs_curvature=true_abs,
        index_gradient_abs_median=float(np.median(index_abs)),
        arclength_gradient_abs_median=float(np.median(arclength_abs)),
        index_gradient_over_true=float(np.median(index_abs) / true_abs),
        arclength_gradient_over_true=float(np.median(arclength_abs) / true_abs),
        index_gradient_max_abs_error=float(np.max(np.abs(index_abs - true_abs))),
        arclength_gradient_max_abs_error=float(np.max(np.abs(arclength_abs - true_abs))),
    )


def run_diagnostic() -> list[CurvatureDiagnosticCase]:
    """Run representative spacing cases around the current unit-spacing assumption."""
    radius = 20.0
    return [
        diagnose_case("fine-spacing arc", radius=radius, target_spacing=0.25),
        diagnose_case("near-unit-spacing arc", radius=radius, target_spacing=1.0),
        diagnose_case("coarse-spacing arc", radius=radius, target_spacing=2.0),
    ]


def format_table(rows: Sequence[CurvatureDiagnosticCase]) -> str:
    """Format diagnostics as a compact plain-text table."""
    header = (
        "case                     n   mean_ds   true_|c|   "
        "index_med_|c|   arc_med_|c|   index/true   arc/true"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for row in rows:
        lines.append(
            f"{row.name:<24} "
            f"{row.n_points:>4d} "
            f"{row.mean_spacing:>9.5f} "
            f"{row.true_abs_curvature:>10.6f} "
            f"{row.index_gradient_abs_median:>14.6f} "
            f"{row.arclength_gradient_abs_median:>12.6f} "
            f"{row.index_gradient_over_true:>12.6f} "
            f"{row.arclength_gradient_over_true:>10.6f}"
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a text table.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = run_diagnostic()
    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2))
    else:
        print("Curvature arclength-scaling diagnostic")
        print()
        print(format_table(rows))
        print()
        print("Interpretation:")
        print("  index-gradient curvature is dtheta per saved point index.")
        print("  arclength-gradient curvature is dtheta/ds.")
        print("  If mean_ds is not close to 1, index-gradient curvature scales with mean_ds.")
        print("  This diagnostic does not change solver behaviour.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
