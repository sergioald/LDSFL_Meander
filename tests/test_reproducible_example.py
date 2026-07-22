"""Check the bundled reproducibility fixture, which nothing checked before.

``examples/reproducible_case1_short/`` ships an expected output tree, an
expected run summary and ~370 KB of reference CSVs. No test read any of it, so
the fixture drifted: ``expected_tree.txt`` was missing a ``var_*_6.csv`` entry
that the documented command produces, and the pinned floating-point values no
longer reproduce on a current scientific-Python stack (``dt_cum`` differs by
about 9% after five steps).

The checks are split by how portable they are:

* structure and integer counters must match exactly everywhere;
* physical invariants must hold everywhere;
* the pinned floating-point values are only compared when
  ``environment.json`` records the same library versions, because BLAS, NumPy
  and SciPy all move the last digits. Regenerate both together with
  ``examples/reproducible_case1_short/regenerate.py``.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ldsfl.main import run_case

FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "reproducible_case1_short"
CASE_ID = "1_9_0005_03_2_05"


@pytest.fixture(scope="module")
def example_run(tmp_path_factory):
    """Run the case documented in the fixture README."""
    workspace = tmp_path_factory.mktemp("reproducible")
    repo_root = Path(__file__).resolve().parents[1]
    (workspace / "Input").mkdir()
    for name in ("Parameter.csv", "xy.csv"):
        (workspace / "Input" / name).write_bytes((repo_root / "Input" / name).read_bytes())

    # Same parameters as the command in the fixture README; Ntstep is the
    # snapshot label baked into output filenames, so it must match too.
    summary = run_case(
        workspace, 1, Nprint=2, Ntstep=100000, Max_Cut=1, max_steps=5, do_plots=False
    )
    return workspace, summary


def _expected_summary() -> dict:
    return json.loads((FIXTURE / "expected_summary.json").read_text(encoding="utf-8"))


def test_output_tree_matches_the_recorded_tree(example_run):
    workspace, _summary = example_run
    produced_root = workspace / "Output"
    produced = sorted(
        str(p.relative_to(produced_root)).replace("\\", "/") + ("/" if p.is_dir() else "")
        for p in produced_root.rglob("*")
    )
    recorded = [
        line.strip().replace("expected_output/", "")
        for line in (FIXTURE / "expected_tree.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert produced == recorded, (
        f"missing from output: {sorted(set(recorded) - set(produced))}; "
        f"unexpected extras: {sorted(set(produced) - set(recorded))}"
    )


def test_integer_counters_match_the_recorded_summary(example_run):
    _workspace, summary = example_run
    expected = _expected_summary()
    for key in ("id_files", "steps", "cut_cnt", "jt"):
        assert summary[key] == expected[key], key


def test_stop_reason_matches_the_recorded_summary(example_run):
    _workspace, summary = example_run
    assert summary["stop_criteria_reached"] == _expected_summary()["stop_criteria_reached"]


def test_outputs_are_physically_sane(example_run):
    """Invariants that hold regardless of library versions."""
    workspace, summary = example_run
    assert summary["sinuo_final"] >= 1.0
    assert summary["dt_cum"] > 0.0

    snapshots = sorted((workspace / "Output" / CASE_ID / "xyu").glob("xyu_*.csv"))
    assert snapshots, "no centreline snapshots were written"
    for path in snapshots:
        frame = pd.read_csv(path)
        assert set(["x", "y", "s", "th", "c", "U"]).issubset(frame.columns)
        assert np.isfinite(frame.to_numpy(dtype=float)).all(), f"non-finite values in {path.name}"
        assert np.all(np.diff(frame["s"].to_numpy()) > 0.0), f"arclength not increasing in {path.name}"


def test_sinuosity_history_is_monotone_and_starts_near_one(example_run):
    workspace, _summary = example_run
    history = pd.read_csv(
        workspace / "Output" / CASE_ID / "files" / f"sinuosity_history_{CASE_ID}.csv"
    )
    values = history.iloc[:, -1].to_numpy(dtype=float)
    assert values[0] == pytest.approx(1.0, abs=1.0e-3)
    assert np.all(values >= 1.0 - 1.0e-12)


def test_recorded_floating_point_values_when_the_environment_matches(example_run):
    """Only meaningful on the machine that generated the fixture."""
    environment_path = FIXTURE / "environment.json"
    if not environment_path.exists():
        pytest.skip(
            "environment.json is absent, so the pinned floats cannot be attributed "
            "to a known stack; run examples/reproducible_case1_short/regenerate.py"
        )

    recorded = json.loads(environment_path.read_text(encoding="utf-8"))
    current = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for name in ("numpy", "scipy", "pandas"):
        try:
            current[name] = __import__(name).__version__
        except Exception:
            current[name] = None
    try:
        current["blas"] = str(np.__config__.show(mode="dicts")).replace("\n", " ")[:400]
    except Exception:
        current["blas"] = None

    keys = ("python", "platform", "numpy", "scipy", "pandas", "blas")
    mismatched = {k: (recorded.get(k), current.get(k)) for k in keys if recorded.get(k) != current.get(k)}
    if mismatched:
        pytest.skip(f"library/BLAS/platform differ from the recorded fixture: {mismatched}")

    _workspace, summary = example_run
    expected = _expected_summary()
    assert summary["dt_cum"] == pytest.approx(expected["dt_cum"], rel=1.0e-6)
    assert summary["sinuo_final"] == pytest.approx(expected["sinuo_final"], rel=1.0e-9)
