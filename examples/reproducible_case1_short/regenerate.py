#!/usr/bin/env python3
"""Regenerate the bundled reproducibility fixture and record the environment.

Run from the repository root on the machine that should define the reference:

    python examples/reproducible_case1_short/regenerate.py

This rewrites ``expected_summary.json``, ``expected_tree.txt``,
``environment.json`` and ``expected_output/``.

The helper runs in a temporary workspace so it does not delete or overwrite
unrelated local simulation outputs under the repository-level ignored
``Output/`` directory.
"""

from __future__ import annotations

import ast
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
REPO = FIXTURE.parents[1]
CASE_ID = "1_9_0005_03_2_05"


def _versions() -> dict:
    record = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for name in ("numpy", "scipy", "pandas", "matplotlib"):
        try:
            record[name] = __import__(name).__version__
        except Exception:
            record[name] = None
    try:
        import numpy as np

        record["blas"] = str(np.__config__.show(mode="dicts")).replace("\n", " ")[:400]
    except Exception:
        record["blas"] = None
    return record


def _run_fixture(workspace: Path) -> dict:
    (workspace / "Input").mkdir()
    for name in ("Parameter.csv", "xy.csv"):
        shutil.copy2(REPO / "Input" / name, workspace / "Input" / name)

    command = [
        sys.executable,
        str(REPO / "run_ldsfl.py"),
        "--base-dir",
        str(workspace),
        "--cases",
        "1",
        "--max-steps",
        "5",
        "--nprint",
        "2",
        "--max-cut",
        "1",
        "--no-plots",
    ]

    completed = subprocess.run(
        command,
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )

    if completed.stdout.strip():
        print(completed.stdout.strip())

    for line in reversed([line.strip() for line in completed.stdout.splitlines() if line.strip()]):
        if line.startswith("{") and line.endswith("}"):
            return ast.literal_eval(line)

    raise SystemExit("could not parse run summary from run_ldsfl.py output")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ldsfl_repro_fixture_") as tmp:
        workspace = Path(tmp)
        summary = _run_fixture(workspace)

        produced = workspace / "Output" / CASE_ID
        if not produced.exists():
            raise SystemExit(f"expected output folder not found: {produced}")

        target = FIXTURE / "expected_output"
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(workspace / "Output", target)

    entries = sorted(
        str(p.relative_to(target)).replace("\\", "/") + ("/" if p.is_dir() else "")
        for p in target.rglob("*")
    )
    (FIXTURE / "expected_tree.txt").write_text(
        "".join(f"expected_output/{entry}\n" for entry in entries),
        encoding="utf-8",
    )
    (FIXTURE / "expected_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (FIXTURE / "environment.json").write_text(
        json.dumps(_versions(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"regenerated fixture from {len(entries)} entries")
    print("refreshed expected_summary.json, expected_tree.txt and environment.json")


if __name__ == "__main__":
    main()
