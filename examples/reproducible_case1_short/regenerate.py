#!/usr/bin/env python3
"""Regenerate the bundled reproducibility fixture and record the environment.

Run from the repository root on the machine that should define the reference:

    python examples/reproducible_case1_short/regenerate.py

This rewrites ``expected_summary.json``, ``expected_tree.txt`` and
``environment.json``. The floating-point entries in ``expected_summary.json``
are only meaningful together with ``environment.json``: BLAS, NumPy and SciPy
versions all influence the last digits, and larger differences indicate an
actual behaviour change that belongs in CHANGELOG.md.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
REPO = FIXTURE.parents[1]
CASE_ID = "1_9_0005_03_2_05"
COMMAND = [
    sys.executable, "run_ldsfl.py", "--base-dir", ".", "--cases", "1",
    "--max-steps", "5", "--nprint", "2", "--max-cut", "1", "--no-plots",
]


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


def main() -> None:
    out_root = REPO / "Output"
    shutil.rmtree(out_root, ignore_errors=True)
    subprocess.run(COMMAND, cwd=REPO, check=True)

    produced = out_root / CASE_ID
    if not produced.exists():
        raise SystemExit(f"expected output folder not found: {produced}")

    target = FIXTURE / "expected_output"
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(out_root, target)

    entries = sorted(
        str(p.relative_to(target)).replace("\\", "/") + ("/" if p.is_dir() else "")
        for p in target.rglob("*")
    )
    (FIXTURE / "expected_tree.txt").write_text(
        "".join(f"expected_output/{e}\n" for e in entries), encoding="utf-8"
    )

    (FIXTURE / "environment.json").write_text(json.dumps(_versions(), indent=2) + "\n", encoding="utf-8")
    print(f"regenerated fixture from {len(entries)} entries; remember to refresh expected_summary.json")


if __name__ == "__main__":
    main()
