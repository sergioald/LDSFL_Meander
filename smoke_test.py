#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from ldsfl.main import run_project


EXPECTED_SUBDIRS = ("files", "plot", "xy_cut", "xyu")


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="ldsfl_smoke_") as tmp:
        base_dir = Path(tmp)
        shutil.copytree(repo_root / "Input", base_dir / "Input")

        results = run_project(
            base_dir,
            cases=[1],
            Nprint=2,
            Ntstep=100000,
            Max_Cut=1,
            max_steps=5,
            do_plots=False,
            flow_backend="numpy",
        )
        result = results[0]

        out_root = base_dir / "Output" / result["id_files"]
        assert out_root.exists(), f"Missing output root: {out_root}"
        for subdir in EXPECTED_SUBDIRS:
            path = out_root / subdir
            assert path.exists(), f"Missing expected subdirectory: {path}"

        xyu_files = list((out_root / "xyu").glob("*.csv"))
        var_files = list((out_root / "files").glob("*.csv"))
        assert xyu_files, "Smoke test expected at least one xyu CSV output."
        assert var_files, "Smoke test expected at least one variables CSV output."
        assert result["steps"] == 5, f"Expected steps == 5, got {result['steps']}"

        summary = {
            "id_files": result["id_files"],
            "steps": result["steps"],
            "cut_cnt": result["cut_cnt"],
            "jt": result["jt"],
            "dt_cum": result["dt_cum"],
        }
        print(json.dumps(summary, indent=2))
        print("Smoke test passed.")


if __name__ == "__main__":
    main()
