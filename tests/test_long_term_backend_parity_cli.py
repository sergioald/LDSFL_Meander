"""Cheap CLI status tests for the standalone long-term parity runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

RUNNER_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "long_term_backend_parity.py"


@pytest.fixture
def runner(monkeypatch):
    # The CLI status tests do not execute either accelerated kernel. Stub the
    # eager vertical import and the CLI's availability check so Numba is optional.
    vertical_stub = ModuleType("ldsfl.vertical_numba")
    vertical_stub.clear_k0123_cache = lambda: None
    vertical_stub.k0123_cache_info = lambda: None
    monkeypatch.setitem(sys.modules, "ldsfl.vertical_numba", vertical_stub)

    spec = importlib.util.spec_from_file_location("long_term_backend_parity_cli_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "numba", ModuleType("numba"))
    return module


def _run_cli(monkeypatch, tmp_path, runner, statuses, *, comparison="vertical-only"):
    monkeypatch.setattr(
        sys,
        "argv",
        ["long_term_backend_parity.py", "--steps", "5", "--comparison", comparison, "--results-dir", str(tmp_path)],
    )
    calls = []
    saved = []

    def fake_run_one(**kwargs):
        calls.append(kwargs)
        completed_steps, run_error = statuses[len(calls) - 1]
        return {
            "completed_steps": completed_steps,
            "run_error": run_error,
            "total_wall_seconds": 0.0,
        }

    def fake_save_comparison(**kwargs):
        saved.append(kwargs)
        output_dir = tmp_path / kwargs["comparison"]
        output_dir.mkdir()
        (output_dir / "report.md").write_text("partial or complete diagnostic report", encoding="utf-8")
        return output_dir

    monkeypatch.setattr(runner, "_run_one", fake_run_one)
    monkeypatch.setattr(runner, "_save_comparison", fake_save_comparison)
    return runner.main(), calls, saved


def test_complete_paired_runs_return_success(monkeypatch, tmp_path, runner):
    result, calls, saved = _run_cli(monkeypatch, tmp_path, runner, [(5, None), (5, None)])

    assert result == 0
    assert len(calls) == 2
    assert len(saved) == 1
    assert (tmp_path / "vertical-only" / "report.md").is_file()


def test_run_error_saves_report_and_returns_failure(monkeypatch, tmp_path, runner):
    error = {"type": "RuntimeError", "message": "synthetic failure"}
    result, calls, saved = _run_cli(monkeypatch, tmp_path, runner, [(5, error), (5, None)])

    assert result != 0
    assert len(calls) == 2
    assert len(saved) == 1
    assert (tmp_path / "vertical-only" / "report.md").is_file()


def test_short_run_saves_report_and_returns_failure(monkeypatch, tmp_path, runner):
    result, calls, saved = _run_cli(monkeypatch, tmp_path, runner, [(5, None), (4, None)])

    assert result != 0
    assert len(calls) == 2
    assert len(saved) == 1
    assert (tmp_path / "vertical-only" / "report.md").is_file()


def test_both_stops_after_incomplete_first_comparison(monkeypatch, tmp_path, runner):
    result, calls, saved = _run_cli(
        monkeypatch, tmp_path, runner, [(5, None), (4, None)], comparison="both",
    )

    assert result != 0
    assert len(calls) == 2
    assert [item["comparison"] for item in saved] == ["vertical-only"]
    assert (tmp_path / "vertical-only" / "report.md").is_file()
    assert not (tmp_path / "full").exists()


def test_both_returns_failure_if_second_comparison_is_incomplete(monkeypatch, tmp_path, runner):
    result, calls, saved = _run_cli(
        monkeypatch, tmp_path, runner,
        [(5, None), (5, None), (5, None), (4, None)], comparison="both",
    )

    assert result != 0
    assert len(calls) == 4
    assert [item["comparison"] for item in saved] == ["vertical-only", "full"]
    assert (tmp_path / "vertical-only" / "report.md").is_file()
    assert (tmp_path / "full" / "report.md").is_file()


def test_invalid_request_generates_no_report_and_exits_nonzero(monkeypatch, tmp_path, runner):
    monkeypatch.setattr(sys, "argv", ["long_term_backend_parity.py", "--steps", "0", "--results-dir", str(tmp_path)])

    with pytest.raises(SystemExit) as exc_info:
        runner.main()

    assert exc_info.value.code != 0
    assert not (tmp_path / "vertical-only").exists()
