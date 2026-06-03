from __future__ import annotations

from gui_smoke_test import main as run_gui_smoke_test
from smoke_test import main as run_solver_smoke_test


def test_solver_smoke_workflow() -> None:
    """Run a short solver smoke test using bundled example inputs."""
    run_solver_smoke_test()


def test_gui_config_smoke_workflow() -> None:
    """Run GUI/config conversion checks and a short solver run."""
    run_gui_smoke_test()
