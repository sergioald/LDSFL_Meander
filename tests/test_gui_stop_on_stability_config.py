from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GUI_PATH = REPO_ROOT / "gui_ldsfl.py"


def _attribute_path(node: ast.AST) -> str:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _run_project_calls(tree: ast.AST) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "run_project":
            calls.append(node)
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "run_project":
            calls.append(node)
    return calls


def test_gui_worker_forwards_sinuosity_stability_stop_to_solver():
    """The GUI must forward the stability stop checkbox to run_project.

    This is a source-level GUI test so CI does not need to instantiate Tkinter.
    The validator already accepts a stability-only stop configuration; this
    guard ensures the actual GUI run path passes the same option to the solver.
    """

    tree = ast.parse(GUI_PATH.read_text(encoding="utf-8"))
    calls = _run_project_calls(tree)
    assert calls, "Expected gui_ldsfl.py to call run_project(...)"

    forwarded_values: list[ast.AST] = []
    for call in calls:
        for keyword in call.keywords:
            if keyword.arg == "stop_on_sinuosity_stability":
                forwarded_values.append(keyword.value)

    assert forwarded_values, (
        "GUI run_project(...) call must pass stop_on_sinuosity_stability so "
        "the Run & diagnostics checkbox affects the actual solver run."
    )

    forwarded_paths = {_attribute_path(value) for value in forwarded_values}
    assert "config.run.stop_on_sinuosity_stability" in forwarded_paths
