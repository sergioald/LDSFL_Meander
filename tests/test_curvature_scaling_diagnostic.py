from __future__ import annotations

import pytest

from tools.diagnose_curvature_scaling import diagnose_case, main, run_diagnostic


def test_arclength_gradient_recovers_constant_curvature_arc():
    result = diagnose_case("unit-test arc", radius=20.0, target_spacing=0.5)

    assert result.arclength_gradient_over_true == pytest.approx(1.0, rel=2.0e-2)
    assert result.arclength_gradient_max_abs_error < 2.0e-3


def test_index_gradient_curvature_scales_with_point_spacing():
    fine = diagnose_case("fine", radius=20.0, target_spacing=0.25)
    coarse = diagnose_case("coarse", radius=20.0, target_spacing=2.0)

    assert fine.index_gradient_over_true == pytest.approx(fine.mean_spacing, rel=3.0e-2)
    assert coarse.index_gradient_over_true == pytest.approx(coarse.mean_spacing, rel=3.0e-2)
    assert coarse.index_gradient_over_true > 6.0 * fine.index_gradient_over_true


def test_diagnostic_runs_representative_spacing_cases():
    rows = run_diagnostic()

    assert [row.name for row in rows] == [
        "fine-spacing arc",
        "near-unit-spacing arc",
        "coarse-spacing arc",
    ]
    assert rows[0].mean_spacing < rows[1].mean_spacing < rows[2].mean_spacing


def test_diagnostic_cli_prints_interpretation(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out

    assert "Curvature arclength-scaling diagnostic" in out
    assert "index-gradient curvature" in out
    assert "arclength-gradient curvature" in out
