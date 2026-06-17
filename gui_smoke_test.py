#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

from ldsfl.gui_utils import (
    DimensionlessInputs,
    DimensionalInputs,
    GuiCaseConfig,
    GeometrySettings,
    RunControls,
    config_from_dict,
    config_to_dict,
    preview_case_config,
    validate_case_config,
    write_case_inputs,
)
from ldsfl.main import run_project


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='ldsfl_gui_smoke_') as tmp:
        work = Path(tmp)
        xy = repo_root / 'Input' / 'xy.csv'

        cfg_dimless = GuiCaseConfig(
            mode='dimensionless',
            xy_csv=xy,
            workspace_dir=work,
            run=RunControls(case_id=11, max_steps=3, max_sim_time=1.0e9, stop_on_steps=True, stop_on_time=True, stop_on_cutoffs=True, stop_mode='first', nprint=2, max_cut=1, do_plots=False, cstab=0.01, sinuo_window=100, sinuo_rel_tol=0.005),
            dimensionless=DimensionlessInputs(beta=9.0, ds=0.005, theta0=0.3, flagbed=2, rpic_0=0.5, Mdat=6),
            geometry=GeometrySettings(mode='as_is', custom_scale=None, smoothing_enabled=True, smoothing_factor=8.0, neck_cutoff_interval=3),
        )
        preview = preview_case_config(cfg_dimless)
        assert abs(preview['beta'] - 9.0) < 1e-12
        assert validate_case_config(cfg_dimless) == []
        restored = config_from_dict(config_to_dict(cfg_dimless))
        assert restored.run.case_id == cfg_dimless.run.case_id
        assert restored.run.stop_mode == 'first'
        assert abs(restored.run.cstab - 0.01) < 1e-12
        assert restored.run.sinuo_window == 100
        summary = write_case_inputs(cfg_dimless)
        assert Path(summary['parameter_csv']).exists()
        assert Path(summary['copied_xy_csv']).exists()
        assert abs(summary['written_geometry_scale'] - 1.0) < 1e-12

        results = run_project(
            work,
            cases=[11],
            Nprint=2,
            Ntstep=100000,
            Max_Cut=1,
            max_steps=3,
            do_plots=False,
        )
        assert results[0]['steps'] == 3
        assert 'sinuosity_stability' in results[0]

        cfg_direct = GuiCaseConfig(
            mode='dimensional',
            xy_csv=xy,
            workspace_dir=work / 'dim_direct',
            run=RunControls(case_id=21),
            dimensional=DimensionalInputs(
                half_width=90.0,
                dref=10.0,
                d50=0.05,
                mobility_mode='direct_shields',
                theta0=0.3,
                flagbed=2,
                rpic_0=0.5,
                Mdat=6,
            ),
        )
        preview_direct = preview_case_config(cfg_direct)
        assert preview_direct['geometry_scale'] == 1.0
        assert abs(preview_direct['beta0'] - 9.0) < 1e-12
        assert abs(preview_direct['ds0'] - 0.005) < 1e-12
        assert abs(preview_direct['theta0'] - 0.3) < 1e-12

        cfg_manning = GuiCaseConfig(
            mode='dimensional',
            xy_csv=xy,
            workspace_dir=work / 'dim_manning',
            run=RunControls(case_id=22),
            dimensional=DimensionalInputs(
                half_width=90.0,
                dref=10.0,
                d50=0.05,
                mobility_mode='depth_velocity_friction_grain',
                velocity=1.2,
                friction_mode='manning_n',
                friction_value=0.03,
                flagbed=2,
                rpic_0=0.5,
                Mdat=6,
            ),
        )
        preview_manning = preview_case_config(cfg_manning)
        assert preview_manning['Cf'] > 0.0
        assert preview_manning['theta0'] > 0.0


        cfg_tau = GuiCaseConfig(
            mode='dimensional',
            xy_csv=xy,
            workspace_dir=work / 'dim_tau',
            run=RunControls(case_id=24),
            dimensional=DimensionalInputs(
                half_width=90.0,
                dref=10.0,
                d50=0.05,
                mobility_mode='direct_shear_stress',
                tau_b=20.0,
                flagbed=2,
                rpic_0=0.5,
                Mdat=6,
            ),
        )
        preview_tau = preview_case_config(cfg_tau)
        assert preview_tau['theta0'] > 0.0

        cfg_discharge = GuiCaseConfig(
            mode='dimensional',
            xy_csv=xy,
            workspace_dir=work / 'dim_discharge',
            run=RunControls(case_id=23),
            dimensional=DimensionalInputs(
                half_width=90.0,
                dref=10.0,
                d50=0.05,
                mobility_mode='discharge_half_width_slope_friction_grain',
                discharge=1080.0,
                slope=0.0002,
                friction_mode='chezy_c',
                friction_value=35.0,
                flagbed=2,
                rpic_0=0.5,
                Mdat=6,
            ),
        )
        preview_discharge = preview_case_config(cfg_discharge)
        assert preview_discharge['D0'] > 0.0
        assert preview_discharge['velocity'] > 0.0
        cfg_geom_width = GuiCaseConfig(
            mode='dimensional',
            xy_csv=xy,
            workspace_dir=work / 'dim_geom_width',
            run=RunControls(case_id=25),
            dimensional=DimensionalInputs(
                half_width=90.0,
                dref=10.0,
                d50=0.05,
                mobility_mode='direct_shields',
                theta0=0.3,
                flagbed=2,
                rpic_0=0.5,
                Mdat=6,
            ),
            geometry=GeometrySettings(mode='scale_by_dimensional_half_width'),
        )
        summary_geom = write_case_inputs(cfg_geom_width)
        assert abs(summary_geom['written_geometry_scale'] - 90.0) < 1e-12


        cfg_dimensional_output = GuiCaseConfig(
            mode='dimensional',
            xy_csv=xy,
            workspace_dir=work / 'dim_output',
            run=RunControls(case_id=26, output_units='dimensional'),
            dimensional=DimensionalInputs(
                half_width=90.0,
                dref=10.0,
                d50=0.05,
                mobility_mode='direct_shields',
                theta0=0.3,
                flagbed=2,
                rpic_0=0.5,
                Mdat=6,
            ),
            geometry=GeometrySettings(mode='scale_by_dimensional_half_width'),
        )
        preview_out = preview_case_config(cfg_dimensional_output)
        assert preview_out['resolved_output_units'] == 'dimensional'
        assert abs(preview_out['output_length_scale'] - 90.0) < 1e-12

        summary2 = write_case_inputs(cfg_discharge)
        assert Path(summary2['parameter_csv']).exists()

        header_xy = work / 'header_xy.csv'
        header_xy.write_text('x,y\n0,0\n10,1\n20,2\n', encoding='utf-8')
        cfg_header = GuiCaseConfig(
            mode='dimensionless',
            xy_csv=header_xy,
            workspace_dir=work / 'header_case',
            run=RunControls(case_id=31, max_steps=1, nprint=1, max_cut=1, do_plots=False),
            dimensionless=DimensionlessInputs(beta=9.0, ds=0.005, theta0=0.3, flagbed=2, rpic_0=0.5, Mdat=6),
        )
        header_warnings = validate_case_config(cfg_header)
        assert any('header row detected' in w.lower() for w in header_warnings)
        summary_header = write_case_inputs(cfg_header)
        assert Path(summary_header['copied_xy_csv']).exists()

        bad_xy = work / 'bad_xy.csv'
        bad_xy.write_text('0,0\n1,foo\n2,2\n', encoding='utf-8')
        cfg_bad = GuiCaseConfig(
            mode='dimensionless',
            xy_csv=bad_xy,
            workspace_dir=work / 'bad_case',
            run=RunControls(case_id=32, max_steps=1, nprint=1, max_cut=1, do_plots=False),
            dimensionless=DimensionlessInputs(beta=9.0, ds=0.005, theta0=0.3, flagbed=2, rpic_0=0.5, Mdat=6),
        )
        try:
            validate_case_config(cfg_bad)
            raise AssertionError('Expected malformed geometry validation to fail')
        except ValueError as exc:
            assert 'nonnumeric' in str(exc).lower() or 'malformed' in str(exc).lower()


    print('GUI smoke test passed.')


if __name__ == '__main__':
    main()
