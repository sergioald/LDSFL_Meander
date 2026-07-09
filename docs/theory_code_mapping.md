# Theory-to-code mapping

This document maps the main LDSFL-Meander modelling concepts to the repository implementation. It is intended to make the code easier to audit, test, and cite as research software.

## Purpose

The repository implements a reduced morphodynamic meander model. The code is organised around a repeatable pipeline:

1. read dimensionless model parameters and an initial centreline;
2. preprocess the centreline geometry;
3. evaluate the hydrodynamic/free-flow response;
4. advance the centreline;
5. update geometry and morphodynamic parameters;
6. save geometry, variable history, sinuosity history, and diagnostics.

The mapping below is deliberately implementation-oriented. It does not replace the paper or original derivation; it helps readers find where each model component lives in the code.

## Pipeline map

| Model step | Implementation | Notes |
|---|---|---|
| Input parameters | `ldsfl.inputs.read_parameter_table`, `ldsfl.inputs.dimensionless_input_table` | Reads `Input/Parameter.csv` and converts each row into the dimensionless parameters used by the solver. |
| Initial centreline | `ldsfl.inputs.read_xy`, `ldsfl.profile.preprof_3` | Reads `Input/xy.csv`, computes arclength, resampled coordinates, tangent angle, wavelength, valley length, and initial sinuosity. |
| Initial curvature | `ldsfl.main.initial_curvature` | Uses the same tangent-angle convention as the geometry update, avoiding a first-step sign flip. |
| Resistance terms | `ldsfl.resistance.resistance_function_flagbed` | Computes resistance-related coefficients for plane-bed and dune-bed cases. |
| Flow-field response | `ldsfl.flowfield.parall_u_free` | Main free-boundary flow response used by default runs. Supports NumPy and optional Numba backends. |
| Periodic flow response | `ldsfl.flowfield_periodic.parall_u_periodic` | Alternative periodic path. Treat as experimental unless specifically validated for the target case. |
| Centreline migration | `ldsfl.evolution.dxdy2` | Converts flow response into centreline displacement increments with a timestep stability coefficient. |
| Geometry update | `ldsfl.geometry.geometry4` | Resamples, smooths, detects cutoffs, updates curvature and sinuosity, and writes cutoff geometries. |
| Parameter update | `ldsfl.evolution.update_parameters` | Updates evolving parameters after geometry changes. |
| Output writing | `ldsfl.outputs` | Writes geometry snapshots, variable-history CSVs, sinuosity history, and figures. |
| Stability diagnostics | `ldsfl.main._sinuosity_stability_metrics`, `ldsfl.stability.sinuosity_equivalence_stability` | Moving-window diagnostic is lightweight. Equivalence/HAC diagnostic is explicit because it can be expensive for long histories. |
| CLI entry point | `run_ldsfl.py` | Provides a command-line interface around `ldsfl.main.run_project`. |
| GUI entry point | `gui_ldsfl.py` | Interactive interface for running and inspecting simulations. |

## Main runtime call graph

The high-level runtime flow is:

```text
run_ldsfl.py
  -> ldsfl.main.run_project()
      -> ldsfl.main.run_case()
          -> read inputs
          -> preprof_3()
          -> initial_curvature()
          -> resistance_function_flagbed()
          -> loop:
              -> parall_u_free() or parall_u_periodic()
              -> save_xystcu() / save_variables() at output intervals
              -> dxdy2()
              -> geometry4()
              -> resistance_function_flagbed()
              -> update_parameters()
              -> sinuosity diagnostics
          -> final snapshot and histories
```

## Boundary-condition status

The default and currently best-supported path is the free-boundary flow solver:

```text
flow_bc="free"
```

The periodic path exists for experimentation and comparison:

```text
flow_bc="periodic"
```

The periodic path should not be presented as equally validated until it has dedicated regression tests and physical comparison cases. For publication-quality runs, use the free-boundary path unless there is a specific reason and validation record for the periodic path.

## Backend status

The default backend is NumPy:

```text
backend="numpy"
```

The optional Numba backend is intended as an acceleration path. It is an optional extra and should be tested against NumPy for the exact solver branch being used.

Current testing distinguishes between:

- default NumPy execution;
- NumPy serial versus threaded mode accumulation;
- optional Numba parity where validated;
- known or suspected backend discrepancies marked as expected failures until investigated.

## Stability diagnostics

Two sinuosity diagnostics are available.

### Lightweight moving-window diagnostic

Implemented in:

```text
ldsfl.main._sinuosity_stability_metrics
```

This diagnostic is cheap and suitable for default return values and GUI display.

### Equivalence/HAC diagnostic

Implemented in:

```text
ldsfl.stability.sinuosity_equivalence_stability
```

This diagnostic estimates total fitted drift over a post-transient analysis window and checks whether the confidence interval lies within a practical drift tolerance. It is more statistically meaningful, but it is more expensive because it uses a HAC/Newey-West-style covariance calculation.

For that reason, it should be computed only when explicitly needed, for example when:

```text
stop_on_sinuosity_stability=True
```

or when the CLI/user explicitly requests it:

```text
--return-equivalence-stability 1
```

## Output map

| Output type | Location | Producer |
|---|---|---|
| Geometry snapshots | `Output/<case_id>/files/xyu_*.csv` | `ldsfl.outputs.save_xystcu` |
| Variable history | `Output/<case_id>/files/var_*.csv` | `ldsfl.outputs.save_variables` |
| Sinuosity history | `Output/<case_id>/files/sinuosity_history_*.csv` | `ldsfl.outputs.save_sinuosity_history` |
| Planform figures | `Output/<case_id>/figures/` | `ldsfl.outputs.plot_it` |
| Cutoff files | `Output/<case_id>/files/` | `ldsfl.geometry.save_xy_cut` |
| Cutoff figures | `Output/<case_id>/figures/` | `ldsfl.geometry.plot_cut` |

## Testing map

| Test topic | Typical files |
|---|---|
| CLI parsing and validation | `tests/test_cli_config.py`, CLI-focused tests |
| Output bookkeeping | `tests/test_variable_history_complete.py` |
| Initial curvature sign | `tests/test_curvature_sign_consistency.py` |
| Plot disabling / no-plots behaviour | `tests/test_cutoff_no_plots.py` |
| Stop criteria and stability | `tests/test_sinuosity_stability_stop.py`, `tests/test_sinuosity_equivalence_performance_guard.py` |
| Flow-field physics/regression | `tests/test_flowfield_physics_regression.py`, `tests/test_flowfield_backend_guardrails.py` |

## Maintenance notes

When changing the solver, update this document if the change affects:

- model inputs;
- boundary-condition behaviour;
- backend behaviour;
- geometry conventions;
- timestep/stability logic;
- output file semantics;
- validation evidence.

Small refactors that do not change behaviour should still preserve the mapping between the model concepts and the public functions listed here.
