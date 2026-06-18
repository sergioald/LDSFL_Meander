# LDSFL-Meander portfolio summary

LDSFL-Meander is a scientific Python project for reduced meander-morphodynamics simulations. It demonstrates how a research model can be packaged into a reproducible command-line workflow and an interactive desktop GUI.

## Project purpose

The project converts a reduced river-meander model into a usable Python software package. The goal is not to replace full hydrodynamic or morphodynamic solvers, but to provide a fast and transparent environment for studying centerline evolution, sensitivity to input parameters, and qualitative planform behaviour.

The software is most appropriate for wide, mildly curved, long-bend cases where reduced modelling assumptions are acceptable.

## What the repository demonstrates

| Capability | Evidence in repository |
|---|---|
| Scientific Python packaging | `pyproject.toml`, importable `ldsfl` package, editable install workflow |
| Reproducible modelling workflow | Bundled `Input/` example, CLI runner, output folders, example metadata |
| GUI development | Tkinter + Matplotlib interface with input validation and live plotting |
| Numerical model maintenance | Geometry resampling, smoothing, cutoff handling, timestep safeguards |
| Diagnostics | Step-vs-sinuosity history, stability/quasi-stability metrics |
| Testing | Pytest unit tests and a minimal integration test |
| Documentation | README, user manual, citation metadata, portfolio summary |

## Main workflow

1. Prepare input geometry in `Input/xy.csv`.
2. Prepare model parameters in `Input/Parameter.csv` or through the GUI.
3. Run the solver from the command line or GUI.
4. Inspect generated planform snapshots, variable histories, cutoff records, and sinuosity diagnostics.
5. Use the saved output folders for reproducibility and comparison between cases.

## Interfaces

### Command line

The command-line workflow is best for reproducibility, batch runs, and testing:

```bash
python run_ldsfl.py --base-dir . --cases 1 --max-steps 50 --nprint 10 --no-plots
```

### GUI

The GUI is best for interactive use, teaching, and visual inspection:

```bash
python gui_ldsfl.py
```

The GUI supports dimensionless and dimensional input modes, geometry scaling, solver stop criteria, backend options, output-unit selection, and visual inspection of final and live planforms.

## Model scope and limitations

LDSFL-Meander should be described as a reduced morphodynamic model. It is not a full 2D or 3D hydrodynamic solver. It should not be used to claim sharp-bend separation physics, detailed turbulence closure, or full sediment-transport fidelity.

A good description is:

> A reduced scientific Python model for reproducible exploration of meandering-river centerline evolution.

A description to avoid is:

> A full hydrodynamic or morphodynamic solver for arbitrary river geometries.

## Reproducibility notes

The repository includes:

- bundled example inputs,
- command-line options for scripted runs,
- deterministic short-run tests,
- output folders separated by generated case identifiers,
- citation metadata through `CITATION.cff`,
- Zenodo DOI metadata for archival releases.

For portfolio review, the recommended reproduction command is:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python run_ldsfl.py --base-dir . --cases 1 --max-steps 50 --nprint 10 --no-plots
```

## Testing strategy

The test suite is intentionally staged:

1. Fast unit tests for parsing, inputs, stop criteria, timestep helpers, and sinuosity metrics.
2. GUI metric tests that avoid launching the full Tkinter app.
3. Minimal integration test that runs a short solver case in a temporary workspace.
4. Future regression tests can compare selected outputs with loose numerical tolerances.

This keeps continuous integration fast while still protecting the main scientific-software workflow.

## Suggested future improvements

- Add GitHub Actions coverage reporting.
- Add a numerical regression test with tolerances for a known short run.
- Add a small gallery of benchmark cases.
- Add API examples for using `ldsfl` directly from Python.
- Keep the README short and move deep modelling notes to documentation files.

## Release workflow

For GitHub and Zenodo releases:

1. Update source files and documentation.
2. Run `python -m pytest`.
3. Commit and push changes.
4. Create a GitHub release.
5. Let Zenodo archive the release and mint a version DOI.
6. Use the Zenodo concept DOI for the evolving project and the version DOI for exact reproducibility.
