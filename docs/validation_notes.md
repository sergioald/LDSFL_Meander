# Validation notes

This document summarises what is currently validated in the `LDSFL-Meander` repository, what is intentionally outside the validation scope, and what should be checked before using the workflows for new research outputs.

## Validation scope

This repository is public research software derived from reduced meander-morphodynamics workflows. The validation focus is on:

- reproducible execution of the main command-line workflow;
- correct handling of the bundled input files and selected configuration options;
- lightweight checks of input reading, parameter parsing and run-case selection;
- smoke testing of package imports and executable scripts;
- basic checks of timestep/evolution helpers and stop criteria;
- GUI-side checks of sinuosity metric calculation without requiring a full manual GUI session;
- a minimal solver integration path that verifies the repository remains runnable after editable installation;
- separation between reusable source code, generated outputs, documentation and local run products.

## What is tested

The test suite and development checks are intended to verify that:

- the Python package can be imported;
- the command-line runner can parse valid case selections;
- invalid command-line case selections fail early and clearly;
- bundled input readers behave as expected on small example inputs;
- stop criteria and sinuosity-stability helpers return consistent state information;
- selected evolution and timestep utilities run on lightweight inputs;
- GUI metric calculation works without starting a full Tkinter application;
- a minimal solver path can execute in a development environment;
- source files remain syntactically valid after cleanup changes.

## What is not fully tested

The repository does not provide full scientific re-validation of every model variant or every historical result. In particular, the following are outside the lightweight public test scope:

- exhaustive reproduction of all original long-run simulations;
- full validation across all possible bed, resistance, boundary-condition or cutoff settings;
- verification of every numerical regime outside the reduced-model assumptions;
- comparison against all full 2D/3D hydrodynamic or morphodynamic solvers;
- calibration against new field or laboratory datasets not included in the repository;
- proof that every bend reaches a mathematical equilibrium when the sinuosity diagnostic is stable;
- operational prediction for site-specific engineering decisions.

## Model and data boundaries

LDSFL-Meander is a reduced centreline-evolution model. It is intended for fast, transparent exploration of reduced meander dynamics, not as a full hydrodynamic or morphodynamic simulator.

The repository separates:

- public reusable code;
- small bundled example inputs;
- documentation and reproducibility notes;
- generated run outputs under `Output/`;
- longer exploratory simulations that should usually remain local.

Large generated outputs, exploratory sweeps and publication-specific post-processing files should not normally be committed to Git unless they are intentionally curated, small and documented.

## Expected behaviour

For correctly formatted bundled or user-provided input files, the workflows should:

- read model parameters and centreline coordinates;
- initialise the selected run case;
- evolve the input centreline according to the reduced model settings;
- write reproducible output folders under `Output/<id_files>/`;
- store centreline, angle, curvature, velocity and diagnostic histories where configured;
- produce planform and sinuosity plots when plotting is enabled;
- compute practical sinuosity-stability diagnostics for long exploratory runs;
- expose the same core workflow through command-line and GUI interfaces.

## Known limitations

- Results depend on input centreline quality, smoothing choices, timestep settings and model parameters.
- The model is intended for wide, mildly curved, long-bend reduced-model studies.
- The stability diagnostic measures practical convergence of bulk sinuosity; it does not prove full geomorphic equilibrium.
- Cutoff handling, boundary conditions and resistance/bed settings should be inspected carefully for each new study.
- The GUI is intended for local interactive use and teaching; scripted CLI runs are preferred for reproducible batch studies.
- The repository is not a full 2D/3D hydrodynamic solver and should not be presented as a replacement for RANS, LES, Delft3D, TELEMAC or full morphodynamic simulations.

## Recommended checks before new use

Before using the repository for a new dataset or research question, users should:

- verify coordinate units and centreline ordering;
- inspect the input planform visually;
- check spacing, smoothing and curvature behaviour;
- confirm that parameter values are physically meaningful for the intended reduced-model regime;
- run a short simulation first before launching a long run;
- inspect intermediate planform snapshots;
- inspect sinuosity histories and cutoff behaviour;
- compare alternative timesteps or selected sensitivity settings when results are used for interpretation;
- document all non-default parameters used in research outputs;
- keep generated outputs separate from source code unless they are deliberately curated examples.

## Development checks

Recommended local checks are:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check ldsfl run_ldsfl.py tests
python -m run_ldsfl --base-dir . --cases 1 --max-steps 1 --no-plots
```

For GUI checks, launch:

```bash
python gui_ldsfl.py
```

The GUI should preload the bundled example inputs and allow a short run to be launched without manually editing repository paths.

## Interpreting stability diagnostics

The sinuosity-stability tools are designed to support exploratory interpretation. They should be read as a practical diagnostic:

- stable or quasi-stable sinuosity means the bulk sinuosity signal has changed little over the selected diagnostic window;
- it does not necessarily mean that every local bend, cutoff process or curvature feature has stopped evolving;
- the diagnostic should be reviewed alongside planform plots, curvature snapshots and output histories.

For publication-quality use, report the window length, tolerance, run length and any stop-on-stability settings used.
