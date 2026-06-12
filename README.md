# LDSFL-Meander

[![Tests](https://github.com/sergioald/LDSFL_Meander/actions/workflows/tests.yml/badge.svg)](https://github.com/sergioald/LDSFL_Meander/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19945291.svg)](https://doi.org/10.5281/zenodo.19945291)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**LDSFL-Meander** is a Python research-software implementation of a reduced morphodynamic model for exploratory studies of meandering-river evolution.

The name refers to **Lopez-Dubon, Sgarabotto, Frascati and Lanzoni**.

<p align="center">
  <img src="docs/figures/cover.png" alt="LDSFL-Meander model overview" width="900">
</p>

<p align="center">
  <em>Reduced meander-evolution workflow with command-line and desktop GUI execution paths.</em>
</p>

---

## What this repository provides

This repository packages the LDSFL meander model as a reproducible source-code project with:

- a solver package in [`ldsfl/`](ldsfl/);
- a command-line runner in [`run_ldsfl.py`](run_ldsfl.py);
- a desktop GUI in [`gui_ldsfl.py`](gui_ldsfl.py);
- bundled example inputs in [`Input/`](Input/);
- smoke tests for solver and GUI/config workflows;
- user documentation, citation metadata and an archived DOI release.

The project is intended for **transparent research-software reuse**, teaching, method comparison and reproducible exploratory studies of meander migration under the assumptions of the reduced model.

---

## Scope and limitations

LDSFL-Meander is intended for reduced-model studies of meander evolution, especially for **wide, mildly curved, long bends**.

It is **not** a full 2D or 3D hydrodynamic solver, and it should not be presented as a sharp-bend separation model. Use the model for exploratory morphodynamic studies within the assumptions described in the documentation and user manual.

---

## Quick start

### 1. Install

From the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The package requires Python 3.10 or newer.

### 2. Run the bundled example from the command line

```bash
python run_ldsfl.py --base-dir . --cases 1 --max-steps 50 --no-plots
```

This uses the example files in [`Input/`](Input/) and writes results to `Output/`, which is created automatically.

Expected project layout:

```text
<base-dir>/
  Input/
    Parameter.csv
    xy.csv
  Output/          created automatically
  run_ldsfl.py
  ldsfl/
```

### 3. Run the first checks

```bash
python smoke_test.py
python gui_smoke_test.py
```

The smoke tests are designed to provide a quick confidence check that the solver, configuration conversion and GUI-related setup are working.

---

## GUI workflow

Launch the desktop GUI with:

```bash
python gui_ldsfl.py
```

On startup, the GUI preloads the bundled example geometry and runnable test values from [`Input/`](Input/), so a first trial run can be launched quickly.

<p align="center">
  <img src="docs/figures/gui_plot_view.png" alt="LDSFL-Meander GUI plot view" width="900">
</p>

<p align="center">
  <em>The GUI supports input preparation, parameter checking, run monitoring, diagnostics and visual review of model output.</em>
</p>

The GUI helps you:

- choose dimensionless or dimensional input mode;
- validate the centreline file;
- keep geometry as provided or scale it by `B_0` before writing `Input/xy.csv`;
- choose stop criteria, output units and solver backends;
- save and load reusable configurations;
- inspect converted parameters before running;
- monitor live snapshots and review the final planform overlay.

---

## Main workflows

| Workflow | Entry point | Purpose |
|---|---|---|
| Command-line example | `run_ldsfl.py` | Fast reproducible run from bundled inputs |
| GUI run | `gui_ldsfl.py` | Interactive input preparation, validation and visual output review |
| Solver smoke test | `smoke_test.py` | Quick solver sanity check |
| GUI/config smoke test | `gui_smoke_test.py` | Quick check for GUI configuration and conversion paths |
| Short regression run | `Reg.py` | Lightweight local regression/smoke workflow |

---

## Inputs and outputs

The bundled example expects:

```text
Input/
  Parameter.csv
  xy.csv
```

The command-line runner creates:

```text
Output/
```

Typical usage:

```bash
python run_ldsfl.py --base-dir . --cases 1 --max-steps 50
```

Use `--no-plots` when you want a non-interactive run, for example in a quick terminal check or CI-like environment.

---

## Important notation

The public documentation and GUI use the following notation:

| Symbol/name | Meaning |
|---|---|
| `B_0` | Reference channel half-width |
| `2B_0` | Full reference channel width |
| `D_0` | Reference flow depth used in dimensional input conversion |
| `D(s,n)` | Reserved for a future local depth field |
| `h(s,n)` | Reserved for a future free-surface elevation field |
| `kappa(s)` | Curvature notation used in the manual, even where some source literature uses `C(s)` |

The reduced input parameters are written as:

| Parameter | Meaning |
|---|---|
| `Beta = B_0 / D_0` | Width-to-depth ratio using the reference half-width |
| `ds = d50 / D_0` | Dimensionless sediment size |
| `Thetha` | Reference Shields stress `theta_0`; historical CSV spelling retained by the code |

---

## Repository structure

```text
LDSFL_Meander/
  ldsfl/                         solver package
  Input/                         bundled example inputs
  Output/                        generated run outputs, created locally
  docs/
    figures/                     README and manual figures
    LDSFL_Meander_user_manual.pdf
  examples/
    reproducible_case1_short/    short reproducible example metadata
  run_ldsfl.py                   command-line runner
  gui_ldsfl.py                   desktop GUI
  smoke_test.py                  solver smoke test
  gui_smoke_test.py              GUI/config smoke test
  USER_MANUAL.md                 short user guide
  BUILD_EXE.md                   optional Windows executable notes
  CITATION.cff                   citation metadata
  pyproject.toml                 package metadata and dependencies
  LICENSE
```

---

## Documentation

Useful supporting files:

- [`USER_MANUAL.md`](USER_MANUAL.md) — short user guide;
- [`docs/LDSFL_Meander_user_manual.pdf`](docs/LDSFL_Meander_user_manual.pdf) — full LaTeX manual;
- [`BUILD_EXE.md`](BUILD_EXE.md) — notes for creating a Windows GUI bundle as a release asset;
- [`examples/reproducible_case1_short/`](examples/reproducible_case1_short/) — short reproducible example metadata.

A Windows GUI executable can be built as a **release asset** rather than committed into the source tree. See [`BUILD_EXE.md`](BUILD_EXE.md).

---

## Development checks

Run the test suite with:

```bash
pytest
```

Run the lightweight smoke checks with:

```bash
python smoke_test.py
python gui_smoke_test.py
```

The project also includes Ruff and pre-commit tooling through the development dependencies.

---

## Citation

If you use this software, please cite the archived release.

Latest software DOI for all versions:

```text
10.5281/zenodo.19945291
```

Specific release DOI for `v0.6.3.1`:

```text
10.5281/zenodo.19945292
```

Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

---

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).
