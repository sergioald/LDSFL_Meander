# LDSFL-Meander

**LDSFL-Meander** is named after **Lopez-Dubon, Sgarabotto, Frascati and Lanzoni**.

It is a Python reduced morphodynamic model for meandering rivers with:

- a solver package in `ldsfl/`
- a command-line runner in `run_ldsfl.py`
- a desktop GUI in `gui_ldsfl.py`

This release focuses on a clean, reproducible source version with improved notation, geometry preprocessing, flexible stop criteria, smoke tests, and plot-based feedback.

## Scope

LDSFL-Meander is intended for reduced-model studies of meander evolution, especially for **wide, mildly curved, long bends**. It is **not** a full 2D or 3D hydrodynamic solver and it should not be presented as a sharp-bend separation model.

## Important notation

In the public documentation and GUI:

- `B_0` is the **reference channel half-width**
- the **full reference channel width** is `2B_0`
- `D_0` is the **reference flow depth** used in dimensional input conversion
- `D(s,n)` is reserved for a future **local depth field**
- `h(s,n)` is reserved for a future **free-surface elevation field**
- `kappa(s)` is used in the manual for **curvature**, even when some source literature uses `C(s)`

Therefore, the reduced input parameters are written as:

- `Beta = B_0 / D_0`
- `ds = d50 / D_0`
- `Thetha` = reference Shields stress `theta_0` (historical CSV spelling retained by the code)

## Repository contents

- `ldsfl/` - solver package
- `run_ldsfl.py` - main command-line entry point
- `gui_ldsfl.py` - desktop GUI
- `Run.py` - convenience local runner
- `Reg.py` - short regression/smoke runner
- `smoke_test.py` - short solver smoke test
- `gui_smoke_test.py` - GUI/config/conversion smoke test
- `Input/` - small example inputs
- `examples/reproducible_case1_short/` - short reproducible example metadata
- `USER_MANUAL.md` - short user guide
- `docs/LDSFL_Meander_user_manual.pdf` - full LaTeX manual
- `CITATION.cff` - citation metadata
- `LICENSE` - software license

## Installation

```bash
pip install -r requirements.txt
```

## Command-line run

From the repository root:

```bash
python run_ldsfl.py --base-dir . --cases 1 --max-steps 50 --no-plots
```

Expected layout:

```text
<base-dir>/
  Input/
    Parameter.csv
    xy.csv
  Output/          created automatically
  run_ldsfl.py
  ldsfl/
```

## GUI

Launch the desktop GUI with:

```bash
python gui_ldsfl.py
```

On startup, the GUI preloads the bundled example geometry and a runnable set of test values from `Input/`, so a first trial run can be launched immediately.

The GUI helps you:

- choose dimensionless or dimensional input mode
- validate the centerline file
- keep geometry as provided or scale it by `B_0` before writing `Input/xy.csv`
- choose stop criteria, output units, and solver backends
- save and load reusable configurations
- inspect converted parameters before the run
- monitor live snapshots and review the final planform overlay

## First check

Run:

```bash
python smoke_test.py
python gui_smoke_test.py
```

## Optional GUI executable

A Windows GUI bundle can be built as a **release asset** rather than committed into the source tree. See `BUILD_EXE.md`.

## DOI

Latest software DOI (all versions): 10.5281/zenodo.19945291
![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19945291.svg)

Specific release DOI for v0.6.3.1: 10.5281/zenodo.19945292
