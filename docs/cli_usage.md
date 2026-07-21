# CLI usage guide

This page gives practical command-line examples for running LDSFL-Meander without the GUI.

The CLI entry point is:

```bash
python -m run_ldsfl
```

or, after installing the package in editable mode:

```bash
ldsfl-run
```

## Quick start

From the repository root:

```bash
python -m run_ldsfl --base-dir . --cases 1 --max-steps 100 --no-plots
```

This runs case `1` from `Input/Parameter.csv`, reads the initial centreline from `Input/xy.csv`, writes results under `Output/`, and disables plot generation.

## Recommended smoke test

Use this before opening a pull request or after changing environments:

```bash
python -m run_ldsfl --base-dir . --cases 1 --max-steps 1 --no-plots
python -m pytest
```

## Selecting cases

The `--cases` option selects one or more case IDs from `Input/Parameter.csv`.

Examples:

```bash
python -m run_ldsfl --base-dir . --cases 1 --max-steps 100 --no-plots
python -m run_ldsfl --base-dir . --cases 1,2 --max-steps 100 --no-plots
python -m run_ldsfl --base-dir . --cases 1-2 --max-steps 100 --no-plots
```

The case IDs must exist in the `Id` column of `Input/Parameter.csv`.

## Common options

| Option | Typical use |
|---|---|
| `--base-dir .` | Use the current repository as the project root. |
| `--cases 1` | Run one selected case from `Input/Parameter.csv`. |
| `--max-steps 1000` | Limit the number of solver iterations. Useful for reproducible short runs. |
| `--nprint 100` | Save periodic outputs every `Nprint` iterations. |
| `--no-plots` | Disable PNG plot generation. Useful for CI, remote servers, and headless sessions. |
| `--backend numpy` | Use the reference NumPy backend. This is the default and safest option. |
| `--backend numba` | Use the optional Numba backend if installed and validated for the selected path. |
| `--flow-bc free` | Use the default free-boundary flow solver. |
| `--flow-bc periodic` | Use the periodic solver path. Treat as experimental unless validated for the target study. |
| `--stop-on-sinuosity-stability 1` | Stop early when the equivalence-style sinuosity stability criterion is satisfied. |
| `--return-equivalence-stability 1` | Compute and return the final equivalence/HAC stability diagnostic without using it as a stopping criterion. |

## Backend guidance

Use NumPy first:

```bash
python -m run_ldsfl --base-dir . --cases 1 --backend numpy --max-steps 100 --no-plots
```

The Numba backend is optional:

```bash
python -m run_ldsfl --base-dir . --cases 1 --backend numba --max-steps 100 --no-plots
```

NumPy should be treated as the reference backend. Numba is an acceleration path and should be compared with NumPy for the selected solver branch before relying on it for a study.

## Output folders

For a case with output identifier `<case_id>`, results are written under:

```text
Output/<case_id>/
```

The main subfolders are:

| Folder | Contents |
|---|---|
| `Output/<case_id>/xyu/` | Geometry snapshots with centreline coordinates, arclength, tangent angle, curvature, and migration velocity. |
| `Output/<case_id>/files/` | Variable-history CSV files and the sinuosity-history CSV file. |
| `Output/<case_id>/plot/` | PNG planform and sinuosity plots when plotting is enabled. |
| `Output/<case_id>/xy_cut/` | Cutoff geometry CSV files when cutoffs occur. |

## Main output files

### Geometry snapshot CSV files

Pattern:

```text
Output/<case_id>/xyu/xyu_<case_id>_*.csv
```

Columns:

| Column | Meaning |
|---|---|
| `x` | Centreline x-coordinate. |
| `y` | Centreline y-coordinate. |
| `s` | Arclength coordinate. |
| `th` | Tangent angle. |
| `c` | Curvature. |
| `U` | Migration velocity response used by the morphodynamic update. |

### Variable-history CSV files

Pattern:

```text
Output/<case_id>/files/var_<case_id>_*.csv
```

These files store model variables accumulated over saved iteration blocks.

### Sinuosity-history CSV file

Pattern:

```text
Output/<case_id>/files/sinuosity_history_<case_id>.csv
```

Columns:

| Column | Meaning |
|---|---|
| `step` | Solver iteration step. |
| `sinuo` | Dimensionless sinuosity. |

The x-axis in the sinuosity history is solver iteration step, not dimensional physical time.

## Headless and CI runs

For automated runs, remote machines, GitHub Actions, or macOS sessions without a stable GUI backend, use:

```bash
python -m run_ldsfl --base-dir . --cases 1 --max-steps 100 --no-plots
```

This avoids GUI display requirements and reduces the chance of matplotlib backend issues.

## Example reproducible run

On Windows `cmd`:

```cmd
python -m run_ldsfl ^
  --base-dir . ^
  --cases 1 ^
  --backend numpy ^
  --max-steps 1000 ^
  --nprint 100 ^
  --no-plots
```

On Linux/macOS shells:

```bash
python -m run_ldsfl \
  --base-dir . \
  --cases 1 \
  --backend numpy \
  --max-steps 1000 \
  --nprint 100 \
  --no-plots
```

## Troubleshooting

### The run is slow

Use a shorter run first:

```bash
python -m run_ldsfl --base-dir . --cases 1 --max-steps 10 --no-plots
```

Then increase `--max-steps`.

### Plotting fails on macOS or a remote machine

Use:

```bash
python -m run_ldsfl --base-dir . --cases 1 --max-steps 100 --no-plots
```

Also see [`macos_gui_notes.md`](macos_gui_notes.md).

### I need the final equivalence/HAC stability diagnostic

Use:

```bash
python -m run_ldsfl --base-dir . --cases 1 --return-equivalence-stability 1
```

This diagnostic is more expensive than the lightweight moving-window diagnostic and is not computed by default for long runs.
