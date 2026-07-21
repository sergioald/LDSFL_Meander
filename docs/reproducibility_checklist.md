# Reproducibility checklist

Use this checklist when running LDSFL-Meander for a report, paper, review, or release archive. It records the information needed for another user to understand and rerun the calculation.

## Minimum record for every run

Record these items for every scientific run:

| Item | What to record |
|---|---|
| Repository | Repository URL and commit SHA. |
| Version/tag | Package version or release tag, if one was used. |
| Python environment | Python version and platform. |
| Installation mode | Editable install, released package, or local script run. |
| CLI command or GUI settings | Exact command line, or a screenshot/export of GUI settings. |
| Input files | Copies or checksums of `Input/Parameter.csv` and `Input/xy.csv`. |
| Case IDs | The selected case IDs from `Input/Parameter.csv`. |
| Backend | `numpy` or `numba`. NumPy is the reference path. |
| Flow boundary condition | `free` or `periodic`. Treat periodic as experimental unless specifically validated. |
| Output units | `dimensionless` or `dimensional`, including length/velocity scales if dimensional outputs are used. |
| Stop criteria | Step/time/cutoff/stability stop settings. |
| Sinuosity diagnostics | Whether equivalence/HAC diagnostics were requested or used for stopping. |
| Output archive | The generated `Output/<case_id>/` folder or a checksum of the archive. |

## Recommended command log

For CLI runs, save the exact command in a text file next to the output archive. Example:

```bash
python -m run_ldsfl \
  --base-dir . \
  --cases 1 \
  --backend numpy \
  --flow-bc free \
  --max-steps 1000 \
  --nprint 100 \
  --no-plots
```

On Windows `cmd`, the same command can be recorded as:

```cmd
python -m run_ldsfl ^
  --base-dir . ^
  --cases 1 ^
  --backend numpy ^
  --flow-bc free ^
  --max-steps 1000 ^
  --nprint 100 ^
  --no-plots
```

## Environment record

Record the active environment before running a long study:

```bash
python --version
python -m pip freeze > environment_pip_freeze.txt
python -m pytest > pytest_before_run.txt
```

For conda environments, also record:

```bash
conda env export > environment.yml
```

## Git record

Record the exact source state:

```bash
git status
git log --oneline -5
git rev-parse HEAD
```

The working tree should usually be clean before a formal run:

```bash
git status
```

Expected state:

```text
nothing to commit, working tree clean
```

If the run intentionally used uncommitted changes, record the diff:

```bash
git diff > uncommitted_changes.patch
```

## Input record

Keep a copy of the input files used for the run:

```text
Input/Parameter.csv
Input/xy.csv
```

For a formal archive, copy them into the output bundle or record checksums.

Example checksum command:

```bash
python -c "from pathlib import Path; import hashlib; [print(p, hashlib.sha256(p.read_bytes()).hexdigest()) for p in [Path('Input/Parameter.csv'), Path('Input/xy.csv')]]"
```

## Output record

Archive the relevant case folder:

```text
Output/<case_id>/
```

The important generated subfolders are:

| Folder | Meaning |
|---|---|
| `xyu/` | Centreline/curvature/migration snapshots. |
| `files/` | Variable-history and sinuosity-history CSV files. |
| `plot/` | PNG plots when plotting is enabled. |
| `xy_cut/` | Cutoff geometry files when cutoffs occur. |

Always include:

```text
Output/<case_id>/files/sinuosity_history_<case_id>.csv
Output/<case_id>/xyu/xyu_<case_id>_*.csv
Output/<case_id>/files/var_<case_id>_*.csv
```

## Backend reproducibility

For reference results, use:

```bash
--backend numpy
```

If using the optional Numba backend, record the Numba settings:

```text
--backend numba
--numba-parallel 0 or 1
--numba-fastmath 0 or 1
```

For any published or reported result produced with Numba, run a small comparison against the NumPy backend for the same input and solver path.

## Boundary-condition reproducibility

Record the selected flow boundary condition:

```text
--flow-bc free
```

or:

```text
--flow-bc periodic
```

The free-boundary NumPy path is the default and best-supported path. Periodic-boundary runs should be described as experimental unless a dedicated validation check is included for the study.

## Sinuosity-stability reproducibility

Record whether the lightweight moving-window diagnostic, the equivalence/HAC diagnostic, or both were used.

Important flags include:

```text
--stop-on-sinuosity-stability
--return-equivalence-stability
--sinuo-window
--sinuo-rel-tol
--sinuo-equiv-transient-step
--sinuo-equiv-drift-tol
--sinuo-equiv-confidence
--sinuo-equiv-min-points
--sinuo-equiv-hac-lags
--sinuo-stability-interval
```

If equivalence/HAC diagnostics are reported, record the full set of tolerance and confidence settings.

## Dimensional-output reproducibility

If using dimensional outputs, record:

```text
--output-units dimensional
--output-length-scale <value>
--output-velocity-scale <value>
```

Also record the physical interpretation of the length and velocity scales. The solver `step` index is an iteration counter and should not be interpreted as dimensional physical time unless a separate scaling is defined.

## Suggested output bundle structure

A reproducible run archive can use this structure:

```text
run_archive/
  README_run.md
  command.txt
  git_status.txt
  git_log.txt
  commit_sha.txt
  environment_pip_freeze.txt
  environment.yml
  pytest_before_run.txt
  Input/
    Parameter.csv
    xy.csv
  Output/
    <case_id>/
      files/
      xyu/
      plot/
      xy_cut/
```

## Report wording template

Use wording like:

```text
Simulations were run with LDSFL-Meander at commit <SHA> using Python <version>. The reference NumPy backend was used with free-boundary flow conditions. The model was run with the command recorded in command.txt. Input files and generated Output/<case_id>/ files are archived with this report. Sinuosity histories are reported against solver iteration step, not dimensional physical time.
```

Adjust the wording if using Numba, periodic boundaries, dimensional outputs, or equivalence/HAC stability stopping.

## Before sharing results

Before sharing a run externally, check:

```bash
python -m pytest
```

For CLI/output runs, also check:

```bash
python -m pytest tests/test_cli_output_smoke.py tests/test_example_input_schema.py
```

## Citation reminder

Use the repository `CITATION.cff` and the release/Zenodo record, if applicable, when citing LDSFL-Meander. If reporting results from a specific unreleased commit, include the commit SHA in the methods or supplementary material.
