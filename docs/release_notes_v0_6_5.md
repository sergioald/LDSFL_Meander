# LDSFL-Meander v0.6.5 release-notes draft

This page records the current release-readiness state of LDSFL-Meander after the recent validation, testing, and documentation work.

It is a draft release note for maintainers and users. It should be reviewed before creating a formal GitHub release or Zenodo/archive update.

## Summary

The current validation state focuses on making the repository safer for users to run from the command line, clearer to inspect, and easier to maintain.

The default recommended path is:

```bash
python -m run_ldsfl --base-dir . --cases 1 --backend numpy --no-plots
```

The NumPy free-boundary path remains the reference route for routine use. Optional and experimental paths are documented separately and should be validated for a target study before being used for scientific conclusions.

## Highlights since the recent maintenance campaign

### Solver and output safety

- Fixed initial curvature sign consistency against the profile preprocessing convention.
- Improved variable-history bookkeeping so completed steps are recorded once and final partial blocks are flushed.
- Ensured cutoff-related output respects plot-disabling options.
- Kept the CLI default backend on NumPy, the reference path.
- Added guardrails for expensive full-history equivalence/HAC sinuosity diagnostics.
- Kept return-only equivalence/HAC diagnostics final-only unless they are actively used as a stopping criterion.

### Sinuosity stability diagnostics

- Added configurable sinuosity-stability stopping controls.
- Preserved the lightweight moving-window diagnostic for routine runs.
- Made the more expensive equivalence/HAC diagnostic opt-in or tied to the explicit stopping path.
- Added tests around default behaviour, return-only behaviour, and CLI forwarding.

### Flow-field regression coverage

- Added physics/regression tests for the free-flow solver path.
- Checked that zero curvature gives zero migration velocity.
- Checked linear response to curvature amplitude.
- Checked NumPy serial/threaded accumulation agreement.
- Added backend guardrails for invalid backend names.
- Documented the optional Numba SL=1 mismatch as expected failure until investigated.

### CLI and generated-output checks

- Added a CLI smoke test that runs a tiny case through the real CLI argument path.
- Checked generated `Output/<case>/` folders.
- Checked geometry snapshot, variable-history, and sinuosity-history CSV files.
- Checked key CSV headers.

### Example input checks

- Added schema and sanity tests for bundled `Input/Parameter.csv` and `Input/xy.csv`.
- Checked required solver columns.
- Checked positive/unique case identifiers.
- Checked finite numeric fields.
- Checked allowed `flagbed` values.
- Checked that the project reader functions accept the bundled example inputs.

### Documentation added

- Theory-to-code mapping.
- Validation strategy and known limitations.
- CLI usage guide.
- macOS GUI and plotting notes.
- Timestep and iteration notes.
- Project health checklist.

## Current recommended validation command

Before a release, run:

```bash
git status
python -m py_compile run_ldsfl.py ldsfl/main.py gui_ldsfl.py
python -m pytest
```

For output and input smoke checks, run:

```bash
python -m pytest tests/test_cli_output_smoke.py tests/test_example_input_schema.py tests/test_variable_history_complete.py
```

For flow-field and backend checks, run:

```bash
python -m pytest tests/test_flowfield_physics_regression.py tests/test_flowfield_backend_guardrails.py
```

For sinuosity stability checks, run:

```bash
python -m pytest tests/test_sinuosity*.py tests/test_cli_sinuosity_stability.py tests/test_cli_return_equivalence_stability.py
```

## Supported/recommended paths

| Area | Current recommendation |
|---|---|
| CLI backend | Use `--backend numpy` as the reference/default path. |
| Boundary condition | Use `--flow-bc free` as the best-supported path. |
| Plotting in CI/headless sessions | Use `--no-plots`. |
| Stability diagnostics | Use default lightweight diagnostics unless equivalence/HAC is explicitly needed. |
| Example inputs | Use bundled `Input/Parameter.csv` and `Input/xy.csv` for smoke tests and first runs. |
| Generated outputs | Expect results under `Output/<case>/`. |

## Optional or experimental paths

| Area | Status |
|---|---|
| Numba backend | Optional acceleration path. Compare against NumPy for the selected case before relying on it. |
| Numba SL=1 path | Known mismatch is documented as expected failure until investigated. |
| Periodic flow boundary condition | Treat as experimental until targeted validation is added. |
| Dimensional output scaling | Available, but users must document chosen length and velocity scales. |
| Equivalence/HAC stability diagnostic | Available but more expensive; opt-in for final diagnostics or stopping studies. |

## User-facing notes

### CLI first-run command

```bash
python -m run_ldsfl --base-dir . --cases 1 --max-steps 1 --no-plots
```

### Where outputs are written

```text
Output/<case>/xyu/
Output/<case>/files/
Output/<case>/plot/
Output/<case>/xy_cut/
```

When `--no-plots` is used, CSV outputs are still expected, but PNG plot outputs may be absent.

### Step/time interpretation

The `step` column is a solver iteration count, not dimensional physical time. Physical-time interpretation requires the relevant nondimensionalisation and scaling assumptions.

## Known limitations to keep visible

- Periodic boundary condition validation remains incomplete.
- Optional Numba acceleration is not the reference path.
- The known Numba SL=1 discrepancy remains to be investigated.
- GUI behaviour can depend on desktop plotting/Tk availability, especially on macOS.
- The bundled examples are smoke-test inputs, not a comprehensive validation benchmark suite.

## Suggested next release tasks

Before a formal release, consider:

1. Re-run the full test suite on all supported Python versions through CI.
2. Confirm the README quick-start commands match the current CLI.
3. Confirm the project health checklist is up to date.
4. Confirm release notes match the actual merged PRs.
5. Create or update a GitHub release tag only after CI is green.
6. Archive or cite the release according to the repository citation policy.

## Maintainer checklist for formal release

```bash
git switch main
git pull origin main
git status
python -m py_compile run_ldsfl.py ldsfl/main.py gui_ldsfl.py
python -m pytest
git log --oneline -10
```

Then check:

- README badges are passing.
- The current version in package metadata is correct.
- The citation metadata is current.
- No untracked generated outputs are present.
- The release branch/tag points to the intended commit.

## Notes on scope

These notes summarise validation and documentation state. They do not claim full scientific validation of every model pathway. Scientific use should still report input files, commit SHA, backend, boundary condition, stopping criteria, output-unit scaling, and any deviations from the default workflow.
