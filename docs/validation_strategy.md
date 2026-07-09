# Validation strategy and known limitations

This document summarises the current validation strategy for LDSFL-Meander and records known limitations that should be considered when using the model for research or portfolio demonstrations.

## Validation philosophy

The repository uses a layered validation approach:

1. **Unit-level checks** for input parsing, CLI behaviour, and small helper functions.
2. **Regression tests** for solver-output bookkeeping and numerical conventions.
3. **Physics-oriented tests** for invariants that should hold independently of a specific benchmark dataset.
4. **Integration tests** for short reproducible runs.
5. **Documented expected failures** for paths that are useful to keep visible but are not yet validated.

This is intentionally different from claiming full physical validation for all possible settings. The tests provide guardrails against implementation errors and regressions.

## Current validated behaviours

| Area | Evidence |
|---|---|
| CLI case parsing | Tests cover empty cases, invalid case IDs, repeated cases, and ranges. |
| Initial curvature convention | Test checks that the first curvature calculation matches the geometry update convention. |
| Variable-history output | Test checks that every solver step is recorded exactly once, including final partial save blocks. |
| Plot disabling | Test checks that cutoff CSVs can still be written while plotting is disabled. |
| Stop-on-sinuosity-stability | Test checks that the stability stop can fire before `max_steps`. |
| Equivalence/HAC performance guard | Test checks that expensive equivalence diagnostics are not computed by default. |
| Flow-field invariants | Tests check zero-curvature response, linearity in curvature amplitude, and serial/threaded NumPy agreement. |

## Known limitations

### Periodic flow boundary condition

The periodic flow path is present but should be treated as experimental until it has dedicated validation against a trusted reference.

Recommended wording:

```text
The free-boundary flow solver is the default validated path. The periodic boundary-condition path is retained for experimental comparison and should be validated for the target study before use.
```

### Optional Numba backend

The Numba backend is optional. It should be used only when installed and when the relevant solver path has been compared against the NumPy backend.

Recommended wording:

```text
NumPy is the reference backend. Numba is an optional acceleration backend and should be checked against NumPy for the selected solver branch.
```

### Statistical stability

The moving-window stability diagnostic is lightweight and useful for monitoring. The equivalence/HAC diagnostic is more rigorous but more expensive. It is therefore opt-in unless used for a stopping criterion.

Recommended wording:

```text
Equivalence-style sinuosity stability is available for publication-style analysis, but it is not computed by default for long runs unless requested.
```

### Demonstration examples

Short example runs are designed for reproducibility and continuous integration. They are not a substitute for full scientific calibration or site-specific validation.

## Suggested validation roadmap

The following additions would strengthen the repository further:

1. Add one or more external benchmark cases with expected summary outputs.
2. Add a documented periodic-boundary comparison test if the periodic path is retained.
3. Add a small analytic transfer-function style test for the free-flow solver if the assumptions can be clearly documented.
4. Add reproducibility artefacts for a paper figure or reference run.
5. Add a versioned `docs/validation_matrix.md` table linking each claim in the README to code/tests.

## What to avoid

Avoid mixing unrelated changes into validation PRs. In particular:

- do not combine solver changes with documentation-only changes;
- do not regenerate large fixture files in the same PR as code refactors;
- do not enable strict lint gates until the repository is fully lint-clean;
- do not mark optional accelerated backends as equivalent unless the tested path supports that claim.

## Review checklist

Before merging a validation-related PR, check:

- Does the PR state whether it changes behaviour or only adds tests/docs?
- Are new tests focused and deterministic?
- Do tests run without optional dependencies unless explicitly skipped?
- Does the PR avoid committing generated output files?
- Does the README avoid overclaiming beyond the available validation evidence?
