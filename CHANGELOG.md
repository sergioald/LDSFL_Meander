# Changelog

All notable user-facing changes to LDSFL-Meander should be documented in this file.

This project follows a lightweight changelog style inspired by [Keep a Changelog](https://keepachangelog.com/). Version numbers should match the metadata in `pyproject.toml`, `CITATION.cff`, and the README.

## [Unreleased]

### Fixed

- Corrected the optional Numba `SL=1` sub-resonant routing so its lambda-2 and lambda-3 terms use the same downstream `SEMIANA2` kernels as the NumPy reference; dedicated CI now exercises Numba parity.
- Centralized scientific parameter and core run-control validation, including exact integer checks for `Id`, `flagbed`, `Mdat`, save intervals, worker counts, and other discrete controls.
- Rejected dimensional output unless finite positive physical length and velocity scales are available; the CLI now requires both scales explicitly.
- Made `--no-plots` suppress every PNG, while sinuosity CSV rows are appended incrementally and exactly once.
- Added `state_step` to variable histories and included the final post-step state, aligning variable and sinuosity histories from step 0 through the final completed step.
- Made missing final numerical outputs fatal and recorded finalization state and errors in `run_status.json`.
- Included freshwater density when converting bed shear stress in pascals to Shields stress.
- Preserved previous simulations by reserving a separate output directory for every run, including continuation segments and colliding historical labels.
- Preserved dimensional output scaling during GUI continuation and aligned initial overlays with output units.
- Selected all cases by their actual CSV IDs; rejected missing, duplicate and invalid IDs before batch execution.
- Excluded disabled limits from combined stopping criteria and rejected runs with no effective stopping criterion.
- Limited moving-window diagnostic array conversion to the requested window, avoiding quadratic copying over long runs.

### Added

- Per-run solver configuration and input hashes in `run_config.json` for CLI and GUI runs.
- Contribution guide for development setup, testing, coverage, optional Numba, branch naming, and PR checklist.
- Project changelog to make future releases easier to review.

### Changed

- Variable histories now contain `N + 1` state rows for an `N`-iteration run; `jt` remains as the legacy counter and `state_step` is the completed-iteration count.
- Documentation and release hygiene are now tracked explicitly through contributor and changelog files.

## [0.6.5] - documented release

This is the first changelog entry added to the repository. Earlier detailed history should be reconstructed from Git history, GitHub releases, and archived Zenodo releases where available.

### Added

- Step-vs-sinuosity stability diagnostics.
- Sinuosity history CSV and plot outputs.
- GUI support for run diagnostics and final planform inspection.
- GUI controls for graceful stopping after the current step.
- GUI continuation workflow from the latest saved output.
- Scrollable GUI input and diagnostics tabs.
- Bundled example inputs and reproducible output references.
- Pytest-based validation suite.
- GitHub Actions workflow for automated testing.
- CI coverage reporting through `pytest-cov`.

### Changed

- README refreshed for a clearer portfolio/reviewer workflow.
- Detailed project explanation moved into `docs/portfolio_summary.md`.
- Package and citation metadata aligned toward the documented `0.6.5` release state.
- Coverage artifacts ignored through `.gitignore`.

### Fixed

- Non-finite input handling in `dxdy2` to avoid unnecessary NumPy runtime warnings before the intended error path.
- GUI sinuosity diagnostics refresh logic so displayed metrics are derived from the plotted sinuosity history when available.

## Earlier versions

Earlier versions are not yet summarized in this changelog.

For historical details, consult:

- Git commit history,
- GitHub release notes,
- Zenodo archived release records,
- previous README/manual revisions.
