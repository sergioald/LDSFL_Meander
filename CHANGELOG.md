# Changelog

All notable user-facing changes to LDSFL-Meander should be documented in this file.

This project follows a lightweight changelog style inspired by [Keep a Changelog](https://keepachangelog.com/). Version numbers should match the metadata in `pyproject.toml`, `CITATION.cff`, and the README.

## [Unreleased]

### Added

- Contribution guide for development setup, testing, coverage, optional Numba, branch naming, and PR checklist.
- Project changelog to make future releases easier to review.

### Changed

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
