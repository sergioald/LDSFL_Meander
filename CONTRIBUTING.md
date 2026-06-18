# Contributing to LDSFL-Meander

Thank you for helping improve LDSFL-Meander.

This project is research software for reduced meander-morphodynamics experiments. Contributions should keep the code reproducible, documented, and easy to run from both the command line and the GUI.

## Development setup

From a local clone of the repository:

```powershell
cd C:\Test\LDSFL_Meander
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The editable install keeps your local code changes active while you run tests and scripts.

## Running tests

Run the full test suite with:

```bash
python -m pytest
```

Run a single test file with:

```bash
python -m pytest tests/test_evolution.py
```

Run tests with coverage using:

```bash
python -m pytest --cov=ldsfl --cov-report=term-missing --cov-report=xml
```

Do not commit generated coverage files such as `.coverage`, `coverage.xml`, or `htmlcov/`.

## Optional Numba backend

The default workflow should work without Numba.

If you are testing the optional accelerated backend, install Numba in your development environment:

```bash
python -m pip install numba
```

Then run the relevant backend-specific checks. Keep Numba-related changes optional unless the project intentionally changes its core installation requirements.

## Running the model locally

Run the bundled example from the command line:

```bash
python run_ldsfl.py --base-dir . --cases 1 --max-steps 50 --nprint 10
```

Run without generating planform PNG files:

```bash
python run_ldsfl.py --base-dir . --cases 1 --max-steps 50 --nprint 10 --no-plots
```

Launch the GUI:

```bash
python gui_ldsfl.py
```

## Branch naming

Use short branch names that describe the change:

```text
fix/dxdy2-nonfinite-inputs
test/add-run-case-integration
docs/readme-portfolio-refresh
ci/add-coverage-report
chore/align-package-version
```

## Pull request checklist

Before opening a pull request:

- Start from an up-to-date `main` branch.
- Keep the PR focused on one topic.
- Run `python -m pytest`.
- For CI or testing changes, also run the relevant coverage command.
- Do not commit generated output folders, coverage files, caches, or local virtual environments.
- Update documentation when user-facing behavior changes.
- Update version metadata only when preparing or aligning a release.

## Release and version notes

When preparing a release or aligning metadata, check these files together:

```text
pyproject.toml
CITATION.cff
README.md
CHANGELOG.md
```

The package version, citation version, README badges/text, and changelog entry should describe the same release state.

## Documentation style

Documentation should be clear for three audiences:

- researchers who want to understand the model scope and assumptions,
- users who want to run a bundled example quickly,
- reviewers or recruiters who want to understand the repository structure and software quality.

Prefer concise examples, reproducible commands, and explicit notes about model scope and limitations.
