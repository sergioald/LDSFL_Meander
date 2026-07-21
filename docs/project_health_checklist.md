# Project health checklist

This checklist summarises the currently recommended safe paths, optional paths, and release checks for LDSFL-Meander.

Use it before opening a pull request, preparing a release, or running a new scientific study with the repository.

## Current health status

The repository is in a good development state when all of the following are true:

- the working tree is clean;
- `main` is up to date with `origin/main`;
- the full test suite passes;
- documentation links in the README are valid;
- any optional backend or boundary-condition path used for a study has been compared against the default path.

Recommended baseline check:

```bash
git switch main
git pull origin main
git status
python -m pytest
```

Expected current test status:

```text
58 passed, 2 skipped
```

The exact number may change as new tests are added, but failures should be investigated before merging or releasing.

## Safe/default paths

These are the recommended default paths for general users and for reproducible examples.

| Area | Recommended path | Notes |
|---|---|---|
| Backend | `numpy` | Reference implementation and default path. |
| Flow boundary condition | `free` | Best-supported solver path. |
| CLI plotting | `--no-plots` for automated runs | Safer for CI, headless systems, and macOS troubleshooting. |
| GUI | Use after CLI smoke test passes | GUI issues can be backend/environment-specific. |
| Stability diagnostic | Lightweight moving-window diagnostic by default | Avoids repeated full-history HAC work in long runs. |
| Equivalence/HAC diagnostic | `--return-equivalence-stability 1` only when needed | Computes the final diagnostic explicitly. |
| Stop-on-stability | `--stop-on-sinuosity-stability 1` only for studies that require it | Uses the equivalence-style diagnostic as a stopping criterion. |

## Optional or experimental paths

These paths are useful but should be treated carefully.

### Numba backend

The Numba backend is an optional acceleration path. Use it only after checking that it agrees with NumPy for the selected solver branch and case.

Recommended comparison workflow:

```bash
python -m run_ldsfl --base-dir . --cases 1 --backend numpy --max-steps 100 --no-plots
python -m run_ldsfl --base-dir . --cases 1 --backend numba --max-steps 100 --no-plots
python -m pytest tests/test_flowfield_backend_guardrails.py
```

Do not assume Numba is valid for every solver path simply because it is installed.

### Periodic flow boundary condition

The free-boundary solver is the default supported path. The periodic flow path should be treated as experimental until it has been validated for the target study.

Before using periodic flow in a result that will be reported, record:

- the exact case and parameters;
- the reason periodic flow is required;
- comparison against the free-boundary path where meaningful;
- any available reference or regression result;
- whether outputs are physically and numerically consistent.

### macOS GUI plotting

The CLI and file-output paths should not require an interactive matplotlib backend. GUI behaviour can depend on the Python, Tk, and matplotlib installation.

Before reporting a macOS GUI problem, confirm:

```bash
python -m run_ldsfl --base-dir . --cases 1 --max-steps 1 --no-plots
python -m pytest
```

If those pass, the issue is likely GUI/backend related rather than a solver failure.

See [`macos_gui_notes.md`](macos_gui_notes.md).

## Pull request checklist

Before opening a PR:

```bash
git status
python -m py_compile run_ldsfl.py ldsfl/main.py gui_ldsfl.py
python -m pytest
```

For code PRs, also consider targeted tests:

| Change area | Suggested targeted test |
|---|---|
| CLI changes | `python -m pytest tests/test_cli_*.py` |
| Sinuosity stability | `python -m pytest tests/test_sinuosity*.py tests/test_cli_sinuosity_stability.py` |
| Flow field | `python -m pytest tests/test_flowfield_*.py` |
| Output writing | `python -m pytest tests/test_*output*.py tests/test_variable_history_complete.py` |
| GUI-related logic | Run available GUI/unit tests and manually smoke-test the GUI if possible. |

For docs-only PRs:

```bash
python -m pytest
git diff --stat
```

Docs-only PRs should not modify solver behaviour, fixtures, generated outputs, or test expectations unless explicitly stated.

## Release checklist

Before tagging or publishing a release:

1. Confirm clean main:

   ```bash
   git switch main
   git pull origin main
   git status
   ```

2. Run tests:

   ```bash
   python -m pytest
   ```

3. Run a CLI smoke test:

   ```bash
   python -m run_ldsfl --base-dir . --cases 1 --max-steps 1 --no-plots
   ```

4. Check metadata:

   - `pyproject.toml` version;
   - `CITATION.cff` metadata;
   - README badges and links;
   - documentation links under `docs/`;
   - release notes or changelog, if used.

5. Check optional-path wording:

   - NumPy is the reference backend;
   - Numba is optional and must be validated for the selected path;
   - free-boundary flow is the default supported path;
   - periodic flow is experimental unless validated for the study;
   - timestep output is solver iteration/computational time, not automatically dimensional physical time.

6. Confirm no accidental generated outputs are staged:

   ```bash
   git status
   git diff --stat --cached
   ```

## Study/reporting checklist

Before using results in a paper, report, or presentation, record:

- repository commit SHA;
- version number if using a release;
- Python version and environment;
- backend used, for example `numpy` or `numba`;
- boundary condition used, for example `free` or `periodic`;
- selected case IDs and input files;
- `max_steps`, `Nprint`, and `cstab`;
- whether plots were generated;
- whether stop-on-stability or final equivalence/HAC diagnostics were enabled;
- location of output files;
- any manual changes to input data.

Suggested command for recording the commit:

```bash
git rev-parse HEAD
```

## Bug-report checklist

When reporting an issue, include:

- operating system;
- Python version;
- installation method, for example conda, venv, system Python, or Homebrew;
- exact command used;
- full traceback or error message;
- `git rev-parse HEAD`;
- whether `python -m pytest` passes;
- whether the minimal CLI smoke test passes;
- small input files or case ID needed to reproduce;
- for plotting/GUI issues, the matplotlib backend and whether `import tkinter` works.

Useful diagnostic commands:

```bash
python --version
python -m pytest
python -m run_ldsfl --base-dir . --cases 1 --max-steps 1 --no-plots
python - <<'PY'
import matplotlib
print(matplotlib.get_backend())
try:
    import tkinter
    print('tkinter OK', tkinter.TkVersion)
except Exception as exc:
    print('tkinter failed:', repr(exc))
PY
git rev-parse HEAD
git status
```

## Maintainer notes

- Keep the default path conservative and reproducible.
- Prefer small PRs with one clear purpose.
- Keep solver changes separate from documentation-only changes.
- Add regression tests for every behaviour-changing bug fix.
- Document limitations honestly rather than hiding optional or experimental status.
- Avoid expensive diagnostics in default long-run paths unless the user explicitly requests them.
