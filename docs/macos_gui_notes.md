# macOS GUI and plotting notes

This page records practical notes for running LDSFL-Meander on macOS and other systems where matplotlib/Tk GUI backends can behave differently from Windows or Linux.

## Recommended first test on macOS

Before launching the GUI, check the CLI path:

```bash
python -m run_ldsfl --base-dir . --cases 1 --max-steps 1 --no-plots
python -m pytest
```

If this passes, the numerical and file-output paths are working. Any remaining issue is likely GUI/backend related rather than a solver failure.

## Why GUI backend issues happen

The GUI uses a desktop plotting stack. On macOS, matplotlib backends can depend on:

- how Python was installed;
- whether Tk is available;
- whether the session is launched from Terminal, an IDE, or Finder;
- whether the environment is headless or remote;
- the installed matplotlib backend configuration.

A common symptom is a crash or freeze when a GUI backend such as `TkAgg` is unavailable or unstable.

## Safe non-GUI mode

For CLI and automated runs, prefer:

```bash
python -m run_ldsfl --base-dir . --cases 1 --max-steps 100 --no-plots
```

This avoids interactive plotting and is the recommended route for:

- GitHub Actions;
- remote servers;
- headless sessions;
- macOS systems with matplotlib/Tk issues;
- reproducibility tests.

## Do not globally force `Agg` for the GUI

The `Agg` backend is useful for non-interactive file rendering, but it is not an interactive GUI backend. Forcing `Agg` globally can avoid some crashes in headless runs, but it can also break or limit interactive GUI behaviour.

Preferred approach:

1. Use non-interactive figure saving for output files.
2. Use `--no-plots` for headless CLI runs.
3. Keep GUI backend configuration local to the GUI path.
4. Avoid changing global matplotlib backend settings inside general output functions.

## If the GUI crashes on macOS

Try the following steps.

### 1. Confirm the CLI works

```bash
python -m run_ldsfl --base-dir . --cases 1 --max-steps 1 --no-plots
```

### 2. Confirm Tk is available

In Python:

```python
import tkinter
print(tkinter.TkVersion)
```

If this fails, install or repair Tk support in the active Python environment.

For conda environments, this may help:

```bash
conda install tk
```

### 3. Check matplotlib backend

In Python:

```python
import matplotlib
print(matplotlib.get_backend())
```

If the backend is not compatible with the session, launch the CLI with `--no-plots` or use a Python environment with a working desktop backend.

### 4. Run from Terminal

On macOS, launching GUI Python programs from Terminal is often more reliable than launching from an IDE or file manager.

```bash
python gui_ldsfl.py
```

## What to report in an issue

If a macOS GUI problem persists, include:

- macOS version;
- Python version;
- installation method, for example conda, pyenv, system Python, or Homebrew;
- matplotlib version;
- output of `matplotlib.get_backend()`;
- whether `import tkinter` works;
- the exact command used;
- the traceback or crash message;
- whether the CLI command with `--no-plots` works.

## Maintainer note

The numerical solver and output-writing path should not require an interactive plotting backend. GUI-specific plotting should remain isolated from CLI/headless output paths so that scientific runs remain reproducible across platforms.
