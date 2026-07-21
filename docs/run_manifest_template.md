# Run manifest template

This page explains how to use [`examples/run_manifest_template.yml`](../examples/run_manifest_template.yml)
to make LDSFL-Meander runs easier to reproduce.

The manifest is not consumed by the solver. It is a lightweight metadata file that
users can copy into a results archive, fill in, and share with outputs, figures,
review packages, or paper supplements.

## Why use a manifest?

A solver output folder alone does not fully describe a run. A reproducible archive
should also record:

- the repository commit or release tag;
- the Python environment;
- the command line or GUI settings;
- the selected input case IDs;
- the generated `id_files` output directory;
- backend and boundary-condition settings;
- stop criteria and stability-diagnostic settings;
- dimensional-output scales, if dimensional outputs were requested.

The template gives these records a consistent structure.

## Recommended workflow

Copy the template into your run archive:

```bash
cp examples/run_manifest_template.yml run_manifest.yml
```

On Windows PowerShell:

```powershell
Copy-Item examples/run_manifest_template.yml run_manifest.yml
```

Fill in the placeholders before or immediately after the run. At minimum, record:

```bash
git rev-parse HEAD
python --version
python -m pip freeze > environment/pip_freeze.txt
```

Then run the model, for example:

```bash
python -m run_ldsfl --base-dir . --cases 1 --max-steps 1 --no-plots
```

The generated output directory is `Output/<id_files>/`, where `id_files` is created
from the selected case ID and relevant parameter values. Do not assume it is simply
`Output/<case_id>/`.

## Suggested archive layout

```text
run_archive/
├── run_manifest.yml
├── README_run_notes.md
├── Input/
│   ├── Parameter.csv
│   └── xy.csv
├── Output/
│   └── <id_files>/
├── environment/
│   ├── pip_freeze.txt
│   └── conda_environment.yml  # optional
└── figures/                  # optional derived figures
```

## Notes

- Use `backend: numpy` for the reference/default path.
- Treat `flow_boundary_condition: periodic` as experimental unless the study has
  a specific validation case for it.
- Record whether equivalence/HAC sinuosity diagnostics were requested or used for
  stopping.
- If dimensional outputs are requested, record the length and velocity scales.
- Keep large archives outside Git unless they are intentionally curated as small
  example artefacts.
