# Reproducible short example

Command run from the repository root:

```bash
python run_ldsfl.py --base-dir . --cases 1 --max-steps 5 --nprint 2 --max-cut 1 --no-plots
```

This example uses the bundled `Input/` folder and creates a short run in `Output/`.

## What is checked, and how

`tests/test_reproducible_example.py` runs this case and compares against the
files here. The checks are deliberately split by how portable they are:

| File | Checked | Portable across machines |
| --- | --- | --- |
| `expected_tree.txt` | exactly | yes |
| `expected_summary.json` integer fields (`steps`, `cut_cnt`, `jt`, `id_files`) | exactly | yes |
| `expected_summary.json` float fields (`dt_cum`, `sinuo_final`) | only when `environment.json` matches | no |
| `expected_output/` CSVs | not compared value-by-value | no |

The floating-point values depend on the BLAS build and on the NumPy/SciPy
versions, so they are only meaningful alongside `environment.json`. When the
recorded versions do not match the ones in use, that check skips rather than
failing.

If the floats differ by more than rounding on the *same* recorded stack, that
is a behaviour change and belongs in `CHANGELOG.md`.

## Regenerating

```bash
python examples/reproducible_case1_short/regenerate.py
```

This rewrites `expected_output/`, `expected_tree.txt` and `environment.json`
together. Update the float fields in `expected_summary.json` from the printed
run summary at the same time, and say in the commit message why they moved.
