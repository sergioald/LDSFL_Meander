# Long-term vertical-only backend parity record

**Authoritative run:** 10,000 migration steps, case 1, completed 2026-09-25. The reference used `backend="numpy", vertical_backend="numpy"`; the accelerated run used `backend="numpy", vertical_backend="numba"`. All other solver controls and input files were identical. This isolates the vertical optimization.

| Run | Completed steps | Wall time | Cutoff events | Final Ns | Final sinuosity |
|---|---:|---:|---:|---:|---:|
| Reference (NumPy vertical) | 10,000 | 4,716.290 s (78 min 36.29 s) | 109 | 3,338 | 1.7199176279642627 |
| Accelerated vertical only (Numba) | 10,000 | 2,547.519 s (42 min 27.52 s) | 109 | 3,336 | 1.7191263031396460 |

Cutoff counts, event sequence, and steps all matched across the two runs (109 each; no differing event). The absolute final sinuosity difference is `0.0007913248246167459`.

## Compatible-grid checkpoint geometry errors

Values below compare corresponding vertex indices without interpolation. Each pair is maximum absolute error / RMS error. The 10,000-step grids are incompatible, so pointwise geometry errors are omitted there (final `Ns` differs by two).

| Step | x error | y error | Curvature error | U error |
|---:|---:|---:|---:|---:|
| 0 | 0 / 0 | 0 / 0 | 0 / 0 | 5.0047e-15 / 1.8030e-15 |
| 1 | 1.7754e-18 / 5.6142e-20 | 1.8943e-15 / 6.8037e-16 | 1.5759e-15 / 5.3745e-16 | 5.4332e-15 / 1.8276e-15 |
| 10 | 1.1369e-13 / 5.2266e-14 | 5.3022e-15 / 1.3874e-15 | 2.5696e-15 / 7.3033e-16 | 1.7195e-14 / 7.2445e-15 |
| 100 | 6.8212e-13 / 2.6529e-13 | 1.7677e-12 / 4.8682e-13 | 1.2640e-13 / 3.3339e-14 | 2.7649e-12 / 8.9923e-13 |
| 1,000 | 6.9576e-11 / 3.9487e-11 | 4.8570e-11 / 2.1721e-11 | 3.0144e-12 / 1.0720e-12 | 3.4133e-11 / 1.1807e-11 |
| 5,000 | 5.0517e-7 / 1.0842e-7 | 5.1888e-7 / 1.3393e-7 | 2.3108e-8 / 3.9984e-9 | 2.8915e-7 / 6.0567e-8 |

## First scalar-error threshold crossings

Entries are the first completed step at which the absolute difference crossed the threshold. `—` means it did not cross during the 10,000-step run.

| Scalar | 1e-12 | 1e-10 | 1e-8 | 1e-6 |
|---|---:|---:|---:|---:|
| Sinuosity | 2,119 | 3,993 | 6,364 | 9,073 |
| beta | 2,203 | 4,062 | 6,447 | 9,225 |
| theta0 | 2,494 | 5,335 | 7,589 | 9,996 |
| ds | 6,285 | 8,956 | 9,996 | — |
| dt_cum | 1 | 1 | 1 | 1 |

`dt_cum` exceeded these absolute thresholds at step 1 (difference `2.0265579223632812e-6`); its accumulated absolute difference is therefore not a near-machine-precision parity measure over long evolution. The discrete `cut_cnt` histories remained identical throughout.

## Interpretation

The isolated vertical optimization agrees initially to near floating-point precision. Small numerical differences accumulate over repeated solver steps. After repeated discrete cutoffs and resampling, exact pointwise topology need not remain identical indefinitely; here both runs had the same 109 cutoff events on the same steps, while their final grid sizes differed by two points. Over the last quarter of the run, sinuosity distributions were descriptively very close (relative mean difference `4.05e-7`, relative standard-deviation difference `8.73e-5`); this is descriptive context, not a separate scientific acceptance criterion.

For this 10,000-step vertical-only comparison, the evidence supports numerical parity of the Numba vertical calculation with the NumPy reference at the trajectory and morphology level, while not claiming bitwise identity or indefinite pointwise topology identity. Machine timings are indicative and may vary.

## 100,000-step extension status

The attempted 100,000-step extension was interrupted at approximately 15,300 reference steps by an external Codex/API authentication failure, not by a solver failure. It is incomplete and is not presented as a completed validation. The 100,000-step runner remains available for future release or publication studies; the completed 10,000-step comparison is the current long-term validation record.

## Saved run artifacts

The detailed machine-readable results and CSVs are in the ignored generated directory `benchmarks/Output/long_term_results/20260925T183530Z_705100/vertical-only/` (`summary.json`, `checkpoint_comparison.csv`, `scalar_history_comparison.csv`, `cutoff_comparison.csv`, and `report.md`).
