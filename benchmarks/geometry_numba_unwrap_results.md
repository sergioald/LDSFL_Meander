# Exact Numba curvature unwrap validation

The production candidate is the serial `unwrap_numba` kernel from the
pre-integration replay study, now available as
`ldsfl.geometry_numba.unwrap_angles_like_matlab_numba`. The validation base was
`583aaf1ab56ad822eddacf3849f3f9c321fd29ad` on branch
`perf/geometry-numba-unwrap`.

Environment: Windows 11; Python 3.12.14; NumPy 2.5.3; SciPy 1.18.1; Numba
0.67.0; llvmlite 0.49.0.

The candidate uses `@njit(cache=False, fastmath=False)`, allocates with
`np.empty_like`, starts the sequential state at `last = 0.0`, and preserves the
strict `<` / `>` comparisons, `np.pi` / `2.0 * np.pi` arithmetic, and original
loop order. It was warmed before solver timing; warm-up took 1.0741 s and
compiled the live 1-D C-contiguous float64 signature in-process. No Numba disk
cache was used.

Five alternating paired full-solver runs each completed 5,000 case-1 migration
steps. Python and Numba results were bitwise exact for checkpoint arrays and
scalars, full scalar histories, cutoff events and selected pairs, points
removed, final point count, and final sinuosity. Every run finished at Ns=2359
with 44 cutoffs and sinuosity 1.7245401220504077.

| Pair | Order | Python wall (s) | Numba wall (s) | Wall speedup | Python curvature (s) | Numba curvature (s) | Curvature speedup | Python geometry (s) | Numba geometry (s) | Geometry speedup |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Python → Numba | 69.6840 | 60.8081 | 1.1460x | 7.6725 | 2.2018 | 3.4846x | 40.2665 | 33.2459 | 1.2112x |
| 2 | Numba → Python | 66.3418 | 60.7245 | 1.0925x | 7.2460 | 2.1931 | 3.3040x | 38.4264 | 33.2093 | 1.1571x |
| 3 | Python → Numba | 65.1179 | 59.8436 | 1.0881x | 7.1997 | 2.1541 | 3.3423x | 37.7758 | 32.7681 | 1.1528x |
| 4 | Numba → Python | 65.4460 | 59.5294 | 1.0994x | 7.2669 | 2.1596 | 3.3649x | 38.0721 | 32.6191 | 1.1672x |
| 5 | Python → Numba | 63.9869 | 61.7494 | 1.0362x | 7.0886 | 2.2564 | 3.1416x | 37.1563 | 33.9586 | 1.0942x |

Across the five pairs, curvature speedup was **3.1416x / 3.3423x / 3.4846x**
(min/median/max), geometry speedup was **1.0942x / 1.1571x / 1.2112x**, and
whole-solver speedup was **1.0362x / 1.0925x / 1.1460x**.

The earlier float64-spline replay produced a false performance signal because
reference and candidate repetitions were timed in separate blocks. This study
used alternating paired order and warmed the compiled path before timing. The
timings are machine- and workload-specific; the 1.09x median is not a universal
speedup claim.

## Public production-path confirmation

After integration, the official `run_case` control was used for a Python-versus-
Numba confirmation pair, with the Numba flow and vertical paths and identical
serial/non-fastmath controls. The public unwrap kernel was warmed separately
(0.943 s, excluded from solver timing); the flow backend was also warmed before
the timed runs. The runner used pass-through recorders for checkpoint and
cutoff capture and did not replace the geometry unwrap binding.

Both runs completed 5,000/5,000 steps and were bitwise exact for all recorded
checkpoint arrays and scalars, the full scalar history, cutoff events and
selected pairs, points removed, final Ns, and final sinuosity. Both finished at
Ns=2359 with 44 cutoffs and sinuosity 1.7245401220504077.

| Metric | Python unwrap | Numba unwrap | Python / Numba ratio |
|---|---:|---:|---:|
| Total wall (s) | 69.253 | 74.989 | 0.924x |
| Geometry (s) | 39.987 | 40.528 | 0.987x |
| Curvature (s) | 7.357 | 1.864 | 3.948x |
| Flowfield total (s) | 25.220 | 30.076 | 0.839x |

This single public-path pair confirms the curvature-stage optimization itself.
Its whole-run timing showed no end-to-end gain: Numba took longer overall, with
the flowfield accounting for most of the opposing timing change. That single
measurement is especially sensitive to unrelated runtime variation. The five
alternating pairs above remain the stronger performance estimate; no universal
end-to-end speedup is claimed. All timings depend on machine and workload.
