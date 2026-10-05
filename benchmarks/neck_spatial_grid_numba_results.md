# Optional Numba spatial-grid neck detector

This report separates the accepted benchmark-prototype evidence from the
final checks made through the public `run_case` and `geometry4` backend
settings. Wall-clock figures are machine- and workload-specific.

Environment: Windows 11, Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1, Numba
0.67.0, and llvmlite 0.49.0. Inputs were case 1 at repository commit
`5b3f4626714fdcf34c4c49fa5108dd617c1154bb`.

## Algorithm and public behavior

The candidate uses the reviewed serial Numba uniform spatial grid: a
conservative cell width, an open-addressed hash table, and a 3-by-3 neighboring
cell search. It retains left-to-right `i` scanning, the original eligible `j`
domain, strict `d2 < r2`, closest-distance selection, and smallest-`j`
tie-breaking. The kernel uses `cache=False`, `fastmath=False`, and
`parallel=False`; no `prange` is used. The candidate shares the production
linear-refinement and original-coordinate validation path with the KDTree
reference.

The independent public option is `neck_detector_backend="kdtree"` or
`"numba_grid"`. `kdtree` remains the default and does not import Numba.
`numba_grid` imports Numba lazily and reports a targeted installation error if
the optional dependency is missing. The setting is independent of both the
flow backend and geometry unwrap backend. No automatic backend selection or
silent dependency fallback is performed.
If an individual `numba_grid` search cannot safely represent its cell
coordinates in the integer grid, it delegates to the KDTree reference. This
is a numerical-safety fallback within the selected backend, not automatic
backend selection or a missing-dependency fallback.

## Benchmark-prototype evidence

The accepted benchmark-only capture at
`benchmarks/Output/neck_spatial_grid_numba/20261004T190123.677262Z` compared
the candidate directly with the KDTree reference. All 1,710 captured
production detector calls and all 12 synthetic/adversarial cases matched
exactly. The captured real corpus contained 44 hits, 1,666 no-hit calls, and
40 refinement searches. Replay speedup was 21.5038x / 22.1593x / 22.6199x
(min / median / max).

The benchmark-only 1,000-step gate matched exactly: both runs completed 1,000
steps with Ns=1187, no cutoffs, and sinuosity 1.1917102476657917.

Three exact 5,000-step solver pairs had these timings:

| Pair | Reference wall (s) | Candidate wall (s) | Reference detector (s) | Candidate detector (s) | Detector speedup | Geometry speedup | Whole-solver speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25.697378 | 15.794629 | 9.796595 | 0.490492 | 19.9730x | 2.7449x | 1.6270x |
| 2 | 26.193979 | 16.732791 | 9.977587 | 0.506822 | 19.6866x | 2.6367x | 1.5654x |
| 3 | 25.320582 | 15.881217 | 9.777893 | 0.490594 | 19.9307x | 2.7116x | 1.5944x |

All three had exact cutoff steps, selected pairs, removed-point counts, scalar
histories, checkpoint states, and resonance flags. Each ended at Ns=2359 with
44 cutoffs and sinuosity 1.7245401220504077. The median speedups were 19.9307x
for detector calls, 2.7116x for geometry, and 1.5944x for the whole solver.
These are validated case-1 measurements, not a universal performance promise.

## Public production-path validation

The public runner is
`benchmarks/neck_spatial_grid_numba_validation.py`. It calls `run_case` with
the selected public setting; observation wrappers capture checkpoints and
normal cutoff-output calls without replacing the detector. Generated runs and
summaries are under the ignored
`benchmarks/Output/neck_spatial_grid_numba_public/` directory.

Both legs used identical controls: Numba flow backend, Numba geometry unwrap,
free-boundary flow, serial/non-fastmath settings, `cstab=0.01`, plots disabled,
and the requested fixed step limit. Only the neck detector setting differed.

### 1,000-step public gate

Both the `kdtree` reference and `numba_grid` candidate completed 1,000 steps.
Both finished with Ns=1187, zero cutoffs, and sinuosity
1.1917102476657917. Checkpoint geometry/state, scalar history, and resonance
flags matched exactly. This satisfies the recorded expected trajectory.

### 5,000-step public confirmation

The order was candidate then reference, alternating from the final
benchmark-prototype pair. Both completed 5,000 steps and matched exactly at
checkpoints 0, 1, 10, 100, 500, 1,000, 2,500, and 5,000. Scalar histories,
resonance flags, cutoff sequence, selected pairs, and removed-point counts
were exact. Both ended at Ns=2359, sinuosity 1.7245401220504077, and 44
cutoffs.

| Measurement | KDTree reference | Numba grid candidate |
|---|---:|---:|
| Wall time | 25.994978 s | 18.792452 s |
| `flowfield_total` | 9.146212 s | 9.893522 s |
| Geometry | 15.276483 s | 6.838052 s |
| Neck detector | 9.978157 s | 1.440138 s |

The candidate wall time was 1.3833x faster and geometry was 2.2340x faster.
The detector figure for this first-use candidate run includes the Numba JIT
compile, so its startup-inclusive detector ratio is 6.9286x. The run had
1,710 detector searches and the expected exact trajectory. The candidate ran
first and paid the cold Numba flow-kernel compilation as well; the reference
ran second with that flow kernel already compiled. The measured
`flowfield_total` values therefore are not a warm paired comparison.

## Fresh post-integration candidate profile

One additional public 5,000-step candidate run followed the A/B pair in the
same process, so the selected Numba kernels were already JIT-compiled. It
completed with the expected 5,000 steps, Ns=2359, 44 cutoffs, and sinuosity
1.7245401220504077.

| Timer | Seconds | Share of wall |
|---|---:|---:|
| Wall | 15.723070 | 100% |
| `flowfield_total` | 8.798102 | 55.96% |
| `semiana_response` | 5.805627 | 36.92% |
| `modal_coefficients` | 1.780775 | 11.33% |
| Geometry | 5.460835 | 34.73% |
| `geometry_neck` | 0.634935 | 4.04% |
| `geometry_neck_detector` | 0.485046 | 3.08% |
| `geometry_initial_uniformization` | 3.912753 | 24.89% |
| `geometry_curvature` | 0.494269 | 3.14% |

The warmed production detector measurement was 20.5716x faster than the
KDTree detector time in the immediately preceding exact 5,000-step reference
run. This aligns with the three prior warmed prototype pairs and explains the
lower startup-inclusive ratio: the public candidate's first measured detector
time includes one-time JIT compilation. The largest aggregate timer is
`flowfield_total`; among the reported leaf timers, `semiana_response` is the
largest. No further optimization was performed.

## Limitations

Exactness was established on the deterministic synthetic corpus, the captured
case-1 detector corpus, and the public case-1 trajectory through 5,000 steps.
The measurements do not establish universal speedups for other geometries,
hardware, Numba versions, or solver controls. First use includes Numba
compilation. The default remains the SciPy KDTree reference path, and Numba
remains optional.
