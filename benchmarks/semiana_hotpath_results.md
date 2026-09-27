# SEMIANA finite-window response: validation record

The recursive recurrence remains a **finite-window** calculation: it updates
the weighted interior while subtracting the point that leaves the `jtoll`
window. It is not an infinite-IIR approximation. The public Numba flowfield
now selects this recursive strategy for the validated serial, cached,
non-fastmath SL0/SL1 paths. The legacy cached implementation remains available
as a private reference for benchmarks and validation.

The benchmark uses case 1 controls from `Input/Parameter.csv` (`Mdat=6`),
`SL=0, beta=9` or `SL=1, beta=6/12` for sub/super-resonance. Synthetic
equally spaced `s` spans `[0, 10]`; curvature is
`1e-3 * sin(2*pi*s/10)`. Each path keeps `TOLL=1e-4`, `paral=0`,
`numba_parallel=False`, and `numba_fastmath=False`. The JIT is warmed before
sampling. Vertical work uses the same warmed Numba backend and exact `Cf0`
in both response strategies. Times are milliseconds on one machine and are
indicative.

## Pre-change legacy baseline

Five timed repetitions followed an untimed warm-up for every scenario and N.
The response column is min / median / max. Other time columns are medians.
The generated JSON holds min / median / max for *all* timing fields and the
`jtoll` value for every mode and eigenvalue.

| Path | N | Flag | jtoll min / med / max | Response ms min / med / max | Total flow ms | Modal ms | Vertical ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| SL0 | 51 | 1 | 5 / 18 / 52 | 1.446 / 1.578 / 1.978 | 3.030 | 1.256 | 0.015 |
| SL0 | 250 | 1 | 20 / 83.5 / 251 | 3.369 / 4.162 / 5.681 | 5.968 | 1.428 | 0.018 |
| SL0 | 1000 | 1 | 75 / 330.5 / 1001 | 27.440 / 33.694 / 39.875 | 35.635 | 1.780 | 0.025 |
| SL0 | 2500 | 1 | 186 / 825 / 2501 | 143.875 / 153.324 / 189.532 | 155.111 | 1.461 | 0.021 |
| SL0 | 5000 | 1 | 370 / 1648.5 / 5001 | 541.864 / 544.387 / 561.070 | 545.636 | 1.312 | 0.020 |
| SL1 sub | 51 | 1 | 5 / 16 / 52 | 1.317 / 1.465 / 1.536 | 2.683 | 1.235 | 0.015 |
| SL1 sub | 250 | 1 | 19 / 74.5 / 251 | 2.955 / 3.186 / 3.595 | 4.437 | 1.210 | 0.017 |
| SL1 sub | 1000 | 1 | 70 / 296 / 1001 | 24.391 / 25.661 / 27.014 | 27.160 | 1.748 | 0.020 |
| SL1 sub | 2500 | 1 | 174 / 738 / 2501 | 130.570 / 131.633 / 141.410 | 132.992 | 1.445 | 0.025 |
| SL1 sub | 5000 | 1 | 347 / 1475 / 5001 | 443.016 / 492.261 / 597.700 | 493.665 | 1.333 | 0.022 |
| SL1 super | 51 | -1 | 5 / 20 / 52 | 1.108 / 1.129 / 1.809 | 2.281 | 1.111 | 0.016 |
| SL1 super | 250 | -1 | 21 / 94 / 251 | 2.432 / 2.478 / 4.210 | 3.479 | 0.951 | 0.017 |
| SL1 super | 1000 | -1 | 79 / 374 / 1001 | 21.158 / 21.782 / 23.543 | 22.958 | 1.074 | 0.022 |
| SL1 super | 2500 | -1 | 195 / 932 / 2501 | 108.645 / 114.789 / 183.501 | 115.941 | 1.109 | 0.021 |
| SL1 super | 5000 | -1 | 388 / 1864 / 5001 | 435.872 / 503.532 / 565.532 | 504.444 | 1.047 | 0.020 |

At N=5000, some `jtoll` values reach `N+1`, making the corresponding finite
windows boundary-limited. The modal and vertical medians remain around 1 ms
and 0.02 ms; SEMIANA dominates the large-N baseline.

## Recurrence and paired performance

The derivation is in `semiana_hotpath_design.md`. Upstream, scan stations
backward with `w=exp(-lambda*deltas)`; downstream, scan forward with
`v=exp(lambda*deltas)`. In each direction, update the weighted interior sum
by adding the entering point and **subtracting the point leaving the finite
`jtoll` window**. Keep the existing endpoint corrections outside that sum.
The scan direction gives a multiplier of magnitude below one for the routed
eigenvalue. Shortened physical-boundary windows and the empty interior for
`jtoll <= 2` are handled explicitly.

The paired comparison used nine warmed repetitions for each strategy in one
process. Response speedup is legacy median divided by recursive median.

| Path | N | Legacy response ms | Recursive response ms | Response speedup | Legacy total flow ms | Recursive total flow ms |
|---|---:|---:|---:|---:|---:|---:|
| SL0 | 51 | 1.265 | 1.243 | 1.02x | 2.255 | 2.360 |
| SL0 | 250 | 3.106 | 1.615 | 1.92x | 4.045 | 2.674 |
| SL0 | 1000 | 24.006 | 2.948 | 8.14x | 25.135 | 4.208 |
| SL0 | 2500 | 134.333 | 5.379 | 24.98x | 135.693 | 6.780 |
| SL0 | 5000 | 533.425 | 9.145 | 58.33x | 534.863 | 10.394 |
| SL1 sub | 51 | 1.217 | 1.205 | 1.01x | 2.175 | 2.292 |
| SL1 sub | 250 | 3.070 | 1.584 | 1.94x | 4.520 | 2.750 |
| SL1 sub | 1000 | 24.206 | 2.837 | 8.53x | 25.560 | 4.116 |
| SL1 sub | 2500 | 128.584 | 5.237 | 24.55x | 130.014 | 6.726 |
| SL1 sub | 5000 | 499.715 | 8.495 | 58.83x | 501.024 | 9.941 |
| SL1 super | 51 | 1.431 | 1.323 | 1.08x | 2.621 | 2.504 |
| SL1 super | 250 | 3.066 | 1.578 | 1.94x | 4.305 | 2.529 |
| SL1 super | 1000 | 25.008 | 2.638 | 9.48x | 26.532 | 3.701 |
| SL1 super | 2500 | 137.862 | 4.774 | 28.88x | 139.166 | 5.904 |
| SL1 super | 5000 | 551.938 | 9.170 | 60.19x | 554.470 | 10.495 |

At N=51 the response medians are effectively tied; the small differences in
total flow time are within the observed timing spread. This synthetic scaling
test does not establish a whole-simulation speedup.

## Numerical parity

Low-level tests compare every station with the legacy cached Numba finite
convolution for `jtoll` of 2, 3, small, intermediate, almost N, N, and N+1;
positive and negative real parts; nonzero imaginary parts; weak and stronger
decay; and both physical boundaries. Complete flowfield tests cover SL0,
SL1 sub-resonant, SL1 super-resonant, and the borderline SL1-to-SL0 fallback.
An earlier three-step solver smoke run comparing the internal legacy and
recursive paths also matched. These tests use the established full-flowfield
`rtol=1e-9, atol=1e-11` without relaxing it.

Across the paired benchmark's 15 legacy-versus-recursive flowfields, the
largest absolute difference was `1.033e-15`, the largest RMS difference was
`5.446e-16`, and the largest relative difference was `1.489e-9` using a
`1e-12` denominator floor (near-zero outputs drive this last figure).
Resonance flags matched in every case. Against the NumPy response with the
same Numba vertical coefficients, the largest absolute difference for
`N <= 1000` was `7.416e-17`. No unexpectedly large numerical error was found
in these flowfield tests; the long-term trajectory evidence follows below.

Generated details are under the ignored `benchmarks/Output/semiana_hotpath/`
directory: `legacy_prechange.json` contains the original baseline;
`paired.json` contains both strategies, all timing min/median/max values,
per-mode/eigenvalue `jtoll`, and per-case parity metrics.

## Long-term trajectory validation

The 10,000-step legacy-versus-recursive A/B completed in both runs without
solver errors. Both recorded 109 cutoff events, with the exact cutoff-event
step sequence matching. This supports numerical parity for the recursive
finite-window SEMIANA response; it does not establish bitwise-equivalent
long-term trajectories.

| Measure | Legacy | Recursive |
|---|---:|---:|
| Completed migration steps | 10,000 | 10,000 |
| Final Ns | 3,338 | 3,336 |
| Final sinuosity | 1.7199176762250783 | 1.7191265744285114 |
| Final sinuosity absolute difference | — | 0.0007911017965669 |
| Total wall time | 549.773 s | 388.956 s |
| Total flowfield time | 230.782 s | 60.793 s |
| SEMIANA response time | 211.532 s | 41.425 s |

The resulting speedups were about **5.106x** for the SEMIANA response,
**3.796x** for the complete flowfield, and **1.413x** for the whole solver.
These machine timings are indicative and vary by hardware and runtime
conditions.

The final two-point Ns difference traces to step 9,996, the shared 109th
cutoff event. Immediately before the cutoff, both geometries had Ns=3,366.
The existing left-to-right first-hit detector uses the strict condition
`distance < dslim3`. At candidate `i=1249`, the legacy trajectory's nearest
eligible pair was `8.30256104365513` against `dslim3=8.302513160824043`, or
`4.7883e-5` outside the threshold. The recursive trajectory's corresponding
pair was `8.302453448870402` against `dslim3=8.302513627057195`, or
`6.0178e-5` inside it. The scan therefore selected `(1250, 1275)` for legacy
and the earlier `(1249, 1276)` for recursive. `geometry4` removed 28 versus
30 points. Both used the original KDTree fast path; the selected pairs and
removal counts agree with the detector and cut indexing. No cutoff indexing
defect was found. The difference is explained by a strict threshold crossing
in the existing first-hit logic.

After repeated discrete cutoffs and resampling, pointwise topology and exact
grid correspondence need not remain identical indefinitely. Small
floating-point trajectory drift begins before the first cutoff; nonlinear
evolution, resampling, and cutoff events amplify that drift. The final
Ns=3,338 versus 3,336 difference is specifically explained by the strict
first-hit cutoff threshold crossing at step 9,996, as detailed above. The
10,000-step study is the current long-term validation record. Longer runs
remain available for future release or publication studies.

## Public-default selector smoke validation

After the selector change, the public Numba path was compared with an
explicit recursive-strategy override in a 500-step `run_case` run. Both runs
completed without errors at Ns=1,000, with zero cutoffs, sinuosity
1.0251146220575202, and resonance flag `1` (sub-resonant). Wall times were
11.538 s for the public-default run and 10.405 s for the explicit-recursive
run; JIT warm-up was timed separately. At the compatible step-500 grid, max
absolute and RMS differences were exactly zero for x, y, s, curvature, and U.
The final scalar differences, including sinuosity and cumulative time, were
also zero. This confirms the public selector routes to the recursive
implementation; these timings are a short wiring check, not a new performance
claim.
