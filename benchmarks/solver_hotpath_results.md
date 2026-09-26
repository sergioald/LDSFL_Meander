# Cache-audited `k0123` benchmark

The corrected runner is `benchmarks/solver_hotpath_benchmark.py`. It separates
direct Python execution, the raw compiled Numba kernel after an untimed JIT or
persistent-cache warm-up, a public-wrapper exact-key miss, and a public-wrapper
exact-key hit. Each measurement reports min/median/max and cache-counter
changes. The regenerated raw report is `benchmarks/solver_hotpath_audit.json`.

The independent cache-controlled audit measured these medians for case 1 at
`Cf0 = 0.01318086081547672`:

| Measurement | Exact function and cache state | Median |
| --- | --- | ---: |
| Python reference, uncached | `ldsfl.vertical.k0123(Cf0)` called directly | ~124.272 ms |
| Compiled Numba, uncached | Raw `ldsfl.vertical_numba._k0123_kernel(Cf0)` after JIT warm-up | ~0.222 ms |
| Numba public wrapper, exact-key miss | `k0123_numba_cached(Cf0)` after clearing its cache | ~0.318 ms |
| Numba public wrapper, exact-key hit | Same wrapper and exact key already cached | ~5.6 µs |

The 30-call independent sample showed zero LRU counter changes for the direct
Python call and raw Numba kernel. The wrapper-miss calls recorded 30 misses and
no hits; the wrapper-hit calls recorded 30 hits and no misses. The raw compiled
kernel timing excludes the untimed JIT warm-up. The original ~9.4 µs median
came from cache hits, not raw compiled execution.

The independent changing-`Cf0` flowfield sample used 12 distinct, positive,
unrounded exact float keys and recorded 12 misses, zero hits, for each vertical
path. The median end-to-end times were ~216.701 ms with the Python vertical
path and ~5.148 ms with the Numba vertical path. These end-to-end values also
use the corresponding NumPy and Numba flow-response backends. The regenerated
JSON adds a matched Numba flow-response comparison with Python versus Numba
vertical backends to isolate the vertical calculation.

For the independently audited five-step `run_case` sample, NumPy took ~2.614 s
total and Numba took ~0.237 s total after an untimed JIT warm-up. The exact
`Cf0` LRU was cleared before each run; it recorded one hit and five misses
during the six flowfield calls, including the final-snapshot recomputation.
There were five completed migration steps. The regenerated report separates
`flowfield_loop`, `flowfield_final`, and
`flowfield_total`, where total is loop plus final. Its fresh run records each
component and wall time separately.

Machine timings vary with CPU, operating-system math libraries, and system
load; these numbers are indicative. Do not interpret the cache-hit timing as
uncached Numba execution. The Numba `njit` import remains routed through the
existing `flowfield_numba` compatibility helper; moving it to a shared helper
was left as future cleanup because it is unrelated to these review fixes.
