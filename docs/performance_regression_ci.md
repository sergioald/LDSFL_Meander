# Performance regression CI guardrails

The dedicated **Performance guardrails** workflow runs short, warmed,
same-runner microbenchmarks for the accepted exact Numba backends. It is
separate from the Tests workflow and does not use whole-solver elapsed time as
a blocking metric. The residual 5,000-step audit showed identical trajectories
with wall times from 19.390 s to 49.308 s, so local absolute seconds would be a
noisy hosted-runner limit.

## What is protected

The ordinary Tests workflow runs without the optional Numba extra in its
Python-version matrix. That continues to exercise reference/default behavior
and confirms that importing and using the default solver does not require
Numba. Its Python 3.12 Numba job runs deterministic numerical and dispatch
tests for vertical coefficients, recursive SEMIANA, geometry unwrap, and the
neck detector. Those tests cover exact-Cf0 cache behavior and parity, public
recursive routing plus unsupported-mode legacy fallback, unwrap edge cases,
neck strict-threshold/tie behavior and unsafe-coordinate KDTree delegation,
and the reference defaults.

The separate performance workflow adds warmed measurements for:

| Metric | Reference | Candidate | Blocking floor |
|---|---|---|---:|
| Vertical `k0123` | Uncached Python reference | Raw warmed Numba kernel | 3x |
| SEMIANA response, N=1,000 and 2,500 | Legacy cached Numba finite-window response | Recursive cached Numba finite-window response | 2x each |
| Neck detector, no-hit and late-hit cases | KDTree reference | Numba spatial grid | 2x each |
| Geometry unwrap | Python reference | Exact Numba unwrap | Correctness only |

The thresholds are stored centrally in
[`ci_performance_guardrails.py`](../benchmarks/ci_performance_guardrails.py).
The vertical floor is intentionally far below the previously measured raw
kernel advantage. The SEMIANA floors are below the historical N=1,000 and
N=2,500 response ratios. The neck floor is below the validated production
replay ratio. These are gross-regression checks, not promises about absolute
throughput. Geometry unwrap operates on the real initial input-sized angle
array; its short timing is reported but not gated because runner noise can
dominate it.

Three local complete invocations after warm-up produced these median speedup
ratios. The JSON reports retain every raw sample and each reference/candidate
`range / median` value.

| Metric | Run 1 | Run 2 | Run 3 | Median of runs | Floor |
|---|---:|---:|---:|---:|---:|
| Vertical `k0123` | 288.38x | 292.81x | 305.28x | 292.81x | 3x |
| SEMIANA N=1,000 | 13.38x | 12.44x | 14.58x | 13.38x | 2x |
| SEMIANA N=2,500 | 38.79x | 34.67x | 43.14x | 38.79x | 2x |
| Neck grid, no hit | 48.11x | 60.91x | 47.77x | 48.11x | 2x |
| Neck grid, later hit | 44.53x | 58.42x | 48.93x | 48.93x | 2x |
| Geometry unwrap (not gated) | 216.98x | 107.47x | 185.49x | 185.49x | — |

The three complete runs took 6.01 s, 7.39 s, and 6.31 s. All blocking
comparisons cleared their floor without remeasurement. No implementation in
these final samples crossed the `range / median > 1` variability warning
marker; the largest was 0.895 for recursive SEMIANA at N=2,500 in run 2. Raw
per-run ranges remain in the generated JSON. These local ratios support the
deliberately low floors but are not hosted-runner performance expectations.

## Measurement method and noise

Each Numba specialization is warmed and its output checked before timing.
Inputs are deterministic; the solver-response shape uses case 1 controls from
`Input/Parameter.csv`, with `Mdat` unchanged. The driver uses five repeated
measurements, alternates reference/candidate order, limits numerical-library
thread pools to one thread, and uses the median as its primary statistic. It
records every sample, min/median/max, `range / median`, parity, measured
speedup, configured floor, and headroom in the JSON artifact. It uses no
fastmath and writes no solver outputs.

If a blocking median misses its floor, the driver makes one fresh five-sample
measurement with the starting order reversed. The guardrail fails only when
that second median still misses. A `range / median` above 1 is reported as a
variability warning rather than hidden. Hosted CPU and dependency-version
variation remains possible; a failing or noisy artifact should be inspected
before interpreting the ratio as a code regression.

Run locally from the repository root:

```powershell
python benchmarks/ci_performance_guardrails.py
python benchmarks/ci_performance_guardrails.py --output benchmarks/Output/ci_performance_guardrails/performance_guardrails.json
```

The current driver targets under 60 seconds of warmed benchmark work and
allows up to 90 seconds for cold JIT and hosted-runner variation. The workflow
also spends time checking out the repository and installing dependencies.
The artifact is named `performance-guardrails-python312`; its `environment`
section records the commit, Python and numerical-library versions, platform,
processor, and CPU count. `threshold_policy` records each implementation pair,
repetition count, floor, and rationale. `metrics` contains raw samples and the
measured median ratios. An exactness or architecture failure exits nonzero and
uploads partial diagnostics when available.

The workflow deliberately does not build a temporary base-revision worktree.
The accepted-backend relative checks are simpler to maintain and do not depend
on a fragile dual-checkout/import setup. A future base-vs-head comparison can
be considered separately if needed.

Passing these microbenchmarks is not a scientific validation substitute. It
does not establish long-term solver trajectory parity, cutoff equivalence, or
validity for other cases and controls; those require the existing numerical
and scientific validation process.
