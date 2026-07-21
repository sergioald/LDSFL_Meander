# Curvature arclength-scaling diagnostic

This page records a diagnostic-only investigation of centreline curvature scaling in LDSFL-Meander.

## Why this matters

The reduced flow response uses centreline curvature as an input. Curvature is normally interpreted as
the change in tangent angle per unit arclength:

```text
c = dtheta / ds
```

A reviewer note identified that the current solver path computes curvature using an index-gradient
pattern:

```text
c = gradient(theta)
```

This is equivalent to a change in tangent angle per saved point index. It is close to `dtheta/ds` only
when the effective point spacing is approximately one solver length unit. If the spacing is not close to
one, the index-gradient value scales with the spacing.

## What this PR does

This PR does not change solver behaviour. It adds a manufactured-geometry diagnostic so the effect can
be measured before changing scientific results.

Added files:

- `tools/diagnose_curvature_scaling.py`
- `tests/test_curvature_scaling_diagnostic.py`

## How to run the diagnostic

From the repository root:

```bash
python tools/diagnose_curvature_scaling.py
```

Machine-readable output is also available:

```bash
python tools/diagnose_curvature_scaling.py --json
```

## Expected interpretation

The diagnostic compares manufactured constant-curvature circular arcs with different point spacings.
For each case it reports:

- the true absolute curvature, `1 / radius`;
- the median absolute curvature from the current index-gradient pattern;
- the median absolute curvature from an arclength-gradient pattern;
- the ratio of each estimate to the true curvature.

The expected qualitative result is:

```text
index-gradient curvature   ≈ true curvature × mean point spacing
arclength-gradient curvature ≈ true curvature
```

Therefore, if `mean_ds` is close to 1, the two estimates may look similar. If `mean_ds` is far from 1,
the current index-gradient pattern is not scale-invariant.

## Follow-up decision

A future behaviour-changing PR should decide whether to replace the curvature calculation with an
arclength-scaled derivative in both relevant solver paths:

- the first-step curvature path; and
- the post-geometry-update curvature path.

That follow-up should be treated as a scientific/numerical change. It should include local comparison
runs, updated validation expectations, and clear release notes because saved trajectories may change.
