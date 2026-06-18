# Sinuosity equivalence stability diagnostic

LDSFL-Meander already reports a simple moving-window sinuosity stability state using relative span and relative trend metrics. The equivalence diagnostic adds a stricter statistical interpretation for long runs.

## Why use equivalence?

A large p-value for a linear trend does not prove that the trend is zero. It only says that the analysis failed to detect a statistically significant trend.

For practical model interpretation, it is better to define how much total drift is negligible and then test whether the fitted drift is smaller than that tolerance.

## Method

After discarding an initial transient, fit a linear trend to the remaining sinuosity history:

```text
S(t) = a + b t + error
```

The diagnostic estimates the total fitted drift over the analysis window:

```text
total drift = b * (final_step - first_analysis_step)
```

Stability is accepted only when the confidence interval for that total drift lies fully inside the predefined tolerance interval:

```text
-drift_tolerance < drift confidence interval < +drift_tolerance
```

The default settings are:

```text
transient_step = 40,000
drift_tolerance = 0.02 sinuosity units
confidence = 0.90
```

The uncertainty calculation uses a lightweight Newey-West/HAC covariance estimate so that the slope uncertainty is less sensitive to autocorrelation in the simulation time series.

## Interpretation

Use this diagnostic as evidence of practical stability, not as proof of exact mathematical equilibrium.

A suitable reporting statement is:

```text
After discarding the initial transient, sinuosity drift was tested using an equivalence criterion. Stability was accepted only when the confidence interval for the fitted total drift lay within the chosen tolerance.
```

## Choosing the tolerance

The tolerance should be chosen for the physical problem. A tolerance of ±0.02 sinuosity units is a reasonable default for exploratory runs, but calibrated studies should justify the chosen tolerance.
