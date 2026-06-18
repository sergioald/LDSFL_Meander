# Stop on sinuosity stability control

This control stops a run automatically when the post-transient equivalence diagnostic accepts sinuosity stability within the configured drift tolerance.

## Recommended behavior

The stop condition should use the equivalence diagnostic, not only the simple moving-window label.

```text
stop when sinuosity_stability["equivalence"]["stable"] is True
```

The default should remain disabled, because the interpretation depends on the chosen transient cutoff and drift tolerance.

## Suggested GUI label

```text
Enable stop when sinuosity is statistically stable
```

## Suggested stop reason

```text
stop criteria reached: sinuosity_stability
```

## Notes

This is intended for long exploratory runs where the user wants the solver to stop once residual post-transient drift is practically negligible. It should not be treated as proof of mathematical equilibrium.
