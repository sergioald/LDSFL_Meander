# Timestep and iteration notes

This page explains how the solver timestep is chosen and how to interpret the saved step history.

## Solver step versus physical time

The `step` column in the output files is the solver iteration number. It is not, by itself, a dimensional physical time.

For example, for a generated output identifier `<id_files>`:

```text
Output/<id_files>/files/sinuosity_history_<id_files>.csv
```

The `id_files` name is generated from the selected input case ID and key parameter values.

contains:

| Column | Meaning |
|---|---|
| `step` | Solver iteration index. |
| `sinuo` | Dimensionless sinuosity at that iteration. |

The relationship between model iterations and dimensional physical time depends on the nondimensionalisation and physical scaling used for the study. Do not interpret `step` as seconds, days, years, or any dimensional unit unless the corresponding scaling has been defined.

## Adaptive computational timestep

The centreline update uses an adaptive computational timestep. The migration velocity components are computed from the flow response and centreline tangent angle:

```text
dx/dt = ERT * U * sin(theta)
dy/dt = ERT * U * cos(theta)
```

The code then estimates the maximum absolute migration component:

```text
maxCSI = max(max(abs(dx/dt)), max(abs(dy/dt)))
```

The computational timestep is:

```text
dt = cstab / maxCSI
```

where `cstab` is the timestep stability coefficient.

If the migration speed is zero or numerically tiny, the code uses a guarded fallback:

```text
dt = cstab
```

and emits a runtime warning.

## Practical interpretation

The adaptive rule means that faster migration responses produce smaller computational timesteps, while slower responses allow larger computational timesteps.

This is a numerical stability mechanism. It should not be read as a direct physical time calibration without additional scaling.

## Relevant CLI option

The timestep stability coefficient can be set from the CLI:

```bash
python -m run_ldsfl --base-dir . --cases 1 --cstab 0.01
```

Smaller `cstab` values generally produce more conservative updates. Larger values may run faster but should be tested carefully because they can affect numerical stability.

## Output implication

Because `step` is an iteration counter, comparisons such as:

```text
sinuosity versus step
```

show numerical evolution through solver iterations. For physical-time interpretation, document the chosen dimensional scaling separately.

## Recommended wording for reports

Use wording like:

```text
The simulation was run for N computational iterations using the adaptive timestep rule implemented in `dxdy2`. The plotted horizontal axis is solver iteration step, not dimensional physical time.
```

If dimensional physical time is needed, add the nondimensionalisation and scaling assumptions used to convert computational time to physical units.
