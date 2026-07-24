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
dx/dt = ER * U * sin(theta)
dy/dt = ER * U * cos(theta)
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

## Bank-erodibility coefficient

The code and public interface use `ER` for the bank-erodibility /
migration-rate coefficient. The historical default is:

```text
ER = 1.0e-8
```

It can be changed from the CLI:

```bash
python -m run_ldsfl --base-dir . --cases 1 --erosion-rate 2e-8
```

The GUI exposes the same value as **Bank erodibility / erosion rate**. Blank,
zero, negative, NaN, and infinite values are rejected.

Because:

```text
migration component proportional to ER
dt proportional to 1 / ER
```

the product used for one adaptive centreline update is normally approximately
independent of `ER`. Doubling `ER` therefore approximately halves `dt` while
leaving displacement per solver iteration nearly unchanged.

This does not make `ER` irrelevant. It changes cumulative computational time,
and it can change the final geometry of runs stopped by a simulated-time
criterion rather than by a step count.

## Relevant CLI options

The timestep stability coefficient and bank-erodibility coefficient can be set independently:

```bash
python -m run_ldsfl --base-dir . --cases 1 --cstab 0.01 --erosion-rate 1e-8
```

Smaller `cstab` values generally produce more conservative geometric updates.
Larger values may become unstable. Changing `erosion_rate` primarily rescales
cumulative computational time because the adaptive timestep compensates for the
migration-speed multiplier.

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
