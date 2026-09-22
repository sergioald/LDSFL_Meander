# LDSFL-Meander User Manual

## 1. What LDSFL-Meander is

**LDSFL-Meander** is named after **Lopez-Dubon, Sgarabotto, Frascati and Lanzoni**.

LDSFL-Meander is a reduced model for meander evolution. The solver works internally with dimensionless parameters, but the GUI can accept either dimensionless or dimensional user inputs. It is intended for reduced-model studies of wide, mildly curved, long bends. Version 0.6.5 adds scrollable GUI tabs plus graceful stop and continuation controls for long runs.

## 2. Main entry points

Use `run_ldsfl.py` when you already have `Input/Parameter.csv` and `Input/xy.csv` prepared.

`--cases` selects values from the CSV's `Id` column, not row numbers. IDs must
be unique positive integers. Omitting `--cases` runs every ID in table order;
an unknown ID is rejected before the batch starts.

Use `gui_ldsfl.py` when you want help building input files, checking conversions, scaling and filtering geometry, selecting stop criteria, and viewing the final planform.

All entry points share the same scientific validation. `Parameter.csv` must
contain `Id`, `Beta`, `ds`, `Thetha`, `flagbed`, `r`, and `Mdat`. Numeric values
must be finite; positive quantities must be greater than zero; `Id`, `flagbed`,
and `Mdat` must be exact integers. `flagbed` must be 1 or 2, IDs must be unique,
and fractional values such as `Mdat=6.5` are rejected rather than truncated.

## 3. Core notation

In this project:

- `B_0` = reference channel **half-width**
- full reference width = `2B_0`
- `D_0` = reference depth used in dimensional conversion
- `Beta = B_0 / D_0`
- `ds = d50 / D_0`
- `Thetha` = reference Shields stress `theta_0` (historical CSV spelling retained in the code)

Direct bed shear stress is supplied in pascals. Its conversion uses freshwater
density `rho_w = 1000 kg/m^3` and submerged specific gravity `1.65`:
`theta_0 = tau_b / (rho_w * 1.65 * g * d50)`.

## 4. Geometry preprocessing

The GUI can:

- validate geometry files
- skip a first header row such as `x,y`
- ignore blank and comment lines
- keep geometry as provided or rescale it by `B_0`
- apply smoothing/filtering and resampling controls

## 5. Advanced controls

The GUI exposes advanced controls for:

- backend selection (`numpy` or `numba`)
- boundary condition (`free` or `periodic`)
- `cstab`
- bank erodibility / erosion rate
- geometry smoothing
- resample upper/lower factors
- neck cutoff interval
- flexible stop criteria
- output units (`dimensionless` or `dimensional` when dimensional inputs are available)

A zero or absent stopping limit disables that criterion, including in `all`
mode. At least one criterion must have a positive limit, or sinuosity stability
stopping must be enabled. Negative and non-finite stopping limits are rejected
by the solver.

These rules also apply to direct `run_case()` calls. Save intervals, iteration
labels, worker counts, smoothing and resampling settings, stability settings,
flow options, and output units are checked before a run creates its output
directory. For example, `--nprint 0` fails immediately with a validation error.

## 6. Bank erodibility and resonance

The advanced GUI includes **Bank erodibility / erosion rate**. The historical
default is `1.0e-8`, and the value must be finite and greater than zero.

Because the solver adapts the timestep inversely to migration speed, changing
this coefficient primarily changes cumulative simulated time. Geometry versus
solver iteration is normally almost unchanged. Record the selected value when
using `max_sim_time` or comparing a simulation with a physical timescale.

The run summary also reports a resonance diagnostic containing the current
state, the estimated resonant aspect ratio, relative distance to resonance, and
the fundamental decay rate. Treat this as a reduced-model interpretation aid,
not as independent physical validation.

## 7. Outputs

LDSFL-Meander writes case outputs under `Output/<id_files>/`, including:

- `xyu/`
- `xy_cut/`
- `plot/`
- `files/` (saved run variables)
- `run_config.json` (solver options, initial parameters, and input SHA-256 hashes)
- `run_status.json` (simulation completion, output completeness, stop reason, and output errors)
- `run_manifest.json`
- `gui_final_overlay.png`

Every run reserves an unused output folder. The first uses the historical
parameter-based name; later runs with that name use `_run2`, `_run3`, etc.
Existing outputs are preserved, including when different parameters produce
the same historical label. Use the returned `id_files` to locate a run rather
than reconstructing its name. `run_config.json` is always written; the GUI's
manifest and overlay remain optional.

Dimensional length outputs use the physical half-width `B_0` independently of
how the input geometry was prepared. Already dimensionless coordinates and
continuation geometry therefore still produce metres when requested.

Dimensional output is accepted only when both `B_0` and a finite positive
physical reference velocity `U0` are known. GUI input modes that cannot derive
`U0` must be supplemented with velocity/friction information or use
dimensionless output. CLI and direct API users must explicitly provide
`output_length_scale` and `output_velocity_scale`; no scale of 1 is silently
treated as physical.

With plotting disabled, including through `--no-plots`, no PNG is created. The
sinuosity CSV is still required and is appended in blocks without rewriting
the complete history. Its `step` values run from 0 through the final completed
iteration exactly once. The optional sinuosity PNG is generated only at final
completion when plots are enabled.

The `var_*.csv` files use two counters:

- `state_step` is the number of completed migration iterations represented by the row;
- `jt` is the retained legacy/internal loop counter and equals `state_step + 1`.

The initial row has `state_step=0`, `jt=1`, `dt=0`, and `dt_cum=0`. The `dt`
field is the adaptive timestep used to reach the represented state, so a run
that completes `N` iterations records states 0 through `N`, including the final
post-step geometry and cumulative time.

The final `xyu` snapshot, variable-history flush, and sinuosity CSV are required
scientific outputs. If any cannot be written, the solver attempts the other
final outputs, records the errors in `run_status.json`, and raises `RuntimeError`
instead of returning a successful result. Plot failures remain non-fatal.

The optional Numba backend now follows the NumPy reference for both
sub-resonant and super-resonant `SL=1` flow evaluation, and CI checks this
equivalence when installing the `numba` extra.

## 8. Full manual

See `docs/LDSFL_Meander_user_manual.pdf` for the full LaTeX manual.

## Stop and continue from the GUI

The **Run & diagnostics** tab now has a vertical scrollbar, so the controls, sinuosity panel, and diagnostic text remain accessible on smaller screens.

Use **Stop after current step** to request a graceful user stop. The solver will finish the current safe iteration boundary, write the final geometry snapshot, and save the current sinuosity history.

Use **Continue from latest output** after a completed or manually stopped run to launch another segment from the latest saved geometry. This is useful when a run stops because `max_steps`, `max_cutoffs`, or another stop criterion was reached before the sinuosity became stable or quasi-stable.

The continuation button writes `Input/xy_continue_from_latest.csv` from the latest `xyu` snapshot and uses it as the next initial centerline. If outputs were saved in dimensional units, the GUI converts the coordinates back to solver units before continuing.

Each continuation segment gets a separate output folder and retains the
physical output scale. It starts a new segment with step/time counters reset;
it is a geometry restart, not a full solver-state checkpoint.

