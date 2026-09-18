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

