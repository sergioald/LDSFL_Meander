# LDSFL-Meander User Manual

## 1. What LDSFL-Meander is

**LDSFL-Meander** is named after **Lopez-Dubon, Sgarabotto, Frascati and Lanzoni**.

LDSFL-Meander is a reduced model for meander evolution. The solver works internally with dimensionless parameters, but the GUI can accept either dimensionless or dimensional user inputs. It is intended for reduced-model studies of wide, mildly curved, long bends. Version 0.6.5 adds scrollable GUI tabs plus graceful stop and continuation controls for long runs.

## 2. Main entry points

Use `run_ldsfl.py` when you already have `Input/Parameter.csv` and `Input/xy.csv` prepared.

Use `gui_ldsfl.py` when you want help building input files, checking conversions, scaling and filtering geometry, selecting stop criteria, and viewing the final planform.

## 3. Core notation

In this project:

- `B_0` = reference channel **half-width**
- full reference width = `2B_0`
- `D_0` = reference depth used in dimensional conversion
- `Beta = B_0 / D_0`
- `ds = d50 / D_0`
- `Thetha` = reference Shields stress `theta_0` (historical CSV spelling retained in the code)

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
- geometry smoothing
- resample upper/lower factors
- neck cutoff interval
- flexible stop criteria
- output units (`dimensionless` or `dimensional` when dimensional inputs are available)

## 6. Outputs

LDSFL-Meander writes case outputs under `Output/<id_files>/`, including:

- `xyu/`
- `xy_cut/`
- `plot/`
- `files/` (saved run variables)
- `run_manifest.json`
- `gui_final_overlay.png`

## 7. Full manual

See `docs/LDSFL_Meander_user_manual.pdf` for the full LaTeX manual.

## Stop and continue from the GUI

The **Run & diagnostics** tab now has a vertical scrollbar, so the controls, sinuosity panel, and diagnostic text remain accessible on smaller screens.

Use **Stop after current step** to request a graceful user stop. The solver will finish the current safe iteration boundary, write the final geometry snapshot, and save the current sinuosity history.

Use **Continue from latest output** after a completed or manually stopped run to launch another segment from the latest saved geometry. This is useful when a run stops because `max_steps`, `max_cutoffs`, or another stop criterion was reached before the sinuosity became stable or quasi-stable.

The continuation button writes `Input/xy_continue_from_latest.csv` from the latest `xyu` snapshot and uses it as the next initial centerline. If outputs were saved in dimensional units, the GUI converts the coordinates back to solver units before continuing.

