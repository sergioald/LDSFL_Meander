# LDSFL-Meander User Manual

## 1. What LDSFL-Meander is

**LDSFL-Meander** is named after **Lopez-Dubon, Sgarabotto, Frascati and Lanzoni**.

LDSFL-Meander is a reduced model for meander evolution. The solver works internally with dimensionless parameters, but the GUI can accept either dimensionless or dimensional user inputs. It is intended for reduced-model studies of wide, mildly curved, long bends.

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
- `files/` (saved run variables and sinuosity history)
- `run_manifest.json`
- `gui_final_overlay.png`
- `sinuosity_history_<id_files>.csv` and `sinuosity_history_<id_files>.png` for step-vs-sinuosity stability checks

## 7. Full manual

See `docs/LDSFL_Meander_user_manual.pdf` for the full LaTeX manual.


## 8. Sinuosity stability

The GUI includes a sinuosity stability panel. It plots step number versus sinuosity and reports whether the recent evolution is not stable, quasi-stable, or stable. The default stability window is 100 stored values. The assessment uses both the relative span of sinuosity over the window and the fitted relative trend per step, so it is more robust than checking only the last two values.
