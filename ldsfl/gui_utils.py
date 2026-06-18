from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import csv
import math
import re
from typing import Literal

import pandas as pd

InputMode = Literal["dimensionless", "dimensional"]
MobilityMode = Literal[
    "direct_shields",
    "direct_shear_stress",
    "depth_slope_grain",
    "depth_velocity_friction_grain",
    "discharge_half_width_slope_friction_grain",
]
FrictionMode = Literal["cf", "darcy_f", "chezy_c", "manning_n"]
GeometryMode = Literal["as_is", "scale_by_dimensional_half_width"]
StopMode = Literal["first", "all"]
OutputUnits = Literal["dimensionless", "dimensional"]

GRAVITY = 9.81
SUBMERGED_SPECIFIC_GRAVITY = 1.65


@dataclass
class DimensionlessInputs:
    beta: float
    ds: float
    theta0: float
    flagbed: int
    rpic_0: float
    Mdat: int


@dataclass
class DimensionalInputs:
    half_width: float
    d50: float
    mobility_mode: MobilityMode
    flagbed: int
    rpic_0: float
    Mdat: int
    dref: float | None = None
    theta0: float | None = None
    tau_b: float | None = None
    slope: float | None = None
    velocity: float | None = None
    discharge: float | None = None
    friction_mode: FrictionMode = "cf"
    friction_value: float | None = None

    def _require_positive(self, name: str, value: float | None) -> float:
        if value is None or float(value) <= 0.0:
            raise ValueError(f"{name} must be > 0")
        return float(value)

    def resolved_depth(self) -> float:
        half_width = self._require_positive("half-width B", self.half_width)
        self._require_positive("d50", self.d50)
        _ = half_width
        if self.mobility_mode == "discharge_half_width_slope_friction_grain":
            q = self._require_positive("discharge", self.discharge)
            slope = self._require_positive("slope", self.slope)
            cf = self.resolved_cf(dref_hint=max(self.dref or 1.0, 1.0))
            if self.friction_mode == "manning_n":
                n = self._require_positive("Manning n", self.friction_value)
                return (q * n / (2.0 * half_width * math.sqrt(slope))) ** (3.0 / 5.0)
            return (q / (2.0 * half_width * math.sqrt(GRAVITY * slope / cf))) ** (2.0 / 3.0)
        return self._require_positive("reference depth D_0", self.dref)

    def resolved_cf(self, dref_hint: float | None = None) -> float:
        value = self._require_positive("friction value", self.friction_value)
        dref = float(dref_hint) if dref_hint is not None else None
        if self.friction_mode == "cf":
            cf = value
        elif self.friction_mode == "darcy_f":
            cf = value / 8.0
        elif self.friction_mode == "chezy_c":
            cf = GRAVITY / (value ** 2)
        elif self.friction_mode == "manning_n":
            if dref is None or dref <= 0.0:
                raise ValueError("reference depth D_0 is required to convert Manning n to Cf")
            cf = GRAVITY * (value ** 2) / (dref ** (1.0 / 3.0))
        else:
            raise ValueError(f"Unsupported friction_mode={self.friction_mode}")
        if cf <= 0.0:
            raise ValueError("resolved Cf must be > 0")
        return float(cf)

    def resolved_theta0(self) -> float:
        d50 = self._require_positive("d50", self.d50)
        dref = self.resolved_depth()
        if self.mobility_mode == "direct_shields":
            theta0 = self._require_positive("Shields stress", self.theta0)
        elif self.mobility_mode == "direct_shear_stress":
            tau_b = self._require_positive("bed shear stress", self.tau_b)
            theta0 = tau_b / (SUBMERGED_SPECIFIC_GRAVITY * GRAVITY * d50)
        elif self.mobility_mode == "depth_slope_grain":
            slope = self._require_positive("slope", self.slope)
            theta0 = dref * slope / (SUBMERGED_SPECIFIC_GRAVITY * d50)
        elif self.mobility_mode == "depth_velocity_friction_grain":
            velocity = self._require_positive("velocity", self.velocity)
            cf = self.resolved_cf(dref)
            theta0 = cf * (velocity ** 2) / (SUBMERGED_SPECIFIC_GRAVITY * GRAVITY * d50)
        elif self.mobility_mode == "discharge_half_width_slope_friction_grain":
            slope = self._require_positive("slope", self.slope)
            theta0 = dref * slope / (SUBMERGED_SPECIFIC_GRAVITY * d50)
        else:
            raise ValueError(f"Unsupported mobility_mode={self.mobility_mode}")
        if theta0 <= 0.0:
            raise ValueError("resolved Shields stress must be > 0")
        return float(theta0)

    def resolved_velocity(self) -> float:
        dref = self.resolved_depth()
        half_width = self._require_positive("half-width B", self.half_width)
        if self.mobility_mode == "discharge_half_width_slope_friction_grain":
            q = self._require_positive("discharge", self.discharge)
            return q / (2.0 * half_width * dref)
        if self.velocity is not None and self.velocity > 0.0:
            return float(self.velocity)
        if self.slope is not None and self.slope > 0.0 and self.friction_value is not None and self.friction_value > 0.0:
            cf = self.resolved_cf(dref)
            return math.sqrt(GRAVITY * dref * float(self.slope) / cf)
        return float("nan")

    def derived_values(self) -> dict:
        dref = self.resolved_depth()
        half_width = self._require_positive("half-width B", self.half_width)
        d50 = self._require_positive("d50", self.d50)
        theta0 = self.resolved_theta0()
        out = {
            "D0": float(dref),
            "beta0": float(half_width / dref),
            "ds0": float(d50 / dref),
            "theta0": float(theta0),
        }
        if self.mobility_mode in ("depth_velocity_friction_grain", "discharge_half_width_slope_friction_grain"):
            out["Cf"] = float(self.resolved_cf(dref))
            vel = self.resolved_velocity()
            if not math.isnan(vel):
                out["velocity"] = float(vel)
        return out

    def to_dimensionless(self) -> DimensionlessInputs:
        vals = self.derived_values()
        return DimensionlessInputs(
            beta=float(vals["beta0"]),
            ds=float(vals["ds0"]),
            theta0=float(vals["theta0"]),
            flagbed=int(self.flagbed),
            rpic_0=float(self.rpic_0),
            Mdat=int(self.Mdat),
        )


@dataclass
class GeometrySettings:
    mode: GeometryMode = "as_is"
    custom_scale: float | None = None
    smoothing_enabled: bool = True
    smoothing_factor: float = 8.0
    neck_cutoff_interval: int = 3
    resample_upper_factor: float = 1.03
    resample_lower_factor: float = 0.97

    def resolved_scale(self, config: "GuiCaseConfig") -> float:
        if self.mode == "as_is":
            return 1.0
        if self.mode == "scale_by_dimensional_half_width":
            if config.dimensional is None:
                raise ValueError("Geometry scaling by dimensional half-width requires dimensional input mode")
            half_width = float(config.dimensional.half_width)
            if half_width <= 0.0:
                raise ValueError("Dimensional half-width B_0 must be > 0 to scale geometry")
            return half_width
        raise ValueError(f"Unsupported geometry mode: {self.mode}")


@dataclass
class RunControls:
    case_id: int = 1
    nprint: int = 10000
    ntstep: int = 100000
    max_cut: int = 100
    max_steps: int = 50
    max_sim_time: float = 0.0
    stop_on_steps: bool = True
    stop_on_time: bool = False
    stop_on_cutoffs: bool = True
    stop_on_sinuosity_stability: bool = False
    stop_mode: StopMode = "first"
    do_plots: bool = False
    save_final_overlay: bool = True
    save_run_manifest: bool = True
    output_units: OutputUnits = "dimensionless"
    flow_bc: str = "free"
    backend: str = "numpy"
    flow_paral: int = 0
    flow_workers: int = 0
    numba_parallel: bool = False
    numba_fastmath: bool = False
    cstab: float = 0.01
    sinuo_window: int = 100
    sinuo_rel_tol: float = 5.0e-3


@dataclass
class GuiCaseConfig:
    mode: InputMode
    xy_csv: Path
    workspace_dir: Path
    run: RunControls
    dimensionless: DimensionlessInputs | None = None
    dimensional: DimensionalInputs | None = None
    geometry: GeometrySettings = field(default_factory=GeometrySettings)

    def resolved_dimensionless(self) -> DimensionlessInputs:
        if self.mode == "dimensionless":
            if self.dimensionless is None:
                raise ValueError("dimensionless inputs are missing")
            validate_dimensionless(self.dimensionless)
            return self.dimensionless
        if self.dimensional is None:
            raise ValueError("dimensional inputs are missing")
        dimless = self.dimensional.to_dimensionless()
        validate_dimensionless(dimless)
        return dimless


PARAMETER_COLUMNS = [
    "Id", "Beta", "ds", "Thetha", "flagbed", "r", "Mdat", "flagbed=1 plane; flagbed=2 dunes"
]


GEOMETRY_HEADER_ALIASES = {
    "x", "xcoord", "xcoordinate", "x_coordinate", "xposition", "abscissa",
    "y", "ycoord", "ycoordinate", "y_coordinate", "yposition", "ordinate",
}


def _normalize_header_token(token: str) -> str:
    token = token.strip().lower()
    token = re.sub(r"[^a-z0-9]+", "", token)
    return token


def _looks_like_geometry_header(row: list[str]) -> bool:
    if len(row) < 2:
        return False
    a = _normalize_header_token(row[0])
    b = _normalize_header_token(row[1])
    if not a or not b:
        return False
    if a in {"x"} and b in {"y"}:
        return True
    return ((a.startswith("x") and b.startswith("y")) or (a in GEOMETRY_HEADER_ALIASES and b in GEOMETRY_HEADER_ALIASES and a != b))


def parse_geometry_csv(path: Path) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Geometry file not found: {path}")

    numeric_rows: list[tuple[float, float]] = []
    bad_rows: list[str] = []
    header_skipped = False
    extra_columns_ignored = False
    first_data_row_seen = False

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for lineno, row in enumerate(reader, start=1):
            if not row or all(str(cell).strip() == "" for cell in row):
                continue
            first_cell = str(row[0]).strip()
            if first_cell.startswith("#"):
                continue
            if len(row) < 2:
                bad_rows.append(f"line {lineno}: expected at least 2 columns, found {len(row)}")
                continue

            first_two = [str(row[0]).strip(), str(row[1]).strip()]
            if not first_data_row_seen:
                first_data_row_seen = True
                if _looks_like_geometry_header(first_two):
                    header_skipped = True
                    continue

            try:
                x = float(first_two[0])
                y = float(first_two[1])
            except ValueError:
                preview = ", ".join(repr(v) for v in first_two)
                bad_rows.append(
                    f"line {lineno}: nonnumeric x/y values {preview}. "
                    "Only the first non-empty row may be a text header like x,y."
                )
                continue

            if any(str(cell).strip() != "" for cell in row[2:]):
                extra_columns_ignored = True
            numeric_rows.append((x, y))

    if bad_rows:
        msg = "Geometry CSV is malformed.\n" + "\n".join(bad_rows[:8])
        if len(bad_rows) > 8:
            msg += f"\n... and {len(bad_rows) - 8} more malformed row(s)."
        raise ValueError(msg)
    if len(numeric_rows) < 2:
        raise ValueError(
            f"Geometry CSV must contain at least 2 numeric x,y rows after removing blank/comment/header rows: {path}"
        )

    table = pd.DataFrame(numeric_rows)
    info = {
        "row_count": len(numeric_rows),
        "header_skipped": header_skipped,
        "extra_columns_ignored": extra_columns_ignored,
    }
    return table, info


def validate_dimensionless(inp: DimensionlessInputs) -> None:
    if inp.beta <= 0:
        raise ValueError("Aspect ratio beta_0 = B_0/D_0 must be > 0")
    if inp.ds <= 0:
        raise ValueError("Relative grain size ds_0 = d50/D_0 must be > 0")
    if inp.theta0 <= 0:
        raise ValueError("Shields stress must be > 0")
    if inp.flagbed not in (1, 2):
        raise ValueError("Bed regime must be 1 (plane bed) or 2 (dune bed)")
    if inp.rpic_0 <= 0:
        raise ValueError("Transverse bed-slope coefficient r must be > 0")
    if inp.Mdat <= 0:
        raise ValueError("Number of Fourier modes must be > 0")


def parameter_row_from_dimensionless(case_id: int, inp: DimensionlessInputs) -> dict:
    validate_dimensionless(inp)
    return {
        "Id": int(case_id),
        "Beta": float(inp.beta),
        "ds": float(inp.ds),
        "Thetha": float(inp.theta0),
        "flagbed": int(inp.flagbed),
        "r": float(inp.rpic_0),
        "Mdat": int(inp.Mdat),
        "flagbed=1 plane; flagbed=2 dunes": int(inp.flagbed),
    }


def build_parameter_table(case_id: int, inp: DimensionlessInputs) -> pd.DataFrame:
    return pd.DataFrame([parameter_row_from_dimensionless(case_id, inp)], columns=PARAMETER_COLUMNS)


def build_scaled_xy_table(config: GuiCaseConfig) -> tuple[pd.DataFrame, float]:
    table, _info = parse_geometry_csv(config.xy_csv)
    scale = config.geometry.resolved_scale(config)
    if scale != 1.0:
        table = table.copy()
        table.iloc[:, 0] = table.iloc[:, 0].astype(float) / scale
        table.iloc[:, 1] = table.iloc[:, 1].astype(float) / scale
    return table, float(scale)


def compute_id_files(config: GuiCaseConfig) -> str:
    dimless = config.resolved_dimensionless()
    parts = [
        str(int(config.run.case_id)),
        format(dimless.beta, ".15g"),
        format(dimless.ds, ".15g"),
        format(dimless.theta0, ".15g"),
        str(int(dimless.flagbed)),
        format(dimless.rpic_0, ".15g"),
    ]
    return ("_".join(parts)).replace(".", "")




def resolve_output_units(config: GuiCaseConfig) -> OutputUnits:
    requested = str(config.run.output_units).lower()
    if requested == "dimensional" and config.mode == "dimensional" and config.dimensional is not None:
        return "dimensional"
    return "dimensionless"


def output_scales(config: GuiCaseConfig) -> dict:
    units = resolve_output_units(config)
    length_scale = 1.0
    velocity_scale = 1.0
    if units == "dimensional" and config.dimensional is not None:
        length_scale = float(config.geometry.resolved_scale(config))
        try:
            velocity = float(config.dimensional.resolved_velocity())
            if math.isfinite(velocity) and velocity > 0.0:
                velocity_scale = velocity
        except Exception:
            velocity_scale = 1.0
    return {
        "requested_output_units": str(config.run.output_units),
        "resolved_output_units": units,
        "output_length_scale": float(length_scale),
        "output_velocity_scale": float(velocity_scale),
    }

def preview_case_config(config: GuiCaseConfig) -> dict:
    dimless = config.resolved_dimensionless()
    scales = output_scales(config)
    preview = {
        "base_dir": str(Path(config.workspace_dir)),
        "xy_csv": str(Path(config.xy_csv)),
        "mode": config.mode,
        "case_id": int(config.run.case_id),
        "id_files": compute_id_files(config),
        "beta": float(dimless.beta),
        "ds": float(dimless.ds),
        "theta0": float(dimless.theta0),
        "flagbed": int(dimless.flagbed),
        "rpic_0": float(dimless.rpic_0),
        "Mdat": int(dimless.Mdat),
        "geometry_mode": config.geometry.mode,
        "geometry_scale": float(config.geometry.resolved_scale(config)),
        "geometry_smoothing_enabled": bool(config.geometry.smoothing_enabled),
        "geometry_smoothing_factor": float(config.geometry.smoothing_factor),
        "neck_cutoff_interval": int(config.geometry.neck_cutoff_interval),
        "resample_upper_factor": float(config.geometry.resample_upper_factor),
        "resample_lower_factor": float(config.geometry.resample_lower_factor),
        "cstab": float(config.run.cstab),
        "sinuo_window": int(config.run.sinuo_window),
        "sinuo_rel_tol": float(config.run.sinuo_rel_tol),
        "stop_mode": config.run.stop_mode,
        "stop_on_steps": bool(config.run.stop_on_steps),
        "stop_on_time": bool(config.run.stop_on_time),
        "stop_on_cutoffs": bool(config.run.stop_on_cutoffs),
        "stop_on_sinuosity_stability": bool(config.run.stop_on_sinuosity_stability),
        "max_steps": int(config.run.max_steps),
        "max_sim_time": float(config.run.max_sim_time),
        "max_cut": int(config.run.max_cut),
    }
    preview.update(scales)
    if config.mode == "dimensional" and config.dimensional is not None:
        preview.update(config.dimensional.derived_values())
        preview["mobility_mode"] = config.dimensional.mobility_mode
        preview["friction_mode"] = config.dimensional.friction_mode
    return preview


def validate_case_config(config: GuiCaseConfig) -> list[str]:
    dimless = config.resolved_dimensionless()
    warnings: list[str] = []
    if not Path(config.xy_csv).exists():
        raise FileNotFoundError(f"xy.csv not found: {config.xy_csv}")
    geometry_table, geometry_info = parse_geometry_csv(Path(config.xy_csv))
    if config.run.nprint <= 0:
        raise ValueError("Saved snapshot interval Nprint must be > 0")
    if config.run.ntstep <= 0:
        raise ValueError("Ntstep must be > 0")
    if config.run.max_cut < 0:
        raise ValueError("Maximum cutoffs must be >= 0")
    if config.run.max_steps < 0:
        raise ValueError("Maximum steps must be >= 0")
    if config.run.max_sim_time < 0:
        raise ValueError("Maximum simulated time must be >= 0")
    if config.run.cstab <= 0:
        raise ValueError("cstab must be > 0")
    if config.run.sinuo_window < 2:
        raise ValueError("Sinuosity stability window must be >= 2")
    if config.run.sinuo_rel_tol <= 0.0:
        raise ValueError("Sinuosity relative tolerance must be > 0")
    if not (
        config.run.stop_on_steps
        or config.run.stop_on_time
        or config.run.stop_on_cutoffs
        or config.run.stop_on_sinuosity_stability
    ):
        raise ValueError("At least one stop criterion must be enabled.")
    if config.run.stop_on_steps and config.run.max_steps == 0:
        warnings.append("Step stopping is enabled but max_steps = 0, so the step criterion will never trigger.")
    if config.run.stop_on_time and config.run.max_sim_time == 0:
        warnings.append("Time stopping is enabled but max_sim_time = 0, so the time criterion will never trigger.")
    if config.run.stop_on_cutoffs and config.run.max_cut == 0:
        warnings.append("Cutoff stopping is enabled but max_cut = 0, so the cutoff criterion will never trigger.")
    if str(config.run.output_units).lower() == "dimensional" and config.mode != "dimensional":
        warnings.append("Dimensional outputs were requested, but only dimensional input mode provides enough information to dimensionalize all outputs. The run will fall back to dimensionless outputs.")
    if dimless.beta < 4 or dimless.beta > 80:
        warnings.append("Aspect ratio beta_0 = B_0/D_0 is outside a typical reduced-model range (roughly 4 to 80).")
    if dimless.ds < 1e-4 or dimless.ds > 0.1:
        warnings.append("Relative grain size ds_0 = d50/D_0 is outside a typical reduced-model range (roughly 1e-4 to 0.1).")
    if dimless.theta0 < 0.02 or dimless.theta0 > 1.5:
        warnings.append("Shields stress is outside a typical transport-focused range (roughly 0.02 to 1.5).")
    if dimless.rpic_0 < 0.2 or dimless.rpic_0 > 1.0:
        warnings.append("Transverse bed-slope coefficient r is outside the common Talmon-style range (roughly 0.2 to 1.0).")
    if dimless.Mdat > 32:
        warnings.append("Number of Fourier modes is high; runtime may increase significantly.")
    geometry_scale = config.geometry.resolved_scale(config)
    if geometry_info.get("header_skipped"):
        warnings.append("Geometry file header row detected and will be skipped automatically.")
    if geometry_info.get("extra_columns_ignored"):
        warnings.append("Geometry file has more than two columns; only the first two columns (x, y) will be used.")
    if geometry_scale <= 0.0:
        raise ValueError("Geometry scale must be > 0")
    if config.geometry.mode != "as_is" and geometry_scale == 1.0:
        warnings.append("Geometry scaling is enabled but the effective scale factor is 1.0.")
    if config.geometry.smoothing_factor <= 0.0:
        raise ValueError("Geometry smoothing factor must be > 0")
    if config.geometry.neck_cutoff_interval < 0:
        raise ValueError("Neck cutoff interval must be >= 0")
    if config.geometry.resample_upper_factor <= 1.0:
        raise ValueError("Resample upper factor must be > 1.0")
    if not (0.0 < config.geometry.resample_lower_factor < 1.0):
        raise ValueError("Resample lower factor must be between 0 and 1")
    if len(geometry_table) < 6:
        warnings.append("Geometry has very few points; curvature estimates and smoothing may be unstable.")
    diffs = ((geometry_table.iloc[:, 0].diff() ** 2 + geometry_table.iloc[:, 1].diff() ** 2) ** 0.5).iloc[1:]
    if (diffs <= 0.0).any():
        warnings.append("Geometry contains duplicated consecutive points or zero-length segments.")
    if len(diffs) >= 3 and diffs.mean() > 0:
        if diffs.max() / diffs.mean() > 4.0:
            warnings.append("Geometry point spacing is highly uneven; resampling and curvature estimates may be noisy.")
    if config.mode == "dimensional" and config.dimensional is not None:
        vals = config.dimensional.derived_values()
        if vals["D0"] <= 0 or vals["beta0"] <= 0:
            raise ValueError("Derived dimensional quantities are invalid.")
        if vals["D0"] < 0.1:
            warnings.append("Derived reference depth D_0 is very small; check units and conversion choices.")
    return warnings


def config_to_dict(config: GuiCaseConfig) -> dict:
    return {
        "mode": config.mode,
        "xy_csv": str(config.xy_csv),
        "workspace_dir": str(config.workspace_dir),
        "run": asdict(config.run),
        "dimensionless": asdict(config.dimensionless) if config.dimensionless is not None else None,
        "dimensional": asdict(config.dimensional) if config.dimensional is not None else None,
        "geometry": asdict(config.geometry),
    }


def config_from_dict(data: dict) -> GuiCaseConfig:
    run = RunControls(**data["run"])
    dimless = DimensionlessInputs(**data["dimensionless"]) if data.get("dimensionless") else None

    dimensional_data = dict(data.get("dimensional") or {})
    if dimensional_data:
        # Backward compatibility with pre-release configs that used width/depth and older option names.
        if "half_width" not in dimensional_data and "width" in dimensional_data:
            dimensional_data["half_width"] = dimensional_data.pop("width")
        if "dref" not in dimensional_data and "depth" in dimensional_data:
            dimensional_data["dref"] = dimensional_data.pop("depth")
        if dimensional_data.get("mobility_mode") == "discharge_width_slope_friction_grain":
            dimensional_data["mobility_mode"] = "discharge_half_width_slope_friction_grain"
    dimensional = DimensionalInputs(**dimensional_data) if dimensional_data else None

    geometry_data = dict(data.get("geometry", {}))
    if geometry_data.get("mode") == "scale_by_dimensional_width":
        geometry_data["mode"] = "scale_by_dimensional_half_width"
    if geometry_data.get("mode") == "scale_by_custom_factor":
        geometry_data["mode"] = "as_is"
    geometry = GeometrySettings(**geometry_data)

    return GuiCaseConfig(
        mode=data["mode"],
        xy_csv=Path(data["xy_csv"]),
        workspace_dir=Path(data["workspace_dir"]),
        run=run,
        dimensionless=dimless,
        dimensional=dimensional,
        geometry=geometry,
    )


def write_case_inputs(config: GuiCaseConfig) -> dict:
    config.xy_csv = Path(config.xy_csv)
    config.workspace_dir = Path(config.workspace_dir)
    if not config.xy_csv.exists():
        raise FileNotFoundError(f"xy.csv not found: {config.xy_csv}")

    dimless = config.resolved_dimensionless()
    in_dir = config.workspace_dir / "Input"
    out_dir = config.workspace_dir / "Output"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = build_parameter_table(config.run.case_id, dimless)
    param_csv = in_dir / "Parameter.csv"
    table.to_csv(param_csv, index=False)

    xy_table, geometry_scale = build_scaled_xy_table(config)
    xy_target = in_dir / "xy.csv"
    xy_table.to_csv(xy_target, header=False, index=False)

    summary = preview_case_config(config)
    summary.update({
        "parameter_csv": str(param_csv),
        "copied_xy_csv": str(xy_target),
        "output_dir": str(out_dir),
        "written_geometry_scale": float(geometry_scale),
    })
    return summary
