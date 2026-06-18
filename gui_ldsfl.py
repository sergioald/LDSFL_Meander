#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

from ldsfl.gui_utils import (
    DimensionlessInputs,
    DimensionalInputs,
    GuiCaseConfig,
    GeometrySettings,
    RunControls,
    build_scaled_xy_table,
    parse_geometry_csv,
    compute_id_files,
    config_from_dict,
    config_to_dict,
    output_scales,
    preview_case_config,
    validate_case_config,
    write_case_inputs,
)
from ldsfl.main import run_project
from ldsfl import __version__

BED_OPTIONS = {
    '1 - plane bed': 1,
    '2 - dune bed': 2,
}
MOBILITY_OPTIONS = {
    'Direct Shields stress': 'direct_shields',
    'Direct bed shear stress + d50': 'direct_shear_stress',
    'Reference depth D_0 + slope + d50': 'depth_slope_grain',
    'Reference depth D_0 + velocity + friction + d50': 'depth_velocity_friction_grain',
    'Discharge + half-width B + slope + friction + d50': 'discharge_half_width_slope_friction_grain',
}
FRICTION_OPTIONS = {
    'Cf': 'cf',
    'Darcy-Weisbach f': 'darcy_f',
    'Chezy C': 'chezy_c',
    'Manning n': 'manning_n',
}
FRICTION_LABELS = {
    'cf': 'Cf [-]',
    'darcy_f': 'Darcy-Weisbach f [-]',
    'chezy_c': 'Chezy C [m^0.5/s]',
    'manning_n': 'Manning n [s/m^(1/3)]',
}
GEOMETRY_OPTIONS = {
    'Use geometry as provided': 'as_is',
    'Scale x and y by half-width B_0': 'scale_by_dimensional_half_width',
}
HELP = {
    'beta': 'Reference aspect ratio beta_0 = B_0/D_0, where B_0 is the half-width and D_0 is the reference depth. Larger values mean wider, shallower channels.',
    'ds': 'Reference relative grain size ds_0 = d50/D_0. This is the grain size scaled by the reference depth D_0.',
    'theta0': 'Reference Shields stress theta_0. Dimensionless bed shear stress controlling sediment mobility.',
    'flagbed': 'Bed regime / resistance package. Plane bed is simpler; dune bed includes bedform effects.',
    'rpic': 'Transverse bed-slope coefficient r (Talmon-style). Keep the default unless you are calibrating it.',
    'mdat': 'Number of retained Fourier modes. Higher values capture finer structure but cost more runtime.',
    'half_width': 'Channel half-width B in metres. The full channel width is 2B.',
    'dref': 'Reference flow depth D_0 in metres.',
    'd50': 'Median sediment grain size d50 in meters.',
    'mobility': 'Choose how the GUI derives Shields stress in dimensional mode.',
    'tau_b': 'Bed shear stress tau_b in Pa (N m^-2).',
    'slope': 'Water-surface or bed slope S (dimensionless).',
    'velocity': 'Mean flow velocity U in m s^-1.',
    'discharge': 'Discharge Q in m^3 s^-1.',
    'friction': 'Choose the friction representation used to derive Cf in dimensional mode.',
    'geometry_mode': 'Select whether the imported geometry is already in solver units or should be scaled before writing Input/xy.csv.',
    'geometry_scale': 'Unused in the current GUI. Geometry can now be kept as provided or scaled by B_0 only.',
    'case_id': 'Case identifier written into Parameter.csv. A case is one row in the parameter table, i.e. one simulation setup with a single set of input parameters.',
    'nprint': 'Saved snapshot interval in steps. The live preview uses this cadence unless a smaller live-preview interval is set.',
    'ntstep': 'Large iteration label used in output naming.',
    'max_cut': 'Maximum number of cutoffs before stopping.',
    'max_steps': 'Maximum number of steps to run. Set 0 for no explicit step cap.',
    'live_preview': 'If enabled, the plot tab refreshes from saved xyu snapshots while the run is in progress.',
    'live_every': 'Requested live-preview cadence in steps. The GUI uses the smaller of this value and Nprint for saved snapshots.',
    'show_original_plot': 'If enabled, the plot view overlays the original imported geometry or initial topography/planform together with live or final results.',
    'flow_bc': 'Flow boundary condition. Use free for open-boundary runs and periodic for repeating-domain tests.',
    'backend': 'Numerical backend for the free-boundary flow solver. NumPy is the reference path; Numba uses JIT acceleration when available.',
    'flow_paral': 'Parallel-flow flag used by the solver for mode computations. Leave at 0 unless you are benchmarking or testing.',
    'flow_workers': 'Number of worker processes/threads for the parallel-flow option. Use 0 for automatic/default behavior.',
    'numba_parallel': 'Enable Numba parallel kernels where supported. This can speed up runs but may oversubscribe CPU cores when combined with flow_paral=1.',
    'numba_fastmath': 'Allow Numba fast-math optimizations. This may improve speed slightly at the cost of stricter IEEE math behavior.',
    'cstab': 'Stability coefficient controlling timestep size. Smaller values are safer but slower; larger values are faster but may become unstable.',
    'sinuo_window': 'Number of stored sinuosity values used for stable/quasi-stable assessment. A default of 100 is conservative for long runs.',
    'sinuo_rel_tol': 'Relative tolerance used for sinuosity stability. Default 0.005 means about 0.5 percent over the stability window.',
    'max_sim_time': 'Maximum cumulative simulated time before stopping. Set 0 to disable this criterion.',
    'stop_mode': 'Choose how multiple enabled stop criteria are combined. First = stop when any enabled criterion triggers. All = stop only when all enabled criteria have triggered.',
    'stop_on_steps': 'Enable or disable the maximum-steps stop criterion.',
    'stop_on_time': 'Enable or disable the maximum simulated time stop criterion.',
    'stop_on_cutoffs': 'Enable or disable the maximum-cutoffs stop criterion.',
    'stop_on_sinuosity_stability': 'Enable automatic stopping once the post-transient equivalence diagnostic accepts sinuosity stability within the drift tolerance.',
    'save_final_overlay': 'Save the current GUI overlay plot as a PNG in the run output folder after the simulation ends.',
    'save_run_manifest': 'Write a run_manifest.json file into the run output folder containing the resolved inputs, run summary, and output-unit settings.',
    'show_completion_popup': 'Show a popup message when the simulation finishes, stops by criteria, or fails.',
    'output_units': 'Choose whether saved solver outputs and exported plots are written in dimensionless or dimensional form. Dimensional output conversion requires dimensional input mode.',
    'geometry_smoothing_enabled': 'Turn the geometry smoothing/filtering stage on or off.',
    'geometry_smoothing_factor': 'Characteristic smoothing wavelength factor used by the geometry filter. Larger values keep more short-scale structure.',
    'neck_cutoff_interval': 'Check for neck cutoffs every N iterations. Set 0 to disable neck-cutoff checks.',
    'resample_upper_factor': 'Upper resampling trigger factor. Values slightly above 1 trigger resampling when average spacing grows too large.',
    'resample_lower_factor': 'Lower resampling trigger factor. Values slightly below 1 trigger resampling when average spacing becomes too small.',
}


class ToolTip:
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind('<Enter>', self.show)
        widget.bind('<Leave>', self.hide)

    def show(self, _event=None):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 2
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f'+{x}+{y}')
        label = tk.Label(
            tw,
            text=self.text,
            justify='left',
            background='#ffffe0',
            relief='solid',
            borderwidth=1,
            wraplength=320,
            padx=6,
            pady=4,
        )
        label.pack()

    def hide(self, _event=None):
        if self.tipwindow is not None:
            self.tipwindow.destroy()
            self.tipwindow = None


class LdslGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title('LDSFL-Meander — Lopez-Dubon, Sgarabotto, Frascati and Lanzoni')
        self.geometry('1280x980')

        repo_root = Path(__file__).resolve().parent
        self.default_xy = repo_root / 'Input' / 'xy.csv'
        self.default_workspace = repo_root
        self.plot_canvas = None
        self.figure = Figure(figsize=(7.2, 5.4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.run_thread = None
        self.run_in_progress = False
        self.stop_requested_event = threading.Event()
        self.monitor_job = None
        self.current_id_files = None
        self.current_xyu_dir = None
        self.last_snapshot_path = None
        self.last_sinuosity_history_mtime = None
        self.latest_result = None
        self.latest_config = None
        self.initial_xy = None
        self.final_xy = None
        self.run_button = None
        self.stop_button = None
        self.continue_button = None

        self.mode_var = tk.StringVar(value='dimensionless')
        self.advanced_var = tk.BooleanVar(value=False)
        self.xy_var = tk.StringVar(value=str(self.default_xy))
        self.workspace_var = tk.StringVar(value=str(self.default_workspace))
        self.status_var = tk.StringVar(value='Ready.')
        self.geometry_mode_label_var = tk.StringVar(value='Use geometry as provided')
        self.geometry_smoothing_enabled_var = tk.BooleanVar(value=True)
        self.geometry_smoothing_factor_var = tk.StringVar(value='8.0')
        self.neck_cutoff_interval_var = tk.StringVar(value='3')
        self.resample_upper_var = tk.StringVar(value='1.03')
        self.resample_lower_var = tk.StringVar(value='0.97')

        self.case_id_var = tk.StringVar(value='1')
        self.nprint_var = tk.StringVar(value='10000')
        self.ntstep_var = tk.StringVar(value='100000')
        self.max_cut_var = tk.StringVar(value='100')
        self.max_steps_var = tk.StringVar(value='50')
        self.max_time_var = tk.StringVar(value='0')
        self.stop_mode_var = tk.StringVar(value='first')
        self.stop_on_steps_var = tk.BooleanVar(value=True)
        self.stop_on_time_var = tk.BooleanVar(value=False)
        self.stop_on_cutoffs_var = tk.BooleanVar(value=True)
        self.stop_on_sinuosity_stability_var = tk.BooleanVar(value=False)
        self.live_preview_var = tk.BooleanVar(value=True)
        self.live_every_var = tk.StringVar(value='5')
        self.do_plots_var = tk.BooleanVar(value=False)
        self.save_final_overlay_var = tk.BooleanVar(value=True)
        self.save_run_manifest_var = tk.BooleanVar(value=True)
        self.show_completion_popup_var = tk.BooleanVar(value=True)
        self.output_units_var = tk.StringVar(value='dimensionless')
        self.show_original_plot_var = tk.BooleanVar(value=True)
        self.flow_bc_var = tk.StringVar(value='free')
        self.backend_var = tk.StringVar(value='numpy')
        self.flow_paral_var = tk.StringVar(value='0')
        self.flow_workers_var = tk.StringVar(value='0')
        self.numba_parallel_var = tk.BooleanVar(value=False)
        self.numba_fastmath_var = tk.BooleanVar(value=False)
        self.cstab_var = tk.StringVar(value='0.01')
        self.sinuo_window_var = tk.StringVar(value='100')
        self.sinuo_rel_tol_var = tk.StringVar(value='0.005')
        self.sinuo_state_var = tk.StringVar(value='Not available')
        self.sinuo_current_var = tk.StringVar(value='—')
        self.sinuo_window_used_var = tk.StringVar(value='—')
        self.sinuo_rel_span_var = tk.StringVar(value='—')
        self.sinuo_rel_trend_var = tk.StringVar(value='—')
        self.sinuosity_figure = Figure(figsize=(6.6, 2.6), dpi=100)
        self.sinuosity_ax = self.sinuosity_figure.add_subplot(111)
        self.sinuosity_canvas = None

        self.beta_var = tk.StringVar(value='9.0')
        self.ds_var = tk.StringVar(value='0.005')
        self.theta0_var = tk.StringVar(value='0.3')
        self.flagbed_label_var = tk.StringVar(value='2 - dune bed')
        self.rpic_var = tk.StringVar(value='0.5')
        self.mdat_var = tk.StringVar(value='6')

        self.half_width_var = tk.StringVar(value='90.0')
        self.dref_var = tk.StringVar(value='10.0')
        self.d50_var = tk.StringVar(value='0.05')
        self.taub_var = tk.StringVar(value='20.0')
        self.slope_var = tk.StringVar(value='0.0002')
        self.velocity_var = tk.StringVar(value='1.2')
        self.discharge_var = tk.StringVar(value='1080.0')
        self.friction_label_var = tk.StringVar(value='Cf')
        self.friction_value_var = tk.StringVar(value='0.01')
        self.mobility_label_var = tk.StringVar(value='Reference depth D_0 + slope + d50')

        self.preview_dref_var = tk.StringVar(value='—')
        self.preview_beta_var = tk.StringVar(value='—')
        self.preview_ds_var = tk.StringVar(value='—')
        self.preview_theta_var = tk.StringVar(value='—')
        self.preview_cf_var = tk.StringVar(value='—')
        self.preview_vel_var = tk.StringVar(value='—')

        self._load_bundled_example_defaults()

        self._build_ui()
        self._bind_preview_updates()
        self._sync_mode()
        self._sync_dimensional_fields()
        self._sync_geometry_fields()
        self._sync_advanced()
        self._update_dimensional_preview()
        self._refresh_initial_plot()
        self._set_run_button_state(running=False)

    def _load_bundled_example_defaults(self) -> None:
        """Load a bundled example case so the GUI is runnable on first launch."""
        try:
            self.workspace_var.set(str(self.default_workspace))
            if self.default_xy.exists():
                self.xy_var.set(str(self.default_xy))
            param_path = self.default_workspace / 'Input' / 'Parameter.csv'
            if not param_path.exists():
                return
            df = pd.read_csv(param_path, encoding='utf-8-sig')
            if df.empty:
                return
            row = df.iloc[0]
            def _get_num(name_candidates, default):
                for name in name_candidates:
                    if name in row.index and pd.notna(row[name]):
                        return row[name]
                return default
            case_id = int(_get_num(['Id', 'id', 'ID'], self.case_id_var.get()))
            beta = float(_get_num(['Beta', 'beta'], self.beta_var.get()))
            ds = float(_get_num(['ds', 'Ds'], self.ds_var.get()))
            theta = float(_get_num(['Thetha', 'Theta', 'theta0', 'theta'], self.theta0_var.get()))
            flagbed = int(_get_num(['flagbed', 'Flagbed'], BED_OPTIONS.get(self.flagbed_label_var.get(), 2)))
            rpic = float(_get_num(['r', 'rpic_0', 'rpic'], self.rpic_var.get()))
            mdat = int(_get_num(['Mdat', 'mdat'], self.mdat_var.get()))
            self.case_id_var.set(str(case_id))
            self.beta_var.set(str(beta))
            self.ds_var.set(str(ds))
            self.theta0_var.set(str(theta))
            self.flagbed_label_var.set(next((label for label, value in BED_OPTIONS.items() if value == flagbed), self.flagbed_label_var.get()))
            self.rpic_var.set(str(rpic))
            self.mdat_var.set(str(mdat))
            # Keep a consistent bundled dimensional example matching the default reduced parameters.
            self.half_width_var.set('90.0')
            self.dref_var.set('10.0')
            self.d50_var.set('0.05')
            self.mobility_label_var.set('Reference depth D_0 + slope + d50')
            self.slope_var.set('0.0002')
            self.friction_label_var.set('Cf')
            self.friction_value_var.set('0.01')
            self.status_var.set('Loaded bundled example values. Choose Run to execute a short example case.')
        except Exception:
            # Leave the hard-coded defaults in place if the bundled example cannot be read.
            pass

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill='both', expand=True)

        header = ttk.Frame(outer)
        header.pack(fill='x', pady=(0, 10))
        self._build_logo(header)
        text_box = ttk.Frame(header)
        text_box.pack(side='left', fill='x', expand=True, padx=(12, 0))
        ttk.Label(text_box, text='LDSFL-Meander', font=('TkDefaultFont', 18, 'bold')).pack(anchor='w')
        ttk.Label(text_box, text='Lopez-Dubon, Sgarabotto, Frascati and Lanzoni', font=('TkDefaultFont', 11, 'italic')).pack(anchor='w')
        ttk.Label(
            text_box,
            text=(
                'LDSFL-Meander is a reduced meander model. The solver stays dimensionless internally. '
                'Dimensional mode converts user inputs to beta_0 = B_0/D_0, ds_0 = d50/D_0, and theta_0 before writing Input/Parameter.csv. '
                'The GUI can also rescale physical geometry before writing Input/xy.csv. In this GUI, B denotes channel half-width, so the full width is 2B.'
            ),
            wraplength=820,
            justify='left',
        ).pack(anchor='w')
        ttk.Button(header, text='About…', command=self._show_about).pack(side='right')

        io_bar = ttk.Frame(outer)
        io_bar.pack(fill='x', pady=(0, 8))
        io_bar.columnconfigure(1, weight=1)
        ttk.Label(io_bar, text='Geometry xy.csv').grid(row=0, column=0, sticky='w', padx=(0, 8), pady=2)
        xy_entry = ttk.Entry(io_bar, textvariable=self.xy_var)
        xy_entry.grid(row=0, column=1, sticky='ew', pady=2)
        ttk.Button(io_bar, text='Browse…', command=self._browse_xy).grid(row=0, column=2, padx=(8, 0), pady=2)
        ttk.Label(io_bar, text='Workspace directory').grid(row=1, column=0, sticky='w', padx=(0, 8), pady=2)
        work_entry = ttk.Entry(io_bar, textvariable=self.workspace_var)
        work_entry.grid(row=1, column=1, sticky='ew', pady=2)
        ttk.Button(io_bar, text='Browse…', command=self._browse_workspace).grid(row=1, column=2, padx=(8, 0), pady=2)
        ToolTip(xy_entry, 'Centerline geometry CSV used by the solver.')
        ToolTip(work_entry, 'Workspace where Input/ and Output/ are written.')

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill='both', expand=True)

        # Inputs and diagnostics can be taller than a laptop screen.
        # These two tabs therefore use a real vertical scrollbar.
        self.inputs_tab_shell = ttk.Frame(self.notebook)
        self.run_tab_shell = ttk.Frame(self.notebook)
        self.inputs_tab = self._make_scrollable_tab(self.inputs_tab_shell)
        self.run_tab = self._make_scrollable_tab(self.run_tab_shell)
        self.plot_tab = ttk.Frame(self.notebook, padding=10)
        self.log_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.inputs_tab_shell, text='Inputs')
        self.notebook.add(self.run_tab_shell, text='Run & diagnostics')
        self.notebook.add(self.plot_tab, text='Plot view')
        self.notebook.add(self.log_tab, text='Log')

        self._build_inputs_tab()
        self._build_run_tab()
        self._build_plot_tab()
        self._build_log_tab()

        status = ttk.Label(outer, textvariable=self.status_var)
        status.pack(fill='x', pady=(8, 0))

    def _make_scrollable_tab(self, parent) -> ttk.Frame:
        """Create a vertically scrollable notebook tab and return its inner frame."""
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        inner = ttk.Frame(canvas, padding=10)
        window_id = canvas.create_window((0, 0), window=inner, anchor='nw')

        def _update_scroll_region(_event=None):
            canvas.configure(scrollregion=canvas.bbox('all'))

        def _resize_inner(event):
            canvas.itemconfigure(window_id, width=event.width)

        def _on_mousewheel(event):
            if getattr(event, 'num', None) == 4:
                canvas.yview_scroll(-3, 'units')
            elif getattr(event, 'num', None) == 5:
                canvas.yview_scroll(3, 'units')
            else:
                delta = int(-1 * (event.delta / 120)) if event.delta else 0
                canvas.yview_scroll(delta, 'units')

        def _bind_mousewheel(_event=None):
            canvas.bind_all('<MouseWheel>', _on_mousewheel)
            canvas.bind_all('<Button-4>', _on_mousewheel)
            canvas.bind_all('<Button-5>', _on_mousewheel)

        def _unbind_mousewheel(_event=None):
            canvas.unbind_all('<MouseWheel>')
            canvas.unbind_all('<Button-4>')
            canvas.unbind_all('<Button-5>')

        inner.bind('<Configure>', _update_scroll_region)
        canvas.bind('<Configure>', _resize_inner)
        canvas.bind('<Enter>', _bind_mousewheel)
        canvas.bind('<Leave>', _unbind_mousewheel)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        return inner

    def _build_inputs_tab(self) -> None:
        mode_frame = ttk.LabelFrame(self.inputs_tab, text='Input style', padding=10)
        mode_frame.pack(fill='x', pady=(0, 8))
        ttk.Radiobutton(mode_frame, text='Dimensionless input', value='dimensionless', variable=self.mode_var, command=self._sync_mode).pack(anchor='w')
        ttk.Radiobutton(mode_frame, text='Dimensional input', value='dimensional', variable=self.mode_var, command=self._sync_mode).pack(anchor='w')
        ttk.Checkbutton(mode_frame, text='Advanced mode', variable=self.advanced_var, command=self._sync_advanced).pack(anchor='w', pady=(6, 0))

        geometry_frame = ttk.LabelFrame(self.inputs_tab, text='Geometry scaling and filtering', padding=10)
        geometry_frame.pack(fill='x', pady=(0, 8))
        self._combobox_row(geometry_frame, 0, 'Geometry mode', self.geometry_mode_label_var, list(GEOMETRY_OPTIONS.keys()), callback=self._sync_geometry_fields, help_key='geometry_mode')
        ttk.Label(geometry_frame, text='Scaled geometry is written to Input/xy.csv before the solver starts.', wraplength=840, justify='left').grid(row=1, column=0, columnspan=2, sticky='w', pady=(4, 0))
        self.geometry_advanced = ttk.Frame(geometry_frame)
        self.geometry_advanced.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(6, 0))
        smooth_btn = ttk.Checkbutton(self.geometry_advanced, text='Enable geometry smoothing/filtering', variable=self.geometry_smoothing_enabled_var)
        smooth_btn.grid(row=0, column=0, columnspan=2, sticky='w')
        ToolTip(smooth_btn, HELP['geometry_smoothing_enabled'])
        self._entry_row(self.geometry_advanced, 1, 'Geometry smoothing factor', self.geometry_smoothing_factor_var, HELP['geometry_smoothing_factor'])
        self._entry_row(self.geometry_advanced, 2, 'Neck cutoff interval', self.neck_cutoff_interval_var, HELP['neck_cutoff_interval'])
        self._entry_row(self.geometry_advanced, 3, 'Resample upper factor', self.resample_upper_var, HELP['resample_upper_factor'])
        self._entry_row(self.geometry_advanced, 4, 'Resample lower factor', self.resample_lower_var, HELP['resample_lower_factor'])

        param_wrap = ttk.Frame(self.inputs_tab)
        param_wrap.pack(fill='x', pady=(0, 8))
        param_wrap.columnconfigure(0, weight=1)
        param_wrap.columnconfigure(1, weight=1)

        self.dimless_frame = ttk.LabelFrame(param_wrap, text='Dimensionless parameters', padding=10)
        self.dimless_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 6))
        self.dimensional_frame = ttk.LabelFrame(param_wrap, text='Dimensional parameters', padding=10)
        self.dimensional_frame.grid(row=0, column=1, sticky='nsew', padx=(6, 0))

        self._entry_row(self.dimless_frame, 0, 'Aspect ratio beta_0 = B_0/D_0', self.beta_var, HELP['beta'])
        self._entry_row(self.dimless_frame, 1, 'Relative grain size ds_0 = d50/D_0', self.ds_var, HELP['ds'])
        self._entry_row(self.dimless_frame, 2, 'Shields stress theta_0', self.theta0_var, HELP['theta0'])
        self._combobox_row(self.dimless_frame, 3, 'Bed regime / resistance package', self.flagbed_label_var, list(BED_OPTIONS.keys()), help_key='flagbed')
        self._entry_row(self.dimless_frame, 4, 'Number of Fourier modes', self.mdat_var, HELP['mdat'])
        self.dimless_advanced = ttk.Frame(self.dimless_frame)
        self.dimless_advanced.grid(row=5, column=0, columnspan=2, sticky='ew')
        self._entry_row(self.dimless_advanced, 0, 'Transverse bed-slope coefficient r', self.rpic_var, HELP['rpic'])
        ttk.Label(self.dimless_advanced, text='Advanced parameter. Default values usually work well for exploratory runs.', wraplength=420, justify='left').grid(row=1, column=0, columnspan=2, sticky='w', pady=(4, 0))

        row = 0
        self._entry_row(self.dimensional_frame, row, 'Half-width B_0 [m]', self.half_width_var, HELP['half_width'])
        row += 1
        self.depth_row = self._entry_row(self.dimensional_frame, row, 'Reference depth D_0 [m]', self.dref_var, HELP['dref'])
        row += 1
        self._entry_row(self.dimensional_frame, row, 'Median grain size d50 [m]', self.d50_var, HELP['d50'])
        row += 1
        self._combobox_row(self.dimensional_frame, row, 'Sediment mobility input method', self.mobility_label_var, list(MOBILITY_OPTIONS.keys()), callback=self._sync_dimensional_fields, help_key='mobility')
        row += 1
        self.direct_theta_row = self._entry_row(self.dimensional_frame, row, 'Shields stress theta_0', self.theta0_var, HELP['theta0'])
        row += 1
        self.direct_tau_row = self._entry_row(self.dimensional_frame, row, 'Bed shear stress τb [Pa]', self.taub_var, HELP['tau_b'])
        row += 1
        self.slope_row = self._entry_row(self.dimensional_frame, row, 'Water-surface / bed slope S [-]', self.slope_var, HELP['slope'])
        row += 1
        self.velocity_row = self._entry_row(self.dimensional_frame, row, 'Mean velocity U [m/s]', self.velocity_var, HELP['velocity'])
        row += 1
        self.discharge_row = self._entry_row(self.dimensional_frame, row, 'Discharge Q [m³/s]', self.discharge_var, HELP['discharge'])
        row += 1
        self.friction_mode_row = self._combobox_row(self.dimensional_frame, row, 'Friction input', self.friction_label_var, list(FRICTION_OPTIONS.keys()), callback=self._sync_friction_label, help_key='friction')
        row += 1
        self.friction_value_row = self._entry_row(self.dimensional_frame, row, FRICTION_LABELS['cf'], self.friction_value_var, HELP['friction'])
        self.friction_value_label = self.friction_value_row[0]
        row += 1
        self._combobox_row(self.dimensional_frame, row, 'Bed regime / resistance package', self.flagbed_label_var, list(BED_OPTIONS.keys()), help_key='flagbed')
        row += 1
        self._entry_row(self.dimensional_frame, row, 'Number of Fourier modes', self.mdat_var, HELP['mdat'])
        row += 1
        self.dimensional_advanced = ttk.Frame(self.dimensional_frame)
        self.dimensional_advanced.grid(row=row, column=0, columnspan=2, sticky='ew', pady=(4, 0))
        self._entry_row(self.dimensional_advanced, 0, 'Transverse bed-slope coefficient r', self.rpic_var, HELP['rpic'])
        ttk.Label(self.dimensional_advanced, text='r stays dimensionless even in dimensional mode.', wraplength=420, justify='left').grid(row=1, column=0, columnspan=2, sticky='w', pady=(4, 0))

        preview_frame = ttk.LabelFrame(self.inputs_tab, text='Derived preview for dimensional mode', padding=10)
        preview_frame.pack(fill='x')
        self._preview_row(preview_frame, 0, 'Derived D_0 [m]', self.preview_dref_var)
        self._preview_row(preview_frame, 1, 'Derived beta_0 = B_0/D_0', self.preview_beta_var)
        self._preview_row(preview_frame, 2, 'Derived ds_0 = d50/D_0', self.preview_ds_var)
        self._preview_row(preview_frame, 3, 'Derived Shields stress theta_0', self.preview_theta_var)
        self._preview_row(preview_frame, 4, 'Derived Cf [-]', self.preview_cf_var)
        self._preview_row(preview_frame, 5, 'Derived velocity U [m/s]', self.preview_vel_var)

    def _build_run_tab(self) -> None:
        upper = ttk.Frame(self.run_tab)
        upper.pack(fill='x')
        upper.columnconfigure(0, weight=1)
        upper.columnconfigure(1, weight=1)

        run_frame = ttk.LabelFrame(upper, text='Run controls', padding=10)
        run_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 6))
        self._entry_row(run_frame, 0, 'Case id', self.case_id_var, HELP['case_id'])
        self._entry_row(run_frame, 1, 'Saved snapshot interval Nprint', self.nprint_var, HELP['nprint'])
        self._entry_row(run_frame, 2, 'Ntstep', self.ntstep_var, HELP['ntstep'])
        ttk.Checkbutton(run_frame, text='Enable live preview', variable=self.live_preview_var).grid(row=3, column=0, columnspan=2, sticky='w', pady=(4, 0))
        self._entry_row(run_frame, 4, 'Live preview every N steps', self.live_every_var, HELP['live_every'])
        ttk.Checkbutton(run_frame, text='Save solver PNG plots', variable=self.do_plots_var).grid(row=5, column=0, columnspan=2, sticky='w', pady=(4, 0))
        save_overlay_btn = ttk.Checkbutton(run_frame, text='Save final GUI overlay plot', variable=self.save_final_overlay_var)
        save_overlay_btn.grid(row=6, column=0, columnspan=2, sticky='w')
        ToolTip(save_overlay_btn, HELP['save_final_overlay'])
        save_manifest_btn = ttk.Checkbutton(run_frame, text='Save run manifest JSON', variable=self.save_run_manifest_var)
        save_manifest_btn.grid(row=7, column=0, columnspan=2, sticky='w')
        ToolTip(save_manifest_btn, HELP['save_run_manifest'])
        self._combobox_row(run_frame, 8, 'Output units', self.output_units_var, ['dimensionless', 'dimensional'], help_key='output_units')
        popup_btn = ttk.Checkbutton(run_frame, text='Show popup when simulation ends', variable=self.show_completion_popup_var)
        popup_btn.grid(row=9, column=0, columnspan=2, sticky='w')
        ToolTip(popup_btn, HELP['show_completion_popup'])

        advanced = ttk.LabelFrame(upper, text='Advanced solver controls', padding=10)
        advanced.grid(row=0, column=1, sticky='nsew', padx=(6, 0))
        self.advanced_frame = advanced
        self._combobox_row(advanced, 0, 'Boundary condition', self.flow_bc_var, ['free', 'periodic'], help_key='flow_bc')
        self._combobox_row(advanced, 1, 'Backend', self.backend_var, ['numpy', 'numba'], help_key='backend')
        self._entry_row(advanced, 2, 'cstab', self.cstab_var, HELP['cstab'])
        self._entry_row(advanced, 3, 'Parallel flow flag', self.flow_paral_var, HELP['flow_paral'])
        self._entry_row(advanced, 4, 'Flow workers', self.flow_workers_var, HELP['flow_workers'])
        numba_parallel_btn = ttk.Checkbutton(advanced, text='Numba parallel', variable=self.numba_parallel_var)
        numba_parallel_btn.grid(row=5, column=0, columnspan=2, sticky='w', pady=(4, 0))
        ToolTip(numba_parallel_btn, HELP['numba_parallel'])
        numba_fastmath_btn = ttk.Checkbutton(advanced, text='Numba fastmath', variable=self.numba_fastmath_var)
        numba_fastmath_btn.grid(row=6, column=0, columnspan=2, sticky='w')
        ToolTip(numba_fastmath_btn, HELP['numba_fastmath'])
        self._entry_row(advanced, 7, 'Sinuosity stability window', self.sinuo_window_var, HELP['sinuo_window'])
        self._entry_row(advanced, 8, 'Sinuosity relative tolerance', self.sinuo_rel_tol_var, HELP['sinuo_rel_tol'])

        stop_frame = ttk.LabelFrame(self.run_tab, text='Stop criteria', padding=10)
        stop_frame.pack(fill='x', pady=(8, 8))
        self._combobox_row(stop_frame, 0, 'Combine enabled criteria', self.stop_mode_var, ['first', 'all'], help_key='stop_mode')
        step_btn = ttk.Checkbutton(stop_frame, text='Enable max steps', variable=self.stop_on_steps_var)
        step_btn.grid(row=1, column=0, sticky='w', pady=(4, 0))
        ToolTip(step_btn, HELP['stop_on_steps'])
        step_entry = ttk.Entry(stop_frame, textvariable=self.max_steps_var)
        step_entry.grid(row=1, column=1, sticky='ew', pady=(4, 0))
        ToolTip(step_entry, HELP['max_steps'])
        time_btn = ttk.Checkbutton(stop_frame, text='Enable max simulated time', variable=self.stop_on_time_var)
        time_btn.grid(row=2, column=0, sticky='w', pady=(4, 0))
        ToolTip(time_btn, HELP['stop_on_time'])
        time_entry = ttk.Entry(stop_frame, textvariable=self.max_time_var)
        time_entry.grid(row=2, column=1, sticky='ew', pady=(4, 0))
        ToolTip(time_entry, HELP['max_sim_time'])
        cut_btn = ttk.Checkbutton(stop_frame, text='Enable max cutoffs', variable=self.stop_on_cutoffs_var)
        cut_btn.grid(row=3, column=0, sticky='w', pady=(4, 0))
        ToolTip(cut_btn, HELP['stop_on_cutoffs'])
        cut_entry = ttk.Entry(stop_frame, textvariable=self.max_cut_var)
        cut_entry.grid(row=3, column=1, sticky='ew', pady=(4, 0))
        ToolTip(cut_entry, HELP['max_cut'])
        sinuo_stop_btn = ttk.Checkbutton(
            stop_frame,
            text='Enable stop when sinuosity is statistically stable',
            variable=self.stop_on_sinuosity_stability_var,
        )
        sinuo_stop_btn.grid(row=4, column=0, columnspan=2, sticky='w', pady=(4, 0))
        ToolTip(sinuo_stop_btn, HELP['stop_on_sinuosity_stability'])
        stop_frame.columnconfigure(1, weight=1)

        action_frame = ttk.LabelFrame(self.run_tab, text='Actions', padding=10)
        action_frame.pack(fill='x', pady=(0, 8))
        ttk.Button(action_frame, text='Preview converted parameters', command=self._preview).pack(side='left', padx=(0, 8))
        ttk.Button(action_frame, text='Validate setup', command=self._validate_setup).pack(side='left', padx=(0, 8))
        ttk.Button(action_frame, text='Save config…', command=self._save_config).pack(side='left', padx=(0, 8))
        ttk.Button(action_frame, text='Load config…', command=self._load_config).pack(side='left', padx=(0, 8))
        ttk.Button(action_frame, text='Write Input/ files', command=self._write_inputs).pack(side='left', padx=(0, 8))
        self.run_button = ttk.Button(action_frame, text='Run case', command=self._run_case_threaded)
        self.run_button.pack(side='left', padx=(0, 8))
        self.stop_button = ttk.Button(action_frame, text='Stop after current step', command=self._request_stop, state='disabled')
        self.stop_button.pack(side='left', padx=(0, 8))
        self.continue_button = ttk.Button(action_frame, text='Continue from latest output', command=self._continue_from_latest_output, state='disabled')
        self.continue_button.pack(side='left', padx=(0, 8))
        ToolTip(self.stop_button, 'Request a graceful stop. The solver stops at the next safe iteration boundary and writes final output files.')
        ToolTip(self.continue_button, 'Start a new continuation run from the latest saved geometry after a stop criterion or manual stop.')

        diag_frame = ttk.LabelFrame(self.run_tab, text='Validation and diagnostic notes', padding=10)
        diag_frame.pack(fill='both', expand=True)
        self.validation_box = tk.Text(diag_frame, height=12, width=100)
        self.validation_box.pack(fill='both', expand=True)

        sinuo_frame = ttk.LabelFrame(self.run_tab, text='Sinuosity stability', padding=10)
        # Keep this panel above the large text diagnostics so it remains visible.
        sinuo_frame.pack(fill='x', pady=(0, 8), before=diag_frame)
        metrics = ttk.Frame(sinuo_frame)
        metrics.pack(fill='x', pady=(0, 6))
        self._metric_label(metrics, 0, 'State', self.sinuo_state_var)
        self._metric_label(metrics, 1, 'Current sinuo', self.sinuo_current_var)
        self._metric_label(metrics, 2, 'Window used', self.sinuo_window_used_var)
        self._metric_label(metrics, 3, 'Rel. span', self.sinuo_rel_span_var)
        self._metric_label(metrics, 4, 'Rel. trend/step', self.sinuo_rel_trend_var)
        ttk.Button(metrics, text='Refresh sinuosity', command=self._refresh_sinuosity_panel).grid(row=0, column=10, padx=(16, 0), sticky='e')
        metrics.columnconfigure(10, weight=1)
        self.sinuosity_canvas = FigureCanvasTkAgg(self.sinuosity_figure, master=sinuo_frame)
        self.sinuosity_canvas.get_tk_widget().pack(fill='x')
        self._clear_sinuosity_panel()

    def _build_plot_tab(self) -> None:
        info = ttk.Label(self.plot_tab, text='This tab shows the initial centerline, the latest live snapshot, and the final geometry after the run completes.')
        info.pack(anchor='w', pady=(0, 6))
        controls = ttk.Frame(self.plot_tab)
        controls.pack(fill='x', pady=(0, 6))
        show_orig_btn = ttk.Checkbutton(controls, text='Show original topography / initial geometry', variable=self.show_original_plot_var, command=lambda: self._update_plot(snapshot_path=self.last_snapshot_path, final=bool(self.latest_result)))
        show_orig_btn.pack(side='left')
        ToolTip(show_orig_btn, HELP['show_original_plot'])
        ttk.Button(controls, text='Refresh plot from latest output', command=self._refresh_plot_from_outputs).pack(side='left', padx=(12, 0))
        self.plot_canvas = FigureCanvasTkAgg(self.figure, master=self.plot_tab)
        self.plot_canvas.get_tk_widget().pack(fill='both', expand=True)

    def _build_log_tab(self) -> None:
        self.log = tk.Text(self.log_tab, wrap='word', height=24)
        self.log.pack(fill='both', expand=True)

    def _build_logo(self, parent) -> None:
        canvas = tk.Canvas(parent, width=140, height=56, highlightthickness=0)
        canvas.pack(side='left')
        canvas.create_text(10, 10, anchor='nw', text='LDSFL-Meander', font=('TkDefaultFont', 16, 'bold'))
        canvas.create_line(10, 38, 32, 22, 54, 42, 76, 18, 98, 38, 130, 24, smooth=True, width=3)

    def _entry_row(self, parent, row: int, label: str, textvar: tk.StringVar, help_text: str | None = None):
        lab = ttk.Label(parent, text=label)
        lab.grid(row=row, column=0, sticky='w', padx=(0, 8), pady=2)
        entry = ttk.Entry(parent, textvariable=textvar)
        entry.grid(row=row, column=1, sticky='ew', pady=2)
        parent.columnconfigure(1, weight=1)
        if help_text:
            ToolTip(lab, help_text)
            ToolTip(entry, help_text)
        return (lab, entry)

    def _combobox_row(self, parent, row: int, label: str, textvar: tk.StringVar, values: list[str], callback=None, help_key: str | None = None):
        lab = ttk.Label(parent, text=label)
        lab.grid(row=row, column=0, sticky='w', padx=(0, 8), pady=2)
        combo = ttk.Combobox(parent, textvariable=textvar, values=values, state='readonly')
        combo.grid(row=row, column=1, sticky='ew', pady=2)
        if callback is not None:
            combo.bind('<<ComboboxSelected>>', lambda _evt: callback())
        parent.columnconfigure(1, weight=1)
        if help_key:
            ToolTip(lab, HELP[help_key])
            ToolTip(combo, HELP[help_key])
        return (lab, combo)

    def _preview_row(self, parent, row, label, textvar):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=(0, 8), pady=1)
        ttk.Label(parent, textvariable=textvar).grid(row=row, column=1, sticky='w', pady=1)

    def _metric_label(self, parent, column: int, label: str, var: tk.StringVar):
        box = ttk.Frame(parent)
        box.grid(row=0, column=column, sticky='w', padx=(0, 14))
        ttk.Label(box, text=label).pack(anchor='w')
        ttk.Label(box, textvariable=var, font=('TkDefaultFont', 9, 'bold')).pack(anchor='w')

    def _bind_preview_updates(self):
        watched = [
            self.mode_var, self.beta_var, self.ds_var, self.theta0_var, self.rpic_var, self.mdat_var,
            self.half_width_var, self.dref_var, self.d50_var, self.taub_var, self.slope_var,
            self.velocity_var, self.discharge_var, self.friction_value_var,
            self.mobility_label_var, self.friction_label_var,
        ]
        for var in watched:
            var.trace_add('write', lambda *_args: self._update_dimensional_preview())
        for var in (self.xy_var, self.geometry_mode_label_var, self.half_width_var):
            var.trace_add('write', lambda *_args: self._refresh_initial_plot())

    def _browse_xy(self):
        path = filedialog.askopenfilename(title='Choose xy.csv', filetypes=[('CSV files', '*.csv'), ('All files', '*.*')], initialdir=str(Path(self.xy_var.get()).parent if self.xy_var.get() else self.default_workspace))
        if path:
            self.xy_var.set(path)

    def _browse_workspace(self):
        path = filedialog.askdirectory(title='Choose workspace directory', initialdir=self.workspace_var.get() or str(self.default_workspace))
        if path:
            self.workspace_var.set(path)

    def _sync_geometry_fields(self):
        self._refresh_initial_plot()

    def _sync_mode(self):
        mode = self.mode_var.get()
        self._walk_frame_state(self.dimless_frame, 'normal' if mode == 'dimensionless' else 'disabled')
        self._walk_frame_state(self.dimensional_frame, 'normal' if mode == 'dimensional' else 'disabled')
        self._sync_dimensional_fields()
        self._sync_geometry_fields()
        self._sync_advanced()

    def _walk_frame_state(self, frame, state: str):
        for child in frame.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                pass
            if isinstance(child, (ttk.Frame, ttk.LabelFrame)):
                self._walk_frame_state(child, state)

    def _sync_advanced(self):
        adv = bool(self.advanced_var.get())
        mode = self.mode_var.get()
        if adv:
            self.dimless_advanced.grid()
            self.dimensional_advanced.grid()
            self.geometry_advanced.grid()
            self.advanced_frame.grid()
        else:
            self.dimless_advanced.grid_remove()
            self.dimensional_advanced.grid_remove()
            self.geometry_advanced.grid_remove()
            self.advanced_frame.grid_remove()
        if mode != 'dimensionless':
            self._walk_frame_state(self.dimless_frame, 'disabled')
        if mode != 'dimensional':
            self._walk_frame_state(self.dimensional_frame, 'disabled')
        self._sync_dimensional_fields()

    def _sync_friction_label(self):
        mode = FRICTION_OPTIONS[self.friction_label_var.get()]
        self.friction_value_label.configure(text=FRICTION_LABELS[mode])
        defaults = {'cf': '0.01', 'darcy_f': '0.08', 'chezy_c': '35.0', 'manning_n': '0.03'}
        if not self.friction_value_var.get().strip():
            self.friction_value_var.set(defaults[mode])
        self._update_dimensional_preview()

    def _sync_dimensional_fields(self):
        mode = self.mode_var.get()
        mobility = MOBILITY_OPTIONS[self.mobility_label_var.get()]
        enabled = mode == 'dimensional'
        show_direct_theta = mobility == 'direct_shields'
        show_direct_tau = mobility == 'direct_shear_stress'
        show_slope = mobility in ('depth_slope_grain', 'discharge_half_width_slope_friction_grain')
        show_velocity = mobility == 'depth_velocity_friction_grain'
        show_discharge = mobility == 'discharge_half_width_slope_friction_grain'
        show_friction = mobility in ('depth_velocity_friction_grain', 'discharge_half_width_slope_friction_grain')
        use_depth = mobility != 'discharge_half_width_slope_friction_grain'

        self._row_visible(self.direct_theta_row, show_direct_theta)
        self._row_visible(self.direct_tau_row, show_direct_tau)
        self._row_visible(self.slope_row, show_slope)
        self._row_visible(self.velocity_row, show_velocity)
        self._row_visible(self.discharge_row, show_discharge)
        self._row_visible(self.friction_mode_row, show_friction)
        self._row_visible(self.friction_value_row, show_friction)
        self._row_visible(self.depth_row, use_depth)

        depth_state = 'normal' if (enabled and use_depth) else 'disabled'
        for widget in self.depth_row:
            try:
                widget.configure(state=depth_state)
            except tk.TclError:
                pass
        if show_friction:
            self._sync_friction_label()
        self._update_dimensional_preview()

    def _row_visible(self, row_widgets, visible: bool):
        for widget in row_widgets:
            if visible:
                widget.grid()
            else:
                widget.grid_remove()

    def _parse_flagbed(self) -> int:
        return BED_OPTIONS[self.flagbed_label_var.get()]

    def _effective_nprint(self) -> int:
        nprint = int(self.nprint_var.get())
        if self.live_preview_var.get() and self.live_every_var.get().strip():
            live_every = max(1, int(self.live_every_var.get()))
            return min(nprint, live_every)
        return nprint

    def _build_config(self) -> GuiCaseConfig:
        run = RunControls(
            case_id=int(self.case_id_var.get()),
            nprint=self._effective_nprint(),
            ntstep=int(self.ntstep_var.get()),
            max_cut=int(self.max_cut_var.get()),
            max_steps=int(self.max_steps_var.get()),
            max_sim_time=float(self.max_time_var.get()),
            stop_on_steps=bool(self.stop_on_steps_var.get()),
            stop_on_time=bool(self.stop_on_time_var.get()),
            stop_on_cutoffs=bool(self.stop_on_cutoffs_var.get()),
            stop_on_sinuosity_stability=bool(self.stop_on_sinuosity_stability_var.get()),
            stop_mode=self.stop_mode_var.get(),
            do_plots=bool(self.do_plots_var.get()),
            save_final_overlay=bool(self.save_final_overlay_var.get()),
            save_run_manifest=bool(self.save_run_manifest_var.get()),
            output_units=self.output_units_var.get(),
            flow_bc=self.flow_bc_var.get(),
            backend=self.backend_var.get(),
            flow_paral=int(self.flow_paral_var.get()),
            flow_workers=int(self.flow_workers_var.get()),
            numba_parallel=bool(self.numba_parallel_var.get()),
            numba_fastmath=bool(self.numba_fastmath_var.get()),
            cstab=float(self.cstab_var.get()),
            sinuo_window=int(self.sinuo_window_var.get()),
            sinuo_rel_tol=float(self.sinuo_rel_tol_var.get()),
        )
        kwargs = {
            'mode': self.mode_var.get(),
            'xy_csv': Path(self.xy_var.get()),
            'workspace_dir': Path(self.workspace_var.get()),
            'run': run,
            'geometry': GeometrySettings(
                mode=GEOMETRY_OPTIONS[self.geometry_mode_label_var.get()],
                custom_scale=None,
                smoothing_enabled=bool(self.geometry_smoothing_enabled_var.get()),
                smoothing_factor=float(self.geometry_smoothing_factor_var.get()),
                neck_cutoff_interval=int(self.neck_cutoff_interval_var.get()),
                resample_upper_factor=float(self.resample_upper_var.get()),
                resample_lower_factor=float(self.resample_lower_var.get()),
            ),
        }
        flagbed = self._parse_flagbed()
        if self.mode_var.get() == 'dimensionless':
            kwargs['dimensionless'] = DimensionlessInputs(beta=float(self.beta_var.get()), ds=float(self.ds_var.get()), theta0=float(self.theta0_var.get()), flagbed=flagbed, rpic_0=float(self.rpic_var.get()), Mdat=int(self.mdat_var.get()))
        else:
            kwargs['dimensional'] = DimensionalInputs(
                half_width=float(self.half_width_var.get()),
                dref=float(self.dref_var.get()) if self.dref_var.get().strip() else None,
                d50=float(self.d50_var.get()),
                mobility_mode=MOBILITY_OPTIONS[self.mobility_label_var.get()],
                theta0=float(self.theta0_var.get()) if self.theta0_var.get().strip() else None,
                tau_b=float(self.taub_var.get()) if self.taub_var.get().strip() else None,
                slope=float(self.slope_var.get()) if self.slope_var.get().strip() else None,
                velocity=float(self.velocity_var.get()) if self.velocity_var.get().strip() else None,
                discharge=float(self.discharge_var.get()) if self.discharge_var.get().strip() else None,
                friction_mode=FRICTION_OPTIONS[self.friction_label_var.get()],
                friction_value=float(self.friction_value_var.get()) if self.friction_value_var.get().strip() else None,
                flagbed=flagbed,
                rpic_0=float(self.rpic_var.get()),
                Mdat=int(self.mdat_var.get()),
            )
        return GuiCaseConfig(**kwargs)

    def _update_dimensional_preview(self):
        try:
            cfg = self._build_config()
            if cfg.mode != 'dimensional' or cfg.dimensional is None:
                raise ValueError
            vals = cfg.dimensional.derived_values()
            self.preview_dref_var.set(f"{vals['D0']:.8g}")
            self.preview_beta_var.set(f"{vals['beta0']:.8g}")
            self.preview_ds_var.set(f"{vals['ds0']:.8g}")
            self.preview_theta_var.set(f"{vals['theta0']:.8g}")
            self.preview_cf_var.set(f"{vals.get('Cf', float('nan')):.8g}" if 'Cf' in vals else '—')
            self.preview_vel_var.set(f"{vals.get('velocity', float('nan')):.8g}" if 'velocity' in vals else '—')
        except Exception:
            self.preview_dref_var.set('—')
            self.preview_beta_var.set('—')
            self.preview_ds_var.set('—')
            self.preview_theta_var.set('—')
            self.preview_cf_var.set('—')
            self.preview_vel_var.set('—')

    def _preview(self):
        try:
            config = self._build_config()
            summary = preview_case_config(config)
            self._log('Converted parameters preview:')
            self._log(json.dumps(summary, indent=2))
            self.validation_box.delete('1.0', 'end')
            self.validation_box.insert('end', json.dumps(summary, indent=2))
        except Exception as exc:
            self._show_error(exc)

    def _validate_setup(self) -> bool:
        try:
            config = self._build_config()
            warnings = validate_case_config(config)
            summary = preview_case_config(config)
            lines = ['Validation successful.', '', json.dumps(summary, indent=2)]
            if warnings:
                lines.extend(['', 'Warnings:'])
                lines.extend(f'- {w}' for w in warnings)
            else:
                lines.extend(['', 'No warnings.'])
            self.validation_box.delete('1.0', 'end')
            self.validation_box.insert('end', '\n'.join(lines))
            self._log('Validation successful.')
            for w in warnings:
                self._log(f'Warning: {w}')
            return True
        except Exception as exc:
            tb = traceback.format_exc()
            self.validation_box.delete('1.0', 'end')
            self.validation_box.insert('end', f'Validation failed:\n{exc}\n\n{tb}')
            self._show_error_message(str(exc), tb)
            return False

    def _write_inputs(self):
        try:
            config = self._build_config()
            warnings = validate_case_config(config)
            summary = write_case_inputs(config)
            self._log('Input files updated successfully.')
            self._log(json.dumps(summary, indent=2))
            if warnings:
                self._log('Warnings before write:')
                for w in warnings:
                    self._log(f'  - {w}')
        except Exception as exc:
            self._show_error(exc)

    def _save_config(self):
        try:
            config = self._build_config()
            payload = {
                'ldsfl_gui_version': 3,
                'project_name': 'LDSFL-Meander',
                'project_expansion': 'Lopez-Dubon, Sgarabotto, Frascati and Lanzoni',
                'config': config_to_dict(config),
                'gui_state': {
                    'advanced': bool(self.advanced_var.get()),
                    'live_preview': bool(self.live_preview_var.get()),
                    'live_every': self.live_every_var.get(),
                    'show_original_plot': bool(self.show_original_plot_var.get()),
                    'show_completion_popup': bool(self.show_completion_popup_var.get()),
                    'stop_on_sinuosity_stability': bool(self.stop_on_sinuosity_stability_var.get()),
                    'sinuo_window': self.sinuo_window_var.get(),
                    'sinuo_rel_tol': self.sinuo_rel_tol_var.get(),
                },
            }
            path = filedialog.asksaveasfilename(title='Save LDSFL-Meander config', defaultextension='.json', filetypes=[('JSON files', '*.json')])
            if not path:
                return
            Path(path).write_text(json.dumps(payload, indent=2))
            self._log(f'Saved GUI config to {path}')
        except Exception as exc:
            self._show_error(exc)

    def _load_config(self):
        try:
            path = filedialog.askopenfilename(title='Load LDSFL-Meander config', filetypes=[('JSON files', '*.json'), ('All files', '*.*')])
            if not path:
                return
            payload = json.loads(Path(path).read_text())
            cfg = config_from_dict(payload['config'])
            self._apply_config(cfg)
            gui_state = payload.get('gui_state', {})
            self.advanced_var.set(bool(gui_state.get('advanced', False)))
            self.live_preview_var.set(bool(gui_state.get('live_preview', True)))
            self.live_every_var.set(str(gui_state.get('live_every', '5')))
            self.show_original_plot_var.set(bool(gui_state.get('show_original_plot', True)))
            self.show_completion_popup_var.set(bool(gui_state.get('show_completion_popup', True)))
            self.stop_on_sinuosity_stability_var.set(
                bool(gui_state.get(
                    'stop_on_sinuosity_stability',
                    getattr(cfg.run, 'stop_on_sinuosity_stability', False),
                ))
            )
            self.sinuo_window_var.set(str(gui_state.get('sinuo_window', getattr(cfg.run, 'sinuo_window', 100))))
            self.sinuo_rel_tol_var.set(str(gui_state.get('sinuo_rel_tol', getattr(cfg.run, 'sinuo_rel_tol', 0.005))))
            self._sync_mode()
            self._sync_advanced()
            self._sync_dimensional_fields()
            self._update_dimensional_preview()
            self._refresh_initial_plot()
            self._log(f'Loaded GUI config from {path}')
        except Exception as exc:
            self._show_error(exc)

    def _apply_config(self, cfg: GuiCaseConfig):
        self.mode_var.set(cfg.mode)
        self.xy_var.set(str(cfg.xy_csv))
        self.workspace_var.set(str(cfg.workspace_dir))
        self.geometry_mode_label_var.set(next(k for k, v in GEOMETRY_OPTIONS.items() if v == cfg.geometry.mode))
        self.geometry_smoothing_enabled_var.set(bool(cfg.geometry.smoothing_enabled))
        self.geometry_smoothing_factor_var.set(str(cfg.geometry.smoothing_factor))
        self.neck_cutoff_interval_var.set(str(cfg.geometry.neck_cutoff_interval))
        self.resample_upper_var.set(str(cfg.geometry.resample_upper_factor))
        self.resample_lower_var.set(str(cfg.geometry.resample_lower_factor))
        self.case_id_var.set(str(cfg.run.case_id))
        self.nprint_var.set(str(cfg.run.nprint))
        self.ntstep_var.set(str(cfg.run.ntstep))
        self.max_cut_var.set(str(cfg.run.max_cut))
        self.max_steps_var.set(str(cfg.run.max_steps))
        self.max_time_var.set(str(cfg.run.max_sim_time))
        self.stop_on_steps_var.set(bool(cfg.run.stop_on_steps))
        self.stop_on_time_var.set(bool(cfg.run.stop_on_time))
        self.stop_on_cutoffs_var.set(bool(cfg.run.stop_on_cutoffs))
        self.stop_on_sinuosity_stability_var.set(
            bool(getattr(cfg.run, 'stop_on_sinuosity_stability', False))
        )
        self.stop_mode_var.set(str(cfg.run.stop_mode))
        self.do_plots_var.set(bool(cfg.run.do_plots))
        self.save_final_overlay_var.set(bool(cfg.run.save_final_overlay))
        self.save_run_manifest_var.set(bool(cfg.run.save_run_manifest))
        self.output_units_var.set(str(getattr(cfg.run, 'output_units', 'dimensionless')))
        self.flow_bc_var.set(cfg.run.flow_bc)
        self.backend_var.set(cfg.run.backend)
        self.flow_paral_var.set(str(cfg.run.flow_paral))
        self.flow_workers_var.set(str(cfg.run.flow_workers))
        self.numba_parallel_var.set(bool(cfg.run.numba_parallel))
        self.numba_fastmath_var.set(bool(cfg.run.numba_fastmath))
        self.cstab_var.set(str(cfg.run.cstab))
        self.sinuo_window_var.set(str(getattr(cfg.run, 'sinuo_window', 100)))
        self.sinuo_rel_tol_var.set(str(getattr(cfg.run, 'sinuo_rel_tol', 0.005)))
        if cfg.dimensionless is not None:
            self.beta_var.set(str(cfg.dimensionless.beta))
            self.ds_var.set(str(cfg.dimensionless.ds))
            self.theta0_var.set(str(cfg.dimensionless.theta0))
            self.flagbed_label_var.set(next(k for k, v in BED_OPTIONS.items() if v == cfg.dimensionless.flagbed))
            self.rpic_var.set(str(cfg.dimensionless.rpic_0))
            self.mdat_var.set(str(cfg.dimensionless.Mdat))
        if cfg.dimensional is not None:
            self.half_width_var.set(str(cfg.dimensional.half_width))
            self.dref_var.set('' if cfg.dimensional.dref is None else str(cfg.dimensional.dref))
            self.d50_var.set(str(cfg.dimensional.d50))
            self.mobility_label_var.set(next(k for k, v in MOBILITY_OPTIONS.items() if v == cfg.dimensional.mobility_mode))
            self.theta0_var.set('' if cfg.dimensional.theta0 is None else str(cfg.dimensional.theta0))
            self.taub_var.set('' if cfg.dimensional.tau_b is None else str(cfg.dimensional.tau_b))
            self.slope_var.set('' if cfg.dimensional.slope is None else str(cfg.dimensional.slope))
            self.velocity_var.set('' if cfg.dimensional.velocity is None else str(cfg.dimensional.velocity))
            self.discharge_var.set('' if cfg.dimensional.discharge is None else str(cfg.dimensional.discharge))
            self.friction_label_var.set(next(k for k, v in FRICTION_OPTIONS.items() if v == cfg.dimensional.friction_mode))
            self.friction_value_var.set('' if cfg.dimensional.friction_value is None else str(cfg.dimensional.friction_value))
            self.flagbed_label_var.set(next(k for k, v in BED_OPTIONS.items() if v == cfg.dimensional.flagbed))
            self.rpic_var.set(str(cfg.dimensional.rpic_0))
            self.mdat_var.set(str(cfg.dimensional.Mdat))

    def _set_run_button_state(self, *, running: bool):
        if self.run_button is not None:
            self.run_button.configure(state='disabled' if running else 'normal')
        if self.stop_button is not None:
            self.stop_button.configure(state='normal' if running else 'disabled')
        if self.continue_button is not None:
            can_continue = (not running) and (self.latest_result is not None or self.last_snapshot_path is not None)
            self.continue_button.configure(state='normal' if can_continue else 'disabled')

    def _request_stop(self):
        if not self.run_in_progress:
            return
        self.stop_requested_event.set()
        self.status_var.set('Stop requested. The solver will stop at the next safe step boundary…')
        self._log('User stop requested. Waiting for the current solver step to finish safely.')
        if self.stop_button is not None:
            self.stop_button.configure(state='disabled')

    def _latest_xyu_snapshot_path(self) -> Path | None:
        candidates: list[Path] = []
        if self.last_snapshot_path is not None and Path(self.last_snapshot_path).exists():
            candidates.append(Path(self.last_snapshot_path))
        if self.current_xyu_dir is not None and Path(self.current_xyu_dir).exists():
            candidates.extend(Path(self.current_xyu_dir).glob('*.csv'))
        if self.latest_result is not None:
            xyu_dir = Path(self.workspace_var.get()) / 'Output' / self.latest_result['id_files'] / 'xyu'
            if xyu_dir.exists():
                candidates.extend(xyu_dir.glob('*.csv'))
        if not candidates:
            return None
        return sorted(set(candidates), key=lambda p: p.stat().st_mtime)[-1]

    def _continue_from_latest_output(self):
        if self.run_in_progress:
            messagebox.showinfo('LDSFL-Meander GUI', 'A run is already in progress.')
            return
        latest = self._latest_xyu_snapshot_path()
        if latest is None or not latest.exists():
            messagebox.showinfo('LDSFL-Meander GUI', 'No saved geometry snapshot is available yet. Run or refresh outputs first.')
            return
        if self.latest_config is None:
            try:
                base_config = self._build_config()
            except Exception as exc:
                self._show_error(exc)
                return
        else:
            base_config = self.latest_config
        try:
            cfg = copy.deepcopy(base_config)
            df = pd.read_csv(latest)
            if not {'x', 'y'}.issubset(df.columns):
                raise ValueError(f'Latest snapshot does not contain x and y columns: {latest}')
            x = df['x'].astype(float).to_numpy()
            y = df['y'].astype(float).to_numpy()

            # If saved outputs were dimensional, convert back to solver units.
            scale = 1.0
            if self.latest_result is not None and str(self.latest_result.get('output_units', 'dimensionless')).lower() == 'dimensional':
                scale = float(self.latest_result.get('output_length_scale', 1.0) or 1.0)
            if scale != 1.0:
                x = x / scale
                y = y / scale

            cont_path = Path(cfg.workspace_dir) / 'Input' / 'xy_continue_from_latest.csv'
            cont_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({'x': x, 'y': y}).to_csv(cont_path, header=False, index=False)
            cfg.xy_csv = cont_path
            cfg.geometry.mode = 'as_is'
            cfg.geometry.custom_scale = None
            self._log(f'Prepared continuation geometry from {latest.name}: {cont_path}')
            self._run_case_threaded(config_override=cfg, continuation=True)
        except Exception as exc:
            self._show_error(exc)

    def _run_case_threaded(self, config_override: GuiCaseConfig | None = None, *, continuation: bool = False):
        if self.run_in_progress:
            messagebox.showinfo('LDSFL-Meander GUI', 'A run is already in progress.')
            return
        if config_override is None:
            if not self._validate_setup():
                return
            config = self._build_config()
            warnings = validate_case_config(config)
            if warnings:
                proceed = messagebox.askyesno('LDSFL-Meander warnings', 'Validation produced warnings. Continue anyway?\n\n' + '\n'.join(f'• {w}' for w in warnings))
                if not proceed:
                    return
        else:
            config = config_override
        self.stop_requested_event.clear()
        self.run_in_progress = True
        self._set_run_button_state(running=True)
        self.status_var.set('Continuation run in progress…' if continuation else 'Run in progress…')
        self.current_id_files = compute_id_files(config)
        self.current_xyu_dir = Path(config.workspace_dir) / 'Output' / self.current_id_files / 'xyu'
        self.last_snapshot_path = None
        self.last_sinuosity_history_mtime = None
        self.latest_result = None
        self.latest_config = config
        self.final_xy = None
        self._refresh_initial_plot()
        self.notebook.select(self.plot_tab)
        self.run_thread = threading.Thread(target=self._run_case_worker, args=(config, continuation), daemon=True)
        self.run_thread.start()
        if self.live_preview_var.get():
            self._schedule_monitor()

    def _run_case_worker(self, config: GuiCaseConfig, continuation: bool = False):
        try:
            summary = write_case_inputs(config)
            self._log('Starting LDSFL-Meander continuation run...' if continuation else 'Starting LDSFL-Meander run...')
            self._log(json.dumps(summary, indent=2))
            scales = output_scales(config)
            results = run_project(
                Path(config.workspace_dir),
                cases=[config.run.case_id],
                Nprint=config.run.nprint,
                Ntstep=config.run.ntstep,
                Max_Cut=config.run.max_cut,
                max_steps=None if config.run.max_steps <= 0 else config.run.max_steps,
                max_sim_time=None if config.run.max_sim_time <= 0 else config.run.max_sim_time,
                stop_on_steps=config.run.stop_on_steps,
                stop_on_time=config.run.stop_on_time,
                stop_on_cutoffs=config.run.stop_on_cutoffs,
                stop_on_sinuosity_stability=config.run.stop_on_sinuosity_stability,
                stop_mode=config.run.stop_mode,
                cstab=config.run.cstab,
                geometry_smoothing_enabled=config.geometry.smoothing_enabled,
                geometry_smoothing_factor=config.geometry.smoothing_factor,
                neck_cutoff_interval=config.geometry.neck_cutoff_interval,
                resample_upper_factor=config.geometry.resample_upper_factor,
                resample_lower_factor=config.geometry.resample_lower_factor,
                do_plots=config.run.do_plots,
                flow_bc=config.run.flow_bc,
                flow_paral=config.run.flow_paral,
                flow_workers=config.run.flow_workers,
                flow_backend=config.run.backend,
                numba_parallel=config.run.numba_parallel,
                numba_fastmath=config.run.numba_fastmath,
                output_units=scales['resolved_output_units'],
                output_length_scale=scales['output_length_scale'],
                output_velocity_scale=scales['output_velocity_scale'],
                sinuo_window=config.run.sinuo_window,
                sinuo_rel_tol=config.run.sinuo_rel_tol,
                stop_requested_callback=self.stop_requested_event.is_set,
            )
            result = results[0]
            if config.run.save_run_manifest:
                manifest = {
                    'project_name': 'LDSFL-Meander',
                    'project_expansion': 'Lopez-Dubon, Sgarabotto, Frascati and Lanzoni',
                    'input_summary': summary,
                    'run_result': result,
                }
                out_root = Path(config.workspace_dir) / 'Output' / result['id_files']
                out_root.mkdir(parents=True, exist_ok=True)
                (out_root / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
            self.after(0, lambda: self._finish_run(result))
        except Exception as exc:
            tb = traceback.format_exc()
            self.after(0, lambda msg=str(exc), tb=tb: self._fail_run(msg, tb))

    def _schedule_monitor(self):
        if self.monitor_job is not None:
            self.after_cancel(self.monitor_job)
        self.monitor_job = self.after(700, self._monitor_snapshots)

    def _monitor_snapshots(self):
        self.monitor_job = None
        try:
            if self.current_xyu_dir is not None and self.current_xyu_dir.exists():
                files = sorted(self.current_xyu_dir.glob('*.csv'), key=lambda p: p.stat().st_mtime)
                if files:
                    latest = files[-1]
                    if self.last_snapshot_path != latest:
                        self.last_snapshot_path = latest
                        self._update_plot(snapshot_path=latest, final=False)
                        self._log(f'Updated live plot from snapshot {latest.name}')

            # The sinuosity history is written independently of the live x/y snapshot.
            # Refresh it whenever its CSV modification time changes.
            history_path = self._sinuosity_history_path()
            if history_path is not None and history_path.exists():
                mtime = history_path.stat().st_mtime
                if self.last_sinuosity_history_mtime != mtime:
                    self.last_sinuosity_history_mtime = mtime
                    self._refresh_sinuosity_panel(log_errors=False)
        except Exception as exc:
            self._log(f'Live preview skipped: {exc}')
        finally:
            if self.run_in_progress and self.live_preview_var.get():
                self._schedule_monitor()

    def _current_output_root(self) -> Path | None:
        if self.latest_result is None:
            return None
        return Path(self.workspace_var.get()) / 'Output' / self.latest_result['id_files']

    def _save_overlay_plot_if_requested(self):
        if not self.save_final_overlay_var.get():
            return
        out_root = self._current_output_root()
        if out_root is None:
            return
        out_root.mkdir(parents=True, exist_ok=True)
        target = out_root / 'gui_final_overlay.png'
        self.figure.savefig(target, dpi=160, bbox_inches='tight')
        self._log(f'Saved GUI overlay plot to {target}')

    def _notify_run_end(self, title: str, message: str, kind: str = 'info'):
        if not self.show_completion_popup_var.get():
            return
        if kind == 'error':
            messagebox.showerror(title, message)
        else:
            messagebox.showinfo(title, message)

    def _finish_run(self, result: dict):
        self.run_in_progress = False
        self.stop_requested_event.clear()
        self.latest_result = result
        self._set_run_button_state(running=False)
        reason = result.get('stop_reason', 'Run complete.')
        self.status_var.set(f'Run complete: {reason}')
        self._log('Run complete.')
        self._log(json.dumps(result, indent=2))
        if self.monitor_job is not None:
            self.after_cancel(self.monitor_job)
            self.monitor_job = None
        self._refresh_plot_from_outputs()
        self._refresh_sinuosity_panel()
        self._save_overlay_plot_if_requested()
        self.notebook.select(self.plot_tab)
        self._notify_run_end('LDSFL-Meander simulation finished', f"Simulation finished.\n\nReason: {reason}\nSteps: {result.get('steps')}\nCutoffs: {result.get('cut_cnt')}")

    def _fail_run(self, message: str, tb: str):
        self.run_in_progress = False
        self.stop_requested_event.clear()
        self._set_run_button_state(running=False)
        self.status_var.set('Run failed.')
        if self.monitor_job is not None:
            self.after_cancel(self.monitor_job)
            self.monitor_job = None
        self._show_error_message(message, tb)

    def _refresh_initial_plot(self):
        try:
            if not self.xy_var.get().strip():
                self.initial_xy = None
            else:
                cfg = self._build_config()
                table, _scale = build_scaled_xy_table(cfg)
                self.initial_xy = (table.iloc[:, 0].to_numpy(), table.iloc[:, 1].to_numpy())
        except Exception:
            self.initial_xy = self._load_xy_path(Path(self.xy_var.get())) if self.xy_var.get().strip() else None
        self._update_plot(snapshot_path=None, final=False)

    def _refresh_plot_from_outputs(self):
        try:
            if self.current_xyu_dir is None or not self.current_xyu_dir.exists():
                if self.latest_result is not None:
                    self.current_xyu_dir = Path(self.workspace_var.get()) / 'Output' / self.latest_result['id_files'] / 'xyu'
                else:
                    raise FileNotFoundError('No output snapshot directory found yet.')
            files = sorted(self.current_xyu_dir.glob('*.csv'), key=lambda p: p.stat().st_mtime)
            if not files:
                raise FileNotFoundError('No xyu snapshot CSV files found yet.')
            self.last_snapshot_path = files[-1]
            self._update_plot(snapshot_path=self.last_snapshot_path, final=True)
        except Exception as exc:
            self._show_error(exc)

    def _load_xy_path(self, path: Path):
        if not path.exists():
            return None
        df, _info = parse_geometry_csv(path)
        return df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()

    def _load_snapshot_xy(self, path: Path):
        df = pd.read_csv(path)
        if 'x' not in df.columns or 'y' not in df.columns:
            raise ValueError(f'Snapshot file does not contain x and y columns: {path}')
        return df['x'].to_numpy(), df['y'].to_numpy()

    def _update_plot(self, snapshot_path: Path | None, final: bool):
        self.ax.clear()
        for legend in list(self.figure.legends):
            legend.remove()
        if self.show_original_plot_var.get() and self.initial_xy is not None:
            self.ax.plot(self.initial_xy[0], self.initial_xy[1], label='Original / initial', linewidth=1.6)
        if snapshot_path is not None and snapshot_path.exists():
            xy = self._load_snapshot_xy(snapshot_path)
            self.final_xy = xy if final else self.final_xy
            label = 'Final' if final else f'Live snapshot ({snapshot_path.name})'
            self.ax.plot(xy[0], xy[1], label=label, linewidth=1.6)
        elif self.final_xy is not None:
            self.ax.plot(self.final_xy[0], self.final_xy[1], label='Final', linewidth=1.6)
        self.ax.set_aspect('equal', adjustable='box')
        units = '[-]'
        if self.latest_config is not None and getattr(self.latest_config.run, 'output_units', 'dimensionless') == 'dimensional' and self.latest_config.mode == 'dimensional':
            units = '[m]'
        self.ax.set_xlabel(f'x {units}')
        self.ax.set_ylabel(f'y {units}')
        self.ax.set_title('')
        if getattr(self, '_figure_title', None) is None:
            self._figure_title = self.figure.suptitle('LDSFL-Meander planform view', y=0.985)
        else:
            self._figure_title.set_text('LDSFL-Meander planform view')
            self._figure_title.set_y(0.985)
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.figure.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.01, 0.965), borderaxespad=0.0, frameon=True)
        self.ax.grid(True, alpha=0.3)
        self.figure.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
        self.plot_canvas.draw_idle()

    def _sinuosity_history_path(self) -> Path | None:
        id_files = None
        if self.latest_result is not None:
            id_files = self.latest_result.get('id_files')
        elif self.current_id_files is not None:
            id_files = self.current_id_files
        if not id_files:
            return None
        return Path(self.workspace_var.get()) / 'Output' / id_files / 'files' / f'sinuosity_history_{id_files}.csv'

    def _layout_sinuosity_figure(self):
        # Avoid tight_layout warnings in the compact embedded Tkinter panel.
        # Fixed margins are more robust here because the panel is small and is
        # refreshed repeatedly while the solver is running.
        self.sinuosity_figure.subplots_adjust(
            left=0.18,
            right=0.97,
            bottom=0.24,
            top=0.83,
        )

    def _clear_sinuosity_panel(self):
        self.sinuosity_ax.clear()
        self.sinuosity_ax.set_title('Sinuosity evolution')
        self.sinuosity_ax.set_xlabel('Step [-]')
        self.sinuosity_ax.set_ylabel('Sinuosity [-]')
        self.sinuosity_ax.grid(True, alpha=0.3)

        self.sinuo_state_var.set('Not available')
        self.sinuo_current_var.set('—')
        self.sinuo_window_used_var.set('—')
        self.sinuo_rel_span_var.set('—')
        self.sinuo_rel_trend_var.set('—')

        # These variables exist only when the equivalence-metrics panel is enabled.
        # Guard with hasattr so older/simpler GUI builds still launch cleanly.
        if hasattr(self, 'sinuo_equiv_state_var'):
            self.sinuo_equiv_state_var.set('—')
        if hasattr(self, 'sinuo_equiv_drift_var'):
            self.sinuo_equiv_drift_var.set('—')
        if hasattr(self, 'sinuo_equiv_ci_var'):
            self.sinuo_equiv_ci_var.set('—')
        if hasattr(self, 'sinuo_equiv_tol_var'):
            self.sinuo_equiv_tol_var.set('±0.02')

        self._layout_sinuosity_figure()
        if self.sinuosity_canvas is not None:
            self.sinuosity_canvas.draw_idle()

    def _compute_sinuosity_stability_from_arrays(self, steps, values) -> dict:
        steps = np.asarray(steps, dtype=np.float64)
        vals = np.asarray(values, dtype=np.float64)
        try:
            window = int(self.sinuo_window_var.get())
        except Exception:
            window = 100
        try:
            rel_tol = float(self.sinuo_rel_tol_var.get())
        except Exception:
            rel_tol = 5.0e-3

        if vals.size == 0:
            return {
                'state': 'not available',
                'stable': False,
                'quasi_stable': False,
                'window_requested': int(window),
                'window_used': 0,
                'rel_span': np.nan,
                'rel_last_change': np.nan,
                'rel_trend_per_step': np.nan,
                'sinuo_final': np.nan,
            }
        if vals.size == 1:
            return {
                'state': 'not enough history',
                'stable': False,
                'quasi_stable': False,
                'window_requested': int(window),
                'window_used': 1,
                'rel_span': 0.0,
                'rel_last_change': 0.0,
                'rel_trend_per_step': 0.0,
                'sinuo_final': float(vals[-1]),
            }

        w = max(2, min(int(window), vals.size))
        tail = vals[-w:]
        tail_steps = steps[-w:] if steps.size >= vals.size else np.arange(vals.size - w, vals.size, dtype=np.float64)
        ref = max(abs(float(tail[-1])), 1.0e-12)
        rel_span = float((tail.max() - tail.min()) / ref)
        rel_last_change = float(abs(tail[-1] - tail[-2]) / ref)
        x = tail_steps - tail_steps[0]
        if np.allclose(x, x[0]):
            slope = 0.0
        else:
            slope = float(np.polyfit(x, tail, 1)[0])
        rel_trend_per_step = float(abs(slope) / ref)
        trend_tol = float(rel_tol) / max(float(w), 1.0)
        stable = (rel_span < rel_tol) and (rel_trend_per_step < trend_tol)
        quasi_stable = stable or ((rel_span < 2.0 * rel_tol) and (rel_trend_per_step < 2.0 * trend_tol))
        if stable:
            state = 'stable'
        elif quasi_stable:
            state = 'quasi-stable'
        else:
            state = 'not stable'
        return {
            'state': state,
            'stable': bool(stable),
            'quasi_stable': bool(quasi_stable),
            'window_requested': int(window),
            'window_used': int(w),
            'rel_span': rel_span,
            'rel_last_change': rel_last_change,
            'rel_trend_per_step': rel_trend_per_step,
            'sinuo_final': float(vals[-1]),
        }

    def _format_metric(self, value):
        try:
            value = float(value)
            if value != value:
                return '—'
            return f'{value:.4g}'
        except Exception:
            return '—'

    def _update_sinuosity_metrics(self, stability: dict | None):
        if not stability:
            self.sinuo_state_var.set('Not available')
            self.sinuo_current_var.set('—')
            self.sinuo_window_used_var.set('—')
            self.sinuo_rel_span_var.set('—')
            self.sinuo_rel_trend_var.set('—')
            return
        self.sinuo_state_var.set(str(stability.get('state', 'not available')))
        self.sinuo_current_var.set(self._format_metric(stability.get('sinuo_final')))
        self.sinuo_window_used_var.set(str(stability.get('window_used', '—')))
        self.sinuo_rel_span_var.set(self._format_metric(stability.get('rel_span')))
        self.sinuo_rel_trend_var.set(self._format_metric(stability.get('rel_trend_per_step')))

    def _refresh_sinuosity_panel(self, log_errors: bool = True):
        try:
            history_path = self._sinuosity_history_path()
            if history_path is None or not history_path.exists():
                if self.latest_result is not None:
                    self._update_sinuosity_metrics(self.latest_result.get('sinuosity_stability'))
                return
            df = pd.read_csv(history_path)
            if 'step' not in df.columns or 'sinuo' not in df.columns:
                raise ValueError(f'Sinuosity history file does not contain step and sinuo columns: {history_path}')
            self.sinuosity_ax.clear()
            self.sinuosity_ax.plot(df['step'].to_numpy(), df['sinuo'].to_numpy(), linewidth=1.7)
            self.sinuosity_ax.set_title('Sinuosity evolution')
            self.sinuosity_ax.set_xlabel('Step [-]')
            self.sinuosity_ax.set_ylabel('Sinuosity [-]')
            self.sinuosity_ax.grid(True, alpha=0.3)
            self._layout_sinuosity_figure()
            if self.sinuosity_canvas is not None:
                self.sinuosity_canvas.draw_idle()
            # Compute metrics directly from the CSV so the panel works during a run,
            # before latest_result is available. After the run, prefer the solver-returned
            # metrics when they exist.
            if self.latest_result is not None and self.latest_result.get('sinuosity_stability'):
                self._update_sinuosity_metrics(self.latest_result.get('sinuosity_stability'))
            else:
                stability = self._compute_sinuosity_stability_from_arrays(
                    df['step'].to_numpy(),
                    df['sinuo'].to_numpy(),
                )
                self._update_sinuosity_metrics(stability)
        except Exception as exc:
            if log_errors:
                self._log(f'Sinuosity panel refresh skipped: {exc}')

    def _show_about(self):
        messagebox.showinfo(
            'About LDSFL-Meander',
            'LDSFL-Meander\n\n'
            'Lopez-Dubon, Sgarabotto, Frascati and Lanzoni\n\n'
            f'GUI version built on package version {__version__}.\n\n'
            'This desktop GUI is a thin front end around the same solver used by the CLI. '
            'The solver remains dimensionless internally; dimensional mode converts hydraulic inputs before writing Input/Parameter.csv, and the geometry panel can optionally rescale x and y by B_0 before writing Input/xy.csv.\n\n'
            'Recommended use: wide, mildly curved, long bends in a reduced-model setting. '
            'Use advanced mode only when you need explicit control over the slope-deflection coefficient or solver options.'
        )

    def _show_error(self, exc: Exception):
        tb = traceback.format_exc()
        self._show_error_message(str(exc), tb)

    def _show_error_message(self, message: str, tb: str):
        self._log(tb)
        self.status_var.set('Error.')
        messagebox.showerror('LDSFL-Meander GUI error', message)

    def _log(self, text: str):
        self.after(0, lambda: self._append_log(text))

    def _append_log(self, text: str):
        self.log.insert('end', text + '\n')
        self.log.see('end')


def main() -> None:
    app = LdslGui()
    app.mainloop()


if __name__ == '__main__':
    main()
