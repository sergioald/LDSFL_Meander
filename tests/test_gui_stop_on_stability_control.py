"""Source-level guards for the GUI sinuosity-stability stop control."""

from __future__ import annotations

from pathlib import Path


SOURCE = Path("gui_ldsfl.py").read_text(encoding="utf-8")


def test_gui_defines_sinuosity_stability_stop_variable():
    assert "self.stop_on_sinuosity_stability_var = tk.BooleanVar(value=False)" in SOURCE


def test_gui_exposes_sinuosity_stability_stop_checkbox():
    assert "Enable sinuosity stability stop" in SOURCE
    assert "HELP['stop_on_sinuosity_stability']" in SOURCE


def test_gui_build_config_reads_sinuosity_stability_stop_state():
    assert (
        "stop_on_sinuosity_stability=bool(self.stop_on_sinuosity_stability_var.get()),"
        in SOURCE
    )


def test_gui_apply_config_restores_sinuosity_stability_stop_state():
    assert (
        "self.stop_on_sinuosity_stability_var.set(bool(getattr(cfg.run, 'stop_on_sinuosity_stability', False)))"
        in SOURCE
    )
