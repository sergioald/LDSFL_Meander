from __future__ import annotations

from ldsfl.main import _criterion_names, _should_stop


def test_criterion_names_reports_sinuosity_stability_when_enabled():
    reached = _criterion_names(
        steps=10,
        dt_cum=0.0,
        cut_cnt=0,
        max_steps=0,
        max_sim_time=0.0,
        max_cutoffs=0,
        stop_on_steps=False,
        stop_on_time=False,
        stop_on_cutoffs=False,
        stop_on_sinuosity_stability=True,
        sinuosity_stability_reached=True,
    )

    assert reached == ["sinuosity_stability"]


def test_criterion_names_ignores_sinuosity_stability_when_disabled():
    reached = _criterion_names(
        steps=10,
        dt_cum=0.0,
        cut_cnt=0,
        max_steps=0,
        max_sim_time=0.0,
        max_cutoffs=0,
        stop_on_steps=False,
        stop_on_time=False,
        stop_on_cutoffs=False,
        stop_on_sinuosity_stability=False,
        sinuosity_stability_reached=True,
    )

    assert reached == []


def test_should_stop_with_only_sinuosity_stability_enabled():
    assert (
        _should_stop(
            ["sinuosity_stability"],
            stop_mode="first",
            stop_on_steps=False,
            stop_on_time=False,
            stop_on_cutoffs=False,
            stop_on_sinuosity_stability=True,
        )
        is True
    )


def test_should_stop_all_mode_counts_sinuosity_stability():
    assert (
        _should_stop(
            ["max_steps"],
            stop_mode="all",
            stop_on_steps=True,
            stop_on_time=False,
            stop_on_cutoffs=False,
            stop_on_sinuosity_stability=True,
        )
        is False
    )

    assert (
        _should_stop(
            ["max_steps", "sinuosity_stability"],
            stop_mode="all",
            stop_on_steps=True,
            stop_on_time=False,
            stop_on_cutoffs=False,
            stop_on_sinuosity_stability=True,
        )
        is True
    )
