from __future__ import annotations

from ldsfl.main import _criterion_names, _should_stop


def test_criterion_names_reports_enabled_reached_limits():
    reached = _criterion_names(
        steps=10,
        dt_cum=2.5,
        cut_cnt=3,
        max_steps=10,
        max_sim_time=2.0,
        max_cutoffs=3,
        stop_on_steps=True,
        stop_on_time=True,
        stop_on_cutoffs=True,
    )

    assert reached == ["max_steps", "max_sim_time", "max_cutoffs"]


def test_criterion_names_ignores_disabled_limits():
    reached = _criterion_names(
        steps=10,
        dt_cum=2.5,
        cut_cnt=3,
        max_steps=10,
        max_sim_time=2.0,
        max_cutoffs=3,
        stop_on_steps=False,
        stop_on_time=False,
        stop_on_cutoffs=True,
    )

    assert reached == ["max_cutoffs"]


def test_should_stop_false_when_no_criteria_enabled():
    assert (
        _should_stop(
            ["max_steps"],
            stop_mode="first",
            stop_on_steps=False,
            stop_on_time=False,
            stop_on_cutoffs=False,
        )
        is False
    )


def test_should_stop_first_mode_when_any_enabled_criterion_is_reached():
    assert (
        _should_stop(
            ["max_steps"],
            stop_mode="first",
            stop_on_steps=True,
            stop_on_time=True,
            stop_on_cutoffs=True,
        )
        is True
    )


def test_should_stop_all_mode_requires_all_enabled_criteria():
    assert (
        _should_stop(
            ["max_steps", "max_sim_time"],
            stop_mode="all",
            stop_on_steps=True,
            stop_on_time=True,
            stop_on_cutoffs=True,
        )
        is False
    )

    assert (
        _should_stop(
            ["max_steps", "max_sim_time", "max_cutoffs"],
            stop_mode="all",
            stop_on_steps=True,
            stop_on_time=True,
            stop_on_cutoffs=True,
        )
        is True
    )
