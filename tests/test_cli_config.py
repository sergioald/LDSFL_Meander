from __future__ import annotations

from run_ldsfl import parse_cases


def test_parse_cases_single_case():
    assert parse_cases("1") == [1]


def test_parse_cases_list_and_range():
    assert parse_cases("1,3-5") == [1, 3, 4, 5]


def test_parse_cases_ignores_empty_parts_and_whitespace():
    assert parse_cases(" 1, ,2 ") == [1, 2]


def test_parse_cases_empty_string_returns_empty_list():
    assert parse_cases("") == []
