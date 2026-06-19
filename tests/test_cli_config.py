from __future__ import annotations

import argparse

import pytest

from run_ldsfl import parse_cases


def test_parse_cases_single_case():
    assert parse_cases("1") == [1]


def test_parse_cases_list_and_range():
    assert parse_cases("1,3-5") == [1, 3, 4, 5]


def test_parse_cases_ignores_empty_parts_and_whitespace():
    assert parse_cases(" 1, ,2 ") == [1, 2]


def test_parse_cases_deduplicates_preserving_order():
    assert parse_cases("1,2,1,2-3") == [1, 2, 3]


@pytest.mark.parametrize("value", ["", "0", "2-1", "a"])
def test_parse_cases_rejects_invalid_selections(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_cases(value)
