"""Exact parity tests for bounded batched neck-detector queries."""

from __future__ import annotations

import numpy as np
import pytest

from ldsfl import mathutils

BATCH_SIZES = (1, 3, 16, 64, 1_000)


def _separated_points(n: int) -> tuple[np.ndarray, np.ndarray]:
    return 10.0 * np.arange(n, dtype=np.float64), np.zeros(n, dtype=np.float64)


def _with_pairs(n: int, pairs: list[tuple[int, int, float]]) -> tuple[np.ndarray, np.ndarray]:
    x, y = _separated_points(n)
    for i, j, distance in pairs:
        x[j] = x[i] + distance
    return x, y


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize(
    ("x", "y", "ss", "radius", "expected"),
    [
        (*_separated_points(20), 3, 1.0, None),
        (*_with_pairs(20, [(0, 3, 0.25)]), 3, 1.0, (0, 3)),
        (*_with_pairs(20, [(8, 11, 0.25)]), 3, 1.0, (8, 11)),
        (*_with_pairs(20, [(13, 16, 0.25)]), 3, 1.0, (13, 16)),
        (*_with_pairs(20, [(2, 5, 0.4), (7, 10, 0.1)]), 3, 1.0, (2, 5)),
        (*_with_pairs(20, [(2, 6, 0.7), (2, 4, 0.2), (2, 5, 0.4)]), 2, 1.0, (2, 4)),
        (*_with_pairs(20, [(2, 5, 0.25), (2, 4, -0.25)]), 2, 1.0, (2, 4)),
        (*_with_pairs(20, [(2, 5, 1.0)]), 3, 1.0, None),
        (*_with_pairs(20, [(0, 3, np.nextafter(1.0, 0.0))]), 3, 1.0, (0, 3)),
        (*_with_pairs(20, [(0, 3, np.nextafter(1.0, np.inf))]), 3, 1.0, None),
        (*_with_pairs(12, [(3, 4, 0.2)]), 1, 1.0, (3, 4)),
        (*_with_pairs(8, [(1, 4, 0.2)]), 3, 1.0, (1, 4)),
        (*_separated_points(3), 1, 1.0, None),
        (np.zeros(12), np.zeros(12), 2, 1.0, (0, 2)),
        (0.005 * np.arange(100), 0.01 * np.sin(np.arange(100)), 3, 1.0, (0, 3)),
    ],
)
def test_batched_matches_scalar_exactly(x, y, ss, radius, expected, batch_size):
    reference = mathutils._kdtree_first_hit_point_pair_scalar(x, y, ss, radius)
    actual = mathutils._kdtree_first_hit_point_pair_batched(
        x, y, ss, radius, batch_size=batch_size
    )
    assert reference == expected
    assert actual == reference


def test_batched_accepts_positive_workers():
    x, y = _with_pairs(40, [(4, 8, 0.25)])
    assert mathutils._kdtree_first_hit_point_pair_batched(
        x, y, 4, 1.0, workers=2, batch_size=3
    ) == mathutils._kdtree_first_hit_point_pair_scalar(x, y, 4, 1.0, workers=2)


def test_production_dispatcher_uses_bounded_batch_size_32(monkeypatch):
    x, y = _with_pairs(20, [(2, 5, 0.25)])
    batch_size = mathutils._NECK_QUERY_BATCH_SIZE
    assert batch_size == 32
    assert mathutils._kdtree_first_hit_point_pair_batched.__kwdefaults__["batch_size"] == batch_size
    seen = {}

    def fake_batched(*args, **kwargs):
        seen.update(kwargs)
        return (2, 5)

    def fail_if_scalar(*_args, **_kwargs):
        raise AssertionError("production dispatcher selected the scalar reference")

    # Change the private constant temporarily to prove the dispatcher reads it
    # rather than repeating a literal production value.
    monkeypatch.setattr(mathutils, "_NECK_QUERY_BATCH_SIZE", batch_size + 1)
    monkeypatch.setattr(mathutils, "_kdtree_first_hit_point_pair_batched", fake_batched)
    monkeypatch.setattr(mathutils, "_kdtree_first_hit_point_pair_scalar", fail_if_scalar)
    assert mathutils._kdtree_first_hit_point_pair(x, y, 3, 1.0) == (2, 5)
    assert seen == {"workers": 1, "batch_size": batch_size + 1}


def test_batched_falls_back_when_scipy_lacks_return_sorted(monkeypatch):
    original_tree = mathutils.cKDTree

    class LegacyTree:
        def __init__(self, *args, **kwargs):
            self._tree = original_tree(*args, **kwargs)

        def query_ball_point(self, points, *args, **kwargs):
            if "return_sorted" in kwargs:
                raise TypeError("unexpected keyword argument 'return_sorted'")
            return self._tree.query_ball_point(points, *args, **kwargs)

    x, y = _with_pairs(24, [(3, 7, 0.25)])
    expected = mathutils._kdtree_first_hit_point_pair_scalar(x, y, 4, 1.0)
    monkeypatch.setattr(mathutils, "cKDTree", LegacyTree)
    actual = mathutils._kdtree_first_hit_point_pair_batched(
        x, y, 4, 1.0, batch_size=3
    )
    assert actual == expected


@pytest.mark.parametrize("batch_size", [0, -1])
def test_batched_rejects_nonpositive_batch_size(batch_size):
    x, y = _separated_points(10)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        mathutils._kdtree_first_hit_point_pair_batched(
            x, y, 2, 1.0, batch_size=batch_size
        )


@pytest.mark.parametrize(
    "x,y,ss,radius,expected",
    [
        (*_with_pairs(30, [(0, 4, 0.25)]), 4, 1.0, (0, 4)),  # original fast hit
        (0.4 * np.arange(30), np.zeros(30), 3, 1.0, None),  # no refine required
        (*_separated_points(30), 3, 1.0, None),  # refinement, still no hit
    ],
)
def test_public_refinement_path_matches_each_search(x, y, ss, radius, expected):
    results = []
    for helper in (
        mathutils._kdtree_first_hit_point_pair_scalar,
        mathutils._kdtree_first_hit_point_pair_batched,
    ):
        original_dispatch = mathutils._kdtree_first_hit_point_pair
        mathutils._kdtree_first_hit_point_pair = lambda *args, _helper=helper, **kwargs: _helper(
            *args, **kwargs
        )
        try:
            results.append(mathutils.find_neck_cutoff_kdtree_with_refine(x, y, ss, radius))
        finally:
            mathutils._kdtree_first_hit_point_pair = original_dispatch
    assert results == [expected, expected]


@pytest.mark.parametrize(
    "mapped_pair,original_x,ss,expected",
    [
        ((0, 2), np.array([0.0, 10.0, 0.25, 30.0, 40.0, 50.0]), 2, (0, 2)),
        ((0, 1), np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0]), 2, None),
        ((0, 2), np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0]), 2, None),
    ],
    ids=("simulated-refined-hit-maps-valid", "mapped-pair-domain-rejected", "strict-distance-rejected"),
)
def test_refinement_mapping_branches_match_with_scripted_query_results(
    monkeypatch, mapped_pair, original_x, ss, expected
):
    """Exercise mapping/revalidation branches, including synthetic-only ones.

    With a faithful original search, a refined pair mapping to an admissible
    original pair inside the strict radius is necessarily found on the first
    pass. The scripted first miss makes that otherwise unreachable path
    testable without changing production cutoff behavior.
    """
    original_y = np.zeros_like(original_x)
    refined_x = np.array([0.0, 0.1, 0.2, 0.3])
    refined_y = np.zeros_like(refined_x)
    mapping = {0: mapped_pair[0], 1: mapped_pair[1]}

    monkeypatch.setattr(
        mathutils,
        "_refine_long_segments_linear",
        lambda *_args: (
            refined_x.copy(),
            refined_y.copy(),
            np.zeros(4, dtype=np.int64),
            np.zeros(4),
        ),
    )
    monkeypatch.setattr(mathutils, "_map_refined_idx_to_original", lambda idx, *_: mapping[idx])

    for helper_name in (
        "_kdtree_first_hit_point_pair_scalar",
        "_kdtree_first_hit_point_pair_batched",
    ):
        calls = 0

        def scripted(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            return None if calls == 1 else (0, 1)

        monkeypatch.setattr(mathutils, helper_name, scripted)
        monkeypatch.setattr(
            mathutils,
            "_kdtree_first_hit_point_pair",
            lambda *args, _name=helper_name, **kwargs: getattr(mathutils, _name)(
                *args, **kwargs
            ),
        )
        result = mathutils.find_neck_cutoff_kdtree_with_refine(
            original_x, original_y, ss, 1.0
        )
        assert calls == 2
        assert result == expected
