"""Comparing two series on different labels raises pandas' ValueError."""

import pytest

import firepanda as pd


@pytest.mark.parametrize("op", ["__eq__", "__ne__", "__lt__", "__ge__"])
def test_misaligned_compare_raises(op):
    left = pd.Series([1, 2, 3], index=["x", "y", "z"])
    right = pd.Series([3, 2, 1], index=["z", "y", "x"])
    with pytest.raises(ValueError, match="Can only compare identically-labeled Series objects"):
        getattr(left, op)(right)


def test_named_method_lines_up():
    left = pd.Series([1, 2], index=["x", "y"])
    assert left.eq(pd.Series([1], index=["x"])).tolist() == [True, False]
