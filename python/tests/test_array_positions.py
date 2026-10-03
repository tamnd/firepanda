"""The methods every pandas array has for picking, dropping and placing values.

`take`, `delete`, `insert`, `repeat` and `equals` are on pandas' ExtensionArray,
so every array has them and keeps its type through them. `reshape`, `swapaxes`
and the reductions are only on the kinds pandas gives them to, and an axis past
the one there is gets numpy's AxisError on the kinds pandas keeps in numpy.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

_BELOW = r"^'indices' contains values less than allowed \(-2 < -1\)$"


def test_take_picks_and_fills(firepanda: ModuleType) -> None:
    """-1 counts from the end, or is a gap or `fill_value` when `allow_fill` is set."""
    values = firepanda.array([1, None, 3])
    assert values.take([2, 0, -1]).tolist()[:2] == [3, 1]
    assert type(values.take([0])).__name__ == "IntegerArray"
    filled = values.take([0, -1], allow_fill=True)
    assert filled[0] == 1 and filled.isna().tolist() == [False, True]
    assert values.take([-1], allow_fill=True, fill_value=9).tolist() == [9]
    assert values.take(np.array([2, 0])).tolist() == [3, 1]
    with pytest.raises(ValueError, match=_BELOW):
        values.take([-2], allow_fill=True)
    with pytest.raises(IndexError, match=r"^indices are out-of-bounds$"):
        values.take([5], allow_fill=True)
    with pytest.raises(IndexError, match=r"^index 5 is out of bounds for axis 0 with size 3$"):
        values.take([5])
    with pytest.raises(IndexError, match=r"^out of bounds value in 'indices'\.$"):
        firepanda.array(["a"]).take([4])


def test_take_with_a_gap_on_numpy_numbers_gives_floats(firepanda: ModuleType) -> None:
    """A numpy type has no gap, so pandas lets a NaN pick the type."""
    numbers = firepanda.arrays.NumpyExtensionArray(np.array([1, 2]))
    filled = numbers.take([0, -1], allow_fill=True)
    assert str(filled.dtype) == "float64"
    assert np.isnan(filled.tolist()[1])


def test_delete_insert_and_repeat(firepanda: ModuleType) -> None:
    """Each is numpy's on the positions, and keeps the array's type."""
    values = firepanda.array([1, None, 3])
    assert values.delete(0).tolist()[1] == 3
    assert values.delete([0, 2]).isna().tolist() == [True]
    with pytest.raises(IndexError, match=r"^index 5 is out of bounds for axis 0 with size 3$"):
        values.delete(5)
    assert values.insert(-1, 9).tolist()[2:] == [9, 3]
    assert values.insert(3, 9).tolist()[3] == 9
    with pytest.raises(IndexError, match=r"^loc must be an integer between -3 and 3$"):
        values.insert(5, 1)
    with pytest.raises(TypeError, match=r"^Invalid value '1' for dtype 'str'"):
        firepanda.array(["a", "b"]).insert(0, 1)
    assert firepanda.array(["a", "b"]).repeat(2).tolist() == ["a", "a", "b", "b"]
    assert firepanda.array([1, 2, 3]).repeat([1, 0, 2]).tolist() == [1, 3, 3]
    with pytest.raises(ValueError, match=r"^operands could not be broadcast together"):
        values.repeat([1, 2])
    with pytest.raises(ValueError, match=r"^the 'axis' parameter is not supported"):
        values.repeat(2, axis=1)


def test_equals_wants_the_same_type(firepanda: ModuleType) -> None:
    """Gaps in the same places are equal; another type or a list is not."""
    values = firepanda.array([1, None, 3])
    assert values.equals(values.copy())
    assert not values.equals(firepanda.array([1, None, 3], dtype="Int32"))
    assert not values.equals(firepanda.array([1, 2, 3]))
    assert not values.equals([1, None, 3])


def test_shape_methods(firepanda: ModuleType) -> None:
    """One dimension is its own transpose; numpy kinds check the axes they are given."""
    values = firepanda.array([1, None, 3])
    assert values.ravel().equals(values)
    assert values.T.equals(values)
    assert values.transpose(1).equals(values)
    assert values.reshape(-1).equals(values)
    assert values.reshape((3,)).equals(values)
    with pytest.raises(ValueError, match=r"^cannot reshape array of size 3 into shape \(2,\)$"):
        values.reshape(2)
    with pytest.raises(np.exceptions.AxisError, match=r"^axis2: axis 1 is out of bounds"):
        values.swapaxes(0, 1)
    numbers = firepanda.arrays.NumpyExtensionArray(np.array([1.5, 2.0]))
    with pytest.raises(np.exceptions.AxisError, match=r"^axis 1 is out of bounds"):
        numbers.transpose(1)
    assert not hasattr(firepanda.array(["a"]), "reshape")


def test_reductions_by_kind(firepanda: ModuleType) -> None:
    """Masked and numpy numbers reduce; a masked gap leaves `all` unknown without skipna."""
    numbers = firepanda.arrays.NumpyExtensionArray(np.array([1, 2, 3]))
    assert numbers.prod() == 6
    assert numbers.median() == 2.0
    assert numbers.std() == 1.0 and numbers.var() == 1.0
    assert numbers.sem() == pytest.approx(0.5773502691896258)
    assert numbers.skew() == 0.0
    assert bool(numbers.all())
    flags = firepanda.array([True, None])
    assert flags.all(skipna=False) is firepanda.NA
    assert bool(flags.any(skipna=False))
    assert firepanda.array([1.25, None], dtype="Float64").round(1).tolist()[0] == 1.2
    assert firepanda.array([1, None, 3]).std() == pytest.approx(1.4142135623730951)
    spans = firepanda.array(firepanda.to_timedelta(["1D", "0s"]))
    assert not bool(spans.all()) and bool(spans.any())
    assert spans.median() == firepanda.Timedelta("12h")


def test_a_period_median(firepanda: ModuleType) -> None:
    """The middle period, the earlier one for an even count, as pandas rounds it."""
    months = firepanda.array(firepanda.period_range("2024-01", periods=3, freq="M"))
    assert str(months.median()) == "2024-02"
    four = firepanda.array(firepanda.period_range("2024-01", periods=4, freq="M"))
    assert str(four.median()) == "2024-02"


def test_categorical_positions_keep_the_categories(firepanda: ModuleType) -> None:
    """A value not among the categories is refused, as `__setitem__` refuses it."""
    values = firepanda.Categorical(["a", "b", None, "a"], categories=["b", "a", "z"])
    assert values.insert(1, "z").tolist()[:3] == ["a", "z", "b"]
    assert values.take([0, -1], allow_fill=True, fill_value="z").tolist() == ["a", "z"]
    assert values.delete([0, 1]).categories.tolist() == ["b", "a", "z"]
    assert values.repeat(2).tolist()[:4] == ["a", "a", "b", "b"]
    assert values.equals(values.copy())
    assert type(values.copy()).__name__ == "Categorical"
    with pytest.raises(TypeError, match=r"new category \(q\), set the categories first$"):
        values.insert(1, "q")
    with pytest.raises(TypeError, match=r"new category \(q\)"):
        values.take([0, -1], allow_fill=True, fill_value="q")
    with pytest.raises(np.exceptions.AxisError, match=r"^axis 1 is out of bounds"):
        values.transpose(1)


def test_categorical_extras(firepanda: ModuleType) -> None:
    """The order checks, the gap mask and the counts table pandas' Categorical has."""
    values = firepanda.Categorical(["a", "b", None, "a"], categories=["b", "a", "z"])
    with pytest.raises(TypeError, match=r"^Categorical is not ordered for operation min\n"):
        values.check_for_ordered("min")
    assert values.as_ordered().check_for_ordered("min") is None
    assert values.set_ordered(True).ordered
    with pytest.raises(TypeError, match=r"^'ordered' must either be 'True' or 'False'$"):
        values.set_ordered(1)
    assert values.notna().tolist() == [True, True, False, True]
    assert values.notnull().tolist() == [True, True, False, True]
    table = values.describe()
    assert table["counts"].tolist() == [1, 2, 0, 1]
    assert table["freqs"].tolist() == [0.25, 0.5, 0.0, 0.25]
    assert table.index.name == "categories"
    assert values.memory_usage() > 4
