"""Fields pandas gives a period index, an interval index or array, and a column's group by.

`asi8`, `is_full` and `resolution` read a period index's ordinals and unit, the
closed flags and `to_tuples` read an interval's ends, and `SeriesGroupBy.dtype`
gives the column's type once for each group.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest


def test_asi8_and_is_full(firepanda: ModuleType) -> None:
    """Ordinals with the least int64 for a gap; no ordinal skipped means full."""
    labels = firepanda.PeriodIndex(["2024-01", None, "2024-03"], freq="M")
    assert labels.asi8.tolist() == [648, np.iinfo(np.int64).min, 650]
    assert firepanda.PeriodIndex(["2024-01", "2024-02", "2024-02"], freq="M").is_full
    assert not firepanda.PeriodIndex(["2024-01", "2024-03"], freq="M").is_full
    assert firepanda.PeriodIndex([], freq="M").is_full
    with pytest.raises(ValueError, match=r"^Index is not monotonic$"):
        _ = firepanda.PeriodIndex(["2024-03", "2024-01"], freq="M").is_full


@pytest.mark.parametrize(
    ("freq", "unit"),
    [("M", "month"), ("D", "day"), ("h", "hour"), ("Q", "quarter"), ("Y", "year")],
)
def test_resolution_names_the_unit(firepanda: ModuleType, freq: str, unit: str) -> None:
    """The unit a frequency counts in, by pandas' name for it."""
    assert firepanda.period_range("2024-01-01", periods=2, freq=freq).resolution == unit


def test_resolution_refuses_a_week_and_a_multiple(firepanda: ModuleType) -> None:
    """pandas has no unit for a week or for a frequency counting by more than one."""
    with pytest.raises(ValueError, match=r"^Invalid frequency: W-SUN$"):
        _ = firepanda.period_range("2024-01-01", periods=2, freq="W").resolution
    with pytest.raises(ValueError, match=r"^Invalid frequency: 2M$"):
        _ = firepanda.period_range("2024-01", periods=2, freq="2M").resolution


@pytest.mark.parametrize(
    ("closed", "flags"),
    [
        ("left", (True, False, False, True)),
        ("right", (False, True, True, False)),
        ("both", (True, True, False, False)),
        ("neither", (False, False, True, True)),
    ],
)
def test_closed_flags(firepanda: ModuleType, closed: str, flags: tuple[bool, ...]) -> None:
    """Each end held or left out, the same on an index and on an array."""
    for owner in (
        firepanda.IntervalIndex.from_breaks([0, 1, 2], closed=closed),
        firepanda.arrays.IntervalArray.from_breaks([0, 1, 2], closed=closed),
    ):
        assert (owner.closed_left, owner.closed_right, owner.open_left, owner.open_right) == flags


def test_to_tuples(firepanda: ModuleType) -> None:
    """Pairs of ends, a gap a pair of NaN or one NaN with `na_tuple` off."""
    plain = firepanda.IntervalIndex.from_breaks([0, 1, 2]).to_tuples()
    assert plain.tolist() == [(0, 1), (1, 2)]
    assert str(plain.dtype) == "object"
    gap = firepanda.IntervalIndex([firepanda.Interval(0, 1), None])
    pairs = gap.to_tuples().tolist()
    assert pairs[0] == (0.0, 1.0) and all(np.isnan(end) for end in pairs[1])
    assert np.isnan(gap.to_tuples(na_tuple=False).tolist()[1])
    values = gap.array.to_tuples()
    assert isinstance(values, np.ndarray) and values.dtype == object
    assert values[0] == (0.0, 1.0)
    assert firepanda.arrays.IntervalArray.can_hold_na


def test_a_column_group_by_dtype(firepanda: ModuleType) -> None:
    """The column's type for each group, an object series named after the column."""
    frame = firepanda.DataFrame({"a": [1, 1, 2], "b": [1.5, 2.5, 3.5]})
    kinds = frame.groupby("a")["b"].dtype
    assert kinds.index.tolist() == [1, 2]
    assert [str(kind) for kind in kinds.tolist()] == ["float64", "float64"]
    assert kinds.name == "b"
    assert str(kinds.dtype) == "object"


def test_a_numpy_array_repeats_along_its_one_axis(firepanda: ModuleType) -> None:
    """numpy repeats these, so axis 0 is taken and any other is numpy's AxisError."""
    numbers = firepanda.Series([1, 2]).array
    assert numbers.repeat(2, axis=0).tolist() == [1, 1, 2, 2]
    with pytest.raises(np.exceptions.AxisError, match=r"^axis 1 is out of bounds"):
        numbers.repeat(2, axis=1)
    with pytest.raises(ValueError, match=r"^the 'axis' parameter is not supported"):
        firepanda.array([1, 2], dtype="Int64").repeat(2, axis=0)
