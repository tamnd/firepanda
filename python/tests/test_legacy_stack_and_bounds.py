"""The older flat stack, a DatetimeIndex bound by text, and an IntervalIndex from one."""

import math

import pytest
from firepanda.errors import InvalidArgumentError

import firepanda as fp


def _frame():
    return fp.DataFrame({"a": [1.0, math.nan], "b": [4, 3]}, index=["p", "q"])


def test_older_stack_leaves_out_each_gap():
    out = _frame().stack(future_stack=False)
    assert out.index.tolist() == [("p", "a"), ("p", "b"), ("q", "b")]
    assert out.tolist() == [1.0, 4.0, 3.0]
    kept = _frame().stack(future_stack=False, dropna=False, sort=True)
    assert len(kept) == 4
    assert math.isnan(kept.tolist()[2])


def test_new_stack_still_refuses_dropna():
    with pytest.raises(InvalidArgumentError, match="dropna must be unspecified"):
        _frame().stack(dropna=True)


def test_datetime_bound_reads_text_as_pandas():
    stamps = fp.date_range("2024-01-01 09:00", periods=4, freq="D")
    assert stamps.get_slice_bound("2024-01-02", side="right") == 2
    assert stamps.get_slice_bound("2024-01-02", side="left") == 1
    assert stamps.get_slice_bound(fp.Timestamp("2024-01-03 09:00"), side="left") == 2
    with pytest.raises(InvalidArgumentError, match="side kwarg"):
        stamps.get_slice_bound("2024-01-02", side="up")


def test_interval_index_from_an_index_keeps_its_name_and_type():
    breaks = fp.IntervalIndex.from_breaks([0, 1, 2], name="w")
    out = fp.IntervalIndex(breaks, dtype="interval[float64, right]")
    assert out.name == "w"
    assert str(out.dtype) == "interval[int64, right]"
    assert str(fp.IntervalIndex(breaks, closed="left").dtype) == "interval[int64, left]"
    assert fp.IntervalIndex(breaks, name="z").name == "z"
