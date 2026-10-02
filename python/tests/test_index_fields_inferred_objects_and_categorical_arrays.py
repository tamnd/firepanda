"""Datetime index fields as numpy arrays, inferred object types and categorical arrays."""

import datetime

import numpy as np

import firepanda as pd
from firepanda import NaT


def _days():
    return pd.DatetimeIndex(["2020-01-01", None, "2021-03-31"])


def test_datetime_index_flags_are_numpy_bools_false_for_a_gap():
    out = _days().is_leap_year
    assert isinstance(out, np.ndarray)
    assert out.dtype == bool
    assert out.tolist() == [True, False, False]
    assert _days().is_month_start.tolist() == [True, False, False]
    assert _days().is_quarter_end.tolist() == [False, False, True]


def test_datetime_index_date_and_time_are_numpy_objects():
    dates = _days().date
    assert isinstance(dates, np.ndarray)
    assert dates.dtype == object
    assert dates[0] == datetime.date(2020, 1, 1)
    assert dates[1] is NaT
    times = pd.date_range("2020-01-01 10:30", periods=2, freq="h").time
    assert times.dtype == object
    assert times.tolist() == [datetime.time(10, 30), datetime.time(11, 30)]


def test_to_pydatetime_is_a_numpy_object_array():
    out = pd.date_range("2020-01-01", periods=2).to_pydatetime()
    assert out.dtype == object
    assert out[1] == datetime.datetime(2020, 1, 2)


def test_frame_infer_objects_types_object_columns():
    frame = pd.DataFrame({"a": pd.Series([1, 2], dtype="object"), "b": [1.5, 2.5]})
    assert [str(dtype) for dtype in frame.infer_objects().dtypes] == ["int64", "float64"]
    mixed = pd.DataFrame({"a": pd.Series(["x", 2], dtype="object")})
    assert str(mixed.infer_objects()["a"].dtype) == "object"


def test_index_infer_objects_types_object_labels():
    out = pd.Index([1.5, 2], dtype="object", name="n").infer_objects()
    assert str(out.dtype) == "float64"
    assert out.name == "n"
    assert str(pd.Index([1, "a"], dtype="object").infer_objects().dtype) == "object"


def test_categorical_array_is_a_categorical():
    out = pd.Series(["a", "b", "a"], dtype="category").array
    assert isinstance(out, pd.Categorical)
    assert out.categories.tolist() == ["a", "b"]
    assert isinstance(pd.CategoricalIndex(["a", "b"]).array, pd.Categorical)
    assert not isinstance(pd.Series([1, 2]).array, pd.Categorical)
