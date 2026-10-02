"""Time indexers and temporal values as numpy, typed level names and mixed column names."""

import numpy as np

import firepanda as pd


def _hours():
    return pd.date_range("2020-01-01 00:00", periods=4, freq="5h")


def test_time_indexers_are_numpy_positions():
    at = _hours().indexer_at_time("10:00")
    assert isinstance(at, np.ndarray)
    assert at.tolist() == [2]
    between = _hours().indexer_between_time("00:00", "06:00", include_end=False)
    assert isinstance(between, np.ndarray)
    assert between.tolist() == [0, 1]
    assert pd.Series(range(4), index=_hours()).at_time("10:00").tolist() == [2]


def test_datetime_and_timedelta_values_are_numpy():
    days = pd.date_range("2020-01-01", periods=2).values
    assert days.dtype.kind == "M"
    zoned = pd.date_range("2020-01-01", periods=1, tz="US/Eastern").values
    assert zoned.dtype.kind == "M"
    assert str(zoned[0]).startswith("2020-01-01T05:00")
    spans = pd.to_timedelta(["1D", "2D"]).values
    assert spans.dtype.kind == "m"


def test_multiindex_dtypes_name_unnamed_levels():
    out = pd.MultiIndex.from_arrays([[1, 2], ["a", "b"]]).dtypes
    assert out.index.tolist() == ["level_0", "level_1"]
    assert str(out.dtype) == "object"


def test_isocalendar_is_uint32_of_pandas():
    out = pd.Series(pd.to_datetime(["2019-12-29", None])).dt.isocalendar()
    assert [str(dtype) for dtype in out.dtypes] == ["UInt32"] * 3
    assert out["week"].tolist()[0] == 52


def test_asof_gives_a_numpy_number():
    out = pd.Series([1.0, 2.0, None, 4.0], index=[10, 20, 30, 40]).asof(20)
    assert isinstance(out, np.float64)
    assert out == 2.0


def test_index_names_keep_their_type():
    assert pd.Index([1], name=5).name == 5
    assert pd.Index([1], name=("x", "y")).name == ("x", "y")
    assert pd.Index([1], name=("x", "y")).names == [("x", "y")]
    series = pd.Series([1, 2], index=pd.Index([3, 4], name=7))
    assert series.index.name == 7


def test_reset_index_names_columns_by_value():
    series = pd.Series([1, 2], index=pd.Index([3, 4], name=7))
    assert series.reset_index().columns.tolist() == [7, 0]
    assert pd.Series([1]).reset_index().columns.tolist() == ["index", 0]
    assert pd.Series([1], name="v").reset_index(name=3).columns.tolist() == ["index", 3]


def test_mixed_column_names_are_an_object_index():
    frame = pd.DataFrame({7: [1, 2], "a": [3, 4]})
    assert frame.columns.tolist() == [7, "a"]
    assert str(frame.columns.dtype) == "object"
    assert frame[7].tolist() == [1, 2]
    reset = pd.DataFrame({"a": [1]}, index=pd.Index([3], name=7)).reset_index()
    assert reset.columns.tolist() == [7, "a"]
