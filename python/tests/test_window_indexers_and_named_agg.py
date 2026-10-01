"""Window indexers, named aggregation, factorize of numpy arrays, option dicts, slash dates."""

import math

import numpy as np
import pytest
from firepanda.api.indexers import (
    BaseIndexer,
    FixedForwardWindowIndexer,
    VariableOffsetWindowIndexer,
    check_array_indexer,
)
from firepanda.errors import InvalidArgumentError

import firepanda as fp


def _same(found, expected):
    assert len(found) == len(expected)
    for a, b in zip(found, expected, strict=True):
        assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b)


def test_forward_window_reductions():
    s = fp.Series([1.0, 2.0, float("nan"), 4.0, 5.0])
    ahead = FixedForwardWindowIndexer(window_size=2)
    nan = float("nan")
    _same(s.rolling(ahead).sum().tolist(), [3.0, nan, nan, 9.0, nan])
    _same(s.rolling(ahead, min_periods=1).sum().tolist(), [3.0, 2.0, 4.0, 9.0, 5.0])
    _same(s.rolling(ahead).count().tolist(), [2.0, 1.0, 1.0, 2.0, nan])
    _same(s.rolling(ahead, min_periods=1).max().tolist(), [2.0, 2.0, 4.0, 5.0, 5.0])
    wide = FixedForwardWindowIndexer(window_size=3)
    _same(
        s.rolling(wide, min_periods=2).std().tolist(),
        [0.7071067811865476, 1.4142135623730951, 0.7071067811865476, 0.7071067811865476, nan],
    )
    doubled = s.rolling(ahead, min_periods=1).apply(lambda w: w.sum() * 2, raw=True)
    _same(doubled.tolist(), [6.0, nan, nan, 18.0, 10.0])
    frame = fp.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert frame.rolling(ahead, min_periods=1).sum()["b"].tolist() == [9.0, 11.0, 6.0]
    with pytest.raises(InvalidArgumentError, match="can't have center=True"):
        s.rolling(ahead, center=True).sum()


def test_custom_and_offset_indexers():
    class Every(BaseIndexer):
        def get_window_bounds(self, num_values, min_periods, center, closed, step):
            start = np.arange(num_values, dtype=np.int64)
            start[1::2] = np.maximum(start[1::2] - self.window_size, 0)
            return start, np.arange(1, num_values + 1, dtype=np.int64)

    s = fp.Series([1.0, 2.0, float("nan"), 4.0, 5.0])
    _same(
        s.rolling(Every(window_size=2), min_periods=1).sum().tolist(),
        [1.0, 3.0, float("nan"), 6.0, 5.0],
    )
    index = fp.date_range("2020-01-01", periods=6, freq="D")
    days = fp.Series(range(6), index=index, dtype="float64")
    business = VariableOffsetWindowIndexer(index=index, offset=fp.offsets.BDay(1))
    assert days.rolling(business, min_periods=1).sum().tolist() == [0.0, 1.0, 2.0, 3.0, 7.0, 12.0]
    with pytest.raises(InvalidArgumentError, match="DateOffset-like"):
        VariableOffsetWindowIndexer(index=index, offset=fp.Timedelta("1D"))
    with pytest.raises(NotImplementedError):
        BaseIndexer(window_size=2).get_window_bounds(3)


def test_check_array_indexer():
    arr = fp.array([1, 2, 3])
    assert check_array_indexer(arr, [True, False, True]).tolist() == [True, False, True]
    assert check_array_indexer(arr, [0, 2]).tolist() == [0, 2]
    flags = fp.array([True, None, True], dtype="boolean")
    assert check_array_indexer(arr, flags).tolist() == [True, False, True]
    assert check_array_indexer(arr, 1) == 1
    with pytest.raises(IndexError, match="wrong length: 2 instead of 3"):
        check_array_indexer(arr, [True, False])
    with pytest.raises(IndexError, match="integer or boolean"):
        check_array_indexer(arr, [1.5])


def test_named_aggregation():
    frame = fp.DataFrame({"A": [1, 2, 3], "B": [4.0, 5.0, 6.0], "C": [7, 8, 10]})
    out = frame.agg(y=("C", "min"), x=("A", "max"), z=("C", "max"))
    assert out.index.tolist() == ["y", "x", "z"]
    assert list(out.columns) == ["C", "A"]
    assert out["C"].tolist()[0] == 7.0
    assert math.isnan(out["A"].tolist()[0])
    assert frame.agg(x=("A", "max"))["A"].tolist() == [3]
    assert fp.Series([1, 2, 3]).agg(x="max", y="min").tolist() == [3, 1]
    with pytest.raises(TypeError, match="Must provide 'func'"):
        frame.agg()


def test_numpy_arrays_factorize_and_unique():
    codes, uniques = fp.factorize(np.array(["b", "b", "a", "c", "b"], dtype="O"))
    assert codes.tolist() == [0, 0, 1, 2, 0]
    assert uniques.dtype == object
    assert uniques.tolist() == ["b", "a", "c"]
    codes, uniques = fp.factorize(np.array(["b", "a", "b"], dtype="O"), sort=True)
    assert codes.tolist() == [1, 0, 1]
    assert fp.unique(np.array([3, 1, 3])).tolist() == [3, 1]


def test_option_dict_and_slash_dates():
    fp.set_option({"display.max_columns": 4, "display.precision": 1})
    try:
        assert fp.get_option("display.max_columns") == 4
        assert fp.get_option("display.precision") == 1
    finally:
        fp.reset_option("display.max_columns")
        fp.reset_option("display.precision")
    assert fp.Timestamp("1/2/2018") == fp.Timestamp("2018-01-02")
    assert fp.Timestamp("2018/01/02 10:30") == fp.Timestamp("2018-01-02 10:30")
    assert len(fp.bdate_range(start="1/1/2018", end="1/08/2018")) == 6
