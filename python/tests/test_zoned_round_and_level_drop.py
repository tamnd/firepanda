"""Zoned rounding policies, fill limits along a MultiIndex, and drops by one level."""

import datetime

import pytest
from firepanda.errors import InvalidArgumentError

import firepanda as fp


def test_rounding_onto_a_repeated_hour_follows_ambiguous():
    moment = fp.Timestamp("2024-11-03 00:31", tz="US/Eastern")
    assert str(moment.ceil("h", ambiguous=True)) == "2024-11-03 01:00:00-04:00"
    assert str(moment.ceil("h", ambiguous=False)) == "2024-11-03 01:00:00-05:00"
    assert moment.ceil("h", ambiguous="NaT") is fp.NaT
    with pytest.raises(InvalidArgumentError, match="Cannot infer dst time"):
        moment.ceil("h")


def test_rounding_onto_a_missing_hour_follows_nonexistent():
    moment = fp.Timestamp("2024-03-10 03:30", tz="US/Eastern")
    assert str(moment.floor("2h", nonexistent="shift_forward")) == "2024-03-10 03:00:00-04:00"
    assert str(moment.floor("2h", nonexistent="shift_backward")) == (
        "2024-03-10 01:59:59.999999-05:00"
    )
    with pytest.raises(InvalidArgumentError, match="nonexistent time"):
        moment.floor("2h")
    with pytest.raises(InvalidArgumentError, match="relocalize"):
        moment.floor("2h", nonexistent=datetime.timedelta(minutes=10))


def test_plain_zoned_rounding_keeps_the_wall_clock():
    moment = fp.Timestamp("2024-07-01 13:45", tz="Europe/Paris")
    assert str(moment.floor("D")) == "2024-07-01 00:00:00+02:00"
    with pytest.raises(InvalidArgumentError, match="'ambiguous' parameter"):
        moment.round("h", ambiguous="x")


def test_fill_limit_along_a_multi_index():
    index = fp.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)])
    target = [("a", 0), ("a", 1), ("a", 3), ("a", 4), ("b", 1), ("b", 2), ("b", 3)]
    assert list(index.get_indexer(target, method="pad", limit=1)) == [-1, 0, 1, -1, 2, 2, -1]
    assert list(index.get_indexer(target, method="bfill", limit=1)) == [0, 0, -1, 2, 2, -1, -1]
    with pytest.raises(InvalidArgumentError, match="greater than 0"):
        index.get_indexer(target, method="pad", limit=0)
    with pytest.raises(InvalidArgumentError, match="monotonic"):
        index.get_indexer([("b", 0), ("a", 3)], method="pad", limit=1)


def test_series_reindex_fills_along_a_multi_index():
    index = fp.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)])
    series = fp.Series([1, 2, 3], index=index)
    target = fp.MultiIndex.from_tuples([("a", 3), ("a", 4), ("b", 5)])
    out = series.reindex(target, method="ffill", limit=1)
    assert out.tolist()[0] == 2.0
    assert out.tolist()[2] == 3.0


def test_drop_by_level():
    index = fp.MultiIndex.from_tuples([("a", 1), ("b", 2), ("a", 3)], names=["k", "n"])
    series = fp.Series([1, 2, 3], index=index)
    assert series.drop("a", level=0).tolist() == [2]
    assert series.drop([2, 3], level="n").tolist() == [1]
    assert series.drop("z", level=0, errors="ignore").tolist() == [1, 2, 3]
    with pytest.raises(KeyError, match="not found in level"):
        series.drop(["a", "z"], level="k")
    with pytest.raises(AssertionError, match="must be a MultiIndex"):
        fp.Series([1, 2]).drop(0, level=0)


def test_drop_by_level_on_repeated_rows_and_columns():
    repeated = fp.Series([1, 2, 3], index=fp.MultiIndex.from_tuples([("a", 1), ("a", 1), ("b", 2)]))
    assert repeated.drop("a", level=0).tolist() == [3]
    with pytest.raises(KeyError, match="not found in axis"):
        repeated.drop("z", level=0)
    columns = fp.MultiIndex.from_tuples([("a", "p"), ("a", "q"), ("b", "p")])
    frame = fp.DataFrame([[1, 2, 3]], columns=columns)
    assert frame.drop("p", axis=1, level=1).columns.tolist() == [("a", "q")]
