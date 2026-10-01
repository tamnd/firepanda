"""Rounding to a frequency of several pieces, numpy's short temporal names, compiled patterns."""

import re

import pytest
from firepanda.errors import InvalidArgumentError

import firepanda as fp


def test_rounding_to_several_pieces_sums_them():
    assert fp.Timestamp("2020-03-14 15:32").ceil("1h30min") == fp.Timestamp("2020-03-14 16:30")
    assert fp.Timedelta("1h10min").ceil("1h30min") == fp.Timedelta("1h30min")
    index = fp.DatetimeIndex(["2020-03-14 15:32"]).ceil("1h30min")
    assert index.tolist() == [fp.Timestamp("2020-03-14 16:30")]
    column = fp.Series(fp.to_datetime(["2020-03-14 15:32"]))
    assert column.dt.floor("1h30min").tolist() == [fp.Timestamp("2020-03-14 15:00")]
    assert column.dt.round("1D2h").tolist() == [fp.Timestamp("2020-03-14 10:00")]
    with pytest.raises(InvalidArgumentError, match="Invalid frequency"):
        fp.Timestamp("2020-03-14").ceil("1h2W")


def test_period_reads_nan_as_nat():
    assert fp.Period("nan", freq="D") is fp.NaT
    assert fp.Period("NaN", freq="D") is fp.NaT


def test_numpy_short_temporal_names_cast():
    text = fp.Series(["2015-03-29 02:30:00"])
    assert str(text.astype("M8[ns]").dtype) == "datetime64[ns]"
    assert str(fp.Series([1, 2]).astype("<m8[s]").dtype) == "timedelta64[s]"
    assert fp.Series([1, 2]).astype("m8[s]").tolist() == [fp.Timedelta("1s"), fp.Timedelta("2s")]


def test_replace_reads_a_compiled_pattern():
    s = fp.Series(["foo", "fuz"])
    assert s.str.replace(re.compile("^f.", re.IGNORECASE), "X", regex=True).tolist() == ["Xo", "Xz"]
    with pytest.raises(InvalidArgumentError, match="regex=False"):
        s.str.replace(re.compile("f"), "X")
    with pytest.raises(InvalidArgumentError, match="case and flags"):
        s.str.replace(re.compile("f"), "X", regex=True, case=False)


def test_index_value_counts_with_repeats_and_a_gap():
    counts = fp.Index([3, 1, 2, 3, 4, float("nan")]).value_counts()
    assert counts.index.tolist() == [3.0, 1.0, 2.0, 4.0]
    assert counts.tolist() == [2, 1, 1, 1]


def test_index_putmask_takes_an_index():
    out = fp.Index([1, 2, 3]).putmask([True, False, True], fp.Index([7, 8, 9]))
    assert out.tolist() == [7, 2, 9]


def test_set_operations_meet_at_a_common_width():
    union = fp.Index([1, 2], dtype="uint8").union(fp.Index([3, 4]))
    assert str(union.dtype) == "int64"
    assert union.tolist() == [1, 2, 3, 4]
    assert fp.Index([1.5, 2.0]).union(fp.Index([2, 4], dtype="int32")).tolist() == [1.5, 2.0, 4.0]
    kept = fp.Index([1, 2], dtype="int8").difference(fp.Index([2, 4], dtype="uint16"))
    assert str(kept.dtype) == "int8"
    assert kept.tolist() == [1]


def test_index_reads_nan_among_text_as_a_gap():
    index = fp.Index([float("nan"), "var1", float("nan")])
    assert str(index.dtype) == "str"
    assert list(index.isna()) == [True, False, True]
    assert str(fp.Index([float("nan"), float("nan")]).dtype) == "float64"
