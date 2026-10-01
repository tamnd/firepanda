"""Spans past the nanosecond range, held at a coarser unit as pandas holds them."""

import firepanda as fp


def test_timedelta_range_in_seconds_past_the_nanosecond_range():
    out = fp.timedelta_range("1 Day", periods=3, freq="100000D", unit="s")
    assert str(out.dtype) == "timedelta64[s]"
    assert [span.days for span in out] == [1, 100001, 200001]


def test_whole_numbers_cast_to_seconds_past_the_nanosecond_range():
    out = fp.Series([86400, 8640086400], dtype="int64").astype("timedelta64[s]")
    assert str(out.dtype) == "timedelta64[s]"
    assert [span.days for span in out] == [1, 100001]


def test_spans_inside_the_range_keep_their_unit():
    assert str(fp.to_timedelta(["1 day", "2h"]).dtype) == "timedelta64[us]"
    assert str(fp.to_timedelta([1.5, 2], unit="s").dtype) == "timedelta64[ns]"
