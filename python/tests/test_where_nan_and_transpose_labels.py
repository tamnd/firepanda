"""where takes NaN from the other side as NaN, and transpose keeps the row labels typed."""

import math

import pytest

import firepanda as fp

pa = pytest.importorskip("pyarrow")


def _bigger(x, y):
    return x.where(x > y, y)


def _exported_nulls(column):
    return pa.table(fp.DataFrame({"a": column})).column("a").null_count


def test_where_takes_nan_from_other():
    x = fp.Series([1.0, math.nan, 3.0, 4.0])
    y = fp.Series([9.0, 9.0, math.nan, math.nan])
    out = x.where(x > y, y)
    assert out.isna().tolist() == [False, False, True, True]
    assert _exported_nulls(out) == 0


def test_mask_and_frame_combine_take_nan():
    assert _exported_nulls(fp.Series([1.0, 2.0]).mask([False, True])) == 0
    frame = fp.DataFrame({"a": [1.0, math.nan]})
    joined = frame.combine(fp.DataFrame({"a": [math.nan, math.nan]}), _bigger)
    assert _exported_nulls(joined["a"]) == 0


def test_nullable_float_keeps_its_gap():
    out = fp.Series([1.0, 2.0], dtype="Float64").where([True, False])
    assert str(out.dtype) == "Float64"
    assert out.isna().tolist() == [False, True]
    assert _exported_nulls(out) == 1


def test_transpose_twice_keeps_typed_row_labels():
    frame = fp.DataFrame({"a": [1.0, 2.0], "b": [4, 3]})
    back = frame.T.T
    assert str(back.index.dtype) == "int64"
    assert fp.concat([frame, back]).index.tolist() == [0, 1, 0, 1]


def test_fill_across_concats_with_its_source():
    frame = fp.DataFrame({"a": [1.0, math.nan], "b": [4, 3], "s": ["x", "y"]})[["a", "b"]]
    joined = fp.concat([frame, frame.bfill(axis=1)], keys=["down", "across"])
    assert joined["b"].tolist() == [4.0, 3.0, 4.0, 3.0]
    assert joined["a"].tolist()[3] == 3.0
