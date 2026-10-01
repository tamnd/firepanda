"""to_numeric's dtype_backend, align's fill against a column, concat levels and integer-na."""

import math

import pytest
from firepanda.errors import InvalidArgumentError

import firepanda as fp


def test_to_numeric_answers_the_nullable_kind_read():
    out = fp.to_numeric(
        fp.Series(["1", None, "x"]), errors="coerce", dtype_backend="numpy_nullable"
    )
    assert str(out.dtype) == "Int64"
    assert out.tolist()[0] == 1
    assert str(fp.to_numeric(fp.Series(["1.5"]), dtype_backend="numpy_nullable").dtype) == "Float64"
    assert str(fp.to_numeric(fp.Series([1.0, 2.0]), dtype_backend="numpy_nullable").dtype) == (
        "Float64"
    )
    assert str(fp.to_numeric(fp.Series(["1.0", "2"]), dtype_backend="numpy_nullable").dtype) == (
        "Float64"
    )
    flags = fp.to_numeric(fp.Series([True, None], dtype="object"), dtype_backend="numpy_nullable")
    assert str(flags.dtype) == "boolean"


def test_to_numeric_answers_the_arrow_kind_read():
    assert str(fp.to_numeric(fp.Series(["1", "2"]), dtype_backend="pyarrow").dtype) == (
        "int64[pyarrow]"
    )
    assert str(fp.to_numeric(fp.Series([1, 2], dtype="int32"), dtype_backend="pyarrow").dtype) == (
        "int32[pyarrow]"
    )
    flags = fp.to_numeric(fp.Series([True, None], dtype="object"), dtype_backend="pyarrow")
    assert str(flags.dtype) == "bool[pyarrow]"


def test_to_numeric_keeps_a_nullable_column_and_downcasts_it():
    assert str(fp.to_numeric(fp.Series([1, None], dtype="Int64")).dtype) == "Int64"
    small = fp.to_numeric(fp.Series([1, 2]), downcast="integer", dtype_backend="numpy_nullable")
    assert str(small.dtype) == "Int8"
    assert str(fp.to_numeric(fp.Series([1, 2], dtype="Int64"), dtype_backend="pyarrow").dtype) == (
        "int64[pyarrow]"
    )
    assert str(fp.to_numeric(fp.Index(["1", "2"]), dtype_backend="numpy_nullable").dtype) == "Int64"
    with pytest.raises(InvalidArgumentError, match="dtype_backend x is invalid"):
        fp.to_numeric(fp.Series(["1"]), dtype_backend="x")


def test_align_against_a_column_fills_every_gap():
    frame = fp.DataFrame({"a": [1.0, float("nan"), 3.0], "b": [4, 3, 2]})
    left, right = frame.align(fp.Series([1, 2], index=["a", "z"]), axis=1, fill_value=0)
    assert left["a"].tolist() == [1.0, 0.0, 3.0]
    assert str(left["z"].dtype) == "float64"
    assert right.tolist() == [1.0, 0.0, 2.0]
    left, right = frame.align(fp.Series([float("nan"), 2.0]), axis=0, join="left", fill_value=7)
    assert left["a"].tolist() == [1.0, 7.0, 3.0]
    assert right.tolist() == [7.0, 2.0, 7.0]


def test_align_of_two_frames_fills_only_new_places():
    frame = fp.DataFrame({"a": [1.0, float("nan")]})
    left, _ = frame.align(fp.DataFrame({"a": [1.0], "b": [2.0]}), fill_value=5)
    assert math.isnan(left["a"].tolist()[1])
    assert left["b"].tolist() == [5.0, 5.0]


def test_concat_levels_beside_keys():
    s = fp.Series([1, 2])
    made = fp.concat([s, s], keys=["p", "q"], levels=[["q", "p", "r"]])
    assert made.index.tolist() == [("p", 0), ("p", 1), ("q", 0), ("q", 1)]
    with pytest.raises(InvalidArgumentError, match="Values not found in passed level"):
        fp.concat([s, s], keys=["p", "q"], levels=[["q"]])


def test_integer_na_and_scalar_to_numpy():
    nan = float("nan")
    assert fp.api.types.infer_dtype([1, nan], skipna=False) == "integer-na"
    assert fp.api.types.infer_dtype([1, None], skipna=False) == "mixed-integer"
    assert fp.api.types.infer_dtype([1, 2.5, nan], skipna=False) == "mixed-integer-float"
    with pytest.raises(InvalidArgumentError, match=r"Timestamp\.to_numpy dtype and copy"):
        fp.Timestamp("2024-01-01").to_numpy(copy=True)
    with pytest.raises(InvalidArgumentError, match=r"Timedelta\.to_numpy dtype and copy"):
        fp.Timedelta("1s").to_numpy(dtype="timedelta64[ms]")
