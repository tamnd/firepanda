"""A reduction or a cell of a numpy typed column is numpy's scalar, as in pandas."""

import numpy
import pytest

import firepanda as fp


@pytest.mark.parametrize(
    ("dtype", "kind", "expected"),
    [
        ("int64", "sum", numpy.int64),
        ("int32", "sum", numpy.int64),
        ("int32", "min", numpy.int32),
        ("uint8", "sum", numpy.uint64),
        ("uint8", "max", numpy.uint8),
        ("int64", "mean", numpy.float64),
        ("float32", "mean", numpy.float32),
        ("float32", "quantile", numpy.float64),
        ("float64", "std", numpy.float64),
        ("bool", "sum", numpy.int64),
        ("bool", "max", numpy.bool_),
        ("Int64", "sum", numpy.int64),
        ("Float64", "mean", numpy.float64),
        ("int64", "count", numpy.int64),
        ("int64", "argmax", numpy.int64),
        ("int64", "all", numpy.bool_),
        ("float64", "any", numpy.bool_),
    ],
)
def test_a_reduction_is_numpys_scalar(dtype, kind, expected):
    values = [True, False, True] if "bool" in dtype else [1, 2, 4]
    answer = getattr(fp.Series(values, dtype=dtype), kind)()
    assert type(answer) is expected


def test_counts_of_distinct_values_and_text_stay_python():
    assert type(fp.Series([1, 2, 2]).nunique()) is int
    assert type(fp.Series(["a", "b"]).sum()) is str
    assert type(fp.Series([1, 2]).iloc[:1].item()) is int


def test_a_cell_is_numpys_scalar():
    s = fp.Series([1, 2], index=["a", "b"])
    for cell in (s.iloc[0], s.loc["a"], s.at["a"], s["a"], s.iat[0]):
        assert type(cell) is numpy.int64
    frame = fp.DataFrame({"a": [1.5], "b": [True]})
    assert type(frame.at[0, "a"]) is numpy.float64
    assert type(frame.iat[0, 1]) is numpy.bool_
    assert type(next(iter(s))) is int


def test_a_numpy_scalar_is_an_operand():
    s = fp.Series([1, 4, 2])
    assert (s == s.max()).tolist() == [False, True, False]
    assert (s + numpy.int64(1)).tolist() == [2, 5, 3]
    frame = fp.DataFrame({"a": [1, 2]})
    assert frame.add(numpy.int32(2))["a"].tolist() == [3, 4]


def test_the_extreme_of_a_masked_column():
    assert fp.Series([1, 4, 2], dtype="Int64").argmax() == 1
    assert fp.Series([1.0, None, 4.0], dtype="Float64").idxmax() == 2
