"""DataFrame.eval reads several assignment lines in turn, as pandas does."""

import pytest
from firepanda.errors import InvalidArgumentError

import firepanda as fp


def test_each_line_sees_the_columns_before_it():
    frame = fp.DataFrame({"a": [1, 2], "b": [1.5, 2.5]})
    made = frame.eval("c = a + b\nd = c * 2")
    assert made.columns.tolist() == ["a", "b", "c", "d"]
    assert made["d"].tolist() == [5.0, 9.0]
    assert frame.columns.tolist() == ["a", "b"]


def test_a_line_may_overwrite_and_use_a_variable():
    frame = fp.DataFrame({"a": [1, 2], "b": [1.5, 2.5]})
    k = 3
    made = frame.eval("\nc = a + @k\na = b - 1\n")
    assert made["c"].tolist() == [k + 1, k + 2]
    assert made["a"].tolist() == [0.5, 1.5]


def test_inplace_puts_every_column_in():
    frame = fp.DataFrame({"a": [1, 2]})
    assert frame.eval("c = a\nd = c + 1", inplace=True) is None
    assert frame["d"].tolist() == [2, 3]


def test_a_line_without_an_assignment_is_refused():
    frame = fp.DataFrame({"a": [1, 2], "b": [1.5, 2.5]})
    with pytest.raises(InvalidArgumentError, match="all expressions contain an assignment"):
        frame.eval("c = a\na + b")
