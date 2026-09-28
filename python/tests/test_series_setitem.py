"""Writing into a series, `s[key] = value` and through `loc`, `iloc`, `at` and `iat`.

pandas 3 writes a value into a column only when the column can hold it as it
is. A gap written into whole numbers makes them float64, a float with no
fraction goes into whole numbers, and anything else that does not fit raises a
`TypeError`. A list is read as a numpy array first, so it is stricter than one
value. A series of values is lined up on the labels under `loc` and a mask, and
taken by position under `iloc` and a slice of numbers. `loc` with a label the
series does not have puts a new row on the end.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def numbers(m: ModuleType) -> Any:
    return m.Series([1, 2, 3, 4], name="n")


def lettered(m: ModuleType) -> Any:
    return m.Series([1, 2, 3, 4], index=["a", "b", "c", "d"], name="n")


def floats(m: ModuleType) -> Any:
    return m.Series([1.5, 2.5, None, 4.5])


def words(m: ModuleType) -> Any:
    return m.Series(["a", "b", None, "d"])


def flags(m: ModuleType) -> Any:
    return m.Series([True, False, True, False])


def moments(m: ModuleType) -> Any:
    return m.to_datetime(m.Series(["2020-01-01", "2020-01-02", None]))


def spans(m: ModuleType) -> Any:
    return m.to_timedelta(m.Series(["1s", "2s", None]))


def at(make: Callable[[ModuleType], Any], write: Callable[[ModuleType, Any], None]) -> Any:
    """A call that writes into a fresh series and answers it."""

    def call(m: ModuleType) -> Any:
        column = make(m)
        write(m, column)
        return column

    return call


def put(column: Any, key: Any, value: Any) -> None:
    column[key] = value


WRITES: dict[str, Callable[[ModuleType], Any]] = {
    "label": at(lettered, lambda m, s: put(s, "b", 20)),
    "new label": at(lettered, lambda m, s: put(s, "z", 9)),
    "new label float": at(numbers, lambda m, s: put(s, 7, 9.5)),
    "new label gap": at(numbers, lambda m, s: put(s, 7, None)),
    "number on text labels": at(lettered, lambda m, s: put(s, "q", 0)),
    "labels": at(lettered, lambda m, s: put(s, ["d", "a"], [40, 10])),
    "label slice": at(lettered, lambda m, s: put(s, slice("b", "c"), 0)),
    "number slice": at(lettered, lambda m, s: put(s, slice(1, None), 0)),
    "stepped slice": at(lettered, lambda m, s: put(s, slice(None, None, 2), [8, 9])),
    "mask": at(numbers, lambda m, s: put(s, s > 2, 0)),
    "mask of a list": at(numbers, lambda m, s: put(s, [True, False, True, False], -1)),
    "mask none marked": at(numbers, lambda m, s: put(s, s > 9, None)),
    "series lined up": at(numbers, lambda m, s: put(s, s > 1, s * 10)),
    "series shuffled": at(
        lettered, lambda m, s: s.loc.__setitem__(["b", "d"], m.Series([10, 20], index=["d", "b"]))
    ),
    "series part": at(
        lettered, lambda m, s: s.loc.__setitem__(["a", "b"], m.Series([5], index=["a"]))
    ),
    "series whole floats": at(numbers, lambda m, s: put(s, s > 1, m.Series([7.0, 8, 9, 10]))),
    "series with a gap": at(numbers, lambda m, s: put(s, s > 1, m.Series([7.0, None, 9, 10]))),
    "iloc": at(lettered, lambda m, s: s.iloc.__setitem__(-1, 0)),
    "iloc list": at(numbers, lambda m, s: s.iloc.__setitem__([3, 1], [30, 10])),
    "iloc repeated": at(numbers, lambda m, s: s.iloc.__setitem__([1, 1], [5, 6])),
    "iloc tuple value": at(numbers, lambda m, s: s.iloc.__setitem__([0, 1], (4, 5))),
    "iloc series by position": at(
        lettered, lambda m, s: s.iloc.__setitem__([1, 3], m.Series([10, 20], index=["d", "b"]))
    ),
    "slice series by position": at(
        lettered, lambda m, s: put(s, slice(1, 3), m.Series([10, 20], index=["c", "b"]))
    ),
    "iloc empty": at(numbers, lambda m, s: s.iloc.__setitem__([], 0)),
    "at": at(lettered, lambda m, s: s.at.__setitem__("c", 0)),
    "at new": at(lettered, lambda m, s: s.at.__setitem__("q", 5)),
    "iat": at(numbers, lambda m, s: s.iat.__setitem__(2, 0)),
    "repeated labels": at(
        lambda m: m.Series([1, 2, 3], index=[0, 0, 1]), lambda m, s: s.loc.__setitem__(0, 9)
    ),
    "whole float": at(numbers, lambda m, s: put(s, 0, 7.0)),
    "gap": at(numbers, lambda m, s: put(s, 0, None)),
    "nan": at(numbers, lambda m, s: put(s, 0, float("nan"))),
    "narrow gap": at(lambda m: m.Series([1, 2], dtype="int8"), lambda m, s: put(s, 0, None)),
    "float": at(floats, lambda m, s: put(s, 2, 1)),
    "float list": at(floats, lambda m, s: s.iloc.__setitem__([0, 1], [1, True])),
    "float gap": at(lambda m: m.Series([1.5, 2.5], dtype="float32"), lambda m, s: put(s, 0, None)),
    "text": at(words, lambda m, s: put(s, 0, "z")),
    "text gap": at(words, lambda m, s: put(s, 0, None)),
    "text list": at(words, lambda m, s: s.iloc.__setitem__([0, 1], [None, "q"])),
    "flag": at(flags, lambda m, s: put(s, 1, True)),
    "flag list": at(flags, lambda m, s: s.iloc.__setitem__([0, 1], [False, True])),
    "whole list": at(numbers, lambda m, s: s.iloc.__setitem__([0, 1], [True, 4])),
    "moment text": at(moments, lambda m, s: put(s, 0, "2021-05-05")),
    "moment": at(moments, lambda m, s: put(s, 2, m.Timestamp("2022-02-02 10:00"))),
    "moment gap": at(moments, lambda m, s: put(s, 0, None)),
    "span text": at(spans, lambda m, s: put(s, 0, "3s")),
    "span gap": at(spans, lambda m, s: put(s, 1, m.NaT)),
}

REFUSED: dict[str, Callable[[ModuleType], Any]] = {
    "fraction": at(numbers, lambda m, s: put(s, 0, 1.5)),
    "text in numbers": at(numbers, lambda m, s: put(s, 0, "x")),
    "flag in numbers": at(numbers, lambda m, s: put(s, 0, True)),
    "too big": at(lambda m: m.Series([1, 2], dtype="int32"), lambda m, s: put(s, 0, 2**40)),
    "below zero": at(lambda m: m.Series([1, 2], dtype="uint8"), lambda m, s: put(s, 0, -1)),
    "flag in floats": at(floats, lambda m, s: put(s, 0, True)),
    "text in floats": at(floats, lambda m, s: put(s, 0, "s")),
    "number in flags": at(flags, lambda m, s: put(s, 0, 1)),
    "gap in flags": at(flags, lambda m, s: put(s, 0, None)),
    "number in text": at(words, lambda m, s: put(s, 0, 1)),
    "number in moments": at(moments, lambda m, s: put(s, 0, 5)),
    "fraction nowhere": at(numbers, lambda m, s: s.iloc.__setitem__([], 1.5)),
    "float list in numbers": at(numbers, lambda m, s: s.iloc.__setitem__([0, 1], [3.0, 4.0])),
    "gap list in numbers": at(numbers, lambda m, s: s.iloc.__setitem__([0, 1], [None, 4])),
    "gap list in floats": at(floats, lambda m, s: s.iloc.__setitem__([0, 1], [None, 4])),
    "mixed list in text": at(words, lambda m, s: s.iloc.__setitem__([0, 1], [1, "c"])),
    "number list in flags": at(flags, lambda m, s: s.iloc.__setitem__([0, 1], [1, 0])),
    "series of fractions": at(numbers, lambda m, s: put(s, s > 1, m.Series([7.0, 8.5, 9, 1]))),
    "series of flags": at(numbers, lambda m, s: put(s, s > 1, s > 2)),
    "series of numbers in text": at(words, lambda m, s: put(s, [True] * 4, m.Series([1, 2, 3, 4]))),
    "wrong length": at(numbers, lambda m, s: s.iloc.__setitem__([1, 3], [10])),
    "list into one": at(numbers, lambda m, s: s.iloc.__setitem__(0, [5])),
    "mask wrong length": at(numbers, lambda m, s: put(s, [True, False], 0)),
    "iloc past the end": at(numbers, lambda m, s: s.iloc.__setitem__(9, 0)),
    "iloc list past the end": at(numbers, lambda m, s: s.iloc.__setitem__([0, 7], 0)),
    "iat past the end": at(numbers, lambda m, s: s.iat.__setitem__(7, 0)),
    "missing labels": at(lettered, lambda m, s: s.loc.__setitem__(["a", "q"], 0)),
    "two axes": at(numbers, lambda m, s: s.loc.__setitem__((0, 1), 0)),
}


def shown(column: Any) -> Any:
    """A column's type, labels and values, with text named `str` in both."""
    printed = str(column.dtype).replace("string", "str")
    return printed, column.index.tolist(), [repr(value) for value in column.tolist()]


@needs_pandas
@pytest.mark.parametrize("name", list(WRITES))
def test_a_write_matches_pandas(firepanda: ModuleType, name: str) -> None:
    """The type, the labels and every value after the write."""
    import pandas as pd

    assert shown(WRITES[name](firepanda)) == shown(WRITES[name](pd))


@needs_pandas
@pytest.mark.parametrize("name", list(REFUSED))
def test_a_write_pandas_refuses_raises_its_kind_of_error(firepanda: ModuleType, name: str) -> None:
    """The same kind of error, and pandas' words for a value that does not fit."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        REFUSED[name](pd)
    kinds = (TypeError, KeyError, IndexError, ValueError)
    kind = next((k for k in kinds if theirs.errisinstance(k)), None)
    with pytest.raises(kind or Exception) as mine:
        REFUSED[name](firepanda)
    if kind is None:
        assert type(mine.value).__name__ == type(theirs.value).__name__
    if kind is TypeError:
        # pandas prints several values as a numpy array, so only the words before them are read.
        assert str(mine.value).split("'")[0] == str(theirs.value).split("'")[0]


def test_a_copy_taken_before_keeps_what_it_had(firepanda: ModuleType) -> None:
    """The write rebinds the series, so a copy and a column read out before are untouched."""
    frame = firepanda.DataFrame({"a": [1, 2, 3]})
    column = frame["a"]
    copy = column.copy()
    column[0] = 10
    column.iloc[1] = 20
    assert column.tolist() == [10, 20, 3]
    assert copy.tolist() == [1, 2, 3]
    assert frame["a"].tolist() == [1, 2, 3]


def test_a_write_keeps_the_name_and_the_index_name(firepanda: ModuleType) -> None:
    """The name and the name of the labels survive each kind of write."""
    column = firepanda.Series([1, 2, 3], index=firepanda.Index([5, 6, 7], name="k"), name="v")
    column[6] = 0
    column.iloc[[0, 2]] = [8, 9]
    column[8] = 4
    assert column.name == "v" and column.index.name == "k"
    assert column.tolist() == [8, 0, 9, 4]
