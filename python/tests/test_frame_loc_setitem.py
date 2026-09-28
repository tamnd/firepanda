"""Writing into a frame through `loc`, `iloc`, `at` and `iat`.

Each column a key names is written the way a series is, so a value that does
not fit the column raises pandas' `TypeError` and a gap written into some rows
of whole numbers makes them float64. A write that covers every row by a slice
refuses to change the type instead. `loc` with a name the frame does not have
makes a new column, and with a row label it does not have puts a new row on
the end, whose cells meet each column the way pandas enlarges it.
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


def mixed(m: ModuleType) -> Any:
    return m.DataFrame(
        {"a": [1, 2, 3], "b": [1.5, 2.5, 3.5], "c": ["x", "y", "z"]}, index=[10, 20, 30]
    )


def flagged(m: ModuleType) -> Any:
    frame = mixed(m)
    frame["d"] = [True, False, True]
    return frame


def numbers(m: ModuleType) -> Any:
    return m.DataFrame({"a": [1, 2], "b": [1.5, 2.5]})


def repeated(m: ModuleType) -> Any:
    return m.DataFrame({"a": [1, 2, 3]}, index=[10, 10, 20])


def named(m: ModuleType) -> Any:
    return m.DataFrame({"a": [1]}, index=m.Index([1], name="k"))


def at(make: Callable[[ModuleType], Any], write: Callable[[ModuleType, Any], None]) -> Any:
    """A call that writes into a fresh frame and answers it."""

    def call(m: ModuleType) -> Any:
        frame = make(m)
        write(m, frame)
        return frame

    return call


def loc(frame: Any, key: Any, value: Any) -> None:
    frame.loc[key] = value


def iloc(frame: Any, key: Any, value: Any) -> None:
    frame.iloc[key] = value


EVERY = slice(None)

WRITES: dict[str, Callable[[ModuleType], Any]] = {
    "cell": at(mixed, lambda m, f: loc(f, (20, "a"), 9)),
    "rows and columns": at(mixed, lambda m, f: loc(f, ([10, 30], ["a", "b"]), 0)),
    "column slice": at(mixed, lambda m, f: loc(f, (10, slice("a", "b")), 0)),
    "row of values": at(mixed, lambda m, f: loc(f, 20, [7, 8.5, "q"])),
    "row from a series": at(mixed, lambda m, f: loc(f, 20, m.Series({"b": 7.5, "a": 7}))),
    "one value a column": at(mixed, lambda m, f: loc(f, ([10, 20], ["a", "b"]), [7, 8])),
    "rows of values": at(mixed, lambda m, f: loc(f, ([10, 20], ["a", "b"]), [[7, 8], [9, 10]])),
    "series on rows": at(
        mixed, lambda m, f: loc(f, ([10, 20], ["a", "b"]), m.Series({"a": 5, "b": 6.0}))
    ),
    "frame": at(
        mixed,
        lambda m, f: loc(
            f, ([10, 20], ["a", "b"]), m.DataFrame({"b": [1.0, 2.0], "a": [3, 4]}, index=[20, 10])
        ),
    ),
    "frame short a column": at(
        mixed,
        lambda m, f: loc(f, ([10, 20], ["a", "b"]), m.DataFrame({"a": [3, 4]}, index=[20, 10])),
    ),
    "frame into one column": at(
        mixed, lambda m, f: loc(f, ([10, 20], "a"), m.DataFrame({"z": [3, 4]}, index=[20, 10]))
    ),
    "mask with a gap": at(mixed, lambda m, f: loc(f, (f["a"] > 1, "a"), None)),
    "mask of floats": at(mixed, lambda m, f: loc(f, (f["a"] > 1, "b"), None)),
    "whole column": at(mixed, lambda m, f: loc(f, (EVERY, "a"), 1.0)),
    "whole column of whole floats": at(
        mixed, lambda m, f: loc(f, (EVERY, "a"), m.Series([1.0, 2.0, 3.0], index=[10, 20, 30]))
    ),
    "new column": at(mixed, lambda m, f: loc(f, ([10], "n"), 5)),
    "new column of text": at(mixed, lambda m, f: loc(f, ([10], "n"), "s")),
    "new whole column": at(mixed, lambda m, f: loc(f, (EVERY, "n"), 5)),
    "new whole column list": at(mixed, lambda m, f: loc(f, (EVERY, "n"), [1, 2, 3])),
    "new column list": at(mixed, lambda m, f: loc(f, ([10, 20], "n"), [1, 2])),
    "new column mask": at(mixed, lambda m, f: loc(f, (f["a"] > 1, "n"), 1.5)),
    "new column series": at(
        mixed, lambda m, f: loc(f, ([10, 20], "n"), m.Series([5, 6], index=[20, 10]))
    ),
    "new among names": at(mixed, lambda m, f: loc(f, (EVERY, ["a", "n"]), 0)),
    "new row": at(mixed, lambda m, f: loc(f, 40, [5, 6.5, "w"])),
    "new row cell": at(mixed, lambda m, f: loc(f, (40, "a"), 5)),
    "new row and column": at(mixed, lambda m, f: loc(f, (40, "n"), 5)),
    "new row whole float": at(flagged, lambda m, f: loc(f, 40, [5.0, 1.0, "w", True])),
    "new row number into flags": at(flagged, lambda m, f: loc(f, 40, [5, 1.0, "w", 1])),
    "new row fraction": at(flagged, lambda m, f: loc(f, 40, [5.5, 1.0, "w", True])),
    "new row of numbers reads as floats": at(numbers, lambda m, f: loc(f, 9, [1, 2.5])),
    "new row of whole numbers": at(numbers, lambda m, f: loc(f, 9, [1, 2])),
    "row of numbers into whole numbers": at(numbers, lambda m, f: loc(f, 1, [7, 8.5])),
    "new row keeps the index name": at(named, lambda m, f: loc(f, (5, "a"), 0)),
    "repeated label": at(repeated, lambda m, f: loc(f, (10, "a"), 0)),
    "iloc cell": at(mixed, lambda m, f: iloc(f, (1, 2), "k")),
    "iloc row": at(mixed, lambda m, f: iloc(f, 1, [0, 1.5, "t"])),
    "iloc one value a column": at(mixed, lambda m, f: iloc(f, ([0, 1], [0, 1]), [7, 8])),
    "iloc column series by position": at(
        mixed, lambda m, f: iloc(f, (slice(0, 2), 0), m.Series([7, 8], index=[30, 10]))
    ),
    "iloc whole column series": at(
        mixed, lambda m, f: iloc(f, (EVERY, 0), m.Series([7, 8, 9], index=[30, 10, 20]))
    ),
    "at": at(mixed, lambda m, f: f.at.__setitem__((20, "b"), 0.25)),
    "at new row": at(mixed, lambda m, f: f.at.__setitem__((40, "b"), 0.25)),
    "at new column": at(mixed, lambda m, f: f.at.__setitem__((20, "n"), 1)),
    "at repeated": at(repeated, lambda m, f: f.at.__setitem__((10, "a"), 0)),
    "iat": at(mixed, lambda m, f: f.iat.__setitem__((0, 0), 5)),
}

REFUSED: dict[str, Callable[[ModuleType], Any]] = {
    "number into text across a row": at(mixed, lambda m, f: loc(f, 20, 0)),
    "text into numbers across a row": at(mixed, lambda m, f: loc(f, 20, "s")),
    "text into a whole column": at(mixed, lambda m, f: loc(f, (EVERY, "a"), "q")),
    "fraction into a whole column": at(mixed, lambda m, f: loc(f, (EVERY, "a"), 1.5)),
    "gap into a whole column": at(mixed, lambda m, f: loc(f, (EVERY, "a"), None)),
    "gap into a whole column by range": at(mixed, lambda m, f: loc(f, (slice(10, 30), "a"), None)),
    "series with a gap into a whole column": at(
        mixed, lambda m, f: loc(f, (EVERY, "a"), m.Series([7, 8], index=[30, 10]))
    ),
    "gap by iloc into a whole column": at(mixed, lambda m, f: iloc(f, (EVERY, 0), None)),
    "number into text by mask": at(mixed, lambda m, f: loc(f, f["a"] > 1, 0)),
    "row too short": at(mixed, lambda m, f: loc(f, 20, [1, 2])),
    "list across rows and columns": at(mixed, lambda m, f: loc(f, (EVERY, ["a", "b"]), [7, 8, 9])),
    "rows too wide": at(
        mixed, lambda m, f: loc(f, ([10, 20], ["a", "b"]), [[7, 8, 1], [9, 10, 1]])
    ),
    "missing label in a list": at(mixed, lambda m, f: loc(f, [10, 99], 0)),
    "iloc row past the end": at(mixed, lambda m, f: iloc(f, (5, 0), 0)),
    "iloc column past the end": at(mixed, lambda m, f: iloc(f, (0, 5), 0)),
    "iloc columns past the end": at(mixed, lambda m, f: iloc(f, (0, [0, 7]), 0)),
    "iat row past the end": at(mixed, lambda m, f: f.iat.__setitem__((9, 0), 5)),
    "iat column past the end": at(mixed, lambda m, f: f.iat.__setitem__((0, 9), 5)),
    "iat text into numbers": at(mixed, lambda m, f: f.iat.__setitem__((0, 0), "s")),
}


def shown(frame: Any) -> Any:
    """A frame's types, labels, index name and every value, with text named `str` in both."""
    types = [str(kind).replace("string", "str") for kind in frame.dtypes.tolist()]
    values = {name: [repr(v) for v in frame[name].tolist()] for name in frame.columns}
    return types, frame.index.tolist(), frame.index.name, values


@needs_pandas
@pytest.mark.parametrize("name", list(WRITES))
def test_a_write_matches_pandas(firepanda: ModuleType, name: str) -> None:
    """The types, the labels and every value after the write."""
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
    kind = next(k for k in kinds if theirs.errisinstance(k))
    with pytest.raises(kind) as mine:
        REFUSED[name](firepanda)
    if kind is not KeyError:
        # pandas prints several values as a numpy array, so only the words before them are read.
        assert str(mine.value).split("'")[0] == str(theirs.value).split("'")[0]


def test_a_refused_write_leaves_the_frame_as_it_was(firepanda: ModuleType) -> None:
    """A row that fails at its last column writes none of the others."""
    frame = firepanda.DataFrame({"a": [1.5, 2.5], "b": [1, 2]})
    with pytest.raises(TypeError):
        frame.loc[0] = 0.5
    assert frame["a"].tolist() == [1.5, 2.5]


def test_a_copy_taken_before_keeps_what_it_had(firepanda: ModuleType) -> None:
    """The write rebinds the frame, so a copy and a column read out before are untouched."""
    frame = firepanda.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    column = frame["a"]
    copy = frame.copy()
    frame.loc[0, "a"] = 10
    frame.iloc[1] = 0
    frame.at[2, "b"] = 9
    assert frame.values.tolist() == [[10, 4], [0, 0], [3, 9]]
    assert column.tolist() == [1, 2, 3]
    assert copy.values.tolist() == [[1, 4], [2, 5], [3, 6]]


def test_a_frame_of_values_under_iloc_is_refused(firepanda: ModuleType) -> None:
    """pandas lines up the columns of a frame under `iloc` by how it stores them."""
    frame = firepanda.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(NotImplementedError):
        frame.iloc[:, [0, 1]] = firepanda.DataFrame({"a": [5, 6], "b": [7, 8]})
