"""`rename` of the row labels, by a mapping or a function, on a frame and a column.

pandas passes each row label through the mapping, keeping a label it has no
key for, or through the function, and builds the index again from what comes
back. On a MultiIndex every level goes through, or only the one `level`
names, and `errors="raise"` refuses a key that is not a label. Both axes can
be renamed in one call. Each test here runs the same code on both libraries
and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def frame(lib: Any) -> Any:
    return lib.DataFrame({"a": [1.0, 2.0, 4.0], "b": [4, 3, 2]}, index=list("xyz"))


def levelled(lib: Any) -> Any:
    labels = lib.MultiIndex.from_arrays([["p", "p", "q"], ["x", "y", "z"]], names=["o", "i"])
    return lib.DataFrame({"a": [1.0, 2.0, 4.0]}, index=labels)


def dated(lib: Any) -> Any:
    return lib.DataFrame({"a": [1, 2]}, index=lib.date_range("2024", periods=2))


BUILDS = {
    "both": lambda lib: frame(lib).rename(index={"x": "X"}, columns={"a": "A"}),
    "both functions": lambda lib: frame(lib).rename(index=str.upper, columns=str.upper),
    "mapping": lambda lib: frame(lib).rename(index={"y": "Y", "q": "Q"}),
    "positional": lambda lib: frame(lib).rename(str.upper),
    "axis": lambda lib: frame(lib).rename({"z": 9}, axis="index").index,
    "range": lambda lib: lib.DataFrame({"a": [1, 2]}).rename(index={0: 10}).index,
    "range function": lambda lib: lib.DataFrame({"a": [1, 2]}).rename(index=lambda i: i * 2).index,
    "levels": lambda lib: levelled(lib).rename(index={"p": "P", "y": "Y"}),
    "one level": lambda lib: levelled(lib).rename(index={"p": "P", "x": "X"}, level="o"),
    "level number": lambda lib: levelled(lib).rename(index=str.upper, level=1),
    "column mapping": lambda lib: frame(lib)["a"].rename({"x": "X"}),
    "column function": lambda lib: frame(lib)["a"].rename(str.upper),
    "column level": lambda lib: levelled(lib)["a"].rename({"q": "Q"}, level=0),
    "named": lambda lib: frame(lib).rename_axis("r").rename(index=str.upper).index,
    "dates": lambda lib: dated(lib).rename(index=lambda t: t + lib.Timedelta("1D")).index,
    "dates columns": lambda lib: dated(lib).rename(columns={"a": "b"}).index,
    "series as mapping": lambda lib: frame(lib).rename(index=lib.Series(["X"], index=["x"])),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_renamed_row_labels_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_a_key_that_is_not_a_label_is_refused_when_asked(firepanda: Any) -> None:
    with pytest.raises(KeyError, match=r"\['q'\] not found in axis"):
        frame(firepanda).rename(index={"q": "Q"}, errors="raise")
    with pytest.raises(KeyError, match="not found in axis"):
        levelled(firepanda).rename(index={"zz": 1}, errors="raise")


def test_row_labels_are_renamed_in_place(firepanda: Any) -> None:
    mine, theirs = frame(firepanda), frame(pd)
    assert mine.rename(index=str.upper, inplace=True) is None
    theirs.rename(index=str.upper, inplace=True)
    assert repr(mine) == repr(theirs)
    column = frame(firepanda)["a"]
    assert column.rename(str.upper, inplace=True) is None
    assert column.index.tolist() == ["X", "Y", "Z"]
