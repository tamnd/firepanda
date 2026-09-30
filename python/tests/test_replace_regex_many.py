"""`replace` with several patterns, read out of a run, a mapping or `regex` itself.

pandas finds the rows each pattern matches in the column as it arrived and only
rewrites those, so a row that a later pattern would match only after an earlier
one rewrote it keeps what the earlier one wrote. A frame reads a mapping of
column names to mappings, or to patterns beside one text, column by column.
Each test runs the same code on both libraries and compares what they print.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import pytest


def words(lib: Any) -> Any:
    return lib.Series(["a", "ab", "cd", float("nan"), "ba"], name="k")


def table(lib: Any) -> Any:
    return lib.DataFrame({"x": ["a", "ab"], "y": ["cd", "b"], "n": [1, 2]})


BUILDS = {
    "run to run": lambda lib: words(lib).replace(["a", "b"], ["b", "c"], regex=True),
    "run to one": lambda lib: words(lib).replace(["a", "c"], "Z", regex=True),
    "tuple": lambda lib: words(lib).replace(("a", "b"), "z", regex=True),
    "mapping": lambda lib: words(lib).replace({"a": "b", "b": "c"}, regex=True),
    "groups": lambda lib: words(lib).replace({r"(a)(b)?": r"\2\1"}, regex=True),
    "anchored": lambda lib: words(lib).replace([r"^b", r"a$"], ["B", "A"], regex=True),
    "regex run": lambda lib: words(lib).replace(regex=["a", "c"], value="Q"),
    "regex mapping": lambda lib: words(lib).replace(regex={"a": "Q", "d": "R"}),
    "compiled": lambda lib: words(lib).replace([re.compile("A", re.I)], ["q"], regex=True),
    "column of values": lambda lib: words(lib).replace(
        ["a", "b"], lib.Series(["1", "2"]), regex=True
    ),
    "empty": lambda lib: words(lib).replace([], [], regex=True),
    "numbers": lambda lib: lib.Series([1, 2]).replace(["1"], ["x"], regex=True),
    "frame run": lambda lib: table(lib).replace(["a", "b"], ["X", "Y"], regex=True),
    "frame mapping": lambda lib: table(lib).replace({"a": "X", "d": "Y"}, regex=True),
    "frame nested": lambda lib: table(lib).replace(
        {"x": {"a": "X"}, "y": {"b": "Y", "c": "C"}}, regex=True
    ),
    "frame by column": lambda lib: table(lib).replace({"n": "1", "x": "a"}, "Z", regex=True),
    "frame regex mapping": lambda lib: table(lib).replace(regex={"a": "Q"}),
    "frame regex run": lambda lib: table(lib).replace(regex=["a", "b"], value="Q"),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_many_patterns_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_inplace_hands_the_column_back_rewritten(firepanda: Any) -> None:
    column = words(firepanda)
    assert column.replace({"a": "q"}, regex=True, inplace=True) is column
    assert repr(column) == repr(words(pd).replace({"a": "q"}, regex=True))


def test_runs_of_different_lengths_are_refused(firepanda: Any) -> None:
    with pytest.raises(ValueError, match="Replacement lists must match in length"):
        words(firepanda).replace(["a", "c"], ["Z"], regex=True)


def test_a_replacement_that_is_not_text_is_refused(firepanda: Any) -> None:
    with pytest.raises(NotImplementedError, match="regex"):
        words(firepanda).replace({r"^a$": 5}, regex=True)
