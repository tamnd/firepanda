"""`DataFrame.quantile(method="table")`, whole rows rather than one quantile per column.

pandas sorts the rows by every column at once, missing values last, and takes
the row at each quantile's place, lower, higher or nearest, so each answer is
a row the frame holds. `linear` is refused, since it would blend two rows.
Each test runs the same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def gapped(lib: Any) -> Any:
    return lib.DataFrame({"a": [3, 1, 2, 1, 5], "b": [1.5, 9.0, None, 2.0, 0.5]})


def whole(lib: Any) -> Any:
    return lib.DataFrame({"a": [3, 1, 2, 4], "b": [7, 8, 5, 6]}, index=list("wxyz"))


def shown(answer: Any) -> str:
    kinds = answer.dtypes if hasattr(answer, "columns") else [answer.dtype]
    return repr((answer.to_string(), [str(kind) for kind in kinds]))


BUILDS = {
    "one": lambda lib: gapped(lib).quantile(0.5, method="table", interpolation="nearest"),
    "several": lambda lib: gapped(lib).quantile(
        [0.1, 0.5, 0.9], method="table", interpolation="nearest"
    ),
    "lower": lambda lib: gapped(lib).quantile([0.25, 0.6], method="table", interpolation="lower"),
    "higher": lambda lib: gapped(lib).quantile([0.25, 0.6], method="table", interpolation="higher"),
    "half to even": lambda lib: whole(lib).quantile(
        [0.5, 1 / 6], method="table", interpolation="nearest"
    ),
    "whole numbers": lambda lib: whole(lib).quantile(0.5, method="table", interpolation="lower"),
    "ends": lambda lib: whole(lib).quantile([0.0, 1.0], method="table", interpolation="nearest"),
    "across": lambda lib: whole(lib).quantile(0.5, axis=1, method="table", interpolation="nearest"),
    "across several": lambda lib: whole(lib).quantile(
        [0.2, 0.8], axis=1, method="table", interpolation="lower"
    ),
    "ties": lambda lib: lib.DataFrame({"a": [1, 1, 1], "b": [3, 2, 1]}).quantile(
        [0.0, 0.5], method="table", interpolation="nearest"
    ),
    "empty": lambda lib: (
        gapped(lib).iloc[:0].quantile(0.5, method="table", interpolation="nearest")
    ),
    "empty several": lambda lib: (
        gapped(lib).iloc[:0].quantile([0.5], method="table", interpolation="nearest")
    ),
    "text": lambda lib: (
        gapped(lib).assign(t=list("vwxyz")).quantile([0.5], method="table", interpolation="nearest")
    ),
    "numbers only": lambda lib: (
        gapped(lib)
        .assign(t=list("vwxyz"))
        .quantile(0.5, method="table", numeric_only=True, interpolation="nearest")
    ),
    "tuple": lambda lib: whole(lib).quantile((0.3, 0.6), method="table", interpolation="nearest"),
}

MISTAKES = {
    "midpoint": (
        "Invalid interpolation: midpoint",
        lambda lib: whole(lib).quantile(0.5, method="table", interpolation="midpoint"),
    ),
    "out of range": (
        "percentiles should all be in the interval",
        lambda lib: whole(lib).quantile(1.5, method="table", interpolation="nearest"),
    ),
    "repeated labels": (
        "The column label '10' is not unique.",
        lambda lib: lib.DataFrame({"a": [1, 2]}, index=[10, 10]).quantile(
            0.5, axis=1, method="table", interpolation="nearest"
        ),
    ),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_quantile_table_is_pandas(firepanda: Any, make: Any) -> None:
    assert shown(make(firepanda)) == shown(make(pd))


@pytest.mark.parametrize(("words", "make"), MISTAKES.values(), ids=MISTAKES.keys())
def test_what_pandas_refuses_is_refused(firepanda: Any, words: str, make: Any) -> None:
    with pytest.raises(ValueError) as theirs:
        make(pd)
    with pytest.raises(ValueError) as mine:
        make(firepanda)
    assert words in str(theirs.value)
    assert words in str(mine.value)
