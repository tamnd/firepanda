"""Categories handed to a frame whole, and rows whose labels repeat.

pandas keeps a `Categorical` put in with `assign`, and an index of categories
made a column, as categories rather than their values. Dropping a label that
repeats drops every row holding it, and a transpose whose row labels repeat
would need repeated columns, which firepanda refuses rather than answering a
frame with rows missing. Each test here runs the same code on both libraries
and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def repeated(lib: Any) -> Any:
    return lib.DataFrame({"v": [1, 2, 3, 4], "w": [5, 6, 7, 8]}, index=["a", "b", "a", "c"])


BUILDS = {
    "assign": lambda lib: (
        lib.DataFrame({"v": [1, 2, 3]}).assign(k=lib.Categorical(["a", "b", "a"])).dtypes
    ),
    "assign ordered": lambda lib: lib.DataFrame({"v": [1, 2]}).assign(
        k=lib.Categorical(["y", "x"], categories=["y", "x"], ordered=True)
    )["k"],
    "assign index": lambda lib: lib.DataFrame({"v": [1, 2]}).assign(
        k=lib.CategoricalIndex(["x", "y"])
    )["k"],
    "assign to nothing": lambda lib: lib.DataFrame().assign(k=lib.Categorical(["a", "b"]))["k"],
    "series of index": lambda lib: lib.Series(lib.CategoricalIndex(["a", "b", "a"], name="k")),
    "reset_index": lambda lib: (
        lib.DataFrame({"k": lib.Categorical(["x", "y"]), "v": [1, 2]})
        .set_index("k")
        .reset_index()
        .dtypes
    ),
    "drop": lambda lib: repeated(lib).drop("a"),
    "drop list": lambda lib: repeated(lib).drop(["a", "c"]),
    "drop both halves": lambda lib: repeated(lib).drop(index="a", columns="w"),
    "drop ignore": lambda lib: repeated(lib).drop(["z", "a"], errors="ignore"),
    "drop column": lambda lib: repeated(lib)["v"].drop("a"),
    "drop numbers": lambda lib: lib.Series([1, 2, 3], index=[1, 1, 2]).drop(1),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_the_answer_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_categories_of_the_wrong_length_are_refused(firepanda: Any) -> None:
    with pytest.raises(ValueError, match="Length of values"):
        firepanda.DataFrame({"v": [1, 2, 3]}).assign(k=firepanda.Categorical(["a"]))


def test_a_missing_label_among_repeated_ones_is_refused(firepanda: Any) -> None:
    with pytest.raises(KeyError, match="not found in axis"):
        repeated(firepanda).drop(["a", "z"])


def test_a_transpose_of_repeated_row_labels_is_refused(firepanda: Any) -> None:
    with pytest.raises(NotImplementedError, match="row labels repeat"):
        _ = repeated(firepanda).T
