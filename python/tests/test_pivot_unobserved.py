"""`pivot_table(observed=False)`, a group for every category a key could hold.

pandas groups a categorical key by all of its categories when `observed` is
False, so a category no row holds still gets a row or a column, holding what
the aggregate of no rows is: nought for a sum or a count, and missing for a
mean, which `dropna` then takes away. Each test runs the same code on both
libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def cats(lib: Any) -> Any:
    return lib.DataFrame(
        {
            "k": lib.Categorical(["x", "x", "y", "y"], categories=["x", "y", "z"]),
            "c": lib.Categorical(["p", "q", "p", "p"], categories=["p", "q", "r"]),
            "v": [1.0, 2.0, 3.0, 4.0],
        }
    )


CASES = [
    (agg, observed, dropna, across)
    for agg in ["sum", "mean", "count"]
    for observed in [True, False]
    for dropna in [True, False]
    for across in [None, "c"]
]


@pytest.mark.parametrize(("agg", "observed", "dropna", "across"), CASES)
def test_pivot_unobserved_is_pandas(
    firepanda: Any, agg: str, observed: bool, dropna: bool, across: Any
) -> None:
    def make(lib: Any) -> Any:
        return lib.pivot_table(
            cats(lib),
            values="v",
            index="k",
            columns=across,
            aggfunc=agg,
            observed=observed,
            dropna=dropna,
        )

    assert repr(make(firepanda)) == repr(make(pd))


@pytest.mark.parametrize("observed", [True, False])
def test_categorical_row_key_labels_categorically(firepanda: Any, observed: bool) -> None:
    def make(lib: Any) -> Any:
        return lib.pivot_table(
            cats(lib), values="v", index="k", columns="c", aggfunc="sum", observed=observed
        ).index

    assert repr(make(firepanda)) == repr(make(pd))


def mixed(lib: Any, first: Any, second: Any) -> Any:
    return lib.concat(
        [lib.DataFrame({"v": [1]}, index=first(lib)), lib.DataFrame({"v": [2]}, index=second(lib))]
    ).index


def text(lib: Any) -> Any:
    return lib.Index(["x"], name="k")


def number(lib: Any) -> Any:
    return lib.Index([1], name="k")


def category(lib: Any) -> Any:
    return lib.CategoricalIndex(["y"], name="k")


def other(lib: Any) -> Any:
    return lib.Index([2], name="j")


PAIRS = {
    "text and category": (text, category),
    "category and text": (category, text),
    "number and text": (number, text),
    "names differ": (text, other),
}


@pytest.mark.parametrize("pair", PAIRS.values(), ids=PAIRS.keys())
def test_mixed_concat_labels_are_pandas(firepanda: Any, pair: Any) -> None:
    assert repr(mixed(firepanda, *pair)) == repr(mixed(pd, *pair))
