"""Reducing a category column, checked against a running pandas.

The column is reduced through its codes, so `min` and `max` of an ordered one
are the categories at the smallest and largest code, `nunique` counts the
categories in use, and `count` counts the rows with one. pandas refuses `min`
and `max` when the order means nothing and refuses the rest outright, both with
a `TypeError`, and the sentences here are pandas' own.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

ROWS = ["b", "a", None, "c", "b"]
ORDER = ["c", "b", "a"]


def both(firepanda: ModuleType, ordered: bool) -> tuple[Any, Any]:
    """The same column in both libraries, ordered backwards or unordered."""
    import pandas as pd

    made = []
    for module in (firepanda, pd):
        column = module.Series(ROWS).astype("category")
        if ordered:
            column = column.cat.set_categories(ORDER, ordered=True)
        made.append(column)
    return made[0], made[1]


def answer(call: Any) -> Any:
    """The value, or the kind of error and its message."""
    try:
        return call()
    except Exception as error:
        return type(error).__name__, str(error)


@pytest.mark.parametrize("kind", ["min", "max"])
def test_an_ordered_column_answers_a_category(firepanda: ModuleType, kind: str) -> None:
    """By the category order, which is not the order of the letters."""
    mine, theirs = both(firepanda, True)
    assert getattr(mine, kind)() == getattr(theirs, kind)()


@pytest.mark.parametrize("kind", ["min", "max"])
def test_an_unordered_column_refuses_as_pandas_refuses(firepanda: ModuleType, kind: str) -> None:
    """Pointing at `as_ordered`."""
    mine, theirs = both(firepanda, False)
    assert answer(getattr(mine, kind)) == answer(getattr(theirs, kind))


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("dropna", [True, False])
def test_nunique_counts_the_categories_in_use(
    firepanda: ModuleType, ordered: bool, dropna: bool
) -> None:
    """And one more for the missing row when it is asked to."""
    mine, theirs = both(firepanda, ordered)
    assert mine.nunique(dropna=dropna) == theirs.nunique(dropna=dropna)


def test_count_is_the_rows_with_a_category(firepanda: ModuleType) -> None:
    """The same count a column of strings gives."""
    mine, theirs = both(firepanda, False)
    assert mine.count() == theirs.count()


@pytest.mark.parametrize(
    "kind", ["sum", "mean", "median", "std", "var", "prod", "sem", "skew", "any", "all"]
)
@pytest.mark.parametrize("ordered", [True, False])
def test_arithmetic_on_positions_is_refused(
    firepanda: ModuleType, kind: str, ordered: bool
) -> None:
    """A sum of codes is not a sum of anything, ordered or not."""
    mine, theirs = both(firepanda, ordered)
    assert answer(getattr(mine, kind)) == answer(getattr(theirs, kind))


def test_skipping_nothing_answers_nan_over_a_gap(firepanda: ModuleType) -> None:
    """As every other reduction does."""
    mine, theirs = both(firepanda, True)
    got = mine.min(skipna=False)
    want = theirs.min(skipna=False)
    assert got != got and want != want


def test_a_column_with_no_category_in_use_has_no_minimum(firepanda: ModuleType) -> None:
    """NaN, where the kernel had nothing to reduce."""
    import pandas as pd

    rows = ["x", None]
    mine = firepanda.Series(rows).astype("category").cat.set_categories(["a"], ordered=True)
    theirs = pd.Series(rows).astype("category").cat.set_categories(["a"], ordered=True)
    got, want = mine.max(), theirs.max()
    assert got != got and want != want
