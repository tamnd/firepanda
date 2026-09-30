"""Reducing across a row with `axis=1`, and sorting with a `key`, checked against pandas.

A frame reduced with `axis=1` answers one value per row, labelled by the row index. An int or
bool sum is int64, a float or a mean is float64, and `skipna=False` leaves a gap in any row that
has one. A frame of text joins for a sum and compares for a max, which is what pandas does.

`sort_index` takes a `key`, `na_position` and `sort_remaining`, and `sort_values` takes a
`key`, which is called with each column and whose answer decides the order.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def numbers(lib: ModuleType) -> Any:
    return lib.DataFrame({"a": [1, 2, 3], "b": [4.5, None, 6.0], "c": [7, 8, 9]})


def words(lib: ModuleType) -> Any:
    return lib.DataFrame({"a": ["x", "yy", ""], "b": ["z", "a", "q"]})


def labelled(lib: ModuleType) -> Any:
    return lib.DataFrame({"v": [1, 2, 3, 4]}, index=[3.0, None, 1.0, 2.0])


def paired(lib: ModuleType) -> Any:
    index = lib.MultiIndex.from_arrays([["b", "a", "b", "a"], [1, 2, 2, 1]], names=["x", "y"])
    return lib.DataFrame({"v": [1, 2, 3, 4]}, index=index)


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "sum": lambda lib: numbers(lib).sum(axis=1),
    "sum-int": lambda lib: numbers(lib)[["a", "c"]].sum(axis=1),
    "sum-skipna": lambda lib: numbers(lib).sum(axis=1, skipna=False),
    "sum-min-count": lambda lib: numbers(lib)[["b"]].sum(axis=1, min_count=1),
    "prod": lambda lib: numbers(lib).prod(axis=1),
    "max": lambda lib: numbers(lib).max(axis="columns"),
    "min": lambda lib: numbers(lib).min(axis=1),
    "mean": lambda lib: numbers(lib).mean(axis=1),
    "median": lambda lib: numbers(lib).median(axis=1),
    "var": lambda lib: numbers(lib).var(axis=1),
    "std": lambda lib: numbers(lib).std(axis=1, ddof=0),
    "sem": lambda lib: numbers(lib).sem(axis=1),
    "skew": lambda lib: numbers(lib).skew(axis=1),
    "kurt": lambda lib: numbers(lib).kurt(axis=1),
    "count": lambda lib: numbers(lib).count(axis=1),
    "nunique": lambda lib: numbers(lib).nunique(axis=1),
    "any": lambda lib: numbers(lib).any(axis=1),
    "all": lambda lib: numbers(lib).all(axis=1),
    "text-sum": lambda lib: words(lib).sum(axis=1),
    "text-max": lambda lib: words(lib).max(axis=1),
    "text-mean": lambda lib: words(lib).mean(axis=1),
    "text-any": lambda lib: words(lib).any(axis=1),
    "numeric-only": lambda lib: numbers(lib).assign(d=list("pqr")).sum(axis=1, numeric_only=True),
    "sort-index-key": lambda lib: labelled(lib).sort_index(key=lambda labels: -labels),
    "sort-index-first": lambda lib: labelled(lib).sort_index(na_position="first"),
    "sort-index-down": lambda lib: labelled(lib).sort_index(ascending=False, na_position="first"),
    "sort-index-level": lambda lib: paired(lib).sort_index(level=1),
    "sort-index-remaining": lambda lib: paired(lib).sort_index(level=1, sort_remaining=False),
    "sort-index-ignore": lambda lib: labelled(lib).sort_index(key=abs, ignore_index=True),
    "sort-values-key": lambda lib: words(lib).sort_values("b", key=lambda col: col.str.upper()),
    "sort-values-length": lambda lib: words(lib).sort_values("a", key=lambda col: col.str.len()),
    "sort-values-two": lambda lib: numbers(lib).sort_values(["b", "a"], key=lambda col: -col),
    "series-sort-key": lambda lib: numbers(lib)["b"].sort_values(key=lambda col: -col),
    "series-sort-first": lambda lib: numbers(lib)["b"].sort_values(
        key=lambda col: col % 4, na_position="first", ascending=False
    ),
}


def outcome(run: Callable[[], Any]) -> Any:
    """What a case answers, as a printed value that can be compared across libraries."""
    try:
        answer = run()
    except Exception as error:
        return type(error).__name__
    return repr(answer)


@needs_pandas
@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_answers_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    import pandas as pd

    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))


def test_a_row_of_text_says_the_operation_is_not_supported() -> None:
    with pytest.raises(TypeError, match="does not support operation 'mean'"):
        words(fp).mean(axis=1)


def test_a_key_on_the_column_axis_is_refused() -> None:
    with pytest.raises(NotImplementedError, match="key"):
        numbers(fp).sort_index(axis=1, key=lambda labels: labels)
