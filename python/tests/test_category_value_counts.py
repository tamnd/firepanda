"""`value_counts` on a category column, checked against pandas.

pandas counts a category column by its categories, so a category nothing uses
is a row with a count of nought, the rows start in the order of the categories
and the sort that follows is stable. The index is categorical with the
column's categories and ordered flag, and a missing value is a row of its own
only when `dropna` is False.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

VALUES = ["b", "a", None, "b", "c", "a", "b"]


def missing(value: Any) -> bool:
    """None, NaN, or pandas' NA, which refuses to say whether it equals itself."""
    try:
        return value is None or bool(value != value)
    except TypeError:
        return True


def plain(values: list[Any]) -> list[Any]:
    """The values with every spelling of missing as one word."""
    return ["missing" if missing(value) else value for value in values]


def agrees(got: Any, want: Any) -> None:
    """The same counts, labels, categories, names and types."""
    assert got.tolist() == want.tolist()
    assert str(got.dtype) == str(want.dtype)
    assert plain(list(got.index)) == plain(list(want.index))
    assert str(got.index.dtype) == "category"
    assert got.name == want.name
    assert got.index.name == want.index.name


def column(m: ModuleType, ordered: bool = False, missing: bool = True) -> Any:
    """A category column with an unused category, in an order that is not sorted."""
    values = VALUES if missing else [value for value in VALUES if value is not None]
    kept = m.Series(values, name="k").astype("category")
    return kept.cat.set_categories(["c", "z", "b", "a"], ordered=ordered)


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: column(m).value_counts(),
    lambda m: column(m, ordered=True).value_counts(),
    lambda m: column(m).value_counts(dropna=False),
    lambda m: column(m, missing=False).value_counts(dropna=False),
    lambda m: column(m).value_counts(ascending=True),
    lambda m: column(m).value_counts(normalize=True),
    lambda m: column(m).value_counts(normalize=True, dropna=False),
    lambda m: column(m).value_counts(sort=False),
    lambda m: column(m).value_counts(sort=False, dropna=False),
    lambda m: column(m, ordered=True).value_counts(dropna=False).sort_index(),
    lambda m: m.Series(["x", "y"]).astype("category").iloc[:0].value_counts(),
    lambda m: m.Series([None, None], dtype="string").astype("category").value_counts(dropna=False),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Unused categories, order, the missing row, normalising and no sort."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


def test_the_index_keeps_the_categories(firepanda: ModuleType) -> None:
    """Crossing to Arrow as an ordered dictionary holding every category."""
    import pyarrow as pa

    counts = column(firepanda, ordered=True).value_counts().reset_index()
    labels = pa.table(counts).column("k").combine_chunks()
    assert pa.types.is_dictionary(labels.type)
    assert labels.type.ordered
    assert labels.dictionary.to_pylist() == ["c", "z", "b", "a"]
