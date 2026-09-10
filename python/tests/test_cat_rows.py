"""A category column stays a category column when its rows are moved about.

Every operation here is one that does not know it is looking at a categorical.
A drop, a head, a fill and a shift are written against the physical layout,
because a filter over a date column runs through the integer that a date is
stored as, and each of them puts the input's logical type back on the result at
the end. For every other type that is the whole of it. A dictionary is the one
type whose meaning is not all in the buffer: the codes are, the category list is
beside it, and the relabelling left it behind.

So what is checked here is not that the values are right, which they were. It is
that the result is still a categorical afterwards, that its categories are the
ones that went in, and that the values read back as words rather than as the
positions they are stored as. The pandas comparison is worth having for the same
reason it is worth having anywhere else: it says the answer is the one a caller
moving off pandas already expects, including the part where a filter leaves an
emptied category in the list.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

ROWS = ["low", None, "high", "low", "mid"]
"""Five rows, three categories and one missing value, which is the shape that failed."""


def made(firepanda: ModuleType, values: list[Any] | None = None) -> Any:
    """A firepanda category column."""
    return firepanda.Series(ROWS if values is None else values).astype("category")


def theirs(values: list[Any] | None = None) -> Any:
    """The same column in pandas."""
    import pandas as pd

    return pd.Series(ROWS if values is None else values).astype("category")


def words(column: Any) -> list[Any]:
    """The rows as a list, with a missing one written the same way on both sides.

    `tolist()` on a pandas categorical writes a missing row as a float `nan` and
    firepanda writes `None`, which is a difference about how pandas spells
    absence in a list of strings rather than anything to do with moving rows. It
    is normalised here so that these tests are about the thing they are about,
    and firepanda's own spelling is asserted directly in the tests that do not
    compare against pandas.
    """
    return [None if value is None or value != value else value for value in column.tolist()]


def categories(column: Any) -> list[Any]:
    """The category list, as a plain list, from either library."""
    return list(column.cat.categories.tolist())


def test_dropping_the_nulls_leaves_a_category_column(firepanda: ModuleType) -> None:
    """The case that found all of this, and it used to raise from the Arrow writer."""
    out = made(firepanda).dropna()
    assert out.dtype == "category"
    assert out.tolist() == ["low", "high", "low", "mid"]


def test_dropping_the_nulls_keeps_the_categories(firepanda: ModuleType) -> None:
    """Reading the categories is what a corrupt column cannot do."""
    assert categories(made(firepanda).dropna()) == ["high", "low", "mid"]


@needs_pandas
def test_dropping_the_nulls_agrees_with_pandas(firepanda: ModuleType) -> None:
    """Values and categories both."""
    ours, theirs_ = made(firepanda).dropna(), theirs().dropna()
    assert words(ours) == words(theirs_)
    assert categories(ours) == categories(theirs_)


def test_a_head_is_still_a_category_column(firepanda: ModuleType) -> None:
    """A slice is its own path and neither the filter nor the take reaches it."""
    out = made(firepanda).head(3)
    assert out.dtype == "category"
    assert out.tolist() == ["low", None, "high"]
    assert categories(out) == ["high", "low", "mid"]


def test_a_tail_is_still_a_category_column(firepanda: ModuleType) -> None:
    """The other end of the same operation."""
    out = made(firepanda).tail(2)
    assert out.dtype == "category"
    assert out.tolist() == ["low", "mid"]


@needs_pandas
def test_a_head_keeps_a_category_no_row_is_left_in(firepanda: ModuleType) -> None:
    """pandas does the same, and `remove_unused_categories` is what undoes it."""
    ours, theirs_ = made(firepanda).head(2), theirs().head(2)
    assert categories(ours) == categories(theirs_) == ["high", "low", "mid"]


def test_filling_forward_keeps_the_categories(firepanda: ModuleType) -> None:
    """A fill takes a value from another row, so it cannot invent a category."""
    out = made(firepanda).ffill()
    assert out.dtype == "category"
    assert out.tolist() == ["low", "low", "high", "low", "mid"]
    assert categories(out) == ["high", "low", "mid"]


def test_filling_backward_keeps_the_categories(firepanda: ModuleType) -> None:
    """The mirror of the one above."""
    out = made(firepanda).bfill()
    assert out.dtype == "category"
    assert out.tolist() == ["low", "high", "high", "low", "mid"]


@needs_pandas
def test_filling_forward_agrees_with_pandas(firepanda: ModuleType) -> None:
    """Filling a categorical is a place pandas is careful and worth checking against."""
    ours, theirs_ = made(firepanda).ffill(), theirs().ffill()
    assert words(ours) == words(theirs_)
    assert categories(ours) == categories(theirs_)


def test_a_shift_is_still_a_category_column(firepanda: ModuleType) -> None:
    """A shift stacks a run of gap onto a slice, so the gap side carries no list."""
    out = made(firepanda).shift(1)
    assert out.dtype == "category"
    assert out.tolist() == [None, "low", None, "high", "low"]
    assert categories(out) == ["high", "low", "mid"]


def test_shifting_backwards_is_still_a_category_column(firepanda: ModuleType) -> None:
    """The gap is at the other end and the answer is the same kind of column."""
    out = made(firepanda).shift(-1)
    assert out.dtype == "category"
    assert out.tolist() == [None, "high", "low", "mid", None]


@needs_pandas
def test_a_shift_agrees_with_pandas(firepanda: ModuleType) -> None:
    """Including the categories, which a shift cannot change."""
    ours, theirs_ = made(firepanda).shift(1), theirs().shift(1)
    assert words(ours) == words(theirs_)
    assert categories(ours) == categories(theirs_)


def test_the_codes_still_line_up_after_a_move(firepanda: ModuleType) -> None:
    """The assertion a corrupt column would pass everything else and fail here.

    A column can hold the right codes and the wrong list, or the right codes and
    no list at all, and the only way to tell is to look up each code and see
    which word comes back.
    """
    out = made(firepanda).dropna()
    names = categories(out)
    assert [names[code] for code in out.cat.codes.tolist()] == out.tolist()


def test_a_category_nothing_points_at_survives_a_move(firepanda: ModuleType) -> None:
    """A filter that empties a category leaves it in the list rather than tidying it."""
    out = made(firepanda, ["low", "high", "high"]).head(1)
    assert categories(out) == ["high", "low"]
    assert out.cat.remove_unused_categories().cat.categories.tolist() == ["low"]
