"""Comparing a category column, checked against a running pandas.

Six rules, and every one of them is measured here rather than asserted against a
constant. Equality works whether the categories are ordered or not. An ordering
comparison needs them ordered. A scalar that is not one of the categories is all
false under equality and a `TypeError` under an ordering. Two categoricals have
to agree about their categories exactly. A categorical against a plain text
column compares by value under equality and is refused under an ordering. And
the ordering follows the categories rather than the words, which is the one a
caller reaches for `ordered=True` to get.

One difference is asserted rather than worked around. A missing row answers
missing here and false in pandas, because a comparison against a value that is
not there has no answer and firepanda has somewhere to say so. That is the same
three valued logic every other firepanda column already has, and a categorical
answering differently from the text it decodes to would be the surprising thing.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

WORDS = ["bolt", "anchor", "rivet"]
"""Three rows over three categories, deliberately not in alphabetical order."""

SIZES = ["medium", "large", "small"]
"""Three rows whose word order and whose meaning order disagree."""

ORDER = ["small", "medium", "large"]
"""What the sizes mean, which is not what they sort as."""


def made(firepanda: ModuleType, values: list[Any] = WORDS, ordered: bool = False) -> Any:
    """A firepanda category column."""
    column = firepanda.Series(values).astype("category")
    return column.cat.as_ordered() if ordered else column


def theirs(values: list[Any] = WORDS, ordered: bool = False) -> Any:
    """The same column in pandas."""
    import pandas as pd

    column = pd.Series(values).astype("category")
    return column.cat.as_ordered() if ordered else column


def message(call: Any) -> str:
    """Runs something that is expected to raise and returns what it said."""
    try:
        call()
    except Exception as error:
        return str(error)
    raise AssertionError("expected a refusal and got an answer")


@needs_pandas
def test_equality_against_a_category_agrees_with_pandas(firepanda: ModuleType) -> None:
    """The ordinary case, and it works without the categories being ordered."""
    assert made(firepanda).eq("bolt").tolist() == theirs().eq("bolt").tolist()
    assert made(firepanda).eq("bolt").tolist() == [True, False, False]


@needs_pandas
def test_inequality_against_a_category_agrees_with_pandas(firepanda: ModuleType) -> None:
    """The mirror of the one above, and the same rule."""
    assert made(firepanda).ne("bolt").tolist() == theirs().ne("bolt").tolist()


@needs_pandas
def test_an_ordering_on_an_unordered_column_says_what_pandas_says(
    firepanda: ModuleType,
) -> None:
    """The message is the pandas message, because programs match on it."""
    mine = made(firepanda)
    them = theirs()
    assert message(lambda: mine < "bolt") == message(lambda: them < "bolt")
    with pytest.raises(TypeError, match="Unordered Categoricals"):
        _ = mine < "bolt"


@needs_pandas
def test_an_ordering_runs_once_the_categories_are_ordered(firepanda: ModuleType) -> None:
    """What `ordered=True` buys, in one line."""
    mine = made(firepanda, ordered=True)
    assert (mine < "bolt").tolist() == (theirs(ordered=True) < "bolt").tolist()
    assert (mine < "bolt").tolist() == [False, True, False]


@needs_pandas
def test_the_four_orderings_all_agree_with_pandas(firepanda: ModuleType) -> None:
    """One test for `lt`, `le`, `gt` and `ge` rather than four almost identical ones."""
    mine = made(firepanda, ordered=True)
    them = theirs(ordered=True)
    for name in ("lt", "le", "gt", "ge"):
        assert getattr(mine, name)("bolt").tolist() == getattr(them, name)("bolt").tolist()


@needs_pandas
def test_a_scalar_that_is_not_a_category_is_equal_to_no_row(firepanda: ModuleType) -> None:
    """Equality can answer without the scalar being a category, and answers false."""
    assert made(firepanda).eq("washer").tolist() == theirs().eq("washer").tolist()
    assert made(firepanda).eq("washer").tolist() == [False, False, False]


@needs_pandas
def test_a_scalar_that_is_not_a_category_is_unequal_to_every_row(
    firepanda: ModuleType,
) -> None:
    """The mirror, and it is all true rather than an error."""
    assert made(firepanda).ne("washer").tolist() == theirs().ne("washer").tolist()


@needs_pandas
def test_an_ordering_against_a_scalar_that_is_not_a_category_is_refused(
    firepanda: ModuleType,
) -> None:
    """The asymmetry. There is no position to compare against, so there is no answer."""
    mine = made(firepanda, ordered=True)
    them = theirs(ordered=True)
    assert message(lambda: mine > "washer") == message(lambda: them > "washer")
    with pytest.raises(TypeError, match="Invalid comparison"):
        _ = mine > "washer"


@needs_pandas
def test_a_number_against_a_text_categorical_is_equal_to_nothing(
    firepanda: ModuleType,
) -> None:
    """A scalar of the wrong kind is not a category either, so the same rule applies."""
    assert made(firepanda).eq(5).tolist() == theirs().eq(5).tolist()


@needs_pandas
def test_the_unordered_refusal_is_the_one_reported_when_both_are_wrong(
    firepanda: ModuleType,
) -> None:
    """pandas reports the ordering first and so does firepanda."""
    mine = made(firepanda)
    assert message(lambda: mine < "washer") == message(lambda: theirs() < "washer")


@needs_pandas
def test_two_columns_with_the_same_categories_compare(firepanda: ModuleType) -> None:
    """Codes against codes, which is what makes this cheap."""
    left = made(firepanda, ["bolt", "anchor", "rivet"])
    right = made(firepanda, ["bolt", "rivet", "anchor"])
    assert left.eq(right).tolist() == [True, False, False]


@needs_pandas
def test_two_columns_that_disagree_about_their_categories_are_refused(
    firepanda: ModuleType,
) -> None:
    """A code means nothing outside the list it indexes."""
    left = made(firepanda, ["bolt", "anchor"])
    right = made(firepanda, ["bolt", "rivet"])
    with pytest.raises(TypeError, match="categories"):
        _ = left.eq(right)


@needs_pandas
def test_a_categorical_against_plain_text_compares_by_value(firepanda: ModuleType) -> None:
    """The one shape where a comparison has to decode, because there is nothing else."""
    left = made(firepanda, ["bolt", "anchor", "rivet"])
    right = firepanda.Series(["bolt", "rivet", "rivet"])
    assert left.eq(right).tolist() == theirs().eq(["bolt", "rivet", "rivet"]).tolist()


@needs_pandas
def test_an_ordering_against_plain_text_is_refused(firepanda: ModuleType) -> None:
    """pandas refuses this and says so, and the searchable part of the message is kept."""
    left = made(firepanda, ["bolt", "anchor"], ordered=True)
    right = firepanda.Series(["bolt", "rivet"])
    with pytest.raises(TypeError, match="Cannot compare a Categorical"):
        _ = left < right


@needs_pandas
def test_the_ordering_follows_the_categories_and_not_the_words(
    firepanda: ModuleType,
) -> None:
    """The whole point of an ordered categorical, and it disagrees with sorting."""
    mine = made(firepanda, SIZES).cat.set_categories(ORDER, ordered=True)
    them = theirs(SIZES).cat.set_categories(ORDER, ordered=True)
    assert (mine < "large").tolist() == (them < "large").tolist()
    assert (mine < "large").tolist() == [True, False, True]
    assert (firepanda.Series(SIZES) < "large").tolist() == [False, False, False]


@needs_pandas
def test_a_missing_row_answers_missing_where_pandas_answers_false(
    firepanda: ModuleType,
) -> None:
    """The one asserted divergence, and it is the library wide one rather than a new one.

    A comparison against a value that is not there has no answer. pandas has to
    say false because its result is a numpy bool array with no room for absence,
    and firepanda says None the way it already does for text and for numbers.
    """
    mine = firepanda.Series(["bolt", None, "anchor"]).astype("category")
    them = theirs(["bolt", None, "anchor"])
    assert mine.eq("bolt").tolist() == [True, None, False]
    assert them.eq("bolt").tolist() == [True, False, False]
    assert firepanda.Series(["bolt", None, "anchor"]).eq("bolt").tolist() == [
        True,
        None,
        False,
    ]


@needs_pandas
def test_an_imported_categorical_compares_the_same(firepanda: ModuleType) -> None:
    """A pandas categorical arrives with int8 codes, and nothing here notices."""
    import pandas as pd

    over = firepanda.from_arrow(pd.DataFrame({"part": theirs()}))
    assert over["part"].eq("bolt").tolist() == theirs().eq("bolt").tolist()
