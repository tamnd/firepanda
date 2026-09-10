"""The `cat` accessor, checked against a running pandas.

Eleven names and three doors under them, so the tests are mostly about the
arithmetic in between: which list gets built, in what order, and which of the
disagreements a caller can cause is a `ValueError` rather than a quiet answer.
Every one of those is measured against pandas rather than against a constant,
for the reason `test_astype.py` gives at more length.

Two differences are asserted rather than worked around, because both are
decisions somebody made on purpose.

The first is the width of the codes. pandas answers int8 for a column of three
categories and firepanda answers int32, which document 26 argues for and which a
caller can see through `Series.cat.codes.dtype`. The values agree, so a program
comparing codes to codes is fine and a program comparing dtypes is not, and it
should find that out here rather than in production.

The second is what a missing row reads back as. pandas puts a NaN in the hole
because a categorical's missing rows come out through numpy, and firepanda says
None because Arrow has a validity bitmap and a value that is absent is not a
value that is a float. That is the same difference every other column has and it
is argued in `firepanda/py/values.mojo`.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

WORDS = ["rivet", "bolt", "rivet", "anchor"]
"""Four rows over three categories, with one of them used twice.

The repeat is what makes the codes worth asserting: a column where every row has
its own category cannot tell a code apart from a row number. The words are also
deliberately not in alphabetical order, so a test that passes has established
that the categories come out sorted rather than in the order they appeared.
"""


def made(firepanda: ModuleType, values: list[Any] = WORDS) -> Any:
    """A firepanda category column."""
    return firepanda.Series(values).astype("category")


def theirs(values: list[Any] = WORDS) -> Any:
    """The same column in pandas."""
    import pandas as pd

    return pd.Series(values).astype("category")


def like(mine: list[Any], them: list[Any]) -> bool:
    """Compares two columns of values, reading a NaN as a None.

    The one difference this file allows through without asserting it row by row,
    since it is the library wide missing value rule rather than anything about
    categories.
    """
    if len(mine) != len(them):
        return False
    for one, other in zip(mine, them, strict=True):
        if one is None:
            if other is None or other != other:
                continue
            return False
        if one != other:
            return False
    return True


@needs_pandas
def test_the_categories_are_the_ones_pandas_finds_in_the_same_order(
    firepanda: ModuleType,
) -> None:
    """Sorted, and sorted is not the order the words appear in."""
    assert made(firepanda).cat.categories.tolist() == theirs().cat.categories.tolist()
    assert made(firepanda).cat.categories.tolist() == ["anchor", "bolt", "rivet"]


@needs_pandas
def test_the_codes_are_the_ones_pandas_writes(firepanda: ModuleType) -> None:
    """The values agree. The width does not, and that is document 26's decision."""
    assert made(firepanda).cat.codes.tolist() == theirs().cat.codes.tolist()
    assert made(firepanda).cat.codes.dtype == "int32"
    assert str(theirs().cat.codes.dtype) == "int8"


@needs_pandas
def test_the_codes_carry_the_row_labels_and_not_the_column_name(
    firepanda: ModuleType,
) -> None:
    """Both of those are pandas, and the second one reads like an oversight there.

    The column name is dropped here and pandas drops it too. What is asserted is
    the empty string rather than the None pandas answers, because a firepanda
    series with no name reports an empty string everywhere and not only here.
    That is a difference of its own and it is filed rather than papered over in
    this test, since fixing it in one place would make the accessor disagree with
    the rest of the library.
    """
    mine = firepanda.Series(WORDS, name="part").astype("category")
    assert mine.name == "part"
    assert mine.cat.codes.name == ""
    assert theirs().cat.codes.name is None
    assert len(mine.cat.codes) == len(mine)


@needs_pandas
def test_a_bare_category_is_not_ordered(firepanda: ModuleType) -> None:
    """`astype("category")` cannot say otherwise, in either library."""
    assert made(firepanda).cat.ordered is False
    assert theirs().cat.ordered is False


@needs_pandas
def test_the_order_can_be_turned_on_and_off_again(firepanda: ModuleType) -> None:
    """And nothing else about the column moves while it happens."""
    ordered = made(firepanda).cat.as_ordered()
    assert ordered.cat.ordered is True
    assert theirs().cat.as_ordered().cat.ordered is True
    assert ordered.cat.codes.tolist() == made(firepanda).cat.codes.tolist()
    assert ordered.cat.as_unordered().cat.ordered is False
    assert theirs().cat.as_ordered().cat.as_unordered().cat.ordered is False


@needs_pandas
def test_a_category_can_be_added_with_nothing_in_it(firepanda: ModuleType) -> None:
    """It goes on the end, and no row changes."""
    mine = made(firepanda).cat.add_categories(["washer"])
    them = theirs().cat.add_categories(["washer"])
    assert mine.cat.categories.tolist() == them.cat.categories.tolist()
    assert like(mine.tolist(), list(them))


@needs_pandas
def test_a_category_that_is_already_there_cannot_be_added(firepanda: ModuleType) -> None:
    """pandas names the clash in the message and so does this."""
    with pytest.raises(ValueError, match="must not include old categories"):
        made(firepanda).cat.add_categories(["bolt"])
    with pytest.raises(ValueError, match="must not include old categories"):
        theirs().cat.add_categories(["bolt"])


@needs_pandas
def test_removing_a_category_makes_its_rows_missing(firepanda: ModuleType) -> None:
    """Which is the whole difference between removing one and renaming one."""
    mine = made(firepanda).cat.remove_categories(["bolt"])
    them = theirs().cat.remove_categories(["bolt"])
    assert like(mine.tolist(), list(them))
    assert mine.cat.categories.tolist() == them.cat.categories.tolist()
    assert mine.tolist()[1] is None


@needs_pandas
def test_a_category_that_is_not_there_cannot_be_removed(firepanda: ModuleType) -> None:
    """A typo in a category name is a mistake rather than a no op."""
    with pytest.raises(ValueError, match="removals must all be in old categories"):
        made(firepanda).cat.remove_categories(["washer"])
    with pytest.raises(ValueError, match="removals must all be in old categories"):
        theirs().cat.remove_categories(["washer"])


@needs_pandas
def test_the_unused_categories_can_be_dropped(firepanda: ModuleType) -> None:
    """The ones that are used keep the order they were in, rather than being resorted."""
    mine = made(firepanda).cat.add_categories(["washer"]).cat.remove_unused_categories()
    them = theirs().cat.add_categories(["washer"]).cat.remove_unused_categories()
    assert mine.cat.categories.tolist() == them.cat.categories.tolist()
    assert "washer" not in mine.cat.categories.tolist()
    assert like(mine.tolist(), list(them))


@needs_pandas
def test_renaming_the_categories_keeps_every_row_where_it_was(
    firepanda: ModuleType,
) -> None:
    """The one operation decided by position rather than by value."""
    mine = made(firepanda).cat.rename_categories(["A", "B", "C"])
    them = theirs().cat.rename_categories(["A", "B", "C"])
    assert like(mine.tolist(), list(them))
    assert mine.cat.codes.tolist() == made(firepanda).cat.codes.tolist()


@needs_pandas
def test_a_rename_can_be_written_as_a_mapping(firepanda: ModuleType) -> None:
    """A category the mapping does not name keeps the label it had."""
    mine = made(firepanda).cat.rename_categories({"bolt": "B"})
    them = theirs().cat.rename_categories({"bolt": "B"})
    assert like(mine.tolist(), list(them))
    assert mine.cat.categories.tolist() == them.cat.categories.tolist()


@needs_pandas
def test_a_rename_can_be_written_as_a_function(firepanda: ModuleType) -> None:
    """Applied to each label, which is how pandas takes it too."""
    mine = made(firepanda).cat.rename_categories(str.upper)
    them = theirs().cat.rename_categories(str.upper)
    assert like(mine.tolist(), list(them))


@needs_pandas
def test_a_rename_needs_one_label_per_category(firepanda: ModuleType) -> None:
    """A short list is a mistake here and is deliberate under `set_categories`."""
    with pytest.raises(ValueError, match="same number of items"):
        made(firepanda).cat.rename_categories(["only"])
    with pytest.raises(ValueError, match="same number of items"):
        theirs().cat.rename_categories(["only"])


@needs_pandas
def test_reordering_the_categories_moves_the_codes_to_match(
    firepanda: ModuleType,
) -> None:
    """The values do not move, which is what makes this the value door."""
    wanted = ["rivet", "bolt", "anchor"]
    mine = made(firepanda).cat.reorder_categories(wanted)
    them = theirs().cat.reorder_categories(wanted)
    assert mine.cat.codes.tolist() == them.cat.codes.tolist()
    assert like(mine.tolist(), list(them))
    assert mine.cat.categories.tolist() == wanted


@needs_pandas
def test_reordering_has_to_name_the_same_categories(firepanda: ModuleType) -> None:
    """Otherwise it is a `set_categories` and the caller should say so."""
    with pytest.raises(ValueError, match="not the same as in old categories"):
        made(firepanda).cat.reorder_categories(["rivet", "bolt", "washer"])
    with pytest.raises(ValueError, match="not the same as in old categories"):
        theirs().cat.reorder_categories(["rivet", "bolt", "washer"])


@needs_pandas
def test_reordering_can_say_the_order_means_something(firepanda: ModuleType) -> None:
    """The one place `ordered` is a keyword rather than its own method."""
    wanted = ["rivet", "bolt", "anchor"]
    assert made(firepanda).cat.reorder_categories(wanted, ordered=True).cat.ordered
    assert theirs().cat.reorder_categories(wanted, ordered=True).cat.ordered


@needs_pandas
def test_setting_the_categories_drops_the_rows_it_leaves_out(
    firepanda: ModuleType,
) -> None:
    """Which is `remove_categories` without having to name what is going."""
    mine = made(firepanda).cat.set_categories(["rivet"])
    them = theirs().cat.set_categories(["rivet"])
    assert like(mine.tolist(), list(them))
    assert mine.cat.categories.tolist() == ["rivet"]


@needs_pandas
def test_setting_the_categories_with_rename_is_by_position(
    firepanda: ModuleType,
) -> None:
    """A short list drops the categories off the end and nulls their rows."""
    mine = made(firepanda).cat.set_categories(["x"], rename=True)
    them = theirs().cat.set_categories(["x"], rename=True)
    assert like(mine.tolist(), list(them))
    assert mine.cat.categories.tolist() == ["x"]


@needs_pandas
def test_a_rename_with_a_longer_list_leaves_the_extra_ones_unused(
    firepanda: ModuleType,
) -> None:
    """The other half of what `rename=True` means, and the half nobody expects."""
    wanted = ["w", "x", "y", "z"]
    mine = made(firepanda).cat.set_categories(wanted, rename=True)
    them = theirs().cat.set_categories(wanted, rename=True)
    assert mine.cat.categories.tolist() == them.cat.categories.tolist() == wanted
    assert like(mine.tolist(), list(them))


@needs_pandas
def test_a_category_list_that_repeats_itself_is_refused(firepanda: ModuleType) -> None:
    """With the pandas message, since a program matching on its text exists."""
    with pytest.raises(ValueError, match="categories must be unique"):
        made(firepanda).cat.set_categories(["rivet", "rivet"])
    with pytest.raises(ValueError, match="categories must be unique"):
        theirs().cat.set_categories(["rivet", "rivet"])


@needs_pandas
def test_the_accessor_refuses_a_column_that_is_not_a_category(
    firepanda: ModuleType,
) -> None:
    """An AttributeError rather than a type error, which is pandas and is right.

    A caller who guards with `hasattr` should be handed a False rather than an
    exception, and that only works if the accessor is what refuses.
    """
    with pytest.raises(AttributeError, match=r"Can only use \.cat accessor"):
        _ = firepanda.Series([1.0, 2.0]).cat
    with pytest.raises(AttributeError, match=r"Can only use \.cat accessor"):
        _ = theirs(WORDS).astype("str").cat


@needs_pandas
def test_a_missing_row_stays_missing_through_every_one_of_them(
    firepanda: ModuleType,
) -> None:
    """A null is not a category, so nothing on the accessor can give it one."""
    values = ["rivet", None, "bolt"]
    mine = made(firepanda, values)
    them = theirs(values)
    assert mine.cat.categories.tolist() == them.cat.categories.tolist()
    assert like(mine.cat.add_categories(["washer"]).tolist(), list(them))
    assert like(mine.cat.as_ordered().tolist(), list(them))
    assert like(mine.cat.rename_categories(["a", "b"]).tolist(), ["b", None, "a"])


@needs_pandas
def test_a_missing_row_has_no_code_rather_than_a_code_of_minus_one(
    firepanda: ModuleType,
) -> None:
    """The third difference on this accessor, and the one worth staring at.

    pandas writes -1 for a row whose category is missing, because its codes are a
    numpy integer array and there is nowhere else to record absence. Firepanda's
    codes are an Arrow column with a validity bitmap, so the code is missing and
    reads back as None.

    That matters more than it looks. A program that filters on `codes >= 0` is
    doing the pandas idiom and gets nothing here, and a program that takes the
    mean of the codes gets a different answer in each library. Both of those are
    better found by a failing comparison than by a number that is quietly wrong.
    """
    mine = made(firepanda, ["rivet", None, "bolt"])
    them = theirs(["rivet", None, "bolt"])
    assert mine.cat.codes.tolist() == [1, None, 0]
    assert them.cat.codes.tolist() == [1, -1, 0]


@needs_pandas
def test_the_accessor_class_is_reachable_off_the_class(firepanda: ModuleType) -> None:
    """What the conformance board reads, and the reason `Namespace` is a descriptor."""
    assert firepanda.Series.cat.__name__ == "CategoricalAccessor"
    assert isinstance(made(firepanda).cat, firepanda.Series.cat)


@needs_pandas
def test_the_accessor_has_the_names_pandas_has_and_no_others(
    firepanda: ModuleType,
) -> None:
    """The board counts what is there, so an extra name here would be a claim."""
    from pandas.core.arrays.categorical import CategoricalAccessor

    mine = {name for name in dir(firepanda.Series.cat) if not name.startswith("_")}
    them = {name for name in dir(CategoricalAccessor) if not name.startswith("_")}
    assert mine == them
