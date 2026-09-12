"""The members that describe a frame or a column rather than reading one.

`empty`, `ndim`, `size`, `shape`, `axes` and `dtypes` are the cheapest members in the library. Not
one of them touches a value, none of them can fail, and every one of them is a line. That is what
makes them worth a file of their own, because the cheap members are the ones nobody thinks about
and they are the ones every other library reads first.

Two of them are the whole point of the file. `empty` is not `len(df) == 0`, it is whether either
axis is empty, so a frame of two columns holding no rows is empty and so is a frame of no columns
at all. And `axes` exists on a class with one axis, where it answers a list of one, which reads
like a mistake and is the reason it exists: code that walks `obj.axes` works on both classes
without asking which one it has.

`dtypes` is the only one with a shape worth arguing about, and the argument is that a frame of
forty columns is asked what it holds so that the names can be read off beside the types, which a
list makes the caller do by hand.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

DATA: dict[str, list[Any]] = {"a": [1, 2, 3], "b": [1.5, 2.5, 3.5], "c": ["x", "y", "z"]}
"""Three columns of three different kinds, so `dtypes` has something to say."""


# ---------------------------------------------------------------------------
# How big it is
# ---------------------------------------------------------------------------


@needs_pandas
@pytest.mark.parametrize("name", ["empty", "ndim", "size", "shape"])
def test_a_frame_measures_itself_the_way_pandas_measures_it(
    firepanda: ModuleType, name: str
) -> None:
    """Four numbers about a frame, all four of them pandas'."""
    import pandas as pd

    assert getattr(firepanda.DataFrame(DATA), name) == getattr(pd.DataFrame(DATA), name)


@needs_pandas
@pytest.mark.parametrize("name", ["empty", "ndim", "size", "shape"])
def test_a_column_measures_itself_the_way_pandas_measures_it(
    firepanda: ModuleType, name: str
) -> None:
    """The same four on a column, where two of them have different answers."""
    import pandas as pd

    assert getattr(firepanda.DataFrame(DATA)["a"], name) == getattr(pd.DataFrame(DATA)["a"], name)


def test_a_frame_is_two_dimensional_and_a_column_is_one(firepanda: ModuleType) -> None:
    """The pair of constants, asserted together because the point is that they differ."""
    frame = firepanda.DataFrame(DATA)
    assert frame.ndim == 2
    assert frame["a"].ndim == 1


def test_size_is_cells_on_a_frame_and_rows_on_a_column(firepanda: ModuleType) -> None:
    """Three rows and three columns is nine, which is the trap in the name.

    A caller who reads `size` as a row count is right on a column and wrong on a frame, and there
    is no error to tell them, which is why it is worth one test that says the number out loud.
    """
    frame = firepanda.DataFrame(DATA)
    assert frame.size == 9
    assert frame["a"].size == 3
    assert len(frame) == 3


# ---------------------------------------------------------------------------
# Empty is about axes, not about rows
# ---------------------------------------------------------------------------


@needs_pandas
def test_a_frame_of_columns_with_no_rows_is_empty(firepanda: ModuleType) -> None:
    """The rule the name does not say, which is that either axis being empty is enough."""
    import pandas as pd

    made = firepanda.DataFrame({"a": [], "b": []})
    assert made.empty is True
    assert made.shape == (0, 2)
    assert made.empty == pd.DataFrame({"a": [], "b": []}).empty


@needs_pandas
def test_a_frame_of_no_columns_is_empty(firepanda: ModuleType) -> None:
    """The other axis, which is the one a caller reaches by filtering all the columns out."""
    import pandas as pd

    made = firepanda.DataFrame({})
    assert made.empty is True
    assert made.shape == (0, 0)
    assert made.size == 0
    assert made.ndim == 2
    assert made.empty == pd.DataFrame({}).empty


def test_a_frame_with_rows_in_it_is_not_empty(firepanda: ModuleType) -> None:
    """The other half, so the property is not passing by answering true to everything."""
    assert firepanda.DataFrame(DATA).empty is False
    assert firepanda.DataFrame(DATA)["a"].empty is False


@needs_pandas
def test_a_column_of_no_rows_is_empty(firepanda: ModuleType) -> None:
    """A column has one axis, so there is only the one way to be empty."""
    import pandas as pd

    made = firepanda.DataFrame({"a": []})["a"]
    assert made.empty is True
    assert made.size == 0
    assert made.empty == pd.DataFrame({"a": []})["a"].empty


def test_empty_cannot_be_set(firepanda: ModuleType) -> None:
    """A property with no setter, the same as pandas, so a typo is caught rather than kept."""
    frame = firepanda.DataFrame(DATA)
    with pytest.raises(AttributeError):
        frame.empty = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The axes, in the order the axis numbers name them
# ---------------------------------------------------------------------------


@needs_pandas
def test_the_axes_of_a_frame_are_the_rows_then_the_columns(firepanda: ModuleType) -> None:
    """The order is the one `shape` reads in and the one `axis=0` and `axis=1` name."""
    import pandas as pd

    made = firepanda.DataFrame(DATA).axes
    theirs = pd.DataFrame(DATA).axes
    assert len(made) == len(theirs) == 2
    assert made[0].tolist() == theirs[0].tolist()
    assert list(made[1]) == theirs[1].tolist()


def test_the_first_axis_of_a_frame_is_the_index_it_was_given(firepanda: ModuleType) -> None:
    """Not a range built on the spot, the labels the frame actually carries, name and all."""
    made = firepanda.DataFrame({"k": ["p", "q"], "v": [1, 2]}).set_index("k")
    assert made.axes[0].tolist() == ["p", "q"]
    assert made.axes[0].name == "k"


def test_the_axes_of_a_column_are_a_list_of_one(firepanda: ModuleType) -> None:
    """The shape that reads like a mistake and is the reason the member exists."""
    column = firepanda.DataFrame({"k": ["p", "q"], "v": [1, 2]}).set_index("k")["v"]
    assert len(column.axes) == 1
    assert column.axes[0].tolist() == ["p", "q"]


def test_the_axes_of_either_class_can_be_walked_without_asking_which_it_is(
    firepanda: ModuleType,
) -> None:
    """The loop the member is for, written once over both classes."""
    frame = firepanda.DataFrame(DATA)
    assert [len(axis) for axis in frame.axes] == [3, 3]
    assert [len(axis) for axis in frame["a"].axes] == [3]


def test_the_first_axis_is_the_index_and_the_second_is_what_columns_answers(
    firepanda: ModuleType,
) -> None:
    """`axes` is not a new answer, it is the two existing ones in a list."""
    frame = firepanda.DataFrame(DATA)
    assert frame.axes[0].tolist() == frame.index.tolist()
    assert list(frame.axes[1]) == frame.columns
    assert frame["a"].axes[0].tolist() == frame["a"].index.tolist()


# ---------------------------------------------------------------------------
# What each column holds
# ---------------------------------------------------------------------------


@needs_pandas
def test_the_types_of_a_frame_are_labelled_by_column_name(firepanda: ModuleType) -> None:
    """A column of answers rather than a list, so the names can be read off beside them."""
    import pandas as pd

    made = firepanda.DataFrame(DATA).dtypes
    assert made.index.tolist() == pd.DataFrame(DATA).dtypes.index.tolist()
    assert made.tolist() == ["int64", "float64", "string"]


@needs_pandas
def test_the_types_carry_no_name_of_their_own(firepanda: ModuleType) -> None:
    """The column is built by making a frame and moving a column into the index.

    That route leaves two names behind, one on the column and one on the index, and neither is a
    name a caller asked for, so both are taken off. pandas leaves both empty and so does this.
    """
    import pandas as pd

    made = firepanda.DataFrame(DATA).dtypes
    theirs = pd.DataFrame(DATA).dtypes
    assert made.name is None
    assert made.index.name is None
    assert (theirs.name, theirs.index.name) == (None, None)


def test_the_types_of_a_frame_agree_with_asking_each_column(firepanda: ModuleType) -> None:
    """The plural is the singular repeated, which is the only rule there is here."""
    frame = firepanda.DataFrame(DATA)
    made = frame.dtypes
    assert [made[name] for name in frame.columns] == [frame[name].dtype for name in frame.columns]


def test_a_column_answers_its_own_type_to_the_plural_name(firepanda: ModuleType) -> None:
    """`dtypes` on a thing with one type is `dtype`, which is pandas' rule and reads oddly."""
    column = firepanda.DataFrame(DATA)["a"]
    assert column.dtypes == column.dtype == "int64"


@needs_pandas
def test_a_frame_of_no_rows_still_says_what_it_holds(firepanda: ModuleType) -> None:
    """The types belong to the columns rather than to the rows, so an empty frame has them."""
    import pandas as pd

    made = firepanda.DataFrame({"a": [], "b": []}).dtypes
    assert made.index.tolist() == ["a", "b"]
    assert made.tolist() == [str(held) for held in pd.DataFrame({"a": [], "b": []}).dtypes]


def test_a_frame_of_no_columns_has_no_types(firepanda: ModuleType) -> None:
    """Nothing to say and nothing raised, which is the shape an empty loop wants."""
    made = firepanda.DataFrame({}).dtypes
    assert made.tolist() == []
    assert made.size == 0


def test_a_category_column_says_category(firepanda: ModuleType) -> None:
    """The one type whose name is not the name of a storage kind, checked once."""
    frame = firepanda.DataFrame({"a": ["x", "y"]})
    assert frame["a"].astype("category").dtype == "category"


@needs_pandas
def test_the_types_are_strings_where_pandas_hands_back_objects(firepanda: ModuleType) -> None:
    """The divergence `dtype` already carries, written down once in its new place.

    pandas answers a numpy dtype or an extension dtype, which prints as its own name and can be
    compared against a string. This answers the string. The two agree under `str` for every type
    but text, where pandas 3 says `str` and this says `string`, and that difference belongs to
    `dtype` rather than to this member.
    """
    import pandas as pd

    made = firepanda.DataFrame(DATA).dtypes
    theirs = pd.DataFrame(DATA).dtypes
    assert all(isinstance(held, str) for held in made.tolist())
    assert made.tolist()[:2] == [str(held) for held in theirs.tolist()[:2]]
