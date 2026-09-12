"""How much memory a frame and a column are using.

Three members, `Series.nbytes` and `memory_usage` on both classes, and the interesting thing
about all three is that the number they answer is not the number pandas answers and cannot be.
firepanda counts the Arrow buffers the data is actually stored in. pandas counts the size of the
numpy representation, which for text is pointers and for an index nobody declared is the size of
three Python integers. Those are two different measurements rather than one measurement taken
twice, and the tests below assert the firepanda one on purpose rather than reaching for pandas.

What is checked against pandas is the shape of the answer and the rules around it: a column of
counts labelled by column name with the index first and called `Index`, `index=False` dropping
that row and nothing else, and `deep` being a parameter that exists.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

DATA: dict[str, list[Any]] = {"k": ["p", "qq", "rrr"], "v": [1, 2, 3], "f": [1.5, 2.5, 3.5]}
"""Text, integers and floats, so the three buffer shapes are all present."""


# ---------------------------------------------------------------------------
# What a column weighs
# ---------------------------------------------------------------------------


def test_a_column_reports_the_bytes_its_buffers_occupy(firepanda: ModuleType) -> None:
    """Three int64 values and the validity bitmap beside them, which is one byte."""
    assert firepanda.DataFrame(DATA)["v"].nbytes == 25


def test_a_wider_column_weighs_more_than_a_narrower_one(firepanda: ModuleType) -> None:
    """The property that makes the number worth having at all."""
    frame = firepanda.DataFrame({"small": [1, 2, 3], "big": [1.5, 2.5, 3.5]})
    assert frame["small"].astype("int8").nbytes < frame["big"].nbytes


def test_a_text_column_counts_the_text(firepanda: ModuleType) -> None:
    """The views and the payload, and not three pointers.

    This is the half of the divergence that matters. pandas reports the size of an array of
    references, so the strings themselves are outside its number, and here they are inside it.

    The arithmetic shows the other thing a string view does, which is that a string of up to
    twelve bytes lives inside its own view and costs no payload at all. Three one letter strings
    are three sixteen byte views and a bitmap byte and nothing else, and replacing one of them
    with a hundred letters moves that one out to the payload whole.
    """
    short = firepanda.DataFrame({"c": ["a", "b", "c"]})["c"].nbytes
    longer = firepanda.DataFrame({"c": ["a" * 100, "b", "c"]})["c"].nbytes
    assert short == 3 * 16 + 1
    assert longer - short == 100


def test_an_empty_column_weighs_nothing(firepanda: ModuleType) -> None:
    """No rows, no buffers, and no constant to report instead."""
    assert firepanda.DataFrame({"a": []})["a"].nbytes == 0


def test_nbytes_cannot_be_set(firepanda: ModuleType) -> None:
    """A property with no setter, so a typo is caught rather than kept."""
    with pytest.raises(AttributeError):
        firepanda.DataFrame(DATA)["v"].nbytes = 4  # type: ignore[misc]


# ---------------------------------------------------------------------------
# What a frame weighs, column by column
# ---------------------------------------------------------------------------


@needs_pandas
def test_a_frame_reports_one_row_per_column_labelled_by_name(firepanda: ModuleType) -> None:
    """The shape is pandas', including the index coming first under the name `Index`."""
    import pandas as pd

    made = firepanda.DataFrame(DATA).memory_usage()
    assert list(made.index) == list(pd.DataFrame(DATA).memory_usage().index)
    assert list(made.index) == ["Index", "k", "v", "f"]


def test_the_rows_are_the_columns_asked_one_at_a_time(firepanda: ModuleType) -> None:
    """The plural is the singular repeated, which is the only rule there is here."""
    frame = firepanda.DataFrame(DATA)
    made = frame.memory_usage(index=False)
    assert made.tolist() == [frame[name].nbytes for name in frame.columns]


@needs_pandas
def test_index_false_drops_the_index_row_and_changes_nothing_else(firepanda: ModuleType) -> None:
    """A column does not get bigger because the labels beside it were counted."""
    import pandas as pd

    frame = firepanda.DataFrame(DATA)
    with_index = frame.memory_usage()
    without = frame.memory_usage(index=False)
    assert list(without.index) == ["k", "v", "f"]
    assert without.tolist() == with_index.tolist()[1:]
    theirs = pd.DataFrame(DATA).memory_usage(index=False)
    assert list(without.index) == list(theirs.index)


def test_the_index_row_is_what_the_index_says_about_itself(firepanda: ModuleType) -> None:
    """Not a separate measurement, the one `Index.nbytes` already answers."""
    frame = firepanda.DataFrame(DATA).set_index("k")
    made = frame.memory_usage()
    assert made.tolist()[0] == frame.index.nbytes
    assert made.tolist()[0] > 0


def test_a_frame_that_declared_no_index_reports_nothing_for_it(firepanda: ModuleType) -> None:
    """Zero, because there are no labels, which is the honest answer and not pandas'.

    pandas says 132 there, which is the size of the three Python integers a `RangeIndex` holds
    and is a fact about a Python object rather than about any data. Copying it would mean
    reporting memory nobody allocated for labels nobody declared.
    """
    made = firepanda.DataFrame(DATA).memory_usage()
    assert made.tolist()[0] == 0


def test_the_counts_carry_no_name_of_their_own(firepanda: ModuleType) -> None:
    """Same rule as `dtypes`, and the same reason: `_labelled` leaves two names behind."""
    made = firepanda.DataFrame(DATA).memory_usage()
    assert made.name is None
    assert made.index.name is None


def test_a_frame_of_no_columns_reports_only_its_index(firepanda: ModuleType) -> None:
    """One row and no error, which is the shape a sum over the answer wants."""
    made = firepanda.DataFrame({}).memory_usage()
    assert list(made.index) == ["Index"]
    assert made.tolist() == [0]


# ---------------------------------------------------------------------------
# What a column weighs, which is a number rather than a column
# ---------------------------------------------------------------------------


def test_a_column_answers_one_number_where_a_frame_answers_a_column(
    firepanda: ModuleType,
) -> None:
    """The same name on both classes, shaped like the class it was asked of."""
    frame = firepanda.DataFrame(DATA)
    assert isinstance(frame["v"].memory_usage(), int)
    assert frame.memory_usage().size == 4


def test_a_column_counts_its_index_by_default_and_nbytes_does_not(firepanda: ModuleType) -> None:
    """The difference between the two members, which the names do not say.

    `memory_usage()` with nothing passed includes the labels and `nbytes` never does. That is
    pandas' rule, it is the opposite of what most readers would guess from the names, and it is
    the reason both members exist rather than one of them being enough.
    """
    frame = firepanda.DataFrame(DATA).set_index("k")
    column = frame["v"]
    assert column.memory_usage(index=False) == column.nbytes
    assert column.memory_usage() == column.nbytes + frame.index.nbytes
    assert column.memory_usage() > column.nbytes


def test_a_column_with_no_index_to_speak_of_answers_the_same_either_way(
    firepanda: ModuleType,
) -> None:
    """Where pandas' two answers differ by 132 and these two do not differ at all."""
    column = firepanda.DataFrame(DATA)["v"]
    assert column.memory_usage() == column.memory_usage(index=False) == column.nbytes


# ---------------------------------------------------------------------------
# The parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("deep", [True, False])
def test_deep_is_accepted_and_changes_nothing(firepanda: ModuleType, deep: bool) -> None:
    """There are no object columns here, so every number is already the deep one.

    Refusing the parameter would tell a caller that something is unavailable, when what is
    unavailable is the shallow answer.
    """
    frame = firepanda.DataFrame(DATA)
    assert frame.memory_usage(deep=deep).tolist() == frame.memory_usage().tolist()
    assert frame["k"].memory_usage(deep=deep) == frame["k"].memory_usage()


@needs_pandas
@pytest.mark.parametrize("flag", [1, 0, None, "", "no", 2.5])
def test_a_flag_that_is_not_a_boolean_is_read_for_its_truth(
    firepanda: ModuleType, flag: Any
) -> None:
    """These two parameters are not checked for being booleans, and that is deliberate.

    Almost every flag in this library goes through the same check and refuses a `1` with a
    sentence naming the type that arrived, because pandas validates almost every flag. These
    two it does not. `memory_usage` reads whatever arrives for its truth, so a `1` counts the
    index and a `0` and a `None` and an empty string do not, and `"no"` counts it because a
    non empty string is true. Copying that is the only way a program that works there works
    here, and the assertion below is against a running pandas rather than against a memory of
    one.
    """
    import pandas as pd

    frame = firepanda.DataFrame(DATA)
    theirs = list(pd.DataFrame(DATA).memory_usage(index=flag).index)
    assert list(frame.memory_usage(index=flag).index) == theirs
    counted = frame["v"].memory_usage(index=flag) > frame["v"].memory_usage(index=False)
    assert counted == ("Index" in theirs and frame.index.nbytes > 0)


def test_a_deep_that_is_not_a_boolean_changes_nothing_either(firepanda: ModuleType) -> None:
    """The flag is ignored, so the type of it cannot matter, and it still must not raise."""
    frame = firepanda.DataFrame(DATA)
    assert frame.memory_usage(deep="yes").tolist() == frame.memory_usage().tolist()
    assert frame["v"].memory_usage(deep=7) == frame["v"].memory_usage()


def test_none_is_a_flag_and_means_false(firepanda: ModuleType) -> None:
    """pandas allows it for these and reads it as False, so this does too."""
    frame = firepanda.DataFrame(DATA)
    assert list(frame.memory_usage(index=None).index) == ["k", "v", "f"]
