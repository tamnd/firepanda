"""`Series.isin` and `DataFrame.isin`, which are a kernel and a pile of rules about sets.

The lookup itself is a hash table in `firepanda/kernel/member.mojo` and it is tested in
`tests/test_member.mojo`, where it is checked against a scalar twin on both sides of the threshold
that picks between the table and the scan. None of that is here. What is here is everything the
kernel cannot decide, and it turns out to be most of the method.

The kernel compares one type against one type and refuses a mismatch. pandas compares by value and
never refuses, so `Series([1, 2]).isin(["1"])` finds nothing and says nothing about it. The gap
between those two is closed above the kernel by dropping the values the column cannot hold, on the
grounds that a value a column cannot hold is a value none of its rows can equal, and by converting
the two cases where pandas does find a row across a kind boundary, which are a flag against a
column of numbers and a zero or a one against a column of flags.

The kernel answers null for a null row, because it is describing SQL's `IN` and an unknown compared
against anything is unknown. pandas answers false unless the set holds that dtype's own missing
value, and there are four of those and firepanda has one. The third section is that rule.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

NAN = float("nan")
"""Spelt once, because a nan written inline reads like a typo."""


@needs_pandas
@pytest.mark.parametrize(
    ("values", "wanted"),
    [
        ([1, 2, 3, 2], [2]),
        ([1, 2, 3, 2], [2, 3]),
        ([1, 2, 3, 2], []),
        ([1, 2, 3, 2], [9]),
        ([1.5, 2.5, 3.5], [2.5]),
        (["a", "b", "c"], ["a", "c"]),
        (["a", "b", "c"], ["z"]),
        ([True, False, True], [True]),
    ],
)
def test_a_series_answers_what_pandas_answers(
    firepanda: ModuleType, values: list[Any], wanted: list[Any]
) -> None:
    """The whole point, over the four types a column can be and a set that misses."""
    import pandas as pd

    assert firepanda.Series(values).isin(wanted).tolist() == pd.Series(values).isin(wanted).tolist()


@needs_pandas
def test_the_answer_is_boolean_and_keeps_the_name_and_the_labels(firepanda: ModuleType) -> None:
    """A mask nobody can line up against its rows is not an answer."""
    made = firepanda.Series([1, 2, 3], name="n").isin([2])
    assert str(made.dtype) == "bool"
    assert made.name == "n"
    assert made.index.tolist() == [0, 1, 2]


@needs_pandas
@pytest.mark.parametrize(
    ("values", "wanted"),
    [
        ([1, 2, 3], [1.0]),
        ([1, 2, 3], [1.5]),
        ([1, 2, 3], ["1"]),
        ([1, 2, 3], [True]),
        ([1.0, 2.0], [2]),
        ([True, False], [1]),
        ([True, False], [0]),
        ([True, False], [1.0]),
        ([True, False], [2]),
        (["a", "b"], [1]),
        (["a", "b"], [True]),
        ([1, 2, 3], [2, "b", 9.5, None]),
    ],
)
def test_a_value_of_another_kind_is_found_or_missed_the_way_pandas_finds_it(
    firepanda: ModuleType, values: list[Any], wanted: list[Any]
) -> None:
    """The rule the kernel does not have, which is that a set may hold anything at all.

    The last row is the one that matters most, because it is the shape real code passes: a set with
    one value of the column's own kind in it and three that are not. pandas finds the one and
    ignores the rest, and a refusal here would be a refusal of a call that works.
    """
    import pandas as pd

    assert firepanda.Series(values).isin(wanted).tolist() == pd.Series(values).isin(wanted).tolist()


@needs_pandas
@pytest.mark.parametrize(
    ("values", "wanted", "answer"),
    [
        (["a", None], [None], [False, True]),
        (["a", None], [NAN], [False, True]),
        (["a", None], ["a"], [True, False]),
        ([1.0, None], [NAN], [False, True]),
        ([1.0, None], [None], [False, False]),
        ([1.0, None], [1.0], [True, False]),
        ([True, None], [None], [False, False]),
        ([True, None], [True], [True, False]),
    ],
)
def test_a_missing_row_is_in_the_set_only_when_the_set_says_it_is(
    firepanda: ModuleType, values: list[Any], wanted: list[Any], answer: list[bool]
) -> None:
    """pandas has a missing value per dtype and firepanda has one, so the set is what is read.

    A column of words finds both `None` and a nan, a column of numbers finds only the nan, and a
    column of flags finds neither. Those are pandas' answers for `object`, `float64` and `boolean`,
    and every one of them is reached here by looking at what is in the set rather than at what is
    in the column, because the column cannot tell them apart.
    """
    assert firepanda.Series(values).isin(wanted).tolist() == answer


@needs_pandas
def test_a_missing_row_is_false_rather_than_missing(firepanda: ModuleType) -> None:
    """The core answers null here and pandas answers false, and pandas wins at this layer."""
    import pandas as pd

    made = firepanda.Series([1.0, None, 3.0]).isin([1.0])
    assert made.tolist() == pd.Series([1.0, None, 3.0]).isin([1.0]).tolist()
    assert made.isna().tolist() == [False, False, False]


@needs_pandas
def test_a_nan_and_a_missing_row_are_both_found_by_a_nan(firepanda: ModuleType) -> None:
    """A nan is not equal to itself, so the kernel misses it and this puts it back.

    A float column can hold a real nan as well as a row with nothing in it, and pandas finds both
    with the same nan in the set. The kernel finds neither, the first because the comparison is
    false and the second because the comparison is unknown, so both are taken from the same place
    here and `notna` is the one question that is false in both of them.
    """
    import pandas as pd

    values = [NAN, 1.0, None]
    assert firepanda.Series(values).isin([NAN]).tolist() == pd.Series(values).isin([NAN]).tolist()


@needs_pandas
@pytest.mark.parametrize(
    ("wanted", "answer"),
    [
        (["a"], [True, False, False]),
        (["a", "b"], [True, True, False]),
        (["z"], [False, False, False]),
        ([], [False, False, False]),
        ([None], [False, False, True]),
        ([1], [False, False, False]),
    ],
)
def test_a_category_column_is_looked_up_through_its_codes(
    firepanda: ModuleType, wanted: list[Any], answer: list[bool]
) -> None:
    """A category stores positions, so the set becomes positions rather than becoming a category.

    A value that is not one of the categories has no position, which is why the third and the sixth
    rows cost nothing and raise nothing. pandas answers the same way and for the same reason.
    """
    import pandas as pd

    column = firepanda.Series(["a", "b", None]).astype("category")
    assert column.isin(wanted).tolist() == answer
    assert pd.Series(["a", "b", None], dtype="category").isin(wanted).tolist() == answer


@needs_pandas
def test_a_category_answers_plain_booleans(firepanda: ModuleType) -> None:
    """The answer is a mask and not a category of two, which is the obvious thing to get wrong."""
    made = firepanda.Series(["a", "b"]).astype("category").isin(["a"])
    assert str(made.dtype) == "bool"


@needs_pandas
@pytest.mark.parametrize(
    "wanted",
    [
        [2, 3],
        (2, 3),
        {2, 3},
        {2: "two", 3: "three"},
        range(2, 4),
        iter([2, 3]),
        {2: "two", 3: "three"}.keys(),
    ],
)
def test_anything_list_like_is_a_set(firepanda: ModuleType, wanted: Any) -> None:
    """A mapping is read as its keys and a generator is read once, both of which are pandas'."""
    assert firepanda.Series([1, 2, 3]).isin(wanted).tolist() == [False, True, True]


@needs_pandas
def test_a_column_is_a_set(firepanda: ModuleType) -> None:
    """The shape a caller reaches for most, which is one column looked up in another."""
    assert firepanda.Series([1, 2, 3]).isin(firepanda.Series([2, 3])).tolist() == [
        False,
        True,
        True,
    ]
    assert firepanda.Series([1, 2, 3]).isin(firepanda.Index([2, 3])).tolist() == [
        False,
        True,
        True,
    ]


@needs_pandas
@pytest.mark.parametrize("wanted", [1, 1.5, None, object()])
def test_a_scalar_is_refused_in_pandas_words(firepanda: ModuleType, wanted: Any) -> None:
    """A `TypeError` with pandas' own sentence in it, backticks and all."""
    import pandas as pd

    with pytest.raises(TypeError) as mine:
        firepanda.Series([1, 2, 3]).isin(wanted)
    with pytest.raises(TypeError) as theirs:
        pd.Series([1, 2, 3]).isin(wanted)
    assert str(mine.value) == str(theirs.value)


@needs_pandas
def test_a_string_is_a_scalar_and_not_a_set_of_letters(firepanda: ModuleType) -> None:
    """`isin("ab")` is a mistake everywhere in pandas and it is one here."""
    import pandas as pd

    with pytest.raises(TypeError) as mine:
        firepanda.Series(["a", "b"]).isin("ab")
    with pytest.raises(TypeError) as theirs:
        pd.Series(["a", "b"]).isin("ab")
    assert str(mine.value) == str(theirs.value)


@needs_pandas
def test_a_frame_asks_every_column_the_same_question(firepanda: ModuleType) -> None:
    """One set against a frame of two types, where each column finds its own half of it."""
    made = firepanda.DataFrame({"a": [1, 2], "b": ["x", "y"]}).isin([1, "y"])
    assert made["a"].tolist() == [True, False]
    assert made["b"].tolist() == [False, True]
    assert str(made["a"].dtype) == "bool"


@needs_pandas
def test_a_mapping_asks_each_column_its_own_question(firepanda: ModuleType) -> None:
    """A column the mapping does not name is false all the way down rather than being dropped."""
    made = firepanda.DataFrame({"a": [1, 2], "b": ["x", "y"]}).isin({"a": [2]})
    assert made.columns == ["a", "b"]
    assert made["a"].tolist() == [False, True]
    assert made["b"].tolist() == [False, False]


@needs_pandas
def test_a_frame_keeps_the_labels_it_was_asked_about(firepanda: ModuleType) -> None:
    """The answer is built through the constructor, so the labels have to be put back by hand."""
    made = firepanda.DataFrame({"a": [1, 2], "b": ["x", "y"]}).set_index("b").isin([2])
    assert made.index.tolist() == ["x", "y"]
    assert made.index.name == "b"
    assert made["a"].tolist() == [False, True]


@needs_pandas
def test_a_frame_refuses_a_scalar_in_pandas_words(firepanda: ModuleType) -> None:
    """A different sentence from the series one, quoted differently, and both are copied."""
    import pandas as pd

    with pytest.raises(TypeError) as mine:
        firepanda.DataFrame({"a": [1, 2]}).isin(1)
    with pytest.raises(TypeError) as theirs:
        pd.DataFrame({"a": [1, 2]}).isin(1)
    assert str(mine.value) == str(theirs.value)


@needs_pandas
@pytest.mark.parametrize("shape", ["series", "frame"])
def test_a_frame_refuses_the_two_arguments_that_are_not_sets(
    firepanda: ModuleType, shape: str
) -> None:
    """pandas answers these and it does not answer them as a membership test.

    `df.isin(other)` where the other is a frame or a series is an equality test lined up by label,
    which is `df == other` wearing this method's name. Reading it as a set would be a wrong answer
    that looks like a right one, so it is refused until the comparison is written.
    """
    frame = firepanda.DataFrame({"a": [1, 2]})
    other = firepanda.Series([1, 2]) if shape == "series" else firepanda.DataFrame({"a": [1, 2]})
    with pytest.raises(NotImplementedError, match="cell against cell"):
        frame.isin(other)


@needs_pandas
def test_an_index_still_answers_a_list(firepanda: ModuleType) -> None:
    """`Index.isin` was here first and is not changed by this, which is worth one line."""
    assert firepanda.Index([1, 2, 3]).isin([2]) == [False, True, False]
