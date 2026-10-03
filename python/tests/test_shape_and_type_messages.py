"""Arguments of the wrong shape or type, refused or read with pandas' words.

Each case is one pandas answers with a sentence of its own, and the sentence is
the one asserted, since a caller who catches the error by its words should not
have to know which library raised it.
"""

from __future__ import annotations

from types import ModuleType

import pytest


@pytest.mark.parametrize(
    ("value", "words"),
    [
        ([1], r'^"value" parameter must be a scalar or dict, but you passed a "list"$'),
        ((1,), r'^"value" parameter must be a scalar or dict, but you passed a "tuple"$'),
        ({1}, r'^"value" parameter must be a scalar, dict or Series, but you passed a "set"$'),
        (range(1), r'^"value" parameter must be a scalar, dict or Series, but you passed a "ra'),
    ],
)
def test_a_column_fill_value_of_several_values(
    firepanda: ModuleType, value: object, words: str
) -> None:
    """A run of values that is not a dict or a column says which shape it was."""
    column = firepanda.Series([1.0, None])
    with pytest.raises(TypeError, match=words):
        column.fillna(value)


def test_a_frame_fill_value_of_several_values(firepanda: ModuleType) -> None:
    """A list gets the column's words and a set gets the frame's own."""
    frame = firepanda.DataFrame({"a": [1.0, None]})
    with pytest.raises(TypeError, match=r'but you passed a "list"$'):
        frame.fillna([1])
    with pytest.raises(ValueError, match=r"^invalid fill value with a <class 'set'>$"):
        frame.fillna({1})


def test_an_index_fill_value_of_several_values(firepanda: ModuleType) -> None:
    """An index takes one value only."""
    with pytest.raises(TypeError, match=r"^'value' must be a scalar, passed: list$"):
        firepanda.Index([1.0, None]).fillna([1])


def test_the_fill_shapes_that_name_places_still_fill(firepanda: ModuleType) -> None:
    """A scalar, a dict and a column are read as before."""
    column = firepanda.Series([1.0, None])
    assert column.fillna(5.0).tolist() == [1.0, 5.0]
    assert column.fillna({1: 7.0}).tolist() == [1.0, 7.0]
    assert column.fillna(firepanda.Series([0.0, 8.0])).tolist() == [1.0, 8.0]


@pytest.mark.parametrize("method", ["std", "var", "sem"])
def test_a_ddof_that_is_not_a_number(firepanda: ModuleType, method: str) -> None:
    """The words `float` uses, as a ValueError rather than the core's RuntimeError."""
    column = firepanda.Series([1.0, 2.0])
    with pytest.raises(ValueError, match=r"^could not convert string to float: 'a'$"):
        getattr(column, method)(ddof="a")
    frame = firepanda.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match=r"^could not convert string to float: 'a'$"):
        getattr(frame, method)(ddof="a")


def test_argmax_past_the_one_axis(firepanda: ModuleType) -> None:
    """`argmax` is numpy's word and gets numpy's sentence, where `idxmax` keeps pandas'."""
    column = firepanda.Series([1.0, 2.0])
    with pytest.raises(ValueError, match=r"^`axis` must be fewer than the number of dimensions"):
        column.argmax(axis=1)
    with pytest.raises(ValueError, match=r"^`axis` must be fewer than the number of dimensions"):
        column.argmin(axis=1)
    with pytest.raises(ValueError, match=r"^No axis named 1 for object type Series$"):
        column.idxmax(axis=1)


def test_ranking_text_is_refused(firepanda: ModuleType) -> None:
    """The frame names the column, and the column names the method and the type."""
    frame = firepanda.DataFrame({"c": ["a", "b"], "d": [1, 2]})
    with pytest.raises(
        TypeError, match=r"^Column 'c' has dtype str, cannot use method 'nlargest' with this dtype$"
    ):
        frame.nlargest(1, "c")
    with pytest.raises(TypeError, match=r"^Column 'c' has dtype str, cannot use method 'nsmall"):
        frame.nsmallest(1, ["d", "c"])
    with pytest.raises(TypeError, match=r"^Cannot use method 'nlargest' with dtype str$"):
        frame["c"].nlargest(1)


def test_ranking_a_category_is_refused(firepanda: ModuleType) -> None:
    """A category is ordered by its categories, which pandas does not rank by here."""
    column = firepanda.Series(["a", "b"]).astype("category")
    with pytest.raises(TypeError, match=r"^Cannot use method 'nsmallest' with dtype category$"):
        column.nsmallest(1)


def test_ranking_flags(firepanda: ModuleType) -> None:
    """A flag column ranks True above False, as pandas ranks it."""
    frame = firepanda.DataFrame({"c": [False, True, False], "d": [1, 2, 3]})
    assert frame.nlargest(1, "c")["d"].tolist() == [2]
    assert frame.nsmallest(2, "c")["d"].tolist() == [1, 3]
    assert frame["c"].nsmallest(1).index.tolist() == [0]


def test_ranking_nullable_whole_numbers(firepanda: ModuleType) -> None:
    """A gap ranks last both ways, and the values keep their type."""
    column = firepanda.Series([3, None, 5, 1], dtype="Int64")
    largest = column.nlargest(2)
    assert largest.index.tolist() == [2, 0]
    assert str(largest.dtype) == "Int64"
    assert column.nsmallest(4).index.tolist() == [3, 0, 2, 1]
    assert column.nlargest(2, keep="all").tolist() == [5, 3]


def test_a_group_quantile_out_of_range_is_printed_as_a_float(firepanda: ModuleType) -> None:
    """pandas prints the bad `q` it was handed after making it a float."""
    frame = firepanda.DataFrame({"k": [1, 1], "v": [1.0, 2.0]})
    with pytest.raises(ValueError, match=r"^Each 'q' must be between 0 and 1. Got '2.0' instead$"):
        frame.groupby("k").quantile(2)
    with pytest.raises(ValueError, match=r"Got '3.0' instead$"):
        frame.groupby("k")["v"].quantile([0.5, 3])


def test_the_validate_words_are_listed_short_ones_first(firepanda: ModuleType) -> None:
    """The four short spellings come before the four long ones, as pandas lists them."""
    frame = firepanda.DataFrame({"k": [1]})
    with pytest.raises(ValueError) as caught:
        frame.merge(frame, on="k", validate="x")
    listed = str(caught.value).split("\n")[1:]
    assert listed == [
        '- "1:1"',
        '- "1:m"',
        '- "m:1"',
        '- "m:m"',
        '- "one_to_one"',
        '- "one_to_many"',
        '- "many_to_one"',
        '- "many_to_many"',
    ]


def test_whole_numbers_floor_divided_by_zero(firepanda: ModuleType) -> None:
    """Infinity with the dividend's sign, NaN for nought over nought, and floats."""
    column = firepanda.Series([-2, 0, 3])
    answer = column // 0
    assert str(answer.dtype) == "float64"
    values = answer.tolist()
    assert values[0] == float("-inf")
    assert values[1] != values[1]
    assert values[2] == float("inf")
    assert str((column // 2).dtype) == "int64"


def test_whole_numbers_modulo_zero(firepanda: ModuleType) -> None:
    """NaN wherever the divisor is zero, and the rest as floats."""
    answer = firepanda.Series([4, 5, 6]) % firepanda.Series([0, 2, 4])
    values = answer.tolist()
    assert str(answer.dtype) == "float64"
    assert values[0] != values[0]
    assert values[1:] == [1.0, 2.0]


def test_a_zero_answer_is_never_negative(firepanda: ModuleType) -> None:
    """pandas divides as whole numbers first, so `0 // -2` is a positive zero."""
    answer = 0 // firepanda.Series([-2, 0])
    assert str(answer.iloc[0]) == "0.0"
    remainder = firepanda.Series([4, 1]) % firepanda.Series([-2, 0])
    assert str(remainder.iloc[0]) == "0.0"


def test_a_frame_divided_by_a_frame_with_a_zero(firepanda: ModuleType) -> None:
    """The whole number columns widen together, as pandas' one block of them does."""
    left = firepanda.DataFrame({"a": [4, 6], "b": [1, 0]})
    right = firepanda.DataFrame({"a": [2, 3], "b": [0, 0]})
    answer = left // right
    assert [str(kind) for kind in answer.dtypes.tolist()] == ["float64", "float64"]
    assert answer["a"].tolist() == [2.0, 2.0]
    assert answer["b"].iloc[0] == float("inf")
    assert str((left // 2).dtypes.tolist()[0]) == "int64"


def test_nullable_whole_numbers_divided_by_zero_stay_nullable(firepanda: ModuleType) -> None:
    """A nullable type has a gap of its own, which is what pandas answers."""
    answer = firepanda.Series([1, 2], dtype="Int64") // 0
    assert str(answer.dtype) == "Int64"
