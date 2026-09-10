"""The twelve `str` methods where a position is a character, checked against pandas.

The accessor has fifty seven names and these are the twelve whose only idea is
that `s.str[1]` means the second character and not the second byte. Everything
else in the library counts bytes and is right to, so the tests that matter here
are the ones on rows where the two counts disagree: `héllo` is five characters
in six bytes and `日本語です` is five characters in fifteen.

Every answer is measured against a running pandas rather than against a written
down constant, for the reason `test_astype.py` gives. Two of them are measured
against Python's own `str` instead, and both times it is because pandas answers
through numpy and loses the shape of the question: `str.len` on a column with a
missing row comes back as float64 there, and `str.find` the same, so the numbers
would have to be compared as floats to be compared at all. Python's `str` is
what pandas is copying in those two cases anyway.

One difference is asserted rather than worked around. A missing row reads back
as None here and as NaN in pandas, which is the library wide rule argued in
`firepanda/py/values.mojo`, so the comparisons go through `like` rather than
through `==`.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

ROWS = ["héllo", "日本語です", "", "abcdef", None, "zebra"]
"""Six rows chosen so that a byte answer and a character answer differ.

The first row is five characters in six bytes and the second is five characters
in fifteen, so any kernel that counted bytes would be caught by both. The empty
string is here because it is the row where `get` has to answer nothing and where
a slice has to answer something. The None is here because a string method on a
missing row is a missing row and never an empty string, which is the mistake
that looks most reasonable while being written.
"""


def made(firepanda: ModuleType, values: list[Any] = ROWS) -> Any:
    """A firepanda text column."""
    return firepanda.Series(values)


def theirs(values: list[Any] = ROWS) -> Any:
    """The same column in pandas, held as object so that None stays None."""
    import pandas as pd

    return pd.Series(values, dtype="object")


def like(mine: list[Any], them: list[Any]) -> bool:
    """Compares two columns of values, reading a NaN as a None."""
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


def by_hand(rows: list[Any], call: Any) -> list[Any]:
    """Runs a Python `str` method over the rows, leaving the missing ones missing."""
    return [None if row is None else call(row) for row in rows]


@needs_pandas
def test_length_counts_characters_and_not_bytes(firepanda: ModuleType) -> None:
    """The whole reason the kernel exists, on the two rows that can tell."""
    assert made(firepanda).str.len().tolist() == [5, 5, 0, 6, None, 5]
    assert like(made(firepanda).str.len().tolist(), by_hand(ROWS, len))


@needs_pandas
def test_length_is_an_integer_column_where_pandas_widens_to_float(
    firepanda: ModuleType,
) -> None:
    """The asserted difference, and the one place in this file it is visible in a dtype.

    pandas has nowhere to put a missing count in an int64 numpy array, so a
    column with one missing row comes back float64 and every count in it is a
    float. Arrow has a validity bitmap, so the count stays a count.
    """
    assert str(made(firepanda).str.len().dtype) == "int64"
    assert str(theirs().str.len().dtype) == "float64"


@needs_pandas
def test_a_slice_takes_characters_under_pythons_own_rules(firepanda: ModuleType) -> None:
    """Three slices that pandas and Python agree on, including a negative start."""
    for start, stop, step in ((1, 4, None), (None, None, 2), (-3, None, None)):
        mine = made(firepanda).str.slice(start, stop, step)
        assert like(mine.tolist(), theirs().str.slice(start, stop, step).tolist())


@needs_pandas
def test_a_negative_step_reverses_every_row(firepanda: ModuleType) -> None:
    """The slice that is easiest to get backwards, since both bounds are absent.

    An absent bound means the far end, and which end that is depends on the sign
    of the step. Getting it wrong gives an empty string on every row rather than
    an error, which is why this is its own test.
    """
    mine = made(firepanda).str.slice(None, None, -1)
    assert like(mine.tolist(), theirs().str.slice(None, None, -1).tolist())
    assert mine.tolist()[0] == "olléh"


@needs_pandas
def test_a_step_of_zero_is_refused(firepanda: ModuleType) -> None:
    """pandas raises here as well, out of Python's own slice machinery."""
    with pytest.raises(ValueError):
        made(firepanda).str.slice(None, None, 0)


@needs_pandas
def test_getting_one_character_answers_nothing_past_the_end(
    firepanda: ModuleType,
) -> None:
    """The row that separates `get` from a slice of one character.

    `"".str[1]` would be the empty string as a slice and is a missing value as a
    get, and pandas is the one being copied here rather than Python, which would
    raise.
    """
    mine = made(firepanda).str.get(1)
    assert like(mine.tolist(), theirs().str.get(1).tolist())
    assert mine.tolist()[0] == "é"
    assert mine.tolist()[2] is None


@needs_pandas
def test_replacing_a_slice_puts_the_new_text_where_the_characters_were(
    firepanda: ModuleType,
) -> None:
    """Two bounds and a replacement, on rows where the bounds land mid character."""
    mine = made(firepanda).str.slice_replace(1, 3, "XX")
    assert like(mine.tolist(), theirs().str.slice_replace(1, 3, "XX").tolist())
    assert mine.tolist()[0] == "hXXlo"
    assert mine.tolist()[1] == "日XXです"


@needs_pandas
def test_find_answers_a_character_position_or_minus_one(firepanda: ModuleType) -> None:
    """Measured against Python, since pandas widens this one to float as well."""
    mine = made(firepanda).str.find("e")
    assert like(mine.tolist(), by_hand(ROWS, lambda row: row.find("e")))


@needs_pandas
def test_rfind_answers_the_last_one(firepanda: ModuleType) -> None:
    """A row with two matches, so that the two directions cannot agree by accident."""
    rows = ["banana", "b", None]
    mine = made(firepanda, rows)
    assert like(mine.str.find("a").tolist(), by_hand(rows, lambda row: row.find("a")))
    assert like(mine.str.rfind("a").tolist(), by_hand(rows, lambda row: row.rfind("a")))
    assert mine.str.find("a").tolist()[0] == 1
    assert mine.str.rfind("a").tolist()[0] == 5


@needs_pandas
def test_find_takes_a_range_and_the_position_is_still_from_the_start(
    firepanda: ModuleType,
) -> None:
    """The bounds narrow the search and do not move where counting begins."""
    rows = ["banana", "aaaa", None]
    mine = made(firepanda, rows)
    assert like(
        mine.str.find("a", 2, 5).tolist(),
        by_hand(rows, lambda row: row.find("a", 2, 5)),
    )
    assert mine.str.find("a", 2, 5).tolist()[0] == 3


@needs_pandas
def test_index_is_find_that_raises_where_find_answers_minus_one(
    firepanda: ModuleType,
) -> None:
    """One missing row is enough to throw away every position that was found."""
    assert made(firepanda, ["ab", "ba"]).str.index("a").tolist() == [0, 1]
    with pytest.raises(ValueError):
        made(firepanda).str.index("e")
    with pytest.raises(ValueError):
        made(firepanda).str.rindex("e")


@needs_pandas
def test_a_missing_row_does_not_make_index_raise(firepanda: ModuleType) -> None:
    """pandas skips the missing rows here rather than counting them as absences."""
    assert like(
        made(firepanda, ["ab", None]).str.index("a").tolist(),
        theirs(["ab", None]).str.index("a").tolist(),
    )


@needs_pandas
def test_startswith_and_endswith_answer_a_mask_with_a_hole_in_it(
    firepanda: ModuleType,
) -> None:
    """The predicates, where a missing row is missing and not false."""
    assert like(
        made(firepanda).str.startswith("a").tolist(),
        theirs().str.startswith("a").tolist(),
    )
    assert like(made(firepanda).str.endswith("a").tolist(), theirs().str.endswith("a").tolist())
    assert made(firepanda).str.startswith("a").tolist()[4] is None


@needs_pandas
def test_a_tuple_asks_several_questions_at_once(firepanda: ModuleType) -> None:
    """Python's own signature for these two, and the empty tuple that comes with it."""
    assert like(
        made(firepanda).str.startswith(("a", "z")).tolist(),
        theirs().str.startswith(("a", "z")).tolist(),
    )
    assert like(
        made(firepanda).str.endswith(("a", "o")).tolist(),
        theirs().str.endswith(("a", "o")).tolist(),
    )


@needs_pandas
def test_na_fills_the_hole_in_the_mask(firepanda: ModuleType) -> None:
    """The one argument these two take beyond the pattern."""
    assert like(
        made(firepanda).str.startswith("a", False).tolist(),
        theirs().str.startswith("a", na=False).tolist(),
    )
    assert made(firepanda).str.startswith("a", True).tolist()[4] is True


@needs_pandas
def test_a_prefix_comes_off_once_and_only_if_it_is_there(
    firepanda: ModuleType,
) -> None:
    """Removing, not stripping, which is the difference this pair exists to make."""
    rows = ["aab", "b", "", None]
    assert like(
        made(firepanda, rows).str.removeprefix("a").tolist(),
        theirs(rows).str.removeprefix("a").tolist(),
    )
    assert made(firepanda, rows).str.removeprefix("a").tolist()[0] == "ab"


@needs_pandas
def test_a_suffix_comes_off_the_same_way(firepanda: ModuleType) -> None:
    """The mirror, on rows where the suffix is a multi byte character."""
    rows = ["日本語", "語", "", None]
    assert like(
        made(firepanda, rows).str.removesuffix("語").tolist(),
        theirs(rows).str.removesuffix("語").tolist(),
    )
    assert made(firepanda, rows).str.removesuffix("語").tolist()[0] == "日本"


@needs_pandas
def test_the_accessor_refuses_a_column_that_is_not_text(firepanda: ModuleType) -> None:
    """Checked when the accessor is built, which is where pandas checks it too.

    An `AttributeError` rather than a type error, so that code guarding with
    `hasattr(s, "str")` gets a False out of it instead of an exception.
    """
    with pytest.raises(AttributeError):
        firepanda.Series([1, 2, 3]).str.len()
    assert not hasattr(firepanda.Series([1, 2, 3]), "str")
    assert hasattr(made(firepanda), "str")
