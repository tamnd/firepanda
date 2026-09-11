"""The nine `str` methods that only ever touch the two ends of a row.

Trimming and padding are one group because nothing in either of them reads the
middle of a row, and they are tested together for the same reason. Every answer
is measured against a running pandas rather than against a written down
constant, which is the rule `test_astype.py` argues for and which matters more
here than usual: three of the rules being copied are undocumented, and a
constant written out by hand would be a copy of what the author believed rather
than of what pandas does.

The three are worth naming. A strip set is a set of characters and not a prefix.
A width is a count of characters, so a row of accented letters pads to the width
that was asked for and not to a byte count. And an odd amount of padding on both
sides goes to whichever side CPython's `unicode_center` puts it on, which is
written down in C and nowhere else.

The arguments that pandas refuses are checked too, because a caller who passes a
width as a string should read the message pandas would have given rather than a
message about the column.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

ROWS = ["  hi  ", "abXba", "-5", "café", "", "\tab\n", None, "日本語", "  spaced  "]
"""Nine rows, each of them there to catch a different way of being wrong.

The first and the sixth carry whitespace that is not a space, the second is the
row where a strip set has to be read as a set, the third is the one where a sign
has to stay in front of the zeros, and the fourth and the eighth are the rows
where a byte count and a character count disagree. The empty string is the row
where padding has to do all the work and stripping none of it, and the None is
there because a string method on a missing row is a missing row.
"""

UNICODE = ["café", "İstanbul", "ß", "ﬁance", "你好", " x ", "\U0001d7d9\U0001d7da", ""]
"""Rows where a character is not a byte, and sometimes not even close.

The seventh row is two mathematical double struck digits, which is two
characters in eight bytes and is the widest the two counts get apart in anything
this library is likely to meet. It is written as escapes rather than as itself
because a digit that is not a digit is the kind of thing a linter is right to
complain about in source, however much it belongs in the data.
"""


def made(firepanda: ModuleType, values: list[Any] = ROWS) -> Any:
    """A firepanda text column."""
    return firepanda.Series(values)


def theirs(values: list[Any] = ROWS) -> Any:
    """The same column in pandas, held as object so that None stays None."""
    import pandas as pd

    return pd.Series(values, dtype="object")


def like(mine: list[Any], them: list[Any]) -> bool:
    """Compares two columns of values, reading a NaN as a None.

    The library wide rule is that a missing row reads back as None and pandas
    reads it back as NaN, so the two lists are the same answer written two ways
    and have to be compared through this rather than through `==`.
    """
    if len(mine) != len(them):
        return False
    for one, two in zip(mine, them, strict=True):
        if two is None or (isinstance(two, float) and two != two):
            if one is not None:
                return False
            continue
        if one != two:
            return False
    return True


@needs_pandas
@pytest.mark.parametrize("name", ["strip", "lstrip", "rstrip"])
def test_stripping_nothing_in_particular_removes_whitespace(
    firepanda: ModuleType, name: str
) -> None:
    """Python's idea of whitespace and not ASCII's, which the tab and the newline check."""
    assert like(
        getattr(made(firepanda).str, name)().tolist(),
        getattr(theirs().str, name)().tolist(),
    )


@needs_pandas
@pytest.mark.parametrize("name", ["strip", "lstrip", "rstrip"])
def test_a_strip_set_is_a_set_and_not_a_prefix(firepanda: ModuleType, name: str) -> None:
    """`abXba` stripped of `ab` is `X`, and every reading that gives `Xba` is wrong."""
    assert like(
        getattr(made(firepanda).str, name)("ab").tolist(),
        getattr(theirs().str, name)("ab").tolist(),
    )
    assert made(firepanda).str.strip("ab").tolist()[1] == "X"


@needs_pandas
def test_an_empty_strip_set_is_not_the_same_as_no_strip_set(firepanda: ModuleType) -> None:
    """Two different requests, and pandas keeps them apart because Python does.

    `strip()` removes whitespace and `strip("")` removes nothing at all, so the
    absence cannot be filled in with a default set on the way across.
    """
    assert like(made(firepanda).str.strip("").tolist(), theirs().str.strip("").tolist())
    assert made(firepanda).str.strip("").tolist()[0] == "  hi  "


@needs_pandas
def test_stripping_counts_characters_and_not_bytes(firepanda: ModuleType) -> None:
    """A strip set of one accented letter takes off letters and not lead bytes."""
    rows = ["ééxéé", "ééxab", None]
    assert like(
        made(firepanda, rows).str.strip("é").tolist(),
        theirs(rows).str.strip("é").tolist(),
    )
    assert made(firepanda, rows).str.strip("é").tolist()[0] == "x"


@needs_pandas
@pytest.mark.parametrize("side", ["left", "right", "both"])
def test_padding_fills_the_side_it_was_told_to(firepanda: ModuleType, side: str) -> None:
    """Three sides, and the default is the left one, which reads backwards and is pandas."""
    assert like(
        made(firepanda).str.pad(12, side=side, fillchar=".").tolist(),
        theirs().str.pad(12, side=side, fillchar=".").tolist(),
    )


@needs_pandas
def test_padding_counts_characters_and_not_bytes(firepanda: ModuleType) -> None:
    """Every row comes out twelve characters wide, whatever it weighs in bytes."""
    assert like(
        made(firepanda, UNICODE).str.center(12).tolist(),
        theirs(UNICODE).str.center(12).tolist(),
    )
    for value in made(firepanda, UNICODE).str.center(12).tolist():
        assert len(value) == 12


@needs_pandas
def test_a_row_that_is_already_wide_enough_is_handed_back(firepanda: ModuleType) -> None:
    """Python pads and never truncates, which is not what a fixed width field would do."""
    rows = ["abcdef", "abcdefghij"]
    assert like(made(firepanda, rows).str.center(6).tolist(), theirs(rows).str.center(6).tolist())
    assert made(firepanda, rows).str.center(6).tolist()[1] == "abcdefghij"


@needs_pandas
def test_the_odd_character_of_a_centred_row_goes_where_cpython_puts_it(
    firepanda: ModuleType,
) -> None:
    """The one rule here that is written down in C and nowhere else.

    `"a".center(4, ".")` is `".a.."` and `"ab".center(5, ".")` is `"..ab."`, and
    the two disagree about which side gets the spare character. Anything that
    guessed at splitting the gap in half gets one of the two wrong.
    """
    rows = ["a", "ab", "abc"]
    for width in (3, 4, 5, 6, 7):
        assert like(
            made(firepanda, rows).str.center(width, ".").tolist(),
            theirs(rows).str.center(width, ".").tolist(),
        )
    assert made(firepanda, rows).str.center(4, ".").tolist()[0] == ".a.."
    assert made(firepanda, rows).str.center(5, ".").tolist()[1] == "..ab."


@needs_pandas
@pytest.mark.parametrize("name", ["ljust", "rjust"])
def test_justifying_is_padding_with_the_side_already_chosen(
    firepanda: ModuleType, name: str
) -> None:
    """And the sides read the way a typesetter would rather than the way `pad` does."""
    assert like(
        getattr(made(firepanda).str, name)(10, ".").tolist(),
        getattr(theirs().str, name)(10, ".").tolist(),
    )


@needs_pandas
def test_zero_filling_puts_the_zeros_after_a_sign(firepanda: ModuleType) -> None:
    """`-5` filled to six is `-00005`, which is a rule about numbers and not about text."""
    rows = ["-5", "+5", "5", "-", "", None, "café"]
    assert like(made(firepanda, rows).str.zfill(6).tolist(), theirs(rows).str.zfill(6).tolist())
    assert made(firepanda, rows).str.zfill(6).tolist()[0] == "-00005"


@needs_pandas
@pytest.mark.parametrize("times", [0, 1, 3])
def test_repeating_writes_a_row_out_end_to_end(firepanda: ModuleType, times: int) -> None:
    """Zero copies is the empty string, which is what multiplying a string by zero does."""
    assert like(
        made(firepanda).str.repeat(times).tolist(),
        theirs().str.repeat(times).tolist(),
    )


@needs_pandas
def test_a_width_that_is_not_a_whole_number_is_refused(firepanda: ModuleType) -> None:
    """With pandas' message, because the problem is the argument and not the column."""
    with pytest.raises(TypeError, match="width must be of integer type"):
        made(firepanda).str.zfill("6")
    with pytest.raises(TypeError, match="width must be of integer type"):
        made(firepanda).str.pad(6.5)
    with pytest.raises(TypeError, match="width must be of integer type"):
        made(firepanda).str.center(True)


@needs_pandas
def test_a_fill_that_is_not_one_character_is_refused(firepanda: ModuleType) -> None:
    """Two checks and two messages, both of them copied from pandas."""
    with pytest.raises(TypeError, match="not int"):
        made(firepanda).str.pad(6, fillchar=1)
    with pytest.raises(TypeError, match="not str"):
        made(firepanda).str.pad(6, fillchar="..")
    with pytest.raises(TypeError, match="not str"):
        made(firepanda).str.center(6, "")


@needs_pandas
def test_a_side_that_is_not_one_of_three_words_is_refused(firepanda: ModuleType) -> None:
    """A `ValueError` here where the fill checks are `TypeError`, which is pandas' split."""
    with pytest.raises(ValueError, match="Invalid side"):
        made(firepanda).str.pad(6, side="middle")


@needs_pandas
def test_one_repeat_count_per_row_says_that_it_is_not_written_yet(
    firepanda: ModuleType,
) -> None:
    """A second method wearing the same name, and it answers a column per row.

    Refusing it is better than repeating by whichever count happens to be first,
    and `NotImplementedError` says which of the two it is rather than reading as
    a bad argument.
    """
    with pytest.raises(NotImplementedError, match="count per row"):
        made(firepanda).str.repeat([1, 2, 3, 1, 1, 1, 1, 1, 1])
