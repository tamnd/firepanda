"""The five `str` questions about what a character is, checked against pandas.

`isalpha`, `isnumeric`, `isdigit`, `isdecimal` and `isalnum` are one group and a
simpler group than the case questions next door. Each of them is the same rule,
which is that the row has a character in it and every character it has is in the
class, so the loop stops at the first character that fails and an empty row is
False for all five. There is no flag to carry and nothing in a row can rule the
row out by being in a second class the way an upper case letter rules out
`islower`.

What makes them worth a file of their own is the classes rather than the rule.
pandas 3 holds text in Arrow and answers all five out of Arrow kernels, and
Arrow's idea of a digit is not Python's. Arrow calls anything written as a
single number sign a digit, so `½` and `¼` are digits to pandas and are not
digits to `str.isdigit`, and that is 877 code points of difference. Arrow also
knows about 8946 letters Python's copy of the Unicode data has not heard of.
Every one of those differences is a row a caller could hit, so this library
follows Arrow and the tests below sweep every code point in Unicode to say so.

The three number questions nest, and the two places they are wider than each
other are worth naming. Decimal is the characters that can be a place in a base
ten number, which is the ten ASCII digits and 760 others that work the same way.
Digit adds the 915 written as one sign rather than a place, the fractions and
the superscripts and the circled forms. Numeric adds the 239 that are a number
and a letter at once, which is the Roman numerals and the Runic counting marks.
No character is both alphabetic and numeric, so `isalnum` needs no class of its
own and is the other two together, and both of those facts are asserted against
Arrow by the generator before it writes a table.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

NAMES = ("isalpha", "isnumeric", "isdigit", "isdecimal", "isalnum")
"""The five, in the order the accessor gained them."""

ROWS = [
    "hello",
    "café",
    "42",
    "abc123",
    "a b",
    "1.5",
    "½",
    "²",
    "Ⅷ",
    "٤",
    "①",
    "三",
    "",
    None,
]
"""Fourteen rows, each of them there to catch a different way of being wrong.

The fourth is the only row that is alphanumeric and neither alphabetic nor
numeric. The seventh and the eighth are the two rows where Arrow and Python
disagree about what a digit is. The ninth is a number that is also a letter and
the twelfth is a letter that reads like a number and is not one. The tenth is a
decimal digit that is not an ASCII one and the eleventh is a digit that is not a
decimal. The empty string is the row all five say no to and the None is the row
they all keep missing.
"""

EVERY_CODE_POINT = [chr(cp) for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]
"""Unicode, one code point to a row, minus the surrogates pandas cannot hold."""


def made(firepanda: ModuleType, values: list[Any] = ROWS) -> Any:
    """A firepanda text column."""
    return firepanda.Series(values)


def theirs(values: list[Any] = ROWS) -> Any:
    """The same column in pandas, held the way pandas holds it by default."""
    import pandas as pd

    return pd.Series(values, dtype="str")


@needs_pandas
def test_the_five_match_pandas_row_for_row(firepanda: ModuleType) -> None:
    """On every row of the fourteen except the missing one, which has its own test."""
    for name in NAMES:
        mine = getattr(made(firepanda).str, name)().tolist()[:-1]
        assert mine == getattr(theirs().str, name)().tolist()[:-1], name


@needs_pandas
def test_the_five_agree_on_every_code_point_there_is(firepanda: ModuleType) -> None:
    """The whole of Unicode through both sides, five times, with nothing left over.

    A single character is the part of the rule that is about the character, and
    since a million rows through either library is a fraction of a second there
    is no reason to assert a sample of something that can be asserted whole. The
    test after this one is the other half, which is the folding.
    """
    mine, them = made(firepanda, EVERY_CODE_POINT), theirs(EVERY_CODE_POINT)
    for name in NAMES:
        assert getattr(mine.str, name)().tolist() == getattr(them.str, name)().tolist(), name


@needs_pandas
def test_the_five_agree_on_words_built_out_of_every_class(firepanda: ModuleType) -> None:
    """Which is the half a sweep of single characters cannot reach.

    The alphabet is chosen rather than random: a letter, a letter that is not
    ASCII, a decimal digit, a decimal digit that is not ASCII, a digit that is
    not decimal, a number that is also a letter, a space and a stop. Every
    arrangement of three of them is 512 rows, and between them they cover a row
    that starts in one class and ends in another, which is the only thing the
    folding can get wrong.
    """
    from itertools import product

    alphabet = ["a", "é", "4", "٤", "½", "Ⅷ", " ", "."]
    rows = ["".join(word) for word in product(alphabet, repeat=3)]
    mine, them = made(firepanda, rows), theirs(rows)
    for name in NAMES:
        assert getattr(mine.str, name)().tolist() == getattr(them.str, name)().tolist(), name


@needs_pandas
def test_a_half_sign_is_a_digit_to_pandas_and_is_not_one_to_python(
    firepanda: ModuleType,
) -> None:
    """The most surprising answer of the five, and it is pandas' rather than ours.

    pandas 3 answers `isdigit` out of Arrow, Arrow calls anything written as one
    number sign a digit, and Python does not. Both sides are written out here so
    that the choice this library made is visible rather than implied, and so
    that the day pandas changes its mind this test says which way.
    """
    rows = ["½", "¼", "²"]
    assert [row.isdigit() for row in rows] == [False, False, True]
    assert theirs(rows).str.isdigit().tolist() == [True, True, True]
    assert made(firepanda, rows).str.isdigit().tolist() == [True, True, True]


def test_the_three_number_questions_narrow_in_that_order(firepanda: ModuleType) -> None:
    """An ASCII four is all three, a half sign is two of them, a Roman numeral is one."""
    column = made(firepanda, ["4", "½", "Ⅷ"])
    assert column.str.isnumeric().tolist() == [True, True, True]
    assert column.str.isdigit().tolist() == [True, True, False]
    assert column.str.isdecimal().tolist() == [True, False, False]


def test_a_decimal_digit_need_not_be_an_ascii_one(firepanda: ModuleType) -> None:
    """Arabic Indic four and Extended Arabic Indic five are places in a base ten number."""
    # The second is written as a number because it is a five that reads as an o,
    # and a test whose reader cannot tell which character it is about is not one.
    rows = ["٤", "\u06f5", "42"]
    assert made(firepanda, rows).str.isdecimal().tolist() == [True, True, True]


def test_a_letter_is_not_a_number_and_a_number_is_not_a_letter(
    firepanda: ModuleType,
) -> None:
    """A Roman numeral looks like both and Arrow calls it one, and the CJK three the other.

    The two classes have nothing in common, which the generator asserts over
    every code point before it writes, and which is the reason there is no
    alphanumeric class anywhere in this library.
    """
    # The third is written as a number for the reason the test above gives: it
    # is the ideographic zero and it reads as a capital O in a source file.
    column = made(firepanda, ["Ⅷ", "三", "\u3007"])
    assert column.str.isalpha().tolist() == [False, True, False]
    assert column.str.isnumeric().tolist() == [True, False, True]


def test_the_alphanumeric_question_is_the_other_two_together(firepanda: ModuleType) -> None:
    """Including the row that is alphanumeric and is neither of the two on its own."""
    column = made(firepanda, ["abc", "123", "abc123", "a b", "1.5"])
    assert column.str.isalnum().tolist() == [True, True, True, False, False]
    assert column.str.isalpha().tolist() == [True, False, False, False, False]
    assert column.str.isnumeric().tolist() == [False, True, False, False, False]


def test_an_empty_row_answers_no_to_all_five(firepanda: ModuleType) -> None:
    """The rule needs a character and an empty row has none, so there is nothing to be."""
    column = made(firepanda, [""])
    for name in NAMES:
        assert getattr(column.str, name)().tolist() == [False], name


def test_all_five_keep_a_missing_row_missing(firepanda: ModuleType) -> None:
    """Which is the difference from pandas that `engine/string-predicate-null` names.

    pandas holding the column the way it holds it by default answers a numpy
    array of bools and has nowhere to put a missing answer, so a missing row
    comes back False and cannot be told from a row that was really not a letter.
    This library answers None, the same way it does for the case questions.
    """
    column = made(firepanda, ["abc", None])
    for name in NAMES:
        assert getattr(column.str, name)().tolist()[1] is None, name
    assert column.str.isalpha().tolist() == [True, None]
    assert column.str.isdigit().tolist() == [False, None]
