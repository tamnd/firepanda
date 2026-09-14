"""The five `str` methods that are about case, checked against pandas.

Two of them write a row out in one case or the other and three of them ask a
question about the case a row is already in. They are one group because they all
rest on the same piece of data, which is the table that says what the other case
of a character is, and because the interesting rows are the same rows for all
five.

Case is not a byte and it is not even a character. `ß` raises to two letters, so
a row can come back longer than it went in, and a capital I with a dot over it
lowers to a letter and a separate dot, so a row can come back with more
characters in it than it started with. Both of those are in here, because an
implementation that walked the bytes or mapped one character to one character
would pass every ASCII test and get both of them wrong.

The last test in the file asserts three differences rather than working around
them. The standard library this is built on carries an older and smaller copy of
the Unicode case data than CPython does, and document 64 measures exactly how
much smaller. Those three are the differences a caller is most likely to meet,
they are written down here so that the day the data is replaced the test fails
and somebody has to come and read the document.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

ROWS = [
    "hello",
    "HÉLLO",
    "Mixed Case",
    "café",
    "straße",
    "İstanbul",
    "\u0131stanbul",
    "",
    "42",
    None,
]
"""Ten rows, each of them there to catch a different way of being wrong.

The second and the fourth are the rows where a byte walk would change the letter
and leave the accent, the fifth is the row that comes back longer than it went
in, and the sixth and the seventh are the Turkish pair, which is the one place
in a Latin alphabet where the two cases are not a pair at all. The empty string
and the row of digits are the two rows where a question about case has no cased
character to answer about, and the None is here because a case change on a
missing row is a missing row rather than an empty string.
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


@needs_pandas
def test_upper_matches_pandas_row_for_row(firepanda: ModuleType) -> None:
    """Every row, including the two that change length and the two Turkish ones."""
    assert like(made(firepanda).str.upper().tolist(), theirs().str.upper().tolist())


@needs_pandas
def test_lower_matches_pandas_row_for_row(firepanda: ModuleType) -> None:
    """The same ten rows the other way, which is the harder direction of the two."""
    assert like(made(firepanda).str.lower().tolist(), theirs().str.lower().tolist())


def test_a_row_can_come_back_longer_than_it_went_in(firepanda: ModuleType) -> None:
    """The row that proves this is not a character for a character rewrite."""
    assert made(firepanda, ["straße"]).str.upper().tolist() == ["STRASSE"]


def test_the_turkish_pair_is_not_a_pair(firepanda: ModuleType) -> None:
    """A capital I with a dot keeps its dot, and a small i without one stays without.

    Both are written out rather than compared, because they are the two answers
    that look like mistakes: the first is nine characters where the input was
    eight, and the second loses nothing at all despite the dotless letter having
    no capital of its own.
    """
    assert made(firepanda, ["İstanbul"]).str.lower().tolist() == ["i̇stanbul"]
    assert made(firepanda, ["\u0131stanbul"]).str.upper().tolist() == ["ISTANBUL"]


def test_changing_case_twice_does_not_come_back(firepanda: ModuleType) -> None:
    """Which is a fact about Unicode rather than about this library."""
    assert made(firepanda, ["straße"]).str.upper().str.lower().tolist() == ["strasse"]


def test_a_case_change_keeps_a_missing_row_missing(firepanda: ModuleType) -> None:
    """And does not turn it into the empty string, which is the tempting mistake."""
    assert made(firepanda, ["a", None]).str.upper().tolist() == ["A", None]
    assert made(firepanda, ["A", None]).str.lower().tolist() == ["a", None]


@needs_pandas
def test_the_three_questions_match_pandas_row_for_row(firepanda: ModuleType) -> None:
    """Held as object on the pandas side, which is where None stays None there too."""
    for name in ("isspace", "islower", "isupper"):
        mine = getattr(made(firepanda).str, name)().tolist()
        assert like(mine, getattr(theirs().str, name)().tolist()), name


def test_a_row_with_no_cased_character_is_neither_lower_nor_upper(
    firepanda: ModuleType,
) -> None:
    """Python's rule, and the one that makes the two questions not opposites."""
    column = made(firepanda, ["42", "", "abc", "ABC"])
    assert column.str.islower().tolist() == [False, False, True, False]
    assert column.str.isupper().tolist() == [False, False, False, True]


def test_whitespace_is_asked_of_every_character(firepanda: ModuleType) -> None:
    """Including the empty row, where there is no character to ask about."""
    column = made(firepanda, [" ", " \t\n", "a b", "", "ab"])
    assert column.str.isspace().tolist() == [True, True, False, False, False]


@needs_pandas
def test_a_question_about_a_missing_row_is_missing_here_and_false_there(
    firepanda: ModuleType,
) -> None:
    """The asserted difference, which is `engine/string-predicate-null` in the registry.

    pandas holding the column in its own string dtype has nowhere to put a
    missing answer, because the answer is a numpy array of bools, so a missing
    row comes back False and cannot be told from a row that was really not upper
    case. Held as object it comes back None, which is what this library answers
    whatever the column is made of.
    """
    import pandas as pd

    assert made(firepanda, ["A", None]).str.isupper().tolist() == [True, None]
    assert pd.Series(["A", None], dtype="str").str.isupper().tolist() == [True, False]
    assert pd.Series(["A", None], dtype="object").str.isupper().tolist() == [True, None]


def test_a_question_answers_a_column_of_bools(firepanda: ModuleType) -> None:
    """The shape of the answer, which is what picks the door these three come through."""
    assert str(made(firepanda, ["a"]).str.isupper().dtype) == "bool"
    assert str(made(firepanda, ["a"]).str.upper().dtype) == "string"


def test_the_accessor_refuses_a_column_that_is_not_text(firepanda: ModuleType) -> None:
    """At the accessor rather than at the method, which is where pandas refuses it."""
    with pytest.raises(AttributeError):
        firepanda.Series([1, 2, 3]).str.upper()


@needs_pandas
def test_three_measured_differences_in_the_case_data(firepanda: ModuleType) -> None:
    """The three gaps a caller is most likely to meet, written down on purpose.

    Document 64 measures the whole of it: about a hundred code points map
    differently and about seventeen hundred answer a question about case
    differently, out of the million or so there are. These three are the ones
    inside a script somebody is likely to be holding. A no break space is
    whitespace in Python and not here, the feminine ordinal is a lower case
    letter in Python and not here, and a Greek sigma at the end of a word lowers
    to its own final form in Python and to the ordinary one here.
    """
    assert made(firepanda, ["\u00a0"]).str.isspace().tolist() == [False]
    assert theirs(["\u00a0"]).str.isspace().tolist() == [True]
    assert made(firepanda, ["ª"]).str.islower().tolist() == [False]
    assert theirs(["ª"]).str.islower().tolist() == [True]
    assert made(firepanda, ["ΟΔΟΣ"]).str.lower().tolist() == ["οδοσ"]
    assert theirs(["ΟΔΟΣ"]).str.lower().tolist() == ["οδος"]
