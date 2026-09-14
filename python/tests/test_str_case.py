"""The five `str` methods that are about case, checked against pandas.

Two of them write a row out in one case or the other and three of them ask a
question about the case a row is already in. They are one group because they all
rest on the same piece of data, which is the table that says what the other case
of a character is, and because the interesting rows are the same rows for all
five.

Which table that is turns out to matter more than anything else here. pandas 3
holds a text column in Arrow and answers `upper` and `lower` out of an Arrow
kernel, which uses the simple case mappings, so a row is never longer coming out
than it was going in. The same column held as object goes through Python's own
string methods, which use the full mappings, and the two disagree: a sharp s
raises to one letter in the first and two in the second, and a Turkish capital I
keeps its dot in the second and loses it in the first. So these tests compare
against the default dtype, which is what a caller gets without asking, and the
test at the end of the file writes both pandas answers out side by side so that
the choice is visible rather than implied.

The last test asserts three differences rather than working around them. The
standard library underneath this carries an older copy of the Unicode data than
Arrow does, and document 64 measures exactly how much older. The three names
that ask a question still answer out of that copy, so they differ on characters
neither table here corrects.
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
and leave the accent, the fifth is the row where Python and Arrow disagree about
how many letters the answer has, and the sixth and the seventh are the Turkish
pair, which is the one place in a Latin alphabet where the two cases are not a
pair at all. The empty string and the row of digits are the two rows where a
question about case has no cased character to answer about, and the None is here
because a case change on a missing row is a missing row rather than an empty
string.
"""


def made(firepanda: ModuleType, values: list[Any] = ROWS) -> Any:
    """A firepanda text column."""
    return firepanda.Series(values)


def theirs(values: list[Any] = ROWS) -> Any:
    """The same column in pandas, held the way pandas holds it by default."""
    import pandas as pd

    return pd.Series(values, dtype="str")


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
    """Every row, including the sharp s and the two Turkish ones."""
    assert like(made(firepanda).str.upper().tolist(), theirs().str.upper().tolist())


@needs_pandas
def test_lower_matches_pandas_row_for_row(firepanda: ModuleType) -> None:
    """The same ten rows the other way, which is the harder direction of the two."""
    assert like(made(firepanda).str.lower().tolist(), theirs().str.lower().tolist())


@needs_pandas
def test_the_two_pandas_string_backends_do_not_agree_with_each_other(
    firepanda: ModuleType,
) -> None:
    """Which is why the tests above name a dtype, and this is the name they chose.

    Held as pandas holds it by default the answer comes out of Arrow, and held
    as object it comes out of Python. This library follows the first, because
    that is what a caller gets from `pd.Series([...])` without asking for
    anything, and because a case change that can make a row longer is a
    different kind of operation from one that cannot.
    """
    import pandas as pd

    rows = ["straße", "İstanbul"]
    assert pd.Series(rows, dtype="str").str.upper().tolist() == ["STRAẞE", "İSTANBUL"]
    assert pd.Series(rows, dtype="object").str.upper().tolist() == ["STRASSE", "İSTANBUL"]
    assert pd.Series(rows, dtype="str").str.lower().tolist() == ["straße", "istanbul"]
    assert pd.Series(rows, dtype="object").str.lower().tolist() == [
        "straße",
        "i̇stanbul",
    ]
    assert made(firepanda, rows).str.upper().tolist() == ["STRAẞE", "İSTANBUL"]
    assert made(firepanda, rows).str.lower().tolist() == ["straße", "istanbul"]


def test_a_row_keeps_its_length_through_a_case_change(firepanda: ModuleType) -> None:
    """The simple mappings are one character in and one character out, always."""
    for row in ("straße", "İstanbul", "ﬁance", "ŉ"):
        assert len(made(firepanda, [row]).str.upper().tolist()[0]) == len(row)
        assert len(made(firepanda, [row]).str.lower().tolist()[0]) == len(row)


def test_changing_case_twice_does_not_come_back(firepanda: ModuleType) -> None:
    """Which is a fact about Unicode rather than about this library."""
    assert made(firepanda, ["İstanbul"]).str.lower().str.upper().tolist() == ["ISTANBUL"]


def test_a_case_change_keeps_a_missing_row_missing(firepanda: ModuleType) -> None:
    """And does not turn it into the empty string, which is the tempting mistake."""
    assert made(firepanda, ["a", None]).str.upper().tolist() == ["A", None]
    assert made(firepanda, ["A", None]).str.lower().tolist() == ["a", None]


@needs_pandas
def test_the_three_questions_match_pandas_row_for_row(firepanda: ModuleType) -> None:
    """On every row of the ten except the missing one, which the next test is about."""
    for name in ("isspace", "islower", "isupper"):
        mine = getattr(made(firepanda).str, name)().tolist()[:-1]
        assert mine == getattr(theirs().str, name)().tolist()[:-1], name


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

    pandas holding the column the way it holds it by default has nowhere to put
    a missing answer, because the answer is a numpy array of bools, so a missing
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
def test_three_measured_differences_in_the_case_questions(firepanda: ModuleType) -> None:
    """The gaps the three questions still have, written down on purpose.

    The two that rewrite a row are corrected against Arrow's table for all
    hundred and forty nine code points where the standard library underneath
    disagrees with it, so they match pandas everywhere. The three that ask a
    question are not corrected, because the same measurement counts more than a
    thousand code points that Arrow calls cased and the library here does not,
    which is a table rather than a list. Document 64 has the counts. These three
    rows are the ones a caller is most likely to meet, and they are here so that
    the day the library's data is replaced a test fails and somebody comes and
    reads the document.
    """
    assert made(firepanda, ["\u00a0"]).str.isspace().tolist() == [False]
    assert theirs(["\u00a0"]).str.isspace().tolist() == [True]
    assert made(firepanda, ["ĸ"]).str.islower().tolist() == [False]
    assert theirs(["ĸ"]).str.islower().tolist() == [True]
    assert made(firepanda, ["\u2102"]).str.isupper().tolist() == [False]
    assert theirs(["\u2102"]).str.isupper().tolist() == [True]
