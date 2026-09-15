"""The four `str` methods that look for a pattern, checked against pandas.

`contains`, `match`, `fullmatch` and `count` are four questions about where a
pattern sits in a row: anywhere, at the front, the whole row, and how many
times. pandas reads the pattern as a regular expression in all four, and this
file is about the patterns where that reading and a byte search are the same
thing.

They are the same thing whenever the pattern holds none of the twelve regular
expression metacharacters, and the byte search is a great deal faster, so that
is the path all four take for a pattern like the ones below. The answers are
therefore pandas' answers rather than an approximation of them, and what is
being checked is that the fast path is the same question and not a cheaper one.

Where the two part company is a pattern with a metacharacter in it. Three of the
four send one to the regular expression engine now, and `test_str_regex.py` is
where that is checked. `count` does not, because counting needs where each match
ends and the engine answers whether there is one, so a metacharacter is still
refused there by name. The refusal is the one thing a compatibility layer must
not get wrong in the other direction: answering `count(".")` as a search for a
full stop would be wrong on every row of a column pandas counts everywhere.

Two things about the group are worth knowing before reading the assertions.

The first is that `count` is counted in bytes when the pattern is empty. Arrow
counts an empty match at every byte offset and once past the end, pandas 3 holds
text in Arrow and answers Arrow, and Python's own `re` module counts characters
and answers one less for every non ASCII character in the row. So
`pandas.Series(["héllo"], dtype="str").str.count("")` is 7 where
`len(re.findall("", "héllo"))` is 6, and this library follows pandas. It is the
same road the case classes took and it is written down for the same reason.

The second is that matches do not overlap. `count("aa")` on a row of four a's is
two, because the cursor moves past the whole pattern after a hit. That is what a
regular expression engine answers and it is what pandas answers.
"""

from __future__ import annotations

import importlib.util
import re
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

ROWS = [
    "abcabc",
    "ab",
    "",
    "ABC",
    "aaaa",
    "a b c",
    "héllo",
    "日本語",
    None,
]
"""Nine rows, each of them there to catch a different way of being wrong.

The first holds the pattern twice and holds it at the front. The second is the
pattern and nothing else, which is the only row `fullmatch` says yes to. The
third is empty and the fourth is the same letters in the other case, which is
the row that would pass if the search folded case. The fifth is the overlap
question. The sixth has the pattern's letters spread out so that a search that
forgot the pattern was several bytes would find it. The seventh and eighth are
non ASCII, which is where counting bytes and counting characters part company.
The last is the missing row.
"""

PATTERNS = ("bc", "a", "abc", "", "aa", "z", "héllo", "本")
"""Eight literal patterns, none of them holding a metacharacter."""


def made(firepanda: ModuleType, values: list[Any] = ROWS) -> Any:
    """A firepanda text column."""
    return firepanda.Series(values)


def theirs(values: list[Any] = ROWS) -> Any:
    """The same column in pandas, held the way pandas holds it by default."""
    import pandas as pd

    return pd.Series(values, dtype="str")


def without_the_missing(values: list[Any]) -> list[Any]:
    """Drops the last row, which is the null every one of these disagrees about.

    pandas hands back a bool column and a missing row becomes False in it, and
    this library keeps the row missing. That is the divergence the board records
    as `engine/string-predicate-null` and it has its own test at the bottom of
    this file, so the row is dropped where it is not the subject.
    """
    return values[:-1]


@needs_pandas
def test_contains_matches_pandas_on_every_literal_pattern(firepanda: ModuleType) -> None:
    """Both ways of asking, since a literal pattern reads the same either way."""
    mine, them = made(firepanda), theirs()
    for pattern in PATTERNS:
        want = without_the_missing(them.str.contains(pattern).tolist())
        assert without_the_missing(mine.str.contains(pattern).tolist()) == want, pattern
        got = mine.str.contains(pattern, regex=False).tolist()
        assert without_the_missing(got) == want, pattern


@needs_pandas
def test_match_and_fullmatch_match_pandas(firepanda: ModuleType) -> None:
    """The front of the row and the whole of it, which are two different questions."""
    mine, them = made(firepanda), theirs()
    for pattern in PATTERNS:
        for name in ("match", "fullmatch"):
            got = getattr(mine.str, name)(pattern).tolist()
            want = getattr(them.str, name)(pattern).tolist()
            assert without_the_missing(got) == without_the_missing(want), (name, pattern)


@needs_pandas
def test_count_matches_pandas(firepanda: ModuleType) -> None:
    """Including the empty pattern, which is the one counted in bytes."""
    mine, them = made(firepanda), theirs()
    for pattern in PATTERNS:
        got = without_the_missing(mine.str.count(pattern).tolist())
        want = without_the_missing(them.str.count(pattern).tolist())
        assert got == want, pattern


@needs_pandas
def test_an_empty_pattern_is_counted_in_bytes_and_python_counts_characters(
    firepanda: ModuleType,
) -> None:
    """All three answers written out, because one of them on its own reads as a typo.

    A row of five characters holding one accented letter is six bytes. pandas
    says seven empty matches, this library says seven, and `re` says six. The
    assertion on `re` is not testing `re`, it is recording that the difference
    was measured rather than assumed.
    """
    rows = ["héllo", "日本", "hello", ""]
    assert [len(re.findall("", row)) for row in rows] == [6, 3, 6, 1]
    assert theirs(rows).str.count("").tolist() == [7, 7, 6, 1]
    assert made(firepanda, rows).str.count("").tolist() == [7, 7, 6, 1]


@needs_pandas
def test_a_count_does_not_let_matches_overlap(firepanda: ModuleType) -> None:
    """Four a's hold two runs of two, and pandas agrees."""
    rows = ["aaaa", "aaaaa", "abab", "aa", "a"]
    assert made(firepanda, rows).str.count("aa").tolist() == [2, 2, 0, 1, 0]
    assert theirs(rows).str.count("aa").tolist() == [2, 2, 0, 1, 0]


@needs_pandas
def test_an_empty_pattern_is_in_every_row_and_is_only_the_empty_row(
    firepanda: ModuleType,
) -> None:
    """Which is three different answers from three methods on the same argument."""
    rows = ["abc", "", "a"]
    for name, want in (("contains", [True] * 3), ("match", [True] * 3)):
        assert getattr(made(firepanda, rows).str, name)("").tolist() == want, name
        assert getattr(theirs(rows).str, name)("").tolist() == want, name
    assert made(firepanda, rows).str.fullmatch("").tolist() == [False, True, False]
    assert theirs(rows).str.fullmatch("").tolist() == [False, True, False]


@needs_pandas
def test_count_refuses_a_pattern_with_a_metacharacter_rather_than_searching(
    firepanda: ModuleType,
) -> None:
    """And the refusal names the character, so a caller can tell which mistake it was.

    pandas answers all of these, so every one of them is a gap and not a
    difference of opinion. The refusal is the honest shape of a gap: the board
    reads it as unimplemented and a caller reads a sentence saying what is
    missing, where a literal search would have read as a wrong answer.

    The other three used to be here and answer these patterns now. What keeps
    `count` out is that it needs where each match ends in order to start looking
    for the next one, and the engine answers whether there is a match rather
    than where it is.
    """
    mine = made(firepanda)
    for pattern in ("a.c", "^a", "a+", "a|b", "[ab]", "a*", "a?", r"a\b", "(a)", "a{2}"):
        with pytest.raises(firepanda.errors.UnsupportedError) as caught:
            mine.str.count(pattern)
        assert "regular expression" in str(caught.value), pattern


@needs_pandas
def test_contains_with_regex_off_searches_for_the_characters_themselves(
    firepanda: ModuleType,
) -> None:
    """Which is the one way a metacharacter can be asked about, and pandas agrees."""
    rows = ["a.c", "abc", "a+c", ""]
    for pattern in (".", "a.c", "+", "[", "$"):
        got = made(firepanda, rows).str.contains(pattern, regex=False).tolist()
        assert got == theirs(rows).str.contains(pattern, regex=False).tolist(), pattern


@needs_pandas
def test_flags_are_refused_rather_than_ignored(
    firepanda: ModuleType,
) -> None:
    """An ignored argument is the one failure mode a compatibility layer must not have.

    `case=False` used to be refused here beside this and is answered now, which
    `test_str_case_insensitive.py` covers. A flag is a statement about how the
    engine is to read the pattern, and the engine reads none of them yet, so
    this one stays. pandas refuses `flags` out of Arrow as well and answers it
    out of its other engine.
    """
    mine = made(firepanda)
    for name in ("contains", "match", "fullmatch", "count"):
        with pytest.raises(firepanda.errors.UnsupportedError):
            getattr(mine.str, name)("abc", flags=re.IGNORECASE)


@needs_pandas
def test_a_pattern_that_is_not_a_string_is_a_type_error(firepanda: ModuleType) -> None:
    """pandas raises here too, and says the same thing."""
    mine = made(firepanda)
    for name in ("contains", "match", "fullmatch", "count"):
        with pytest.raises(TypeError):
            getattr(mine.str, name)(1)


@needs_pandas
def test_na_fills_the_missing_row_in_the_three_that_take_it(
    firepanda: ModuleType,
) -> None:
    """`count` has no `na`, which is the one place the four disagree about themselves."""
    mine = made(firepanda)
    for name in ("contains", "match", "fullmatch"):
        assert getattr(mine.str, name)("ab", na=False).tolist()[-1] is False, name
        assert getattr(mine.str, name)("ab", na=True).tolist()[-1] is True, name


@needs_pandas
def test_all_four_keep_a_missing_row_missing(firepanda: ModuleType) -> None:
    """Where pandas answers False for the three masks and a missing number for the count.

    The masks are the divergence the board records as
    `engine/string-predicate-null`, which every question the accessor asks about
    a row already sits inside. The count is the other one: pandas widens an
    int64 column to float64 to make room for the missing row and this library
    keeps it int64 and holds the row missing, which is the same disagreement the
    whole library has with pandas about where a missing integer lives.
    """
    mine = made(firepanda)
    for name in ("contains", "match", "fullmatch", "count"):
        assert getattr(mine.str, name)("ab").tolist()[-1] is None, name
    assert theirs().str.contains("ab").tolist()[-1] is False
    assert str(theirs().str.count("ab").dtype) == "float64"
