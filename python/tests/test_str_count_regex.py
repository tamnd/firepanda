"""`str.count` with a pattern in it, against pandas.

The fourth method that asks a regular expression engine, and the first one that
does not ask it whether. `contains`, `match` and `fullmatch` want one answer per
row and stop at the first match. `count` wants the number of them, which means
running the pattern again from after each match, and the loop that does that is
where the three surprises live.

`test_str_regex.py` is the test of the other three and says what the two kinds
of refusal mean. This file is the test of the same wiring for `count`, plus the
loop, because the loop is not shared with anything and is not what a reader
would guess. The rules it follows were measured out of pandas rather than read
out of either engine, `firepanda/kernel/regex/pike.mojo` states them and
`tests/test_regex_count.mojo` holds them one at a time. What is asserted here is
that the whole path from the accessor down agrees with pandas on a column.

The one thing this file does not compare is the dtype. pandas hands back a float
column because the missing row has to be a `nan`, and this library hands back an
int64 column with the row missing. That is the same disagreement the three
predicates have and is nothing to do with counting, so the comparisons below are
between numbers and `2` equals `2.0`.
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
    "abcabc",
    "ab",
    "",
    "ABC",
    "aaaa",
    "a b c",
    "héllo",
    "日本語",
    "a\nb",
    "007",
    None,
]
"""The same rows the other three are run over, for the same reasons.

Two of them earn their place twice over here. The row with characters wider than
a byte matters because the loop counts a match of no width once per byte rather
than once per character, and the empty row matters because a pattern that can
match nothing matches it once.
"""

PATTERNS = (
    "a.c",
    "^a",
    "a+",
    "a|b",
    "[ab]",
    "a*",
    "a?",
    r"a\b",
    "(a)",
    "a{2}",
    r"\d",
    "b$",
    "[^a]",
    r"\w+\s\w+",
    "(abc)+",
    "a.*c",
    r"\Aab",
    r"c\Z",
    "(?s)a.b",
    "(?m)^b",
)
"""The twenty patterns the other three are asked, asked again as a count."""


def made(firepanda: ModuleType, values: list[Any] = ROWS) -> Any:
    """A firepanda text column."""
    return firepanda.Series(values)


def theirs(values: list[Any] = ROWS) -> Any:
    """The same column in pandas, held the way pandas holds it by default."""
    import pandas as pd

    return pd.Series(values, dtype="str")


def without_the_missing(values: list[Any]) -> list[Any]:
    """Drops the last row, which is the null the two libraries type differently."""
    return values[:-1]


@needs_pandas
def test_every_pattern_counts_what_pandas_counts(firepanda: ModuleType) -> None:
    """Two hundred comparisons, one per pattern per row.

    Most of these would pass under a loop that was wrong about all three rules,
    because most patterns match a fixed run of characters in the middle of a row
    and every sensible loop agrees about those. The ones that would not are
    `^a`, which counts four in a row of four letters, and `a*` and `a?`, which
    count the bytes of a row and one more.
    """
    mine, them = made(firepanda), theirs()
    for pattern in PATTERNS:
        got = without_the_missing(mine.str.count(pattern).tolist())
        want = without_the_missing(them.str.count(pattern).tolist())
        assert got == want, pattern


@needs_pandas
def test_an_anchor_is_read_against_what_is_left_of_the_row(
    firepanda: ModuleType,
) -> None:
    """The rule that reads wrong and is what pandas does.

    After a match the rest of the row becomes the text, so the start of the text
    moves with it and `count("^a")` on a row of four letters is four. Python's
    own `re` answers one for the same pattern and the same row. Nobody would
    choose this, and pandas has it because Arrow hands RE2 a fresh piece of text
    after each match rather than an offset into the old one.
    """
    rows = ["aaaa", "abab", "baaa", ""]
    for pattern in ("^a", r"\Aa", "(?m)^a", "^"):
        got = made(firepanda, rows).str.count(pattern).tolist()
        assert got == theirs(rows).str.count(pattern).tolist(), pattern


@needs_pandas
def test_a_match_of_no_width_is_counted_once_per_byte(firepanda: ModuleType) -> None:
    """Which is a different number from once per character for half the world.

    A five character word with one accented letter in it counts seven, because
    the letter is two bytes and there are six bytes to stand between plus the
    end. pandas counts an empty literal the same way and document 66 measured it
    there first, so the two paths through the accessor agree.
    """
    rows = ["héllo", "日本語", "٣٤", "abc", ""]
    for pattern in ("x*", "a{0}", "(?:q)?"):
        got = made(firepanda, rows).str.count(pattern).tolist()
        assert got == theirs(rows).str.count(pattern).tolist(), pattern


@needs_pandas
def test_a_boundary_found_ahead_of_the_scan_is_counted_twice(
    firepanda: ModuleType,
) -> None:
    """The third rule, and the one that took the longest to find.

    A word boundary in the middle of a row is found from the start of the row,
    which moves the scan to it, and then found again from where the scan now is.
    So a word with a space on each side of it has two boundaries and pandas
    counts two, where a loop that stepped past an empty match would say one and
    a loop that stepped one byte after every empty match would say three.
    """
    rows = ["  a  ", "a  ", "  a", "ab ba", "aaaa", "٣٤"]
    for pattern in (r"\b", r"\ba", r"a\b", "$", r"\z"):
        got = made(firepanda, rows).str.count(pattern).tolist()
        assert got == theirs(rows).str.count(pattern).tolist(), pattern


@needs_pandas
def test_the_first_arm_of_an_alternation_wins(firepanda: ModuleType) -> None:
    """Which changes the count even though it never changes whether.

    `a|aa` ends after one character and `aa|a` after two, so the same row of
    four letters holds four of the first and two of the second. Both engines
    prefer the arm the pattern wrote first, which is leftmost first, and a
    leftmost longest engine would answer two for both.
    """
    rows = ["aaaa", "aa", "a", ""]
    for pattern in ("a|aa", "aa|a", "a|ab", "ab|a"):
        got = made(firepanda, rows).str.count(pattern).tolist()
        assert got == theirs(rows).str.count(pattern).tolist(), pattern


@needs_pandas
def test_a_pattern_with_no_metacharacter_still_takes_the_byte_search(
    firepanda: ModuleType,
) -> None:
    """The fast path, which `count` had before it had an engine.

    Nothing in the answer says which path a pattern took, so what is asserted is
    that the two paths agree wherever they can both be asked, and the empty
    pattern is the one place they can be asked the same question and get the
    interesting answer.
    """
    mine, them = made(firepanda), theirs()
    for pattern in ("abc", "ab", "", "日本", "a"):
        got = without_the_missing(mine.str.count(pattern).tolist())
        want = without_the_missing(them.str.count(pattern).tolist())
        assert got == want, pattern


@needs_pandas
def test_a_missing_row_stays_missing(firepanda: ModuleType) -> None:
    """`count` has no `na`, so there is nothing to fill the row with.

    pandas types the column as float to fit a `nan` in it. This library keeps
    the count an integer and keeps the row missing, which is the better of the
    two answers and is still a difference, so it is asserted rather than
    compared.
    """
    assert made(firepanda).str.count("a.c").tolist()[-1] is None
    assert made(firepanda).str.count("abc").tolist()[-1] is None


@needs_pandas
def test_a_pattern_the_other_engine_would_run_is_answered(
    firepanda: ModuleType,
) -> None:
    """An atomic group goes to Python's `re` upstream, and `count` answers it.

    Upstream routes `count` by the same rule it routes the other three by, so
    the same patterns leave Arrow. Both halves of the lookaround used to be on
    this list and so did the backreference, documents 93, 94 and 95 answered
    those, and #994 answered the atomic group, so the list is empty and the row
    compares instead of refusing. The route did not change, only what waits at
    the end of it, and the lookahead in front of each pattern below is what
    routes it.
    """
    mine, them = made(firepanda), theirs()
    for pattern in (r"(?=a)(?>a)b", r"(?=a)a*+b"):
        got = without_the_missing(mine.str.count(pattern).tolist())
        want = without_the_missing(them.str.count(pattern).tolist())
        assert got == want, pattern


def test_a_pattern_the_engine_refuses_is_a_value_error(firepanda: ModuleType) -> None:
    """Which is what pandas raises for these too, out of Arrow."""
    mine = made(firepanda)
    for pattern in ("a*+", "(?#note)a", r"a\Zb"):
        with pytest.raises(ValueError) as caught:
            mine.str.count(pattern)
        assert pattern in str(caught.value), pattern


def test_flags_move_the_call_to_the_other_engine(firepanda: ModuleType) -> None:
    """`count` takes flags and upstream serves them by handing the pattern to
    Python's `re`, which counts by a different rule from the Arrow kernel this
    file is otherwise about.

    Both rules are asserted here on one pattern, because the pair is the point.
    An empty pattern is a match of no width at every position, Arrow steps one
    byte past one and Python steps one character, so a row holding a two byte
    letter is counted two different ways one keyword apart.
    `test_str_count_replace_python_engine.py` has the rest of that engine.
    """
    import re

    mine = made(firepanda, ["\u00df"])
    assert mine.str.count("").tolist() == [3]
    assert mine.str.count("", flags=re.IGNORECASE).tolist() == [2]


def test_a_pattern_that_is_not_a_string_is_a_type_error(firepanda: ModuleType) -> None:
    """The refusal that has to come first, before anything decides which engine
    would have taken the pattern."""
    with pytest.raises(TypeError):
        made(firepanda).str.count(1)
