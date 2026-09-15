"""`str.replace` with a pattern in it, against pandas.

The fifth method that asks a regular expression engine and the first whose
answer is text. That changes two things at once. It needs to know where a match
started and where each of its groups matched rather than only whether there was
one, which is a different program. And it runs a different scan down the row
from the one `str.count` runs, in the same Arrow library, on the same pattern.

The three ways the two scans differ are all visible in ordinary answers and all
three are asserted below. Counting cuts the row after each match and replacing
does not. Counting moves its cursor a byte at a time and replacing moves it a
character at a time. Both refuse a match of no width that lands where the last
match ended, and only replacing has to decide what to do instead, which is to
copy one character across.

`test_str_count_regex.py` is the test of the other scan and `test_str_regex.py`
is the test of the wiring the two share, including what the two kinds of refusal
mean. What is asserted here is the whole path from the accessor down, plus the
grammar of the replacement, which is RE2's and is narrower than Python's.
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
    "  a  ",
    "x1y22z333",
    None,
]
"""The rows the other four regular expression methods are run over, with three
more that only a replacement can tell apart.

The row of spaces around one letter is where a pattern of no width lands in the
middle rather than at an end, the row of letters and digits is where a group
reference has something to put back, and the row with characters wider than a
byte is where a cursor that moved in bytes would cut a character in half.
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
    r"\b",
    "x*",
    "(?m)$",
)
"""The twenty the other four are asked, and three more for the scan itself."""


def made(firepanda: ModuleType, values: list[Any] = ROWS) -> Any:
    """A firepanda text column."""
    return firepanda.Series(values)


def theirs(values: list[Any] = ROWS) -> Any:
    """The same column in pandas, held the way pandas holds it by default."""
    import pandas as pd

    return pd.Series(values, dtype="str")


def without_the_missing(values: list[Any]) -> list[Any]:
    """Drops the last row, which is the null the two libraries write differently.

    Both libraries leave a missing row missing, which is the part that matters,
    but pandas hands a missing text row out of `tolist` as a float nan and this
    library hands it out as None. That is the same disagreement the count test
    works around and it belongs to the way a column is read rather than to
    anything a replacement does, so the last row is checked on its own below.
    """
    return values[:-1]


@needs_pandas
def test_every_pattern_replaces_what_pandas_replaces(firepanda: ModuleType) -> None:
    """Two hundred and seventy six comparisons, one per pattern per row."""
    mine, them = made(firepanda), theirs()
    for pattern in PATTERNS:
        got = mine.str.replace(pattern, "#", regex=True).tolist()
        want = them.str.replace(pattern, "#", regex=True).tolist()
        assert without_the_missing(got) == without_the_missing(want), pattern


@needs_pandas
def test_the_row_is_not_cut_so_an_anchor_stays_where_it_was(
    firepanda: ModuleType,
) -> None:
    """The rule where the two Arrow scans part company.

    `str.count("^a")` on a row of four letters is four, because counting makes
    the rest of the row into a text of its own after every match. Replacing does
    not, so `^a` replaces once and a multiline `^` replaces once per line.
    """
    rows = ["aaaa", "abab", "a\nb\nc", "", "\n"]
    for pattern in ("^a", r"\Aa", "(?m)^", "^", "(?m)$", "$", r"\z"):
        got = made(firepanda, rows).str.replace(pattern, "#", regex=True).tolist()
        want = theirs(rows).str.replace(pattern, "#", regex=True).tolist()
        assert got == want, pattern


@needs_pandas
def test_the_cursor_moves_a_character_at_a_time(firepanda: ModuleType) -> None:
    """Which is the other Arrow scan's rule the other way round.

    A five character word written in six bytes takes six markers from a pattern
    that matches nothing, where `str.count` of the same pattern on the same row
    answers seven.
    """
    rows = ["héllo", "日本語", "٣٤", "abc", ""]
    for pattern in ("x*", "a{0}", "(?:q)?", r"\b"):
        got = made(firepanda, rows).str.replace(pattern, "#", regex=True).tolist()
        want = theirs(rows).str.replace(pattern, "#", regex=True).tolist()
        assert got == want, pattern


@needs_pandas
def test_an_empty_match_where_the_last_one_ended_is_thrown_away(
    firepanda: ModuleType,
) -> None:
    """The rule Python's own `re` does not have.

    `re.sub("a*", "#", "abc")` is `##b#c#` because the empty match just after
    the `a` is a match like any other. Arrow refuses it and copies the `b` out
    instead, so pandas answers `#b#c#`.
    """
    rows = ["abc", "aaaa", "  a  ", "", "ab ba"]
    for pattern in ("a*", "b?", "a{0,2}", r"a|\b"):
        got = made(firepanda, rows).str.replace(pattern, "#", regex=True).tolist()
        want = theirs(rows).str.replace(pattern, "#", regex=True).tolist()
        assert got == want, pattern
    assert made(firepanda, ["abc"]).str.replace("a*", "#", regex=True).tolist() == ["#b#c#"]


@needs_pandas
def test_a_group_is_written_where_the_replacement_asks_for_it(
    firepanda: ModuleType,
) -> None:
    """The reason this method needed a different program from the other four.

    `\\0` is the whole match and `\\1` through `\\9` are the groups, numbered by
    the order their brackets opened. A group that did not take part contributes
    nothing rather than raising.
    """
    rows = ["x1y22z333", "abc", "", "aaaa", "ab ba"]
    pairs = (
        ("[0-9]+", "<\\0>"),
        ("([a-z])([0-9]+)", "\\2\\1"),
        ("(a+)", "\\1\\1"),
        ("(a)|(b)", "[\\1\\2]"),
        ("(a)(b)?", "<\\1|\\2>"),
        ("(a)", "\\10"),
    )
    for pattern, repl in pairs:
        got = made(firepanda, rows).str.replace(pattern, repl, regex=True).tolist()
        want = theirs(rows).str.replace(pattern, repl, regex=True).tolist()
        assert got == want, (pattern, repl)


@needs_pandas
def test_a_backslash_in_the_replacement_is_read_by_re2(firepanda: ModuleType) -> None:
    """Which is why `regex=True` goes to the engine whatever the pattern is.

    `str.replace("a", chr(92) * 2, regex=True)` puts one backslash in and
    `regex=False` puts two, on a pattern holding no metacharacter at all. So the
    replacement decides the path as much as the pattern does.
    """
    rows = ["abc", "aaaa", ""]
    for repl in ("\\\\", "x\\\\y", "\\\\\\\\"):
        got = made(firepanda, rows).str.replace("a", repl, regex=True).tolist()
        want = theirs(rows).str.replace("a", repl, regex=True).tolist()
        assert got == want, repl
    literal = made(firepanda, rows).str.replace("a", "\\\\", regex=False).tolist()
    assert literal == theirs(rows).str.replace("a", "\\\\", regex=False).tolist()


@needs_pandas
def test_a_replacement_re2_cannot_read_is_a_value_error(firepanda: ModuleType) -> None:
    """The three ways a rewrite string is wrong, all of them `ValueError`.

    pandas raises an Arrow error for each of these, which is a `ValueError` in
    Python, so a program written against pandas catches the same thing here.
    """
    mine = made(firepanda)
    for pattern, repl in (("a", "x\\"), ("a", "\\n"), ("a", "\\1")):
        with pytest.raises(ValueError):
            mine.str.replace(pattern, repl, regex=True)
        with pytest.raises(ValueError):
            theirs().str.replace(pattern, repl, regex=True)


def test_a_pattern_the_other_engine_would_run_is_not_implemented(
    firepanda: ModuleType,
) -> None:
    """A lookaround or a backreference goes to Python's `re` upstream."""
    mine = made(firepanda)
    for pattern in ("a(?=b)", "a(?!b)", r"(a)\1", "(?<=a)b"):
        with pytest.raises(NotImplementedError) as caught:
            mine.str.replace(pattern, "#", regex=True)
        assert pattern in str(caught.value), pattern


def test_a_pattern_the_engine_refuses_is_a_value_error(firepanda: ModuleType) -> None:
    """Which is what pandas raises for these too, out of Arrow."""
    mine = made(firepanda)
    for pattern in ("a*+", "(?#note)a", r"a\Zb"):
        with pytest.raises(ValueError) as caught:
            mine.str.replace(pattern, "#", regex=True)
        assert pattern in str(caught.value), pattern


def test_a_count_with_a_pattern_is_refused(firepanda: ModuleType) -> None:
    """Because upstream answers it out of a loop this library will not copy.

    Arrow's bounded replace finds a match and then asks RE2 to replace inside
    the text it found, which has no rule about a match of no width and does not
    move the cursor. `str.replace("a*", "#", n=5)` puts five markers at the
    front of a row it then leaves alone, and a pattern of no width raises on
    every row. A gap on the board is the honest answer to that.
    """
    mine = made(firepanda)
    for pattern in ("a.c", "a*", "(a)"):
        with pytest.raises(NotImplementedError):
            mine.str.replace(pattern, "#", n=2, regex=True)
    with pytest.raises(NotImplementedError):
        mine.str.replace("abc", "\\0", n=2, regex=True)


@needs_pandas
def test_a_count_with_a_plain_pattern_still_works(firepanda: ModuleType) -> None:
    """The refusal above is narrowed to the calls that would have needed the
    engine, so a literal pattern and a literal replacement keep their count."""
    rows = ["aaaa", "abcabc", ""]
    for limit in (0, 1, 2, 10):
        got = made(firepanda, rows).str.replace("a", "#", n=limit, regex=True).tolist()
        want = theirs(rows).str.replace("a", "#", n=limit, regex=True).tolist()
        assert got == want, limit


def test_a_named_group_in_the_replacement_is_not_implemented(
    firepanda: ModuleType,
) -> None:
    """pandas reads a replacement holding one out of Python's `re` rather than
    out of Arrow, and that engine is not written."""
    mine = made(firepanda)
    with pytest.raises(NotImplementedError):
        mine.str.replace("(a)", "\\g<1>", regex=True)


@needs_pandas
def test_a_missing_row_stays_missing(firepanda: ModuleType) -> None:
    """A text answer has somewhere to put a missing row, so both libraries do.

    This is the one kernel of the three where the builder writes the null rather
    than a repair pass clearing it afterwards, because a builder has to be told
    what a row is before it can be told what the next one is, so it is worth
    asserting on its own rather than leaving it inside the sweep.
    """
    import pandas as pd

    assert made(firepanda).str.replace("a.c", "#", regex=True).tolist()[-1] is None
    assert pd.isna(theirs().str.replace("a.c", "#", regex=True).tolist()[-1])


def test_a_pattern_that_is_not_a_string_is_a_type_error(firepanda: ModuleType) -> None:
    """The refusal that has to come first, before anything decides which engine
    would have taken the pattern."""
    with pytest.raises(TypeError):
        made(firepanda).str.replace(1, "#", regex=True)
