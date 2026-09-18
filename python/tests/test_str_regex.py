"""The three `str` methods that ask a regular expression engine, against pandas.

`contains`, `match` and `fullmatch` are one question upstream and one question
here: whether a pattern matches somewhere in a row. The other two are that one
with the pattern anchored, and the anchoring is a rewrite rather than a mode, so
`str.match("a|b")` asks whether a row starts with either letter and not whether
it starts with `a` or holds a `b` anywhere.

The engine behind them is firepanda's own, and what it is measured against is
pandas over thirty thousand generated patterns in
`tests/differential/regex_match.mojo`. That is the test of what a pattern
matches. This file is the test of the wiring: that the accessor sends a pattern
to the engine when it has to, sends it to the byte search when it can, refuses
what neither can answer, and refuses it as the right kind of exception.

The two kinds of refusal are the part worth reading carefully. A pattern RE2
would refuse is refused upstream too, out of Arrow, as a `ValueError`, so that
is what it is here and a program written against pandas keeps working. A pattern
this library has not learned yet is a `NotImplementedError`, because a caller
who catches `ValueError` around a pattern they know to be good should not be
told they wrote a bad one.
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
"""The rows every pattern below is run over.

The first few are the ordinary cases. The row with a newline in it is there for
`$` and for the full stop, which read it differently, and the digits are there
for `\\d`, which reads a character outside ASCII differently again. The last row
is the missing one, which every one of these disagrees with pandas about for a
reason that is nothing to do with the engine.
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
"""Twenty patterns, each of which has to reach the engine.

The last two open with a global flag group, which is the one part of the rewrite
that is this library's own rather than a copy of upstream's, so both anchored
methods have something to say about them.
"""


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
    as `engine/string-predicate-null`, it has nothing to do with the engine, and
    `test_str_pattern.py` is where it is asserted.
    """
    return values[:-1]


@needs_pandas
def test_every_pattern_answers_what_pandas_answers(firepanda: ModuleType) -> None:
    """Sixty comparisons, each of them a column of ten rows.

    The differential is the real test of this and runs the same engine against
    the same pandas, so what a failure here means is that the accessor did not
    reach the engine or reached it with the wrong pattern rather than that the
    engine is wrong.
    """
    mine, them = made(firepanda), theirs()
    for pattern in PATTERNS:
        for name in ("contains", "match", "fullmatch"):
            got = without_the_missing(getattr(mine.str, name)(pattern).tolist())
            want = without_the_missing(getattr(them.str, name)(pattern).tolist())
            assert got == want, (name, pattern)


@needs_pandas
def test_match_anchors_the_whole_pattern_rather_than_its_first_arm(
    firepanda: ModuleType,
) -> None:
    """The one place the rewrite is visible in an answer rather than in a pattern.

    `match("a|b")` reads as `^(a|b)` and not as `^a|b`, so a row starting with
    `b` matches and a row holding one later does not. A rewrite that put the
    anchor on without the group would answer True for every row below.
    """
    rows = ["ba", "ab", "cb", "cab"]
    assert made(firepanda, rows).str.match("a|b").tolist() == [True, True, False, False]
    assert theirs(rows).str.match("a|b").tolist() == [True, True, False, False]


@needs_pandas
def test_a_flag_group_in_front_reaches_the_engine_rather_than_the_rewrite(
    firepanda: ModuleType,
) -> None:
    """The hoist, which is the rewrite this library does differently on purpose.

    `(?s)` makes the full stop match a newline and `(?m)` makes `^` match at
    every line, and both have to survive being anchored. Upstream leaves the
    group where it was and hands the result to Arrow. This library moves it to
    the front, because it reads its own rewrite with Python's grammar and that
    grammar will not have a global flag group anywhere else.

    `fullmatch("(?m)a")` is the case that pays for the care taken over the move.
    The anchors it adds are the ends of the row and not the ends of a line, even
    though the pattern asked for `m`, because upstream adds them outside the
    group the flag sits in. A hoist that let the flag reach them would say the
    first row below matches, which is a different column and not a slower one.
    """
    rows = ["a\nb", "ab", "b\na", "b"]
    for pattern in ("(?s)a.b", "(?m)^b", "(?m)a$", "(?m)a", "(?sm)a.b$", "(?s).b"):
        for name in ("contains", "match", "fullmatch"):
            got = getattr(made(firepanda, rows).str, name)(pattern).tolist()
            want = getattr(theirs(rows).str, name)(pattern).tolist()
            assert got == want, (name, pattern)


@needs_pandas
def test_an_end_of_text_at_the_end_survives_the_anchoring(
    firepanda: ModuleType,
) -> None:
    """`\\Z` is Python's spelling and RE2 has no such escape, so pandas rewrites
    it into `\\z` before it anchors anything. Rewriting it afterwards would be
    too late, since the anchoring puts a bracket after it and Arrow refuses one
    that is not at the end."""
    rows = ["ab", "abc", "ab\n"]
    for name in ("contains", "match", "fullmatch"):
        got = getattr(made(firepanda, rows).str, name)(r"ab\Z").tolist()
        assert got == getattr(theirs(rows).str, name)(r"ab\Z").tolist(), name


@needs_pandas
def test_a_pattern_with_no_metacharacter_still_takes_the_byte_search(
    firepanda: ModuleType,
) -> None:
    """The fast path, which is where most patterns a program writes end up.

    There is nothing in the answer that says which path a pattern took, so what
    is asserted is the answer and the fact that the engine and the search agree
    about it. `abc` is the same question either way and `a.c` is not, which is
    the line the accessor draws.
    """
    mine, them = made(firepanda), theirs()
    for pattern in ("abc", "ab", "", "日本"):
        for name in ("contains", "match", "fullmatch"):
            got = without_the_missing(getattr(mine.str, name)(pattern).tolist())
            want = without_the_missing(getattr(them.str, name)(pattern).tolist())
            assert got == want, (name, pattern)


@needs_pandas
def test_a_missing_row_stays_missing_and_na_fills_it(firepanda: ModuleType) -> None:
    """The engine never reads a null row, and `na` fills what it left."""
    mine = made(firepanda)
    for name in ("contains", "match", "fullmatch"):
        assert getattr(mine.str, name)("a.c").tolist()[-1] is None, name
        assert getattr(mine.str, name)("a.c", na=False).tolist()[-1] is False, name
        assert getattr(mine.str, name)("a.c", na=True).tolist()[-1] is True, name


def test_a_pattern_the_other_engine_would_run_is_not_implemented(
    firepanda: ModuleType,
) -> None:
    """A lookbehind or a backreference goes to Python's `re` upstream, and this
    engine has not got either of them.

    pandas answers these, so the refusal is a gap rather than a difference of
    opinion, and `NotImplementedError` is what a gap is spelled. The message
    says which gap it is and quotes the pattern back.

    The lookahead used to be on this list. Document 93 answered it, and the row
    that used to be here is now in `test_str_lookahead.py` asking the opposite
    question.
    """
    mine = made(firepanda)
    for pattern in (r"(a)\1", "(?<=a)b", "(?<!a)b"):
        for name in ("contains", "match", "fullmatch"):
            with pytest.raises(NotImplementedError) as caught:
                getattr(mine.str, name)(pattern)
            assert pattern in str(caught.value), (name, pattern)


def test_a_pattern_the_engine_refuses_is_a_value_error(firepanda: ModuleType) -> None:
    """Which is what pandas raises for these too, out of Arrow.

    A possessive quantifier and a comment group are both Python syntax that RE2
    has never had, and pandas hands the pattern to RE2 anyway, so a caller sees
    a `ValueError` upstream and sees one here. The wording is this library's own
    rather than Arrow's, because a caller here did not call Arrow.
    """
    mine = made(firepanda)
    for pattern in ("a*+", "(?#note)a", r"a\Zb"):
        for name in ("contains", "match", "fullmatch"):
            with pytest.raises(ValueError) as caught:
                getattr(mine.str, name)(pattern)
            assert pattern in str(caught.value), (name, pattern)


def test_case_folding_reaches_the_engine_rather_than_stopping_at_it(
    firepanda: ModuleType,
) -> None:
    """`case=False` folds the pattern against a table that is written now.

    pandas serves it by handing RE2 its own ignore case flag, and so does this.
    A literal pattern still goes to the byte search and folds through the other
    table, so the argument is one word to a caller and two paths underneath, and
    both of them answer here. What they answer is measured against pandas in
    `test_str_regex_case_and_flags.py`.
    """
    mine = made(firepanda)
    for name in ("contains", "match", "fullmatch"):
        folded = getattr(mine.str, name)("a.c", case=False)
        assert len(folded.tolist()) == len(mine.tolist()), name
        getattr(mine.str, name)("abc", case=False)


@needs_pandas
def test_regex_off_searches_for_the_characters_themselves(
    firepanda: ModuleType,
) -> None:
    """The way to ask about a metacharacter, and the engine never sees it."""
    rows = ["a.c", "abc", "a+c", "", None]
    for pattern in (".", "a.c", "+", "[", "$", "a|b"):
        got = made(firepanda, rows).str.contains(pattern, regex=False).tolist()
        want = theirs(rows).str.contains(pattern, regex=False).tolist()
        assert without_the_missing(got) == without_the_missing(want), pattern


def test_a_pattern_that_is_not_a_string_is_a_type_error(firepanda: ModuleType) -> None:
    """The refusal that has to come first, before anything decides which engine
    would have taken the pattern."""
    mine = made(firepanda)
    for name in ("contains", "match", "fullmatch"):
        with pytest.raises(TypeError):
            getattr(mine.str, name)(1)
