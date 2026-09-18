"""The `case` and `flags` arguments on the pattern methods, against pandas.

`(?i)` written inside the pattern is next door in `test_str_fold_regex.py`. This
file is the same fold asked for from outside, which upstream turns into the
inside one by compiling the pattern with the argument before anything routes it.
So the two spellings have to answer alike, and the first test here asserts that
as an equality over rows rather than trusting the route.

`flags` is a stranger argument than it looks. Only `match` reads it and gets to
Arrow, because `match` alone compiles the pattern first and the routing test
then asks what the compiled pattern holds rather than what was passed with it, and
a pattern carrying nothing but ignore case is not a pattern with flags to that
test. `contains`, `fullmatch` and `count` hand any flag at all straight to
Python's engine, whose scan is a different scan, so they are refused here until
that scan is written.

That leaves `match` with two refusals of its own, both `ValueError` upstream and
both reproduced. A `case` beside a `flags` that disagrees with it is caught
first, by reading the compiled pattern's own ignore case bit back out. A flag
beyond ignore case is caught second, by a check that compares the flags the
accessor just zeroed against the ones it just compiled in. The order is measured
rather than guessed: a call that trips both gets the first message.
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
    "abc",
    "ABC",
    "AbC",
    "",
    "k",
    "K",
    "\u212a",
    "\u017f",
    "s",
    "\u00df",
    "\u1e9e",
    "\u03c3",
    "\u03c2",
    "\u03a3",
    "0",
    None,
]
"""The same rows the written flag is measured on, so the two files compare.

The Kelvin sign, the long s and the three sigmas are the rows where a fold that
went through a lowercase table rather than through the whole cycle answers
wrongly, and they are also the rows where the byte search and the engine could
have drifted apart without anybody noticing."""

PATTERNS = [
    "a",
    "A",
    "abc",
    "ABC",
    "aBc",
    "[a-z]",
    "[A-Z]",
    "[^a]",
    "\\w",
    "\\d",
    "k",
    "s",
    "\u00df",
    "\u03c3",
    "a.c",
    "^a",
    "c$",
    "a|B",
]
"""Half of these hold no metacharacter, which is the point.

A pattern with nothing special in it goes to the byte search and a pattern with
something special in it goes to the engine, and the two fold through different
tables: one maps a character to a character and the other takes every code point
that folds onto the one written down. They agree on every row here and they are
not the same rule, so both halves of the list are needed."""


def made(firepanda: ModuleType, values: list[Any] = ROWS) -> Any:
    """A firepanda text column."""
    return firepanda.Series(values)


def theirs(values: list[Any] = ROWS) -> Any:
    """The same column in pandas, held the way pandas holds one by default."""
    import pandas as pd

    return pd.Series(values, dtype="str")


def without_the_missing(values: list[Any]) -> list[Any]:
    """Drops the last row, which is the null every mask here disagrees about.

    pandas hands back a bool column and a missing row becomes False in it, and
    this library keeps the row missing. That is the divergence the board records
    as `engine/string-predicate-null` and an argument on the call changes
    nothing about it.
    """
    return values[:-1]


def mask_of(them: Any) -> list[Any]:
    """pandas' mask as a list, with every flavour of missing read as None."""
    return [None if one is None or one != one else bool(one) for one in them.tolist()]


@needs_pandas
def test_the_case_argument_answers_what_pandas_answers(firepanda: ModuleType) -> None:
    """Eighteen patterns through each of the three methods, against live pandas.

    This is the assertion that matters, because the argument is served by two
    different searches depending on what is in the pattern and pandas serves all
    of it with one engine.
    """
    mine, them = made(firepanda), theirs()
    for name in ("contains", "match", "fullmatch"):
        for pattern in PATTERNS:
            got = getattr(mine.str, name)(pattern, case=False)
            want = getattr(them.str, name)(pattern, case=False)
            assert without_the_missing(got.tolist()) == without_the_missing(mask_of(want)), (
                name,
                pattern,
            )


@needs_pandas
def test_the_argument_and_the_written_flag_are_one_fact(firepanda: ModuleType) -> None:
    """`case=False` and a leading `(?i)` reach the compiler as the same tree.

    They get there by different routes, one seeded into the parser and one read
    out of the pattern, and a route that dropped the seed on the way would show
    up here as the engine patterns disagreeing while the literal ones agreed.
    """
    mine = made(firepanda)
    for name in ("contains", "match", "fullmatch"):
        for pattern in PATTERNS:
            argued = getattr(mine.str, name)(pattern, case=False)
            written = getattr(mine.str, name)("(?i)" + pattern)
            assert argued.tolist() == written.tolist(), (name, pattern)


@needs_pandas
def test_the_argument_reaches_a_pattern_that_was_anchored_first(firepanda: ModuleType) -> None:
    """`match` and `fullmatch` compile a second time on a pattern this library
    built, and the argument has to be handed to that parse as well. Dropping it
    there would leave `contains` folding and the other two not, which is one
    method out of three quietly disagreeing."""
    rows = ["abc", "ABC", "xabc", "ABCx"]
    mine, them = made(firepanda, rows), theirs(rows)
    for pattern in ("[a-z]+", "a.c", "^abc$", "a|B"):
        for name in ("match", "fullmatch"):
            got = getattr(mine.str, name)(pattern, case=False)
            assert got.tolist() == mask_of(getattr(them.str, name)(pattern, case=False)), (
                name,
                pattern,
            )


@needs_pandas
def test_case_true_is_the_ordinary_call(firepanda: ModuleType) -> None:
    """The default, spelled out, which has to stay the search it always was.

    `match` is the one whose default moved, from True to nothing, because it now
    has to tell a caller who passed True apart from a caller who passed nothing.
    Both of them mean the same search when no flag came with them.
    """
    mine, them = made(firepanda), theirs()
    for name in ("contains", "match", "fullmatch"):
        for pattern in ("a", "[a-z]", "abc"):
            got = getattr(mine.str, name)(pattern, case=True)
            plain = getattr(mine.str, name)(pattern)
            assert got.tolist() == plain.tolist(), (name, pattern)
            assert without_the_missing(got.tolist()) == without_the_missing(
                mask_of(getattr(them.str, name)(pattern, case=True))
            ), (name, pattern)


@needs_pandas
def test_match_takes_the_ignore_case_flag_as_an_argument(firepanda: ModuleType) -> None:
    """The one name and the one flag that reach Arrow upstream.

    A compiled pattern carrying nothing but ignore case and the Unicode bit is
    not a pattern with flags to the routing test, so it stays on Arrow, and this
    is the only way a `flags` argument gets an answer out of any of the four.
    """
    mine, them = made(firepanda), theirs()
    for pattern in ("a", "[a-z]", "k", "\u03c3", "a.c"):
        got = mine.str.match(pattern, flags=re.IGNORECASE)
        assert without_the_missing(got.tolist()) == without_the_missing(
            mask_of(them.str.match(pattern, flags=re.IGNORECASE))
        ), pattern
        assert got.tolist() == mine.str.match(pattern, case=False).tolist(), pattern


@needs_pandas
def test_match_takes_the_unicode_flag_and_it_means_nothing(firepanda: ModuleType) -> None:
    """`re.UNICODE` is on for every string pattern in Python 3 and upstream ors
    it in whatever the caller passed, so a pattern compiled with it alone is
    still a pattern with no flags as far as the route is concerned."""
    mine, them = made(firepanda), theirs()
    got = mine.str.match("a", flags=re.UNICODE)
    assert without_the_missing(got.tolist()) == without_the_missing(
        mask_of(them.str.match("a", flags=re.UNICODE))
    )
    assert got.tolist() == mine.str.match("a").tolist()


@needs_pandas
def test_match_refuses_a_case_that_disagrees_with_its_flag(firepanda: ModuleType) -> None:
    """Upstream reads the ignore case bit back out of the pattern it just
    compiled and compares it to the argument, so the two ways of asking for the
    same fold have to agree with each other. `case=False` beside `re.I` is the
    one combination that gets through."""
    mine = made(firepanda)
    for case, flags in ((True, re.IGNORECASE), (False, re.MULTILINE), (False, re.DOTALL)):
        with pytest.raises(ValueError, match="conflicting case-sensitivity"):
            mine.str.match("a", case=case, flags=flags)
    assert mine.str.match("a", case=False, flags=re.IGNORECASE).tolist() == (
        mine.str.match("a", case=False).tolist()
    )


@needs_pandas
def test_match_refuses_a_flag_beyond_ignore_case(firepanda: ModuleType) -> None:
    """The second of the two, and the one that fires only after the first has
    let the call through. `str.fullmatch` with the same argument answers
    upstream, so this is `match` being `match` rather than a rule about
    flags."""
    mine = made(firepanda)
    for flags in (re.MULTILINE, re.DOTALL, re.VERBOSE, re.IGNORECASE | re.MULTILINE):
        with pytest.raises(ValueError, match=r"do not match pat\.flags"):
            mine.str.match("a", flags=flags)


@needs_pandas
def test_the_conflict_is_reported_before_the_flag_is(firepanda: ModuleType) -> None:
    """A call that trips both checks gets the first message, because upstream
    makes the comparison in the accessor and leaves the flag check to the array.
    The order is the only thing being asserted and it was measured."""
    mine, them = made(firepanda), theirs()
    with pytest.raises(ValueError, match="conflicting case-sensitivity"):
        mine.str.match("a", case=False, flags=re.MULTILINE)
    with pytest.raises(ValueError, match="conflicting case-sensitivity"):
        them.str.match("a", case=False, flags=re.MULTILINE)


@needs_pandas
def test_the_other_three_take_the_flag_to_the_other_engine(
    firepanda: ModuleType,
) -> None:
    """`contains`, `fullmatch` and `count` hand a pattern with any flag argument
    to Python's engine upstream, ignore case included, and that engine scans
    differently from Arrow. All three are answered out of it now.
    `test_str_flags_python_engine.py` measures the two that ask for a mask and
    `test_str_count_replace_python_engine.py` measures the counting loop, which
    is the rule the two engines do not share."""
    mine = made(firepanda)
    for flags in (re.IGNORECASE, re.MULTILINE, re.UNICODE):
        for name in ("contains", "fullmatch"):
            assert getattr(mine.str, name)("abc", flags=flags).tolist()[0] is True
        assert mine.str.count("a", flags=flags).tolist()[0] >= 0


@needs_pandas
def test_extract_takes_a_flag_without_taking_a_route_with_it(
    firepanda: ModuleType,
) -> None:
    """The last of the six, and the only one where the flag moves nothing.

    Every other name on this accessor reads a flags argument as two facts, what
    the letters mean and that the call has left Arrow. `extract` is one of the
    three pandas never sends to Arrow, so the second fact has nowhere to go and
    the letters cross on their own. It is also the one pattern method with no
    `case` argument, so it is the one where the two spellings of a fold cannot
    disagree.
    """
    mine = made(firepanda)
    assert mine.str.replace("a", "-", flags=re.IGNORECASE, regex=True).tolist()[0] is not None
    assert mine.str.extract("(A)", flags=re.IGNORECASE)["0"].tolist()[0] == "a"
    assert theirs().str.extract("(A)", flags=re.IGNORECASE)[0].tolist()[0] == "a"
