"""The inline `(?i)` flag on the pattern methods, against pandas.

The flag is not an argument. `case=False` is next door in
`test_str_case_insensitive.py` and is a literal search with a folding compare;
this file is the flag written inside the pattern, which reaches the regular
expression compiler and is spent there. pandas sends such a pattern to whichever
engine it would have sent the pattern to anyway, so an inline `(?i)` never moves
a call between Arrow and Python upstream, and the same is true here.

That matters because the two engines do not fold the same alphabet. They agree
about 2923 of the 2927 code points either of them considers cased, and they
disagree about the four Turkish I ones: Python reads `I`, `i`, the dotted `\u0130`
and the dotless `\u0131` as one letter, and RE2 reads the first two as one letter
and the other two as themselves. So `str.contains` and `str.extract` on the same
accessor with the same pattern and the same row answer differently, in pandas
and here, and the tests below assert that against live pandas rather than
against the rule, because the rule came from measuring pandas.

Everything else in here is the ordinary business of folding a pattern: both
cases of a letter match, a class folds every code point in its range rather than
the endpoints it was written with, a negated class folds before the caret is
applied rather than after, and the word and digit classes were already closed
under folding and do not move.
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
"""Rows picked so that a fold read off a lowercase table gets several wrong.

The Kelvin sign, the long s, the capital sharp s and the final sigma are each a
code point that folds onto a letter that is spelled nothing like it, and the
digit and the empty row are there so that a pattern that folded too widely would
be caught rather than merely be right."""

TURKISH = ["i", "I", "\u0130", "\u0131", "istanbul", "\u0130stanbul", None]
"""The four code points the two engines disagree about, and two words."""


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
    as `engine/string-predicate-null` and folding the pattern changes nothing
    about it.
    """
    return values[:-1]


def mask_of(them: Any) -> list[Any]:
    """pandas' mask as a list, with every flavour of missing read as None."""
    return [None if one is None or one != one else bool(one) for one in them.tolist()]


PATTERNS = [
    "a",
    "A",
    "abc",
    "ABC",
    "aBc",
    "[a-z]",
    "[A-Z]",
    "[^a]",
    "[^A]",
    "\\w",
    "\\W",
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
"""Twenty patterns, every one of them run with the flag on the front."""


@needs_pandas
def test_the_three_questions_fold_the_way_pandas_folds_them(firepanda: ModuleType) -> None:
    """contains, match and fullmatch, with the flag written into the pattern.

    All three of these stay on Arrow upstream and on RE2 here, so the alphabet
    is RE2's on both sides and the rows that separate the two engines are not
    in this frame.
    """
    mine, them = made(firepanda), theirs()
    for name in ("contains", "match", "fullmatch"):
        for pattern in PATTERNS:
            got = getattr(mine.str, name)("(?i)" + pattern)
            want = getattr(them.str, name)("(?i)" + pattern)
            assert without_the_missing(got.tolist()) == without_the_missing(mask_of(want)), (
                name,
                pattern,
            )


@needs_pandas
def test_counting_folds_the_same_way(firepanda: ModuleType) -> None:
    """`count` is the fourth method on the same engine and answers a number, so
    a fold that was one code point too wide shows up as a bigger difference here
    than in a mask."""
    mine, them = made(firepanda), theirs()
    for pattern in PATTERNS:
        got = mine.str.count("(?i)" + pattern)
        want = them.str.count("(?i)" + pattern)
        assert without_the_missing(got.tolist()) == without_the_missing(
            [None if one is None or one != one else int(one) for one in want.tolist()]
        ), pattern


@needs_pandas
def test_replacing_folds_the_same_way(firepanda: ModuleType) -> None:
    """The fifth method, which answers text, so a wrong fold is a wrong row
    rather than a wrong flag and is the easiest of the five to read."""
    mine, them = made(firepanda), theirs()
    for pattern in ("a", "[a-z]", "k", "\u00df", "\u03c3"):
        got = mine.str.replace("(?i)" + pattern, "#", regex=True)
        want = them.str.replace("(?i)" + pattern, "#", regex=True)
        assert got.tolist() == [
            None if one is None or one != one else one for one in want.tolist()
        ], pattern


@needs_pandas
def test_a_class_folds_every_code_point_in_its_range(firepanda: ModuleType) -> None:
    """The Kelvin sign and the long s arrive from a range that names neither,
    which is the assertion that the fold walks the code points a class covers
    rather than the endpoints it was written with."""
    rows = ["\u212a", "\u017f", "K", "k", "s"]
    mine, them = made(firepanda, rows), theirs(rows)
    for pattern in ("(?i)[a-z]", "(?i)[A-Z]", "(?i)[k-s]"):
        got = mine.str.contains(pattern)
        assert got.tolist() == mask_of(them.str.contains(pattern)), pattern


@needs_pandas
def test_a_negated_class_is_folded_before_it_is_negated(firepanda: ModuleType) -> None:
    """Folding the complement instead would put the small `a` back in through
    the other case of every letter that is not `a`, so `(?i)[^a]` would match
    everything. Both libraries answer the other way."""
    rows = ["a", "A", "b", "B"]
    mine, them = made(firepanda, rows), theirs(rows)
    for pattern in ("(?i)[^a]", "(?i)[^A]", "(?i)[^a-c]"):
        got = mine.str.contains(pattern)
        assert got.tolist() == mask_of(them.str.contains(pattern)), pattern
    assert mine.str.contains("(?i)[^a]").tolist() == [False, False, True, True]


@needs_pandas
def test_the_two_engines_disagree_about_the_turkish_i(firepanda: ModuleType) -> None:
    """The four code points that are the whole of the difference between them.

    `contains` is RE2 on both sides and `extract` is Python's `re` on both
    sides, so the same flag and the same letter over the same rows answers two
    different things in one accessor. That is upstream's behaviour and it is
    reproduced rather than smoothed over, which is what the two engines being
    two engines means.
    """
    mine, them = made(firepanda, TURKISH), theirs(TURKISH)
    got = mine.str.contains("(?i)i")
    assert without_the_missing(got.tolist()) == without_the_missing(
        mask_of(them.str.contains("(?i)i"))
    )
    # `\u0130stanbul` opens with the dotted capital and holds no plain `i` at
    # all, so RE2 answers False for it and Python answers True for the letter
    # on its own two rows above.
    assert without_the_missing(got.tolist()) == [True, True, False, False, True, False]

    ours = mine.str.extract("(?i)(i)", expand=False)
    upstream = them.str.extract("(?i)(i)", expand=False)
    assert ours.tolist() == [
        None if one is None or one != one else one for one in upstream.tolist()
    ]
    assert ours.tolist()[2] == "\u0130"
    assert ours.tolist()[3] == "\u0131"
    assert ours.tolist()[5] == "\u0130"


@needs_pandas
def test_the_plain_pair_folds_on_both_engines(firepanda: ModuleType) -> None:
    """The half of that family the two agree about, which is what makes the
    other half a difference rather than a missing table."""
    mine, them = made(firepanda, TURKISH), theirs(TURKISH)
    for pattern, method in (("(?i)i", "contains"), ("(?i)I", "contains")):
        got = getattr(mine.str, method)(pattern)
        assert without_the_missing(got.tolist())[:2] == [True, True], pattern
        assert without_the_missing(got.tolist()) == without_the_missing(
            mask_of(getattr(them.str, method)(pattern))
        ), pattern


@needs_pandas
def test_the_flag_reaches_a_pattern_that_was_anchored_first(firepanda: ModuleType) -> None:
    """`match` and `fullmatch` wrap the pattern before compiling it and the flag
    group has to stay in front of the wrapper for the grammar, so this is the
    case where a working fold and a broken hoist would answer differently."""
    rows = ["abc", "ABC", "xabc"]
    mine, them = made(firepanda, rows), theirs(rows)
    for pattern in ("(?i)abc", "(?i)a", "(?i)^abc$"):
        for name in ("match", "fullmatch"):
            got = getattr(mine.str, name)(pattern)
            assert got.tolist() == mask_of(getattr(them.str, name)(pattern)), (name, pattern)


@needs_pandas
def test_a_scoped_flag_group_is_still_refused(firepanda: ModuleType) -> None:
    """The parser reads `(?i:a)` and throws the letters away, so a program built
    from that tree would answer without folding while both engines fold. It is a
    gap and says so."""
    mine = made(firepanda)
    with pytest.raises(NotImplementedError):
        mine.str.contains("(?i:a)")


@needs_pandas
def test_the_case_argument_is_still_a_gap_on_a_regular_expression(
    firepanda: ModuleType,
) -> None:
    """The flag inside the pattern is carried and the argument beside it is not,
    which is the next slice rather than this one. A caller is told which."""
    mine = made(firepanda)
    with pytest.raises(NotImplementedError):
        mine.str.contains("a.c", case=False)
