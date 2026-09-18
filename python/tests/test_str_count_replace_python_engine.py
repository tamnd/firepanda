"""`count` and `replace` on Python's engine, against pandas.

The two mask methods went to that engine first because a mask is the one answer
both engines already knew how to give. These two need a loop around the engine
as well, and the loop is where they differ: `re.finditer` never cuts the row
down, looks at a match of no width a second time before it moves along, and
judges an anchor against the row rather than against what is left of it.
Arrow's two kernels each do something else, and they do not agree with each
other either. So the same `count("^")` is four upstream without a flag and one
with `re.M`, and that pair is a fact about pandas rather than about this
library.

Four shapes of call land here and only two of them mention case. A flag and a
`case=False` are the ordinary two. The other two are `regex=True` with an empty
pattern, which pandas sends to `re.sub` because pyarrow used not to terminate on
one, and a replacement holding a named group reference, which pandas reads out
of `re` because Arrow's rewrite grammar cannot spell one. Both of those arrive
with no flags at all, which is why the engine has to travel as a word rather
than as a number.

`case=False` on `replace` is the change most likely to surprise a reader of the
four names above it. There it picks a byte fold that maps one character to one
character. Here it does not, because pandas turns it into `re.IGNORECASE` and
compiles the pattern with it, escaping the pattern first when `regex` is False
rather than taking a different path. So the fold is the engine's both ways
round, and the four Turkish I code points are where that shows.
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
    "aaa",
    "",
    "  a  ",
    "a b",
    "K",
    "\u212a",
    "k",
    "s",
    "\u017f",
    "\u00df",
    "STRASSE",
    "stra\u00dfe",
    "i",
    "I",
    "\u0130",
    "\u0131",
    "a\nb",
    "a\n",
    "\nb",
    "h\u00e9llo",
    "0",
    None,
]
"""The rows the mask file uses, plus the ones a scan needs.

A row of spaces around a letter is where an empty match and a boundary both
land several times, a row of three letters is where a greedy star and a cut row
part company, and the empty row is where a scan that asks about a position it
has already answered at runs off the end. The rest are the fold and class rows,
which matter here for the same reason they matter there.
"""

PATTERNS = [
    "a",
    "a*",
    "a+",
    "",
    "\\b",
    "\\B",
    "^",
    "$",
    "^a",
    "a$",
    "\\w",
    "\\W",
    "\\s",
    "[a-z]",
    "k",
    "s",
    ".",
    ".*",
    "(a)(b)",
    "(?:ab)+",
    "a|b",
    "\u00df",
    "a b",
    "[a b]",
    "a*?",
    "b*|a",
    "\\b|a",
]
"""Twenty seven patterns, every one of them run with every flag combination.

`a b` and `[a b]` are there for verbose mode, which reads one of them as two
characters and the other as three because the skip happens outside a class and
not inside one. Every other letter reads both of them as written.

The last three are there for the rule about looking at a position twice. Each of
them prefers to match nothing where it could have matched something, a lazy star
by being lazy and the other two by the order their arms are written in, so each
of them is a row where a scan that stepped one character on after a match of no
width would lose the wider match that upstream finds. Document 93.
"""

FLAGS = [
    re.IGNORECASE,
    re.MULTILINE,
    re.DOTALL,
    re.MULTILINE | re.DOTALL,
    re.IGNORECASE | re.MULTILINE,
    re.UNICODE,
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
    re.ASCII,
    re.IGNORECASE | re.ASCII,
    re.VERBOSE,
    re.VERBOSE | re.ASCII,
]
"""The combinations of the six letters this library reads, plus the one that
asks for what this engine does anyway."""


def made(firepanda: ModuleType, values: list[Any] = ROWS) -> Any:
    """A firepanda text column."""
    return firepanda.Series(values)


def theirs(values: list[Any] = ROWS) -> Any:
    """The same column in pandas, held the way pandas holds one by default."""
    import pandas as pd

    return pd.Series(values, dtype="str")


def counts_of(them: Any) -> list[Any]:
    """pandas' counts as a list, with every flavour of missing read as None."""
    return [None if one is None or one != one else int(one) for one in them.tolist()]


def texts_of(them: Any) -> list[Any]:
    """pandas' text answers as a list, with every flavour of missing read as None."""
    return [None if one is None or one != one else one for one in them.tolist()]


@needs_pandas
def test_count_answers_what_pandas_answers_under_every_flag(
    firepanda: ModuleType,
) -> None:
    """The sweep, which is 154 columns of numbers compared row for row.

    Nothing is held out. A missing row counts as missing on both sides, which is
    the one thing `count` agrees about with pandas where the mask methods do
    not, because a count column carries its own missing values rather than
    collapsing them into False.
    """
    mine, them = made(firepanda), theirs()
    for pattern in PATTERNS:
        for flags in FLAGS:
            got = mine.str.count(pattern, flags=flags)
            want = them.str.count(pattern, flags=flags)
            assert got.tolist() == counts_of(want), (pattern, flags)


@needs_pandas
def test_replace_answers_what_pandas_answers_under_every_flag(
    firepanda: ModuleType,
) -> None:
    """The same sweep for the replacing loop, with a marker that cannot be
    confused with anything in a row and with a limit on half the runs, since the
    limit is the one argument that can tell where the cursor was left."""
    mine, them = made(firepanda), theirs()
    for pattern in PATTERNS:
        for flags in FLAGS:
            for n in (-1, 0, 1, 2):
                got = mine.str.replace(pattern, "#", n=n, flags=flags, regex=True)
                want = them.str.replace(pattern, "#", n=n, flags=flags, regex=True)
                assert got.tolist() == texts_of(want), (pattern, flags, n)


@needs_pandas
def test_a_match_of_no_width_gets_a_second_look_rather_than_a_step(
    firepanda: ModuleType,
) -> None:
    """The rule upstream changed in 3.7 and this library had wrong until now.

    A match of no width does not move the cursor. The cursor stays where it is
    and the pattern is asked again at that one position with its end refused, so
    an arm that reads a character gets its turn where the arm that read nothing
    had already answered. `a b` holds one space and `count(r"(?!x)|\\s")` is
    five: nothing at each of the four positions, and the space as well.

    A scan that stepped one character on instead answers four and leaves the
    space in the row when it replaces, which is what this file used to do. Both
    halves are asserted, because the two loops are two pieces of code and the
    counting one was wrong in a way that only adds up to a number.
    """
    rows = ["a b", " ", "x {} y", "abc"]
    mine, them = made(firepanda, rows), theirs(rows)
    for pattern in (r"(?!x)|\s", "a*?", r"\b|a", "b*|a"):
        got = mine.str.count(pattern, flags=re.IGNORECASE)
        want = them.str.count(pattern, flags=re.IGNORECASE)
        assert got.tolist() == counts_of(want), pattern
        wrote = mine.str.replace(pattern, "#", flags=re.IGNORECASE, regex=True)
        theirs_wrote = them.str.replace(pattern, "#", flags=re.IGNORECASE, regex=True)
        assert wrote.tolist() == texts_of(theirs_wrote), pattern
    assert mine.str.count(r"(?!x)|\s", flags=re.IGNORECASE).tolist() == [5, 3, 8, 4]


@needs_pandas
def test_the_two_counting_loops_disagree_about_a_cut_row(
    firepanda: ModuleType,
) -> None:
    """Arrow makes the rest of the row into a new text after every match, so a
    front anchor is at the front again and again. Python searches from an offset
    into a row that stays whole. One keyword apart, four and one."""
    mine, them = made(firepanda, ["abc"]), theirs(["abc"])
    assert mine.str.count("^").tolist() == [4]
    assert mine.str.count("^", flags=re.MULTILINE).tolist() == [1]
    assert them.str.count("^").tolist() == [4]
    assert them.str.count("^", flags=re.MULTILINE).tolist() == [1]


@needs_pandas
def test_the_two_counting_loops_disagree_about_bytes_and_characters(
    firepanda: ModuleType,
) -> None:
    """Arrow steps one byte past a match of no width and a sharp s is two bytes,
    so an empty pattern counts the bytes of a row rather than its characters.
    Python steps one character. Three and two on the same row."""
    rows = ["\u00df", "h\u00e9llo"]
    mine, them = made(firepanda, rows), theirs(rows)
    assert mine.str.count("").tolist() == [3, 7]
    assert mine.str.count("", flags=re.IGNORECASE).tolist() == [2, 6]
    assert them.str.count("").tolist() == [3, 7]
    assert them.str.count("", flags=re.IGNORECASE).tolist() == [2, 6]


@needs_pandas
def test_a_limit_stops_the_scan_where_the_last_match_ended(
    firepanda: ModuleType,
) -> None:
    """Where the cursor was left is what this asserts. A scan stopped by its
    count writes the rest of the row out from the end of the last match, so the
    letter an empty match sat in front of is still there. `##bc` and not
    `##c`."""
    mine, them = made(firepanda, ["abc"]), theirs(["abc"])
    for kwargs in ({"case": False}, {"flags": re.IGNORECASE}):
        assert mine.str.replace("a*", "#", n=2, regex=True, **kwargs).tolist() == ["##bc"]
        assert them.str.replace("a*", "#", n=2, regex=True, **kwargs).tolist() == ["##bc"]


@needs_pandas
def test_a_count_of_zero_means_all_of_them_on_this_engine(
    firepanda: ModuleType,
) -> None:
    """`re.sub` reads a count of zero as unlimited and has done since long
    before pandas existed, and the Arrow path reads it as none, so the same
    `n=0` is two arguments depending on where the call landed. Upstream widens
    it by writing `n if n >= 0 else 0` and this widens it on the way out."""
    mine, them = made(firepanda, ["aaa"]), theirs(["aaa"])
    assert mine.str.replace("a", "#", n=0, flags=re.IGNORECASE, regex=True).tolist() == ["###"]
    assert them.str.replace("a", "#", n=0, flags=re.IGNORECASE, regex=True).tolist() == ["###"]
    assert mine.str.replace("a", "#", n=0, regex=True).tolist() == ["aaa"]
    assert them.str.replace("a", "#", n=0, regex=True).tolist() == ["aaa"]


@needs_pandas
def test_an_empty_pattern_goes_to_python_with_no_flags_at_all(
    firepanda: ModuleType,
) -> None:
    """pandas sends this one shape to `re.sub` because pyarrow used not to
    terminate on it, so the route cannot be read off the flags. It also means
    the replacement is read by Python's grammar there, which is the reason a
    backslash in it used to be refused rather than answered."""
    mine, them = made(firepanda, ["abc", "", "\u00df"]), theirs(["abc", "", "\u00df"])
    assert mine.str.replace("", "#", regex=True).tolist() == ["#a#b#c#", "#", "#\u00df#"]
    assert them.str.replace("", "#", regex=True).tolist() == ["#a#b#c#", "#", "#\u00df#"]
    assert mine.str.replace("", "\\n", regex=True).tolist() == texts_of(
        them.str.replace("", "\\n", regex=True)
    )


@needs_pandas
def test_a_named_group_in_the_replacement_moves_the_call(
    firepanda: ModuleType,
) -> None:
    """Upstream tests for `\\g<` in the replacement without looking at `regex`,
    which reads like it moves a literal replacement too and does not: the call
    it moves lands in a branch that asks `regex or flags or callable(repl)`
    before it reads the replacement as a template, and a literal call answers no
    to all three. So `\\g<0>` is the whole match one way and four characters of
    text the other, and both are asserted."""
    mine, them = made(firepanda, ["aXb"]), theirs(["aXb"])
    assert mine.str.replace("X", "\\g<0>", regex=True).tolist() == ["aXb"]
    assert them.str.replace("X", "\\g<0>", regex=True).tolist() == ["aXb"]
    assert mine.str.replace("X", "\\g<0>", regex=False).tolist() == ["a\\g<0>b"]
    assert them.str.replace("X", "\\g<0>", regex=False).tolist() == ["a\\g<0>b"]


@needs_pandas
def test_the_replacement_grammar_is_pythons_when_the_engine_is(
    firepanda: ModuleType,
) -> None:
    """Two grammars, one per engine, and the pattern picks which one reads the
    replacement. That is the one place where a fact about the pattern decides
    something about an argument that is not the pattern, and it is upstream's
    arrangement rather than one made here."""
    mine, them = made(firepanda, ["ab"]), theirs(["ab"])
    for template in ("\\g<2>\\g<1>", "\\2\\1", "\\g<0>", "\\123", "\\n", "\\\\", "\\-"):
        got = mine.str.replace("(a)(b)", template, regex=True, flags=re.IGNORECASE)
        want = them.str.replace("(a)(b)", template, regex=True, flags=re.IGNORECASE)
        assert got.tolist() == texts_of(want), template


@needs_pandas
def test_case_false_on_replace_folds_the_way_the_engine_folds(
    firepanda: ModuleType,
) -> None:
    """The four names above this one answer `case=False` out of a byte fold that
    maps one character to one character. `replace` does not, because pandas
    turns the argument into `re.IGNORECASE` and compiles the pattern with it
    whichever way `regex` was written, so the fold is the engine's.

    The two folds agree about more than a reader would expect, since the one to
    one table already carries a Kelvin sign onto a `k` and a long s onto an `s`.
    The four Turkish I code points are the whole of the difference between them,
    and they show it in both directions: `replace("i", case=False)` swaps a
    dotted capital I and `contains("i", case=False)` does not find one. Both
    halves are measured rather than reasoned."""
    rows = ["i", "I", "\u0130", "\u0131", "k", "\u212a"]
    mine, them = made(firepanda, rows), theirs(rows)
    for regex in (True, False):
        got = mine.str.replace("i", "#", case=False, regex=regex)
        assert got.tolist() == texts_of(them.str.replace("i", "#", case=False, regex=regex))
        assert got.tolist() == ["#", "#", "#", "#", "k", "\u212a"]
    assert mine.str.contains("i", case=False).tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert mine.str.replace("k", "#", case=False, regex=True).tolist()[5] == "#"


@needs_pandas
def test_a_literal_pattern_is_escaped_rather_than_taking_another_path(
    firepanda: ModuleType,
) -> None:
    """`regex=False` beside a `case` or a flag still reaches the engine, with
    the pattern run through `re.escape` first. That is upstream's own line and
    it is what keeps a dot a dot."""
    rows = ["a.c", "abc", "A.C"]
    mine, them = made(firepanda, rows), theirs(rows)
    for kwargs in ({"case": False}, {"flags": re.IGNORECASE}):
        got = mine.str.replace("a.c", "#", regex=False, **kwargs)
        assert got.tolist() == texts_of(them.str.replace("a.c", "#", regex=False, **kwargs))
        assert got.tolist() == ["#", "abc", "#"]


@needs_pandas
def test_a_bad_replacement_is_refused_in_a_class_pandas_does_not_use(
    firepanda: ModuleType,
) -> None:
    """Every one of these is an error upstream too and none of them is a
    `ValueError` there. `re` raises a `PatternError`, whose bases are `Exception`
    and nothing else, and an unknown group name comes back as an `IndexError`.
    This library raises the same class for a bad rewrite on either engine, so a
    caller catching `ValueError` catches one more thing here than upstream.
    Document 86 has that written down as a divergence rather than as a bug."""
    mine = made(firepanda, ["ab"])
    for template in ("\\8", "\\s", "a\\", "\\400", "\\g<>", "\\g<zz>", "\\g<1"):
        with pytest.raises(ValueError):
            mine.str.replace("(a)", template, regex=True, flags=re.IGNORECASE)


@needs_pandas
def test_the_two_spellings_of_the_fold_still_reach_two_engines_for_count(
    firepanda: ModuleType,
) -> None:
    """`count` has no `case` argument at all, which is the one place the five
    pattern methods disagree about their own signature, so the only way to ask
    it for a fold is the flag and the flag always moves it."""
    rows = ["\u212a", "k", "\u0130", "i"]
    mine, them = made(firepanda, rows), theirs(rows)
    assert mine.str.count("k", flags=re.IGNORECASE).tolist() == [1, 1, 0, 0]
    assert them.str.count("k", flags=re.IGNORECASE).tolist() == [1, 1, 0, 0]
    assert mine.str.count("k").tolist() == [0, 1, 0, 0]


def test_a_pattern_that_is_not_text_is_refused_before_the_flag_is_read(
    firepanda: ModuleType,
) -> None:
    """`re.compile` is the first thing to look at a pattern down every route
    upstream, and it is what raises for one that is not a string, so the type
    refusal comes before anything this library would have said about the
    flag."""
    mine = made(firepanda)
    with pytest.raises(TypeError):
        mine.str.count(5, flags=re.MULTILINE)
    with pytest.raises(TypeError):
        mine.str.replace(5, "-", flags=re.MULTILINE, regex=True)


def test_a_flag_value_that_names_no_letter_is_refused(firepanda: ModuleType) -> None:
    """The stray bits, which are refused rather than dropped. `re.DEBUG` is a
    letter `re` names and answers by printing the parsed pattern on the way
    past, and the other two are numbers that belong to nothing. All of it
    happens while the pattern compiles and the scan is downstream of that."""
    mine = made(firepanda)
    for flags in (re.DEBUG, 1024, 1 << 30):
        with pytest.raises(NotImplementedError):
            mine.str.count("a", flags=flags)
        with pytest.raises(NotImplementedError):
            mine.str.replace("a", "-", flags=flags, regex=True)


def test_the_locale_flag_is_refused_the_way_python_refuses_it(
    firepanda: ModuleType,
) -> None:
    """Python turns the locale flag down on a pattern made of text, and every
    pattern here is made of text, so the refusal is an error on both sides
    rather than this library falling short."""
    mine = made(firepanda)
    with pytest.raises(ValueError) as caught:
        mine.str.count("a", flags=re.LOCALE)
    assert "locale" in str(caught.value)
    with pytest.raises(ValueError) as again:
        mine.str.replace("a", "-", flags=re.LOCALE, regex=True)
    assert "locale" in str(again.value)


def test_a_count_with_a_pattern_is_still_refused_on_the_other_engine(
    firepanda: ModuleType,
) -> None:
    """Nothing here lifts the Arrow refusal, which is about a loop that replaces
    nothing after the first match and raises on a pattern of no width. A call
    with no flag, no `case` and a real pattern still lands on it."""
    mine = made(firepanda, ["abc"])
    with pytest.raises(NotImplementedError):
        mine.str.replace("a*", "#", n=1, regex=True)
    assert mine.str.replace("a*", "#", n=1, case=False, regex=True).tolist() == ["#bc"]
