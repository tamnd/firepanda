"""The `flags` argument on `contains` and `fullmatch`, against pandas.

A flag written `(?i)` inside the pattern never moves a call between the two
engines and is in `test_str_fold_regex.py`. A flag passed as an argument moves
four of the six pattern methods, and this file is the two of those four that ask
for a mask. `contains` and `fullmatch` want the one answer both engines already
knew how to give, so they were wired before either of the two scans Python's
engine needed. `count` and `replace` want those scans and are in
`test_str_count_replace_python_engine.py`.

The move is not cosmetic and this file is mostly about the ways it shows. The
two engines fold four code points differently, read `\\w` as 63 characters and
as 138558, disagree about a dollar sign in front of a trailing newline, and
disagreed about whether `\\B` matches an empty row until 3.14 settled it in
RE2's favour, which document 90 is about. All four of those are
reachable from here with one keyword, in pandas and in this library, and every
assertion below is made against live pandas rather than against that list.

Anchoring moves with the engine as well, because pandas stops rewriting the
pattern the moment it stops talking to Arrow. A call that went to Python's
engine is answered by `regex.fullmatch` rather than by a pattern with `^` and
`$` glued to it, and under the multiline flag those two are different questions.
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
    "\u03c3",
    "\u03a3",
    "0",
    "a\nb",
    "a\n",
    "\nb",
    "h\u00e9llo",
    "i",
    "I",
    "\u0130",
    "\u0131",
    "istanbul",
    "aaa",
    "a b",
    "  ",
    "a\u000bb",
    None,
]
"""Rows picked so that every one of the four engine differences has a row.

The Kelvin sign and the long s are where a fold read off a lowercase table goes
wrong, the two Turkish I code points are where the two engines part company, the
accented letter is where the two word classes part company, and the rows holding
a newline are where the two dollar signs do. The empty row is there twice over,
once for the fold and once for `\\B`.

The vertical tab is the row the ascii flag needs and it is the only one in this
list that discriminates a whole letter on its own. Python's ASCII `\\s` holds one
and RE2's `\\s` does not, so without this row every cell of the ascii column would
agree whether or not the narrow sets were right.
"""

PATTERNS = [
    "a",
    "[a-z]",
    "[^a]",
    "\\w+",
    "\\W",
    "\\d",
    "\\b",
    "\\B",
    "k",
    "s",
    "\u03c3",
    "a.c",
    "^a",
    "c$",
    "a|B",
    "a*",
    "(?i)k",
    "^b",
    "a.b",
    ".*",
    "\\s",
    "i",
    "(?:ab)+",
    "[a-z]{2}",
    "a b",
    "[a b]",
    "a # c",
]
"""Twenty seven patterns, every one of them run with every flag combination.

The last three are there for verbose mode and are ordinary patterns under every
other letter, which is the point of putting them in the sweep rather than in a
test of their own. A space and a hash mean one thing to six of the letters and
another thing to the seventh, and both readings are compared against pandas.
"""

FLAGS = [
    re.IGNORECASE,
    re.MULTILINE,
    re.DOTALL,
    re.MULTILINE | re.DOTALL,
    re.IGNORECASE | re.MULTILINE,
    re.UNICODE,
    re.IGNORECASE | re.UNICODE,
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
    re.ASCII,
    re.IGNORECASE | re.ASCII,
    re.VERBOSE,
    re.VERBOSE | re.ASCII,
]
"""The combinations of the six letters this library reads, plus the one that
means nothing. `re.LOCALE` is not in here because Python itself refuses it on a
pattern made of text, and it has a test of its own further down."""


def made(firepanda: ModuleType, values: list[Any] = ROWS) -> Any:
    """A firepanda text column."""
    return firepanda.Series(values)


def theirs(values: list[Any] = ROWS) -> Any:
    """The same column in pandas, held the way pandas holds one by default."""
    import pandas as pd

    return pd.Series(values, dtype="str")


def mask_of(them: Any) -> list[Any]:
    """pandas' mask as a list, with every flavour of missing read as None."""
    return [None if one is None or one != one else bool(one) for one in them.tolist()]


def without_the_missing(values: list[Any]) -> list[Any]:
    """Drops the last row, which is the null every mask here disagrees about.

    pandas hands back a bool column and a missing row becomes False in it, and
    this library keeps the row missing. That is the divergence the board records
    as `engine/string-predicate-null` and a flag changes nothing about it.
    """
    return values[:-1]


@needs_pandas
def test_the_two_masks_answer_what_pandas_answers_under_every_flag(
    firepanda: ModuleType,
) -> None:
    """The sweep, which is 648 columns of booleans compared row for row.

    A rule stated in a docstring and a rule the engine actually runs are two
    different things, and the only way to tell them apart is to ask both
    libraries the same question rather often. Nothing in here is held out.
    """
    mine, them = made(firepanda), theirs()
    for name in ("contains", "fullmatch"):
        for pattern in PATTERNS:
            for flags in FLAGS:
                got = getattr(mine.str, name)(pattern, flags=flags)
                want = getattr(them.str, name)(pattern, flags=flags)
                assert without_the_missing(got.tolist()) == without_the_missing(mask_of(want)), (
                    name,
                    pattern,
                    flags,
                )


@needs_pandas
def test_the_same_fold_asked_for_two_ways_reaches_two_engines(
    firepanda: ModuleType,
) -> None:
    """`case=False` stays on RE2 and `flags=re.IGNORECASE` does not.

    They are one argument by the time anything runs, upstream turns the second
    into the first, and they still answer differently, because the routing test
    upstream asks how the caller spelled it rather than what they asked for. The
    four Turkish I code points are the whole of the difference between the two
    fold tables and are what this asserts on.
    """
    rows = ["i", "I", "\u0130", "\u0131", "istanbul", "\u0130stanbul"]
    mine, them = made(firepanda, rows), theirs(rows)
    on_re2 = mine.str.contains("i", case=False)
    on_python = mine.str.contains("i", flags=re.IGNORECASE)
    assert on_re2.tolist() == mask_of(them.str.contains("i", case=False))
    assert on_python.tolist() == mask_of(them.str.contains("i", flags=re.IGNORECASE))
    assert on_re2.tolist() == [True, True, False, False, True, False]
    assert on_python.tolist() == [True, True, True, True, True, True]


@needs_pandas
def test_a_flag_widens_what_a_word_character_is(firepanda: ModuleType) -> None:
    """The second of the four differences and the one that covers the most
    patterns. A word character is 63 code points to RE2 and 138558 to Python, so
    a flag that moved the call moved what `\\w` means with it."""
    rows = ["h\u00e9llo", "\u00e9", "abc", "0", " "]
    mine, them = made(firepanda, rows), theirs(rows)
    for pattern in ("\\w", "\\W", "^\\w+$", "\\b"):
        got = mine.str.contains(pattern, flags=re.MULTILINE)
        assert got.tolist() == mask_of(them.str.contains(pattern, flags=re.MULTILINE)), pattern
    assert mine.str.contains("\\w", flags=re.MULTILINE).tolist()[1] is True
    assert mine.str.contains("\\w").tolist()[1] is False


@needs_pandas
def test_a_flag_changes_where_a_dollar_sign_may_sit(firepanda: ModuleType) -> None:
    """The third difference. RE2's dollar sign is the end of the text and
    Python's is the end of the text or just in front of a newline that ends it,
    so the same pattern over the same row answers two things one keyword
    apart."""
    rows = ["a\n", "a", "a\nb"]
    mine, them = made(firepanda, rows), theirs(rows)
    assert mine.str.contains("a$", flags=re.DOTALL).tolist() == mask_of(
        them.str.contains("a$", flags=re.DOTALL)
    )
    assert mine.str.contains("a$", flags=re.DOTALL).tolist() == [True, True, False]
    assert mine.str.contains("a$").tolist() == [False, True, False]


@needs_pandas
def test_the_non_boundary_gives_up_on_an_empty_row(firepanda: ModuleType) -> None:
    """The fourth difference, which was found by running this sweep.

    CPython up to 3.13 fails a `\\B` on an empty subject, which is a special
    case about the subject rather than a consequence of any rule about word
    characters. 3.14 took the case out and made `\\B` the plain negation of
    `\\b`, which is what RE2 has always had, so the empty row is one answer
    under one interpreter and the other answer under the next. Document 90 is
    where that was measured and this library reads the running interpreter
    rather than the one it was written on, which is why the expected value here
    is computed and not spelled.

    Only one of the two flag states is reachable here. The pattern with no flag
    beside it goes to RE2, which asks the question between bytes rather than
    between characters, and that is refused rather than answered wrongly. The
    refusal is asserted alongside so that the day it becomes an answer this test
    says so.
    """
    rows = ["", " ", "ab", "a"]
    empty = re.search("\\B", "") is not None
    mine, them = made(firepanda, rows), theirs(rows)
    assert mine.str.contains("\\B", flags=re.MULTILINE).tolist() == mask_of(
        them.str.contains("\\B", flags=re.MULTILINE)
    )
    assert mine.str.contains("\\B", flags=re.MULTILINE).tolist() == [
        empty,
        True,
        True,
        False,
    ]
    assert them.str.contains("\\B").tolist()[0] is True
    with pytest.raises(NotImplementedError):
        mine.str.contains("\\B")


@needs_pandas
def test_fullmatch_under_a_flag_is_anchored_from_outside_the_pattern(
    firepanda: ModuleType,
) -> None:
    """pandas answers a moved `fullmatch` with `regex.fullmatch` rather than
    with a pattern wrapped in `^` and `$`, and under the multiline flag those
    two are different questions: the wrapped one matches the first line of a row
    and the method does not."""
    rows = ["a\n", "a", "a\nb", "b"]
    mine, them = made(firepanda, rows), theirs(rows)
    for pattern, flags in (("a", re.MULTILINE), ("^a$", re.MULTILINE), ("a", re.DOTALL)):
        got = mine.str.fullmatch(pattern, flags=flags)
        assert got.tolist() == mask_of(them.str.fullmatch(pattern, flags=flags)), (
            pattern,
            flags,
        )
    assert mine.str.fullmatch("a", flags=re.MULTILINE).tolist() == [
        False,
        True,
        False,
        False,
    ]


@needs_pandas
def test_the_multiline_flag_reaches_an_anchor_the_caller_wrote(
    firepanda: ModuleType,
) -> None:
    """The flag has to be spent while the pattern compiles rather than after it,
    which is what a caret written by the caller is the test of."""
    rows = ["a\nb", "b\na", "ab"]
    mine, them = made(firepanda, rows), theirs(rows)
    for pattern in ("^b", "a$", "^b$"):
        got = mine.str.contains(pattern, flags=re.MULTILINE)
        assert got.tolist() == mask_of(them.str.contains(pattern, flags=re.MULTILINE)), pattern
    assert mine.str.contains("^b", flags=re.MULTILINE).tolist() == [True, True, False]
    assert mine.str.contains("^b").tolist() == [False, True, False]


@needs_pandas
def test_a_pattern_with_nothing_special_in_it_still_reaches_the_engine(
    firepanda: ModuleType,
) -> None:
    """The byte search is the faster path for a pattern holding no
    metacharacter, and under a flag it is the wrong one rather than merely an
    unnecessary one, because it is on the other engine and the two do not fold
    the same alphabet. A single letter is the shortest pattern there is and it
    still has to take the long way, which the dotted capital I is the proof of:
    the shortcut folds it onto nothing and the engine folds it onto `i`."""
    rows = ["\u212a", "k", "K", "\u017f", "s", "\u0130", "i"]
    mine, them = made(firepanda, rows), theirs(rows)
    for pattern in ("k", "s", "i"):
        got = mine.str.contains(pattern, flags=re.IGNORECASE)
        assert got.tolist() == mask_of(them.str.contains(pattern, flags=re.IGNORECASE)), pattern
    assert mine.str.contains("i", flags=re.IGNORECASE).tolist()[5] is True
    assert mine.str.contains("i", case=False).tolist()[5] is False


@needs_pandas
def test_the_case_argument_may_be_given_beside_the_flag(firepanda: ModuleType) -> None:
    """Upstream ors the one into the other and so does this, which is why a call
    passing both agrees with itself rather than raising the way `match` does."""
    mine, them = made(firepanda), theirs()
    for pattern in ("abc", "[a-z]", "k"):
        for case in (True, False):
            got = mine.str.contains(pattern, case=case, flags=re.MULTILINE)
            want = them.str.contains(pattern, case=case, flags=re.MULTILINE)
            assert without_the_missing(got.tolist()) == without_the_missing(mask_of(want)), (
                pattern,
                case,
            )


def test_the_locale_flag_is_refused_the_way_python_refuses_it(
    firepanda: ModuleType,
) -> None:
    """One of the three letters that does not go through, and the only one of
    the three that is not this library falling short. Python turns the locale
    flag down on a pattern made of text, and every pattern here is made of
    text, so the refusal is an error on both sides."""
    mine = made(firepanda)
    for name in ("contains", "fullmatch"):
        with pytest.raises(ValueError) as caught:
            getattr(mine.str, name)("a", flags=re.LOCALE)
        assert "locale" in str(caught.value)


def test_the_verbose_and_ascii_flags_answer_rather_than_refuse(
    firepanda: ModuleType,
) -> None:
    """The other two letters, which were the last two this library turned down.

    They are in the sweep above as well, and this is the pair of rows that says
    what each of them does rather than that it agrees. Verbose mode throws away
    a space that would otherwise have to be matched, and the ascii flag takes
    `\\w` back down from 138558 characters to 63.
    """
    mine = made(firepanda)
    verbose = mine.str.contains("a b", flags=re.VERBOSE).tolist()
    plain = mine.str.contains("a b", flags=re.MULTILINE).tolist()
    assert verbose[ROWS.index("abc")] is True
    assert verbose[ROWS.index("a b")] is False
    assert plain[ROWS.index("abc")] is False
    assert plain[ROWS.index("a b")] is True
    narrow = mine.str.contains("\\w", flags=re.ASCII).tolist()
    wide = mine.str.contains("\\w", flags=re.UNICODE).tolist()
    assert narrow[ROWS.index("\u212a")] is False
    assert wide[ROWS.index("\u212a")] is True


def test_a_flag_value_that_names_no_letter_is_refused(firepanda: ModuleType) -> None:
    """`re.DEBUG` is the one anybody reaches for by accident. Upstream answers
    it by printing the parsed pattern to the screen on the way past, which is
    not something this library can do quietly or loudly, so dropping the bit
    and answering would be answering a question nobody asked."""
    mine = made(firepanda)
    with pytest.raises(NotImplementedError):
        mine.str.contains("a", flags=re.DEBUG)


def test_a_flag_beside_a_literal_search_is_refused(firepanda: ModuleType) -> None:
    """A flag passed with `regex=False` looks ignored upstream and is not.

    It is spent on a route, and the route it picks happens to agree with the one
    without it until a `case` is beside it as well. Then the flag is the whole
    of the difference: Arrow compares without case and Python upper cases both
    sides, a sharp s upper cases to two letters and a Kelvin sign upper cases to
    itself, so the two disagree in both directions on the rows below. The
    upstream half is asserted here rather than described, because it is the
    reason this is a refusal instead of an answer.
    """
    rows = ["\u00df", "STRASSE", "\u212a"]
    them = theirs(rows)
    plain = {"case": False, "regex": False}
    flagged = {"case": False, "regex": False, "flags": re.IGNORECASE}
    assert them.str.contains("ss", **plain).tolist() == [False, True, False]
    assert them.str.contains("ss", **flagged).tolist() == [True, True, False]
    assert them.str.contains("k", **plain).tolist() == [False, False, True]
    assert them.str.contains("k", **flagged).tolist() == [False, False, False]
    mine = made(firepanda, rows)
    assert mine.str.contains("ss", **plain).tolist() == [False, True, False]
    assert mine.str.contains("k", **plain).tolist() == [False, False, True]
    for flags in (re.MULTILINE, re.IGNORECASE):
        with pytest.raises(NotImplementedError):
            mine.str.contains("a", flags=flags, regex=False)


def test_every_name_takes_a_flag_now(firepanda: ModuleType) -> None:
    """`count` and `replace` are served out of this engine and live in
    `test_str_count_replace_python_engine.py`. `extract` was the last one left
    and it never needed a route, only a number, because it is on this engine
    whatever anybody passes. `match` keeps its flags on Arrow and has two
    refusals of its own in `test_str_regex_case_and_flags.py`, which are about
    what upstream lets it keep rather than about what is written here."""
    mine = made(firepanda)
    assert mine.str.extract("(a)", flags=re.MULTILINE).shape[1] == 1
    assert mine.str.extract("(A)", flags=re.IGNORECASE)["0"].tolist()[0] == "a"


def test_a_bit_that_re_never_named_is_refused_like_any_other(firepanda: ModuleType) -> None:
    """A flag value holding a bit none of the seven letters own is refused, and the bits `re` does
    not name are refused the same way as the bits it does.

    This is one assertion about two different things. `re.DEBUG` belongs to the flag enumeration
    and was always refused. Bit ten belongs to nothing at all, and the mask the refusal compares
    against was built by oring seven enumeration members together, which makes it an enumeration
    member too, and the complement of one of those is bounded by the bits the enumeration defines
    rather than by the integer. So every bit `re` never named read as already known and was thrown
    away without a word. Every pattern method here shares the one helper, so every one of them had
    it.
    """
    mine = made(firepanda)
    for name in ("contains", "fullmatch", "count"):
        for flags in (1024, re.DEBUG, 1 << 30):
            with pytest.raises(firepanda.errors.UnsupportedError, match="flag value"):
                getattr(mine.str, name)("a", flags=flags)
    with pytest.raises(firepanda.errors.UnsupportedError, match="flag value"):
        mine.str.replace("a", "-", flags=1024, regex=True)
    with pytest.raises(firepanda.errors.UnsupportedError, match="flag value"):
        mine.str.extract("(a)", flags=1024)


def test_a_pattern_that_is_not_text_is_refused_before_the_flag_is_read(
    firepanda: ModuleType,
) -> None:
    """`re.compile` is the first thing to look at a pattern down every route
    upstream, and it is what raises for one that is not a string, so the type
    refusal comes before anything this library would have said about the
    flag."""
    mine = made(firepanda)
    with pytest.raises(TypeError):
        mine.str.contains(5, flags=re.MULTILINE)
    with pytest.raises(TypeError):
        mine.str.count(5, flags=re.MULTILINE)
