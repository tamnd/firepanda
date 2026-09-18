"""A lookbehind, which is the second construct RE2 has not got that this has.

The file beside this one landed the lookahead and said what the lookbehind would
need that the lookahead did not, which is a width. A lookahead starts its body
where the pattern has got to and so can be answered without the compiler knowing
anything about the body. A lookbehind has to start the body far enough back that
it ends where the pattern has got to, so the compiler has to work out how many
characters the body always reads and has to refuse the bodies that do not always
read the same number.

Python refuses exactly those, with `look-behind requires fixed-width pattern`,
so the refusal is agreement with upstream rather than a shortfall here, and the
last row in this file is the one that says so: both libraries raise for the same
patterns and both raise a `ValueError` underneath.

Nothing here passes a flag, for the same reason nothing does next door. The
route is decided by what the pattern holds, so these calls reach Python's engine
with an empty `flags` and would have reached Arrow without the brackets.

The rows compare against pandas rather than against a written down answer,
because the claim of the slice is agreement and not correctness in the abstract.

Document 94.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

ROWS = ["ab", "cb", "b", "ba", "abab", "", None]
"""Rows picked so that a lookbehind decides most of them.

The first two are the pair a lookbehind was invented for, one where the letter
behind is the one asked about and one where it is not. The single letter is the
row where there is nothing behind at all, which is the case a negative
lookbehind gets right and which a plain pattern cannot say anything about. The
reversed pair puts the letter after rather than before. The repeated pair is the
row that matches more than once. The empty row and the missing one are the two
every text file here carries.
"""


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


def counts_of(them: Any) -> list[Any]:
    """pandas' counts as a list, with every flavour of missing read as None."""
    return [None if one is None or one != one else int(one) for one in them.tolist()]


def texts_of(them: Any) -> list[Any]:
    """pandas' text answers as a list, with every flavour of missing read as None."""
    return [None if one is None or one != one else one for one in them.tolist()]


def without_the_missing(values: list[Any]) -> list[Any]:
    """Drops the last row, which is the null every mask here disagrees about.

    pandas hands back a bool column and a missing row becomes False in it, and
    this library keeps the row missing. That is the divergence the board records
    as `engine/string-predicate-null` and a lookbehind changes nothing about it.
    """
    return values[:-1]


@needs_pandas
def test_the_positive_form_agrees_with_pandas_row_by_row(
    firepanda: ModuleType,
) -> None:
    """The call that used to raise. `(?<=a)b` is a `b` with an `a` in front of
    it and the `a` is not consumed, which is visible in the replacement rather
    than in the mask and is why the replacement is a row of its own below."""
    mine, them = made(firepanda), theirs()
    ours = mine.str.contains("(?<=a)b").tolist()
    assert without_the_missing(ours) == without_the_missing(mask_of(them.str.contains("(?<=a)b")))
    assert without_the_missing(ours) == [True, False, False, False, True, False]
    assert mine.str.count("(?<=a)b").tolist() == counts_of(them.str.count("(?<=a)b"))


@needs_pandas
def test_the_negative_form_agrees_as_well(firepanda: ModuleType) -> None:
    """`(?<!a)b` holds on the row where there is nothing in front of the `b` at
    all, which is the case that separates a negative lookbehind from a plain
    `[^a]b` and is why the single letter row is in the list."""
    mine, them = made(firepanda), theirs()
    ours = mine.str.contains("(?<!a)b").tolist()
    assert without_the_missing(ours) == without_the_missing(mask_of(them.str.contains("(?<!a)b")))
    assert without_the_missing(ours) == [False, True, True, True, False, False]


@needs_pandas
def test_the_width_of_one_is_nothing_so_the_replacement_says_so(
    firepanda: ModuleType,
) -> None:
    """A lookbehind is a question about a position rather than a piece of text,
    so what it asked about is still there afterwards. `(?<=a)b` replaced by a
    hash turns `ab` into `a#` and not into `#`."""
    mine, them = made(firepanda), theirs()
    assert mine.str.replace("(?<=a)b", "#", regex=True).tolist() == texts_of(
        them.str.replace("(?<=a)b", "#", regex=True)
    )
    assert mine.str.replace("(?<=a)b", "#", regex=True).tolist()[0] == "a#"


@needs_pandas
def test_a_body_of_no_width_is_a_lookbehind_too(firepanda: ModuleType) -> None:
    """`(?<=\\b)` and `(?<=^)` are bodies that read nothing, so the position the
    body is started from is the position the pattern is already at, and they
    come out right by the ordinary rule rather than by a case of their own."""
    mine, them = made(firepanda), theirs()
    for pattern in ["(?<=\\b)b", "(?<=^)b", "(?<=(?=b)b)a"]:
        assert without_the_missing(mine.str.contains(pattern).tolist()) == without_the_missing(
            mask_of(them.str.contains(pattern))
        )


@needs_pandas
def test_the_anchored_pair_take_it_too(firepanda: ModuleType) -> None:
    """`fullmatch` is answered by writing anchors around the caller's pattern,
    and a lookbehind inside those anchors is the same instruction in a different
    place. `match` is not asked here for the reason document 86 has: pandas
    refuses a `match` on Python's engine through this accessor, so there is
    nothing to agree with."""
    mine, them = made(firepanda), theirs()
    assert without_the_missing(mine.str.fullmatch("a(?<=a)b").tolist()) == without_the_missing(
        mask_of(them.str.fullmatch("a(?<=a)b"))
    )


@needs_pandas
def test_extract_takes_a_group_beside_one_but_not_inside_one(
    firepanda: ModuleType,
) -> None:
    """A group inside a lookbehind keeps what it matched upstream, and the
    second machine this engine runs the body with carries no slots, so that
    shape is refused rather than answered wrongly. A group beside one is
    ordinary. The same pair of rules the lookahead has."""
    mine, them = made(firepanda), theirs()
    assert mine.str.extract("(?<=a)(b)").iloc[:, 0].tolist() == texts_of(
        them.str.extract("(?<=a)(b)")[0]
    )
    assert them.str.extract("(?<=(a))b")[0].tolist()[0] == "a"
    with pytest.raises(NotImplementedError):
        mine.str.extract("(?<=(a))b")


@needs_pandas
def test_a_body_without_one_width_is_refused_by_both_libraries(
    firepanda: ModuleType,
) -> None:
    """The refusal that is agreement rather than a gap, which is why it is a
    `ValueError` here rather than the `NotImplementedError` every construct
    this engine has not got raises.

    The classes are the divergence document 86 registered and not a difference
    this slice introduced. `re` raises a `PatternError`, whose only base is
    `Exception`, and this library raises a `ValueError` for every pattern either
    engine refuses. What matters for a caller is the other half: a pattern
    refused here is a pattern refused there, and the four below are the four
    shapes that get refused.
    """
    import re

    mine, them = made(firepanda), theirs()
    for pattern in ["(?<=a*)b", "(?<=a?)b", "(?<=a|bc)d", "(?<=a{2,3})b"]:
        with pytest.raises(re.error):
            them.str.contains(pattern)
        with pytest.raises(ValueError):
            mine.str.contains(pattern)


@needs_pandas
def test_a_fixed_body_is_read_by_both_the_same_way(firepanda: ModuleType) -> None:
    """The other side of that, which is the shapes Python does call fixed width.
    An alternation whose arms agree, a repeat whose two bounds are the same
    number, and a class, which reads one character however many members it
    has."""
    mine, them = made(firepanda), theirs()
    for pattern in ["(?<=a|c)b", "(?<=ab|cb)a", "(?<=a{1})b", "(?<=[ac])b"]:
        assert without_the_missing(mine.str.contains(pattern).tolist()) == without_the_missing(
            mask_of(them.str.contains(pattern))
        )


@needs_pandas
def test_the_brackets_are_what_moved_the_call_off_arrow(
    firepanda: ModuleType,
) -> None:
    """No flag is passed anywhere in this file and every call above reached
    Python's engine anyway, because the route is decided by what the pattern
    holds. The same pattern without the brackets is a pattern Arrow answers, and
    the two give the same column here for a reason that has nothing to do with
    either engine: they are asking different questions that agree on these
    rows."""
    mine, them = made(firepanda), theirs()
    assert mine.str.contains("ab").tolist() == mine.str.contains("a(?<=a)b").tolist()
    assert without_the_missing(mine.str.contains("a(?<=a)b").tolist()) == (
        without_the_missing(mask_of(them.str.contains("a(?<=a)b")))
    )
