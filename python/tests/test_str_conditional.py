"""A conditional group, which asks whether another group took part.

`(a)?(?(1)b|c)` reads a `b` when the group before it matched and a `c` when it
did not, so the pattern picks its own arm out of what the path has already done.
That is a question about the path rather than about the position, which is what
a backreference is, and it is why this construct lands on the same engine a
backreference lands on rather than on the one an atomic group lands on.

Getting here needs a flag, the same door the atomic group came through. pandas
picks the engine by whether the pattern holds a lookaround or a backreference,
and a conditional holds neither, so a call with no flags on it goes to Arrow and
Arrow is RE2 and RE2 has never had the construct. Naming a flag is what moves
the call onto `re`, which has answered it since the beginning.

`case=False` is not a flag for this purpose, and it has its own row for the
reason it has one next door. pandas turns it into something Arrow can read
rather than into a move onto `re`, so a call that says only that still reaches
RE2 and still raises.

The rows compare against pandas rather than against a written down answer,
because the claim of the slice is agreement and not correctness in the abstract.

Document 100.
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

ROWS = ["ab", "ac", "b", "c", "aab", "", None]
"""Rows picked so that which arm was taken is what decides them.

The first two are the pair the whole construct is about: the same leading `a`
and a different letter after it, so one row takes the first arm and the other
has to give the group back and take the second. The next two hold the letter an
arm wants and no `a` in front of it, which is the group never taking part at
all. The fifth puts the pair one along, so a rule that only looked at the front
of the row would show. The empty row and the missing one are the two every text
file here carries.
"""

PATTERNS = [
    r"(a)?(?(1)b|c)",
    r"(a)?(?(1)b)",
    r"(a)?(?(1)|c)",
    r"(?P<n>a)?(?(n)b|c)",
    r"((a))?(?(2)b|c)",
    r"(?(1)a|b)(x)",
    r"(a*)(?(1)b|c)",
    r"(?:(a)|a)(?(1)b|c)",
]
"""Every shape of the construct this slice claims, in one list.

Two arms and one arm, an empty first arm, a name instead of a number, a nested
group number, a group the pattern has not opened yet, a group that takes part
without reading anything, and the two paths to the same place that the bitmap
had to be given up for.
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
    as `engine/string-predicate-null` and a conditional changes nothing about
    it.
    """
    return values[:-1]


@needs_pandas
def test_the_two_arms_are_chosen_by_a_group(firepanda: ModuleType) -> None:
    """The call that used to raise, and the row that says what the construct is
    for. `ab` takes the first arm because the group took part, and `ac` cannot:
    the group is given back so that it never took part, and what is left is the
    second arm matching the `c` on its own one character along."""
    mine, them = made(firepanda), theirs()
    ours = mine.str.contains(r"(a)?(?(1)b|c)", flags=re.IGNORECASE).tolist()
    assert without_the_missing(ours) == without_the_missing(
        mask_of(them.str.contains(r"(a)?(?(1)b|c)", flags=re.IGNORECASE))
    )
    assert without_the_missing(ours) == [True, True, False, True, True, False]


@needs_pandas
def test_every_shape_of_it_agrees(firepanda: ModuleType) -> None:
    """One row per entry in the list above, asked of both libraries over every
    text. This is the width of the claim: not that the construct works but that
    each of the eight ways of writing one answers what upstream answers."""
    mine, them = made(firepanda), theirs()
    for pattern in PATTERNS:
        assert without_the_missing(
            mine.str.contains(pattern, flags=re.IGNORECASE).tolist()
        ) == without_the_missing(mask_of(them.str.contains(pattern, flags=re.IGNORECASE))), pattern


@needs_pandas
def test_a_group_that_matched_nothing_has_still_taken_part(
    firepanda: ModuleType,
) -> None:
    """Which is the difference between a slot pair that was written and a slot
    pair that holds something. `(a*)` matches the empty string in front of a
    `b`, so the group took part, so the first arm is the one taken and the `b`
    is read. Against `ac` there is then no way to reach the second arm at
    all."""
    mine, them = made(firepanda), theirs()
    ours = mine.str.contains(r"(a*)(?(1)b|c)", flags=re.IGNORECASE).tolist()
    assert without_the_missing(ours) == without_the_missing(
        mask_of(them.str.contains(r"(a*)(?(1)b|c)", flags=re.IGNORECASE))
    )
    assert without_the_missing(ours) == [True, False, True, False, True, False]


@needs_pandas
def test_two_paths_to_the_same_place_can_answer_it_differently(
    firepanda: ModuleType,
) -> None:
    """The row the bitmap had to be given up for. `(?:(a)|a)(?(1)b|c)` has two
    ways to read an `a`, one that sets the group and one that does not, and both
    of them arrive at the test in the same place. Remembering the arrival would
    drop the second and the `c` would never be read, so `ac` would be a miss
    here and a hit upstream."""
    mine, them = made(firepanda), theirs()
    ours = mine.str.contains(r"(?:(a)|a)(?(1)b|c)", flags=re.IGNORECASE).tolist()
    assert without_the_missing(ours) == without_the_missing(
        mask_of(them.str.contains(r"(?:(a)|a)(?(1)b|c)", flags=re.IGNORECASE))
    )
    assert without_the_missing(ours) == [True, True, False, False, True, False]


@needs_pandas
def test_the_counting_and_replacing_scans_read_one(firepanda: ModuleType) -> None:
    """Both scans go through the same door the search does, so what is being
    checked here is that the rule for where to look next is unchanged by an arm
    that read nothing. A one armed conditional matches the empty string
    everywhere, which makes the scans the interesting half of this file."""
    mine, them = made(firepanda), theirs()
    for pattern in PATTERNS:
        assert mine.str.count(pattern, flags=re.IGNORECASE).tolist() == counts_of(
            them.str.count(pattern, flags=re.IGNORECASE)
        ), pattern
        assert mine.str.replace(pattern, "#", regex=True, flags=re.IGNORECASE).tolist() == texts_of(
            them.str.replace(pattern, "#", regex=True, flags=re.IGNORECASE)
        ), pattern


@needs_pandas
def test_the_anchored_one_takes_it_too(firepanda: ModuleType) -> None:
    """`fullmatch` is answered by writing anchors around the caller's pattern,
    and the bracket that goes with them must not be one the caller can count. A
    conditional names a group by number, so a capturing bracket written around
    the pattern would move every number in it by one and `(?(1)` would ask about
    the wrong group. `match` is not asked here for the reason document 86 has:
    pandas refuses a `match` with flags on it through this accessor."""
    mine, them = made(firepanda), theirs()
    for pattern in PATTERNS:
        assert without_the_missing(
            mine.str.fullmatch(pattern, flags=re.IGNORECASE).tolist()
        ) == without_the_missing(mask_of(them.str.fullmatch(pattern, flags=re.IGNORECASE))), pattern


@needs_pandas
def test_extract_reads_the_group_the_arms_asked_about(firepanda: ModuleType) -> None:
    """A test reads the slots and writes none, so the groups at the end of a
    match are the ones the arms left there, and a group given back so that the
    second arm could be taken is missing rather than empty. `extract` is the
    accessor name that asks."""
    mine, them = made(firepanda), theirs()
    for pattern in [r"(a)?(?(1)b|c)", r"(a*)(?(1)b|c)", r"(?:(a)|a)(?(1)b|c)"]:
        assert mine.str.extract(pattern, flags=re.IGNORECASE).iloc[:, 0].tolist() == texts_of(
            them.str.extract(pattern, flags=re.IGNORECASE)[0]
        ), pattern


@needs_pandas
def test_extract_needs_no_flag_to_reach_it(firepanda: ModuleType) -> None:
    """`extract` never goes to Arrow in the first place, on either library, so
    it is the one accessor name that answers a conditional with no flag on the
    call at all. pandas compiles the pattern with `re` and loops in Python, and
    this library compiles it for the engine that copies Python without asking
    the router anything."""
    mine, them = made(firepanda), theirs()
    ours = mine.str.extract(r"(a)?(?(1)b|c)").iloc[:, 0].tolist()
    assert ours == texts_of(them.str.extract(r"(a)?(?(1)b|c)")[0])
    assert ours == ["a", None, None, None, "a", None, None]


@needs_pandas
def test_without_a_flag_both_libraries_send_it_to_re2(firepanda: ModuleType) -> None:
    """Which is what makes the flag the door. pandas asks Arrow, Arrow is RE2,
    and RE2 has never had the construct, so the call raises. This library routes
    the same way and raises the same class, and the words differ because RE2 is
    reached here through a compiler of this library's own rather than through
    Arrow."""
    mine, them = made(firepanda), theirs()
    for pattern in PATTERNS:
        with pytest.raises(ValueError):
            them.str.contains(pattern)
        with pytest.raises(ValueError) as caught:
            mine.str.contains(pattern)
        assert "RE2 has no conditional group" in str(caught.value), pattern


@needs_pandas
def test_asking_for_no_case_is_not_the_same_as_naming_a_flag(
    firepanda: ModuleType,
) -> None:
    """pandas turns `case=False` into something Arrow can read rather than into
    a move onto `re`, so a call that says only that still reaches RE2 and still
    raises. It is the one place in this file where a call that looks like it
    passed a flag did not, and this library follows pandas rather than reading
    the argument as a route."""
    mine, them = made(firepanda), theirs()
    for pattern in [r"(a)?(?(1)b|c)", r"(a*)(?(1)b|c)"]:
        with pytest.raises(ValueError):
            them.str.contains(pattern, case=False)
        with pytest.raises(ValueError):
            mine.str.contains(pattern, case=False)


@needs_pandas
def test_a_group_number_nothing_opens_is_refused_by_both(
    firepanda: ModuleType,
) -> None:
    """`(?(2)a|b)` asks about a group the pattern does not have. Both libraries
    refuse it and the classes differ, which is upstream's doing rather than this
    slice's: `re.error` is a subclass of `Exception` and not of `ValueError`, so
    a pattern `re` cannot compile comes out of pandas as something a caller
    catching `ValueError` does not catch. This library raises a `ValueError` for
    every pattern its own parser turns down, and that is the older divergence
    rather than a new one. The row is here so that it is measured and not
    assumed."""
    mine, them = made(firepanda), theirs()
    with pytest.raises(re.error):
        them.str.contains(r"(?(2)a|b)", flags=re.IGNORECASE)
    assert not issubclass(re.error, ValueError)
    with pytest.raises(ValueError):
        mine.str.contains(r"(?(2)a|b)", flags=re.IGNORECASE)


@needs_pandas
def test_a_lookaround_beside_one_is_still_a_gap(firepanda: ModuleType) -> None:
    """For the reason a lookaround beside a backreference is. A lookaround is a
    search inside a search and the engine that keeps a path has one stack to run
    it on, so a program holding one goes to the engine that merges threads, and
    that engine cannot answer a test. A pattern holding both has nowhere to go.

    pandas answers it, so this is a gap as well, and it is the one thing this
    slice took away from what the flags path could do before it.
    """
    mine, them = made(firepanda), theirs()
    assert mask_of(them.str.contains(r"(?=a)(a)?(?(1)b|c)", flags=re.IGNORECASE))[0] is True
    with pytest.raises(NotImplementedError) as caught:
        mine.str.contains(r"(?=a)(a)?(?(1)b|c)", flags=re.IGNORECASE)
    assert "lookaround beside a conditional group" in str(caught.value)
    with pytest.raises(NotImplementedError) as second:
        mine.str.contains(r"(?<=a)(b)?(?(1)c|d)", flags=re.IGNORECASE)
    assert "lookaround beside a conditional group" in str(second.value)


@needs_pandas
def test_a_test_reads_characters_rather_than_bytes(firepanda: ModuleType) -> None:
    """A test is about the slots rather than about the text, so there is nothing
    here that could read a byte where a character was meant. The row is here so
    that the next person to change the arms has a two byte character in front of
    them."""
    rows = ["ßx", "ßy", "y", None]
    mine, them = made(firepanda, rows), theirs(rows)
    for pattern in [r"(ß)?(?(1)x|y)", r"(ß*)(?(1)x|y)"]:
        assert without_the_missing(
            mine.str.contains(pattern, flags=re.IGNORECASE).tolist()
        ) == without_the_missing(mask_of(them.str.contains(pattern, flags=re.IGNORECASE)))
        assert mine.str.replace(pattern, "#", regex=True, flags=re.IGNORECASE).tolist() == texts_of(
            them.str.replace(pattern, "#", regex=True, flags=re.IGNORECASE)
        )
