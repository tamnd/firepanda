"""An atomic group and a possessive quantifier, which are one construct twice.

`(?>a*)b` says take every `a` there is and never give one back, and `a*+b` says
the same thing in two characters fewer. Upstream compiles the second into the
first and so does this library, so every row below is written twice and the two
spellings have to answer the same.

Getting here needs a flag, and that is the whole of what makes the construct
reachable. pandas picks the engine by whether the pattern holds a lookaround or
a backreference, and an atomic group holds neither, so a call with no flags on
it goes to Arrow and Arrow is RE2 and RE2 has never had either construct. Naming
a flag is what moves the call onto `re`, which has answered both since Python
3.11. So the rows that answer all name a flag and the rows that raise do not,
and both halves of that are agreement with pandas rather than a choice this
library made.

`case=False` is not a flag for this purpose, which is worth its own row. pandas
turns it into something Arrow can read rather than into a move onto `re`, so a
call that says `case=False` and nothing else still reaches Arrow and still
raises, and this library follows it there.

The rows compare against pandas rather than against a written down answer,
because the claim of the slice is agreement and not correctness in the abstract.

Document 99.
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

ROWS = ["aaab", "aab", "ab", "b", "aa", "", None]
"""Rows picked so that giving a character back is what decides them.

The first three hold a run of `a` in front of a `b`, which is the shape the
construct was invented for, and they are three different lengths so that a rule
about the last one would show. The fourth holds the `b` and no run at all, which
is the row where the repeat matched nothing and had nothing to give back. The
fifth holds the run and no `b`, which is the row a cut turns from a match into
no match. The empty row and the missing one are the two every text file here
carries.
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
    as `engine/string-predicate-null` and an atomic group changes nothing about
    it.
    """
    return values[:-1]


PAIRS = [(r"(?>a*)b", r"a*+b"), (r"(?>a+)b", r"a++b"), (r"(?>a?)b", r"a?+b")]
"""The same three patterns written as a group and written as a quantifier."""


@needs_pandas
def test_a_group_keeps_the_first_way_it_matched(firepanda: ModuleType) -> None:
    """The call that used to raise. `(?>a*)b` matches every row with a `b` in it
    because the repeat had nothing to give back, and `(?>a*)a` matches nothing
    at all because the repeat took every `a` and the cut means it keeps them."""
    mine, them = made(firepanda), theirs()
    ours = mine.str.contains(r"(?>a*)b", flags=re.IGNORECASE).tolist()
    assert without_the_missing(ours) == without_the_missing(
        mask_of(them.str.contains(r"(?>a*)b", flags=re.IGNORECASE))
    )
    assert without_the_missing(ours) == [True, True, True, True, False, False]
    kept = mine.str.contains(r"(?>a*)a", flags=re.IGNORECASE).tolist()
    assert without_the_missing(kept) == without_the_missing(
        mask_of(them.str.contains(r"(?>a*)a", flags=re.IGNORECASE))
    )
    assert without_the_missing(kept) == [False] * 6


@needs_pandas
def test_the_two_spellings_answer_the_same(firepanda: ModuleType) -> None:
    """`a*+` is `(?>a*)` and is compiled as that, so there is one mechanism here
    rather than two and this is the row that says so. Both spellings are asked
    of pandas as well, since upstream reading them as one thing is where the
    decision came from."""
    mine, them = made(firepanda), theirs()
    for group, quantifier in PAIRS:
        ours = mine.str.contains(group, flags=re.IGNORECASE).tolist()
        assert ours == mine.str.contains(quantifier, flags=re.IGNORECASE).tolist()
        assert without_the_missing(ours) == without_the_missing(
            mask_of(them.str.contains(group, flags=re.IGNORECASE))
        )
        assert without_the_missing(ours) == without_the_missing(
            mask_of(them.str.contains(quantifier, flags=re.IGNORECASE))
        )


@needs_pandas
def test_a_choice_made_outside_the_group_survives_it(firepanda: ModuleType) -> None:
    """The cut throws away the choices the group made and nothing older than
    that, so a question mark written outside the group is still a choice the
    pattern can come back to. `(?>a)?ab` matches `ab` and `(?>a?)ab` does
    not."""
    mine, them = made(firepanda), theirs()
    for pattern in [r"(?>a)?ab", r"(?>a?)ab", r"(?>a|ab)b", r"(?>ab|a)b"]:
        assert without_the_missing(
            mine.str.contains(pattern, flags=re.IGNORECASE).tolist()
        ) == without_the_missing(mask_of(them.str.contains(pattern, flags=re.IGNORECASE)))


@needs_pandas
def test_one_inside_another_cuts_only_its_own_choices(firepanda: ModuleType) -> None:
    """Nesting is where a cut that threw away too much would show, and the third
    pattern is the one that catches it: the inner group takes `ab` and never
    tries `a`, so the outer `b` has nothing left to read."""
    mine, them = made(firepanda), theirs()
    for pattern in [r"(?>a(?>a)b)", r"(?>(?>a)|b)b", r"(?>(?>ab|a)b)", r"(?>a*)*b"]:
        assert without_the_missing(
            mine.str.contains(pattern, flags=re.IGNORECASE).tolist()
        ) == without_the_missing(mask_of(them.str.contains(pattern, flags=re.IGNORECASE)))


@needs_pandas
def test_the_counting_and_replacing_scans_read_one(firepanda: ModuleType) -> None:
    """Both scans go through the same door the search does, so what is being
    checked here is that the rule for where to look next is unchanged by a match
    the cut made shorter than the pattern would otherwise have taken."""
    mine, them = made(firepanda), theirs()
    for group, quantifier in PAIRS:
        for pattern in [group, quantifier]:
            assert mine.str.count(pattern, flags=re.IGNORECASE).tolist() == counts_of(
                them.str.count(pattern, flags=re.IGNORECASE)
            )
            assert mine.str.replace(
                pattern, "#", regex=True, flags=re.IGNORECASE
            ).tolist() == texts_of(them.str.replace(pattern, "#", regex=True, flags=re.IGNORECASE))
    for pattern in [r"(?>a*)", r"a*+"]:
        assert mine.str.count(pattern, flags=re.IGNORECASE).tolist() == counts_of(
            them.str.count(pattern, flags=re.IGNORECASE)
        )
        assert mine.str.replace(pattern, "#", regex=True, flags=re.IGNORECASE).tolist() == texts_of(
            them.str.replace(pattern, "#", regex=True, flags=re.IGNORECASE)
        )


@needs_pandas
def test_the_anchored_one_takes_it_too(firepanda: ModuleType) -> None:
    """`fullmatch` is answered by writing anchors around the caller's pattern and
    a bracket around the middle, and the bracket has to sit outside the cut
    rather than inside it. `match` is not asked here for the reason document 86
    has: pandas refuses a `match` with flags on it through this accessor, so
    there is nothing to agree with."""
    mine, them = made(firepanda), theirs()
    for pattern in [r"(?>a*)b", r"a*+b", r"(?>a*)ab"]:
        assert without_the_missing(
            mine.str.fullmatch(pattern, flags=re.IGNORECASE).tolist()
        ) == without_the_missing(mask_of(them.str.fullmatch(pattern, flags=re.IGNORECASE)))


@needs_pandas
def test_extract_reads_the_groups_the_cut_kept(firepanda: ModuleType) -> None:
    """A cut throws away the choices and keeps the saves, so a group written
    inside one still says where it matched. `extract` is the accessor name that
    asks."""
    mine, them = made(firepanda), theirs()
    for pattern in [r"(?>(a+))b", r"(a+)*+b"]:
        assert mine.str.extract(pattern, flags=re.IGNORECASE).iloc[:, 0].tolist() == texts_of(
            them.str.extract(pattern, flags=re.IGNORECASE)[0]
        )


@needs_pandas
def test_without_a_flag_both_libraries_send_it_to_re2(firepanda: ModuleType) -> None:
    """Which is what makes the flag the door. pandas asks Arrow, Arrow is RE2,
    and RE2 has neither construct, so the call raises. This library routes the
    same way and raises the same class, and the words differ because RE2 is
    reached here through a compiler of this library's own rather than through
    Arrow."""
    mine, them = made(firepanda), theirs()
    for group, quantifier in PAIRS:
        for pattern in [group, quantifier]:
            with pytest.raises(ValueError):
                them.str.contains(pattern)
            with pytest.raises(ValueError) as caught:
                mine.str.contains(pattern)
            assert "RE2 has no" in str(caught.value), pattern


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
    for pattern in [r"(?>a*)b", r"a*+b"]:
        with pytest.raises(ValueError):
            them.str.contains(pattern, case=False)
        with pytest.raises(ValueError):
            mine.str.contains(pattern, case=False)


@needs_pandas
def test_a_lookaround_beside_one_is_still_a_gap(firepanda: ModuleType) -> None:
    """For the same reason a lookaround beside a backreference is. A lookaround
    is a search inside a search and the engine that keeps a path has one stack to
    run it on, so a program holding one goes to the engine that merges threads,
    and that engine cannot obey a cut. A pattern holding both has nowhere to go.

    pandas answers it, so this is a gap as well, and it is the one thing this
    slice took away from what the flags path could do before it.
    """
    mine, them = made(firepanda), theirs()
    assert mask_of(them.str.contains(r"(?=a)(?>a*)b", flags=re.IGNORECASE))[0] is True
    with pytest.raises(NotImplementedError) as caught:
        mine.str.contains(r"(?=a)(?>a*)b", flags=re.IGNORECASE)
    assert "lookaround beside an atomic group" in str(caught.value)
    with pytest.raises(NotImplementedError) as second:
        mine.str.contains(r"(?<=a)a*+b", flags=re.IGNORECASE)
    assert "lookaround beside an atomic group" in str(second.value)


@needs_pandas
def test_a_cut_reads_characters_rather_than_bytes(firepanda: ModuleType) -> None:
    """A cut is about the stack rather than about the text, so there is nothing
    here that could read a byte where a character was meant. The row is here so
    that the next person to change the stack has a two byte character in front of
    them."""
    rows = ["ßßx", "ßß", "x", None]
    mine, them = made(firepanda, rows), theirs(rows)
    for pattern in [r"(?>ß*)x", r"ß*+x"]:
        assert without_the_missing(
            mine.str.contains(pattern, flags=re.IGNORECASE).tolist()
        ) == without_the_missing(mask_of(them.str.contains(pattern, flags=re.IGNORECASE)))
        assert mine.str.replace(pattern, "#", regex=True, flags=re.IGNORECASE).tolist() == texts_of(
            them.str.replace(pattern, "#", regex=True, flags=re.IGNORECASE)
        )
