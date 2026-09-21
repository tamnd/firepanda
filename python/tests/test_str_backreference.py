"""A backreference, which is the third construct RE2 has not got that this has.

The two files beside this one landed both halves of the lookaround. This one
lands the construct those two documents both pointed at, and it is the one that
decided which engine answers rather than only what the answer is. A lookaround
is a second search from the position the pattern has got to, which is still a
question about a program and a position. A backreference asks what the path that
arrived matched, and the two engines this library reaches for first are both
built on that question never being asked, so the third one answers it alone.

None of that is visible from here, which is the point of the file. What is
visible is that the patterns that used to raise now answer, and that they answer
what pandas answers.

Nothing here passes a flag except the one row that is about a flag. The route is
decided by what the pattern holds, so these calls reach Python's engine with an
empty `flags` and would have reached Arrow without the reference in them.

The rows compare against pandas rather than against a written down answer,
because the claim of the slice is agreement and not correctness in the abstract.

Documents 95 and 97.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

ROWS = ["aa", "ab", "abb", "aabb", "abab", "aA", "", None]
"""Rows picked so that a backreference decides most of them.

The first two are the pair the construct was invented for, one where the letter
repeats and one where it does not. The third has the repeat somewhere other than
the front, so a scan has to walk to it. The fourth has two of them. The fifth
repeats a pair rather than a letter, which is the row where the width the
reference reads is not one. The sixth is the same letter in two cases, which is
the row the ignore case flag decides. The empty row and the missing one are the
two every text file here carries.
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
    as `engine/string-predicate-null` and a backreference changes nothing about
    it.
    """
    return values[:-1]


@needs_pandas
def test_a_reference_agrees_with_pandas_row_by_row(firepanda: ModuleType) -> None:
    """The call that used to raise. A character followed by the same character
    again, which is the shape the construct exists for."""
    mine, them = made(firepanda), theirs()
    ours = mine.str.contains(r"(\w)\1").tolist()
    assert without_the_missing(ours) == without_the_missing(mask_of(them.str.contains(r"(\w)\1")))
    assert without_the_missing(ours) == [True, False, True, True, False, False, False]
    assert mine.str.count(r"(\w)\1").tolist() == counts_of(them.str.count(r"(\w)\1"))


@needs_pandas
def test_a_reference_reads_the_width_the_group_took(firepanda: ModuleType) -> None:
    """Which is not a width the pattern has got written into it. `(\\w\\w)\\1`
    reads two characters back because the group took two, and the row that
    repeats a pair is the only one it holds on."""
    mine, them = made(firepanda), theirs()
    ours = mine.str.contains(r"(\w\w)\1").tolist()
    assert without_the_missing(ours) == without_the_missing(mask_of(them.str.contains(r"(\w\w)\1")))
    assert without_the_missing(ours) == [False, False, False, False, True, False, False]


@needs_pandas
def test_a_named_reference_is_the_same_instruction(firepanda: ModuleType) -> None:
    """`(?P=x)` names the group the number would have counted to, and nothing
    downstream of the parser can tell the two spellings apart."""
    mine, them = made(firepanda), theirs()
    for pattern in [r"(?P<x>\w)(?P=x)", r"(\w)\1"]:
        assert without_the_missing(mine.str.contains(pattern).tolist()) == without_the_missing(
            mask_of(them.str.contains(pattern))
        )


@needs_pandas
def test_a_group_that_never_took_part_fails_the_reference(
    firepanda: ModuleType,
) -> None:
    """And a group that took part and matched nothing does not, which is the
    whole of the difference between `(a)?\\1b` and `(a?)\\1b`. Upstream fails a
    reference to a group whose two ends were never written rather than reading
    it as empty, and that is a rule worth a row because the two patterns look
    alike and answer differently."""
    mine, them = made(firepanda), theirs()
    for pattern in [r"(a)?\1b", r"(a?)\1b"]:
        assert without_the_missing(mine.str.contains(pattern).tolist()) == without_the_missing(
            mask_of(them.str.contains(pattern))
        )


@needs_pandas
def test_the_replacing_scan_reads_one_the_same_way(firepanda: ModuleType) -> None:
    """The scan is the engine's own and nothing in it knows what a
    backreference is, so what is being checked here is that the rule for where
    to look next is unchanged by a match whose width came out of a group."""
    mine, them = made(firepanda), theirs()
    for pattern in [r"(\w)\1", r"(\w\w)\1", r"(a*)\1"]:
        assert mine.str.replace(pattern, "#", regex=True).tolist() == texts_of(
            them.str.replace(pattern, "#", regex=True)
        )


@needs_pandas
def test_a_reference_can_be_written_into_the_replacement_too(
    firepanda: ModuleType,
) -> None:
    """The group the pattern read back is the same group the replacement can
    write out, and the two are different mechanisms that happen to share a
    number."""
    mine, them = made(firepanda), theirs()
    assert mine.str.replace(r"(\w)\1", r"<\1>", regex=True).tolist() == texts_of(
        them.str.replace(r"(\w)\1", r"<\1>", regex=True)
    )


@needs_pandas
def test_the_anchored_pair_take_it_too(firepanda: ModuleType) -> None:
    """`fullmatch` is answered by writing anchors around the caller's pattern
    and a bracket around the middle, and that bracket has to be a non capturing
    one or every group the caller wrote is numbered one higher and the reference
    follows the numbering. `match` is not asked here for the reason document 86
    has: pandas refuses a `match` on Python's engine through this accessor, so
    there is nothing to agree with."""
    mine, them = made(firepanda), theirs()
    for pattern in [r"(\w)\1", r"(\w)\1\w*"]:
        assert without_the_missing(mine.str.fullmatch(pattern).tolist()) == without_the_missing(
            mask_of(them.str.fullmatch(pattern))
        )


@needs_pandas
def test_extract_takes_a_reference_beside_the_group_it_reads(
    firepanda: ModuleType,
) -> None:
    """`extract` hands back what the groups matched, and a group that is read
    back later still matched what it matched."""
    mine, them = made(firepanda), theirs()
    assert mine.str.extract(r"(\w)\1").iloc[:, 0].tolist() == texts_of(
        them.str.extract(r"(\w)\1")[0]
    )


@needs_pandas
def test_the_ignore_case_flag_reads_the_two_characters_lowered(
    firepanda: ModuleType,
) -> None:
    """Upstream compares the two characters here by simple lowercase where it
    compares a literal by its whole fold orbit, and the two disagree on real
    text. Both tables are carried now, so both questions are answered, and the
    row that settles it is the sixth: the same letter in two cases.

    The two rows after it are the disagreement itself. A column holding the long
    s answers True to `(?i)ss` and False to `(?i)(s)\\1`, and pandas answers it
    the same two ways, because upstream is where the two rules came from.
    Document 97.
    """
    mine, them = made(firepanda), theirs()
    assert mask_of(them.str.contains(r"(\w)\1", case=False))[5] is True
    assert mask_of(mine.str.contains(r"(\w)\1", case=False))[5] is True
    assert mask_of(mine.str.contains(r"(\w)\1", case=False))[1] is False
    odd = ["s\u017f"]  # a long s, which folds onto s and lowers to itself
    assert mask_of(theirs(odd).str.contains(r"ss", case=False))[0] is True
    assert mask_of(made(firepanda, odd).str.contains(r"ss", case=False))[0] is True
    assert mask_of(theirs(odd).str.contains(r"(s)\1", case=False))[0] is False
    assert mask_of(made(firepanda, odd).str.contains(r"(s)\1", case=False))[0] is False


@needs_pandas
def test_a_lookaround_beside_one_is_still_a_gap(firepanda: ModuleType) -> None:
    """The two constructs live on different engines here. A lookaround needs a
    second search, which the engine that merges threads runs, and a
    backreference is the one thing that engine cannot be handed. A pattern
    holding both has nowhere to go, so it is refused rather than answered by
    whichever of the two was met first.

    pandas answers it, so this is a gap as well, and it is the one thing this
    slice took away from what the two before it could do.
    """
    mine, them = made(firepanda), theirs()
    assert mask_of(them.str.contains(r"(?=\w)(\w)\1"))[0] is True
    with pytest.raises(NotImplementedError) as caught:
        mine.str.contains(r"(?=\w)(\w)\1")
    assert "lookaround beside a backreference" in str(caught.value)


@needs_pandas
def test_a_reference_to_a_group_that_is_not_there_is_refused_by_both(
    firepanda: ModuleType,
) -> None:
    """Both libraries refuse all four, and both call it a bad pattern.

    pandas decides which engine answers by whether `re` will compile the
    pattern, and none of these compiles, so all four are handed to Arrow and
    come back with Arrow's words about an escape sequence or a perl operator
    rather than with `re`'s words about a group reference that is not there.

    This library refuses them in its parser too, and it used to report a
    pattern its parser could not read as a gap, because the parser reads
    Python's grammar and RE2's was not written down anywhere. It is now, in
    `firepanda/kernel/regex/re2.mojo`, so the second question gets asked: RE2
    will not read these either, and a pattern neither grammar takes is a broken
    pattern rather than a missing feature. The class is a `ValueError` on both
    sides, which is the whole point of reading RE2's grammar, and the reason
    that comes back is RE2's rather than `re`'s because RE2 is the engine that
    would have had to run it.
    """
    mine, them = made(firepanda), theirs()
    for pattern in [r"(a)\2", r"\1", r"(?P=nope)", r"(?P<x>a)(?P=y)"]:
        with pytest.raises(ValueError):
            them.str.contains(pattern)
        with pytest.raises(ValueError) as caught:
            mine.str.contains(pattern)
        assert pattern in str(caught.value), pattern


@needs_pandas
def test_a_reference_is_read_in_characters_rather_than_bytes(
    firepanda: ModuleType,
) -> None:
    """The two ends of a group are positions in characters, so a row with a two
    byte character in it is read back the way Python reads it rather than the
    way a byte count would."""
    rows = ["ßß", "ßs", "sß", "ss", None]
    mine, them = made(firepanda, rows), theirs(rows)
    assert without_the_missing(mine.str.contains(r"(.)\1").tolist()) == without_the_missing(
        mask_of(them.str.contains(r"(.)\1"))
    )
    assert mine.str.replace(r"(.)\1", "#", regex=True).tolist() == texts_of(
        them.str.replace(r"(.)\1", "#", regex=True)
    )
