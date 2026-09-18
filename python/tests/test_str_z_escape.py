"""`\\z`, which is a spelling Python was late to and RE2 always had.

RE2 spells the end of the string `\\z`. Python spells the same position `\\Z`
and read `\\z` as a `bad escape \\z` until 3.14, which added it. So nothing
became sayable in 3.14 that was not sayable before and a spelling stopped being
an error, and on one pattern a supported interpreter raises and the next
answers. `pixi.toml` says this project supports 3.12 and up, so both answers are
live and neither of them is the answer.

That is the second rule in one CPython release this library has to know a
version for, after the `\\B` one in document 90, and the two together are why
the compiler carries a version number rather than a bit per rule. Document 91.

Nothing below spells whether the escape is legal. It asks the running `re`
once at import, which is the only honest way to write a file whose answers
depend on which interpreter opened it, and it is what this library does at its
own door.

The route matters here more than in most of these files, because the two
engines disagree about the pattern rather than about the row. A call with no
flags beside it goes to Arrow, which has always had the spelling, so no version
of Python is involved and every row is answered. A call with a flag goes to
Python's engine and the version decides. `extract` is the one method upstream
never routes, so it asks the question with no flag in sight.

`re.MULTILINE` is the flag used to move a call, since it says nothing about the
end of a string and nothing about an escape, so it moves the route and nothing
else.

One upstream defect keeps a method out of this file entirely. `Series.str.match(
pat, flags=anything)` raises `ValueError: Cannot pass flags that do not match
pat.flags` on pandas 3.0.6, and this library reproduces the refusal because a
caller catching it today is catching something real, so there is no way to reach
`match` on Python's engine through this accessor at all. The rows about `match`
are in the Mojo tests, which ask the compiler rather than the accessor.

The other four methods are all here. `contains` carries the full row by row
comparison and the rest carry one line each, since the question is about the
pattern and not about the row and reading it four times over ten rows would say
the same thing four times.
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

ROWS = ["a", "ab", "ba", "a\n", "", None]
"""Rows picked so the anchor is doing work on most of them.

The single letter ends where it begins, the pair ends on the wrong letter, the
reversed pair ends on the right one and is the row that tells an end of string
anchor from a whole row match, the letter with a newline after it is where
Python's `\\Z` parts company with Perl's since Python's is the absolute end, and
the empty row and the missing one are the two rows every text file here carries.
"""

ZED_IS_AN_ESCAPE = True
"""Whether the running interpreter reads `\\z` at all.

False up to 3.13 and True from 3.14. Asked rather than spelled, and asked once
at import rather than in each row, so a reader who wants to know which
interpreter a run was made on has one place to look.
"""

try:
    re.compile("\\z")
except re.error:
    ZED_IS_AN_ESCAPE = False


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
    as `engine/string-predicate-null` and an end of string anchor changes
    nothing about it.
    """
    return values[:-1]


@needs_pandas
def test_a_flagged_call_asks_which_interpreter_it_is_beside(
    firepanda: ModuleType,
) -> None:
    """The whole rule, on the four methods a flag can move.

    Either both raise or both answer, and which of those happens is not written
    down here. It is whatever the interpreter running the file says, which is
    the same question this library asks at its door.
    """
    mine, them = made(firepanda), theirs()
    if not ZED_IS_AN_ESCAPE:
        with pytest.raises(re.error):
            them.str.contains("a\\z", flags=re.MULTILINE)
        with pytest.raises(ValueError):
            mine.str.contains("a\\z", flags=re.MULTILINE)
        with pytest.raises(ValueError):
            mine.str.count("a\\z", flags=re.MULTILINE)
        with pytest.raises(ValueError):
            mine.str.fullmatch("a\\z", flags=re.MULTILINE)
        with pytest.raises(ValueError):
            mine.str.replace("a\\z", "#", regex=True, flags=re.MULTILINE)
        return
    ours = mine.str.contains("a\\z", flags=re.MULTILINE).tolist()
    assert without_the_missing(ours) == without_the_missing(
        mask_of(them.str.contains("a\\z", flags=re.MULTILINE))
    )
    assert without_the_missing(ours) == [True, False, True, False, False]


@needs_pandas
def test_the_spelling_python_always_had_is_answered_everywhere(
    firepanda: ModuleType,
) -> None:
    """`\\Z` is the same position and no version of Python has ever refused it,
    which is what makes the change a spelling rather than a meaning. Every row
    here is the row the test above would have if the escape were legal."""
    mine, them = made(firepanda), theirs()
    ours = mine.str.contains("a\\Z", flags=re.MULTILINE).tolist()
    assert without_the_missing(ours) == without_the_missing(
        mask_of(them.str.contains("a\\Z", flags=re.MULTILINE))
    )
    assert without_the_missing(ours) == [True, False, True, False, False]
    assert mine.str.count("a\\Z", flags=re.MULTILINE).tolist() == counts_of(
        them.str.count("a\\Z", flags=re.MULTILINE)
    )
    assert mine.str.replace("a\\Z", "#", regex=True, flags=re.MULTILINE).tolist() == texts_of(
        them.str.replace("a\\Z", "#", regex=True, flags=re.MULTILINE)
    )


@needs_pandas
def test_an_unflagged_call_never_asks_the_question(firepanda: ModuleType) -> None:
    """With nothing beside the pattern the call goes to Arrow, which has always
    had the spelling, so an interpreter that would refuse it never sees it. That
    is upstream's answer as well, and it is the reason the same pattern raises
    or answers on one interpreter depending on a keyword that has nothing to do
    with escapes."""
    mine, them = made(firepanda), theirs()
    ours = mine.str.contains("a\\z").tolist()
    assert without_the_missing(ours) == without_the_missing(mask_of(them.str.contains("a\\z")))
    assert without_the_missing(ours) == [True, False, True, False, False]
    assert mine.str.count("a\\z").tolist() == counts_of(them.str.count("a\\z"))
    assert mine.str.replace("a\\z", "#", regex=True).tolist() == texts_of(
        them.str.replace("a\\z", "#", regex=True)
    )
    assert without_the_missing(mine.str.fullmatch("a\\z").tolist()) == [
        True,
        False,
        False,
        False,
        False,
    ]


@needs_pandas
def test_extract_asks_it_with_no_flag_in_sight(firepanda: ModuleType) -> None:
    """`extract` is the one method upstream never routes. It is answered in
    Python whether or not a flag was passed, so the version rule reaches it
    through the front door and a caller who passed nothing still gets the
    refusal."""
    mine, them = made(firepanda), theirs()
    if not ZED_IS_AN_ESCAPE:
        with pytest.raises(re.error):
            them.str.extract("(a)\\z")
        with pytest.raises(ValueError):
            mine.str.extract("(a)\\z")
    else:
        assert texts_of(mine.str.extract("(a)\\z").iloc[:, 0]) == texts_of(
            them.str.extract("(a)\\z")[0]
        )
    assert texts_of(mine.str.extract("(a)\\Z").iloc[:, 0]) == texts_of(
        them.str.extract("(a)\\Z")[0]
    )


@needs_pandas
def test_inside_a_class_it_is_an_error_in_every_version(firepanda: ModuleType) -> None:
    """3.14 added an anchor and a character class holds characters, so `[\\z]`
    is a bad escape on 3.14 too. RE2 refuses it in a class as well, which is why
    the unflagged route is an error here rather than an answer, and no version
    number moves either of them.

    The unflagged route is the one place a class where every side is an error
    still disagrees about which error. pandas hands the pattern to Arrow and an
    `ArrowInvalid` comes back out of the accessor, this library reads the
    pattern with Python's grammar before either engine sees it and calls a
    pattern that grammar cannot read a gap, so a caller gets a
    `NotImplementedError` where upstream gives something that is not an
    `re.error` either. Document 78 has that shape written down and this slice
    does not move it, which is why the row asserts what happens rather than what
    ought to."""
    mine, them = made(firepanda), theirs()
    for pattern in ("[\\z]", "[a\\z]"):
        with pytest.raises(re.error):
            them.str.contains(pattern, flags=re.MULTILINE)
        with pytest.raises(ValueError):
            mine.str.contains(pattern, flags=re.MULTILINE)
        with pytest.raises(NotImplementedError):
            mine.str.contains(pattern)


@needs_pandas
def test_the_anchor_this_library_writes_is_not_the_callers(
    firepanda: ModuleType,
) -> None:
    """The row the slice turns on, and the reason it is a slice rather than a
    line.

    `fullmatch` on Python's engine is answered here by writing anchors around
    the caller's pattern. Those anchors used to be spelled RE2's way, so every
    anchored call carried a `\\z` nobody wrote, and a refusal driven off the
    pattern being compiled would have refused all of them on an older
    interpreter. The closing anchor is `\\Z` now, which is the same position and
    is legal in every version.

    Only `fullmatch` can be asked from here. `match` is the other method that
    gets anchors written round it, and no caller can reach that code through
    this accessor, because upstream refuses `str.match(pat, flags=anything)`
    outright and this library reproduces the refusal. So the `match` half of
    this row lives in the Mojo tests, which can ask the compiler directly.
    """
    mine, them = made(firepanda), theirs()
    ours = mine.str.fullmatch("a", flags=re.MULTILINE).tolist()
    assert without_the_missing(ours) == without_the_missing(
        mask_of(them.str.fullmatch("a", flags=re.MULTILINE))
    )
    assert without_the_missing(ours) == [True, False, False, False, False]
    assert without_the_missing(
        mine.str.fullmatch("a(b)?", flags=re.MULTILINE).tolist()
    ) == without_the_missing(mask_of(them.str.fullmatch("a(b)?", flags=re.MULTILINE)))


@needs_pandas
def test_the_two_spellings_are_one_position_where_both_are_legal(
    firepanda: ModuleType,
) -> None:
    """Beside an interpreter that has both, they answer alike on every row and
    every method. That is the claim that makes the refusal a refusal rather than
    a second reading of the pattern, and it is the reason the anchoring above
    could change spelling without changing an answer."""
    if not ZED_IS_AN_ESCAPE:
        pytest.skip("this interpreter has no \\z escape")
    mine = made(firepanda)
    for method in ("contains", "count", "fullmatch"):
        zed = getattr(mine.str, method)("a\\z", flags=re.MULTILINE).tolist()
        big = getattr(mine.str, method)("a\\Z", flags=re.MULTILINE).tolist()
        assert zed == big, method
