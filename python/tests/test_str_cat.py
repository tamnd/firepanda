"""`str.cat`, checked against pandas.

The first name on this accessor whose answer is narrower than a column. Every
other one reads a column and writes a column, and this one reads a column and
writes a single string.

It is really two operations sharing a name, and which one you get depends on
whether you passed `others`. With it, pandas lines a second column up against
this one and concatenates row by row, and the answer is a column. Without it,
pandas folds this column into one string, and the answer is a scalar. Only the
second is written here, and the first is refused by name: pandas aligns the two
columns on their labels before concatenating anything, so a row of `others` is
matched to the row of this column carrying the same label rather than the one
sitting in the same position, and alignment is not written in this library yet.
Doing it by position instead would answer a different question without saying
so. There is a test at the bottom holding the refusal in place.

Two things about the form that is written are worth knowing before reading the
assertions, and both are about a row that is missing.

The first is that a missing row is dropped rather than blanked, and dropped
means it takes its separator with it. Three rows with the middle one missing
join to two pieces and one separator, not to two separators with nothing
between them. That is the single thing about this method an implementation gets
wrong, because a join written the obvious way replaces the row with an empty
string and leaves the separator where it was.

The second is that an empty row is not a missing one. It is readable, it is
never dropped, and it does leave two separators with nothing between them. So
the two cases that look identical in the output are reached by opposite rules,
and there is a test below that puts them side by side for exactly that reason.

`na_rep` turns the first case into the second. Given one, a missing row is not
missing any more and behaves like an empty row carrying that text. pandas reads
whether to drop off whether the argument was supplied, which means `na_rep=""`
is a real request and is not the same as leaving it out: it keeps the row and
its separator and drops only the text. That distinction is a test of its own.

One last measurement worth recording rather than matching. pandas checks
neither `sep` nor `na_rep`, and lets both fall into `str.join`, so `sep=1`
comes back as an `AttributeError` about `int` having no attribute `join` and
`na_rep=1` as a `TypeError` about sequence items. Both name an implementation
rather than the mistake, so this library refuses them with a message that says
which argument was wrong, and the two tests at the bottom assert the type of
the error rather than its sentence.
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
    "a",
    None,
    "",
    "bb",
    "ccc",
    None,
    "héllo",
    "日本",
    " ",
    "z",
]
"""Ten rows, each of them there to catch a different way of being wrong.

The first is ordinary. The second is missing and sits between two readable rows,
which is where a dropped separator can be counted. The third is empty and
readable, which is the case that looks like the second in the output and is not
it. The fourth and fifth are ordinary and of different lengths. The sixth is
missing and is followed by readable rows rather than ending the column, so a
trailing separator has somewhere to appear. The two after that hold characters
wider than a byte, because a length added up before anything is written is the
one thing that can be counted in the wrong unit. The ninth is a single space,
which is readable and is easy to mistake for a separator. The last is ordinary.
"""

SEPARATORS = ("", ",", ", ", "--", "•", " ")
"""Six separators: absent, short, long, wide, and one that rows also contain."""


def made(firepanda: ModuleType, values: list[Any] = ROWS) -> Any:
    """A firepanda text column."""
    return firepanda.Series(values)


def theirs(values: list[Any] = ROWS) -> Any:
    """The same column in pandas, held the way pandas holds it by default."""
    import pandas as pd

    return pd.Series(values, dtype="str")


@needs_pandas
def test_a_fold_matches_pandas_on_every_separator(firepanda: ModuleType) -> None:
    """The whole column joined, for six separators and both missing row rules."""
    for sep in SEPARATORS:
        assert made(firepanda).str.cat(sep=sep) == theirs().str.cat(sep=sep)
        assert made(firepanda).str.cat(sep=sep, na_rep="?") == theirs().str.cat(sep=sep, na_rep="?")


@needs_pandas
def test_a_missing_row_is_dropped_and_takes_its_separator_with_it(
    firepanda: ModuleType,
) -> None:
    """Three rows with the middle one missing join to one separator, not two.

    This is the rule the obvious implementation gets wrong, and it is worth an
    assertion that names the answer rather than only comparing to pandas.
    """
    rows = ["a", None, "b"]
    assert made(firepanda, rows).str.cat(sep="-") == "a-b"
    assert theirs(rows).str.cat(sep="-") == "a-b"


@needs_pandas
def test_an_empty_row_is_readable_and_keeps_its_separator(
    firepanda: ModuleType,
) -> None:
    """The case that looks like a dropped row in the output and is not one.

    An empty row leaves two separators with nothing between them. A missing row
    leaves one. The two are put side by side here because every other test in
    this file would pass with the two treated as the same thing.
    """
    assert made(firepanda, ["a", "", "b"]).str.cat(sep="-") == "a--b"
    assert theirs(["a", "", "b"]).str.cat(sep="-") == "a--b"
    assert made(firepanda, ["a", None, "b"]).str.cat(sep="-") == "a-b"
    assert theirs(["a", None, "b"]).str.cat(sep="-") == "a-b"


@needs_pandas
def test_a_stand_in_makes_a_missing_row_into_a_row(firepanda: ModuleType) -> None:
    """With `na_rep` the row is not dropped and its separator stays."""
    rows = ["a", None, "b"]
    assert made(firepanda, rows).str.cat(sep="-", na_rep="?") == "a-?-b"
    assert theirs(rows).str.cat(sep="-", na_rep="?") == "a-?-b"


@needs_pandas
def test_an_empty_stand_in_is_not_the_same_as_no_stand_in(
    firepanda: ModuleType,
) -> None:
    """`na_rep=""` keeps the row and its separator and drops only the text.

    pandas decides whether to drop a missing row from whether the argument was
    supplied and not from what it holds, so the empty string is a real answer
    here rather than a way of spelling the default.
    """
    rows = ["a", None, "b"]
    assert made(firepanda, rows).str.cat(sep="-", na_rep="") == "a--b"
    assert theirs(rows).str.cat(sep="-", na_rep="") == "a--b"
    assert made(firepanda, rows).str.cat(sep="-") == "a-b"


@needs_pandas
def test_no_separator_at_all_is_the_default(firepanda: ModuleType) -> None:
    """`sep=None` is the empty string and is what a bare call uses."""
    assert made(firepanda).str.cat() == theirs().str.cat()
    assert made(firepanda).str.cat() == made(firepanda).str.cat(sep="")
    assert made(firepanda, ["a", "b", "c"]).str.cat() == "abc"


@needs_pandas
def test_a_separator_never_appears_at_either_end(firepanda: ModuleType) -> None:
    """Whatever is joined, the answer starts and ends with a row."""
    for sep in SEPARATORS:
        if sep == "":
            continue
        mine = made(firepanda, ["aa", "bb"]).str.cat(sep=sep)
        assert mine == theirs(["aa", "bb"]).str.cat(sep=sep)
        assert not mine.startswith(sep)
        assert not mine.endswith(sep)


@needs_pandas
def test_a_missing_row_at_an_end_loses_no_separator(firepanda: ModuleType) -> None:
    """A dropped row at an end takes a separator that was never there.

    So a column whose first or last row is missing folds to exactly what the
    readable rows alone would have folded to, with no separator left dangling.
    """
    expected = {
        ("a", "b", None): "a-b",
        (None, "a", "b"): "a-b",
        (None, "a", None): "a",
    }
    for rows, answer in expected.items():
        assert made(firepanda, list(rows)).str.cat(sep="-") == answer
        assert theirs(list(rows)).str.cat(sep="-") == answer


@needs_pandas
def test_one_row_never_reaches_the_separator(firepanda: ModuleType) -> None:
    """A column of one row folds to that row, whatever the separator is."""
    for sep in SEPARATORS:
        assert made(firepanda, ["only"]).str.cat(sep=sep) == "only"
        assert theirs(["only"]).str.cat(sep=sep) == "only"


@needs_pandas
def test_a_fold_counts_bytes_and_the_answer_reads_back(firepanda: ModuleType) -> None:
    """Characters wider than a byte survive being measured before being written.

    The length of the answer is added up in one pass and written in another, and
    a row whose character count and byte count differ is the only thing that can
    tell a length counted in the wrong unit from one that was not.
    """
    rows = ["héllo", "wörld"]
    mine = made(firepanda, rows).str.cat(sep="•")
    assert mine == theirs(rows).str.cat(sep="•")
    assert mine == "héllo•wörld"
    assert len(mine) == 11
    assert len(mine.encode()) == 15


@needs_pandas
def test_a_column_with_nothing_readable_folds_to_nothing(
    firepanda: ModuleType,
) -> None:
    """A column of no rows and a column of only missing rows both give "".

    firepanda's own constructor cannot make a typed empty text column directly,
    because `Series([])` is a float column, so the two are reached by cutting a
    text column down.
    """
    assert made(firepanda, ["a"]).head(0).str.cat(sep=",") == ""
    assert theirs(["a"]).head(0).str.cat(sep=",") == ""
    assert made(firepanda, ["a", None]).tail(1).str.cat(sep=",") == ""
    assert theirs(["a", None]).tail(1).str.cat(sep=",") == ""


@needs_pandas
def test_a_column_of_only_missing_rows_folds_to_the_stand_in(
    firepanda: ModuleType,
) -> None:
    """Given a stand in there is something readable after all."""
    assert made(firepanda, ["a", None]).tail(1).str.cat(sep=",", na_rep="?") == "?"
    assert theirs(["a", None]).tail(1).str.cat(sep=",", na_rep="?") == "?"


@needs_pandas
def test_the_answer_is_a_plain_string(firepanda: ModuleType) -> None:
    """Not a column of one row, and not a wrapper around one."""
    mine = made(firepanda).str.cat(sep=",")
    assert isinstance(mine, str)
    assert isinstance(theirs().str.cat(sep=","), str)


@needs_pandas
def test_cat_with_others_is_refused_by_name(firepanda: ModuleType) -> None:
    """The row by row half needs alignment, which is not written yet.

    pandas matches a row of `others` to the row of this column carrying the same
    label rather than the one in the same position, so serving this by position
    would answer a different question. The refusal is a `NotImplementedError`
    and says so, which puts it on the board as a gap rather than as a
    disagreement.
    """
    with pytest.raises(NotImplementedError, match="alignment"):
        made(firepanda).str.cat(made(firepanda))
    with pytest.raises(NotImplementedError, match="alignment"):
        made(firepanda).str.cat(others=["a"] * len(ROWS))


@needs_pandas
def test_a_separator_that_is_not_a_string_is_refused(firepanda: ModuleType) -> None:
    """pandas lets this fall into `str.join` and answers for the wrong thing.

    Its sentence is about an `int` having no attribute `join`, which names an
    implementation detail. The refusal here is the same kind of error and says
    which argument was wrong.
    """
    import pandas as pd

    with pytest.raises(TypeError):
        made(firepanda).str.cat(sep=1)
    with pytest.raises((TypeError, AttributeError)):
        pd.Series(["a"], dtype="str").str.cat(sep=1)


@needs_pandas
def test_a_stand_in_that_is_not_a_string_is_refused(firepanda: ModuleType) -> None:
    """The same, for the other argument pandas does not check."""
    import pandas as pd

    with pytest.raises(TypeError):
        made(firepanda).str.cat(sep=",", na_rep=1)
    with pytest.raises(TypeError):
        pd.Series(["a", None], dtype="str").str.cat(sep=",", na_rep=1)


@needs_pandas
def test_join_is_accepted_and_has_nothing_to_do(firepanda: ModuleType) -> None:
    """It says how to line `others` up and there is no `others`.

    pandas ignores it here too, and in fact ignores it so thoroughly that it
    takes a value that is not one of the four it documents without a word, even
    when there is an `others` for it to have applied to.
    """
    assert made(firepanda).str.cat(sep=",", join="outer") == theirs().str.cat(sep=",", join="outer")
    assert made(firepanda).str.cat(sep=",", join="bogus") == theirs().str.cat(sep=",", join="bogus")


@needs_pandas
def test_cat_on_a_column_that_is_not_text_is_refused(firepanda: ModuleType) -> None:
    """The accessor refuses before the method is reached, which is pandas."""
    with pytest.raises(AttributeError):
        firepanda.Series([1, 2, 3]).str.cat(sep=",")
