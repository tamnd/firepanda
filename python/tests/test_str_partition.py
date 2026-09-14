"""`str.partition` and `str.rpartition`, checked against pandas.

The first two names on this accessor whose answer is a frame rather than a
column. Each cuts every row at one occurrence of a separator and hands back
three columns: what came before it, the separator itself, and what came after.

Two things about the pair are worth knowing before reading the assertions.

The first is that the two names differ in two places and only one of them is the
one the names suggest. `partition` cuts at the first occurrence and `rpartition`
cuts at the last, which is what a reader expects. The other difference is what
happens to a row the separator is not in at all: the whole row survives, and
`partition` puts it in the first column while `rpartition` puts it in the third.
That is Python's rule, pandas hands this name to Python's own `str.partition`,
and an implementation written from the name alone puts the row in the first
column both times and passes every other test in this file.

The second is the two places this cannot match pandas, and both are at the
bottom of the file with a test each so neither can drift unnoticed.

pandas labels the three columns with the integers 0, 1 and 2, and a firepanda
frame holds text column labels, so the same three come back as "0", "1" and "2".
That is registered on the board as a divergence rather than worked around,
because what would fix it is a column label type and not a different string
written here. Every other test below compares the values under the labels rather
than the labels themselves.

pandas also decides both of its refusals and the width of its answer one row at
a time, because it calls Python's own `str.partition` on every row and lets
pyarrow infer the shape from what came back. So on a column with nothing
readable in it pandas raises nothing at all and the answer is not three columns
wide: an empty column gives back a frame with no columns and a column of only
missing rows gives back one. This library checks the separator once before it
starts and always answers three columns, which is what the pandas documentation
describes, and the tests below assert that rather than the measurement.
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
    "a b c",
    "abc",
    "",
    " ",
    "  x",
    "x  ",
    "héllo wörld",
    "日本 語",
    "a--b--c",
    None,
]
"""Ten rows, each of them there to catch a different way of being wrong.

The first holds the separator twice, so the two names choose different
occurrences. The second does not hold it at all, which is the row that tells the
two apart in the way a reader would not guess. The third is empty and the fourth
is the separator and nothing else. The two after those have the separator at an
end, so one of the three columns comes out empty and the other two do not. The
next two hold characters wider than a byte on both sides of the cut, because an
offset into a row is a byte offset and a row that is all ASCII cannot notice an
implementation that thought it was a character offset. The ninth is for a
separator of more than one character and the last is the missing row.
"""

SEPARATORS = (" ", "--", "-", "b", "x", "ö", "日", "語", "z", "a b c")
"""Ten separators: present, absent, repeated, wide, and the whole of a row."""


def made(firepanda: ModuleType, values: list[Any] = ROWS) -> Any:
    """A firepanda text column."""
    return firepanda.Series(values)


def theirs(values: list[Any] = ROWS) -> Any:
    """The same column in pandas, held the way pandas holds it by default."""
    import pandas as pd

    return pd.Series(values, dtype="str")


def mine_rows(frame: Any) -> list[list[Any]]:
    """The three columns of a firepanda answer, read row by row."""
    columns = [frame[name].tolist() for name in frame.columns]
    return [[column[i] for column in columns] for i in range(len(columns[0]))]


def their_rows(frame: Any) -> list[list[Any]]:
    """The same for pandas, with a missing row read as None.

    pandas hands a missing row back as a NaN float and this library hands back
    None, which is `engine/missing-spelling` and is not what any test here is
    about.
    """
    return [
        [None if value is None or value != value else value for value in row]
        for row in frame.to_dict("split")["data"]
    ]


@needs_pandas
def test_both_names_match_pandas_on_every_separator(firepanda: ModuleType) -> None:
    """Which is the whole claim, made once over ten separators and ten rows."""
    mine, them = made(firepanda), theirs()
    for name in ("partition", "rpartition"):
        for sep in SEPARATORS:
            got = mine_rows(getattr(mine.str, name)(sep))
            assert got == their_rows(getattr(them.str, name)(sep)), (name, sep)


@needs_pandas
def test_the_two_names_choose_different_occurrences(firepanda: ModuleType) -> None:
    """The half of the difference a reader would guess."""
    rows = ["a b c", "a--b--c"]
    mine = made(firepanda, rows)
    assert mine_rows(mine.str.partition(" "))[0] == ["a", " ", "b c"]
    assert mine_rows(mine.str.rpartition(" "))[0] == ["a b", " ", "c"]
    assert mine_rows(mine.str.partition("--"))[1] == ["a", "--", "b--c"]
    assert mine_rows(mine.str.rpartition("--"))[1] == ["a--b", "--", "c"]


@needs_pandas
def test_a_row_without_the_separator_goes_to_opposite_ends(firepanda: ModuleType) -> None:
    """The half a reader would not guess, and the one worth writing out twice.

    The row is not dropped and is not blanked. It survives whole, and which of
    the three columns it lands in is the only thing that says which name was
    called. An implementation that put it first both times would be wrong only
    here.
    """
    rows = ["abc", "héllowörld"]
    mine, them = made(firepanda, rows), theirs(rows)
    assert mine_rows(mine.str.partition(" ")) == [["abc", "", ""], ["héllowörld", "", ""]]
    assert mine_rows(mine.str.rpartition(" ")) == [["", "", "abc"], ["", "", "héllowörld"]]
    assert their_rows(them.str.partition(" ")) == mine_rows(mine.str.partition(" "))
    assert their_rows(them.str.rpartition(" ")) == mine_rows(mine.str.rpartition(" "))


@needs_pandas
def test_a_separator_at_an_end_leaves_one_part_empty(firepanda: ModuleType) -> None:
    """And which part is left empty is not the same for the two names."""
    rows = ["  x", "x  "]
    mine = made(firepanda, rows)
    assert mine_rows(mine.str.partition(" ")) == [["", " ", " x"], ["x", " ", " "]]
    assert mine_rows(mine.str.rpartition(" ")) == [[" ", " ", "x"], ["x ", " ", ""]]


@needs_pandas
def test_a_cut_counts_bytes_and_the_parts_still_read_back(firepanda: ModuleType) -> None:
    """A row wider than a byte on both sides of the separator, and a wide one.

    A separator is matched whole, so a cut can only ever land on a character
    boundary, which is why nothing here has to decode the row to be safe.
    """
    rows = ["héllo wörld", "日本 語"]
    mine, them = made(firepanda, rows), theirs(rows)
    for sep in (" ", "ö", "本", "語"):
        for name in ("partition", "rpartition"):
            got = mine_rows(getattr(mine.str, name)(sep))
            assert got == their_rows(getattr(them.str, name)(sep)), (name, sep)
    assert mine_rows(mine.str.partition("ö"))[0] == ["héllo w", "ö", "rld"]


@needs_pandas
def test_a_missing_row_is_missing_in_all_three(firepanda: ModuleType) -> None:
    """There is nothing to cut, so there is no part before the cut either."""
    mine, them = made(firepanda), theirs()
    for name in ("partition", "rpartition"):
        assert mine_rows(getattr(mine.str, name)(" "))[-1] == [None, None, None], name
        assert getattr(them.str, name)(" ").iloc[-1].isna().tolist() == [True] * 3, name


@needs_pandas
def test_the_default_separator_is_a_space(firepanda: ModuleType) -> None:
    """Which pandas chose and which is the one argument with a default here."""
    mine, them = made(firepanda), theirs()
    for name in ("partition", "rpartition"):
        assert mine_rows(getattr(mine.str, name)()) == mine_rows(getattr(mine.str, name)(" "))
        assert mine_rows(getattr(mine.str, name)()) == their_rows(getattr(them.str, name)())


@needs_pandas
def test_an_empty_separator_is_refused_the_way_pandas_refuses_it(
    firepanda: ModuleType,
) -> None:
    """Both names, and with pandas' own sentence, which is Python's sentence."""
    import pandas as pd

    mine, them = made(firepanda), theirs()
    for name in ("partition", "rpartition"):
        with pytest.raises(ValueError) as caught:
            getattr(mine.str, name)("")
        assert "empty separator" in str(caught.value), name
        with pytest.raises(ValueError):
            getattr(them.str, name)("")
    assert pd.__version__  # the refusal above is pandas 3's and was measured


@needs_pandas
def test_a_separator_that_is_not_a_string_is_a_type_error(firepanda: ModuleType) -> None:
    """pandas raises here too and names the type it was handed."""
    mine = made(firepanda)
    for name in ("partition", "rpartition"):
        for bad, written in ((1, "int"), (None, "NoneType"), (1.5, "float"), ([" "], "list")):
            with pytest.raises(TypeError) as caught:
                getattr(mine.str, name)(bad)
            assert written in str(caught.value), (name, bad)


@needs_pandas
def test_expand_false_is_refused_rather_than_approximated(firepanda: ModuleType) -> None:
    """The other half of the pandas argument, and the half that needs a type.

    `expand=False` answers one column of three element tuples. There is no
    column type here that holds a tuple, so this is a gap rather than a
    disagreement, and it is refused by name so the board reads it as one.
    """
    mine = made(firepanda)
    for name in ("partition", "rpartition"):
        with pytest.raises(firepanda.errors.UnsupportedError) as caught:
            getattr(mine.str, name)(" ", expand=False)
        assert "expand=False" in str(caught.value), name


@needs_pandas
def test_the_three_columns_are_labelled_with_text_and_pandas_uses_numbers(
    firepanda: ModuleType,
) -> None:
    """The one place this pair cannot match, asserted so it cannot drift.

    pandas labels the three columns 0, 1 and 2 as integers. A firepanda frame
    holds text column labels, so the same three are "0", "1" and "2". Every
    value under them agrees, which the tests above check, and the labels do not,
    which this checks. What would close it is a column label type, not a
    different string written in this method.
    """
    mine, them = made(firepanda), theirs()
    assert mine.str.partition(" ").columns == ["0", "1", "2"]
    assert list(them.str.partition(" ").columns) == [0, 1, 2]
    assert mine.str.rpartition(" ").columns == ["0", "1", "2"]


@needs_pandas
def test_a_separator_as_long_as_the_row_still_cuts(firepanda: ModuleType) -> None:
    """The edge where both parts come out empty and the middle is everything."""
    rows = ["a b c", "a b"]
    mine, them = made(firepanda, rows), theirs(rows)
    assert mine_rows(mine.str.partition("a b c")) == [["", "a b c", ""], ["a b", "", ""]]
    assert their_rows(them.str.partition("a b c")) == mine_rows(mine.str.partition("a b c"))


@needs_pandas
def test_a_column_with_nothing_readable_still_answers_three_columns(
    firepanda: ModuleType,
) -> None:
    """The second divergence, and the one where this library is the stricter.

    pandas hands every row to Python's `str.partition` and lets pyarrow work out
    the shape from the tuples that come back, so a column with no readable rows
    tells it nothing about the width. An empty column comes back with no columns
    at all and a column of only missing rows comes back with one, which is a
    frame the pandas documentation for this name does not describe.

    Here the width is decided by the method and not by the data, so both come
    back three wide. Nothing about a cut depends on how many rows there are.
    """
    import pandas as pd

    empty = made(firepanda, ["a"]).head(0)
    assert empty.str.partition(" ").shape == (0, 3)
    assert empty.str.rpartition(" ").shape == (0, 3)
    assert pd.Series(["a"], dtype="str").head(0).str.partition(" ").shape == (0, 0)

    nulls = made(firepanda, ["a", None]).tail(1)
    assert nulls.str.partition(" ").shape == (1, 3)
    assert pd.Series([None], dtype="str").str.partition(" ").shape == (1, 1)


@needs_pandas
def test_a_column_with_nothing_readable_is_still_refused_a_bad_separator(
    firepanda: ModuleType,
) -> None:
    """The same divergence seen from the other side, which is the useful side.

    pandas raises inside CPython on the first row it reaches, so a column with no
    rows to reach accepts an empty separator and accepts a separator that is not
    a string at all. The argument is just as wrong in both cases and the only
    reason it goes unremarked is that there was no work to do.
    """
    import pandas as pd

    empty = made(firepanda, ["a"]).head(0)
    for name in ("partition", "rpartition"):
        with pytest.raises(ValueError):
            getattr(empty.str, name)("")
        with pytest.raises(TypeError):
            getattr(empty.str, name)(1)
    assert pd.Series(["a"], dtype="str").head(0).str.partition("").shape == (0, 0)


@needs_pandas
def test_partition_on_a_column_that_is_not_text_is_refused_at_the_accessor(
    firepanda: ModuleType,
) -> None:
    """Which is where pandas refuses it, before either name is reached."""
    column = firepanda.Series([1, 2, 3])
    with pytest.raises(AttributeError):
        column.str.partition(" ")
