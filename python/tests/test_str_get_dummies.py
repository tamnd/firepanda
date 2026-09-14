"""`str.get_dummies`, checked against pandas.

The only name on this accessor whose answer has a width nobody can work out
before the column has been read. `partition` always answers three columns and
`cat` always answers one string. This one splits every row at a separator and
answers one column per distinct token, so both how wide the frame is and what
the columns are called are properties of the data.

Four things are worth knowing before reading the assertions.

The columns are labelled with the tokens, which are text, and a firepanda frame
holds text column labels. So this method is clear of the divergence `partition`
ran into and the labels match pandas exactly, which is why the tests below
compare them directly rather than comparing their types.

The columns are in byte order, which is code point order because that is what
UTF-8 was designed to give, so a digit sorts before a capital before an
underscore before a lowercase letter before anything accented. No locale is
consulted by either library.

What falls out between two separators is a token even when it is nothing. An
empty row contributes the empty string, and so does a row that starts or ends
with the separator. The empty string is therefore a column label a dummy frame
can really have, which reads like an accident and is what pandas does.

A missing row is a row of zeros and not a row of nulls. That is the one place on
this accessor where a missing row does not stay missing, and it follows from the
answer being a frame of counts: a count of a row that says nothing is nothing
rather than unknown.

There is one disagreement, and it is pandas failing rather than choosing. A
column with no readable rows has no tokens, so the frame has no columns. pandas
agrees for a column with no rows at all and answers a frame of shape (0, 0). For
a column that has rows but no readable ones it raises `ValueError: Empty data
passed with indices specified.` out of its own frame constructor, which is an
internal error about building a frame rather than a rule about this method. This
library answers the frame in both cases, and the test at the bottom asserts
pandas' failure so the difference is recorded as a decision.
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
    "a|b",
    "b|c",
    None,
    "",
    "a",
    "b|b",
    "|a",
    "a|",
    "é|ö",
    "日|本",
    "a||b",
    "one whole token",
]
"""Twelve rows, each of them there to catch a different way of being wrong.

The first holds two tokens and the second shares one of them and brings a new
one, so the token set is a union rather than a copy. The third is missing. The
fourth is empty, which is not missing and which contributes the empty token. The
fifth holds one token and no separator. The sixth holds the same token twice, so
the answer is a set and not a count. The three after that put the separator at an
end or in the middle twice, which are the other routes to the empty token. The
two after those hold characters wider than a byte on both sides of a separator,
because a split is a byte offset and a row that is all ASCII cannot notice an
implementation that thought it was a character offset. The last holds spaces so
that a token is not assumed to be a word.
"""

SEPARATORS = ("|", "||", "a", "|a", "-", " ")
"""Six separators: present, doubled, a separator that rows also hold as text."""


def made(firepanda: ModuleType, values: list[Any] = ROWS) -> Any:
    """A firepanda text column."""
    return firepanda.Series(values)


def theirs(values: list[Any] = ROWS) -> Any:
    """The same column in pandas, held the way pandas holds it by default."""
    import pandas as pd

    return pd.Series(values, dtype="str")


def mine_rows(frame: Any) -> list[list[Any]]:
    """A firepanda dummy frame read out row by row."""
    labels = list(frame.columns)
    return [[frame[label][i] for label in labels] for i in range(frame.shape[0])]


@needs_pandas
def test_the_labels_match_pandas_on_every_separator(firepanda: ModuleType) -> None:
    """Same tokens, same order, for six separators."""
    for sep in SEPARATORS:
        assert list(made(firepanda).str.get_dummies(sep=sep).columns) == list(
            theirs().str.get_dummies(sep=sep).columns
        )


@needs_pandas
def test_the_flags_match_pandas_on_every_separator(firepanda: ModuleType) -> None:
    """Same ones and zeros under the same labels."""
    for sep in SEPARATORS:
        assert (
            mine_rows(made(firepanda).str.get_dummies(sep=sep))
            == theirs().str.get_dummies(sep=sep).values.tolist()
        )


@needs_pandas
def test_the_columns_are_in_byte_order(firepanda: ModuleType) -> None:
    """Which is code point order, so a digit sorts before a letter.

    The empty token sorts first because a prefix sorts before what it is a
    prefix of, and the empty string is a prefix of everything.
    """
    rows = ["b|a|C|_|1|é|"]
    assert list(made(firepanda, rows).str.get_dummies().columns) == [
        "",
        "1",
        "C",
        "_",
        "a",
        "b",
        "é",
    ]
    assert list(theirs(rows).str.get_dummies().columns) == [
        "",
        "1",
        "C",
        "_",
        "a",
        "b",
        "é",
    ]


@needs_pandas
def test_a_token_seen_twice_is_one_column_holding_one(firepanda: ModuleType) -> None:
    """The answer is a membership and not a count."""
    rows = ["a|a|a"]
    assert list(made(firepanda, rows).str.get_dummies().columns) == ["a"]
    assert mine_rows(made(firepanda, rows).str.get_dummies()) == [[1]]
    assert theirs(rows).str.get_dummies().values.tolist() == [[1]]


@needs_pandas
def test_an_empty_piece_is_a_token(firepanda: ModuleType) -> None:
    """An empty row and a separator at either end all reach the empty label.

    All three come to the same token by different routes, and an implementation
    that skipped empty pieces would answer no columns for the first and one
    column for the other two.
    """
    rows = ["", "|a", "a|", "a||b"]
    assert list(made(firepanda, rows).str.get_dummies().columns) == ["", "a", "b"]
    assert list(theirs(rows).str.get_dummies().columns) == ["", "a", "b"]
    assert (
        mine_rows(made(firepanda, rows).str.get_dummies())
        == theirs(rows).str.get_dummies().values.tolist()
    )


@needs_pandas
def test_a_missing_row_is_zeros_and_not_nulls(firepanda: ModuleType) -> None:
    """The one place on this accessor a missing row does not stay missing."""
    frame = made(firepanda).str.get_dummies()
    for label in frame.columns:
        assert frame[label][2] == 0
        assert frame[label].isna().sum() == 0
    assert theirs().str.get_dummies().isna().sum().sum() == 0


@needs_pandas
def test_a_missing_row_contributes_no_label(firepanda: ModuleType) -> None:
    """It is skipped rather than being read as an empty row."""
    rows = ["a", None]
    assert list(made(firepanda, rows).str.get_dummies().columns) == ["a"]
    assert list(theirs(rows).str.get_dummies().columns) == ["a"]


@needs_pandas
def test_the_flags_are_int64_by_default(firepanda: ModuleType) -> None:
    """Which is pandas, and is not the bool a reader might expect."""
    frame = made(firepanda).str.get_dummies()
    assert {str(frame[label].dtype) for label in frame.columns} == {"int64"}
    assert set(map(str, theirs().str.get_dummies().dtypes)) == {"int64"}


@needs_pandas
def test_dtype_bool_gives_flags_rather_than_counts(firepanda: ModuleType) -> None:
    """The one value of `dtype` that is written."""
    frame = made(firepanda).str.get_dummies(dtype=bool)
    assert {str(frame[label].dtype) for label in frame.columns} == {"bool"}
    assert set(map(str, theirs().str.get_dummies(dtype=bool).dtypes)) == {"bool"}
    assert [[bool(v) for v in row] for row in mine_rows(frame)] == (
        theirs().str.get_dummies(dtype=bool).values.tolist()
    )


@needs_pandas
def test_a_dtype_that_is_not_written_is_refused(firepanda: ModuleType) -> None:
    """It says which two are written rather than saying the method is missing."""
    with pytest.raises(NotImplementedError, match="int64"):
        made(firepanda).str.get_dummies(dtype=float)


@needs_pandas
def test_a_separator_of_more_than_one_character(firepanda: ModuleType) -> None:
    """The separator is text and nothing assumes it is one character."""
    rows = ["a--b", "b--c"]
    assert list(made(firepanda, rows).str.get_dummies(sep="--").columns) == [
        "a",
        "b",
        "c",
    ]
    assert list(theirs(rows).str.get_dummies(sep="--").columns) == ["a", "b", "c"]


@needs_pandas
def test_a_row_without_the_separator_is_one_whole_token(firepanda: ModuleType) -> None:
    """Not dropped, and not split into characters."""
    rows = ["one whole token"]
    assert list(made(firepanda, rows).str.get_dummies().columns) == ["one whole token"]
    assert list(theirs(rows).str.get_dummies().columns) == ["one whole token"]


@needs_pandas
def test_a_split_counts_bytes_not_characters(firepanda: ModuleType) -> None:
    """A row that is all ASCII cannot notice an offset counted in characters."""
    rows = ["é|ö", "日|本"]
    assert list(made(firepanda, rows).str.get_dummies().columns) == [
        "é",
        "ö",
        "日",
        "本",
    ]
    assert (
        mine_rows(made(firepanda, rows).str.get_dummies())
        == theirs(rows).str.get_dummies().values.tolist()
    )


@needs_pandas
def test_an_empty_separator_is_refused(firepanda: ModuleType) -> None:
    """pandas refuses it too, from inside Arrow's split kernel."""
    import pandas as pd

    with pytest.raises(ValueError):
        made(firepanda).str.get_dummies(sep="")
    with pytest.raises(Exception, match=r"[Ee]mpty separator"):
        pd.Series(["ab"], dtype="str").str.get_dummies(sep="")


@needs_pandas
def test_a_separator_that_is_not_a_string_is_refused(firepanda: ModuleType) -> None:
    """pandas refuses it with Arrow's sentence about bytes."""
    import pandas as pd

    with pytest.raises(TypeError):
        made(firepanda).str.get_dummies(sep=1)
    with pytest.raises(TypeError):
        pd.Series(["ab"], dtype="str").str.get_dummies(sep=1)


@needs_pandas
def test_a_column_with_no_rows_has_no_columns(firepanda: ModuleType) -> None:
    """No tokens, so no columns, and both libraries agree.

    A typed empty text column is reached by cutting one down, because
    `Series([])` is a float column and the accessor would refuse it first.
    """
    assert made(firepanda, ["a"]).head(0).str.get_dummies().shape == (0, 0)
    assert theirs(["a"]).head(0).str.get_dummies().shape == (0, 0)


@needs_pandas
def test_a_column_of_only_missing_rows_is_where_pandas_cannot_build_it(
    firepanda: ModuleType,
) -> None:
    """The one disagreement, and it is pandas failing rather than choosing.

    No row is readable, so there are no tokens and the frame has no columns,
    which is the same answer as the test above. pandas raises out of its own
    frame constructor instead, with a message about empty data and indices that
    describes building a frame rather than anything about this method. The
    measurement is asserted here so the difference is on record as a decision.
    """
    import pandas as pd

    assert made(firepanda, ["a", None]).tail(1).str.get_dummies().shape == (0, 0)
    with pytest.raises(ValueError, match="Empty data passed with indices specified"):
        pd.Series(["a", None], dtype="str").tail(1).str.get_dummies()


@needs_pandas
def test_get_dummies_on_a_column_that_is_not_text_is_refused(
    firepanda: ModuleType,
) -> None:
    """The accessor refuses before the method is reached, which is pandas."""
    with pytest.raises(AttributeError):
        firepanda.Series([1, 2, 3]).str.get_dummies()
