"""The group by door, checked against a running pandas rather than a table.

`df.groupby(...)` is the busiest thing in pandas that firepanda had no door to
at all. The core had the grouping and seventeen reductions over it before any of
this was written, so these tests are not about whether a group's total is right,
which `tests/test_group.mojo` measures in Mojo over more dtypes than are
reachable from here. They are about whether the object in between behaves the
way pandas' one does: the same groups in the same order, the same shape of
answer, the same column names, and the same error from the same line.

Comparing against a live pandas rather than against constants is the same choice
`test_reductions.py` makes and for the same reason. What is deliberately not
compared is the index. pandas puts the keys in one and firepanda has positional
labels only, which document 26 records, so the values and the column names are
compared and the labels are left to the index work.

The refusals are tested as carefully as the answers, because a declared argument
that is quietly dropped is the failure that takes longest to find.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

KEYS = ["b", "a", "b", "a", "b", "a", "b", "a"]
"""Two groups of four, interleaved, and not in sorted order to start with.

Interleaved so that a reduction that read a run of rows rather than a group
would get a different answer, and unsorted so that `sort=True` has something to
do and `sort=False` has something else to do.
"""

VALUES = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
OTHER = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

DATA: dict[str, Any] = {"k": KEYS, "v": VALUES, "w": OTHER}
"""Floats rather than integers for the same reason `test_reductions.py` gives.

Four rows per group, which is what `skew` and `kurt` need before it has an answer at all,
so the nineteen can be tested as one list rather than with an exception in it.
"""

OVER_A_FRAME = [
    "sum",
    "mean",
    "min",
    "max",
    "count",
    "first",
    "last",
    "median",
    "nunique",
    "std",
    "var",
    "sem",
    "skew",
    "kurt",
    "quantile",
    "prod",
    "any",
    "all",
]
"""The eighteen that answer a frame. `size` answers a column and is its own test."""


def same_values(mine: Any, theirs: Any) -> None:
    """Compares a firepanda frame against a pandas one, column by column.

    The column names and the numbers in them, which is everything the two
    libraries claim to agree about here. The row labels are not compared, for
    the reason the module docstring gives.
    """
    assert list(mine.columns) == list(theirs.columns)
    for name in mine.columns:
        assert mine[name].tolist() == pytest.approx(theirs[name].tolist()), name


@needs_pandas
@pytest.mark.parametrize("name", OVER_A_FRAME)
def test_a_grouped_reduction_gives_the_pandas_answer(firepanda: ModuleType, name: str) -> None:
    """The whole point, once per reduction."""
    import pandas as pd

    mine = getattr(firepanda.DataFrame(DATA).groupby("k"), name)()
    theirs = getattr(pd.DataFrame(DATA).groupby("k"), name)()
    same_values(mine, theirs)


@needs_pandas
def test_the_key_column_is_not_reduced_along_with_the_rest(firepanda: ModuleType) -> None:
    """It is what the groups are, so it is not one of the columns they are over."""
    import pandas as pd

    mine = firepanda.DataFrame(DATA).groupby("k").sum()
    assert list(mine.columns) == ["v", "w"]
    assert list(pd.DataFrame(DATA).groupby("k").sum().columns) == ["v", "w"]


@needs_pandas
def test_size_counts_rows_and_answers_a_column(firepanda: ModuleType) -> None:
    """The one reduction that does not read a column, so it does not answer per column.

    pandas gives a series of counts here rather than a frame, since the answer is
    one number per group either way, and this gives the same one.
    """
    import pandas as pd

    mine = firepanda.DataFrame(DATA).groupby("k").size()
    theirs = pd.DataFrame(DATA).groupby("k").size()
    assert isinstance(mine, firepanda.Series)
    assert mine.tolist() == theirs.tolist()
    assert mine.name is theirs.name is None


@needs_pandas
def test_as_index_false_puts_the_key_back_as_a_column(firepanda: ModuleType) -> None:
    """Which is the whole of what the flag does, since there is no index to move it to."""
    import pandas as pd

    mine = firepanda.DataFrame(DATA).groupby("k", as_index=False).mean()
    theirs = pd.DataFrame(DATA).groupby("k", as_index=False).mean()
    same_values(mine, theirs)
    assert mine["k"].tolist() == ["a", "b"]


@needs_pandas
def test_size_without_the_index_is_a_frame_with_a_count_column(firepanda: ModuleType) -> None:
    """pandas names the count column `size` here and gives back a frame, not a column."""
    import pandas as pd

    mine = firepanda.DataFrame(DATA).groupby("k", as_index=False).size()
    theirs = pd.DataFrame(DATA).groupby("k", as_index=False).size()
    same_values(mine, theirs)


@needs_pandas
@pytest.mark.parametrize("name", ["sum", "mean", "count", "nunique", "std", "prod", "all"])
def test_one_column_of_a_grouping_answers_a_column(firepanda: ModuleType, name: str) -> None:
    """`df.groupby(k)[v]` reduces the one column and hands back a column named after it."""
    import pandas as pd

    mine = getattr(firepanda.DataFrame(DATA).groupby("k")["v"], name)()
    theirs = getattr(pd.DataFrame(DATA).groupby("k")["v"], name)()
    assert isinstance(mine, firepanda.Series)
    assert mine.name == theirs.name
    assert mine.tolist() == pytest.approx(theirs.tolist())


@needs_pandas
def test_a_list_of_columns_of_a_grouping_answers_a_frame(firepanda: ModuleType) -> None:
    """One name narrows to a column and a list of one narrows to a frame, as in pandas."""
    import pandas as pd

    mine = firepanda.DataFrame(DATA).groupby("k")[["v"]].sum()
    theirs = pd.DataFrame(DATA).groupby("k")[["v"]].sum()
    assert isinstance(mine, firepanda.DataFrame)
    same_values(mine, theirs)


@needs_pandas
def test_two_keys_group_by_the_pair(firepanda: ModuleType) -> None:
    """The pair rather than one after the other, which is a different set of groups."""
    import pandas as pd

    data = {"a": ["x", "x", "y", "y"], "b": [1, 2, 1, 2], "v": [10.0, 20.0, 30.0, 40.0]}
    mine = firepanda.DataFrame(data).groupby(["a", "b"], as_index=False).sum()
    theirs = pd.DataFrame(data).groupby(["a", "b"], as_index=False).sum()
    same_values(mine, theirs)


@needs_pandas
def test_sort_false_keeps_the_order_the_groups_first_appear_in(firepanda: ModuleType) -> None:
    """Which is the order the rows are in and not the order the keys sort in."""
    import pandas as pd

    mine = firepanda.DataFrame(DATA).groupby("k", as_index=False, sort=False).sum()
    theirs = pd.DataFrame(DATA).groupby("k", as_index=False, sort=False).sum()
    assert mine["k"].tolist() == ["b", "a"]
    same_values(mine, theirs)


@needs_pandas
def test_a_missing_key_is_dropped_unless_it_is_asked_for(firepanda: ModuleType) -> None:
    """`dropna=True` is the default and leaves the rows out, and `False` groups them."""
    import pandas as pd

    data: dict[str, Any] = {"k": ["a", None, "a", None], "v": [1.0, 2.0, 3.0, 4.0]}
    dropped = firepanda.DataFrame(data).groupby("k", as_index=False).sum()
    same_values(dropped, pd.DataFrame(data).groupby("k", as_index=False).sum())
    assert dropped["v"].tolist() == [4.0]

    kept = firepanda.DataFrame(data).groupby("k", as_index=False, dropna=False).sum()
    assert sorted(kept["v"].tolist()) == [4.0, 6.0]


@needs_pandas
@pytest.mark.parametrize("name", ["std", "var", "sem"])
@pytest.mark.parametrize("ddof", [0, 1, 2])
def test_the_delta_degrees_of_freedom_reaches_the_divisor(
    firepanda: ModuleType, name: str, ddof: int
) -> None:
    """A `ddof` that is declared and dropped is right at the default and nowhere else."""
    import pandas as pd

    mine = getattr(firepanda.DataFrame(DATA).groupby("k"), name)(ddof=ddof)
    theirs = getattr(pd.DataFrame(DATA).groupby("k"), name)(ddof=ddof)
    same_values(mine, theirs)


@needs_pandas
@pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.9, 1.0])
def test_the_quantile_lands_where_pandas_lands(firepanda: ModuleType, q: float) -> None:
    """Including the two ends, where there is nothing to interpolate between."""
    import pandas as pd

    mine = firepanda.DataFrame(DATA).groupby("k").quantile(q)
    same_values(mine, pd.DataFrame(DATA).groupby("k").quantile(q))


def test_a_key_that_is_not_a_column_is_a_key_error(firepanda: ModuleType) -> None:
    """Raised out of the `groupby` call and not out of the reduction after it.

    pandas raises here too, and a program that catches the wrong line is a
    program whose error handling does not run.
    """
    with pytest.raises(KeyError, match="nope"):
        firepanda.DataFrame(DATA).groupby("nope")


def test_a_column_that_is_not_there_is_a_key_error_too(firepanda: ModuleType) -> None:
    """Narrowing to a column that does not exist fails at the narrowing."""
    with pytest.raises(KeyError, match="nope"):
        firepanda.DataFrame(DATA).groupby("k")["nope"]


def test_grouping_by_nothing_says_what_pandas_says(firepanda: ModuleType) -> None:
    """Both of the two ways of asking for no groups, with the two pandas messages."""
    with pytest.raises(TypeError, match="one of 'by' and 'level'"):
        firepanda.DataFrame(DATA).groupby()
    with pytest.raises(ValueError, match="No group keys passed"):
        firepanda.DataFrame(DATA).groupby([])


def test_a_key_that_is_neither_a_column_nor_values_is_a_key_error(firepanda: ModuleType) -> None:
    """A number is read as a column's name, as pandas reads it."""
    with pytest.raises(KeyError):
        firepanda.DataFrame(DATA).groupby(0)


def test_a_key_and_a_level_together_refuse(firepanda: ModuleType) -> None:
    """By name, with the reason in the message, rather than being ignored."""
    with pytest.raises(NotImplementedError, match="level"):
        firepanda.DataFrame(DATA).groupby("k", level=0)


@pytest.mark.parametrize(
    ("call", "arguments", "expected"),
    [
        ("sum", {"engine": "numba"}, "engine"),
        ("sum", {"engine_kwargs": {}}, "engine_kwargs"),
        ("min", {"engine": "numba"}, "engine"),
        ("std", {"engine": "numba"}, "engine"),
        ("nunique", {"dropna": False}, "dropna"),
        ("quantile", {"interpolation": "lower"}, "interpolation"),
        ("quantile", {"q": [0.1, 0.9]}, "single quantile"),
        ("any", {"skipna": False}, "skipna"),
        ("all", {"skipna": False}, "skipna"),
    ],
)
def test_a_declared_reduction_argument_that_is_not_implemented_refuses(
    firepanda: ModuleType, call: str, arguments: dict[str, Any], expected: str
) -> None:
    """Every one of them, on the frame's group by and on the column's alike."""
    grouped = firepanda.DataFrame(DATA).groupby("k")
    with pytest.raises(NotImplementedError, match=expected):
        getattr(grouped, call)(**arguments)
    with pytest.raises(NotImplementedError, match=expected):
        getattr(grouped["v"], call)(**arguments)


@needs_pandas
def test_a_grouped_product_of_a_group_with_nothing_in_it_is_one(
    firepanda: ModuleType,
) -> None:
    """The one row where a product and a sum part company.

    A missing value is a zero underneath, which is what a sum wants it to be and
    is the one number a product must not read, so a group of nothing but gaps
    would come out zero if the product were written the way the sum is. pandas
    answers one, because a product over no values is one, and so does this.
    """
    import pandas as pd

    data = {"k": ["a", "a", "b", "b"], "v": [2.0, 3.0, None, None]}
    mine = firepanda.DataFrame(data).groupby("k").prod()
    theirs = pd.DataFrame(data).groupby("k").prod()
    assert mine["v"].tolist() == theirs["v"].tolist() == [6.0, 1.0]


@needs_pandas
def test_a_grouped_truth_reads_a_zero_as_false_and_a_gap_as_neither(
    firepanda: ModuleType,
) -> None:
    """Three groups that separate the three answers a truth reduction can give.

    Group `a` holds a number and a zero, `b` holds nothing but gaps, and `c`
    holds two numbers. A gap is neither true nor false and is stepped over, so
    `b` falls back on the identity of whichever question was asked, which is
    False for `any` and True for `all`.
    """
    import pandas as pd

    data = {"k": ["a", "a", "b", "b", "c", "c"], "v": [0.0, 1.0, None, None, 2.0, 3.0]}
    for name, expected in (("any", [True, False, True]), ("all", [False, True, True])):
        mine = getattr(firepanda.DataFrame(data).groupby("k"), name)()
        theirs = getattr(pd.DataFrame(data).groupby("k"), name)()
        assert mine["v"].tolist() == theirs["v"].tolist() == expected


@needs_pandas
def test_a_grouped_truth_asks_a_text_column_whether_it_is_empty(
    firepanda: ModuleType,
) -> None:
    """Which is the rule Python has for a string and the one pandas keeps.

    `any` and `all` are the only two of the eighteen that read a column of words
    and answer a number, so they are the only two that go over a whole frame
    without the text column having to be taken out of it first.
    """
    import pandas as pd

    data = {"k": ["a", "a", "b", "b"], "t": ["oslo", "", "", ""]}
    mine = firepanda.DataFrame(data).groupby("k").any()
    theirs = pd.DataFrame(data).groupby("k").any()
    assert mine["t"].tolist() == theirs["t"].tolist() == [True, False]


@needs_pandas
def test_a_grouped_product_of_words_is_refused(firepanda: ModuleType) -> None:
    """Because multiplying two strings together is not an operation in pandas
    either, and the refusal is pandas' `TypeError` in pandas' words."""
    import pandas as pd

    data = {"k": ["a", "a"], "t": ["oslo", "lima"]}
    with pytest.raises(TypeError) as ours:
        firepanda.DataFrame(data).groupby("k").prod()
    with pytest.raises(TypeError) as theirs:
        pd.DataFrame(data).groupby("k").prod()
    assert str(ours.value) == str(theirs.value)


@needs_pandas
@pytest.mark.parametrize(
    ("call", "arguments"),
    [
        ("prod", {"min_count": 0}),
        ("prod", {"min_count": -1}),
        ("min", {"min_count": -1}),
        ("min", {"min_count": 0}),
        ("sum", {"min_count": 2}),
        ("sum", {"skipna": False}),
        ("median", {"skipna": False}),
        ("first", {"skipna": False}),
        ("last", {"skipna": False}),
        ("sum", {"numeric_only": True}),
        ("mean", {"numeric_only": True}),
    ],
)
def test_skipna_min_count_and_numeric_only_answer_as_pandas(
    firepanda: ModuleType, call: str, arguments: dict[str, Any]
) -> None:
    """A group with a gap or with fewer values than `min_count` answers NaN, as in pandas."""
    import pandas as pd

    data = {"k": ["a", "a", "b", "b", "c"], "v": [1.0, None, 3.0, 4.0, 5.0], "n": [1, 2, 3, 4, 5]}
    mine = getattr(firepanda.DataFrame(data).groupby("k"), call)(**arguments)
    theirs = getattr(pd.DataFrame(data).groupby("k"), call)(**arguments)
    assert repr(mine) == repr(theirs)
    mine = getattr(firepanda.DataFrame(data).groupby("k")["v"], call)(**arguments)
    theirs = getattr(pd.DataFrame(data).groupby("k")["v"], call)(**arguments)
    assert repr(mine) == repr(theirs)


@needs_pandas
@pytest.mark.parametrize(
    "build",
    [
        lambda g: g.first(),
        lambda g: g.last(),
        lambda g: g.first(skipna=False),
        lambda g: g["c"].last(skipna=False),
        lambda g: g["c"].min(),
        lambda g: g["o"].max(),
        lambda g: g["o"].min(),
    ],
)
def test_a_picked_category_stays_a_category(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """The first, the last and the extremes of a category answer categories, as in pandas.

    Text beside it is blanked by a first that may not skip, and the extremes of a
    list with no order are refused with pandas' words.
    """
    import pandas as pd

    def grouped(lib: ModuleType) -> Any:
        c = lib.Series(["u", None, "v", "w"], dtype="category")
        frame = lib.DataFrame({"k": ["a", "a", "b", "b"], "s": [None, "p", "q", "r"], "c": c})
        return frame.assign(o=c.cat.as_ordered()).groupby("k")

    def outcome(lib: ModuleType) -> str:
        try:
            return repr(build(grouped(lib)))
        except TypeError as error:
            return f"TypeError: {error}"

    assert outcome(firepanda) == outcome(pd)


def test_a_grouping_computes_nothing_until_it_is_reduced(firepanda: ModuleType) -> None:
    """Which is pandas' arrangement and is why the object holds a plan and a frame.

    The class is checked by name because it is reached the way pandas' one is,
    off the frame rather than off the package. `pandas.DataFrameGroupBy` does not
    exist either, and both libraries put the name one import deeper.
    """
    grouped = firepanda.DataFrame(DATA).groupby("k")
    assert type(grouped).__name__ == "DataFrameGroupBy"
    assert type(grouped["v"]).__name__ == "SeriesGroupBy"
    assert not hasattr(grouped, "_inner")
