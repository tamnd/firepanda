"""The flags that change what a reduction answers rather than how it computes.

`skipna=False`, `min_count` and `nunique(dropna=False)` on a column and on
a frame, and `dropna(how="all")` on a frame. Each one was refused by name until this file,
and each one is a rule applied to the answer, so the tests are about the edges
where the rule decides: a column with one gap, a column with none, an integer
column, an empty one, and a floor exactly at the count.

Everything is compared against a running pandas, for the reason
`test_reductions.py` gives.
"""

from __future__ import annotations

import importlib.util
import math
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

GAPPY = [1.0, None, 3.0]
WHOLE = [1, 2, 3]


def same(mine: Any, theirs: Any) -> bool:
    """Equal, where two NaNs count as the same answer."""
    if isinstance(theirs, float) and math.isnan(theirs):
        return isinstance(mine, float) and math.isnan(mine)
    return mine == pytest.approx(theirs)


@pytest.mark.parametrize(
    "name", ["sum", "mean", "min", "max", "std", "var", "prod", "median", "sem", "skew"]
)
@pytest.mark.parametrize("values", [GAPPY, WHOLE], ids=["gappy", "whole"])
def test_skipna_false_answers_what_pandas_answers(
    firepanda: ModuleType, name: str, values: list[Any]
) -> None:
    """NaN when a value is missing, and the ordinary answer when none is."""
    import pandas as pd

    mine = getattr(firepanda.Series(values), name)(skipna=False)
    theirs = getattr(pd.Series(values), name)(skipna=False)
    assert same(mine, theirs)


def test_skipna_false_on_text_is_nan_as_well(firepanda: ModuleType) -> None:
    """The largest of some words and a gap is not a word."""
    import pandas as pd

    words = ["a", None, "c"]
    assert same(firepanda.Series(words).max(skipna=False), pd.Series(words).max(skipna=False))


@pytest.mark.parametrize("floor", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("values", [GAPPY, WHOLE, []], ids=["gappy", "whole", "empty"])
def test_min_count_is_a_floor_on_the_values_present(
    firepanda: ModuleType, floor: int, values: list[Any]
) -> None:
    """Below the floor is NaN, on an integer column too, and at it is the sum."""
    import pandas as pd

    for name in ("sum", "prod"):
        mine = getattr(firepanda.Series(values, dtype="float64"), name)(min_count=floor)
        theirs = getattr(pd.Series(values, dtype="float64"), name)(min_count=floor)
        assert same(mine, theirs), (name, floor)
    mine = firepanda.Series(WHOLE).sum(min_count=floor)
    theirs = pd.Series(WHOLE).sum(min_count=floor)
    assert same(mine, theirs)


def test_a_reduction_that_cannot_read_the_column_still_says_so(firepanda: ModuleType) -> None:
    """The flag is applied after the reduction, so its error comes first."""
    with pytest.raises(TypeError):
        firepanda.Series(["a", None, "c"]).mean(skipna=False)


@pytest.mark.parametrize(
    "values",
    [[1.0, None, 3.0], ["a", None, "c"], [None, None], [1.0, 2.0, 2.0], []],
    ids=["float", "text", "all-missing", "whole", "empty"],
)
def test_nunique_counts_a_missing_value_once(firepanda: ModuleType, values: list[Any]) -> None:
    """However many gaps there are, they are one more distinct value."""
    import pandas as pd

    dtype = "str" if values and isinstance(values[0], str) else "float64"
    for dropna in (True, False):
        mine = firepanda.Series(values, dtype=dtype).nunique(dropna=dropna)
        theirs = pd.Series(values, dtype=dtype).nunique(dropna=dropna)
        assert mine == theirs, dropna


def test_a_frame_nunique_counts_the_gap_per_column(firepanda: ModuleType) -> None:
    """Only the columns that have one get the extra count."""
    import pandas as pd

    data = {"a": [1.0, None, 1.0], "b": [1.0, 2.0, 3.0], "c": ["x", None, None]}
    mine = firepanda.DataFrame(data).nunique(dropna=False)
    theirs = pd.DataFrame(data).nunique(dropna=False)
    assert mine.tolist() == theirs.tolist()
    assert list(mine.index) == list(theirs.index)
    assert mine.name == theirs.name


MIXED = {"a": [1, 2, 3], "b": [1.0, None, 3.0]}
INTEGERS = {"a": [1, 2], "b": [5, 6]}


@pytest.mark.parametrize("name", ["sum", "prod", "mean", "min", "max", "std", "var", "median"])
@pytest.mark.parametrize("data", [MIXED, INTEGERS], ids=["mixed", "integers"])
def test_a_frame_with_skipna_false_answers_nan_per_column(
    firepanda: ModuleType, name: str, data: dict[str, list[Any]]
) -> None:
    """Only a column with a gap is NaN, and the answer widens only when one is."""
    import pandas as pd

    mine = getattr(firepanda.DataFrame(data), name)(skipna=False)
    theirs = getattr(pd.DataFrame(data), name)(skipna=False)
    assert mine.dtype == str(theirs.dtype)
    assert all(same(a, b) for a, b in zip(mine.tolist(), theirs.tolist(), strict=True))


@pytest.mark.parametrize("name", ["sum", "prod"])
@pytest.mark.parametrize("floor", [1, 2, 3])
@pytest.mark.parametrize("data", [MIXED, INTEGERS], ids=["mixed", "integers"])
def test_a_frame_min_count_is_a_floor_per_column(
    firepanda: ModuleType, name: str, floor: int, data: dict[str, list[Any]]
) -> None:
    """A column with fewer values than the floor is NaN."""
    import pandas as pd

    mine = getattr(firepanda.DataFrame(data), name)(min_count=floor)
    theirs = getattr(pd.DataFrame(data), name)(min_count=floor)
    assert mine.dtype == str(theirs.dtype)
    assert all(same(a, b) for a, b in zip(mine.tolist(), theirs.tolist(), strict=True))


@pytest.mark.parametrize("flags", [{"skipna": False}, {"min_count": 2}, {"min_count": 5}], ids=str)
def test_the_whole_frame_takes_the_flags_too(firepanda: ModuleType, flags: dict[str, Any]) -> None:
    """With `axis=None` the floor counts every cell rather than each column."""
    import pandas as pd

    data = {"a": [1.0, None], "b": [None, 2.0]}
    mine = firepanda.DataFrame(data).sum(axis=None, **flags)
    theirs = pd.DataFrame(data).sum(axis=None, **flags)
    assert same(mine, float(theirs))


FRAME = {
    "a": [1.0, None, None, 4.0],
    "b": [None, None, 3.0, 4.0],
    "c": ["x", None, None, "y"],
}


@pytest.mark.parametrize("subset", [None, ["a"], ["a", "b"], "c"])
def test_dropna_how_all_drops_only_the_empty_rows(firepanda: ModuleType, subset: Any) -> None:
    """A row survives while any column looked at has something in it."""
    import pandas as pd

    for how in ("any", "all"):
        mine = firepanda.DataFrame(FRAME).dropna(how=how, subset=subset)
        theirs = pd.DataFrame(FRAME).dropna(how=how, subset=subset)
        assert list(mine.index) == list(theirs.index), (how, subset)
        assert mine.shape == theirs.shape


def test_dropna_how_all_on_a_column_of_nothing_leaves_no_rows(firepanda: ModuleType) -> None:
    """The shape pandas gives, which keeps the column."""
    import pandas as pd

    data = {"a": [None, None]}
    shaped = pd.DataFrame(data, dtype="float64").dropna(how="all").shape
    assert firepanda.DataFrame(data, dtype="float64").dropna(how="all").shape == shaped


def test_dropna_how_all_in_place(firepanda: ModuleType) -> None:
    """The frame itself loses the rows and the call answers None."""
    frame = firepanda.DataFrame(FRAME)
    assert frame.dropna(how="all", inplace=True) is None
    assert list(frame.index) == [0, 2, 3]


def test_dropna_refuses_a_how_it_does_not_know(firepanda: ModuleType) -> None:
    """With pandas' message."""
    with pytest.raises(ValueError, match="invalid how option: bogus"):
        firepanda.DataFrame(FRAME).dropna(how="bogus")


def test_dropna_refuses_how_and_thresh_together(firepanda: ModuleType) -> None:
    """Which pandas checks before it looks at either one."""
    with pytest.raises(TypeError, match="both the how and thresh"):
        firepanda.DataFrame(FRAME).dropna(how="all", thresh=1)


@pytest.mark.parametrize("name", ["sum", "prod", "min", "max"])
@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2], "b": [float("inf"), -float("inf")]},
        {"a": [1.0, 2.0], "b": [None, None]},
        {"a": [None, None], "b": [None, None]},
    ],
    ids=["inf-minus-inf", "one-empty-column", "all-empty"],
)
def test_the_whole_frame_keeps_a_nan_the_arithmetic_made(
    firepanda: ModuleType, name: str, data: dict[str, list[Any]]
) -> None:
    """A NaN a column's own total made is the answer, and an empty column is skipped."""
    import pandas as pd

    mine = getattr(firepanda.DataFrame(data, dtype="float64"), name)(axis=None)
    theirs = getattr(pd.DataFrame(data, dtype="float64"), name)(axis=None)
    assert same(mine, float(theirs))
