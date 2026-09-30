"""Parameters that used to refuse and now answer as pandas does, checked against pandas.

Each of these was a declared argument held at its default: a fill value for the
rows a shift opens, a scan that does not skip gaps, a fill that stops after a
number of gaps, a sort of the columns rather than the rows, and rows numbered
again after a sort. Every test builds the same thing in both libraries and
compares what comes back.
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

FLOATS = [1.0, None, 3.0, None, 5.0]


def same(got: list[Any], want: list[Any]) -> bool:
    """Two lists of cells as equal, with NaN equal to NaN."""
    if len(got) != len(want):
        return False
    for mine, theirs in zip(got, want, strict=True):
        both_gaps = isinstance(mine, float) and isinstance(theirs, float)
        if both_gaps and math.isnan(mine) and math.isnan(theirs):
            continue
        if mine != theirs:
            return False
    return True


def frames(firepanda: ModuleType, data: dict[str, list[Any]], **kwargs: Any) -> tuple[Any, Any]:
    """The same frame in both libraries."""
    import pandas as pd

    return firepanda.DataFrame(data, **kwargs), pd.DataFrame(data, **kwargs)


@pytest.mark.parametrize(("periods", "fill"), [(1, 0), (-2, 9), (5, 0), (0, 7), (2, 0.5)])
def test_a_shift_fills_the_rows_it_opens(firepanda: ModuleType, periods: int, fill: Any) -> None:
    import pandas as pd

    mine = firepanda.Series([1, 2, 3, 4], name="v").shift(periods, fill_value=fill)
    theirs = pd.Series([1, 2, 3, 4], name="v").shift(periods, fill_value=fill)
    assert mine.tolist() == theirs.tolist()
    assert str(mine.dtype) == str(theirs.dtype)
    assert mine.name == theirs.name


def test_a_shift_fill_leaves_the_gaps_that_were_there(firepanda: ModuleType) -> None:
    mine, theirs = frames(firepanda, {"a": [1, 2, 3], "c": [1.0, None, 2.0]})
    got = mine.shift(1, fill_value=0).to_dict("list")
    want = theirs.shift(1, fill_value=0).to_dict("list")
    assert got.keys() == want.keys()
    for name in want:
        assert same(got[name], want[name]), name


@pytest.mark.parametrize("name", ["cumsum", "cumprod", "cummax", "cummin"])
def test_a_scan_that_does_not_skip_is_a_gap_after_the_first_gap(
    firepanda: ModuleType, name: str
) -> None:
    import pandas as pd

    got = getattr(firepanda.Series(FLOATS), name)(skipna=False).tolist()
    want = getattr(pd.Series(FLOATS), name)(skipna=False).tolist()
    assert same(got, want)


def test_a_frame_scan_takes_skipna_and_numeric_only(firepanda: ModuleType) -> None:
    data = {"a": [1, 2, 3], "c": [1.0, None, 2.0], "s": ["x", "y", "z"]}
    mine, theirs = frames(firepanda, data)
    got = mine[["a", "c"]].cumsum(skipna=False).to_dict("list")
    want = theirs[["a", "c"]].cumsum(skipna=False).to_dict("list")
    assert all(same(got[name], want[name]) for name in want)
    got = mine.cumsum(numeric_only=True).to_dict("list")
    want = theirs.cumsum(numeric_only=True).to_dict("list")
    assert got.keys() == want.keys()
    assert all(same(got[name], want[name]) for name in want)


@pytest.mark.parametrize("limit", [1, 2, 5])
def test_a_fill_stops_after_the_limit(firepanda: ModuleType, limit: int) -> None:
    import pandas as pd

    got = firepanda.Series(FLOATS).fillna(0.0, limit=limit).tolist()
    assert same(got, pd.Series(FLOATS).fillna(0.0, limit=limit).tolist())
    mine, theirs = frames(firepanda, {"a": FLOATS, "b": FLOATS[::-1]})
    got = mine.fillna({"a": 7.0}, limit=limit).to_dict("list")
    want = theirs.fillna({"a": 7.0}, limit=limit).to_dict("list")
    assert all(same(got[name], want[name]) for name in want)


def test_a_fill_limit_across_the_rows_is_still_refused(firepanda: ModuleType) -> None:
    with pytest.raises(NotImplementedError, match="limit="):
        firepanda.DataFrame({"a": FLOATS}).fillna(0.0, axis=1, limit=1)


@pytest.mark.parametrize("ascending", [True, False])
def test_sort_index_sorts_the_columns(firepanda: ModuleType, ascending: bool) -> None:
    mine, theirs = frames(firepanda, {"b": [1], "a": [2], "c": [3]})
    got = mine.sort_index(axis=1, ascending=ascending)
    want = theirs.sort_index(axis=1, ascending=ascending)
    assert list(got.columns) == list(want.columns)
    assert got.to_dict("list") == want.to_dict("list")


def test_sort_index_numbers_the_rows_again(firepanda: ModuleType) -> None:
    mine, theirs = frames(firepanda, {"v": [1, 2, 3]}, index=[3, 1, 2])
    got = mine.sort_index(ignore_index=True)
    want = theirs.sort_index(ignore_index=True)
    assert got.to_dict("list") == want.to_dict("list")
    assert list(got.index) == list(want.index)


def test_a_pattern_replaces_inside_every_text_column(firepanda: ModuleType) -> None:
    data = {"b": ["ax", "bx"], "a": [1, 2], "c": ["xa", None]}
    mine, theirs = frames(firepanda, data)
    got = mine.replace("x", "Y", regex=True).to_dict("list")
    want = theirs.replace("x", "Y", regex=True).to_dict("list")
    assert all(same(got[name], want[name]) for name in want)
