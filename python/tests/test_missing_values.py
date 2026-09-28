"""How a missing value is spelled when it is read out of a column, against pandas.

pandas has no one word for a gap. A float column holds a NaN, an integer column
with a gap is a float column, and a text column in pandas 3 carries its gaps as
NaN too, so all three hand out `nan`. A column of flags with a gap is an object
column and hands out None. firepanda keeps every gap in a validity bitmap and
hands it out the way pandas does for the column's type, through `tolist`,
iteration and reading one cell.
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

COLUMNS: list[list[Any]] = [
    [1.5, None, 3.5],
    [1, None, 3],
    ["a", None, "c"],
    [True, None, False],
]

READS: list[Callable[[Any], Any]] = [
    lambda s: s.tolist(),
    lambda s: s.to_list(),
    lambda s: list(s),
    lambda s: [s[1]],
    lambda s: [s.iloc[1]],
    lambda s: [s.iat[1]],
    lambda s: [s.loc[1]],
    lambda s: [s.at[1]],
    lambda s: [value for _, value in s.items()],
]


def spelled(values: list[Any]) -> list[Any]:
    """The values with a NaN and a None named, so a NaN compares equal to a NaN.

    An integer column with a gap is a float column in pandas and stays an
    integer column here, so the values are compared as numbers, where 1 and 1.0
    are equal.
    """
    out: list[Any] = []
    for value in values:
        if value is None:
            out.append("None")
        elif isinstance(value, float) and math.isnan(value):
            out.append("nan")
        else:
            out.append(value.item() if hasattr(value, "item") else value)
    return out


@needs_pandas
@pytest.mark.parametrize("values", COLUMNS)
@pytest.mark.parametrize("read", READS)
def test_a_gap_reads_out_as_pandas_reads_it(
    firepanda: ModuleType, values: list[Any], read: Callable[[Any], Any]
) -> None:
    """Every door a value leaves a column by, on every type with a gap."""
    import pandas as pd

    assert spelled(read(firepanda.Series(values))) == spelled(read(pd.Series(values)))


@needs_pandas
@pytest.mark.parametrize("values", COLUMNS)
def test_a_gap_in_a_frame_reads_out_as_pandas_reads_it(
    firepanda: ModuleType, values: list[Any]
) -> None:
    """A cell of a frame, by position and by label, and a column of it."""
    import pandas as pd

    mine = firepanda.DataFrame({"v": values})
    them = pd.DataFrame({"v": values})
    assert spelled([mine.iat[1, 0], mine.at[1, "v"]]) == spelled([them.iat[1, 0], them.at[1, "v"]])
    assert spelled(mine["v"].tolist()) == spelled(them["v"].tolist())


@needs_pandas
def test_a_gap_in_an_index_reads_out_as_pandas_reads_it(firepanda: ModuleType) -> None:
    """An index hands its labels out the same way a column does."""
    import pandas as pd

    for values in ([1.5, None], ["a", None]):
        assert spelled(firepanda.Index(values).tolist()) == spelled(pd.Index(values).tolist())


@needs_pandas
def test_a_gap_in_a_category_column_reads_out_as_nan(firepanda: ModuleType) -> None:
    """A category column hands a missing row out as a NaN, and its code is -1."""
    import pandas as pd

    mine = firepanda.Series(["a", None, "b"]).astype("category")
    them = pd.Series(["a", None, "b"]).astype("category")
    assert spelled(mine.tolist()) == spelled(them.tolist())
    assert mine.cat.codes.tolist() == them.cat.codes.tolist()


@needs_pandas
def test_a_nan_among_text_builds_a_text_column_with_a_gap(firepanda: ModuleType) -> None:
    """What `tolist` hands out goes back in, as it does in pandas."""
    import pandas as pd

    rows = ["a", math.nan, "c"]
    mine = firepanda.Series(rows)
    them = pd.Series(rows)
    assert mine.isna().tolist() == them.isna().tolist()
    assert spelled(firepanda.Series(mine.tolist()).tolist()) == spelled(them.tolist())
    frame = firepanda.DataFrame({"s": rows})
    assert frame["s"].isna().tolist() == [False, True, False]


@needs_pandas
def test_a_nan_asked_for_finds_the_gap_in_an_index(firepanda: ModuleType) -> None:
    """`get_indexer` finds a gap for a NaN, and for an index holding one."""
    import pandas as pd

    for make in (lambda m: [2.0, math.nan], lambda m: m.Index([2.0, None])):
        mine = firepanda.Index([None, 2.0]).get_indexer(make(firepanda))
        assert list(mine) == list(pd.Index([None, 2.0]).get_indexer(make(pd)))


def test_a_nan_and_a_hole_both_read_out_as_nan(firepanda: ModuleType) -> None:
    """Both read out as NaN, and both are counted as missing."""
    column = firepanda.Series([1.0, math.nan, None])
    values = column.tolist()
    assert math.isnan(values[1]) and math.isnan(values[2])
    assert column.isna().tolist() == [False, True, True]
    assert column.count() == 1
