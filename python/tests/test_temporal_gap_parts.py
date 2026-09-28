"""A part read from a column of instants or spans with a gap in it, against pandas.

numpy has no integer that can hold a NaN, so pandas answers a float for a whole
number read from a column with a gap, and false for a flag. Without a gap the
number stays an integer.
"""

from __future__ import annotations

import importlib.util
import math
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

FIELDS = [
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "microsecond",
    "nanosecond",
    "dayofweek",
    "weekday",
    "dayofyear",
    "quarter",
    "days_in_month",
    "is_month_start",
    "is_month_end",
    "is_leap_year",
]


def spelled(values: list[Any]) -> list[Any]:
    """The values with a NaN named, so a NaN compares equal to a NaN."""
    return ["nan" if isinstance(v, float) and math.isnan(v) else v for v in values]


def instants(m: ModuleType, gap: bool) -> list[Any]:
    rows = [m.Timestamp("2020-01-31 03:04:05"), m.Timestamp("2021-03-01 05:06")]
    return [*rows, None] if gap else rows


@needs_pandas
@pytest.mark.parametrize("gap", [True, False])
@pytest.mark.parametrize("field", FIELDS)
def test_a_part_of_a_column_is_typed_as_in_pandas(
    firepanda: ModuleType, field: str, gap: bool
) -> None:
    """`Series.dt`, with and without a gap."""
    import pandas as pd

    mine = getattr(firepanda.Series(instants(firepanda, gap)).dt, field)
    them = getattr(pd.Series(instants(pd, gap)).dt, field)
    assert str(mine.dtype) == str(them.dtype)
    assert spelled(mine.tolist()) == spelled(them.tolist())


@needs_pandas
@pytest.mark.parametrize("field", FIELDS)
def test_a_part_of_an_index_is_typed_as_in_pandas(firepanda: ModuleType, field: str) -> None:
    """`DatetimeIndex`, with a gap."""
    import pandas as pd

    mine = getattr(firepanda.DatetimeIndex(instants(firepanda, True)), field)
    them = getattr(pd.DatetimeIndex(instants(pd, True)), field)
    assert str(mine.dtype) == str(them.dtype)
    assert spelled(list(mine)) == spelled([v.item() if hasattr(v, "item") else v for v in them])


@needs_pandas
@pytest.mark.parametrize("field", ["days", "seconds", "microseconds", "nanoseconds"])
def test_a_part_of_spans_is_typed_as_in_pandas(firepanda: ModuleType, field: str) -> None:
    """`Series.dt` and `TimedeltaIndex` on spans with a gap."""
    import pandas as pd

    rows = ["1 day 2h", None]
    mine = getattr(firepanda.Series([firepanda.Timedelta(rows[0]), None]).dt, field)
    them = getattr(pd.Series([pd.Timedelta(rows[0]), None]).dt, field)
    assert str(mine.dtype) == str(them.dtype)
    assert spelled(mine.tolist()) == spelled(them.tolist())
