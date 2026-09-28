"""Names, text and Python values read from a `DatetimeIndex` with a gap, against pandas.

The name of a day or a month and `strftime` are text, and a gap in the labels
is a gap in the text, which pandas shows as NaN. The time of day and the Python
datetime of a gap are `NaT`.
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


def labels(m: ModuleType) -> Any:
    rows = [m.Timestamp("2020-01-31 03:04"), None, m.Timestamp("2021-06-01")]
    return m.DatetimeIndex(rows).tz_localize("Asia/Tokyo")


def spelled(value: Any) -> Any:
    """A value named so two libraries' answers compare, with `NaT` and NaN as words."""
    if type(value).__name__ == "NaTType":
        return "NaT"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return repr(value)


CALLS: dict[str, Callable[[ModuleType], Any]] = {
    "day_name": lambda m: labels(m).day_name(),
    "month_name": lambda m: labels(m).month_name(),
    "strftime": lambda m: labels(m).strftime("%Y-%m-%d %H"),
    "time": lambda m: labels(m).time,
    "timetz": lambda m: labels(m).timetz,
    "to_pydatetime": lambda m: labels(m).to_pydatetime(),
}


@needs_pandas
@pytest.mark.parametrize("name", list(CALLS))
def test_a_gap_reads_out_as_in_pandas(firepanda: ModuleType, name: str) -> None:
    """Every value matches, and the gap is NaN in text and `NaT` otherwise."""
    import pandas as pd

    mine = CALLS[name](firepanda)
    them = CALLS[name](pd)
    assert [spelled(v) for v in mine] == [spelled(v) for v in them]
