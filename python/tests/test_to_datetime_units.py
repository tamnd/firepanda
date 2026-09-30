"""`to_datetime` with a unit coarser than a second, or on floats, against pandas.

Arrow counts time in seconds and finer, so days, hours and minutes are turned
into seconds before the column is read, and pandas holds those at seconds too.
Floats that are all whole are read as integers are. A float with a fraction makes
pandas count every row in nanoseconds, and a gap stays a gap either way. The last
case is a format that parses nothing under `errors="coerce"`, which pandas holds
at seconds whatever the format was.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def shown(column: Any) -> tuple[str, list[str]]:
    """The dtype and the cells of a column, as text either library writes alike."""
    return str(column.dtype), [str(cell) for cell in column.tolist()]


@pytest.mark.parametrize("unit", ["D", "h", "m", "s", "ms", "us", "ns"])
@pytest.mark.parametrize(
    "data",
    [[1, 2, 0, -3], [1.5, 2.0, 0.0, -0.25], [0.3, -1.7, 0.1], [1.0, None, 3.0], [2.5, None]],
    ids=[
        "integers",
        "fractions",
        "inexact fractions",
        "whole floats with a gap",
        "fractions with a gap",
    ],
)
def test_a_count_in_a_unit_is_the_pandas_instant(
    firepanda: ModuleType, data: list[Any], unit: str
) -> None:
    import pandas as pd

    mine = firepanda.to_datetime(firepanda.Series(data), unit=unit)
    theirs = pd.to_datetime(pd.Series(data), unit=unit)
    assert shown(mine) == shown(theirs)


@pytest.mark.parametrize("unit", ["D", "s", "ms"])
@pytest.mark.parametrize("data", [[1, 2], [1.5, None]], ids=["integers", "a fraction and a gap"])
def test_utc_marks_the_counts_as_utc(firepanda: ModuleType, data: list[Any], unit: str) -> None:
    import pandas as pd

    mine = firepanda.to_datetime(firepanda.Series(data), unit=unit, utc=True)
    theirs = pd.to_datetime(pd.Series(data), unit=unit, utc=True)
    assert shown(mine) == shown(theirs)


def test_a_format_that_parses_nothing_is_held_at_seconds(firepanda: ModuleType) -> None:
    import pandas as pd

    rows = ["x", "y"]
    mine = firepanda.to_datetime(firepanda.Series(rows), errors="coerce", format="%Y-%m-%d")
    theirs = pd.to_datetime(pd.Series(rows), errors="coerce", format="%Y-%m-%d")
    assert str(mine.dtype) == str(theirs.dtype) == "datetime64[s]"
    assert mine.isna().tolist() == [True, True]
