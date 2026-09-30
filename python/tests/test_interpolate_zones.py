"""`interpolate` and `describe` over zoned instants, intervals and periods.

pandas draws the line through a zoned column's UTC instants and hands the
answer back in the column's zone. It passes an interval column with no gaps
through untouched and refuses one with gaps, refuses a period column outright,
and describes an interval column by counting its values the way it counts text.
Each test runs the same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

NAN = float("nan")


def zoned(lib: Any) -> Any:
    days = ["2024-01-01", None, "2024-01-04", None, None, "2024-01-10"]
    return lib.Series(lib.to_datetime(days).tz_localize("UTC"))


def gapped(lib: Any) -> Any:
    return lib.Series(lib.IntervalIndex.from_tuples([(0, 1), None, (2, 3)]))


BUILDS = {
    "zoned": lambda lib: zoned(lib).interpolate(),
    "zoned limit": lambda lib: zoned(lib).interpolate(limit=1),
    "zoned backward": lambda lib: zoned(lib).interpolate(limit_direction="backward", limit=1),
    "zoned frame": lambda lib: lib.DataFrame(
        {"d": zoned(lib).dt.tz_convert("Asia/Tokyo"), "v": [1.0, NAN, 3.0, NAN, 5.0, 6.0]}
    ).interpolate(),
    "zoned whole": lambda lib: zoned(lib).dropna().interpolate(),
    "intervals whole": lambda lib: lib.DataFrame(
        {"i": lib.interval_range(0, 3), "v": [1.0, NAN, 3.0]}
    ).interpolate(),
    "describe intervals": lambda lib: gapped(lib).describe(),
    "describe interval frame": lambda lib: lib.DataFrame({"i": gapped(lib)}).describe(),
    "describe all": lambda lib: lib.DataFrame({"i": gapped(lib), "v": [1, 2, 3]}).describe(
        include="all"
    ),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_interpolate_zones_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_intervals_with_gaps_and_periods_are_refused(firepanda: Any) -> None:
    with pytest.raises(NotImplementedError, match="IntervalArray does not implement"):
        gapped(firepanda).interpolate()
    periods = firepanda.Series(firepanda.period_range("2024-01", periods=3, freq="M"))
    with pytest.raises(NotImplementedError, match="PeriodArray does not implement"):
        periods.interpolate()
