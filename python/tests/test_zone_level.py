"""`tz_localize` and `tz_convert` on one level of a MultiIndex.

pandas takes `level=` to change the zone of that level alone and leaves the
other levels as they were. A level that is not there, or one that does not
hold dates, is refused in pandas' words. Each test here runs the same code on
both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def framed(lib: Any) -> Any:
    times = lib.date_range("2024-01-01", periods=3, freq="h")
    labels = lib.MultiIndex.from_arrays([["a", "b", "c"], times], names=["k", "t"])
    return lib.DataFrame({"v": [1, 2, 3]}, index=labels)


BUILDS = {
    "frame by name": lambda lib: framed(lib).tz_localize("UTC", level="t"),
    "frame by number": lambda lib: framed(lib).tz_localize("UTC", level=1),
    "column": lambda lib: framed(lib)["v"].tz_localize("Asia/Tokyo", level="t"),
    "converted": lambda lib: (
        framed(lib).tz_localize("UTC", level="t").tz_convert("US/Eastern", level="t")
    ),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_a_zone_on_one_level_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_a_level_that_is_not_there_is_refused(firepanda: Any) -> None:
    with pytest.raises(ValueError, match="The level z is not valid"):
        framed(firepanda).tz_localize("UTC", level="z")


def test_a_level_without_dates_is_refused(firepanda: Any) -> None:
    with pytest.raises(TypeError, match="not a valid DatetimeIndex"):
        framed(firepanda).tz_localize("UTC", level="k")
