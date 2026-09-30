"""Resampling timestamps that carry a zone.

pandas lays days and calendar steps on the zone's own clock, so a day around
a change of the clocks is 23 or 25 hours long, and lays a fixed step on the
instants themselves, counted from midnight of the first day on the zone's
clock, or from the epoch on that clock. The labels carry the zone. Each test
here runs the same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def spring(lib: Any) -> Any:
    stamps = lib.date_range("2024-03-09 20:00", periods=10, freq="3h", tz="US/Eastern")
    return lib.Series(range(10), index=stamps)


def autumn(lib: Any) -> Any:
    stamps = lib.date_range("2024-11-02 22:00", periods=8, freq="2h", tz="US/Eastern")
    return lib.Series(range(8), index=stamps)


def half_hour(lib: Any) -> Any:
    stamps = lib.date_range("2024-01-01 05:00", periods=3, freq="7h", tz="Asia/Kolkata")
    return lib.Series([1.0, 2.0, 3.0], index=stamps)


BUILDS = {
    "spring day": lambda lib: spring(lib).resample("D").sum(),
    "spring 6h": lambda lib: spring(lib).resample("6h").sum(),
    "spring 5h": lambda lib: spring(lib).resample("5h").mean(),
    "autumn day": lambda lib: autumn(lib).resample("D").sum(),
    "autumn 4h": lambda lib: autumn(lib).resample("4h").sum(),
    "autumn 3h": lambda lib: autumn(lib).resample("3h").count(),
    "half hour 4h": lambda lib: half_hour(lib).resample("4h").sum(),
    "half hour day": lambda lib: half_hour(lib).resample("D").sum(),
    "month end": lambda lib: spring(lib).resample("ME").sum(),
    "week": lambda lib: autumn(lib).resample("W").sum(),
    "start": lambda lib: half_hour(lib).resample("4h", origin="start").sum(),
    "epoch": lambda lib: half_hour(lib).resample("4h", origin="epoch").sum(),
    "right": lambda lib: spring(lib).resample("6h", closed="right", label="right").sum(),
    "asfreq": lambda lib: half_hour(lib).resample("7h").asfreq(),
    "ffill": lambda lib: half_hour(lib).resample("3h").ffill(),
    "transform": lambda lib: spring(lib).resample("D").transform("sum"),
    "binner": lambda lib: spring(lib).resample("D").binner,
    "frame": lambda lib: spring(lib).to_frame("v").resample("12h").max(),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_zoned_resample_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))
