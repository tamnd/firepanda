"""A zone given as a `tzinfo` object rather than a name, compared with pandas.

Each case runs in both libraries and compares the text of the answer, which
carries the zone in its type and in each moment. A fixed offset is named after
UTC, as pandas names it.
"""

from __future__ import annotations

import datetime as dt
import zoneinfo
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")

EAST = dt.timezone(dt.timedelta(hours=7))
WEST = dt.timezone(dt.timedelta(hours=-3, minutes=-30))
TOKYO = zoneinfo.ZoneInfo("Asia/Tokyo")


def moments(lib: ModuleType) -> Any:
    return lib.Series(lib.to_datetime(["2024-01-01 12:00", None]), name="d")


def outcome(call: Callable[[ModuleType], Any], lib: ModuleType) -> Any:
    got = call(lib)
    if isinstance(got, (fp.Series, pd.Series)):
        return str(got.dtype), str(got.tolist())
    return str(got)


CASES: list[Callable[[ModuleType], Any]] = [
    lambda lib: moments(lib).dt.tz_localize(dt.UTC),
    lambda lib: moments(lib).dt.tz_localize(TOKYO),
    lambda lib: moments(lib).dt.tz_localize(EAST),
    lambda lib: moments(lib).dt.tz_localize(WEST),
    lambda lib: lib.DatetimeIndex(["2024-01-01 12:00"]).tz_localize(TOKYO),
    lambda lib: lib.DatetimeIndex(["2024-01-01 12:00"]).tz_localize(EAST),
    lambda lib: lib.date_range("2024-01-01", periods=2, freq="D").tz_localize(dt.UTC),
    lambda lib: moments(lib).dt.tz_localize("UTC").dt.tz_convert(WEST),
    lambda lib: lib.Series([dt.datetime(2024, 1, 1, tzinfo=WEST)]),
    lambda lib: lib.date_range("2024-01-01", periods=2, tz=EAST),
    lambda lib: lib.date_range(dt.datetime(2024, 1, 1, tzinfo=WEST), periods=2),
]


@pytest.mark.parametrize("call", CASES)
def test_a_tzinfo_zone_is_pandas_zone(call: Callable[[ModuleType], Any]) -> None:
    assert outcome(call, fp) == outcome(call, pd)
