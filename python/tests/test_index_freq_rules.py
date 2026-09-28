"""The frequency pandas passes on to the index an operation builds, compared with pandas.

Each case builds a daily, business, hourly or month end index in both libraries,
runs an operation and compares the frequency of the answer as pandas spells it,
or the class and message of a mistake.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")


def days(lib: ModuleType) -> Any:
    return lib.date_range("2024-01-01", periods=5, freq="D")


def plain(lib: ModuleType) -> Any:
    return lib.DatetimeIndex(list(days(lib)))


def workdays(lib: ModuleType) -> Any:
    return lib.date_range("2024-01-01", periods=5, freq="B")


def hours(lib: ModuleType) -> Any:
    return lib.date_range("2024-01-01", periods=5, freq="12h")


def month_ends(lib: ModuleType) -> Any:
    return lib.date_range("2024-01-31", periods=5, freq="ME")


def spans(lib: ModuleType) -> Any:
    return lib.timedelta_range("1h", periods=4, freq="h")


def outcome(call: Callable[[ModuleType], Any], lib: ModuleType) -> Any:
    try:
        return call(lib).freqstr
    except Exception as error:
        return type(error).__name__, str(error)


CASES: list[Callable[[ModuleType], Any]] = [
    lambda lib: days(lib).unique(),
    lambda lib: days(lib)[::-1].unique(),
    lambda lib: plain(lib).unique(),
    lambda lib: days(lib).append(days(lib)).unique(),
    lambda lib: days(lib).drop_duplicates(),
    lambda lib: days(lib).sort_values(),
    lambda lib: days(lib).sort_values(ascending=False),
    lambda lib: days(lib)[::-1].sort_values(),
    lambda lib: days(lib)[::2].sort_values(ascending=False),
    lambda lib: days(lib).sort_values(key=lambda x: -x.day),
    lambda lib: days(lib).sort_values(return_indexer=True)[0],
    lambda lib: plain(lib).sort_values(),
    lambda lib: workdays(lib).sort_values(ascending=False),
    lambda lib: month_ends(lib).sort_values(ascending=False),
    lambda lib: days(lib).take([1, 2]),
    lambda lib: days(lib).take([2, 1]),
    lambda lib: days(lib).take([0, 2, 4]),
    lambda lib: days(lib).take([4, 3, 2, 1, 0]),
    lambda lib: days(lib).take([-1, -2]),
    lambda lib: days(lib).take([1]),
    lambda lib: days(lib).take([]),
    lambda lib: days(lib).take([1, 1]),
    lambda lib: days(lib)[[True] * 5],
    lambda lib: days(lib)[[False, True, True, False, False]],
    lambda lib: days(lib)[[True, False, True, False, True]],
    lambda lib: days(lib)[days(lib).isin(days(lib)[1:3])],
    lambda lib: days(lib).delete(0),
    lambda lib: days(lib).delete(-1),
    lambda lib: days(lib).delete(-5),
    lambda lib: days(lib).delete(1),
    lambda lib: days(lib).delete([0, 1]),
    lambda lib: days(lib).delete([3, 4]),
    lambda lib: days(lib).delete([0, 2]),
    lambda lib: days(lib).delete([0, 4]),
    lambda lib: days(lib).delete([-1]),
    lambda lib: days(lib).delete(slice(0, 2)),
    lambda lib: days(lib).drop(days(lib)[0]),
    lambda lib: days(lib).insert(5, days(lib)[-1] + lib.Timedelta("1D")),
    lambda lib: days(lib).insert(0, days(lib)[0] - lib.Timedelta("1D")),
    lambda lib: days(lib).insert(-5, days(lib)[0] - lib.Timedelta("1D")),
    lambda lib: days(lib).insert(-1, days(lib)[-1] + lib.Timedelta("1D")),
    lambda lib: days(lib).insert(1, days(lib)[0]),
    lambda lib: days(lib).insert(5, days(lib)[-1] + lib.Timedelta("2D")),
    lambda lib: days(lib).insert(5, lib.NaT),
    lambda lib: days(lib)[:0].insert(0, lib.Timestamp("2024-02-01")),
    lambda lib: workdays(lib)[:0].insert(0, lib.Timestamp("2024-02-03")),
    lambda lib: days(lib)[:2].union(days(lib)[1:]),
    lambda lib: days(lib)[:2].union(days(lib)[2:]),
    lambda lib: days(lib)[:2].union(days(lib)[3:]),
    lambda lib: days(lib).union(days(lib)[::2]),
    lambda lib: days(lib)[:2].union(plain(lib)[2:]),
    lambda lib: plain(lib)[:2].union(days(lib)[2:]),
    lambda lib: plain(lib).union(plain(lib)),
    lambda lib: days(lib).union(hours(lib)),
    lambda lib: days(lib).intersection(days(lib)[1:]),
    lambda lib: days(lib).intersection(days(lib)[::2]),
    lambda lib: days(lib).intersection(plain(lib)),
    lambda lib: plain(lib).intersection(days(lib)),
    lambda lib: plain(lib).intersection(plain(lib)),
    lambda lib: days(lib).difference(days(lib)[3:]),
    lambda lib: days(lib).difference(days(lib)[1:4]),
    lambda lib: days(lib).difference(days(lib)[:0]),
    lambda lib: plain(lib).difference(plain(lib)[:1]),
    lambda lib: days(lib).where([True] * 5),
    lambda lib: days(lib).where([True, False, True, True, True]),
    lambda lib: days(lib).putmask([False] * 5, days(lib)[0]),
    lambda lib: days(lib).putmask([True, False, False, False, False], days(lib)[0]),
    lambda lib: days(lib).fillna(days(lib)[0]),
    lambda lib: days(lib).dropna(),
    lambda lib: days(lib).astype("datetime64[s]"),
    lambda lib: days(lib).view(),
    lambda lib: days(lib).repeat(1),
    lambda lib: days(lib).append(days(lib)[:0]),
    lambda lib: days(lib)[:2].append(days(lib)[2:]),
    lambda lib: days(lib)[:2].append([days(lib)[2:4], days(lib)[4:]]),
    lambda lib: days(lib)[:2].append(days(lib)[3:]),
    lambda lib: days(lib)[:0].append(days(lib)),
    lambda lib: days(lib).reindex(days(lib))[0],
    lambda lib: days(lib).reindex(days(lib)[:2])[0],
    lambda lib: days(lib).reindex(list(days(lib)))[0],
    lambda lib: hours(lib).normalize(),
    lambda lib: month_ends(lib).normalize(),
    lambda lib: hours(lib).tz_localize("UTC"),
    lambda lib: hours(lib).tz_localize("America/New_York"),
    lambda lib: hours(lib)[:1].tz_localize("America/New_York"),
    lambda lib: workdays(lib).tz_localize("UTC"),
    lambda lib: hours(lib).tz_localize("UTC").tz_localize(None),
    lambda lib: hours(lib).tz_localize("UTC").tz_convert("Asia/Tokyo"),
    lambda lib: days(lib).tz_localize("UTC").tz_convert("Asia/Tokyo"),
    lambda lib: workdays(lib).tz_localize("UTC").tz_convert("Asia/Tokyo"),
    lambda lib: days(lib) + lib.Timedelta("1h"),
    lambda lib: days(lib) + lib.offsets.Day(1),
    lambda lib: days(lib) - lib.Timedelta("1h"),
    lambda lib: days(lib) + dt.timedelta(hours=1),
    lambda lib: hours(lib) + lib.offsets.Day(1),
    lambda lib: hours(lib) + lib.offsets.Hour(2),
    lambda lib: workdays(lib) + lib.Timedelta("1h"),
    lambda lib: month_ends(lib) + lib.Timedelta("1h"),
    lambda lib: month_ends(lib) + lib.offsets.MonthEnd(1),
    lambda lib: days(lib) - days(lib)[0],
    lambda lib: days(lib) - dt.datetime(2024, 1, 1),
    lambda lib: lib.Timestamp("2025-01-01") - days(lib),
    lambda lib: workdays(lib) - workdays(lib)[0],
    lambda lib: days(lib) - days(lib),
    lambda lib: spans(lib) + lib.Timedelta("1D"),
    lambda lib: spans(lib) - lib.Timedelta("1h"),
    lambda lib: lib.Timedelta("1D") - spans(lib),
    lambda lib: spans(lib) + lib.Timestamp("2024-01-01"),
    lambda lib: spans(lib) + dt.datetime(2024, 1, 1),
    lambda lib: spans(lib) + spans(lib),
    lambda lib: -spans(lib),
    lambda lib: +spans(lib),
    lambda lib: -spans(lib).__neg__(),
    lambda lib: abs(spans(lib)),
    lambda lib: spans(lib).unique(),
    lambda lib: spans(lib).sort_values(ascending=False),
    lambda lib: spans(lib).delete(0),
    lambda lib: spans(lib).insert(4, lib.Timedelta("5h")),
    lambda lib: spans(lib).intersection(spans(lib)[1:]),
]


@pytest.mark.parametrize("call", CASES)
def test_the_frequency_is_pandas_frequency(call: Callable[[ModuleType], Any]) -> None:
    assert outcome(call, fp) == outcome(call, pd)
