"""`Period` against pandas, compared by repr."""

from __future__ import annotations

import datetime
import random
import warnings
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy
import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")

FREQS = ["Y", "Y-JUN", "Y-JAN", "Q", "Q-JAN", "Q-NOV", "M", "2M", "W", "W-WED", "W-SAT", "B"]
FREQS += ["D", "3D", "h", "5min", "s", "ms", "us", "ns"]

SPANS = [
    ("2026-03-15", "Y"),
    ("2026-03-15", "Y-JUN"),
    ("2026-03-15", "Q"),
    ("2026-03-15", "Q-JAN"),
    ("2026-03-15", "M"),
    ("2026-03-15", "2M"),
    ("2026-03-15", "W"),
    ("2026-03-15", "W-WED"),
    ("2026-03-15", "2W"),
    ("2026-03-15", "D"),
    ("2026-03-15", "3D"),
    ("2026-03-15", "B"),
    ("2026-03-14", "B"),
    ("2026-03-15 13:45:12", "h"),
    ("2026-03-15 13:45:12", "3h"),
    ("2026-03-15 13:45:12", "min"),
    ("2026-03-15 13:45:12", "s"),
    ("2026-03-15 13:45:12.5", "ms"),
    ("2026-03-15", "us"),
    ("2026-03-15", "ns"),
    ("1969-12-31", "W"),
    ("1960-01-01", "Q"),
]

FIELDS = ["year", "month", "day", "hour", "minute", "second", "dayofweek", "dayofyear"]
FIELDS += ["days_in_month", "quarter", "qyear", "week", "is_leap_year"]


def described(lib: ModuleType, value: str, freq: str) -> Any:
    """What a period says of itself, from its text to where it starts and ends."""
    p = lib.Period(value, freq)
    return (
        p,
        str(p),
        p.ordinal,
        p.freqstr,
        p.freq,
        p.start_time,
        p.end_time,
        p.to_timestamp(freq.lstrip("0123456789")),
        p.to_timestamp(freq.lstrip("0123456789"), "end"),
        p.to_timestamp(how="e"),
        [getattr(p, name) for name in FIELDS],
    )


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "P('2026-03-15', 'ME')": lambda lib: lib.Period("2026-03-15", "ME"),
    "P('2026-03-15', 'YE')": lambda lib: lib.Period("2026-03-15", "YE"),
    "P('2026-03-15', 'QE')": lambda lib: lib.Period("2026-03-15", "QE"),
    "P('2026-03-15', 'A')": lambda lib: lib.Period("2026-03-15", "A"),
    "P('2026-03-15', 'H')": lambda lib: lib.Period("2026-03-15", "H"),
    "P('2026-03-15', 'T')": lambda lib: lib.Period("2026-03-15", "T"),
    "P('2026-03-15', 'S')": lambda lib: lib.Period("2026-03-15", "S"),
    "P('2026-03-15', 'L')": lambda lib: lib.Period("2026-03-15", "L"),
    "P('2026-03-15', 'U')": lambda lib: lib.Period("2026-03-15", "U"),
    "P('2026-03-15', 'N')": lambda lib: lib.Period("2026-03-15", "N"),
    "P('2026-03-15', 'BM')": lambda lib: lib.Period("2026-03-15", "BM"),
    "P('2026-03-15', 'SM')": lambda lib: lib.Period("2026-03-15", "SM"),
    "P('2026-03-15', '2W')": lambda lib: lib.Period("2026-03-15", "2W"),
    "P('2026-03-15', 'W-FOO')": lambda lib: lib.Period("2026-03-15", "W-FOO"),
    "P('2026-03-15', 'Q-FOO')": lambda lib: lib.Period("2026-03-15", "Q-FOO"),
    "P('2026-03-15', 'MS')": lambda lib: lib.Period("2026-03-15", "MS"),
    "P('2026-03-15', 'h')": lambda lib: lib.Period("2026-03-15", "h"),
    "P('2026-03-15', '5min')": lambda lib: lib.Period("2026-03-15", "5min"),
    "P('2026-03-15', '1D')": lambda lib: lib.Period("2026-03-15", "1D"),
    "P('2026-03-15', '-1M')": lambda lib: lib.Period("2026-03-15", "-1M"),
    "P('2026-03-15', '0M')": lambda lib: lib.Period("2026-03-15", "0M"),
    "P('2026-03-15', 'C')": lambda lib: lib.Period("2026-03-15", "C"),
    "P('2026-03-15', 'bh')": lambda lib: lib.Period("2026-03-15", "bh"),
    "P('2026-03-15', 'BME')": lambda lib: lib.Period("2026-03-15", "BME"),
    "P('2026-03-15', 'QS')": lambda lib: lib.Period("2026-03-15", "QS"),
    "P('2026-03-15', 'SME')": lambda lib: lib.Period("2026-03-15", "SME"),
    "P('2026-03-15', 'BQE')": lambda lib: lib.Period("2026-03-15", "BQE"),
    "P('2026-03-15', 'YS')": lambda lib: lib.Period("2026-03-15", "YS"),
    "P('2026-03-15', 'BYE')": lambda lib: lib.Period("2026-03-15", "BYE"),
    "P('2026-03-15', 'W-MON')": lambda lib: lib.Period("2026-03-15", "W-MON"),
    "P('2026-03-15', '2Q-JUN')": lambda lib: lib.Period("2026-03-15", "2Q-JUN"),
    "P('2026-03-15', 'cbh')": lambda lib: lib.Period("2026-03-15", "cbh"),
    "P('2026-03-15', 'WOM-1MON')": lambda lib: lib.Period("2026-03-15", "WOM-1MON"),
    "P('2026-03-15', 'D-FOO')": lambda lib: lib.Period("2026-03-15", "D-FOO"),
    "P('2026-03-15', 'h-X')": lambda lib: lib.Period("2026-03-15", "h-X"),
    "P('2026-03-15', 'Q-JAN-X')": lambda lib: lib.Period("2026-03-15", "Q-JAN-X"),
    "P('2026-03-15', '1.5h')": lambda lib: lib.Period("2026-03-15", "1.5h"),
    "P('2026-03-15', '2D3h')": lambda lib: lib.Period("2026-03-15", "2D3h"),
    "P('2026-03-15', 'ns')": lambda lib: lib.Period("2026-03-15", "ns"),
    "P('2026-03-15', '10ns')": lambda lib: lib.Period("2026-03-15", "10ns"),
    "P('2026-03','M').end_time": lambda lib: lib.Period("2026-03", "M").end_time,
    "P('2026-03','M').start_time": lambda lib: lib.Period("2026-03", "M").start_time,
    "P('2026-03-15 13:00','h').end_time": lambda lib: lib.Period("2026-03-15 13:00", "h").end_time,
    "P('2026-03','M')+offsets.MonthEnd(2)": lambda lib: (
        lib.Period("2026-03", "M") + lib.offsets.MonthEnd(2)
    ),
    "P('2026-03','M')+offsets.Day(2)": lambda lib: lib.Period("2026-03", "M") + lib.offsets.Day(2),
    "P('2026-03-15','D')+Timedelta('2D')": lambda lib: (
        lib.Period("2026-03-15", "D") + lib.Timedelta("2D")
    ),
    "P('2026-03-15','D')+Timedelta('2h')": lambda lib: (
        lib.Period("2026-03-15", "D") + lib.Timedelta("2h")
    ),
    "P('2026-03-15 10:00','h')+Timedelta('2h')": lambda lib: (
        lib.Period("2026-03-15 10:00", "h") + lib.Timedelta("2h")
    ),
    "P('2026-03-15 10:00','h')+datetime.timedelta(hours=3)": lambda lib: (
        lib.Period("2026-03-15 10:00", "h") + datetime.timedelta(hours=3)
    ),
    "P('2026-03-15 10:00','h')+offsets.Minute(60)": lambda lib: (
        lib.Period("2026-03-15 10:00", "h") + lib.offsets.Minute(60)
    ),
    "P('2026-03-15 10:00','h')+offsets.Minute(30)": lambda lib: (
        lib.Period("2026-03-15 10:00", "h") + lib.offsets.Minute(30)
    ),
    "P('2026-03-15','D')-NaT": lambda lib: lib.Period("2026-03-15", "D") - lib.NaT,
    "P('2026-03-15','D')+NaT": lambda lib: lib.Period("2026-03-15", "D") + lib.NaT,
    "P('2026-03-15','D')==NaT": lambda lib: lib.Period("2026-03-15", "D") == lib.NaT,
    "P('2026-03-15','D')<NaT": lambda lib: lib.Period("2026-03-15", "D") < lib.NaT,
    "P('2026-03-15','D')==1": lambda lib: lib.Period("2026-03-15", "D") == 1,
    "P('2026-03-15','D')<1": lambda lib: lib.Period("2026-03-15", "D") < 1,
    "1+P('2026-03-15','D')": lambda lib: 1 + lib.Period("2026-03-15", "D"),
    "1-P('2026-03-15','D')": lambda lib: 1 - lib.Period("2026-03-15", "D"),
    "P('2026-03-15','D')+1.5": lambda lib: lib.Period("2026-03-15", "D") + 1.5,
    "P('2026-03-15','D')-P('2026-03-10','D')": lambda lib: (
        lib.Period("2026-03-15", "D") - lib.Period("2026-03-10", "D")
    ),
    "P('2026-03-15','2D')-P('2026-03-10','2D')": lambda lib: (
        lib.Period("2026-03-15", "2D") - lib.Period("2026-03-10", "2D")
    ),
    "P('2026-03-15','D')-P('2026-03','M')": lambda lib: (
        lib.Period("2026-03-15", "D") - lib.Period("2026-03", "M")
    ),
    "P(Timestamp('2026-03-15 10:30'))": lambda lib: lib.Period(lib.Timestamp("2026-03-15 10:30")),
    "P(datetime.datetime(2026,3,15,10,30))": lambda lib: lib.Period(
        datetime.datetime(2026, 3, 15, 10, 30)
    ),
    "P(datetime.date(2026,3,15))": lambda lib: lib.Period(datetime.date(2026, 3, 15)),
    "P(datetime.date(2026,3,15),'M')": lambda lib: lib.Period(datetime.date(2026, 3, 15), "M"),
    "P(Timestamp('2026-03-15 10:30',tz='UTC'),'h')": lambda lib: lib.Period(
        lib.Timestamp("2026-03-15 10:30", tz="UTC"), "h"
    ),
    "P(P('2026-03','M'),'D')": lambda lib: lib.Period(lib.Period("2026-03", "M"), "D"),
    "P(P('2026-03','M'))": lambda lib: lib.Period(lib.Period("2026-03", "M")),
    "P(20260315,'D')": lambda lib: lib.Period(20260315, "D"),
    "P(2026,'Y')": lambda lib: lib.Period(2026, "Y"),
    "P(2026.0,'Y')": lambda lib: lib.Period(2026.0, "Y"),
    "P('2026-03-15T10:30:15.123456789')": lambda lib: lib.Period("2026-03-15T10:30:15.123456789"),
    "P('2026-03-15 10:30:15.123')": lambda lib: lib.Period("2026-03-15 10:30:15.123"),
    "P('2026-03-15 10:30:15')": lambda lib: lib.Period("2026-03-15 10:30:15"),
    "P('2026-03-15 10')": lambda lib: lib.Period("2026-03-15 10"),
    "P('2026Q3','M')": lambda lib: lib.Period("2026Q3", "M"),
    "P('2026-Q3')": lambda lib: lib.Period("2026-Q3"),
    "P('Q3 2026')": lambda lib: lib.Period("Q3 2026"),
    "P('2026W10')": lambda lib: lib.Period("2026W10"),
    "P('2026-03-09/2026-03-15','W')": lambda lib: lib.Period("2026-03-09/2026-03-15", "W"),
    "P('March 2026')": lambda lib: lib.Period("March 2026"),
    "P('2026-03','D')": lambda lib: lib.Period("2026-03", "D"),
    "P('2026','D')": lambda lib: lib.Period("2026", "D"),
    "P(year=2026,quarter=2,freq='Q')": lambda lib: lib.Period(year=2026, quarter=2, freq="Q"),
    "P(year=2026,month=3,day=15,hour=5,freq='h')": lambda lib: lib.Period(
        year=2026, month=3, day=15, hour=5, freq="h"
    ),
    "P(year=2026)": lambda lib: lib.Period(year=2026),
    "P(ordinal=5)": lambda lib: lib.Period(ordinal=5),
    "P('2026-03','M',ordinal=4)": lambda lib: lib.Period("2026-03", "M", ordinal=4),
    "P(2026, 'M')": lambda lib: lib.Period(2026, "M"),
    "P('2026-03-15', 'D').strftime('%Y %q %f %F %l %u %n %j %a %b')": lambda lib: lib.Period(
        "2026-03-15", "D"
    ).strftime("%Y %q %f %F %l %u %n %j %a %b"),
    "P('2026-03-15', 'Q-JAN').strftime('%q %f %F %Y')": lambda lib: lib.Period(
        "2026-03-15", "Q-JAN"
    ).strftime("%q %f %F %Y"),
    "P('2026-03-15', 'Q-JAN').strftime(None)": lambda lib: lib.Period(
        "2026-03-15", "Q-JAN"
    ).strftime(None),
    "P('2026-03-15 10:30:15.123456789', 'ns').strftime('%l %u %n')": lambda lib: lib.Period(
        "2026-03-15 10:30:15.123456789", "ns"
    ).strftime("%l %u %n"),
    "str(P('2026-03-15', 'Q-JAN').asfreq('M'))": lambda lib: str(
        lib.Period("2026-03-15", "Q-JAN").asfreq("M")
    ),
    "P('2026-03-15', 'Q-JAN').asfreq('M','s')": lambda lib: lib.Period(
        "2026-03-15", "Q-JAN"
    ).asfreq("M", "s"),
    "P('2026-03-15', 'Q-JAN').asfreq('M','S')": lambda lib: lib.Period(
        "2026-03-15", "Q-JAN"
    ).asfreq("M", "S"),
    "P('2026-03-15', 'Q-JAN').asfreq('M','x')": lambda lib: lib.Period(
        "2026-03-15", "Q-JAN"
    ).asfreq("M", "x"),
    "P('2026-03-15', 'Q-JAN').asfreq('B')": lambda lib: lib.Period("2026-03-15", "Q-JAN").asfreq(
        "B"
    ),
    "P('2026-03-15', 'W').asfreq('B')": lambda lib: lib.Period("2026-03-15", "W").asfreq("B"),
    "P('2026-03-15', 'W').asfreq('B','start')": lambda lib: lib.Period("2026-03-15", "W").asfreq(
        "B", "start"
    ),
    "P('2026-03-14', 'D').asfreq('B')": lambda lib: lib.Period("2026-03-14", "D").asfreq("B"),
    "P('2026-03-14', 'D').asfreq('B','s')": lambda lib: lib.Period("2026-03-14", "D").asfreq(
        "B", "s"
    ),
    "P('2026-03-16', 'B').asfreq('W')": lambda lib: lib.Period("2026-03-16", "B").asfreq("W"),
    "P('2026-03-15', 'D').asfreq('2M')": lambda lib: lib.Period("2026-03-15", "D").asfreq("2M"),
    "P('2026-03-15', 'M').to_timestamp('D')": lambda lib: lib.Period(
        "2026-03-15", "M"
    ).to_timestamp("D"),
    "P('2026-03-15', 'M').to_timestamp('D','end')": lambda lib: lib.Period(
        "2026-03-15", "M"
    ).to_timestamp("D", "end"),
    "P('2026-03-15', 'M').to_timestamp('h','end')": lambda lib: lib.Period(
        "2026-03-15", "M"
    ).to_timestamp("h", "end"),
    "P('2026-03-15', 'Y').to_timestamp('M','e')": lambda lib: lib.Period(
        "2026-03-15", "Y"
    ).to_timestamp("M", "e"),
    "P('2026-03-15', 'Y').to_timestamp(how='E')": lambda lib: lib.Period(
        "2026-03-15", "Y"
    ).to_timestamp(how="E"),
    "P('2026-03-15', 'W').to_timestamp(how='end')": lambda lib: lib.Period(
        "2026-03-15", "W"
    ).to_timestamp(how="end"),
    "Timestamp('2026-03-15 10:30').to_period('M')": lambda lib: lib.Timestamp(
        "2026-03-15 10:30"
    ).to_period("M"),
    "Timestamp('2026-03-15 10:30').to_period()": lambda lib: lib.Timestamp(
        "2026-03-15 10:30"
    ).to_period(),
    "Timestamp('2026-03-15 10:30',tz='UTC').to_period('D')": lambda lib: lib.Timestamp(
        "2026-03-15 10:30", tz="UTC"
    ).to_period("D"),
    "NaT.to_period('D')": lambda lib: lib.NaT.to_period("D"),
    "P('2026-03-15','D').n": lambda lib: lib.Period("2026-03-15", "D").n,
    "P('2026-03-15','3D').freq": lambda lib: lib.Period("2026-03-15", "3D").freq,
    "P('2026-03-15','3D')+1": lambda lib: lib.Period("2026-03-15", "3D") + 1,
    "(P('2026-03-15','3D')-P('2026-03-10','3D'))": lambda lib: (
        lib.Period("2026-03-15", "3D") - lib.Period("2026-03-10", "3D")
    ),
    "{P('2026-03-15','D'):1}[P('2026-03-15','D')]": lambda lib: {lib.Period("2026-03-15", "D"): 1}[
        lib.Period("2026-03-15", "D")
    ],
    "P('2026-03-15','D') == P('2026-03-15','2D')": lambda lib: (
        lib.Period("2026-03-15", "D") == lib.Period("2026-03-15", "2D")
    ),
    "P('2026-03-15','D') == P('2026-03-15','B')": lambda lib: (
        lib.Period("2026-03-15", "D") == lib.Period("2026-03-15", "B")
    ),
    "P('2026-03-15','D') < P('2026-03-15','2D')": lambda lib: (
        lib.Period("2026-03-15", "D") < lib.Period("2026-03-15", "2D")
    ),
    "P('2026-03-15','D') == '2026-03-15'": lambda lib: (
        lib.Period("2026-03-15", "D") == "2026-03-15"
    ),
    "P('2026-03-15','D') < '2026-03-15'": lambda lib: lib.Period("2026-03-15", "D") < "2026-03-15",
    "P('2026-03-15','D') == Timestamp('2026-03-15')": lambda lib: (
        lib.Period("2026-03-15", "D") == lib.Timestamp("2026-03-15")
    ),
    "P('2026-03-15','D') < Timestamp('2026-03-15')": lambda lib: (
        lib.Period("2026-03-15", "D") < lib.Timestamp("2026-03-15")
    ),
    "format(P('2026-03-15','D'))": lambda lib: format(lib.Period("2026-03-15", "D")),
    "P('0001-01-01','D')": lambda lib: lib.Period("0001-01-01", "D"),
    "P('2026-03-15','D') - offsets.Day(3)": lambda lib: (
        lib.Period("2026-03-15", "D") - lib.offsets.Day(3)
    ),
    "P('2026-03-15','M') - offsets.MonthEnd(3)": lambda lib: (
        lib.Period("2026-03-15", "M") - lib.offsets.MonthEnd(3)
    ),
    "P('2026-03-15','Y') + offsets.YearEnd(3)": lambda lib: (
        lib.Period("2026-03-15", "Y") + lib.offsets.YearEnd(3)
    ),
    "P('2026-03-15','Y') + offsets.YearEnd(3, month=6)": lambda lib: (
        lib.Period("2026-03-15", "Y") + lib.offsets.YearEnd(3, month=6)
    ),
    "P('2026-03-15','Q') + offsets.QuarterEnd(1)": lambda lib: (
        lib.Period("2026-03-15", "Q") + lib.offsets.QuarterEnd(1)
    ),
    "P('2026-03-15','W') + offsets.Week(1)": lambda lib: (
        lib.Period("2026-03-15", "W") + lib.offsets.Week(1)
    ),
    "P('2026-03-15','W') + offsets.Week(1, weekday=6)": lambda lib: (
        lib.Period("2026-03-15", "W") + lib.offsets.Week(1, weekday=6)
    ),
    "P('2026-03-15','B') + offsets.BusinessDay(2)": lambda lib: (
        lib.Period("2026-03-15", "B") + lib.offsets.BusinessDay(2)
    ),
    "P('2026-03-15','D') + offsets.Hour(24)": lambda lib: (
        lib.Period("2026-03-15", "D") + lib.offsets.Hour(24)
    ),
    "P('2026-03-15','D') + offsets.Hour(25)": lambda lib: (
        lib.Period("2026-03-15", "D") + lib.offsets.Hour(25)
    ),
    "P('2026-03-15','D') + Timedelta(days=1, hours=1)": lambda lib: (
        lib.Period("2026-03-15", "D") + lib.Timedelta(days=1, hours=1)
    ),
    "P('2026-03-15','M') + Timedelta(days=1)": lambda lib: (
        lib.Period("2026-03-15", "M") + lib.Timedelta(days=1)
    ),
    "P('2026-03-15','D') + [1]": lambda lib: lib.Period("2026-03-15", "D") + [1],  # noqa: RUF005
    "P('2026-03-15', offsets.MonthEnd())": lambda lib: lib.Period(
        "2026-03-15", lib.offsets.MonthEnd()
    ),
    "P('2026-03-15', offsets.MonthBegin())": lambda lib: lib.Period(
        "2026-03-15", lib.offsets.MonthBegin()
    ),
    "P('2026-03-15', offsets.Week())": lambda lib: lib.Period("2026-03-15", lib.offsets.Week()),
    "P('2026-03-15', offsets.Day(2))": lambda lib: lib.Period("2026-03-15", lib.offsets.Day(2)),
    "P('2026-03-15', Timedelta('1D'))": lambda lib: lib.Period("2026-03-15", lib.Timedelta("1D")),
    "P('2026-03-15', offsets.Hour(-1))": lambda lib: lib.Period("2026-03-15", lib.offsets.Hour(-1)),
    "P('2026-03-15 10:30', 'D').hour": lambda lib: lib.Period("2026-03-15 10:30", "D").hour,
    "P('2026-03-15 10:30', 'h').minute": lambda lib: lib.Period("2026-03-15 10:30", "h").minute,
    "P('2026-03-15 10:30', 'D') < P('2026-03-16', 'D')": lambda lib: (
        lib.Period("2026-03-15 10:30", "D") < lib.Period("2026-03-16", "D")
    ),
    "P('2026-03-15 10:30', 'D') + 0": lambda lib: lib.Period("2026-03-15 10:30", "D") + 0,
    "P('2026-03-15','D').asfreq('h')": lambda lib: lib.Period("2026-03-15", "D").asfreq("h"),
    "P('2026-03-15','D').asfreq('h', 'start')": lambda lib: lib.Period("2026-03-15", "D").asfreq(
        "h", "start"
    ),
    "P('2026-03-15','D').asfreq('min')": lambda lib: lib.Period("2026-03-15", "D").asfreq("min"),
    "P('2026-03-15 10:00','h').asfreq('D')": lambda lib: lib.Period("2026-03-15 10:00", "h").asfreq(
        "D"
    ),
    "P('2026-03-15 10:00','h').asfreq('M')": lambda lib: lib.Period("2026-03-15 10:00", "h").asfreq(
        "M"
    ),
    "P('2026-03-15 10:00','h').asfreq('Y-JUN')": lambda lib: lib.Period(
        "2026-03-15 10:00", "h"
    ).asfreq("Y-JUN"),
    "P('2026','Y').asfreq('Q')": lambda lib: lib.Period("2026", "Y").asfreq("Q"),
    "P('2026','Y').asfreq('Q','s')": lambda lib: lib.Period("2026", "Y").asfreq("Q", "s"),
    "P('2026','Y-JUN').asfreq('Q','s')": lambda lib: lib.Period("2026", "Y-JUN").asfreq("Q", "s"),
    "P('2026','Y-JUN').asfreq('M','s')": lambda lib: lib.Period("2026", "Y-JUN").asfreq("M", "s"),
    "P('2026Q1','Q-JAN').asfreq('Y')": lambda lib: lib.Period("2026Q1", "Q-JAN").asfreq("Y"),
    "P('2026-03-15','W-WED').asfreq('M')": lambda lib: lib.Period("2026-03-15", "W-WED").asfreq(
        "M"
    ),
    "P('2026-03-15','W-WED').asfreq('M','s')": lambda lib: lib.Period("2026-03-15", "W-WED").asfreq(
        "M", "s"
    ),
    "P('2026-03-15','D').asfreq(offsets.MonthEnd())": lambda lib: lib.Period(
        "2026-03-15", "D"
    ).asfreq(lib.offsets.MonthEnd()),
    "P('2026-03-15','D').asfreq('ME')": lambda lib: lib.Period("2026-03-15", "D").asfreq("ME"),
    "P('2026-03-15','D').to_timestamp('ME')": lambda lib: lib.Period(
        "2026-03-15", "D"
    ).to_timestamp("ME"),
    "P('2026-03-15','D').to_timestamp('M')": lambda lib: lib.Period("2026-03-15", "D").to_timestamp(
        "M"
    ),
    "P('2026-03','M').to_timestamp('h')": lambda lib: lib.Period("2026-03", "M").to_timestamp("h"),
    "P('2026-03','M').to_timestamp('W')": lambda lib: lib.Period("2026-03", "M").to_timestamp("W"),
    "P('2026-03','M').to_timestamp('B')": lambda lib: lib.Period("2026-03", "M").to_timestamp("B"),
    "P('2026-03','M').to_timestamp('B', how='end')": lambda lib: lib.Period(
        "2026-03", "M"
    ).to_timestamp("B", how="end"),
    "P('2026-03-14','D').to_timestamp('B')": lambda lib: lib.Period("2026-03-14", "D").to_timestamp(
        "B"
    ),
    "P('2026-03-15 10:00','h').to_timestamp()": lambda lib: lib.Period(
        "2026-03-15 10:00", "h"
    ).to_timestamp(),
    "P('2026-03-15 10:00','h').to_timestamp(how='end')": lambda lib: lib.Period(
        "2026-03-15 10:00", "h"
    ).to_timestamp(how="end"),
    "P('2026-03','2M').end_time": lambda lib: lib.Period("2026-03", "2M").end_time,
    "P('2026-03','2M').to_timestamp(how='end')": lambda lib: lib.Period(
        "2026-03", "2M"
    ).to_timestamp(how="end"),
    "P('2026-03-15','B').end_time": lambda lib: lib.Period("2026-03-15", "B").end_time,
    "P('2026-03-15','W').start_time": lambda lib: lib.Period("2026-03-15", "W").start_time,
    "P('2026-03-15 10:00','h') == P('2026-03-15 10:00','60min')": lambda lib: (
        lib.Period("2026-03-15 10:00", "h") == lib.Period("2026-03-15 10:00", "60min")
    ),
    "repr(P('2026-03-15','W-MON'))": lambda lib: repr(lib.Period("2026-03-15", "W-MON")),
    "P('2026-03-15','D').freq.freqstr": lambda lib: lib.Period("2026-03-15", "D").freq.freqstr,
    "P('2026-03-15','Y').freq.freqstr": lambda lib: lib.Period("2026-03-15", "Y").freq.freqstr,
    "P('2026-03-15','W').freq.freqstr": lambda lib: lib.Period("2026-03-15", "W").freq.freqstr,
    "P('2026-03-15','B').week": lambda lib: lib.Period("2026-03-15", "B").week,
    "P('2026-03-15','W').day": lambda lib: lib.Period("2026-03-15", "W").day,
    "P('2026-03-15','W').dayofweek": lambda lib: lib.Period("2026-03-15", "W").dayofweek,
    "P('1969-06','M').ordinal": lambda lib: lib.Period("1969-06", "M").ordinal,
    "P('1969-12-30 10:00','h').ordinal": lambda lib: lib.Period("1969-12-30 10:00", "h").ordinal,
    "P('1969-12-30','B').ordinal": lambda lib: lib.Period("1969-12-30", "B").ordinal,
    "P('1969-12-30','W-WED').ordinal": lambda lib: lib.Period("1969-12-30", "W-WED").ordinal,
    "P('1969-12-30','W-MON').ordinal": lambda lib: lib.Period("1969-12-30", "W-MON").ordinal,
    "P(ordinal=0,freq='W-MON')": lambda lib: lib.Period(ordinal=0, freq="W-MON"),
    "P(ordinal=0,freq='W-SUN')": lambda lib: lib.Period(ordinal=0, freq="W-SUN"),
    "P(ordinal=0,freq='W-SAT')": lambda lib: lib.Period(ordinal=0, freq="W-SAT"),
    "P(ordinal=0,freq='B')": lambda lib: lib.Period(ordinal=0, freq="B"),
    "P(ordinal=-1,freq='B')": lambda lib: lib.Period(ordinal=-1, freq="B"),
    "P(ordinal=0,freq='Q-JAN')": lambda lib: lib.Period(ordinal=0, freq="Q-JAN"),
    "P(ordinal=0,freq='Y-JAN')": lambda lib: lib.Period(ordinal=0, freq="Y-JAN"),
    "P(year=2026, month=5, freq='Q-JAN')": lambda lib: lib.Period(year=2026, month=5, freq="Q-JAN"),
    "P(year=2026, quarter=1, freq='Q-JAN')": lambda lib: lib.Period(
        year=2026, quarter=1, freq="Q-JAN"
    ),
    "P(year=2026, quarter=1, freq='M')": lambda lib: lib.Period(year=2026, quarter=1, freq="M"),
    "P(year=2026, month=5, freq='Y-JAN')": lambda lib: lib.Period(year=2026, month=5, freq="Y-JAN"),
    "P(year=2026, month=13, freq='M')": lambda lib: lib.Period(year=2026, month=13, freq="M"),
    "P(year=2026, month=3, day=15, freq='W')": lambda lib: lib.Period(
        year=2026, month=3, day=15, freq="W"
    ),
    "P('2026-03-15','D') - Timestamp('2026-03-15')": lambda lib: (
        lib.Period("2026-03-15", "D") - lib.Timestamp("2026-03-15")
    ),
    "Timestamp('2026-03-15') - P('2026-03-15','D')": lambda lib: (
        lib.Timestamp("2026-03-15") - lib.Period("2026-03-15", "D")
    ),
    "Timedelta('1D') + P('2026-03-15','D')": lambda lib: (
        lib.Timedelta("1D") + lib.Period("2026-03-15", "D")
    ),
    "offsets.Day(1) + P('2026-03-15','D')": lambda lib: (
        lib.offsets.Day(1) + lib.Period("2026-03-15", "D")
    ),
    "P('2026-03-15','D') - Timedelta('1D')": lambda lib: (
        lib.Period("2026-03-15", "D") - lib.Timedelta("1D")
    ),
    "P('2026-03-15','D') - 1.0": lambda lib: lib.Period("2026-03-15", "D") - 1.0,
    "P('2026-03-15','D') + True": lambda lib: lib.Period("2026-03-15", "D") + True,
    "P('2026-03-15','D') + numpy.int64(2)": lambda lib: (
        lib.Period("2026-03-15", "D") + numpy.int64(2)
    ),
    "P('2026-03-15','D') + numpy.timedelta64(2,'D')": lambda lib: (
        lib.Period("2026-03-15", "D") + numpy.timedelta64(2, "D")
    ),
    "P('2026-03-15','D') + numpy.array([1,2])": lambda lib: (
        lib.Period("2026-03-15", "D") + numpy.array([1, 2])
    ),
    "P('2026-03-15','D').strftime('%H:%M %%')": lambda lib: lib.Period("2026-03-15", "D").strftime(
        "%H:%M %%"
    ),
    "P('2026-03-15','D') == P('2026-03-15','D').asfreq('D')": lambda lib: (
        lib.Period("2026-03-15", "D") == lib.Period("2026-03-15", "D").asfreq("D")
    ),
}


def mistake(error: Exception) -> str:
    kind = next(kind.__name__ for kind in type(error).__mro__ if kind.__module__ == "builtins")
    return f"{kind}: {error}"


def outcome(build: Callable[[], Any]) -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return repr(build()).replace("string", "str")
    except Exception as error:
        return mistake(error)


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_answers_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))


@pytest.mark.parametrize(("value", "freq"), SPANS)
def test_a_period_describes_itself_as_pandas(value: str, freq: str) -> None:
    assert outcome(lambda: described(fp, value, freq)) == outcome(
        lambda: described(pd, value, freq)
    )


def moments(seed: int) -> list[tuple[str, str, str, str]]:
    """Random moments, each with a frequency to read it at, one to move it to, and a side."""
    chosen = random.Random(seed)
    drawn = []
    for _ in range(40):
        day = datetime.datetime(
            chosen.randint(1900, 2100), chosen.randint(1, 12), chosen.randint(1, 28)
        )
        day += datetime.timedelta(
            seconds=chosen.randint(0, 86399), microseconds=chosen.randint(0, 999999)
        )
        drawn.append(
            (day.isoformat(" "), chosen.choice(FREQS), chosen.choice(FREQS), chosen.choice("SE"))
        )
    return drawn


def travelled(lib: ModuleType, text: str, freq: str, other: str, how: str) -> Any:
    p = lib.Period(text, freq)
    other = other.lstrip("0123456789")
    return (
        p,
        p.ordinal,
        lib.Period(ordinal=p.ordinal, freq=freq),
        p.asfreq(other, how),
        [getattr(p, name) for name in FIELDS],
        p + 3,
        p - 7,
        p - lib.Period("2000-01-01", freq),
        p.strftime("%Y-%m-%d %H:%M:%S %q %F %f %l %u %n %j %a"),
        lib.Timestamp(text).to_period(freq),
        p.start_time,
        p.end_time,
        # pandas reads the ordinal of a period of nanoseconds moved to a coarser
        # start as a count of the wrong unit, so that one is left out.
        None if freq == "ns" and how == "S" else p.to_timestamp(other, how),
    )


@pytest.mark.parametrize("moment", moments(7), ids=lambda moment: " ".join(moment))
def test_random_moments_travel_as_in_pandas(moment: tuple[str, str, str, str]) -> None:
    assert outcome(lambda: travelled(fp, *moment)) == outcome(lambda: travelled(pd, *moment))
