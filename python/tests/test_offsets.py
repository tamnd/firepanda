"""Date offsets, `firepanda.offsets`, compared with `pandas.offsets`.

Each offset is built the same way out of both libraries and asked the same
questions: what it prints as, what moving a set of moments forward and back by
it gives, whether each moment is on it, and where each rolls to. The moments
cover the ends and starts of months, weekends, a leap day, times of day and a
change of year, which is where the rules of each offset differ.
"""

from __future__ import annotations

import datetime
import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

MOMENTS = [
    "2023-12-31",
    "2024-01-01",
    "2024-01-05",
    "2024-01-06 10:30",
    "2024-01-07",
    "2024-01-15 10:00",
    "2024-01-31 10:00",
    "2024-02-29",
    "2024-03-29",
    "2024-03-30",
    "2024-03-31",
    "2024-04-01",
    "2024-06-15",
    "2024-06-28",
    "2024-06-30",
    "2024-07-01",
    "2024-09-30",
    "2024-12-27",
    "2024-12-30",
    "2024-12-31 23:59",
    "2025-01-01",
    "2025-01-03",
]

HOURS = [
    "2024-01-05 08:00",
    "2024-01-05 09:00",
    "2024-01-05 10:30",
    "2024-01-05 12:00",
    "2024-01-05 12:30",
    "2024-01-05 16:59:30",
    "2024-01-05 17:00",
    "2024-01-05 18:00",
    "2024-01-06 12:00",
    "2024-01-08 09:00",
    "2024-01-08 09:30",
    "2024-01-08 13:00",
    "2024-01-08 17:00",
]

KINDS: dict[str, tuple[str, dict[str, Any]]] = {
    "MonthEnd": ("MonthEnd", {}),
    "MonthBegin": ("MonthBegin", {}),
    "BMonthEnd": ("BMonthEnd", {}),
    "BMonthBegin": ("BMonthBegin", {}),
    "QuarterEnd": ("QuarterEnd", {}),
    "QuarterEnd FEB": ("QuarterEnd", {"startingMonth": 2}),
    "QuarterBegin": ("QuarterBegin", {}),
    "QuarterBegin JAN": ("QuarterBegin", {"startingMonth": 1}),
    "BQuarterEnd": ("BQuarterEnd", {}),
    "BQuarterBegin": ("BQuarterBegin", {}),
    "HalfYearEnd": ("HalfYearEnd", {}),
    "HalfYearBegin": ("HalfYearBegin", {}),
    "BHalfYearEnd": ("BHalfYearEnd", {}),
    "BHalfYearBegin": ("BHalfYearBegin", {"startingMonth": 4}),
    "YearEnd": ("YearEnd", {}),
    "YearEnd JUN": ("YearEnd", {"month": 6}),
    "YearBegin": ("YearBegin", {}),
    "BYearEnd": ("BYearEnd", {}),
    "BYearBegin": ("BYearBegin", {}),
    "SemiMonthEnd": ("SemiMonthEnd", {}),
    "SemiMonthEnd 1": ("SemiMonthEnd", {"day_of_month": 1}),
    "SemiMonthBegin": ("SemiMonthBegin", {"day_of_month": 20}),
    "Week": ("Week", {}),
    "Week FRI": ("Week", {"weekday": 4}),
    "WeekOfMonth": ("WeekOfMonth", {"week": 2, "weekday": 3}),
    "LastWeekOfMonth": ("LastWeekOfMonth", {"weekday": 4}),
    "BDay": ("BDay", {}),
    "BDay offset": ("BDay", {"offset": datetime.timedelta(hours=2)}),
    "CDay": ("CDay", {"weekmask": "Mon Wed Fri", "holidays": ["2024-01-01", "2024-03-29"]}),
    "CDay digits": ("CDay", {"weekmask": "1111001"}),
    "CBMonthEnd": ("CBMonthEnd", {"holidays": ["2024-03-29", "2024-12-31"]}),
    "CBMonthBegin": ("CBMonthBegin", {"weekmask": "Tue Thu"}),
    "Easter": ("Easter", {}),
    "Easter julian": ("Easter", {"method": 1}),
    "Easter orthodox": ("Easter", {"method": 2}),
    "FY5253": ("FY5253", {"weekday": 4, "startingMonth": 12}),
    "FY5253 last": ("FY5253", {"weekday": 0, "startingMonth": 6, "variation": "last"}),
    "FY5253Quarter": ("FY5253Quarter", {"weekday": 4, "startingMonth": 12}),
    "FY5253Quarter extra": (
        "FY5253Quarter",
        {"weekday": 5, "startingMonth": 3, "qtr_with_extra_week": 3, "variation": "last"},
    ),
    "Day": ("Day", {}),
    "Hour": ("Hour", {}),
    "Minute": ("Minute", {}),
    "Second": ("Second", {}),
    "Milli": ("Milli", {}),
    "Micro": ("Micro", {}),
    "Nano": ("Nano", {}),
    "DateOffset": ("DateOffset", {}),
    "DateOffset months": ("DateOffset", {"months": 1}),
    "DateOffset mixed": ("DateOffset", {"years": 1, "months": 2, "days": 3, "hours": 4}),
    "DateOffset day": ("DateOffset", {"months": 1, "day": 31}),
    "DateOffset set": ("DateOffset", {"month": 2, "day": 30, "hour": 6}),
    "DateOffset weekday": ("DateOffset", {"weekday": 2}),
    "DateOffset spans": ("DateOffset", {"weeks": 1, "minutes": 30}),
}

HOURLY: dict[str, tuple[str, dict[str, Any]]] = {
    "BusinessHour": ("BusinessHour", {}),
    "BusinessHour two": ("BusinessHour", {"start": ["09:00", "13:00"], "end": ["12:00", "17:00"]}),
    "BusinessHour short": ("BusinessHour", {"start": "10:00", "end": "13:30"}),
    "CustomBusinessHour": (
        "CustomBusinessHour",
        {"weekmask": "Mon Tue Wed Thu", "holidays": ["2024-01-08"]},
    ),
}

STEPS = [-3, -1, 0, 1, 2, 5]
HOUR_STEPS = [-17, -9, -8, -2, -1, 0, 1, 2, 7, 8, 9, 16]


def offset(m: ModuleType, kind: str, n: int, arguments: dict[str, Any], **more: Any) -> Any:
    return getattr(m.offsets, kind)(n, **arguments, **more)


def outcome(call: Callable[[], Any]) -> Any:
    """What a call answers, or the kind of error it raises."""
    try:
        return repr(call())
    except Exception as error:
        return type(error).__name__


def moved(m: ModuleType, kind: str, n: int, arguments: dict[str, Any], moments: list[str]) -> Any:
    if n == 0 and outcome(lambda: offset(m, kind, n, arguments)) == "ValueError":
        return "ValueError"
    step = offset(m, kind, n, arguments)
    answers = []
    for text in moments:
        moment = m.Timestamp(text)
        calls = (
            lambda at=moment: at + step,
            lambda at=moment: at - step,
            lambda at=moment: step.is_on_offset(at),
            lambda at=moment: step.rollforward(at),
            lambda at=moment: step.rollback(at),
        )
        answers.append((text, *(outcome(call) for call in calls)))
    return answers


def described(m: ModuleType, kind: str, n: int, arguments: dict[str, Any]) -> Any:
    """What an offset prints as and says about itself."""
    step = offset(m, kind, n, arguments)
    kwds = {k: v for k, v in step.kwds.items() if k not in ("calendar", "holidays")}
    return (
        repr(step),
        step.freqstr,
        outcome(lambda: step.rule_code),
        outcome(lambda: step.name),
        kwds,
        step.n,
        step.normalize,
        outcome(lambda: step.nanos),
        repr(step.base),
        step == step.copy(),
        outcome(lambda: step == step.freqstr),
        repr(-step),
        repr(step * 3),
        repr(2 * step),
    )


@needs_pandas
@pytest.mark.parametrize("n", STEPS)
@pytest.mark.parametrize("label", list(KINDS))
def test_moving_moments_matches_pandas(firepanda: ModuleType, label: str, n: int) -> None:
    """Adding, taking away, being on the offset and rolling, for every moment."""
    import pandas as pd

    kind, arguments = KINDS[label]
    mine = moved(firepanda, kind, n, arguments, MOMENTS)
    theirs = moved(pd, kind, n, arguments, MOMENTS)
    assert mine == theirs


@needs_pandas
@pytest.mark.parametrize("n", HOUR_STEPS)
@pytest.mark.parametrize("label", list(HOURLY))
def test_business_hours_match_pandas(firepanda: ModuleType, label: str, n: int) -> None:
    """Working hours forward and back from moments in, before, between and after them."""
    import pandas as pd

    kind, arguments = HOURLY[label]
    mine = moved(firepanda, kind, n, arguments, HOURS)
    theirs = moved(pd, kind, n, arguments, HOURS)
    assert mine == theirs


@needs_pandas
@pytest.mark.parametrize("n", [1, 2, -1])
@pytest.mark.parametrize("label", list(KINDS) + list(HOURLY))
def test_an_offset_describes_itself_as_pandas_does(
    firepanda: ModuleType, label: str, n: int
) -> None:
    """The repr, the frequency text, the rule code, the parameters and the arithmetic of steps."""
    import pandas as pd

    kind, arguments = {**KINDS, **HOURLY}[label]
    assert described(firepanda, kind, n, arguments) == described(pd, kind, n, arguments)


def normalized(m: ModuleType) -> Any:
    step = m.offsets.MonthEnd(normalize=True)
    moment = m.Timestamp("2024-01-15 10:00")
    return (
        repr(step),
        moment + step,
        step.is_on_offset(m.Timestamp("2024-01-31 10:00")),
        step.is_on_offset(m.Timestamp("2024-01-31")),
        step == m.offsets.MonthEnd(),
    )


def zoned(m: ModuleType) -> Any:
    moment = m.Timestamp("2024-03-09 12:00", tz="US/Eastern")
    return [
        repr(moment + step)
        for step in (
            m.offsets.MonthEnd(),
            m.offsets.Day(),
            m.offsets.Hour(24),
            m.offsets.Week(),
            m.offsets.DateOffset(hours=24),
            m.offsets.DateOffset(days=1),
            m.offsets.DateOffset(months=1),
            m.offsets.BDay(),
        )
    ]


def ticks(m: ModuleType) -> Any:
    o = m.offsets
    return [
        repr(o.Hour(2) + o.Minute(30)),
        repr(o.Hour(2) - o.Minute(30)),
        repr(o.Hour(1) + datetime.timedelta(minutes=1)),
        repr(o.Hour(1) + o.Day(1)),
        repr(o.Day(1) + o.Day(2)),
        repr(o.Hour(1) * 1.5),
        o.Hour(1) < o.Minute(90),
        o.Hour(3) / o.Hour(1),
        o.Hour(2) == o.Minute(120),
        o.Hour(1) == m.Timedelta("1h"),
        o.Hour(24) == o.Day(),
        o.Day() == m.Timedelta("1D"),
        hash(o.Hour(2)) == hash(o.Minute(120)),
        o.Day().nanos,
        o.Nano(5).nanos,
    ]


def units(m: ModuleType) -> Any:
    moment = m.Timestamp("2024-01-05")
    return [
        (moment + m.offsets.MonthEnd()).unit,
        (moment + m.offsets.Hour()).unit,
        (moment + m.offsets.Nano()).unit,
        repr(m.Timestamp("2024-01-05 00:00:00.000000001") + m.offsets.MonthEnd()),
        repr(m.Timestamp("2024-01-05") + m.offsets.DateOffset(nanoseconds=5)),
        repr(m.Timestamp("2024-01-05") + m.offsets.DateOffset(months=1, nanoseconds=5)),
    ]


def columns(m: ModuleType) -> Any:
    column = m.Series(m.to_datetime(m.Series(["2024-01-15", None, "2024-02-29"])), name="d")
    index = m.DatetimeIndex(["2024-01-15", "2024-03-31"], name="i")
    answers = []
    for step in (m.offsets.MonthEnd(), m.offsets.BDay(2), m.offsets.Hour(3)):
        for value in (column + step, step + column, column - step):
            answers.append((str(value.dtype), value.name, [repr(v) for v in value.tolist()]))
        for value in (index + step, index - step):
            answers.append((str(value.dtype), value.name, [repr(v) for v in value.tolist()]))
    zoned = m.Series(m.to_datetime(m.Series(["2024-03-09 12:00"])).dt.tz_localize("US/Eastern"))
    moved_zoned = zoned + m.offsets.MonthEnd()
    answers.append((str(moved_zoned.dtype), [repr(v) for v in moved_zoned.tolist()]))
    return answers


def ranges(m: ModuleType) -> Any:
    o = m.offsets
    made = [
        m.date_range("2024-01-01", periods=4, freq=o.MonthEnd(2)),
        m.date_range("2024-01-10", "2024-06-30", freq=o.BQuarterEnd()),
        m.date_range(end="2024-06-30", periods=3, freq=o.SemiMonthEnd()),
        m.date_range("2024-01-01", periods=3, freq=o.DateOffset(months=1, days=1)),
        m.date_range("2024-01-01", "2024-03-01", freq=o.WeekOfMonth(week=1, weekday=2)),
        m.date_range("2024-01-01", periods=3, freq=o.Hour(6)),
        m.date_range("2024-01-01", periods=3, freq=o.Day(2)),
        m.date_range("2024-01-31", "2024-04-30", freq=o.MonthEnd(), inclusive="neither"),
        m.date_range("2024-01-01", periods=2, freq=o.Easter(), tz="UTC"),
        m.date_range("2024-01-05 16:00", periods=3, freq=o.BusinessHour()),
    ]
    return [(str(index.dtype), [repr(v) for v in index.tolist()]) for index in made]


def plain(m: ModuleType) -> Any:
    return [
        repr(datetime.datetime(2024, 1, 5) + m.offsets.MonthEnd()),
        repr(datetime.date(2024, 1, 5) + m.offsets.MonthEnd()),
        repr(m.NaT + m.offsets.MonthEnd()),
        repr(m.offsets.MonthEnd() + m.NaT),
        repr(m.offsets.MonthEnd(2.0)),
        m.offsets.MonthEnd() == "ME",
        m.offsets.QuarterEnd() == "QE",
        m.offsets.QuarterEnd(startingMonth=12) == "QE",
        m.offsets.Week(weekday=6) == "W",
        m.offsets.Hour(2) == "2h",
        m.offsets.MonthEnd() == 1,
        m.offsets.Week() == m.offsets.Week(weekday=None),
        isinstance(m.offsets.MonthEnd(), m.offsets.DateOffset),
        isinstance(m.offsets.Hour(), m.offsets.Tick),
        isinstance(m.offsets.Day(), m.offsets.Tick),
        m.offsets.DateOffset(months=2, day=3).kwds,
        m.offsets.DateOffset(months=2).months,
        repr(
            m.offsets.FY5253(weekday=4, startingMonth=12).get_year_end(
                datetime.datetime(2024, 5, 1)
            )
        ),
        m.offsets.FY5253Quarter().get_weeks(m.Timestamp("2024-05-01")),
        m.offsets.FY5253Quarter().year_has_extra_week(m.Timestamp("2024-05-01")),
        m.offsets.FY5253Quarter().year_has_extra_week(m.Timestamp("2021-05-01")),
        repr(m.offsets.BusinessHour().next_bday),
        m.offsets.CDay(weekmask="1010100").weekmask,
        m.offsets.BMonthEnd().is_month_end(m.Timestamp("2024-03-29")),
        m.offsets.MonthEnd().is_month_start(m.Timestamp("2024-03-01")),
        m.DateOffset is m.offsets.DateOffset,
    ]


SCENES = {
    "normalized": normalized,
    "zoned": zoned,
    "ticks": ticks,
    "units": units,
    "columns": columns,
    "ranges": ranges,
    "plain": plain,
}


@needs_pandas
@pytest.mark.parametrize("name", list(SCENES))
def test_offsets_in_use_match_pandas(firepanda: ModuleType, name: str) -> None:
    """Normalizing, zones, fixed steps, units, columns, ranges and comparisons."""
    import pandas as pd

    assert SCENES[name](firepanda) == SCENES[name](pd)


REFUSED: dict[str, Callable[[ModuleType], Any]] = {
    "fraction of a step": lambda m: m.offsets.MonthEnd(1.5),
    "text as a step": lambda m: m.offsets.MonthEnd("2"),
    "unknown part": lambda m: m.offsets.DateOffset(foo=1),
    "fraction of a month": lambda m: m.offsets.DateOffset(months=1.5),
    "normalized tick": lambda m: m.offsets.Hour(normalize=True),
    "week past three": lambda m: m.offsets.WeekOfMonth(week=4),
    "weekday past six": lambda m: m.offsets.Week(weekday=7),
    "month past twelve": lambda m: m.offsets.YearEnd(month=13),
    "day of month past 27": lambda m: m.offsets.SemiMonthEnd(day_of_month=28),
    "semi month begin on the first": lambda m: m.offsets.SemiMonthBegin(day_of_month=1),
    "variation": lambda m: m.offsets.FY5253(variation="middle"),
    "fiscal year of no steps": lambda m: m.offsets.FY5253(0),
    "adding two offsets": lambda m: m.offsets.MonthEnd() + m.offsets.MonthEnd(),
    "offset less a moment": lambda m: m.offsets.MonthEnd() - m.Timestamp("2024-01-01"),
    "hours that overlap": lambda m: m.offsets.BusinessHour(
        start=["09:00", "11:00"], end=["12:00", "17:00"]
    ),
    "hours with seconds": lambda m: m.offsets.BusinessHour(start="09:00:30"),
    "uneven hours": lambda m: m.offsets.BusinessHour(start=["09:00", "13:00"], end="17:00"),
    "no fixed length": lambda m: m.offsets.MonthEnd().nanos,
    "no prefix": lambda m: m.offsets.DateOffset(months=1).rule_code,
    "a column less an offset the other way": lambda m: (
        m.offsets.MonthEnd() - m.Series(m.to_datetime(m.Series(["2024-01-01"])))
    ),
}


@needs_pandas
@pytest.mark.parametrize("name", list(REFUSED))
def test_what_pandas_refuses_is_refused(firepanda: ModuleType, name: str) -> None:
    """The same kind of error, and for a bad argument, pandas' words."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        REFUSED[name](pd)
    kind = next(k for k in (ValueError, TypeError, NotImplementedError) if theirs.errisinstance(k))
    with pytest.raises(kind) as mine:
        REFUSED[name](firepanda)
    if kind is ValueError:
        assert str(mine.value) == str(theirs.value)


def test_business_hours_over_midnight_are_refused(firepanda: ModuleType) -> None:
    """A working day that runs past midnight is the one shape of hours not built."""
    with pytest.raises(NotImplementedError):
        firepanda.offsets.BusinessHour(start="22:00", end="06:00")


def test_the_namespace_has_every_name_pandas_has(firepanda: ModuleType) -> None:
    """Every public class of `pandas.offsets`, and each one callable."""
    names = [
        "BDay", "BHalfYearBegin", "BHalfYearEnd", "BMonthBegin", "BMonthEnd", "BQuarterBegin",
        "BQuarterEnd", "BYearBegin", "BYearEnd", "BaseOffset", "BusinessDay", "BusinessHour",
        "BusinessMonthBegin", "BusinessMonthEnd", "CBMonthBegin", "CBMonthEnd", "CDay",
        "CustomBusinessDay", "CustomBusinessHour", "CustomBusinessMonthBegin",
        "CustomBusinessMonthEnd", "DateOffset", "Day", "Easter", "FY5253", "FY5253Quarter",
        "HalfYearBegin", "HalfYearEnd", "Hour", "LastWeekOfMonth", "Micro", "Milli", "Minute",
        "MonthBegin", "MonthEnd", "Nano", "QuarterBegin", "QuarterEnd", "Second",
        "SemiMonthBegin", "SemiMonthEnd", "Tick", "Week", "WeekOfMonth", "YearBegin", "YearEnd",
    ]  # fmt: skip
    assert all(callable(getattr(firepanda.offsets, name)) for name in names)


def test_easter_matches_dateutil() -> None:
    """The three methods of `dateutil.easter`, for every year from 1600 to 2400."""
    easter = pytest.importorskip("dateutil.easter")
    from firepanda.offsets import _easter

    for year in range(1600, 2401):
        for method in (1, 2, 3):
            assert _easter(year, method) == easter.easter(year, method)


def hourly(m: ModuleType) -> Any:
    return m.Series(range(6), index=m.date_range("2024-01-01", periods=6, freq="5h"))


ELSEWHERE: dict[str, Callable[[ModuleType], Any]] = {
    "resample by hours": lambda m: hourly(m).resample(m.offsets.Hour(12)).sum().tolist(),
    "resample by a day": lambda m: hourly(m).resample(m.offsets.Day()).sum().tolist(),
    "resample by minutes": lambda m: hourly(m).resample(m.offsets.Minute(300)).sum().tolist(),
    "span of hours": lambda m: (
        m.Timedelta(m.offsets.Hour(2)),
        m.Timedelta(m.offsets.Hour(2)).unit,
    ),
    "span of a step back": lambda m: m.Timedelta(m.offsets.Second(-1)),
    "span of nanoseconds": lambda m: (
        m.Timedelta(m.offsets.Nano(5)),
        m.Timedelta(m.offsets.Nano(5)).unit,
    ),
    "span of milliseconds": lambda m: m.Timedelta(m.offsets.Milli(3)).unit,
}

REFUSED_ELSEWHERE: dict[str, Callable[[ModuleType], Any]] = {
    "span of a day": lambda m: m.Timedelta(m.offsets.Day()),
    "span of a month end": lambda m: m.Timedelta(m.offsets.MonthEnd()),
    "resample by a number": lambda m: hourly(m).resample(5),
}


@needs_pandas
@pytest.mark.parametrize("name", list(ELSEWHERE))
def test_a_tick_is_read_where_pandas_reads_one(firepanda: ModuleType, name: str) -> None:
    """`resample` takes a tick or a day as its rule, and `Timedelta` reads a tick."""
    import pandas as pd

    assert repr(ELSEWHERE[name](firepanda)) == repr(ELSEWHERE[name](pd))


@needs_pandas
@pytest.mark.parametrize("name", list(REFUSED_ELSEWHERE))
def test_an_offset_pandas_does_not_read_raises_its_error(firepanda: ModuleType, name: str) -> None:
    """A day or a calendar offset is not a span, and a number is not a rule."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        REFUSED_ELSEWHERE[name](pd)
    kind = ValueError if theirs.errisinstance(ValueError) else TypeError
    with pytest.raises(kind) as mine:
        REFUSED_ELSEWHERE[name](firepanda)
    assert str(mine.value) == str(theirs.value)


def test_a_calendar_offset_as_a_rule_is_refused(firepanda: ModuleType) -> None:
    """Bins of a calendar offset are not all one length, which resample does not do yet."""
    with pytest.raises(NotImplementedError):
        hourly(firepanda).resample(firepanda.offsets.MonthEnd())
