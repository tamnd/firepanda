"""Date offsets, the steps of the calendar, which is `pandas.offsets`.

An offset is a step that a moment can be moved by. Some steps are fixed
lengths of time, like an hour, and the rest land on dates of the calendar, like
the last day of a month or the next business day. The rules below were measured
against pandas 3.0.

- An offset that lands on dates moves a moment to the next such date for each
  step forward, and to the previous one for each step back. A moment that is
  not on a landing date counts its first move as the step that reaches one, so
  `MonthEnd()` takes the fifteenth of January to its thirty first and the
  thirty first to the end of February. Zero steps rolls forward to a landing
  date and stays on one it is already on. The time of day rides along.
- Whether a moment is on a landing date is read from its date alone, unless the
  offset normalizes, when only midnight is on it.
- A fixed step (`Hour` down to `Nano`) adds elapsed time, so on a zoned moment
  it crosses a change of clocks as elapsed time. `Day` is a calendar day in
  pandas 3 and moves the wall clock, as every calendar offset does.
- `DateOffset` made only of fixed spans (days, hours and finer) adds elapsed
  time. With years, months or a value to set, it moves the wall clock the way
  `dateutil.relativedelta` does, clipping the day to the end of the month.
- Business hours count the working time between opening and closing. Going
  forward the answer lands in an opening, never on a closing, and going back it
  lands on a closing, never on an opening.
- An answer is quoted at microseconds or finer.

What is refused: business hours that run over midnight, and a holiday calendar
other than numpy's `busdaycalendar`.
"""

from __future__ import annotations

import bisect
import calendar as _calendar
import datetime
import inspect
import itertools
import re
from collections.abc import Callable
from typing import Any

from ._scalars import NaT, Timedelta, Timestamp

__all__ = [
    "FY5253",
    "BDay",
    "BHalfYearBegin",
    "BHalfYearEnd",
    "BMonthBegin",
    "BMonthEnd",
    "BQuarterBegin",
    "BQuarterEnd",
    "BYearBegin",
    "BYearEnd",
    "BaseOffset",
    "BusinessDay",
    "BusinessHour",
    "BusinessMonthBegin",
    "BusinessMonthEnd",
    "CBMonthBegin",
    "CBMonthEnd",
    "CDay",
    "CustomBusinessDay",
    "CustomBusinessHour",
    "CustomBusinessMonthBegin",
    "CustomBusinessMonthEnd",
    "DateOffset",
    "Day",
    "Easter",
    "FY5253Quarter",
    "HalfYearBegin",
    "HalfYearEnd",
    "Hour",
    "LastWeekOfMonth",
    "Micro",
    "Milli",
    "Minute",
    "MonthBegin",
    "MonthEnd",
    "Nano",
    "QuarterBegin",
    "QuarterEnd",
    "Second",
    "SemiMonthBegin",
    "SemiMonthEnd",
    "Tick",
    "Week",
    "WeekOfMonth",
    "YearBegin",
    "YearEnd",
]

_WEEKDAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
_ONE_DAY = datetime.timedelta(days=1)
_ZERO = datetime.timedelta(0)
_UNITS = {"s": 0, "ms": 1, "us": 2, "ns": 3}


def _whole(n: Any) -> int:
    """The number of steps, which pandas takes as a whole number or a float with no fraction."""
    if isinstance(n, int) and not isinstance(n, bool):
        return n
    if isinstance(n, float) and n.is_integer():
        return int(n)
    if hasattr(n, "__index__") and not isinstance(n, bool):
        return int(n)
    raise ValueError(f"`n` argument must be an integer, got {n}")


def _last_day(year: int, month: int) -> int:
    return _calendar.monthrange(year, month)[1]


def _stamp(value: Any) -> Any:
    """A moment as a `Timestamp`, or `NaT` for a gap."""
    if value is NaT or value is None:
        return NaT
    if isinstance(value, Timestamp):
        return value
    if type(value).__module__ == "numpy":
        # numpy's datetime64 reaches a plain datetime through its own `item`, at microseconds.
        text = str(value)
        return NaT if text == "NaT" else Timestamp(text)
    return Timestamp(value)


def _finer(unit: str, other: str) -> str:
    return max(unit, other, key=_UNITS.__getitem__)


def _placed(wall: datetime.datetime, nanos: int, zone: Any, unit: str) -> Timestamp:
    """A wall clock reading put back together as a `Timestamp` at the unit and zone given."""
    stamp = Timestamp(wall)
    if nanos:
        stamp = stamp + Timedelta._from_nanos(nanos, "ns")
    if stamp.unit != unit:
        stamp = stamp.as_unit(unit)
    if zone is not None:
        stamp = stamp.tz_localize(zone)
    return stamp


class BaseOffset:
    """The base of every offset, which is `pandas.offsets.BaseOffset`."""

    _prefix: str | None = None
    _shown_name: str | None = None
    _shown_params: tuple[str, ...] = ()
    # Whether the step is elapsed time, added to a zoned moment as it is, rather than a
    # move of the wall clock.
    _elapsed = False

    def __init__(self, n: Any = 1, normalize: bool = False) -> None:
        self.n = _whole(n)
        self.normalize = bool(normalize)

    # What an offset is.

    def _arguments(self) -> dict[str, Any]:
        """The keyword arguments that build this offset again, beside n and normalize."""
        return {}

    @property
    def kwds(self) -> dict[str, Any]:
        """The offset's own parameters."""
        return self._arguments()

    def _key(self) -> tuple[Any, ...]:
        arguments = tuple(sorted((k, _hashable(v)) for k, v in self._arguments().items()))
        return (type(self).__name__, self.n, self.normalize, arguments)

    def _with(self, n: int) -> Any:
        return type(self)(n, normalize=self.normalize, **self._arguments())

    def copy(self) -> Any:
        """The same offset."""
        return self._with(self.n)

    @property
    def base(self) -> Any:
        """The same offset with one step."""
        return self._with(1)

    @property
    def rule_code(self) -> str:
        """The code the offset is written with, without its count."""
        if self._prefix is None:
            raise NotImplementedError("Prefix not defined")
        return self._prefix + self._suffix()

    def _suffix(self) -> str:
        return ""

    @property
    def name(self) -> str:
        """The rule code."""
        return self.rule_code

    @property
    def freqstr(self) -> str:
        """The frequency text, the count then the rule code, or the repr when there is no code."""
        try:
            code = self.rule_code
        except NotImplementedError:
            return repr(self)
        code += self._freq_tail()
        return code if self.n == 1 else f"{self.n}{code}"

    def _freq_tail(self) -> str:
        return ""

    @property
    def nanos(self) -> int:
        """The length in nanoseconds, which only a fixed step has."""
        raise ValueError(f"{self!r} is a non-fixed frequency")

    def _repr_params(self) -> list[str]:
        values = self._arguments()
        return [f"{name}={values[name]}" for name in sorted(self._shown_params)]

    def __repr__(self) -> str:
        count = "" if self.n == 1 else f"{self.n} * "
        plural = "" if abs(self.n) == 1 else "s"
        params = self._repr_params()
        shown = f": {', '.join(params)}" if params else ""
        return f"<{count}{self._shown_name or type(self).__name__}{plural}{shown}>"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            try:
                other = _parsed(other)
            except ValueError:
                return False
        if isinstance(other, BaseOffset):
            return self._key() == other._key()
        return False

    def __ne__(self, other: Any) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(self._key())

    # Arithmetic.

    def __neg__(self) -> Any:
        return self._with(-self.n)

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, int) and not isinstance(other, bool):
            return self._with(self.n * other)
        return NotImplemented

    __rmul__ = __mul__

    def __add__(self, other: Any) -> Any:
        if other is NaT:
            return NaT
        if isinstance(other, datetime.date) or type(other).__name__ == "datetime64":
            return self._apply(_stamp(other))
        many = _many(self, other, 1)
        return NotImplemented if many is None else many

    __radd__ = __add__

    def __sub__(self, other: Any) -> Any:
        if isinstance(other, datetime.date):
            raise TypeError("Cannot subtract datetime from offset.")
        return NotImplemented

    def __rsub__(self, other: Any) -> Any:
        if other is NaT:
            return NaT
        if isinstance(other, datetime.date) or type(other).__name__ == "datetime64":
            return (-self)._apply(_stamp(other))
        many = _many(self, other, -1)
        return NotImplemented if many is None else many

    # Moving a moment.

    def _apply(self, stamp: Any) -> Any:
        """The moment moved by this offset."""
        if stamp is NaT:
            return NaT
        unit = _finer(stamp.unit, "us")
        if self._elapsed:
            moved = self._elapse(stamp)
            if moved.unit != _finer(unit, moved.unit):
                moved = moved.as_unit(_finer(unit, moved.unit))
            return moved.normalize() if self.normalize else moved
        zone = stamp.tzinfo
        wall = datetime.datetime(
            stamp.year,
            stamp.month,
            stamp.day,
            stamp.hour,
            stamp.minute,
            stamp.second,
            stamp.microsecond,
        )
        nanos = stamp.nanosecond
        moved = self._move(wall)
        if self.normalize:
            moved, nanos = datetime.datetime.combine(moved.date(), datetime.time()), 0
        return _placed(moved, nanos, zone, unit)

    def _move(self, wall: datetime.datetime) -> datetime.datetime:
        raise NotImplementedError(f"{type(self).__name__} does not move a moment")

    def _elapse(self, stamp: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__} does not move a moment")

    def _on(self, stamp: Any) -> bool:
        return True

    def is_on_offset(self, dt: Any) -> bool:
        """Whether a moment is one this offset lands on."""
        stamp = _stamp(dt)
        if self.normalize and (stamp.hour, stamp.minute, stamp.second, stamp.microsecond) != (
            0,
            0,
            0,
            0,
        ):
            return False
        return self._on(stamp)

    def rollforward(self, dt: Any) -> Any:
        """The moment itself when it is on the offset, and otherwise the next one that is."""
        stamp = _stamp(dt)
        if self.is_on_offset(stamp):
            return stamp
        return type(self)(1, normalize=self.normalize, **self._arguments())._apply(stamp)

    def rollback(self, dt: Any) -> Any:
        """The moment itself when it is on the offset, and otherwise the last one before it."""
        stamp = _stamp(dt)
        if self.is_on_offset(stamp):
            return stamp
        return type(self)(-1, normalize=self.normalize, **self._arguments())._apply(stamp)

    def _field(self, ts: Any, field: str) -> bool:
        """A start or end field of a moment, read by this offset's business days and months.

        pandas reads these the way the offset reads the calendar: a business offset
        counts its first and last business days, and an offset anchored on a month
        counts its quarters and years from that month.
        """
        stamp = _stamp(ts)
        try:
            name = self.name
        except NotImplementedError:
            name = ""
        anchor = self.kwds.get("startingMonth", self.kwds.get("month", 12))
        if name.lstrip("B")[0:2] in ("MS", "QS", "YS"):
            end_month, start_month = (12 if anchor == 1 else anchor - 1), anchor
        else:
            end_month, start_month = anchor, anchor % 12 + 1
        year, month = stamp.year, stamp.month
        if name.startswith("B"):
            first = _first_business(year, month).day == stamp.day
            last = _last_business(year, month).day == stamp.day
        else:
            first, last = stamp.day == 1, stamp.day == _last_day(year, month)
        return {
            "is_month_start": first,
            "is_month_end": last,
            "is_quarter_start": first and (month - start_month) % 3 == 0,
            "is_quarter_end": last and (month - end_month) % 3 == 0,
            "is_year_start": first and month == start_month,
            "is_year_end": last and month == end_month,
        }[field]

    def is_month_start(self, ts: Any) -> bool:
        """Whether a moment starts a month, as this offset counts months."""
        return self._field(ts, "is_month_start")

    def is_month_end(self, ts: Any) -> bool:
        """Whether a moment ends a month, as this offset counts months."""
        return self._field(ts, "is_month_end")

    def is_quarter_start(self, ts: Any) -> bool:
        """Whether a moment starts a quarter, as this offset counts quarters."""
        return self._field(ts, "is_quarter_start")

    def is_quarter_end(self, ts: Any) -> bool:
        """Whether a moment ends a quarter, as this offset counts quarters."""
        return self._field(ts, "is_quarter_end")

    def is_year_start(self, ts: Any) -> bool:
        """Whether a moment starts a year, as this offset counts years."""
        return self._field(ts, "is_year_start")

    def is_year_end(self, ts: Any) -> bool:
        """Whether a moment ends a year, as this offset counts years."""
        return self._field(ts, "is_year_end")


def _hashable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _many(offset: BaseOffset, other: Any, sign: int) -> Any:
    """A series or an index of moments moved by an offset, or None for anything else."""
    from ._datetime import DatetimeIndex
    from ._frame import Series

    if isinstance(other, Series):
        return _series_moved(other, offset if sign > 0 else -offset)
    if isinstance(other, DatetimeIndex):
        moved = _series_moved(other.to_series(), offset if sign > 0 else -offset)
        return DatetimeIndex(moved, name=other.name)
    return None


def _series_moved(column: Any, offset: BaseOffset) -> Any:
    """A column of moments moved one by one, which keeps its labels and its name."""
    from ._frame import Series

    kind = str(column.dtype)
    if not kind.startswith("datetime64"):
        raise TypeError(f"cannot add {type(offset).__name__} to a column of {kind}")
    if isinstance(offset, Tick):
        return column + Timedelta._from_nanos(offset.nanos, offset._unit)
    moved = [offset._apply(value) for value in column.tolist()]
    unit = "ns" if kind.startswith("datetime64[ns") else "us"
    zone = kind[kind.index(",") + 2 : -1] if "," in kind else None
    answer = Series(moved, index=column.index, name=column.name)
    wanted = f"datetime64[{unit}, {zone}]" if zone else f"datetime64[{unit}]"
    return answer if str(answer.dtype) == wanted else answer.astype(wanted)


# Offsets that land on dates.


class _Landing(BaseOffset):
    """An offset that lands on dates of the calendar, from the dates each year holds."""

    def _dates(self, year: int) -> list[datetime.date]:
        raise NotImplementedError

    def _in_year(self, year: int) -> list[datetime.date]:
        """The landing dates in one calendar year, in order."""
        found = self.__dict__.setdefault("_years", {})
        dates = found.get(year)
        if dates is None:
            near = [d for y in (year - 1, year, year + 1) for d in self._dates(y)]
            dates = sorted({d for d in near if d.year == year})
            found[year] = dates
        return dates

    def _after(self, day: datetime.date) -> datetime.date:
        year = day.year
        while year <= 9999:
            dates = self._in_year(year)
            at = bisect.bisect_right(dates, day)
            if at < len(dates):
                return dates[at]
            year += 1
        raise OverflowError("offset moved past the last date a calendar reaches")

    def _before(self, day: datetime.date) -> datetime.date:
        year = day.year
        while year >= 1:
            dates = self._in_year(year)
            at = bisect.bisect_left(dates, day)
            if at > 0:
                return dates[at - 1]
            year -= 1
        raise OverflowError("offset moved past the first date a calendar reaches")

    def _landed(self, day: datetime.date) -> bool:
        dates = self._in_year(day.year)
        at = bisect.bisect_left(dates, day)
        return at < len(dates) and dates[at] == day

    def _move(self, wall: datetime.datetime) -> datetime.datetime:
        day = wall.date()
        if self.n == 0 and not self._landed(day):
            day = self._after(day)
        for _ in range(self.n):
            day = self._after(day)
        for _ in range(-self.n):
            day = self._before(day)
        return datetime.datetime.combine(day, wall.time()) + self._shift()

    def _shift(self) -> datetime.timedelta:
        return _ZERO

    def _on(self, stamp: Any) -> bool:
        return self._landed(datetime.date(stamp.year, stamp.month, stamp.day))


def _weekday(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 6:
        raise ValueError(f"Day must be 0<=day<=6, got {value}")
    return value


def _month(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 12:
        raise ValueError("Month must go from 1 to 12")
    return value


def _first_business(year: int, month: int) -> datetime.date:
    day = datetime.date(year, month, 1)
    while day.weekday() >= 5:
        day += _ONE_DAY
    return day


def _last_business(year: int, month: int) -> datetime.date:
    day = datetime.date(year, month, _last_day(year, month))
    while day.weekday() >= 5:
        day -= _ONE_DAY
    return day


class _Monthly(_Landing):
    """The first or last day, plain or business, of every month, quarter, half year or year."""

    _every = 1
    _first = False
    _business = False
    _anchor: str | None = None
    _default = 12

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Each class names its own anchor, as pandas' classes do, so it can be passed by
        # position and shows in the signature.
        super().__init_subclass__(**kwargs)
        shown = [
            inspect.Parameter("n", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=1),
            inspect.Parameter("normalize", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=False),
        ]
        if cls._anchor is not None:
            kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
            shown.append(inspect.Parameter(cls._anchor, kind, default=None))
        cls.__signature__ = inspect.Signature(shown)

    def __init__(self, n: Any = 1, normalize: bool = False, *by: Any, **anchor: Any) -> None:
        if len(by) > (self._anchor is not None):
            taken = 2 + (self._anchor is not None)
            raise TypeError(
                f"{type(self).__name__}() takes at most {taken} positional arguments"
                f" ({2 + len(by)} given)"
            )
        if by:
            if self._anchor in anchor:
                raise TypeError(
                    f"{type(self).__name__}() got multiple values for keyword argument"
                    f" '{self._anchor}'"
                )
            anchor[self._anchor] = by[0]
        super().__init__(n, normalize)
        if self._anchor is not None:
            given = anchor.pop(self._anchor, None)
            setattr(self, self._anchor, self._default if given is None else _month(given))
        if anchor:
            name = next(iter(anchor))
            raise TypeError(f"{type(self).__name__}() got an unexpected keyword argument '{name}'")

    def _arguments(self) -> dict[str, Any]:
        return {} if self._anchor is None else {self._anchor: getattr(self, self._anchor)}

    @property
    def _shown_params(self) -> tuple[str, ...]:  # type: ignore[override]
        return () if self._anchor is None else (self._anchor,)

    def _suffix(self) -> str:
        return "" if self._anchor is None else "-" + _MONTHS[getattr(self, self._anchor) - 1]

    def _dates(self, year: int) -> list[datetime.date]:
        start = 1 if self._anchor is None else getattr(self, self._anchor)
        months = [m for m in range(1, 13) if (m - start) % self._every == 0]
        if self._business:
            pick = _first_business if self._first else _last_business
            return [pick(year, m) for m in months]
        return [datetime.date(year, m, 1 if self._first else _last_day(year, m)) for m in months]


class MonthEnd(_Monthly):
    """The last day of each month, which is `pandas.offsets.MonthEnd`."""

    _prefix = "ME"


class MonthBegin(_Monthly):
    """The first day of each month, which is `pandas.offsets.MonthBegin`."""

    _prefix = "MS"
    _first = True


class BusinessMonthEnd(_Monthly):
    """The last business day of each month, which is `pandas.offsets.BusinessMonthEnd`."""

    _prefix = "BME"
    _business = True


class BusinessMonthBegin(_Monthly):
    """The first business day of each month, which is `pandas.offsets.BusinessMonthBegin`."""

    _prefix = "BMS"
    _business = True
    _first = True


class QuarterEnd(_Monthly):
    """The last day of each quarter, which is `pandas.offsets.QuarterEnd`."""

    _prefix = "QE"
    _every = 3
    _anchor = "startingMonth"
    _default = 3


class QuarterBegin(_Monthly):
    """The first day of each quarter, which is `pandas.offsets.QuarterBegin`."""

    _prefix = "QS"
    _every = 3
    _first = True
    _anchor = "startingMonth"
    _default = 3


class BQuarterEnd(_Monthly):
    """The last business day of each quarter, which is `pandas.offsets.BQuarterEnd`."""

    _prefix = "BQE"
    _shown_name = "BusinessQuarterEnd"
    _every = 3
    _business = True
    _anchor = "startingMonth"
    _default = 3


class BQuarterBegin(_Monthly):
    """The first business day of each quarter, which is `pandas.offsets.BQuarterBegin`."""

    _prefix = "BQS"
    _shown_name = "BusinessQuarterBegin"
    _every = 3
    _first = True
    _business = True
    _anchor = "startingMonth"
    _default = 3


class HalfYearEnd(_Monthly):
    """The last day of each half year, which is `pandas.offsets.HalfYearEnd`."""

    _prefix = "HYE"
    _every = 6
    _anchor = "startingMonth"
    _default = 6


class HalfYearBegin(_Monthly):
    """The first day of each half year, which is `pandas.offsets.HalfYearBegin`."""

    _prefix = "HYS"
    _every = 6
    _first = True
    _anchor = "startingMonth"
    _default = 1


class BHalfYearEnd(_Monthly):
    """The last business day of each half year, which is `pandas.offsets.BHalfYearEnd`."""

    _prefix = "BHYE"
    _shown_name = "BusinessHalfYearEnd"
    _every = 6
    _business = True
    _anchor = "startingMonth"
    _default = 6


class BHalfYearBegin(_Monthly):
    """The first business day of each half year, which is `pandas.offsets.BHalfYearBegin`."""

    _prefix = "BHYS"
    _shown_name = "BusinessHalfYearBegin"
    _every = 6
    _first = True
    _business = True
    _anchor = "startingMonth"
    _default = 1


class YearEnd(_Monthly):
    """The last day of each year, which is `pandas.offsets.YearEnd`."""

    _prefix = "YE"
    _every = 12
    _anchor = "month"
    _default = 12


class YearBegin(_Monthly):
    """The first day of each year, which is `pandas.offsets.YearBegin`."""

    _prefix = "YS"
    _every = 12
    _first = True
    _anchor = "month"
    _default = 1


class BYearEnd(_Monthly):
    """The last business day of each year, which is `pandas.offsets.BYearEnd`."""

    _prefix = "BYE"
    _every = 12
    _business = True
    _anchor = "month"
    _default = 12


class BYearBegin(_Monthly):
    """The first business day of each year, which is `pandas.offsets.BYearBegin`."""

    _prefix = "BYS"
    _every = 12
    _first = True
    _business = True
    _anchor = "month"
    _default = 1


class SemiMonthEnd(_Landing):
    """A day of each month and its last day, which is `pandas.offsets.SemiMonthEnd`."""

    _prefix = "SME"
    _shown_params = ("day_of_month",)
    _lowest = 1

    def __init__(self, n: Any = 1, normalize: bool = False, day_of_month: Any = None) -> None:
        super().__init__(n, normalize)
        day = 15 if day_of_month is None else day_of_month
        if not isinstance(day, int) or not self._lowest <= day <= 27:
            raise ValueError(f"day_of_month must be {self._lowest}<=day_of_month<=27, got {day}")
        self.day_of_month = day

    def _arguments(self) -> dict[str, Any]:
        return {"day_of_month": self.day_of_month}

    def _suffix(self) -> str:
        return f"-{self.day_of_month}"

    def _dates(self, year: int) -> list[datetime.date]:
        return [
            datetime.date(year, month, day)
            for month in range(1, 13)
            for day in (self.day_of_month, _last_day(year, month))
        ]


class SemiMonthBegin(SemiMonthEnd):
    """The first of each month and a day in it, which is `pandas.offsets.SemiMonthBegin`."""

    _prefix = "SMS"
    _lowest = 2

    def _dates(self, year: int) -> list[datetime.date]:
        return [
            datetime.date(year, month, day)
            for month in range(1, 13)
            for day in (1, self.day_of_month)
        ]


class Week(_Landing):
    """Seven days, or one weekday of every week, which is `pandas.offsets.Week`."""

    _prefix = "W"
    _shown_params = ("weekday",)

    def __init__(self, n: Any = 1, normalize: bool = False, weekday: Any = None) -> None:
        super().__init__(n, normalize)
        self.weekday = None if weekday is None else _weekday(weekday)

    def _arguments(self) -> dict[str, Any]:
        return {"weekday": self.weekday}

    @property
    def kwds(self) -> dict[str, Any]:
        return {} if self.weekday is None else {"weekday": self.weekday}

    def _suffix(self) -> str:
        return "" if self.weekday is None else "-" + _WEEKDAYS[self.weekday]

    def _dates(self, year: int) -> list[datetime.date]:
        day = datetime.date(year, 1, 1)
        day += datetime.timedelta(days=(self.weekday - day.weekday()) % 7)
        dates = []
        while day.year == year:
            dates.append(day)
            day += datetime.timedelta(days=7)
        return dates

    def _move(self, wall: datetime.datetime) -> datetime.datetime:
        if self.weekday is None:
            return wall + datetime.timedelta(days=7 * self.n)
        return super()._move(wall)

    def _on(self, stamp: Any) -> bool:
        return self.weekday is None or super()._on(stamp)


class WeekOfMonth(_Landing):
    """One weekday of one week of each month, which is `pandas.offsets.WeekOfMonth`."""

    _prefix = "WOM"
    _shown_params = ("week", "weekday")

    def __init__(
        self, n: Any = 1, normalize: bool = False, week: Any = 0, weekday: Any = 0
    ) -> None:
        super().__init__(n, normalize)
        if not isinstance(week, int) or not 0 <= week <= 3:
            raise ValueError(f"Week must be 0<=week<=3, got {week}")
        self.week = week
        self.weekday = _weekday(weekday)

    def _arguments(self) -> dict[str, Any]:
        return {"week": self.week, "weekday": self.weekday}

    def _suffix(self) -> str:
        return f"-{self.week + 1}{_WEEKDAYS[self.weekday]}"

    def _dates(self, year: int) -> list[datetime.date]:
        dates = []
        for month in range(1, 13):
            first = datetime.date(year, month, 1)
            ahead = (self.weekday - first.weekday()) % 7 + 7 * self.week
            dates.append(first + datetime.timedelta(days=ahead))
        return dates


class LastWeekOfMonth(_Landing):
    """One weekday of the last week of each month, which is `pandas.offsets.LastWeekOfMonth`."""

    _prefix = "LWOM"
    _shown_params = ("weekday",)

    def __init__(self, n: Any = 1, normalize: bool = False, weekday: Any = 0) -> None:
        super().__init__(n, normalize)
        if self.n == 0:
            raise ValueError("N cannot be 0")
        self.weekday = _weekday(weekday)

    def _arguments(self) -> dict[str, Any]:
        return {"weekday": self.weekday}

    def _suffix(self) -> str:
        return "-" + _WEEKDAYS[self.weekday]

    def _dates(self, year: int) -> list[datetime.date]:
        dates = []
        for month in range(1, 13):
            last = datetime.date(year, month, _last_day(year, month))
            dates.append(last - datetime.timedelta(days=(last.weekday() - self.weekday) % 7))
        return dates


def _easter(year: int, method: int) -> datetime.date:
    """Easter Sunday, by the three methods `dateutil.easter` has.

    Method 1 is the Julian date as it stands, method 2 is the Orthodox date moved
    onto the Gregorian calendar, and method 3 is the Western date.
    """
    g = year % 19
    e = 0
    if method < 3:
        i = (19 * g + 15) % 30
        j = (year + year // 4 + i) % 7
        if method == 2:
            e = 10
            if year > 1600:
                e = e + year // 100 - 16 - (year // 100 - 16) // 4
    else:
        c = year // 100
        h = (c - c // 4 - (8 * c + 13) // 25 + 19 * g + 15) % 30
        i = h - (h // 28) * (1 - (h // 28) * (29 // (h + 1)) * ((21 - g) // 11))
        j = (year + year // 4 + i + 2 - c + c // 4) % 7
    p = i - j + e
    day = 1 + (p + 27 + (p + 6) // 40) % 31
    month = 3 + (p + 26) // 30
    return datetime.date(year, month, day)


class Easter(_Landing):
    """Easter Sunday of each year, which is `pandas.offsets.Easter`."""

    _shown_params = ("method",)

    def __init__(self, n: Any = 1, normalize: bool = False, method: Any = 3) -> None:
        super().__init__(n, normalize)
        if method not in (1, 2, 3):
            raise ValueError("invalid method")
        self.method = method

    def _arguments(self) -> dict[str, Any]:
        return {"method": self.method}

    def _dates(self, year: int) -> list[datetime.date]:
        return [_easter(year, self.method)]

    def _move(self, wall: datetime.datetime) -> datetime.datetime:
        # pandas counts from this year's Easter, so zero steps before it goes back a year.
        n = self.n
        this = datetime.datetime.combine(_easter(wall.year, self.method), datetime.time())
        if n >= 0 and wall < this:
            n -= 1
        elif n < 0 and wall > this:
            n += 1
        return datetime.datetime.combine(_easter(wall.year + n, self.method), wall.time())


class FY5253(_Landing):
    """A fiscal year of 52 or 53 weeks that ends on a weekday, which is `pandas.offsets.FY5253`.

    The year ends on the last `weekday` of `startingMonth`, or with
    `variation="nearest"` on the one nearest the last day of that month, which
    can be a few days into the next month.
    """

    _prefix = "RE"
    _shown_params = ("startingMonth", "variation", "weekday")

    def __init__(
        self,
        n: Any = 1,
        normalize: bool = False,
        weekday: Any = 0,
        startingMonth: Any = 1,
        variation: Any = "nearest",
    ) -> None:
        super().__init__(n, normalize)
        if self.n == 0:
            raise ValueError("N cannot be 0")
        if variation not in ("nearest", "last"):
            raise ValueError(f"{variation} is not a valid variation")
        self.weekday = _weekday(weekday)
        self.startingMonth = _month(startingMonth)
        self.variation = variation

    def _arguments(self) -> dict[str, Any]:
        return {
            "weekday": self.weekday,
            "startingMonth": self.startingMonth,
            "variation": self.variation,
        }

    def get_rule_code_suffix(self) -> str:
        """The part of the rule code after its prefix."""
        letter = "N" if self.variation == "nearest" else "L"
        return f"{letter}-{_MONTHS[self.startingMonth - 1]}-{_WEEKDAYS[self.weekday]}"

    def _suffix(self) -> str:
        return "-" + self.get_rule_code_suffix()

    def _end_of(self, year: int) -> datetime.date:
        last = datetime.date(year, self.startingMonth, _last_day(year, self.startingMonth))
        back = (last.weekday() - self.weekday) % 7
        if self.variation == "last" or back <= 3:
            return last - datetime.timedelta(days=back)
        return last + datetime.timedelta(days=7 - back)

    def _dates(self, year: int) -> list[datetime.date]:
        return [self._end_of(year)]

    def get_year_end(self, dt: Any) -> datetime.datetime:
        """The end of the fiscal year named by the year of a moment."""
        return datetime.datetime.combine(self._end_of(dt.year), datetime.time())


class FY5253Quarter(_Landing):
    """The quarters of a 52 or 53 week fiscal year, which is `pandas.offsets.FY5253Quarter`.

    Each quarter is thirteen weeks, and in a year of 53 weeks the quarter named by
    `qtr_with_extra_week` has fourteen.
    """

    _prefix = "REQ"
    _shown_params = ("qtr_with_extra_week", "startingMonth", "variation", "weekday")

    def __init__(
        self,
        n: Any = 1,
        normalize: bool = False,
        weekday: Any = 0,
        startingMonth: Any = 1,
        qtr_with_extra_week: Any = 1,
        variation: Any = "nearest",
    ) -> None:
        super().__init__(n, normalize)
        if self.n == 0:
            raise ValueError("N cannot be 0")
        self._year = FY5253(1, False, weekday, startingMonth, variation)
        self.weekday = self._year.weekday
        self.startingMonth = self._year.startingMonth
        self.variation = variation
        if qtr_with_extra_week not in (1, 2, 3, 4):
            raise ValueError(f"qtr_with_extra_week must be 1 to 4, got {qtr_with_extra_week}")
        self.qtr_with_extra_week = qtr_with_extra_week

    def _arguments(self) -> dict[str, Any]:
        return {
            "weekday": self.weekday,
            "startingMonth": self.startingMonth,
            "qtr_with_extra_week": self.qtr_with_extra_week,
            "variation": self.variation,
        }

    def get_rule_code_suffix(self) -> str:
        """The part of the rule code after its prefix."""
        return f"{self._year.get_rule_code_suffix()}-{self.qtr_with_extra_week}"

    def _suffix(self) -> str:
        return "-" + self.get_rule_code_suffix()

    def _weeks(self, year: int) -> list[int]:
        weeks = [13, 13, 13, 13]
        if (self._year._end_of(year) - self._year._end_of(year - 1)).days == 371:
            weeks[self.qtr_with_extra_week - 1] = 14
        return weeks

    def _dates(self, year: int) -> list[datetime.date]:
        day = self._year._end_of(year - 1)
        dates = []
        for weeks in self._weeks(year):
            day += datetime.timedelta(weeks=weeks)
            dates.append(day)
        return dates

    def _fiscal_year(self, dt: Any) -> int:
        day = datetime.date(dt.year, dt.month, dt.day)
        year = dt.year
        while self._year._end_of(year) < day:
            year += 1
        while self._year._end_of(year - 1) >= day:
            year -= 1
        return year

    def year_has_extra_week(self, dt: Any) -> bool:
        """Whether the fiscal year a moment is in has 53 weeks."""
        year = self._fiscal_year(dt)
        return (self._year._end_of(year) - self._year._end_of(year - 1)).days == 371

    def get_weeks(self, dt: Any) -> list[int]:
        """The weeks in each quarter of the fiscal year a moment is in."""
        return self._weeks(self._fiscal_year(dt))


# Business days, plain and custom.

_DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _mask(weekmask: Any) -> tuple[bool, ...]:
    """The seven working days of a week mask given as names, as seven digits or as a list."""
    if isinstance(weekmask, str):
        text = weekmask.strip()
        if re.fullmatch(r"[01]{7}", text):
            days = tuple(ch == "1" for ch in text)
        else:
            names = text.split()
            unknown = [name for name in names if name not in _DAY_NAMES]
            if unknown:
                raise ValueError(f"Invalid weekday name {unknown[0]!r} in weekmask")
            days = tuple(name in names for name in _DAY_NAMES)
    else:
        days = tuple(bool(flag) for flag in weekmask)
        if len(days) != 7:
            raise ValueError("A business day weekmask array must have length 7")
    if not any(days):
        raise ValueError("Cannot construct a numpy busdaycal with a weekmask of all zeros")
    return days


def _holidays(holidays: Any) -> tuple[datetime.date, ...]:
    if holidays is None:
        return ()
    days = set()
    for value in holidays:
        stamp = _stamp(value)
        days.add(datetime.date(stamp.year, stamp.month, stamp.day))
    return tuple(sorted(days))


def _shown_holidays(days: tuple[datetime.date, ...]) -> tuple[Any, ...]:
    try:
        import numpy
    except ImportError:
        return days
    return tuple(numpy.datetime64(day.isoformat(), "D") for day in days)


def _offset_of(offset: Any) -> datetime.timedelta:
    if offset is None:
        return _ZERO
    if not isinstance(offset, datetime.timedelta):
        raise TypeError(f"offset must be a timedelta, got {type(offset).__name__}")
    return offset


class _Working:
    """Which days are worked, the plain five days or a custom mask with holidays."""

    _days: tuple[bool, ...] = (True, True, True, True, True, False, False)
    _off: tuple[datetime.date, ...] = ()

    def _custom(self, weekmask: Any, holidays: Any, calendar: Any) -> None:
        if calendar is not None:
            if not (hasattr(calendar, "weekmask") and hasattr(calendar, "holidays")):
                raise NotImplementedError(
                    "a holiday calendar other than numpy's busdaycalendar is not supported"
                )
            if weekmask == "Mon Tue Wed Thu Fri":
                weekmask = list(calendar.weekmask)
            holidays = list(holidays or []) + [str(day) for day in calendar.holidays]
        self._weekmask_given = weekmask
        self._days = _mask(weekmask)
        self._off = _holidays(holidays)
        self._calendar = calendar

    @property
    def weekmask(self) -> Any:
        """The working days of the week, as they were given."""
        return self._weekmask_given

    @property
    def holidays(self) -> tuple[Any, ...]:
        """The days off, in order."""
        return _shown_holidays(self._off)

    @property
    def calendar(self) -> Any:
        """The holiday calendar given, if there was one."""
        return self._calendar

    def _worked(self, day: datetime.date) -> bool:
        if not self._days[day.weekday()]:
            return False
        if not self._off:
            return True
        at = bisect.bisect_left(self._off, day)
        return not (at < len(self._off) and self._off[at] == day)

    def _worked_dates(self, year: int, month: int | None = None) -> list[datetime.date]:
        months = range(1, 13) if month is None else (month,)
        return [
            datetime.date(year, m, d)
            for m in months
            for d in range(1, _last_day(year, m) + 1)
            if self._worked(datetime.date(year, m, d))
        ]


class BusinessDay(_Working, _Landing):
    """The next working day, Monday to Friday, which is `pandas.offsets.BusinessDay`."""

    _prefix = "B"

    def __init__(self, n: Any = 1, normalize: bool = False, offset: Any = _ZERO) -> None:
        super().__init__(n, normalize)
        self.offset = _offset_of(offset)

    def _arguments(self) -> dict[str, Any]:
        return {"offset": self.offset}

    def _repr_params(self) -> list[str]:
        return [f"offset={self.offset!r}"] if self.offset else []

    def _freq_tail(self) -> str:
        if not self.offset:
            return ""
        sign, span = ("+", self.offset) if self.offset >= _ZERO else ("-", -self.offset)
        return sign + _span_text(span)

    def _shift(self) -> datetime.timedelta:
        return self.offset

    def _dates(self, year: int) -> list[datetime.date]:
        return self._worked_dates(year)

    def _landed(self, day: datetime.date) -> bool:
        return self._worked(day)


def _span_text(span: datetime.timedelta) -> str:
    """A span the way pandas writes the offset of a business day into its frequency text."""
    text = f"{span.days}D" if span.days > 0 else ""
    hours, rest = divmod(span.seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    text += f"{hours}h" if hours else ""
    text += f"{minutes}Min" if minutes else ""
    text += f"{seconds}s" if seconds else ""
    return text + (f"{span.microseconds}us" if span.microseconds else "")


class CustomBusinessDay(BusinessDay):
    """The next working day of a week mask, less holidays, which is `CustomBusinessDay`."""

    _prefix = "C"

    def __init__(
        self,
        n: Any = 1,
        normalize: bool = False,
        weekmask: Any = "Mon Tue Wed Thu Fri",
        holidays: Any = None,
        calendar: Any = None,
        offset: Any = _ZERO,
    ) -> None:
        super().__init__(n, normalize, offset)
        self._custom(weekmask, holidays, calendar)

    def _arguments(self) -> dict[str, Any]:
        return {
            "weekmask": self.weekmask,
            "holidays": self._off,
            "calendar": self.calendar,
            "offset": self.offset,
        }

    @property
    def kwds(self) -> dict[str, Any]:
        return {**self._arguments(), "holidays": self.holidays}

    def _on(self, stamp: Any) -> Any:
        # pandas asks numpy's `is_busday`, so the answer is numpy's own true or false.
        answer = super()._on(stamp)
        try:
            import numpy
        except ImportError:
            return answer
        return numpy.bool_(answer)


class _CustomMonthly(_Working, _Landing):
    _first = False

    def __init__(
        self,
        n: Any = 1,
        normalize: bool = False,
        weekmask: Any = "Mon Tue Wed Thu Fri",
        holidays: Any = None,
        calendar: Any = None,
        offset: Any = _ZERO,
    ) -> None:
        super().__init__(n, normalize)
        self.offset = _offset_of(offset)
        self._custom(weekmask, holidays, calendar)

    _arguments = CustomBusinessDay._arguments
    kwds = CustomBusinessDay.kwds
    _repr_params = BusinessDay._repr_params
    _shift = BusinessDay._shift

    @property
    def _shown_name(self) -> str:  # type: ignore[override]
        return type(self).__name__

    def _dates(self, year: int) -> list[datetime.date]:
        dates = []
        for month in range(1, 13):
            worked = self._worked_dates(year, month)
            if worked:
                dates.append(worked[0] if self._first else worked[-1])
        return dates


class CustomBusinessMonthEnd(_CustomMonthly):
    """The last custom working day of each month, which is `CustomBusinessMonthEnd`."""

    _prefix = "CBME"


class CustomBusinessMonthBegin(_CustomMonthly):
    """The first custom working day of each month, which is `CustomBusinessMonthBegin`."""

    _prefix = "CBMS"
    _first = True


# Business hours.


def _clock(value: Any) -> datetime.time:
    if isinstance(value, datetime.time):
        clock = value
    else:
        try:
            clock = datetime.datetime.strptime(str(value), "%H:%M").time()
        except ValueError:
            raise ValueError("time data must match '%H:%M' format") from None
    if clock.second or clock.microsecond:
        raise ValueError("time data must be specified only with hour and minute")
    return clock


def _clocks(value: Any) -> tuple[datetime.time, ...]:
    values = [value] if isinstance(value, str | datetime.time) else list(value)
    if not values:
        raise ValueError("Must include at least 1 start time")
    return tuple(_clock(one) for one in values)


class BusinessHour(_Working, BaseOffset):
    """Hours of work, which is `pandas.offsets.BusinessHour`."""

    _prefix = "bh"

    def __init__(
        self,
        n: Any = 1,
        normalize: bool = False,
        start: Any = "09:00",
        end: Any = "17:00",
        offset: Any = _ZERO,
    ) -> None:
        super().__init__(n, normalize)
        self._set_hours(start, end)
        self.offset = _offset_of(offset)

    def _set_hours(self, start: Any, end: Any) -> None:
        opens, closes = _clocks(start), _clocks(end)
        if len(opens) != len(closes):
            raise ValueError("number of starting time and ending time must be the same")
        pairs = sorted(zip(opens, closes, strict=True))
        if any(close <= open_ for open_, close in pairs):
            raise NotImplementedError("business hours that run over midnight are not supported")
        for (_, close), (open_, _) in itertools.pairwise(pairs):
            if open_ <= close:
                raise ValueError(
                    "invalid starting and ending time(s): opening hours should not touch or"
                    " overlap with one another"
                )
        self.start = tuple(open_ for open_, _ in pairs)
        self.end = tuple(close for _, close in pairs)
        # The working stretches of a day, as microseconds after midnight.
        self._spans = [(_micros(open_), _micros(close)) for open_, close in pairs]
        self._length = sum(close - open_ for open_, close in self._spans)

    def _arguments(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "offset": self.offset}

    def _repr_params(self) -> list[str]:
        hours = ",".join(
            f"{open_.strftime('%H:%M')}-{close.strftime('%H:%M')}"
            for open_, close in zip(self.start, self.end, strict=True)
        )
        return [f"{self._prefix}={hours}"]

    def __repr__(self) -> str:
        # pandas writes the offset of business hours as a part of its own before the hours.
        shown = super().__repr__()
        if not self.offset:
            return shown
        at = shown.index(": ")
        return f"{shown[:at]}: offset={self.offset!r}{shown[at:]}"

    @property
    def next_bday(self) -> BusinessDay:
        """The business day step that goes the same way as this one."""
        return BusinessDay(1 if self.n >= 0 else -1)

    def _step_days(self, day: datetime.date, count: int) -> datetime.date:
        """A working day moved by a number of working days."""
        step = 1 if count > 0 else -1
        for _ in range(abs(count)):
            day += datetime.timedelta(days=step)
            while not self._worked(day):
                day += datetime.timedelta(days=step)
        return day

    def _position(self, day: datetime.date, micros: int, forward: bool) -> tuple[Any, int]:
        """A moment as a working day and the working time before it in that day.

        Going forward, a moment outside the hours moves on to the next opening,
        and a closing counts as outside. Going back, a moment outside them moves
        back to the last closing, and an opening counts as outside.
        """
        if self._worked(day):
            done = 0
            if forward:
                for open_, close in self._spans:
                    if micros < close:
                        return day, done + max(micros - open_, 0)
                    done += close - open_
            else:
                found = None
                for open_, close in self._spans:
                    if micros <= open_:
                        break
                    found = done + min(micros, close) - open_
                    done += close - open_
                if found is not None:
                    return day, found
        if forward:
            return self._step_days(day, 1), 0
        return self._step_days(day, -1), self._length

    def _wall(self, day: datetime.date, done: int, forward: bool) -> datetime.datetime:
        for open_, close in self._spans:
            width = close - open_
            if done < width or (not forward and done == width):
                return datetime.datetime.combine(day, datetime.time()) + datetime.timedelta(
                    microseconds=open_ + done
                )
            done -= width
        raise AssertionError("working time past the end of the day")

    def _move(self, wall: datetime.datetime) -> datetime.datetime:
        forward = self.n >= 0
        micros = _micros(wall.time())
        day, done = self._position(wall.date(), micros, forward)
        total = done + self.n * 3_600_000_000
        if forward:
            days, done = divmod(total, self._length)
        else:
            days = -((-total) // self._length) - 1
            done = total - days * self._length
        return self._wall(self._step_days(day, days), done, forward) + self.offset

    def _on(self, stamp: Any) -> bool:
        day = datetime.date(stamp.year, stamp.month, stamp.day)
        if not self._worked(day):
            return False
        micros = _micros(datetime.time(stamp.hour, stamp.minute, stamp.second, stamp.microsecond))
        return any(open_ <= micros <= close for open_, close in self._spans)

    def _opening(self, stamp: Any, forward: bool) -> Any:
        wall = datetime.datetime(
            stamp.year,
            stamp.month,
            stamp.day,
            stamp.hour,
            stamp.minute,
            stamp.second,
            stamp.microsecond,
        )
        day, done = self._position(wall.date(), _micros(wall.time()), forward)
        moved = self._wall(day, done, forward)
        return _placed(moved, 0, stamp.tzinfo, _finer(stamp.unit, "us"))

    def rollforward(self, dt: Any) -> Any:
        """The moment itself in working hours, and otherwise the next opening."""
        stamp = _stamp(dt)
        return stamp if self.is_on_offset(stamp) else self._opening(stamp, True)

    def rollback(self, dt: Any) -> Any:
        """The moment itself in working hours, and otherwise the last closing."""
        stamp = _stamp(dt)
        return stamp if self.is_on_offset(stamp) else self._opening(stamp, False)


def _micros(clock: datetime.time) -> int:
    return ((clock.hour * 60 + clock.minute) * 60 + clock.second) * 1_000_000 + clock.microsecond


class CustomBusinessHour(BusinessHour):
    """Hours of work on the days of a week mask, less holidays, which is `CustomBusinessHour`."""

    _prefix = "cbh"

    def __init__(
        self,
        n: Any = 1,
        normalize: bool = False,
        weekmask: Any = "Mon Tue Wed Thu Fri",
        holidays: Any = None,
        calendar: Any = None,
        start: Any = "09:00",
        end: Any = "17:00",
        offset: Any = _ZERO,
    ) -> None:
        super().__init__(n, normalize, start, end, offset)
        self._custom(weekmask, holidays, calendar)

    def _arguments(self) -> dict[str, Any]:
        return {
            "weekmask": self.weekmask,
            "holidays": self._off,
            "calendar": self.calendar,
            "start": self.start,
            "end": self.end,
            "offset": self.offset,
        }

    @property
    def kwds(self) -> dict[str, Any]:
        return {**self._arguments(), "holidays": self.holidays}

    @property
    def next_bday(self) -> CustomBusinessDay:
        """The custom business day step that goes the same way as this one."""
        return CustomBusinessDay(
            1 if self.n >= 0 else -1, weekmask=self.weekmask, holidays=self._off
        )


# Fixed steps.


class Day(BaseOffset):
    """A calendar day, which is `pandas.offsets.Day`.

    In pandas 3 a day is a step of the calendar and not a fixed 24 hours, so on a
    zoned moment it keeps the time on the wall clock across a change of clocks.
    """

    _prefix = "D"
    _unit = "us"

    @property
    def nanos(self) -> int:
        return self.n * 86_400_000_000_000

    def _move(self, wall: datetime.datetime) -> datetime.datetime:
        return wall + datetime.timedelta(days=self.n)

    def __add__(self, other: Any) -> Any:
        if isinstance(other, Day):
            return Day(self.n + other.n)
        if isinstance(other, Tick) or (
            isinstance(other, datetime.timedelta) and not isinstance(other, datetime.datetime)
        ):
            return Timedelta._from_nanos(self.nanos, "us") + _elapsed(other)
        return super().__add__(other)

    __radd__ = __add__

    def __sub__(self, other: Any) -> Any:
        if isinstance(other, Day):
            return Day(self.n - other.n)
        return super().__sub__(other)


class Tick(BaseOffset):
    """A fixed length of time, which is `pandas.offsets.Tick`."""

    _elapsed = True
    _step = 0
    _unit = "us"

    def __init__(self, n: Any = 1, normalize: bool = False) -> None:
        super().__init__(n, normalize)
        if self.normalize:
            raise ValueError("Tick offset with `normalize=True` are not allowed.")

    @property
    def nanos(self) -> int:
        if not self._step:
            raise ValueError(f"{self!r} is a non-fixed frequency")
        return self.n * self._step

    def _elapse(self, stamp: Any) -> Any:
        return stamp + Timedelta._from_nanos(self.nanos, self._unit)

    def _on(self, stamp: Any) -> bool:
        return True

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Tick | datetime.timedelta):
            return self._step != 0 and self.nanos == _nanos_of(other)
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self._key())

    def _compared(self, other: Any, test: Callable[[int, int], bool]) -> Any:
        if isinstance(other, Tick | datetime.timedelta):
            return test(self.nanos, _nanos_of(other))
        return NotImplemented

    def __lt__(self, other: Any) -> Any:
        return self._compared(other, lambda a, b: a < b)

    def __le__(self, other: Any) -> Any:
        return self._compared(other, lambda a, b: a <= b)

    def __gt__(self, other: Any) -> Any:
        return self._compared(other, lambda a, b: a > b)

    def __ge__(self, other: Any) -> Any:
        return self._compared(other, lambda a, b: a >= b)

    def __add__(self, other: Any) -> Any:
        if isinstance(other, Tick):
            return _tick_of(self.nanos + other.nanos)
        if isinstance(other, Day) or (
            isinstance(other, datetime.timedelta) and not isinstance(other, datetime.datetime)
        ):
            return Timedelta._from_nanos(self.nanos, self._unit) + _elapsed(other)
        return super().__add__(other)

    __radd__ = __add__

    def __sub__(self, other: Any) -> Any:
        if isinstance(other, Tick):
            return _tick_of(self.nanos - other.nanos)
        if isinstance(other, datetime.timedelta) and not isinstance(other, datetime.datetime):
            return Timedelta._from_nanos(self.nanos, self._unit) - _elapsed(other)
        return super().__sub__(other)

    def __rsub__(self, other: Any) -> Any:
        if isinstance(other, datetime.timedelta) and not isinstance(other, datetime.datetime):
            return _elapsed(other) - Timedelta._from_nanos(self.nanos, self._unit)
        return super().__rsub__(other)

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, float) and self._step:
            return _tick_of(self.nanos * other)
        return super().__mul__(other)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Any:
        if isinstance(other, Tick | datetime.timedelta):
            return self.nanos / _nanos_of(other)
        if isinstance(other, int | float) and not isinstance(other, bool):
            return _tick_of(self.nanos / other)
        return NotImplemented


def _nanos_of(value: Any) -> int:
    if isinstance(value, BaseOffset):
        return value.nanos
    return (value if isinstance(value, Timedelta) else Timedelta(value))._nanos


def _elapsed(value: Any) -> Timedelta:
    if isinstance(value, BaseOffset):
        return Timedelta._from_nanos(value.nanos, getattr(value, "_unit", "us"))
    return value if isinstance(value, Timedelta) else Timedelta(value)


class Hour(Tick):
    """An hour, which is `pandas.offsets.Hour`."""

    _prefix = "h"
    _step = 3_600_000_000_000


class Minute(Tick):
    """A minute, which is `pandas.offsets.Minute`."""

    _prefix = "min"
    _step = 60_000_000_000


class Second(Tick):
    """A second, which is `pandas.offsets.Second`."""

    _prefix = "s"
    _step = 1_000_000_000


class Milli(Tick):
    """A millisecond, which is `pandas.offsets.Milli`."""

    _prefix = "ms"
    _step = 1_000_000


class Micro(Tick):
    """A microsecond, which is `pandas.offsets.Micro`."""

    _prefix = "us"
    _step = 1_000


class Nano(Tick):
    """A nanosecond, which is `pandas.offsets.Nano`."""

    _prefix = "ns"
    _step = 1
    _unit = "ns"


_TICKS: tuple[type[Tick], ...] = (Hour, Minute, Second, Milli, Micro, Nano)


def _tick_of(nanos: float) -> Tick:
    """The coarsest fixed step that holds a length exactly, as pandas answers a sum of two."""
    for kind in _TICKS:
        count = nanos / kind._step
        if count == int(count):
            return kind(int(count))
    raise ValueError(f"Could not convert to integer offset at any resolution: {nanos}")


# DateOffset.

_RELATIVE = (
    "years",
    "months",
    "weeks",
    "days",
    "hours",
    "minutes",
    "seconds",
    "milliseconds",
    "microseconds",
    "nanoseconds",
)
_ABSOLUTE = (
    "year",
    "month",
    "day",
    "weekday",
    "hour",
    "minute",
    "second",
    "microsecond",
    "nanosecond",
)
_ELAPSED = frozenset(_RELATIVE) - {"years", "months"}


class _AnyOffset(type):
    """Makes every offset an instance of `DateOffset`, as pandas does."""

    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, BaseOffset)

    def __subclasscheck__(cls, subclass: Any) -> bool:
        return isinstance(subclass, type) and issubclass(subclass, BaseOffset)


class DateOffset(BaseOffset, metaclass=_AnyOffset):
    """A step made of calendar parts, which is `pandas.DateOffset`.

    The plural parts (`months=1`) move by that much, and the singular ones
    (`day=31`) set that part, the way `dateutil.relativedelta` reads them. With no
    parts at all it is one calendar day.
    """

    def __init__(self, n: Any = 1, normalize: bool = False, **kwds: Any) -> None:
        super().__init__(n, normalize)
        unknown = [name for name in kwds if name not in _RELATIVE and name not in _ABSOLUTE]
        if unknown:
            raise ValueError(f"Invalid argument/s or bad combination of arguments: {unknown}")
        for name in ("years", "months"):
            value = kwds.get(name, 0)
            if isinstance(value, float) and not value.is_integer():
                raise ValueError(
                    "Non-integer years and months are ambiguous and not currently supported."
                )
        self._parts = dict(kwds)
        for name, value in kwds.items():
            setattr(self, name, value)
        # Without years, months or a value to set, days and weeks move the wall clock and
        # the finer parts are elapsed time added after.
        self._plain = bool(kwds) and set(kwds) <= _ELAPSED

    def _arguments(self) -> dict[str, Any]:
        return dict(self._parts)

    @property
    def _shown_params(self) -> tuple[str, ...]:  # type: ignore[override]
        return tuple(self._parts)

    def _span(self, count: int, nanoseconds: bool = True) -> Timedelta:
        parts = self._parts
        micros = (
            parts.get("weeks", 0) * 604_800_000_000
            + parts.get("days", 0) * 86_400_000_000
            + parts.get("hours", 0) * 3_600_000_000
            + parts.get("minutes", 0) * 60_000_000
            + parts.get("seconds", 0) * 1_000_000
            + parts.get("milliseconds", 0) * 1_000
            + parts.get("microseconds", 0)
        )
        nanos = round(micros * 1_000) + (parts.get("nanoseconds", 0) if nanoseconds else 0)
        return Timedelta._from_nanos(nanos * count, "ns" if nanos % 1_000 else "us")

    def _move(self, wall: datetime.datetime) -> datetime.datetime:
        parts = self._parts
        if not parts:
            return wall + datetime.timedelta(days=self.n)
        n = self.n
        if self._plain:
            days = parts.get("weeks", 0) * 7 + parts.get("days", 0)
            return wall + datetime.timedelta(days=days * n)
        year = parts.get("year", wall.year) + parts.get("years", 0) * n
        month = parts.get("month", wall.month)
        months = parts.get("months", 0) * n
        if months:
            year, month = year + (month - 1 + months) // 12, (month - 1 + months) % 12 + 1
        day = min(_last_day(year, month), parts.get("day", wall.day))
        moved = wall.replace(
            year=year,
            month=month,
            day=day,
            hour=parts.get("hour", wall.hour),
            minute=parts.get("minute", wall.minute),
            second=parts.get("second", wall.second),
            microsecond=parts.get("microsecond", wall.microsecond),
        )
        moved += datetime.timedelta(microseconds=_nanos_of(self._span(n, False)) // 1_000)
        weekday = parts.get("weekday")
        if weekday is not None:
            moved += datetime.timedelta(days=(int(weekday) - moved.weekday()) % 7)
        return moved

    def _apply(self, stamp: Any) -> Any:
        moved = super()._apply(stamp)
        if moved is NaT:
            return moved
        if self._plain:
            finer = {k: v for k, v in self._parts.items() if k not in ("weeks", "days")}
            return moved + DateOffset(**finer)._span(self.n) if finer else moved
        extra = self._parts.get("nanoseconds", 0) * self.n
        if "nanosecond" in self._parts:
            extra += self._parts["nanosecond"] - moved.nanosecond
        return moved + Timedelta._from_nanos(extra, "ns") if extra else moved


# The frequency text an offset is written as, read back for comparing.


def _parsed(text: str) -> BaseOffset:
    """The offset a frequency text names, for comparing an offset with text."""
    found = re.fullmatch(r"(-?\d+)?([A-Za-z]+)(?:-([\w-]+))?(?:\+(\w+))?", text.strip())
    if found is None:
        raise ValueError(f"Invalid frequency: {text}")
    count = int(found.group(1)) if found.group(1) else 1
    prefix, suffix, tail = found.group(2), found.group(3), found.group(4)
    if tail is not None:
        # pandas reads a business day with an offset as a plain business day, even from `C`.
        if prefix not in ("B", "C") or suffix is not None:
            raise ValueError(f"Invalid frequency: {text}")
        # A leading minus sign turns the offset around as well as the count.
        span = Timedelta(tail.replace("Min", "min")).to_pytimedelta()
        return BusinessDay(count, offset=-span if count < 0 else span)
    for kind in _BY_PREFIX.get(prefix, ()):
        try:
            return _built(kind, count, suffix)
        except (ValueError, TypeError):
            continue
    raise ValueError(f"Invalid frequency: {text}")


def _built(kind: type[BaseOffset], count: int, suffix: str | None) -> BaseOffset:
    if issubclass(kind, _Monthly) and kind._anchor is not None:
        if suffix is None:
            month = 1 if kind._first else 12
        elif suffix.upper() in _MONTHS:
            month = _MONTHS.index(suffix.upper()) + 1
        else:
            raise ValueError(suffix)
        return kind(count, **{kind._anchor: month})
    if kind is Week:
        return Week(count, weekday=6 if suffix is None else _WEEKDAYS.index(suffix.upper()))
    if kind is SemiMonthEnd or kind is SemiMonthBegin:
        return kind(count, day_of_month=None if suffix is None else int(suffix))
    if kind is LastWeekOfMonth and suffix is not None:
        return LastWeekOfMonth(count, weekday=_WEEKDAYS.index(suffix.upper()))
    if kind is WeekOfMonth and suffix is not None:
        return WeekOfMonth(count, week=int(suffix[0]) - 1, weekday=_WEEKDAYS.index(suffix[1:]))
    if kind is FY5253 or kind is FY5253Quarter:
        parts = (suffix or "").upper().split("-")
        if len(parts) != (3 if kind is FY5253 else 4) or parts[0] not in ("N", "L"):
            raise ValueError(suffix)
        shape = {
            "variation": "nearest" if parts[0] == "N" else "last",
            "startingMonth": _MONTHS.index(parts[1]) + 1,
            "weekday": _WEEKDAYS.index(parts[2]),
        }
        if kind is FY5253Quarter:
            shape["qtr_with_extra_week"] = int(parts[3])
        return kind(count, **shape)
    if suffix is not None:
        raise ValueError(suffix)
    return kind(count)


_BY_PREFIX: dict[str, tuple[type[BaseOffset], ...]] = {}
for _kind in (
    MonthEnd,
    MonthBegin,
    BusinessMonthEnd,
    BusinessMonthBegin,
    QuarterEnd,
    QuarterBegin,
    BQuarterEnd,
    BQuarterBegin,
    HalfYearEnd,
    HalfYearBegin,
    BHalfYearEnd,
    BHalfYearBegin,
    YearEnd,
    YearBegin,
    BYearEnd,
    BYearBegin,
    SemiMonthEnd,
    SemiMonthBegin,
    Week,
    WeekOfMonth,
    LastWeekOfMonth,
    BusinessDay,
    CustomBusinessDay,
    CustomBusinessMonthEnd,
    CustomBusinessMonthBegin,
    BusinessHour,
    CustomBusinessHour,
    FY5253,
    FY5253Quarter,
    Day,
    *_TICKS,
):
    _BY_PREFIX[_kind._prefix or ""] = (*_BY_PREFIX.get(_kind._prefix or "", ()), _kind)
del _kind

# The short names pandas gives the same classes.
BDay = BusinessDay
BMonthEnd = BusinessMonthEnd
BMonthBegin = BusinessMonthBegin
CDay = CustomBusinessDay
CBMonthEnd = CustomBusinessMonthEnd
CBMonthBegin = CustomBusinessMonthBegin


def _range_points(offset: BaseOffset, start: Any, end: Any, periods: int | None) -> list[Any]:
    """The moments of `date_range` with a calendar offset as its step, as pandas makes them.

    The start rolls forward onto the offset and the end rolls back, then the
    points step from the start, or back from the end when there is no start.
    """
    if offset.n == 0:
        raise ValueError("Offset <0 * ...> did not increment date")
    first = None if start is None else offset.rollforward(start)
    last = None if end is None else offset.rollback(end)
    points: list[Any] = []
    if first is not None:
        point = first
        forward = offset.n > 0
        while (periods is None or len(points) < periods) and (
            last is None or (point <= last if forward else point >= last)
        ):
            points.append(point)
            moved = offset._apply(point)
            if moved == point:
                raise ValueError(f"Offset {offset!r} did not increment date")
            point = moved
        return points
    point = last
    while len(points) < (periods or 0):
        points.append(point)
        point = (-offset)._apply(point)
    return points[::-1]
