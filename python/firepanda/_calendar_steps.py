"""The calendar steps of `date_range` and `bdate_range`, whose lengths are not all one.

A calendar step lands on dates: business days, one weekday of every week, or
the first or last day, plain or business, of every month, quarter or year. The
time of day of the first point rides along, as it does in pandas. The rules
below were measured against pandas 3.0.

- A start that is not on a landing date moves forward to the next one, or back
  to the previous one when the step goes backwards. An end given with periods
  and no start moves the other way, and the points count back from it.
- With both ends the points run from the start until they pass the end.
- `C` is a business day with a week mask and holidays, which `bdate_range`
  passes in.

Semi month steps, business hours and the week of month steps are refused by name.
"""

from __future__ import annotations

import calendar
import datetime
import re
from typing import Any

from .errors import InvalidArgumentError

__all__ = ["CalendarStep", "calendar_step"]

_WEEKDAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")

_MONTHLY = {
    "ME": (1, 12, "last", False),
    "MS": (1, 1, "first", False),
    "BME": (1, 12, "last", True),
    "BMS": (1, 1, "first", True),
    "QE": (3, 12, "last", False),
    "QS": (3, 1, "first", False),
    "BQE": (3, 12, "last", True),
    "BQS": (3, 1, "first", True),
    "YE": (12, 12, "last", False),
    "YS": (12, 1, "first", False),
    "BYE": (12, 12, "last", True),
    "BYS": (12, 1, "first", True),
}
"""Every month-like step: its stride in months, anchor month, day and whether it is business."""

_ANCHORED = frozenset({"QE", "QS", "BQE", "BQS", "YE", "YS", "BYE", "BYS"})
"""The month-like steps that take a month after a dash, like `QE-JAN`."""

_RENAMED = {
    "M": "ME",
    "Q": "QE",
    "Y": "YE",
    "BY": "BYE",
    "BM": "BME",
    "BQ": "BQE",
    "SM": "SME",
    "CBM": "CBME",
}
"""The old names pandas 3 no longer reads, and the names it points to instead."""

_REFUSED = frozenset({"SME", "SMS", "CBME", "CBMS", "BH", "CBH", "WOM", "LWOM"})
"""Calendar steps firepanda does not count along yet."""

_DAY_NANOS = 86_400 * 10**9
_EPOCH = datetime.date(1970, 1, 1)


def _invalid(freq: str, why: str) -> InvalidArgumentError:
    """pandas' words for a frequency it cannot read, nesting the reason it gives."""
    return InvalidArgumentError(
        f"Invalid frequency: {freq}. Failed to parse with error message: {why}"
    )


class CalendarStep:
    """One calendar step, able to find its landing dates and move along them."""

    __slots__ = (
        "anchor",
        "business",
        "day",
        "holidays",
        "kind",
        "n",
        "stride",
        "weekday",
        "workdays",
    )

    def __init__(self, kind: str, n: int) -> None:
        self.kind = kind
        self.n = n
        self.weekday = 6
        self.stride = 1
        self.anchor = 12
        self.day = "last"
        self.business = False
        self.workdays = frozenset(range(5))
        self.holidays: frozenset[datetime.date] = frozenset()

    def with_calendar(self, weekmask: Any, holidays: Any) -> CalendarStep:
        """The same step working on the given week mask and holidays."""
        if weekmask is not None:
            self.workdays = _workdays(weekmask)
        if holidays is not None:
            self.holidays = frozenset(_day_of(value) for value in holidays)
        return self

    def _works(self, day: datetime.date) -> bool:
        """Whether a date is a business day."""
        return day.weekday() in self.workdays and day not in self.holidays

    def _pick(self, month: int) -> datetime.date:
        """The landing date of a month, counted from year zero."""
        year, month = divmod(month, 12)
        month += 1
        if self.day == "first":
            found = datetime.date(year, month, 1)
            while self.business and found.weekday() >= 5:
                found += datetime.timedelta(days=1)
        else:
            found = datetime.date(year, month, calendar.monthrange(year, month)[1])
            while self.business and found.weekday() >= 5:
                found -= datetime.timedelta(days=1)
        return found

    def _counts(self, month: int) -> bool:
        """Whether a month, counted from year zero, has a landing date."""
        return (month % 12 + 1 - self.anchor) % self.stride == 0

    def on(self, day: datetime.date) -> bool:
        """Whether a date is a landing date."""
        if self.kind == "B":
            return self._works(day)
        if self.kind == "W":
            return day.weekday() == self.weekday
        month = day.year * 12 + day.month - 1
        return self._counts(month) and self._pick(month) == day

    def roll(self, day: datetime.date, forward: bool) -> datetime.date:
        """The nearest landing date at or after a date, or at or before it."""
        if self.on(day):
            return day
        way = 1 if forward else -1
        if self.kind == "B":
            while not self._works(day):
                day += datetime.timedelta(days=way)
            return day
        if self.kind == "W":
            gap = (self.weekday - day.weekday()) % 7
            return day + datetime.timedelta(days=gap if forward else gap - 7)
        month = day.year * 12 + day.month - 1
        while not (
            self._counts(month)
            and (self._pick(month) >= day if forward else self._pick(month) <= day)
        ):
            month += way
        return self._pick(month)

    def move(self, day: datetime.date, count: int) -> datetime.date:
        """The landing date `count` landings on from a landing date, backwards when negative."""
        if self.kind == "B":
            way = 1 if count > 0 else -1
            for _ in range(abs(count)):
                day += datetime.timedelta(days=way)
                while not self._works(day):
                    day += datetime.timedelta(days=way)
            return day
        if self.kind == "W":
            return day + datetime.timedelta(days=7 * count)
        return self._pick(day.year * 12 + day.month - 1 + count * self.stride)

    def points(self, first: int | None, last: int | None, periods: int | None) -> list[int]:
        """The wall clock readings in nanoseconds along the step, from three of the four."""
        start = None if first is None else _split(first)
        if start is not None:
            begin = (self.roll(start[0], self.n > 0), start[1])
            if periods is not None:
                return [_join(self.move(begin[0], self.n * k), begin[1]) for k in range(periods)]
            assert last is not None
            found = []
            day = begin[0]
            while True:
                point = _join(day, begin[1])
                if (point > last) if self.n > 0 else (point < last):
                    return found
                found.append(point)
                day = self.move(day, self.n)
        assert last is not None and periods is not None
        end = _split(last)
        finish = self.roll(end[0], self.n < 0)
        return [_join(self.move(finish, -self.n * k), end[1]) for k in range(periods)][::-1]


def _split(nanos: int) -> tuple[datetime.date, int]:
    """A wall clock reading as its date and its nanoseconds into the day."""
    days, into = divmod(nanos, _DAY_NANOS)
    return _EPOCH + datetime.timedelta(days=days), into


def _join(day: datetime.date, into: int) -> int:
    """A date and nanoseconds into it as one wall clock reading."""
    return (day - _EPOCH).days * _DAY_NANOS + into


def _day_of(value: Any) -> datetime.date:
    """A holiday as a date, from text, a date or a timestamp."""
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    from ._scalars import Timestamp

    return Timestamp(value).to_pydatetime().date()


def _workdays(weekmask: Any) -> frozenset[int]:
    """The weekdays a week mask works, from `Mon Wed` or `1010100` or a list of flags."""
    if isinstance(weekmask, str):
        text = weekmask.strip()
        if re.fullmatch(r"[01]{7}", text):
            return frozenset(i for i, flag in enumerate(text) if flag == "1")
        names = [name[:3].upper() for name in text.split()]
        if not names or any(name not in _WEEKDAYS for name in names):
            raise InvalidArgumentError(f"Invalid business day weekmask {weekmask!r}")
        return frozenset(_WEEKDAYS.index(name) for name in names)
    flags = list(weekmask)
    if len(flags) != 7:
        raise InvalidArgumentError("A business day weekmask array must have length 7")
    return frozenset(i for i, flag in enumerate(flags) if flag)


def calendar_step(freq: Any) -> CalendarStep | None:
    """The calendar step a frequency names, or None when it is not a calendar step.

    Raises:
        InvalidArgumentError: For a calendar step pandas cannot read, in its words.
        NotImplementedError: For a calendar step firepanda does not count along yet.
    """
    if not isinstance(freq, str):
        return None
    found = re.fullmatch(r"(-?\d+(?:\.\d*)?)?([A-Za-z]+)(?:-(\w+))?", freq.strip())
    if found is None:
        return None
    count, name, suffix = found.groups()
    if name in _RENAMED:
        raise _invalid(
            freq,
            f"ValueError(\"'{name}' is no longer supported for offsets. Please use"
            f" '{_RENAMED[name]}' instead.\")",
        )
    if name not in _MONTHLY and name not in ("B", "C", "W") and name not in _REFUSED:
        return None
    if name in _REFUSED:
        raise NotImplementedError(
            f"date_range: freq={freq!r} is a calendar offset firepanda does not count along"
            " yet; business days, weeks and the first or last day of a month, quarter or"
            " year are supported"
        )
    if count is not None and not re.fullmatch(r"-?\d+", count):
        raise _invalid(freq, f"ValueError(\"invalid literal for int() with base 10: '{count}'\")")
    n = int(count) if count is not None else 1
    if n == 0:
        raise NotImplementedError("date_range: a calendar step of zero is not supported")
    step = CalendarStep("B" if name == "C" else name[0] if name in ("B", "W") else "M", n)
    if name == "W":
        if suffix is not None:
            if suffix not in _WEEKDAYS:
                raise _invalid(
                    freq,
                    f'ValueError("Invalid frequency: {freq}. Failed to parse with error'
                    f" message: KeyError('{suffix}').\")",
                )
            step.weekday = _WEEKDAYS.index(suffix)
        return step
    if name in _MONTHLY:
        step.stride, step.anchor, step.day, step.business = _MONTHLY[name]
        if suffix is not None and name in _ANCHORED:
            if suffix not in _MONTHS:
                raise _invalid(
                    freq,
                    f'ValueError("Invalid frequency: {freq}. Failed to parse with error'
                    f" message: KeyError('{suffix}').\")",
                )
            step.anchor = _MONTHS.index(suffix) + 1
            return step
    if suffix is not None:
        raise _invalid(
            freq,
            f'ValueError("Invalid frequency: {freq}. Failed to parse with error message:'
            f" ValueError('Bad freq suffix {suffix}').\")",
        )
    return step
