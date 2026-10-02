"""The frequency a run of labels keeps, which is `pandas.infer_freq`.

A port of pandas' `_FrequencyInferer`, measured against pandas 3.0. The labels are
read as the whole numbers they are stored as, both as they are, which is UTC for
labels on a clock, and as the wall clock shows them. The steps between wall clock
readings decide whether the answer is a calendar rule, a year, a quarter, a month,
a week, a business day or a week of the month, and the steps between the stored
numbers decide the rules shorter than a day. The answer is pandas' alias for the
rule, with a count in front of it when the step is more than one.
"""

from __future__ import annotations

import calendar
import datetime
import itertools
from typing import Any

from .errors import DTypeError

__all__ = ["infer_freq"]

_PER_DAY = {
    "s": 86_400,
    "ms": 86_400_000,
    "us": 86_400_000_000,
    "ns": 86_400_000_000_000,
}
_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
_DAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
_NUMERIC = ("int", "uint", "float", "bool")


def _steps(values: list[int]) -> list[int]:
    """The distinct steps between neighbours, smallest first."""
    return sorted({after - before for before, after in itertools.pairwise(values)})


def _counted(base: str, count: float) -> str:
    """The alias with its count in front, which is left off for a count of one."""
    if count != 1:
        return f"{int(count)}{base}"
    return base


def _month_position(
    years: list[int], months: list[int], days: list[int], weekdays: list[int]
) -> str | None:
    """Whether every label sits at the start or the end of its month.

    `ce` and `cs` are the calendar end and start, and `be` and `bs` the business
    end and start, where a label may sit on the Friday before a month ends on a
    weekend, or the Monday after one starts on a weekend.
    """
    calendar_end = business_end = calendar_start = business_start = True
    for year, month, day, weekday in zip(years, months, days, weekdays, strict=True):
        if calendar_start:
            calendar_start = day == 1
        if business_start:
            business_start = day == 1 or (day <= 3 and weekday == 0)
        if calendar_end or business_end:
            last = calendar.monthrange(year, month)[1]
            if calendar_end:
                calendar_end = day == last
            if business_end:
                business_end = day == last or (last - day < 3 and weekday == 4)
        elif not calendar_start and not business_start:
            break
    if calendar_end:
        return "ce"
    if business_end:
        return "be"
    if calendar_start:
        return "cs"
    if business_start:
        return "bs"
    return None


class _Inferer:
    """The rule a run of labels keeps, or None when it keeps none."""

    def __init__(
        self,
        stored: list[int],
        wall: list[int],
        unit: str,
        fields: tuple[list[int], list[int], list[int], list[int]] | None,
        first: datetime.datetime,
    ) -> None:
        self.stored = stored
        self.wall = wall
        self.per_day = _PER_DAY[unit]
        self.fields = fields
        self.first = first
        self.steps = _steps(wall)

    def rule(self) -> str | None:
        stored = self.stored
        rising = all(a <= b for a, b in itertools.pairwise(stored))
        falling = all(a >= b for a, b in itertools.pairwise(stored))
        if not (rising or falling) or len(set(stored)) != len(stored):
            return None
        step = self.steps[0]
        if step and step % self.per_day == 0:
            return self._daily()
        per_hour = self.per_day // 24
        if [one / per_hour for one in self.steps] in ([1, 17], [1, 65], [1, 17, 65]):
            return "bh"
        stored_steps = _steps(stored)
        if len(stored_steps) != 1:
            return None
        step = stored_steps[0]
        per_minute = per_hour // 60
        per_second = per_minute // 60
        for base, size in (
            ("h", per_hour),
            ("min", per_minute),
            ("s", per_second),
            ("ms", per_second // 1000),
            ("us", per_second // 1_000_000),
        ):
            if size and step % size == 0:
                return _counted(base, step / size)
        return _counted("ns", step)

    def _daily(self) -> str | None:
        if self.fields is None:
            return self._by_days() if len(self.steps) == 1 else None
        years, months, days, weekdays = self.fields
        year_steps = _steps(years)
        month_steps = _steps([year * 12 + month for year, month in zip(years, months, strict=True)])
        position = _month_position(years, months, days, weekdays)
        if len(year_steps) <= 1 and len(set(months)) <= 1 and position is not None:
            rule = {"cs": "YS", "bs": "BYS", "ce": "YE", "be": "BYE"}[position]
            return _counted(f"{rule}-{_MONTHS[self.first.month - 1]}", year_steps[0])
        if len(month_steps) <= 1 and month_steps[0] % 3 == 0 and position is not None:
            rule = {"cs": "QS", "bs": "BQS", "ce": "QE", "be": "BQE"}[position]
            month = {0: 12, 2: 11, 1: 10}[self.first.month % 3]
            return _counted(f"{rule}-{_MONTHS[month - 1]}", month_steps[0] / 3)
        if len(month_steps) <= 1 and position is not None:
            rule = {"cs": "MS", "bs": "BMS", "ce": "ME", "be": "BME"}[position]
            return _counted(rule, month_steps[0])
        if len(self.steps) == 1:
            return self._by_days()
        if self._business_daily(weekdays):
            return "B"
        return self._week_of_month(days, weekdays)

    def _by_days(self) -> str:
        days = self.steps[0] / self.per_day
        if days % 7 == 0:
            return _counted(f"W-{_DAYS[self.first.weekday()]}", days / 7)
        return _counted("D", days)

    def _business_daily(self, weekdays: list[int]) -> bool:
        if [one / self.per_day for one in self.steps] != [1, 3]:
            return False
        weekday = weekdays[0]
        for before, after in itertools.pairwise(self.wall):
            shift = (after - before) // self.per_day
            weekday = (weekday + shift) % 7
            if not ((weekday == 0 and shift == 3) or (0 < weekday <= 4 and shift == 1)):
                return False
        return True

    def _week_of_month(self, days: list[int], weekdays: list[int]) -> str | None:
        if len(set(weekdays)) > 1:
            return None
        weeks = {(day - 1) // 7 for day in days}
        weeks = {week for week in weeks if week < 4}
        if len(weeks) != 1:
            return None
        return f"WOM-{weeks.pop() + 1}{_DAYS[weekdays[0]]}"


def _inferred(index: Any) -> str | None:
    """The rule of a DatetimeIndex or a TimedeltaIndex."""
    from ._timedelta import TimedeltaIndex

    if len(index) < 3:
        raise ValueError("Need at least 3 dates to infer frequency")
    stored = index._stamps
    if any(one is None for one in stored):
        return None
    unit = index.unit
    if isinstance(index, TimedeltaIndex):
        epoch = datetime.datetime(1970, 1, 1)
        first = epoch + datetime.timedelta(seconds=stored[0] / (_PER_DAY[unit] // 86_400))
        return _Inferer(stored, stored, unit, None, first).rule()
    local = index.tz_localize(None) if index.tz is not None else index
    wall = local._stamps
    fields = (
        local.year.tolist(),
        local.month.tolist(),
        local.day.tolist(),
        local.dayofweek.tolist(),
    )
    first = datetime.datetime(fields[0][0], fields[1][0], fields[2][0])
    return _Inferer(stored, wall, unit, fields, first).rule()


def infer_freq(index: Any) -> str | None:
    """The frequency the labels keep, as pandas' alias for it, or None.

    Takes a DatetimeIndex, a TimedeltaIndex, a column of either, or anything
    `DatetimeIndex` reads as instants. At least three labels are needed.

    Raises:
        TypeError: For a column or index of numbers, or a column of text.
        ValueError: For fewer than three labels.
    """
    from ._datetime import DatetimeIndex
    from ._pandas import SeriesMixin
    from ._timedelta import TimedeltaIndex

    if isinstance(index, SeriesMixin):
        kind = str(index.dtype)
        if kind.startswith("timedelta64"):
            return _inferred(TimedeltaIndex(index))
        if not kind.startswith(("datetime64", "object")):
            shown = "str" if kind == "string" else kind
            raise DTypeError(
                f"cannot infer freq from a non-convertible dtype on a Series of {shown}"
            )
        return _inferred(DatetimeIndex(index))
    if isinstance(index, TimedeltaIndex):
        return _inferred(index)
    kind = str(getattr(index, "dtype", ""))
    if kind.startswith("timedelta64"):
        return _inferred(TimedeltaIndex(index))
    if kind.startswith(_NUMERIC):
        raise DTypeError(f"cannot infer freq from a non-convertible index of dtype {kind}")
    if not isinstance(index, DatetimeIndex):
        index = DatetimeIndex(index)
    return _inferred(index)


def _offset_of(freq: Any) -> Any:
    """The offset a frequency is written as: text, an offset, a span or None."""
    from . import offsets
    from ._scalars import Timedelta

    if freq is None or isinstance(freq, offsets.BaseOffset):
        return freq
    if isinstance(freq, str):
        try:
            return offsets._parsed(freq)
        except ValueError as error:
            # Text like `1.5h` or `2D3h` is a fixed length, which pandas holds as one tick.
            try:
                nanos = Timedelta(freq).value
            except ValueError:
                raise error from None
            return offsets._tick_of(nanos)
    # A span is a fixed step even when it is whole days, so two days is 48 hours.
    return offsets._tick_of(Timedelta(freq).value)


def _conforming(index: Any, offset: Any) -> None:
    """Refuses a frequency the labels do not keep, in pandas' words.

    The labels keep it when they are the range that starts at the first of them
    and steps by it, which is how pandas checks, and an empty index keeps any.
    """
    from ._date_range import date_range
    from ._timedelta import TimedeltaIndex, timedelta_range

    if not len(index):
        return
    inferred = index.inferred_freq
    if inferred == offset.freqstr:
        return
    spans = isinstance(index, TimedeltaIndex)
    # timedelta_range reads a step from text, and every offset it can take has one.
    step = offset.freqstr if spans else offset
    ranged = timedelta_range if spans else date_range
    try:
        made = ranged(start=index[0], periods=len(index), freq=step, unit=index.unit)
        kept = made._stamps == index._stamps
    except (ValueError, TypeError, NotImplementedError):
        kept = False
    if not kept:
        raise ValueError(
            f"Inferred frequency {inferred} from passed values does not conform to passed"
            f" frequency {offset.freqstr}"
        )


def _hold(index: Any, freq: Any, data: Any) -> None:
    """Gives a newly built index the frequency pandas' constructor gives it.

    Left out, the frequency comes along from an index of the same kind. `infer`
    takes the one the labels keep, and anything else has to be one they keep.
    """
    from ._pandas import NO_DEFAULT

    if freq is NO_DEFAULT:
        held = getattr(data, "_freq", None) if type(data) is type(index) else None
    elif isinstance(freq, str) and freq == "infer":
        held = _offset_of(index.inferred_freq)
    else:
        held = _offset_of(freq)
        if held is not None:
            _conforming(index, held)
    index._freq = held


def _held(index: Any, freq: Any) -> Any:
    """The index, holding the frequency it was made at."""
    index._freq = _offset_of(freq)
    return index
