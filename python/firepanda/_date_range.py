"""`date_range`, which is a `DatetimeIndex` of evenly stepped instants.

Three of start, end, periods and freq say where the range is, and the work is
to turn the two ends into whole numbers in one unit, count along them in that
unit, and hand the numbers to `to_datetime` to become instants. The rules below
were measured against pandas 3.0.

- The unit is the finest one the ends carry. Text is `us`, or `ns` when it
  writes nanoseconds, a `Timestamp` keeps its own, a `datetime` is `us` and a
  `date` is `s`. A step finer than that unit raises the unit to the step's.
- With no zone a day is 24 hours. With a zone a step of days is a step on the
  wall clock, so midnight stays midnight across a transition, and every other
  step is a step of absolute time, so the wall clock readings move.
- With start, end and periods and no freq the points are spaced evenly between
  the two ends, as numpy's `linspace` spaces them.
- `inclusive` drops the first point only when it is the start and the last
  point only when it is the end.

Steps of weeks, months, quarters, years and business days are calendar offsets
whose steps are not all one length. They land on dates, counted on the wall
clock, and `_calendar_steps` finds them.
"""

from __future__ import annotations

import datetime
import re
from typing import Any

from ._calendar_steps import CalendarStep, calendar_step
from ._datetime import DatetimeIndex
from ._frame import Series
from ._pandas import to_datetime
from ._resample import _CALENDAR, _NANOS
from ._scalars import Timestamp
from .errors import InvalidArgumentError

__all__ = ["bdate_range", "date_range"]

_UNITS = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}
"""The units a range can be counted in, finest last, in nanoseconds."""

_OFFSETS = {
    "D": ("Day", "Days"),
    "h": ("Hour", "Hours"),
    "min": ("Minute", "Minutes"),
    "s": ("Second", "Seconds"),
    "ms": ("Milli", "Millis"),
    "us": ("Micro", "Micros"),
    "ns": ("Nano", "Nanos"),
}
"""How pandas writes each step as an offset, once and more than once."""

_EPOCH = datetime.datetime(1970, 1, 1)
_DAY = _NANOS["D"]
_INCLUSIVE = ("both", "neither", "left", "right")


class _End:
    """One end of a range, as a wall clock reading in nanoseconds.

    `utc` is the same instant counted from the epoch in UTC when the end came
    with a zone, and None when it came without one.
    """

    __slots__ = ("unit", "utc", "wall", "zone")

    def __init__(self, wall: int, unit: str, zone: str | None, utc: int | None) -> None:
        self.wall = wall
        self.unit = unit
        self.zone = zone
        self.utc = utc


def _zone_name(zone: Any) -> str | None:
    """A zone as the name firepanda reads, or None for no zone."""
    if zone is None:
        return None
    if isinstance(zone, str):
        return zone
    key = getattr(zone, "key", None)
    if isinstance(key, str):
        return key
    if isinstance(zone, datetime.timezone):
        return "UTC" if zone == datetime.UTC else str(zone)
    return str(zone)


def _nanos_since_epoch(moment: datetime.datetime, nanosecond: int) -> int:
    """A wall clock reading as nanoseconds since 1970, with the zone left off."""
    since = moment.replace(tzinfo=None) - _EPOCH
    return (since.days * 86_400 + since.seconds) * 10**9 + since.microseconds * 1000 + nanosecond


def _end(value: Any) -> _End | None:
    """Reads one end of a range, the way pandas' `Timestamp` would.

    Raises:
        InvalidArgumentError: For text that is not an instant.
    """
    if value is None:
        return None
    if type(value).__name__ == "datetime64":
        import numpy as np

        unit = np.datetime_data(value.dtype)[0]
        value = Timestamp(str(value))
        return _End(
            _nanos_since_epoch(value.to_pydatetime(), value.nanosecond),
            unit if unit in _UNITS else "s",
            None,
            None,
        )
    if isinstance(value, str):
        value = Timestamp(value)
    if isinstance(value, datetime.datetime):
        unit = getattr(value, "unit", "us")
        nanosecond = getattr(value, "nanosecond", 0)
        wall = _nanos_since_epoch(
            datetime.datetime(
                value.year,
                value.month,
                value.day,
                value.hour,
                value.minute,
                value.second,
                value.microsecond,
            ),
            nanosecond,
        )
        offset = value.utcoffset()
        if offset is None:
            return _End(wall, unit, None, None)
        shift = (offset.days * 86_400 + offset.seconds) * 10**9 + offset.microseconds * 1000
        return _End(wall, unit, _zone_name(value.tzinfo), wall - shift)
    if isinstance(value, datetime.date):
        return _End(
            _nanos_since_epoch(datetime.datetime(value.year, value.month, value.day), 0),
            "s",
            None,
            None,
        )
    return _end(Timestamp(value))


def _offset(count: int, unit: str) -> str:
    """A step written the way pandas writes its offset, like `<Milli>` or `<2 * Days>`."""
    one, many = _OFFSETS[unit]
    return f"<{one}>" if count == 1 else f"<{count} * {many}>"


def _frequency(freq: Any) -> tuple[int, bool, str]:
    """The length of a step in nanoseconds, whether it is days, and its offset.

    Raises:
        InvalidArgumentError: For a frequency pandas cannot read, with its message.
        NotImplementedError: For a calendar offset.
    """
    if isinstance(freq, datetime.timedelta):
        nanos = _nanos_since_epoch(_EPOCH + freq, getattr(freq, "nanoseconds", 0))
        return nanos, False, _offset(nanos, "ns")
    if not isinstance(freq, str):
        raise NotImplementedError(
            "date_range: freq is read from text or a timedelta for now, because an offset"
            " object is pandas' offsets namespace, which firepanda does not have"
        )
    found = re.fullmatch(r"\s*(-?\d+(?:\.\d*)?)?\s*([A-Za-z]+)(-[A-Za-z]+)?\s*", freq)
    unit = found.group(2) if found else ""
    if unit in _CALENDAR:
        raise NotImplementedError(
            f"date_range: freq={freq!r} is a calendar offset, whose steps are not all one"
            " length, and firepanda steps by days and less for now"
        )
    if found is None or unit not in _NANOS or found.group(3) is not None:
        raise InvalidArgumentError(
            f"Invalid frequency: {freq}. Failed to parse with error message:"
            f" ValueError('Invalid frequency: {freq}.')"
        )
    count = found.group(1)
    length = float(count) * _NANOS[unit] if count else _NANOS[unit]
    if length != int(length):
        raise InvalidArgumentError(
            f"Invalid frequency: {freq}. Failed to parse with error message:"
            f" ValueError('Invalid frequency: {freq}.')"
        )
    if length == 0:
        raise InvalidArgumentError("range() arg 3 must not be zero")
    nanos = int(length)
    if unit == "D" and nanos % _DAY == 0:
        return nanos, True, _offset(nanos // _DAY, "D")
    names = list(_OFFSETS)
    finest = next(name for name in names[names.index(unit) :] if nanos % _NANOS[name] == 0)
    return nanos, False, _offset(nanos // _NANOS[finest], finest)


def _points(
    first: int | None, last: int | None, periods: int | None, step: int | None
) -> list[int]:
    """The counts along a range, from whichever three of the four were given."""
    if step is None:
        assert first is not None and last is not None and periods is not None
        if periods == 1:
            return [first]
        across = (last - first) / (periods - 1)
        inner = [first + int(position * across) for position in range(periods - 1)]
        return [*inner, last] if periods else []
    if first is not None and last is not None:
        if step > 0:
            return list(range(first, last + 1, step))
        return list(range(first, last - 1, step))
    if first is not None:
        assert periods is not None
        return [first + position * step for position in range(periods)]
    assert last is not None and periods is not None
    return [last - (periods - 1 - position) * step for position in range(periods)]


def date_range(
    start: Any = None,
    end: Any = None,
    periods: Any = None,
    freq: Any = None,
    tz: Any = None,
    normalize: bool = False,
    name: Any = None,
    inclusive: Any = "both",
    *,
    unit: Any = None,
    **kwargs: Any,
) -> DatetimeIndex:
    """A fixed frequency `DatetimeIndex`, which is `pandas.date_range`.

    Args:
        start: The first instant, as text, a `Timestamp`, a `datetime` or a `date`.
        end: The last instant, in the same forms.
        periods: How many instants.
        freq: The step, as text like `"6h"`, `"2D"`, `"B"`, `"W-WED"` or `"QE"`, or
            a timedelta. Left out it is a day, unless start, end and periods are
            all given, when the points are spaced evenly between the two ends.
        tz: The zone the instants are read against, by name.
        normalize: Moves both ends to midnight before counting.
        name: The level name.
        inclusive: Which ends may be in the answer.
        unit: The resolution of the answer.
        **kwargs: Refused with pandas' TypeError.

    Raises:
        TypeError: When periods is not a whole number, or for an unknown keyword.
        ValueError: When not exactly three of start, end, periods and freq are
            given, or for an unknown inclusive or unit.
        NotImplementedError: For a semi month, business hour or week of month step.
    """
    return _ranged(start, end, periods, freq, tz, normalize, name, inclusive, unit, kwargs, None)


def _ranged(
    start: Any,
    end: Any,
    periods: Any,
    freq: Any,
    tz: Any,
    normalize: bool,
    name: Any,
    inclusive: Any,
    unit: Any,
    kwargs: dict[str, Any],
    calendar: CalendarStep | None,
) -> DatetimeIndex:
    """`date_range`, with the business day step `bdate_range` built standing in for freq."""
    if kwargs:
        raise TypeError(
            "DatetimeArray._generate_range() got an unexpected keyword argument"
            f" '{next(iter(kwargs))}'"
        )
    if freq is None and None in (start, end, periods):
        freq = "D"
    if periods is not None:
        if isinstance(periods, bool) or not hasattr(periods, "__index__"):
            raise TypeError(f"periods must be an integer, got {periods}")
        periods = int(periods)
        if periods < 0:
            raise OverflowError(f"Python integer {periods} out of bounds for uint64")
    if sum(given is not None for given in (start, end, periods, freq)) != 3:
        raise InvalidArgumentError(
            "Of the four parameters: start, end, periods, and freq, exactly three must be specified"
        )
    if inclusive not in _INCLUSIVE:
        raise InvalidArgumentError(
            "Inclusive has to be either 'both', 'neither', 'left' or 'right'"
        )
    if unit is not None and unit not in _UNITS:
        raise InvalidArgumentError("'unit' must be one of 's', 'ms', 'us', 'ns'")
    steps = calendar_step(freq) if calendar is None else calendar
    step, in_days, offset = (
        _frequency(freq) if freq is not None and steps is None else (None, steps is not None, "")
    )
    ends = [_end(start), _end(end)]
    zone = _zone_name(tz)
    for found in ends:
        if found is not None and found.zone is not None:
            if zone is not None and zone != found.zone:
                raise AssertionError("Inferred time zone not equal to passed time zone")
            zone = found.zone
    if unit is None:
        units = [found.unit for found in ends if found is not None]
        unit = min(units, key=_UNITS.__getitem__)
        while step is not None and step % _UNITS[unit]:
            unit = {"s": "ms", "ms": "us", "us": "ns"}[unit]
    elif step is not None and step % _UNITS[unit]:
        raise InvalidArgumentError(
            f"freq={offset} is incompatible with unit={unit}. Use a lower freq or a higher"
            " unit instead."
        )
    scale = _UNITS[unit]
    if normalize:
        for found in ends:
            if found is not None:
                found.wall = found.wall // _DAY * _DAY
                found.utc = None
    on_the_wall = zone is None or in_days
    if not on_the_wall:
        _place(ends, zone)
    first, last = (
        None if found is None else (found.wall if on_the_wall else found.utc) // scale
        for found in ends
    )
    if steps is not None:
        walls = [None if found is None else found.wall for found in ends]
        counts = [point // scale for point in steps.points(walls[0], walls[1], periods)]
    else:
        counts = _points(first, last, periods, None if step is None else step // scale)
    if counts and inclusive in ("neither", "right") and counts[0] == first:
        counts = counts[1:]
    if counts and inclusive in ("neither", "left") and counts[-1] == last:
        counts = counts[:-1]
    stamps = to_datetime(Series(counts, dtype="int64"), unit=unit)
    if zone is not None:
        if on_the_wall:
            stamps = _localized(stamps, zone)
        else:
            stamps = stamps.dt.tz_localize("UTC").dt.tz_convert(zone)
    return DatetimeIndex(stamps, name=name)


def bdate_range(
    start: Any = None,
    end: Any = None,
    periods: Any = None,
    freq: Any = "B",
    tz: Any = None,
    normalize: bool = True,
    name: Any = None,
    weekmask: Any = None,
    holidays: Any = None,
    inclusive: Any = "both",
    **kwargs: Any,
) -> DatetimeIndex:
    """A `DatetimeIndex` of business days, which is `pandas.bdate_range`.

    It is `date_range` with a business day step and both ends moved to midnight.
    A week mask or holidays need the custom business day `C`. `unit` arrives
    among the keywords, as it does in pandas' signature.

    Raises:
        TypeError: When freq is None.
        ValueError: For a week mask or holidays without a `C` step, and for
            everything `date_range` refuses.
    """
    if freq is None:
        raise TypeError("freq must be specified for bdate_range; use date_range instead")
    custom = isinstance(freq, str) and freq.startswith("C")
    if not custom and (weekmask is not None or holidays is not None):
        raise InvalidArgumentError(
            "a custom frequency string is required when holidays or weekmask are passed,"
            f" got frequency {freq}"
        )
    step = calendar_step(freq) if custom else None
    if step is not None:
        step.with_calendar(weekmask, holidays)
    unit = kwargs.pop("unit", None)
    return _ranged(start, end, periods, freq, tz, normalize, name, inclusive, unit, kwargs, step)


def _place(ends: list[_End | None], zone: str) -> None:
    """Finds the instant in UTC of every end that has only a wall clock reading."""
    for found in ends:
        if found is None or found.utc is not None:
            continue
        wall = to_datetime(Series([found.wall], dtype="int64"), unit="ns")
        found.utc = _localized(wall, zone).astype("int64").tolist()[0]


def _localized(stamps: Series, zone: str) -> Series:
    """Wall clock readings put on a zone, refused with pandas' words in a gap or a fold."""
    try:
        return stamps.dt.tz_localize(zone)
    except InvalidArgumentError as error:
        raise InvalidArgumentError(str(error).removeprefix("temporal: ")) from None
