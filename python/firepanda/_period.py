"""`Period`, a span of time named by a frequency, which is `pandas.Period`.

A pure Python scalar: a frequency and an ordinal, the count of such spans
since the one holding the start of 1970. The rules below were measured
against pandas 3.0.

- A frequency is a year or a quarter ending in some month, a month, a week
  ending on some weekday, a business day, a day, or an hour, minute, second,
  milli, micro or nanosecond, each possibly several long. The offsets that
  name them are `YearEnd`, `QuarterEnd`, `MonthEnd`, `Week`, `BusinessDay`,
  `Day` and the ticks, and a period hands its frequency back as one of them.
- Years count from 1970, quarters and months from its first, weeks so that
  the one ending on 4 January 1970 is the first, business days so that 1
  January 1970 is the zeroth, and days and finer units from its midnight.
- The fields a period shows, its year, month, day and so on, are those of
  its last day for a week or longer, and of its first instant for finer ones.
- Adding a whole number moves a period by that many spans, and adding an
  offset of its own frequency moves it by the offset's count. A day or a finer
  period also moves by a span that is a whole number of its units.
"""

from __future__ import annotations

import calendar
import datetime
import numbers
import operator
import re
import warnings
from typing import Any

from ._scalars import NaT, Timedelta, Timestamp
from .errors import IncompatibleFrequency, InvalidArgumentError

__all__ = ["Period"]

_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
_WEEKDAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
_DAY = 86_400_000_000_000
_UNITS = {
    "D": _DAY,
    "h": 3_600_000_000_000,
    "min": 60_000_000_000,
    "s": 1_000_000_000,
    "ms": 1_000_000,
    "us": 1_000,
    "ns": 1,
}
"""The nanoseconds in each unit a period can be counted in, a day and finer."""

_KINDS = ("Y", "Q", "M", "W", "B", *_UNITS)
_CALENDAR = ("Y", "Q", "M", "W")
"""The kinds whose fields are those of their last day."""

_RENAMED = {
    "ME": "M",
    "MS": "M",
    "BME": "M",
    "BMS": "M",
    "SME": "M",
    "SMS": "M",
    "CBME": "M",
    "CBMS": "M",
    "YE": "Y",
    "YS": "Y",
    "BYE": "Y",
    "BYS": "Y",
    "QE": "Q",
    "QS": "Q",
    "BQE": "Q",
    "BQS": "Q",
}
"""The offsets pandas turns away for a period, pointing at the one it means."""

_OLD = {"A": "Y", "H": "h", "T": "min", "S": "s", "L": "ms", "U": "us", "N": "ns"}
"""Spellings pandas dropped, with the one it suggests instead."""

_RESOLUTIONS = {
    "year": "Y",
    "quarter": "Q",
    "month": "M",
    "day": "D",
    "hour": "h",
    "minute": "min",
    "second": "s",
    "millisecond": "ms",
    "microsecond": "us",
    "nanosecond": "ns",
}
"""The frequency a period read from text takes, from how finely the text is written."""

_EPOCH = datetime.date(1970, 1, 1).toordinal()


class _Freq:
    """A period frequency: its kind, the month or weekday it ends on, and its count."""

    __slots__ = ("anchor", "kind", "n")

    def __init__(self, kind: str, anchor: int | None, n: int) -> None:
        self.kind = kind
        self.anchor = anchor
        self.n = n

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Freq):
            return NotImplemented
        return (self.kind, self.anchor, self.n) == (other.kind, other.anchor, other.n)

    def __hash__(self) -> int:
        return hash((self.kind, self.anchor, self.n))

    @property
    def code(self) -> str:
        """The frequency's name without its count, as `Q-DEC` or `min`."""
        if self.kind in ("Y", "Q"):
            return f"{self.kind}-{_MONTHS[self.anchor - 1]}"  # type: ignore[operator]
        if self.kind == "W":
            return f"W-{_WEEKDAYS[self.anchor]}"  # type: ignore[index]
        return self.kind

    @property
    def text(self) -> str:
        """The frequency's name as a period prints it, as `2M`."""
        return self.code if self.n == 1 else f"{self.n}{self.code}"

    def offset(self, n: int | None = None) -> Any:
        """The offset that steps by this frequency, `n` times when given."""
        from . import offsets

        count = self.n if n is None else n
        if self.kind == "Y":
            return offsets.YearEnd(count, month=self.anchor)
        if self.kind == "Q":
            return offsets.QuarterEnd(count, startingMonth=self.anchor)
        if self.kind == "W":
            return offsets.Week(count, weekday=self.anchor)
        kind = {
            "M": offsets.MonthEnd,
            "B": offsets.BusinessDay,
            "D": offsets.Day,
            "h": offsets.Hour,
            "min": offsets.Minute,
            "s": offsets.Second,
            "ms": offsets.Milli,
            "us": offsets.Micro,
            "ns": offsets.Nano,
        }[self.kind]
        return kind(count)

    @property
    def rule_month(self) -> int:
        """The month a year of this frequency ends in, December unless it says."""
        return self.anchor if self.kind in ("Y", "Q") else 12  # type: ignore[return-value]


def _shown(cause: Exception) -> str:
    """An error as pandas quotes it inside another, as `KeyError('X')` or `ValueError('...')`."""
    kind = "KeyError" if isinstance(cause, KeyError) else "ValueError"
    return f"{kind}({cause.args[0]!r})"


def _invalid(text: str, cause: Exception, hint: str = "") -> InvalidArgumentError:
    inner = f"Invalid frequency: {text}. Failed to parse with error message: {_shown(cause)}."
    return _refused(text, InvalidArgumentError(inner + hint), hint)


def _refused(text: str, cause: Exception, hint: str = "") -> InvalidArgumentError:
    return InvalidArgumentError(
        f"Invalid frequency: {text}. Failed to parse with error message: {_shown(cause)}{hint}"
    )


def _positive(freq: _Freq) -> _Freq:
    if freq.n <= 0:
        raise InvalidArgumentError(
            f"Frequency must be positive, because it represents span: {freq.offset().freqstr}"
        )
    return freq


def _of_offset(offset: Any) -> _Freq | None:
    """The period frequency an offset names, or None when no period is that long."""
    from . import offsets

    kind = type(offset)
    if kind is offsets.YearEnd:
        return _Freq("Y", offset.month, offset.n)
    if kind is offsets.QuarterEnd:
        return _Freq("Q", offset.startingMonth, offset.n)
    if kind is offsets.Week and offset.weekday is not None:
        return _Freq("W", offset.weekday, offset.n)
    ticks = {
        offsets.MonthEnd: "M",
        offsets.BusinessDay: "B",
        offsets.Day: "D",
        offsets.Hour: "h",
        offsets.Minute: "min",
        offsets.Second: "s",
        offsets.Milli: "ms",
        offsets.Micro: "us",
        offsets.Nano: "ns",
    }
    if kind in ticks and (kind is not offsets.BusinessDay or not offset.offset):
        return _Freq(ticks[kind], None, offset.n)
    return None


def _of_span(nanos: int) -> _Freq:
    """A fixed length as a period frequency, counted in the largest unit that divides it."""
    for kind in ("h", "min", "s", "ms", "us", "ns"):
        if nanos % _UNITS[kind] == 0:
            return _Freq(kind, None, nanos // _UNITS[kind])
    raise AssertionError(nanos)


def _freq(freq: Any, signed: bool = False) -> _Freq:
    """A frequency as a period reads it, from text, an offset or a span.

    Raises:
        ValueError: With pandas' message for a frequency no period has, for one
            that is not positive unless `signed` allows it, and for text that
            names nothing.
    """
    from . import offsets

    check: Any = (lambda found: found) if signed else _positive
    if isinstance(freq, _Freq):
        return freq
    if isinstance(freq, offsets.BaseOffset):
        if isinstance(freq, offsets.Week) and freq.weekday is None:
            raise InvalidArgumentError("Unknown freq: 0")
        found = _of_offset(freq)
        if found is None:
            raise InvalidArgumentError(f"{freq!r} is not supported as period frequency")
        return check(found)
    if isinstance(freq, (datetime.timedelta, Timedelta)):
        return check(_of_span(Timedelta(freq).value))
    if not isinstance(freq, str):
        raise InvalidArgumentError(f"Invalid frequency: {freq}")
    text = freq.strip()
    found = re.fullmatch(r"(-?\d+)?([A-Za-z]+)(?:-(\w+))?", text)
    if found is None:
        try:
            nanos = Timedelta(text).value
        except ValueError:
            letters = re.match(r"-?\d*([A-Za-z]*)", text)
            raise _invalid(text, KeyError(letters.group(1) or text)) from None  # type: ignore[union-attr]
        return check(_of_span(nanos))
    count = int(found.group(1)) if found.group(1) else 1
    prefix, suffix = found.group(2), found.group(3)
    if prefix in _RENAMED:
        cause = InvalidArgumentError(
            f"for Period, please use '{_RENAMED[prefix]}' instead of '{prefix}'"
        )
        raise _refused(text, cause)
    if prefix in _OLD:
        raise _invalid(text, KeyError(prefix), f" Did you mean {_OLD[prefix]}?")
    if prefix not in _KINDS:
        try:
            offsets._parsed(text)
        except ValueError:
            raise _invalid(text, KeyError(prefix)) from None
        raise InvalidArgumentError(f"{text} is not supported as period frequency")
    anchor: int | None = None
    if prefix in ("Y", "Q"):
        anchor = 12
        if suffix is not None:
            if suffix not in _MONTHS:
                raise _invalid(text, KeyError(prefix))
            anchor = _MONTHS.index(suffix) + 1
    elif prefix == "W":
        anchor = 6
        if suffix is not None:
            if suffix not in _WEEKDAYS:
                raise _invalid(text, KeyError(suffix))
            anchor = _WEEKDAYS.index(suffix)
    elif suffix is not None:
        raise _invalid(text, InvalidArgumentError(f"Bad freq suffix {suffix}"))
    return check(_Freq(prefix, anchor, count))


def _weekday(days: int) -> int:
    """The weekday of a count of days since 1970, Monday being 0."""
    return (days + 3) % 7


def _days(year: int, month: int, day: int) -> int:
    return datetime.date(year, month, day).toordinal() - _EPOCH


def _date(days: int) -> datetime.date:
    return datetime.date.fromordinal(days + _EPOCH)


def _ordinal(freq: _Freq, moment: datetime.datetime, nanos: int, back: bool = False) -> int:
    """The ordinal of the period of a frequency that holds a moment.

    A moment on a weekend has no business day, so it moves to the next one,
    or back to the one before when `back` is set.
    """
    kind, anchor = freq.kind, freq.anchor
    year, month = moment.year, moment.month
    if kind in ("Y", "Q"):
        fiscal = year + (1 if month > anchor else 0)  # type: ignore[operator]
        if kind == "Y":
            return fiscal - 1970
        quarter = ((month - anchor - 1) % 12) // 3  # type: ignore[operator]
        return (fiscal - 1970) * 4 + quarter
    if kind == "M":
        return (year - 1970) * 12 + month - 1
    days = _days(year, month, moment.day)
    if kind == "W":
        last = days + (anchor - _weekday(days)) % 7  # type: ignore[operator]
        return (last - 3) // 7 + 1
    if kind == "B":
        weekday = _weekday(days)
        if weekday > 4:
            days += 4 - weekday if back else 7 - weekday
        return ((days + 4) // 7) * 5 + (days + 4) % 7 - 4
    clock = moment.hour * 3600 + moment.minute * 60 + moment.second
    total = days * _DAY + clock * 1_000_000_000 + moment.microsecond * 1000 + nanos
    return total // _UNITS[kind]


def _first_day(freq: _Freq, ordinal: int) -> int:
    """The first day of a week-or-longer period, as days since 1970."""
    kind, anchor = freq.kind, freq.anchor
    if kind == "W":
        guess = 7 * (ordinal - 1) + 3
        return guess + (anchor - _weekday(guess)) % 7 - 6  # type: ignore[operator]
    if kind == "Y":
        month = (ordinal + 1970) * 12 + anchor - 12  # type: ignore[operator]
    elif kind == "Q":
        fiscal, quarter = divmod(ordinal, 4)
        month = (fiscal + 1970) * 12 + anchor - 1 - 3 * (3 - quarter) - 2  # type: ignore[operator]
    else:
        month = ordinal + 1970 * 12
    return _days(month // 12, month % 12 + 1, 1)


def _bounds(freq: _Freq, ordinal: int) -> tuple[int, int]:
    """The first nanosecond of a period and the one just past it, since 1970."""
    kind = freq.kind
    if kind == "B":
        weeks, weekday = divmod(ordinal + 3, 5)
        day = weeks * 7 + weekday - 3
        return day * _DAY, (day + 1) * _DAY
    if kind in _UNITS:
        return ordinal * _UNITS[kind], (ordinal + 1) * _UNITS[kind]
    return _first_day(freq, ordinal) * _DAY, _first_day(freq, ordinal + 1) * _DAY


def _moment(nanos: int) -> tuple[datetime.datetime, int]:
    """A count of nanoseconds since 1970 as a moment and the nanoseconds past its microsecond."""
    days, rest = divmod(nanos, _DAY)
    micros, extra = divmod(rest, 1000)
    moment = datetime.datetime.combine(_date(days), datetime.time()) + datetime.timedelta(
        microseconds=micros
    )
    return moment, extra


def _how(how: str) -> bool:
    """Whether a `how` names the end of a period rather than its start.

    Raises:
        ValueError: For anything but start, end or their shorter spellings.
    """
    ends = {"S": False, "E": True, "START": False, "FINISH": True, "BEGIN": False, "END": True}
    spelled = how.upper() if isinstance(how, str) else how
    if spelled not in ends:
        raise InvalidArgumentError("How must be one of S or E")
    return ends[spelled]


_ISO = re.compile(
    r"(\d{4})(?:-(\d{1,2})(?:-(\d{1,2})"
    r"(?:[ T](\d{1,2})(?::(\d{2})(?::(\d{2})(?:\.(\d{1,9}))?)?)?)?)?)?"
)
_QUARTER = re.compile(r"(\d{4})-?Q([1-4])|([1-4])Q-?(\d{4})")
_WEEKLY = re.compile(r"(\d{4}-\d{2}-\d{2})/(\d{4}-\d{2}-\d{2})")
_NAMED = re.compile(r"([A-Z]+)[ -](\d{4})")


def _quartered(year: int, quarter: int, rule_month: int) -> tuple[int, int]:
    """The year and month a fiscal quarter starts in, for a year ending in `rule_month`."""
    month = (rule_month + (quarter - 1) * 3) % 12 + 1
    return (year - 1 if month > rule_month else year), month


def _read(text: str, freq: _Freq | None) -> tuple[datetime.datetime, int, str]:
    """A moment written as text, its nanoseconds past the microsecond, and its resolution.

    Raises:
        ValueError: For text that is not a date pandas reads.
    """
    from ._row_dates import DateParseError

    rule_month = 12 if freq is None else freq.rule_month
    found = _QUARTER.fullmatch(text)
    if found is not None:
        year = int(found.group(1) or found.group(4))
        quarter = int(found.group(2) or found.group(3))
        year, month = _quartered(year, quarter, rule_month)
        return datetime.datetime(year, month, 1), 0, "quarter"
    found = _ISO.fullmatch(text)
    if found is None and re.fullmatch(r"\d{8}", text):
        found = _ISO.fullmatch(f"{text[:4]}-{text[4:6]}-{text[6:]}")
    if found is not None:
        parts = found.groups()
        numbers_ = [int(part) for part in parts[:6] if part is not None]
        fraction = parts[6]
        names = ("year", "month", "day", "hour", "minute", "second")
        resolution = names[len(numbers_) - 1]
        numbers_ += [1] * (3 - len(numbers_)) if len(numbers_) < 3 else []
        nanos = 0
        if fraction is not None:
            resolution = ("millisecond", "microsecond", "nanosecond")[(len(fraction) - 1) // 3]
            nanos = int(fraction.ljust(9, "0"))
        try:
            moment = datetime.datetime(*numbers_, microsecond=nanos // 1000)  # type: ignore[misc]
        except ValueError:
            raise DateParseError(
                f"Unknown datetime string format, unable to parse: {text}"
            ) from None
        return moment, nanos % 1000, resolution
    found = _NAMED.fullmatch(text)
    if found is not None:
        names = [name.upper() for name in calendar.month_name[1:]]
        name = found.group(1)
        if name in names or name in _MONTHS:
            month = names.index(name) + 1 if name in names else _MONTHS.index(name) + 1
            return datetime.datetime(int(found.group(2)), month, 1), 0, "month"
    raise DateParseError(f"Unknown datetime string format, unable to parse: {text}")


def _is_nat(value: Any) -> bool:
    if value is NaT:
        return True
    if isinstance(value, float) and value != value:
        return True
    return type(value).__name__ in ("datetime64", "timedelta64") and str(value) == "NaT"


def _span_nanos(value: Any) -> int | None:
    """The nanoseconds in a span or a fixed offset, or None when the value is neither."""
    from . import offsets

    if isinstance(value, (offsets.Tick, offsets.Day)):
        return int(value.nanos)
    if type(value).__name__ == "timedelta64":
        return int(value.astype("timedelta64[ns]").astype("int64"))
    if isinstance(value, (datetime.timedelta, Timedelta)):
        return int(Timedelta(value).value)
    return None


class Period:
    """A span of time named by its frequency and its place among such spans."""

    __slots__ = ("_freq", "_ordinal")

    _freq: _Freq
    _ordinal: int

    def __new__(
        cls,
        value: Any = None,
        freq: Any = None,
        ordinal: Any = None,
        year: Any = None,
        month: Any = None,
        quarter: Any = None,
        day: Any = None,
        hour: Any = None,
        minute: Any = None,
        second: Any = None,
    ) -> Any:
        """Builds a period from a value, from an ordinal, or from its fields.

        Args:
            value: A date as text, a whole number read as text, a moment or a period.
            freq: The frequency, as text or an offset.
            ordinal: The period's place among those of its frequency.
            year: The year, when building from fields.
            month: The month, when building from fields.
            quarter: The quarter of the year, which moves the month to its start.
            day: The day, when building from fields.
            hour: The hour, when building from fields.
            minute: The minute, when building from fields.
            second: The second, when building from fields.

        Returns:
            The period, or `NaT` for a missing value or when nothing is given.

        Raises:
            ValueError: With pandas' message for a frequency no period has, for
                a moment or an ordinal with no frequency, for both a value and an
                ordinal, and for a value of another kind.
        """
        parsed = None if freq is None else _freq(freq)
        if ordinal is not None and value is not None:
            raise InvalidArgumentError(
                "Only value or ordinal but not both should be given but not both"
            )
        if ordinal is not None:
            if parsed is None:
                raise InvalidArgumentError("Must supply freq for ordinal value")
            return cls._made(parsed, int(ordinal))
        if value is None:
            fields = (year, month, quarter, day, hour, minute, second)
            if all(field is None for field in fields):
                return NaT
            if parsed is None:
                raise InvalidArgumentError("If value is None, freq cannot be None")
            return cls._made(parsed, _from_fields(parsed, *fields))
        if isinstance(value, Period):
            if parsed is None:
                return cls._made(value._freq, value._ordinal)
            return value.asfreq(parsed)
        if _is_nat(value):
            return NaT
        nanos = 0
        if isinstance(value, (str, numbers.Integral)) and not isinstance(value, bool):
            text = str(value).upper()
            if text == "NAT":
                return NaT
            weekly = _WEEKLY.fullmatch(text)
            if weekly is not None:
                start, stop = (datetime.date.fromisoformat(part) for part in weekly.groups())
                if (stop - start).days != 6:
                    raise InvalidArgumentError("Could not parse as weekly-freq Period")
                if parsed is None:
                    parsed = _Freq("W", stop.weekday(), 1)
                moment = datetime.datetime.combine(start, datetime.time())
            else:
                moment, nanos, resolution = _read(text, parsed)
                if parsed is None:
                    parsed = _Freq(_RESOLUTIONS[resolution], None, 1)
                    if parsed.kind in ("Y", "Q"):
                        parsed.anchor = 12
        elif isinstance(value, datetime.datetime) or type(value).__name__ == "datetime64":
            moment = Timestamp(value)
            if parsed is None:
                raise InvalidArgumentError("Must supply freq for datetime value")
            if moment.tzinfo is not None:
                warnings.warn(
                    "Converting to Period representation will drop timezone information.",
                    UserWarning,
                    stacklevel=2,
                )
                moment = moment.tz_localize(None)
            nanos = moment.nanosecond
        elif isinstance(value, datetime.date):
            if parsed is None:
                raise InvalidArgumentError("Must supply freq for datetime value")
            moment = datetime.datetime(value.year, value.month, value.day)
        else:
            raise InvalidArgumentError("Value must be Period, string, integer, or datetime")
        return cls._made(parsed, _ordinal(parsed, moment, nanos))

    @classmethod
    def _made(cls, freq: _Freq, ordinal: int) -> Period:
        period = object.__new__(cls)
        period._freq = freq
        period._ordinal = ordinal
        return period

    @classmethod
    def now(cls, freq: Any = None) -> Period:
        """The period holding this moment.

        Raises:
            ValueError: With no frequency, as in pandas.
        """
        return cls(datetime.datetime.now(), freq=freq)

    @property
    def ordinal(self) -> int:
        """The period's place among those of its frequency, counted from 1970."""
        return self._ordinal

    @property
    def freq(self) -> Any:
        """The frequency, as the offset that steps by it."""
        return self._freq.offset()

    @property
    def freqstr(self) -> str:
        """The frequency's name, as `M` or `Q-DEC`."""
        return self._freq.text

    def _fields(self) -> tuple[datetime.datetime, int]:
        """The moment the fields are read from, and its nanoseconds past the microsecond."""
        start, stop = _bounds(self._freq, self._ordinal)
        if self._freq.kind in _CALENDAR:
            return _moment((stop - 1) // _DAY * _DAY)
        return _moment(start)

    @property
    def year(self) -> int:
        """The year."""
        return self._fields()[0].year

    @property
    def month(self) -> int:
        """The month."""
        return self._fields()[0].month

    @property
    def day(self) -> int:
        """The day of the month."""
        return self._fields()[0].day

    @property
    def hour(self) -> int:
        """The hour."""
        return self._fields()[0].hour

    @property
    def minute(self) -> int:
        """The minute."""
        return self._fields()[0].minute

    @property
    def second(self) -> int:
        """The second."""
        return self._fields()[0].second

    @property
    def dayofweek(self) -> int:
        """The day of the week, Monday being 0."""
        return self._fields()[0].weekday()

    day_of_week = dayofweek
    weekday = dayofweek

    @property
    def dayofyear(self) -> int:
        """The day of the year, from 1."""
        return self._fields()[0].timetuple().tm_yday

    day_of_year = dayofyear

    @property
    def days_in_month(self) -> int:
        """How many days the month has."""
        moment = self._fields()[0]
        return calendar.monthrange(moment.year, moment.month)[1]

    daysinmonth = days_in_month

    @property
    def is_leap_year(self) -> bool:
        """Whether the year is a leap year."""
        return calendar.isleap(self._fields()[0].year)

    @property
    def week(self) -> int:
        """The ISO week of the year."""
        return self._fields()[0].isocalendar()[1]

    weekofyear = week

    def _quarter(self) -> tuple[int, int]:
        """The fiscal year and quarter, for a quarterly period, or the calendar ones."""
        if self._freq.kind == "Q":
            fiscal, quarter = divmod(self._ordinal, 4)
            return fiscal + 1970, quarter + 1
        moment = self._fields()[0]
        return moment.year, (moment.month - 1) // 3 + 1

    @property
    def quarter(self) -> int:
        """The quarter, of the fiscal year for a quarterly period."""
        return self._quarter()[1]

    @property
    def qyear(self) -> int:
        """The fiscal year the quarter falls in, the calendar year unless quarterly."""
        return self._quarter()[0]

    def asfreq(self, freq: Any, how: str = "E") -> Period:
        """The period of another frequency that holds this one's start or end.

        A weekend has no business day. The end of a week or longer period on
        a weekend moves back to the Friday before and its start on to the
        Monday after, while a day or a finer period moves the other way, as
        in pandas.

        Args:
            freq: The frequency to move to.
            how: `E` or `end` for the period holding the end, `S` or `start` for
                the start.

        Raises:
            ValueError: For an unknown how, or a frequency no period has.
        """
        end = _how(how)
        target = _freq(freq)
        if end:
            instant = _bounds(self._freq, self._ordinal + self._freq.n - 1)[1] - 1
        else:
            instant = _bounds(self._freq, self._ordinal)[0]
        moment, nanos = _moment(instant)
        back = end if self._freq.kind in _CALENDAR else not end
        return Period._made(target, _ordinal(target, moment, nanos, back))

    def _stamp(self, nanos: int, fine: bool = False) -> Timestamp:
        if fine or self._freq.kind == "ns":
            return Timestamp(nanos)
        return Timestamp(nanos // 1000, unit="us")

    def to_timestamp(self, freq: Any = None, how: str = "start") -> Timestamp:
        """The first moment of the period, or the last one.

        Args:
            freq: With the start, the frequency whose period holding the start
                gives the moment: its last day for a week or longer, its first
                instant for finer ones.
            how: `start` or `end`.

        Raises:
            ValueError: For an unknown how, or a frequency no period has.
        """
        end = _how(how)
        target = None if freq is None else _freq(freq)
        if end:
            fine = self._freq.kind == "ns" or (target is not None and target.kind == "ns")
            step = 1 if fine else 1000
            if self._freq.kind == "B" or (target is not None and target.kind == "B"):
                start = _bounds(self._freq, self._ordinal)[0]
                return self._stamp(start + _DAY - step, fine)
            stop = _bounds(self._freq, self._ordinal + self._freq.n - 1)[1]
            return self._stamp(stop - step, fine)
        if target is None:
            kind = self._freq.kind
            if kind in _CALENDAR:
                kind = "D"
            elif kind in ("h", "min"):
                kind = "s"
            target = _Freq(kind, None, 1)
        moved = self.asfreq(target, "S")
        start, stop = _bounds(target, moved._ordinal)
        if target.kind in _CALENDAR:
            start = (stop - 1) // _DAY * _DAY
        return self._stamp(start)

    @property
    def start_time(self) -> Timestamp:
        """The first moment of the period."""
        return self.to_timestamp(how="S")

    @property
    def end_time(self) -> Timestamp:
        """The last moment of the period."""
        return self.to_timestamp(how="E")

    def _text(self) -> str:
        kind = self._freq.kind
        if kind == "Q":
            fiscal, quarter = self._quarter()
            return f"{fiscal}Q{quarter}"
        if kind == "W":
            first = _date(_bounds(self._freq, self._ordinal)[0] // _DAY)
            last = first + datetime.timedelta(days=6)
            return f"{_day_text(first)}/{_day_text(last)}"
        moment, nanos = self._fields()
        if kind == "Y":
            return str(moment.year)
        if kind == "M":
            return f"{moment.year}-{moment.month:02d}"
        text = _day_text(moment)
        if kind in ("B", "D"):
            return text
        if kind == "h":
            return f"{text} {moment.hour:02d}:00"
        text += f" {moment.hour:02d}:{moment.minute:02d}"
        if kind == "min":
            return text
        text += f":{moment.second:02d}"
        fraction = moment.microsecond * 1000 + nanos
        digits = {"s": 0, "ms": 3, "us": 6, "ns": 9}[kind]
        if digits:
            text += "." + f"{fraction:09d}"[:digits]
        return text

    def strftime(self, fmt: str | None) -> str:
        """The period written with a format, which also reads pandas' own directives.

        `%q` is the quarter, `%F` the fiscal year and `%f` its last two digits,
        and `%l`, `%u` and `%n` the milli, micro and nanoseconds.
        """
        if fmt is None:
            return self._text()
        moment, nanos = self._fields()
        fiscal, quarter = self._quarter()
        fraction = f"{moment.microsecond * 1000 + nanos:09d}"
        own = {
            "q": str(quarter),
            "F": str(fiscal),
            "f": f"{fiscal % 100:02d}",
            "l": fraction[:3],
            "u": fraction[:6],
            "n": fraction,
            "%": "%",
        }
        return re.sub(
            r"%(.)",
            lambda found: own.get(found.group(1)) or moment.strftime(found.group(0)),
            fmt,
        )

    def __str__(self) -> str:
        return self._text()

    def __repr__(self) -> str:
        return f"Period('{self._text()}', '{self._freq.text}')"

    def __format__(self, spec: str) -> str:
        return format(self._text(), spec)

    def __hash__(self) -> int:
        return hash((self._ordinal, self._freq.text))

    def __reduce__(self) -> tuple[Any, ...]:
        return (Period, (None, self.freq, self._ordinal))

    def _different(self, other: str) -> IncompatibleFrequency:
        return IncompatibleFrequency(
            f"Input has different freq={other} from Period(freq={self.freqstr})"
        )

    def _compare(self, other: Any, op: Any) -> Any:
        if isinstance(other, Period):
            if other._freq != self._freq:
                if op is operator.eq:
                    return False
                if op is operator.ne:
                    return True
                raise self._different(other.freqstr)
            return op(self._ordinal, other._ordinal)
        if other is NaT:
            return op is operator.ne
        return NotImplemented

    def __eq__(self, other: object) -> Any:
        return self._compare(other, operator.eq)

    def __ne__(self, other: object) -> Any:
        return self._compare(other, operator.ne)

    def __lt__(self, other: Any) -> Any:
        return self._compare(other, operator.lt)

    def __le__(self, other: Any) -> Any:
        return self._compare(other, operator.le)

    def __gt__(self, other: Any) -> Any:
        return self._compare(other, operator.gt)

    def __ge__(self, other: Any) -> Any:
        return self._compare(other, operator.ge)

    def _moved(self, other: Any, sign: int) -> Any:
        from . import offsets

        if other is NaT:
            return NaT
        if isinstance(other, bool):
            return NotImplemented
        # numpy's spans count as whole numbers, so they are read as spans first.
        if isinstance(other, numbers.Integral) and type(other).__name__ != "timedelta64":
            return Period._made(self._freq, self._ordinal + sign * int(other) * self._freq.n)
        nanos = _span_nanos(other)
        if nanos is not None:
            if type(other).__name__ == "timedelta64" and str(other) == "NaT":
                return NaT
            unit = _UNITS.get(self._freq.kind)
            if unit is None or nanos % unit:
                raise IncompatibleFrequency(
                    f"Input cannot be converted to Period(freq={self.freqstr})"
                )
            return Period._made(self._freq, self._ordinal + sign * nanos // unit)
        if isinstance(other, offsets.BaseOffset):
            found = _of_offset(other)
            mine = (self._freq.kind, self._freq.anchor)
            if found is None or (found.kind, found.anchor) != mine:
                shown = "None" if found is None and isinstance(other, offsets.Week) else None
                if shown is None:
                    shown = other.freqstr if found is None else found.text
                raise self._different(shown)
            return Period._made(self._freq, self._ordinal + sign * other.n)
        return NotImplemented

    def __add__(self, other: Any) -> Any:
        if isinstance(other, Period):
            raise TypeError("unsupported operand type(s) for +: 'Period' and 'Period'")
        return self._moved(other, 1)

    def __radd__(self, other: Any) -> Any:
        return self._moved(other, 1)

    def __sub__(self, other: Any) -> Any:
        if isinstance(other, Period):
            if other._freq != self._freq:
                raise self._different(other.freqstr)
            return self._freq.offset(self._ordinal - other._ordinal)
        return self._moved(other, -1)


def _day_text(moment: datetime.date) -> str:
    return f"{moment.year}-{moment.month:02d}-{moment.day:02d}"


def _from_fields(freq: _Freq, *fields: Any) -> int:
    """The ordinal of the period holding a moment given as its fields."""
    year, month, quarter, day, hour, minute, second = fields
    month = 1 if month is None else month
    if quarter is not None:
        if not 1 <= quarter <= 4:
            raise InvalidArgumentError("Quarter must be 1 <= q <= 4")
        year, month = _quartered(year, quarter, freq.rule_month)
    extra, month = divmod(month - 1, 12)
    moment = datetime.datetime(
        year + extra, month + 1, 1 if day is None else day, hour or 0, minute or 0, second or 0
    )
    return _ordinal(freq, moment, 0)


class PeriodDtype(str):
    """pandas' type for a column of periods, equal to its name, like `period[M]`.

    The frequency may count backwards or not at all here, as `period[-1D]` and
    `period[0D]`, because pandas takes those for the type even though no single
    period can have them.
    """

    type = Period
    """The class of each value."""

    def __new__(cls, freq: Any) -> PeriodDtype:
        from . import offsets

        if isinstance(freq, PeriodDtype):
            return freq
        if not isinstance(freq, (str, offsets.BaseOffset)):
            raise TypeError(
                f"PeriodDtype argument should be string or BaseOffset, got {type(freq).__name__}"
            )
        if isinstance(freq, str):
            inside = re.fullmatch(r"[Pp]eriod\[(.*)\]", freq)
            if inside is not None:
                freq = inside.group(1)
        found = _freq(freq, signed=True)
        made = super().__new__(cls, f"period[{found.text}]")
        made._freq = found
        return made

    def __getnewargs__(self) -> tuple[str]:
        return (self.name,)

    @classmethod
    def construct_from_string(cls, string: str) -> PeriodDtype:
        """The type named by text like `period[M]`.

        Raises:
            TypeError: For text that does not name one, in pandas' words.
        """
        if not isinstance(string, str):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        if re.fullmatch(r"[Pp]eriod\[.*\]", string):
            try:
                return cls(string)
            except ValueError:
                pass
        raise TypeError(f"Cannot construct a 'PeriodDtype' from '{string}'")

    @classmethod
    def is_dtype(cls, dtype: Any) -> bool:
        """Whether a value is this type or text that names it."""
        if isinstance(dtype, str):
            try:
                cls.construct_from_string(dtype)
            except TypeError:
                return False
            return True
        return isinstance(dtype, cls)

    @property
    def freq(self) -> Any:
        """The frequency, as the offset that steps by it."""
        return self._freq.offset()

    @property
    def name(self) -> str:
        """The text, like `period[M]`."""
        return str.__str__(self)

    @property
    def kind(self) -> str:
        """numpy's letter for the type, `O`, since pandas holds periods as objects."""
        return "O"

    @property
    def index_class(self) -> Any:
        """The index a type of periods labels rows with, `PeriodIndex`."""
        from ._period_index import PeriodIndex

        return PeriodIndex

    @property
    def na_value(self) -> Any:
        """What a gap reads as, `NaT`."""
        return NaT

    @property
    def base(self) -> Any:
        """The numpy type pandas reports underneath, object."""
        import numpy

        return numpy.dtype("O")

    @property
    def str(self) -> str:
        """numpy's short text for the type, `|O08`."""
        return "|O08"

    @property
    def num(self) -> int:
        """The number pandas gives the type, 102."""
        return 102

    @property
    def itemsize(self) -> int:
        """The bytes each value takes, 8."""
        return 8

    @property
    def shape(self) -> tuple[()]:
        """The shape of each value, which is none."""
        return ()

    @property
    def names(self) -> None:
        """The field names, which a period type has none of."""
        return None

    @property
    def subdtype(self) -> None:
        """The type inside, which a period type has none of."""
        return None

    @property
    def isbuiltin(self) -> int:
        """numpy's flag for a built in type, 0."""
        return 0

    @property
    def isnative(self) -> int:
        """numpy's flag for native byte order, which pandas gives as 0."""
        return 0

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(("period", self._freq))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PeriodDtype):
            return self._freq == other._freq
        if isinstance(other, str):
            return other[:1].lower() + other[1:] == self.name
        return False

    def __ne__(self, other: object) -> bool:
        return not self == other


_TYPES: dict[str, PeriodDtype] = {}
"""Each period type a column was read with, by its name, so a cell is not parsed each time."""


def period_type(name: Any) -> PeriodDtype:
    """The period type a name or a type stands for, made once for each name."""
    found = _TYPES.get(name) if isinstance(name, str) else None
    if found is None:
        found = PeriodDtype(name)
        _TYPES[found.name] = found
    return found


def period_at(name: str, ordinal: int) -> Period:
    """The period of a column of type `name` with this ordinal."""
    return Period._made(period_type(name)._freq, ordinal)


def period_kind(values: Any) -> str | None:
    """The period type of a list whose values are periods of one frequency and gaps.

    Returns:
        The type's name, or None when a value is anything else, the periods do
        not share a frequency, or there is no period at all.
    """
    from ._objects import is_gap

    found = None
    for value in values:
        if isinstance(value, Period):
            if found is not None and value._freq != found:
                return None
            found = value._freq
        elif not is_gap(value):
            return None
    return None if found is None else f"period[{found.text}]"


def period_ordinals(values: Any, name: str) -> list[int | None]:
    """Each value as the ordinal of its period in type `name`, None for a gap.

    Raises:
        ValueError: For a period of another frequency, as pandas refuses it,
            or a value no period of that frequency can be read from.
    """
    from ._objects import is_gap

    freq = period_type(name)._freq
    ordinals: list[int | None] = []
    for value in values:
        if is_gap(value):
            ordinals.append(None)
            continue
        if isinstance(value, Period):
            if value._freq != freq:
                raise IncompatibleFrequency(
                    f"Input has different freq={value.freqstr} from PeriodIndex(freq={freq.text})"
                )
            ordinals.append(value._ordinal)
            continue
        made = Period(value, freq=freq)
        ordinals.append(None if made is NaT else made._ordinal)
    return ordinals
