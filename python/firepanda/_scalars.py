"""`Timestamp` and `Timedelta`, the two things a temporal column is made of.

A datetime column has to hand back something when you index into it, take its
maximum, or read one value out of a grouped result. In pandas that something is
a `Timestamp`, and the reason it is not a `datetime.datetime` is three digits:
`datetime` stops at microseconds and Arrow does not. So pandas subclasses
`datetime.datetime` and carries the nanoseconds alongside, which is why
`Timestamp("2020-01-01") == datetime(2020, 1, 1)` is True, why the two hash the
same, and why every `datetime` method still works on one. `Timedelta` is the
same arrangement over `datetime.timedelta` and it is what subtracting two
timestamps produces.

Both are written that way here, for the same reason: a scalar that is not a
`datetime` is a scalar that breaks every piece of code holding a `datetime`, and
almost all of that code belongs to somebody else.

### The whole number underneath

Each one holds an integer and a unit and everything else is derived. For
`Timestamp` the integer counts units since the Unix epoch, for `Timedelta` it
counts units of elapsed time, and the unit is one of `s`, `ms`, `us` and `ns`.
That is the same pair the Arrow column carries, deliberately, because a scalar
whose arithmetic disagrees with the column it came out of is worse than no
scalar at all.

The unit is not always `ns`, and this is where pandas 3.0 differs from every
version anyone remembers. `Timestamp("2026-09-05")` is microseconds,
`Timestamp(datetime.date(2020, 1, 1))` is seconds, and `Timestamp(1600000000)`
is nanoseconds. The inference is measured rather than documented and it is
copied here, because `as_unit`, `resolution` and the precision of every value
that comes back downstream depend on it.

### The frequency string, parsed twice

`round`, `floor` and `ceil` take a frequency, and firepanda already parses
frequencies, in `firepanda/kernel/temporal.mojo`, for `Series.dt.floor` and its
two neighbours. A scalar cannot reach that parser without building a column to
carry one value through it, so there is a second parser here. Two parsers for
one vocabulary is a drift risk, and the thing that manages it is not the comment
you are reading: it is `test_the_scalar_and_the_column_round_the_same_way`,
which asks both of them for the same answer over every alias and compares.

The rounding rule is copied out of the kernel rather than reinvented: floor
division so a negative count behaves, and half to even on an exact tie.

### Zones, which the scalar can do and the column cannot

`tz_localize` and `tz_convert` need a zone database. The kernel does not have
one and #349 is the record of that. This half is Python, and `zoneinfo` is in
the standard library and reads the system database, so the scalar can answer
about `Europe/Paris` while a column still cannot.

That asymmetry is real and it is worth stating rather than leaving to be found.
It is the better of the two available choices: refusing here would mean a
correct `Timestamp` that cannot do the one thing people reach for a `Timestamp`
to do, and it would not make the column any more capable.

### numpy, which is a different problem here

In `api.types` the numpy question is one of inspection, and it is answered
without a dependency, because a value the caller is holding cannot be a numpy
scalar unless numpy is already imported. That does not work here. `to_numpy`,
`asm8` and `to_datetime64` have to produce a numpy object, and nothing can
produce one out of an install that has no numpy in it. So those three import it
when called and say plainly what is missing when the import fails.

### Where these stop short of pandas, which is four places and no more

Every other answer in this file was compared against a running pandas 3.0.3 and
matches it, including which exception class comes out. Four do not, and they are
written down here because a list of known differences is worth more than a
promise of none.

`NaT` does not exist. pandas turns `Timestamp(None)` and `Timedelta(None)` into
a missing value singleton that is equal to nothing including itself, and there
is no such object here yet, so both of those inputs are refused. A missing
temporal value inside a column is an Arrow null and works; a missing one on its
own has nowhere to go.

A moment past the year 9999 is out of range. pandas will build one, because its
`Timestamp` says it is a `datetime` and is not really one, so `Timestamp(1600000000,
unit="D")` reaches the year 4382621 there. This one really is a `datetime`, which
is the property that makes every library holding a `datetime` keep working, and a
`datetime` stops at 9999.

`to_julian_date` hands back a plain float where pandas hands back a
`numpy.float64`. That is the same call as `Series.dtype` being a string: a value
whose type depends on what else is installed is worse than one that is always
the same, and the two compare equal.

`replace` works on a moment before 1678 where pandas refuses. pandas holds such
a moment, prints it and does arithmetic on it, and then refuses to replace a
field on it, because its `replace` checks the nanosecond range rather than the
range of the unit the value is quoted at. That is an inconsistency inside pandas
rather than a rule, and a moment that exists should stay usable.
"""

from __future__ import annotations

import datetime as _datetime
import re
import warnings
from typing import Any, NamedTuple

from .errors import (
    DTypeError,
    InvalidArgumentError,
    NumericOverflowError,
    UnsupportedError,
)

__all__ = ["Components", "Timedelta", "Timestamp"]

# How many nanoseconds each unit is worth. These four are the units Arrow counts
# timestamps in and they are the four a scalar can carry, which is not an
# accident: the scalar and the column have to agree about what a value means.
_UNITS: dict[str, int] = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}

# The unit codes a numpy datetime64 knows, which is a longer list than the four
# above and is not the same question. `as_unit` in pandas answers a day with a
# `NotImplementedError`, because a day is a real unit that pandas has not built
# a timestamp for, and it answers a word that is not a unit at all with a
# `TypeError` from numpy's parser. Two different classes for two different
# mistakes, and telling them apart needs the longer list.
_NUMPY_UNITS: frozenset[str] = frozenset(
    {"Y", "M", "W", "D", "h", "m", "s", "ms", "us", "μs", "ns", "ps", "fs", "as"}
)

# The seven fixed frequencies, with the same nanosecond lengths the kernel gives
# them in `_alias_nanos`. Everything else pandas knows about, a week, a month
# end, a quarter, is a non fixed frequency whose length depends on where in the
# calendar it lands, and pandas refuses to round to those rather than picking an
# average, so they are refused here too.
_FIXED: dict[str, int] = {
    "D": 86_400_000_000_000,
    "h": 3_600_000_000_000,
    "min": 60_000_000_000,
    "s": 1_000_000_000,
    "ms": 1_000_000,
    "us": 1_000,
    "ns": 1,
}

_FREQUENCY = re.compile(r"^\s*([+-]?)(\d*)\s*([A-Za-z]+)\s*$")

# The keyword units `Timedelta(days=1, hours=2)` accepts, in nanoseconds. pandas
# takes the plural spellings here and the singular ones through `unit=`, which
# is why there are two tables rather than one.
_SPANS: dict[str, int] = {
    "weeks": 604_800_000_000_000,
    "days": 86_400_000_000_000,
    "hours": 3_600_000_000_000,
    "minutes": 60_000_000_000,
    "seconds": 1_000_000_000,
    "milliseconds": 1_000_000,
    "microseconds": 1_000,
    "nanoseconds": 1,
}

# What `unit=` means on the constructors. The single letters are pandas'
# abbreviations and they are accepted because a great deal of code passes them.
_ABBREVIATIONS: dict[str, int] = {
    "W": 604_800_000_000_000,
    "D": 86_400_000_000_000,
    "d": 86_400_000_000_000,
    "day": 86_400_000_000_000,
    "days": 86_400_000_000_000,
    "h": 3_600_000_000_000,
    "hour": 3_600_000_000_000,
    "hours": 3_600_000_000_000,
    "m": 60_000_000_000,
    "min": 60_000_000_000,
    "minute": 60_000_000_000,
    "minutes": 60_000_000_000,
    "s": 1_000_000_000,
    "sec": 1_000_000_000,
    "second": 1_000_000_000,
    "seconds": 1_000_000_000,
    "ms": 1_000_000,
    "milli": 1_000_000,
    "millisecond": 1_000_000,
    "milliseconds": 1_000_000,
    "us": 1_000,
    "micro": 1_000,
    "microsecond": 1_000,
    "microseconds": 1_000,
    "ns": 1,
    "nano": 1,
    "nanosecond": 1,
    "nanoseconds": 1,
}

# What a unit written inside a string means, as in `Timedelta("1h30min")`. This
# is a third table rather than a reuse of the one above, because the two
# vocabularies are not the same and measuring is the only way to find that out.
# `Timedelta(1, unit="W")` is a week and `Timedelta("1 week")` is refused, while
# `Timedelta("1hr")` is an hour and `Timedelta(1, unit="hr")` is refused. The
# keys here are lower case and the lookup lowers what it is given, because this
# form is case insensitive where `unit=` is not.
_TEXT_UNITS: dict[str, int] = {
    "w": 604_800_000_000_000,
    "d": 86_400_000_000_000,
    "day": 86_400_000_000_000,
    "days": 86_400_000_000_000,
    "h": 3_600_000_000_000,
    "hr": 3_600_000_000_000,
    "hour": 3_600_000_000_000,
    "hours": 3_600_000_000_000,
    "m": 60_000_000_000,
    "min": 60_000_000_000,
    "minute": 60_000_000_000,
    "minutes": 60_000_000_000,
    "s": 1_000_000_000,
    "sec": 1_000_000_000,
    "second": 1_000_000_000,
    "seconds": 1_000_000_000,
    "ms": 1_000_000,
    "milli": 1_000_000,
    "millis": 1_000_000,
    "millisecond": 1_000_000,
    "milliseconds": 1_000_000,
    "us": 1_000,
    "micro": 1_000,
    "micros": 1_000,
    "microsecond": 1_000,
    "microseconds": 1_000,
    "ns": 1,
    "nano": 1,
    "nanos": 1,
    "nanosecond": 1,
    "nanoseconds": 1,
}

# The eight spellings that still work and warn, mapped to what to write instead.
# Deprecation is part of the contract: a program running under
# `-W error::DeprecationWarning` has to break in the same place in both
# libraries, so a compatibility layer that quietly stops warning has changed the
# behaviour of every strict test suite that imports it. The class here is
# `DeprecationWarning` rather than pandas' `Pandas4Warning`, for the reason
# document 31 gives: firepanda has no version four to point at, and the pandas
# class is a subclass of this one so a filter written against the base catches
# both.
_DEPRECATED_TEXT_UNITS: dict[str, str] = {
    "w": "W",
    "d": "D",
    "H": "h",
    "S": "s",
    "MIN": "min",
    "MS": "ms",
    "US": "us",
    "NS": "ns",
}

# A month and a year are refused rather than averaged, because neither has a
# fixed length and picking one would silently answer a question nobody asked.
# `m` is minutes and `M` is the ambiguous one, which is why this is checked on
# the spelling as written rather than on the lowered form.
_AMBIGUOUS_TEXT_UNITS: frozenset[str] = frozenset({"M", "Y", "y"})

# Which of the four Arrow units a whole number counted in a given unit lands on.
# A count of weeks, days, hours or minutes is a whole number of seconds, so it
# is quoted in seconds, and nothing coarser than a second exists to quote it in.
_COARSEST: dict[int, str] = {
    604_800_000_000_000: "s",
    86_400_000_000_000: "s",
    3_600_000_000_000: "s",
    60_000_000_000: "s",
    1_000_000_000: "s",
    1_000_000: "ms",
    1_000: "us",
    1: "ns",
}

_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


# What `Timedelta.components` hands back. It is a named tuple in pandas and it is
# one here, because callers unpack it positionally as often as they read a field
# off it.
class Components(NamedTuple):
    """The seven pieces an elapsed time breaks into."""

    days: int
    hours: int
    minutes: int
    seconds: int
    milliseconds: int
    microseconds: int
    nanoseconds: int


_ROUND_DOWN = 0
_ROUND_UP = 1
_ROUND_HALF_EVEN = 2

# The sentinel a keyword uses when None is a real answer a caller might pass.
# `Timestamp.replace(tzinfo=None)` means strip the zone, so None cannot also
# mean leave it alone.
_KEEP: Any = object()


def _stepped(value: int, period: int, mode: int) -> int:
    """Moves a whole number to a multiple of a period, one of three ways.

    This is `_stepped_range` out of `firepanda/kernel/temporal.mojo`, written
    again in Python because a scalar has no column to send through the kernel.
    The quotient is a floor division, which is what makes a negative value round
    the way pandas rounds it, and the tie in the half to even case is detected on
    twice the remainder against the signed period rather than against half of it,
    because pandas accepts a negative frequency and its answer for one does not
    agree with its own floor.

    Args:
        value: The whole number, in nanoseconds.
        period: How long one period is, in nanoseconds, never zero.
        mode: Down, up, or half to even.

    Returns:
        The nearest multiple of the period, under the rule the mode names.
    """
    quotient = value // period
    if mode == _ROUND_UP:
        if quotient * period != value:
            quotient += 1
    elif mode == _ROUND_HALF_EVEN:
        twice = (value - quotient * period) * 2
        if twice > period or (twice == period and quotient % 2 != 0):
            quotient += 1
    return quotient * period


def _kind(value: Any) -> str:
    """Names the type of a value the way a CPython error message names one.

    Returns:
        The bare type name, `int` rather than `<class 'int'>`, because that is
        the shape the messages this mirrors are written in.
    """
    return type(value).__name__


def _period(freq: Any) -> int:
    """Turns a pandas frequency string into a length in nanoseconds.

    A frequency is an optional sign, an optional count, and one of the seven
    fixed aliases. This is the same shape `frequency_period` accepts in the
    kernel, and the two are held together by a test rather than by this comment.

    An offset object is accepted as well as a string, and it is accepted by
    asking it to spell itself rather than by knowing anything about it. Every
    pandas offset has a `freqstr` that reads back the way it was written, so
    `Hour(2).freqstr` is `2h` and the parser below takes it from there. That is
    a deliberately shallow reading: it works for the seven fixed offsets and
    refuses the rest for being non fixed, which is what pandas does with them
    too, and it means somebody holding a `pandas.tseries.offsets.Minute` does
    not have to translate it by hand to call `round`.

    Args:
        freq: The frequency, as pandas spells it, or an offset that can spell
            itself.

    Returns:
        The length of one period in nanoseconds, negative if the sign was.

    Raises:
        TypeError: If it is neither of those two, which is the answer pandas
            gives for a bare number.
        ValueError: If it does not parse, or names a non fixed frequency.
    """
    if not isinstance(freq, str):
        spelling = getattr(freq, "freqstr", None)
        if not isinstance(spelling, str):
            raise DTypeError(
                f"Argument 'freq' has incorrect type (expected str, got {_kind(freq)})"
            )
        freq = spelling
    found = _FREQUENCY.match(freq)
    if found is None:
        raise InvalidArgumentError(f"Invalid frequency: {freq}")
    sign, count, alias = found.groups()
    if alias not in _FIXED:
        raise InvalidArgumentError(
            f"<{freq}> is a non-fixed frequency, and the ones that can be rounded"
            " to are D, h, min, s, ms, us and ns, each with an optional count in"
            " front of it"
        )
    period = _FIXED[alias] * (int(count) if count else 1)
    if period == 0:
        raise InvalidArgumentError(f"Invalid frequency: {freq}")
    return -period if sign == "-" else period


def _zone(tz: Any) -> _datetime.tzinfo | None:
    """Reads a zone the way pandas reads one, out of the standard library.

    Args:
        tz: A name, a `tzinfo`, an offset in the form `+01:00`, or None.

    Returns:
        The `tzinfo`, or None for no zone.

    Raises:
        KeyError: If the name is not a zone the system knows about. That is
            zoneinfo's own answer, a `ZoneInfoNotFoundError`, and it is let
            through rather than translated, because pandas lets the same one
            through and code that catches a KeyError around a zone lookup is
            catching what both of them raise.
    """
    if tz is None or isinstance(tz, _datetime.tzinfo):
        return tz
    if not isinstance(tz, str):
        raise InvalidArgumentError(f"Invalid tz argument: {tz!r}")
    if tz in {"UTC", "utc"}:
        return _datetime.UTC
    signed = re.match(r"^([+-])(\d{2}):?(\d{2})$", tz)
    if signed is not None:
        sign, hours, minutes = signed.groups()
        offset = _datetime.timedelta(hours=int(hours), minutes=int(minutes))
        return _datetime.timezone(-offset if sign == "-" else offset)
    import zoneinfo

    return zoneinfo.ZoneInfo(tz)


def _numpy(what: str) -> Any:
    """Imports numpy, or says clearly why the answer cannot be given.

    Three names on these two classes return numpy scalars and nothing can make
    one of those out of an install that has no numpy in it. firepanda has no
    dependencies, so this is a real possibility rather than a defensive check,
    and the message says which of the two sides is missing.

    Args:
        what: The name of the thing being asked for, for the message.

    Returns:
        The numpy module.

    Raises:
        NotImplementedError: If numpy is not installed.
    """
    try:
        import numpy
    except ImportError as missing:
        raise UnsupportedError(
            f"{what} hands back a numpy scalar and numpy is not installed."
            " firepanda has no dependencies, so numpy is yours to install if you"
            " want the numpy forms of these"
        ) from missing
    return numpy


def _unit_of(left: str, right: str) -> str:
    """Picks the finer of two units, which is what pandas keeps in a binary op.

    Args:
        left: One unit.
        right: The other.

    Returns:
        Whichever counts in the smaller pieces.
    """
    return left if _UNITS[left] <= _UNITS[right] else right


def _scaled(value: int, unit: str) -> tuple[int, str]:
    """Checks a unit and returns the value in nanoseconds beside it.

    Args:
        value: The count, in the given unit.
        unit: One of the four.

    Returns:
        The nanosecond count and the unit.

    Raises:
        ValueError: If the unit is not one a column can carry.
    """
    if unit not in _UNITS:
        raise InvalidArgumentError(
            f"unit must be one of {sorted(_UNITS)}, got {unit!r}, because those"
            " are the four resolutions an Arrow timestamp column carries"
        )
    return value * _UNITS[unit], unit


class Timestamp(_datetime.datetime):
    """A moment, to the nanosecond, which is `pandas.Timestamp`.

    It is a `datetime.datetime` with three more digits and a unit. Everything a
    `datetime` can do it can do, comparisons and hashing against a plain
    `datetime` work in both directions, and the extra nanoseconds ride alongside
    rather than in place of the microseconds.

    Attributes:
        value: The whole number of nanoseconds since the Unix epoch.
        unit: The resolution the value is quoted at, one of `s`, `ms`, `us`, `ns`.
        nanosecond: The part below the microsecond, 0 to 999.
    """

    __slots__ = ("_nanosecond", "_unit")

    _nanosecond: int
    _unit: str

    def __new__(
        cls,
        ts_input: Any = _KEEP,
        year: Any = None,
        month: Any = None,
        day: Any = None,
        hour: Any = None,
        minute: Any = None,
        second: Any = None,
        microsecond: Any = None,
        tzinfo: Any = None,
        *,
        nanosecond: Any = None,
        tz: Any = _KEEP,
        unit: Any = None,
        fold: Any = None,
    ) -> Timestamp:
        """Builds a moment out of whatever names one.

        pandas' signature is two constructors sharing one door: a value in the
        first position, or the calendar fields spelled out. The first positional
        argument is a sentinel rather than None because `Timestamp(None)` is a
        thing a caller can write and it is not the same as writing nothing.

        Args:
            ts_input: Text, a whole number, a `datetime`, a `date`, or nothing.
            year: The year, when the fields are being spelled out.
            month: The month.
            day: The day.
            hour: The hour.
            minute: The minute.
            second: The second.
            microsecond: The microsecond.
            tzinfo: The zone, in the position `datetime` puts it.
            nanosecond: The part below the microsecond.
            tz: The zone, by name, which is where pandas puts it.
            unit: What a whole number in `ts_input` counts.
            fold: Which side of a repeated hour, for the standard library.

        Returns:
            The moment.

        Raises:
            TypeError: If the two forms are mixed, or the input names no moment.
            ValueError: If the text does not parse.
        """
        zone = _zone(tz if tz is not _KEEP else tzinfo)
        if ts_input is _KEEP or (isinstance(ts_input, int) and year is not None):
            return cls._from_fields(
                ts_input, year, month, day, hour, minute, second, microsecond,
                nanosecond, zone, fold,
            )  # fmt: skip
        nanos, spelled, found = cls._read(ts_input, unit)
        if nanosecond is not None:
            nanos += int(nanosecond)
            spelled = "ns"
        if zone is not None and found is None:
            # A zone named beside a value that carries none localizes rather
            # than converts, so `Timestamp("13:45", tz="Europe/Paris")` is a
            # quarter to two in Paris and not a quarter to two in London seen
            # from Paris. The two readings differ by the offset and only one of
            # them is pandas', which is worth a line here because handing the
            # zone straight to the builder below gives the other one.
            made = cls._from_nanos(nanos, spelled, None, fold)
            return made.tz_localize(zone)
        return cls._from_nanos(nanos, spelled, zone if zone is not None else found, fold)

    @classmethod
    def _from_fields(
        cls,
        ts_input: Any,
        year: Any,
        month: Any,
        day: Any,
        hour: Any,
        minute: Any,
        second: Any,
        microsecond: Any,
        nanosecond: Any,
        zone: Any,
        fold: Any,
    ) -> Timestamp:
        """Builds from the calendar fields, which is the second constructor.

        Returns:
            The moment.

        Raises:
            TypeError: If the year, month and day are not all there.
        """
        if ts_input is not _KEEP:
            year, month, day = ts_input, year, month
        if year is None or month is None or day is None:
            raise DTypeError(
                "a Timestamp built from fields needs at least a year, a month and a day"
            )
        made = _datetime.datetime(
            int(year), int(month), int(day), int(hour or 0), int(minute or 0),
            int(second or 0), int(microsecond or 0), zone, fold=int(fold or 0),
        )  # fmt: skip
        extra = int(nanosecond or 0)
        return cls._wrap(made, extra, "ns" if extra else "us")

    @classmethod
    def _read(cls, ts_input: Any, unit: Any) -> tuple[int, str, Any]:
        """Reads whatever was passed into nanoseconds, a unit and a zone.

        The unit that comes back is pandas 3.0's inference and it is measured
        rather than documented. A whole number is nanoseconds unless `unit=` says
        otherwise, a `date` is seconds because a date has no time in it, a
        `datetime` is microseconds because that is as fine as one goes, and text
        is microseconds unless it spelled out more digits than that.

        Returns:
            The nanoseconds since the epoch, the unit, and the zone if the input
            carried one.

        Raises:
            TypeError: If nothing here knows how to read it.
        """
        if isinstance(ts_input, Timestamp):
            return ts_input._total, ts_input.unit, ts_input.tzinfo
        if isinstance(ts_input, bool):
            raise DTypeError("Cannot convert input [True] of type <class 'bool'> to Timestamp")
        if isinstance(ts_input, int):
            scale = _ABBREVIATIONS.get(unit, 1) if unit is not None else 1
            if unit is not None and unit not in _ABBREVIATIONS:
                raise InvalidArgumentError(f"Invalid unit: {unit!r}")
            return ts_input * scale, cls._unit_for(scale), None
        if isinstance(ts_input, float):
            scale = _ABBREVIATIONS.get(unit, 1) if unit is not None else 1
            return round(ts_input * scale), "ns", None
        if isinstance(ts_input, _datetime.datetime):
            return cls._epoch(ts_input), "us", ts_input.tzinfo
        if isinstance(ts_input, _datetime.date):
            midnight = _datetime.datetime(ts_input.year, ts_input.month, ts_input.day)
            return cls._epoch(midnight), "s", None
        if isinstance(ts_input, str):
            return cls._parse(ts_input)
        raise DTypeError(
            f"Cannot convert input [{ts_input!r}] of type {type(ts_input)} to Timestamp"
        )

    @staticmethod
    def _unit_for(scale: int) -> str:
        """Names the coarsest unit that holds a scale exactly.

        Returns:
            One of the four unit names.
        """
        for name in ("s", "ms", "us"):
            if scale == _UNITS[name]:
                return name
        return "ns"

    @staticmethod
    def _epoch(made: _datetime.datetime) -> int:
        """Counts nanoseconds from the Unix epoch to a datetime.

        A naive datetime is read as if it were UTC, which is what pandas does and
        what makes `Timestamp(datetime(1970, 1, 1)).value` zero rather than an
        offset that depends on where the machine is.

        The plain `datetime` is rebuilt field by field rather than with
        `replace`, because `replace` on a `Timestamp` hands back a `Timestamp`,
        whose subtraction asks for `value`, which is what called this. Field by
        field is the only spelling here that is not a recursion.

        Returns:
            The nanosecond count, which may be negative.
        """
        bare = _datetime.datetime(
            made.year,
            made.month,
            made.day,
            made.hour,
            made.minute,
            made.second,
            made.microsecond,
        )
        whole = bare - _datetime.datetime(1970, 1, 1)
        nanos = (whole.days * 86_400 + whole.seconds) * 1_000_000_000
        nanos += whole.microseconds * 1_000
        if made.tzinfo is not None:
            offset = made.utcoffset()
            if offset is not None:
                nanos -= int(offset.total_seconds() * 1_000_000_000)
        return nanos

    @classmethod
    def _parse(cls, text: str) -> tuple[int, str, Any]:
        """Reads an ISO 8601 string, which is the only shape a scalar accepts.

        Returns:
            The nanoseconds, the unit, and the zone if the text carried one.

        Raises:
            ValueError: If it does not parse.
        """
        trimmed = text.strip()
        fraction = re.search(r"[.,](\d+)", trimmed)
        digits = fraction.group(1) if fraction is not None else ""
        cleaned = trimmed
        if fraction is not None and len(digits) > 6:
            cleaned = trimmed[: fraction.start() + 7] + trimmed[fraction.end() :]
        try:
            made = _datetime.datetime.fromisoformat(cleaned.replace(" ", "T", 1))
        except ValueError as bad:
            raise InvalidArgumentError(
                f"Could not parse {text!r} as a Timestamp. A scalar takes the ISO"
                " 8601 forms; use firepanda.to_datetime for a column, which reads"
                " more of them"
            ) from bad
        nanos = cls._epoch(made)
        if len(digits) > 6:
            nanos += int(digits[6:9].ljust(3, "0"))
            return nanos, "ns", made.tzinfo
        return nanos, "us", made.tzinfo

    @classmethod
    def _from_nanos(cls, nanos: int, unit: str, zone: Any, fold: Any = None) -> Timestamp:
        """Builds a moment from a nanosecond count, which is the one true door.

        Returns:
            The moment.

        The range is checked in the unit the moment is quoted at and not in
        nanoseconds, which is the difference between reaching 2262 and reaching
        9999. A moment counted in microseconds has the same sixty four bits to
        spend and each one buys a thousand times as much, so `Timestamp("1000-01-01")`
        is an ordinary microsecond moment and only `value` has a problem with it.

        Raises:
            ValueError: If it does not fit in a signed 64 bit count of its own
                unit. It is a ValueError rather than firepanda's
                OutOfBoundsError, which is an IndexError, because pandas raises
                OutOfBoundsDatetime and that is a ValueError.
        """
        counted = nanos // _UNITS[unit]
        if not -9_223_372_036_854_775_808 <= counted <= 9_223_372_036_854_775_807:
            raise InvalidArgumentError(f"Out of bounds nanosecond timestamp: {nanos}")
        seconds, rest = divmod(nanos, 1_000_000_000)
        try:
            made = _datetime.datetime(1970, 1, 1, tzinfo=_datetime.UTC) + _datetime.timedelta(
                seconds=seconds, microseconds=rest // 1_000
            )
        except OverflowError as over:
            raise InvalidArgumentError(f"Out of bounds nanosecond timestamp: {nanos}") from over
        made = made.replace(tzinfo=None) if zone is None else made.astimezone(zone)
        if fold is not None:
            made = made.replace(fold=int(fold))
        return cls._wrap(made, rest % 1_000, unit)

    @classmethod
    def _wrap(cls, made: _datetime.datetime, extra: int, unit: str) -> Timestamp:
        """Puts the datetime fields and the nanosecond remainder into one object.

        Returns:
            The moment.
        """
        self = _datetime.datetime.__new__(
            cls, made.year, made.month, made.day, made.hour, made.minute,
            made.second, made.microsecond, made.tzinfo, fold=made.fold,
        )  # fmt: skip
        object.__setattr__(self, "_nanosecond", extra)
        object.__setattr__(self, "_unit", unit)
        return self

    @property
    def nanosecond(self) -> int:
        """The part below the microsecond, 0 to 999."""
        return self._nanosecond

    @property
    def unit(self) -> str:
        """The resolution this moment is quoted at."""
        return self._unit

    @property
    def _total(self) -> int:
        """The nanoseconds since the epoch, with no range check on the way out.

        This is what the arithmetic here reads and `value` is what a caller
        reads. The two differ for a moment outside the nanosecond window, which
        `value` refuses to hand back and the arithmetic has to go on working
        with, so a subtraction between two moments in the year 1000 still gives
        an answer rather than tripping over its own guard rail.
        """
        return self._epoch(self) + self._nanosecond

    @property
    def value(self) -> int:
        """The whole number of nanoseconds since the Unix epoch.

        Raises:
            OverflowError: If this moment is outside the roughly six hundred
                year window a signed 64 bit nanosecond count covers. A moment
                quoted in microseconds reaches the year 9999 and one quoted in
                nanoseconds stops in 2262, so the moment can exist and still
                have no nanosecond count, and pandas raises here for the same
                reason and with the same words.
        """
        total = self._total
        if not -9_223_372_036_854_775_808 <= total <= 9_223_372_036_854_775_807:
            raise NumericOverflowError(
                "Cannot convert Timestamp to nanoseconds without overflow. Use"
                " `.asm8.view('i8')` to cast represent Timestamp in its own unit"
                f" (here, {self._unit})."
            )
        return total

    @property
    def asm8(self) -> Any:
        """The same moment as a numpy datetime64."""
        return self.to_datetime64()

    @property
    def tz(self) -> Any:
        """The zone, which is another name for `tzinfo` and the one pandas uses."""
        return self.tzinfo

    @property
    def dayofweek(self) -> int:
        """Monday is 0 and Sunday is 6, which is `datetime.weekday`."""
        return self.weekday()

    day_of_week = dayofweek

    @property
    def dayofyear(self) -> int:
        """Which day of the year this is, counting from 1."""
        return self.timetuple().tm_yday

    day_of_year = dayofyear

    @property
    def days_in_month(self) -> int:
        """How many days the month this falls in has."""
        import calendar

        return calendar.monthrange(self.year, self.month)[1]

    daysinmonth = days_in_month

    @property
    def quarter(self) -> int:
        """Which quarter of the year this falls in, 1 to 4."""
        return (self.month - 1) // 3 + 1

    @property
    def week(self) -> int:
        """The ISO week number."""
        return self.isocalendar()[1]

    weekofyear = week

    @property
    def is_leap_year(self) -> bool:
        """Whether this year has a twenty ninth of February in it."""
        import calendar

        return calendar.isleap(self.year)

    @property
    def is_month_start(self) -> bool:
        """Whether this is the first day of a month, at midnight or not."""
        return self.day == 1

    @property
    def is_month_end(self) -> bool:
        """Whether this is the last day of a month."""
        return self.day == self.days_in_month

    @property
    def is_quarter_start(self) -> bool:
        """Whether this is the first day of a quarter."""
        return self.day == 1 and self.month in {1, 4, 7, 10}

    @property
    def is_quarter_end(self) -> bool:
        """Whether this is the last day of a quarter."""
        return self.day == self.days_in_month and self.month in {3, 6, 9, 12}

    @property
    def is_year_start(self) -> bool:
        """Whether this is the first day of a year."""
        return self.day == 1 and self.month == 1

    @property
    def is_year_end(self) -> bool:
        """Whether this is the last day of a year."""
        return self.day == 31 and self.month == 12

    def day_name(self, locale: Any = None) -> str:
        """The name of the day of the week.

        Args:
            locale: Refused, because firepanda does not carry a locale database.

        Returns:
            The English name, from Monday to Sunday.

        Raises:
            NotImplementedError: If a locale is asked for.
        """
        if locale is not None:
            raise UnsupportedError(
                "locale is not supported yet, because naming a day in another"
                " language needs a locale database that firepanda does not carry"
            )
        return _DAYS[self.weekday()]

    def month_name(self, locale: Any = None) -> str:
        """The name of the month.

        Args:
            locale: Refused, for the same reason as `day_name`.

        Returns:
            The English name, from January to December.

        Raises:
            NotImplementedError: If a locale is asked for.
        """
        if locale is not None:
            raise UnsupportedError(
                "locale is not supported yet, because naming a month in another"
                " language needs a locale database that firepanda does not carry"
            )
        return _MONTHS[self.month - 1]

    def normalize(self) -> Timestamp:
        """Midnight on the same day, keeping the zone.

        Returns:
            The moment, with the time of day removed.
        """
        bare = self.replace(hour=0, minute=0, second=0, microsecond=0, nanosecond=0)
        return bare

    def as_unit(self, unit: str, round_ok: bool = True) -> Timestamp:
        """The same moment quoted at a different resolution.

        Args:
            unit: One of `s`, `ms`, `us`, `ns`.
            round_ok: Whether losing precision is allowed.

        Returns:
            The moment at the new unit.

        Raises:
            NotImplementedError: If the unit is a real one that a scalar cannot
                be quoted at, such as a day, which is the class pandas raises
                here and not the one a bad argument usually gets.
            TypeError: If the word is not a unit at all, which is the class
                numpy's parser raises underneath pandas.
            ValueError: If it would lose precision and `round_ok` is False.
        """
        if unit not in _UNITS:
            if unit not in _NUMPY_UNITS:
                raise DTypeError(f'Invalid datetime unit in metadata string "[{unit}]"')
            raise UnsupportedError("Only resolutions 's', 'ms', 'us', 'ns' are supported.")
        size = _UNITS[unit]
        moved = self._total // size * size
        if moved != self._total and not round_ok:
            raise InvalidArgumentError(
                f"Cannot losslessly convert units from {self.unit} to {unit}"
            )
        return type(self)._from_nanos(moved, unit, self.tzinfo)

    def round(self, freq: Any, ambiguous: Any = "raise", nonexistent: Any = "raise") -> Timestamp:
        """The nearest multiple of a frequency, with a tie going to the even one.

        Args:
            freq: A fixed frequency, as pandas spells it.
            ambiguous: What to do on a repeated local hour.
            nonexistent: What to do on a local hour that does not exist.

        Returns:
            The rounded moment.

        Raises:
            ValueError: If the frequency is not a fixed one.
        """
        return self._rounded(freq, _ROUND_HALF_EVEN, ambiguous, nonexistent)

    def floor(self, freq: Any, ambiguous: Any = "raise", nonexistent: Any = "raise") -> Timestamp:
        """The multiple of a frequency at or before this one.

        Args:
            freq: A fixed frequency.
            ambiguous: What to do on a repeated local hour.
            nonexistent: What to do on a local hour that does not exist.

        Returns:
            The floored moment.

        Raises:
            ValueError: If the frequency is not a fixed one.
        """
        return self._rounded(freq, _ROUND_DOWN, ambiguous, nonexistent)

    def ceil(self, freq: Any, ambiguous: Any = "raise", nonexistent: Any = "raise") -> Timestamp:
        """The multiple of a frequency at or after this one.

        Args:
            freq: A fixed frequency.
            ambiguous: What to do on a repeated local hour.
            nonexistent: What to do on a local hour that does not exist.

        Returns:
            The raised moment.

        Raises:
            ValueError: If the frequency is not a fixed one.
        """
        return self._rounded(freq, _ROUND_UP, ambiguous, nonexistent)

    def _rounded(self, freq: Any, mode: int, ambiguous: Any, nonexistent: Any) -> Timestamp:
        """Rounds against the local clock rather than against the instant.

        A zoned moment rounds on the wall clock, which is what makes midnight in
        Paris the floor of a Paris afternoon rather than an hour before it, so the
        offset comes off before the step and goes back on afterwards.

        Returns:
            The moment, moved.

        Raises:
            NotImplementedError: If either of the zone policies is asked for,
                since both of them need the zone database the kernel does not
                have.
        """
        for name, given in (("ambiguous", ambiguous), ("nonexistent", nonexistent)):
            if given != "raise":
                raise UnsupportedError(
                    f"{name}= is not supported yet, because choosing a side of a"
                    " daylight saving change needs the zone transition table,"
                    " which is firepanda#349"
                )
        period = _period(freq)
        offset = self.utcoffset()
        shift = 0 if offset is None else int(offset.total_seconds() * 1_000_000_000)
        moved = _stepped(self._total + shift, period, mode) - shift
        return type(self)._from_nanos(moved, self.unit, self.tzinfo)

    # Deliberately not the signature `datetime.replace` has. pandas takes None
    # here to mean leave the field alone, where the standard library takes a
    # missing argument to mean that, and pandas puts `nanosecond` in the middle
    # of the list. Matching pandas is the point of the class, so the override is
    # incompatible with the base and that is the right way round.
    def replace(  # type: ignore[override]
        self,
        year: Any = None,
        month: Any = None,
        day: Any = None,
        hour: Any = None,
        minute: Any = None,
        second: Any = None,
        microsecond: Any = None,
        nanosecond: Any = None,
        tzinfo: Any = _KEEP,
        fold: Any = None,
    ) -> Timestamp:
        """The same moment with some fields changed.

        pandas takes None here to mean leave it alone, which `datetime.replace`
        does not, and it takes `tzinfo=None` to mean strip the zone, which is why
        that one argument has a sentinel of its own.

        Args:
            year: The year, or None to keep it.
            month: The month, or None.
            day: The day, or None.
            hour: The hour, or None.
            minute: The minute, or None.
            second: The second, or None.
            microsecond: The microsecond, or None.
            nanosecond: The nanosecond, or None.
            tzinfo: The zone, None to remove it, or left out to keep it.
            fold: Which side of a repeated hour, or None.

        Returns:
            The changed moment.
        """
        zone = self.tzinfo if tzinfo is _KEEP else _zone(tzinfo)
        made = _datetime.datetime(
            self.year if year is None else int(year),
            self.month if month is None else int(month),
            self.day if day is None else int(day),
            self.hour if hour is None else int(hour),
            self.minute if minute is None else int(minute),
            self.second if second is None else int(second),
            self.microsecond if microsecond is None else int(microsecond),
            zone,
            fold=self.fold if fold is None else int(fold),
        )
        extra = self.nanosecond if nanosecond is None else int(nanosecond)
        return type(self)._wrap(made, extra, self.unit)

    def tz_localize(
        self, tz: Any, ambiguous: Any = "raise", nonexistent: Any = "raise"
    ) -> Timestamp:
        """Attaches a zone to a moment that had none, keeping the wall clock.

        Args:
            tz: The zone, or None to strip one.
            ambiguous: What to do on a repeated local hour.
            nonexistent: What to do on a local hour that does not exist.

        Returns:
            The moment in that zone, reading the same on the clock.

        Raises:
            TypeError: If it already has a zone.
            NotImplementedError: If either policy is asked for.
        """
        for name, given in (("ambiguous", ambiguous), ("nonexistent", nonexistent)):
            if given != "raise":
                raise UnsupportedError(
                    f"{name}= is not supported yet, because choosing a side of a"
                    " daylight saving change needs the zone transition table,"
                    " which is firepanda#349"
                )
        if tz is None:
            # Stripping a zone off something that has none is a no operation and
            # not a mistake. pandas hands the moment straight back, which is the
            # behaviour a caller writing `tz_localize(None)` over a mixed pile of
            # timestamps is relying on.
            return self if self.tzinfo is None else self.replace(tzinfo=None)
        if self.tzinfo is not None:
            raise DTypeError("Cannot localize tz-aware Timestamp, use tz_convert for conversions")
        return self.replace(tzinfo=_zone(tz))

    def tz_convert(self, tz: Any) -> Timestamp:
        """Moves a moment into another zone, keeping the instant.

        Args:
            tz: The zone, or None to strip one and read it as UTC.

        Returns:
            The same instant, on a different clock.

        Raises:
            TypeError: If it has no zone to convert from.
        """
        if self.tzinfo is None:
            raise DTypeError("Cannot convert tz-naive Timestamp, use tz_localize to localize")
        return type(self)._from_nanos(self._total, self.unit, _zone(tz))

    def astimezone(self, tz: Any = None) -> Timestamp:
        """The standard library's name for `tz_convert`.

        Args:
            tz: The zone.

        Returns:
            The same instant, on a different clock.
        """
        return self.tz_convert(tz)

    def to_pydatetime(self, warn: bool = True) -> _datetime.datetime:
        """The plain `datetime` underneath, which cannot carry the nanoseconds.

        Args:
            warn: Whether to say so when there are nanoseconds to lose.

        Returns:
            A `datetime.datetime`, microseconds at best.
        """
        if warn and self.nanosecond:
            import warnings

            warnings.warn(
                "Discarding nonzero nanoseconds in conversion.",
                UserWarning,
                stacklevel=2,
            )
        return _datetime.datetime(
            self.year, self.month, self.day, self.hour, self.minute,
            self.second, self.microsecond, self.tzinfo, fold=self.fold,
        )  # fmt: skip

    def to_datetime64(self) -> Any:
        """The same moment as a numpy datetime64.

        Returns:
            A `numpy.datetime64` at this moment's own unit.

        Raises:
            NotImplementedError: If numpy is not installed.
        """
        numpy = _numpy("Timestamp.to_datetime64")
        size = _UNITS[self.unit]
        return numpy.datetime64(self._total // size, self.unit)

    def to_numpy(self, dtype: Any = None, copy: bool = False) -> Any:
        """The same moment as a numpy datetime64, which is what pandas calls this.

        Args:
            dtype: Refused, since the only answer is a datetime64.
            copy: Ignored, since a scalar is a copy already.

        Returns:
            A `numpy.datetime64`.

        Raises:
            ValueError: If a dtype is asked for.
        """
        if dtype is not None:
            raise InvalidArgumentError("dtype and copy arguments are ignored")
        return self.to_datetime64()

    def timestamp(self) -> float:
        """Seconds since the Unix epoch, as a float.

        The standard library reads a naive `datetime` as local time here and
        pandas reads it as UTC, which is a seven hour difference on this machine
        and a different one on the next. pandas is what this follows, so the
        answer comes off the nanosecond count rather than off the base class.

        Returns:
            The seconds, rounded to microseconds the way pandas rounds them,
            because a float has no room for the last three digits anyway.
        """
        return round(self._total / 1_000_000_000, 6)

    def to_julian_date(self) -> float:
        """The Julian day number, which is days since noon on the first of January 4713 BC.

        Returns:
            The day number, with the time of day in the fraction.
        """
        year, month = self.year, self.month
        if month <= 2:
            year -= 1
            month += 12
        a = year // 100
        b = 2 - a + a // 4
        days = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + self.day + b - 1524.5
        seconds = self.hour * 3600 + self.minute * 60 + self.second
        fraction = seconds / 86_400 + self.microsecond / 86_400_000_000
        return days + fraction + self.nanosecond / 86_400_000_000_000

    def to_period(self, freq: Any = None) -> Any:
        """The period this moment falls in, which needs a type firepanda has not got.

        Args:
            freq: The period length.

        Raises:
            NotImplementedError: Always, since there is no `Period` yet.
        """
        raise UnsupportedError(
            "to_period is not supported yet, because it hands back a Period and"
            " firepanda has no Period type; it is its own namespace on the board"
        )

    def isoformat(self, sep: str = "T", timespec: str = "auto") -> str:
        """The ISO 8601 spelling, with the nanoseconds in it when there are any.

        Args:
            sep: What goes between the date and the time.
            timespec: How much of the time to print.

        Returns:
            The text.
        """
        base = self.to_pydatetime(warn=False).isoformat(sep=sep, timespec=timespec)
        if not self.nanosecond or timespec not in {"auto", "nanoseconds"}:
            return base
        if self.microsecond == 0:
            head, _, tail = base.partition("+")
            head = f"{head}.000000"
            base = head + ("+" + tail if tail else "")
        cut = base.find(".") + 7
        return base[:cut] + f"{self.nanosecond:03d}" + base[cut:]

    def strftime(self, format: str) -> str:
        """The moment rendered through a C style format string.

        This is what `datetime.strftime` does and nothing else, and it is written
        out rather than inherited for one reason: the inherited one is a C level
        callable whose signature `inspect.signature` cannot read, and pandas spells
        its own out. A caller reading either library with `inspect` has to see the
        same parameter, so this one is spelled out too.

        The nanoseconds are not in the answer, because `%f` is microseconds and
        there is no directive below it. pandas drops them here as well.

        Args:
            format: The format string.

        Returns:
            The text.
        """
        return super().strftime(format)

    @classmethod
    def fromtimestamp(cls, ts: float, tz: Any = None) -> Timestamp:
        """The moment a Unix timestamp names, in local time unless a zone is given.

        Args:
            ts: Seconds since the epoch.
            tz: The zone.

        Returns:
            The moment.
        """
        return cls(_datetime.datetime.fromtimestamp(ts, _zone(tz)))

    @classmethod
    def utcfromtimestamp(cls, ts: float) -> Timestamp:
        """The moment a Unix timestamp names, in UTC.

        Args:
            ts: Seconds since the epoch.

        Returns:
            The moment, zoned to UTC.
        """
        return cls(_datetime.datetime.fromtimestamp(ts, _datetime.UTC))

    @classmethod
    def fromordinal(cls, ordinal: int, tz: Any = None) -> Timestamp:
        """Midnight on the day a proleptic Gregorian ordinal names.

        Args:
            ordinal: Days since the first of January of year 1.
            tz: The zone.

        Returns:
            The moment.
        """
        made = _datetime.datetime.fromordinal(ordinal)
        return cls(made.replace(tzinfo=_zone(tz)))

    @classmethod
    def now(cls, tz: Any = None) -> Timestamp:
        """This moment.

        Args:
            tz: The zone, or None for local time with no zone attached.

        Returns:
            The moment.
        """
        return cls(_datetime.datetime.now(_zone(tz)))

    @classmethod
    def today(cls, tz: Any = None) -> Timestamp:
        """This moment, which is what pandas means by today rather than midnight.

        Args:
            tz: The zone.

        Returns:
            The moment.
        """
        return cls.now(tz)

    @classmethod
    def utcnow(cls) -> Timestamp:
        """This moment, in UTC.

        Returns:
            The moment, zoned to UTC.
        """
        return cls(_datetime.datetime.now(_datetime.UTC))

    @classmethod
    def combine(cls, date: Any, time: Any) -> Timestamp:  # type: ignore[override]
        """A date and a time of day put together.

        Args:
            date: The day.
            time: The time of day.

        Returns:
            The moment.
        """
        return cls(_datetime.datetime.combine(date, time))

    @classmethod
    def fromisoformat(cls, object: str, /) -> Timestamp:
        """The moment an ISO 8601 string names.

        The parameter is named `object` because that is what pandas names it, and
        the conformance board compares parameter names. It is positional only in
        both libraries, so no caller can be holding the name either way.

        Args:
            object: The text.

        Returns:
            The moment.
        """
        return cls(object)

    @classmethod
    def strptime(cls, date_string: Any, format: Any) -> Timestamp:
        """Refused, which is what pandas does with this one too.

        Args:
            date_string: The text.
            format: The format.

        Raises:
            NotImplementedError: Always.
        """
        raise UnsupportedError(
            "Timestamp.strptime() is not implemented. Use to_datetime() to parse date strings."
        )

    def __repr__(self) -> str:
        """The spelling pandas uses, which quotes the text and names the zone.

        Returns:
            The text.
        """
        zone = f", tz='{self.tzinfo}'" if self.tzinfo is not None else ""
        return f"Timestamp('{self._spelled(offset=True)}'{zone})"

    def __str__(self) -> str:
        """The moment without the wrapper, which is what printing gives.

        The offset carries a colon here where `__repr__` writes it without one,
        so the same moment is `13:45:06+02:00` printed and `13:45:06+0200`
        echoed. There is no reason for the two to differ and pandas differs
        anyway, so both are copied.

        Returns:
            The text.
        """
        return self._spelled(offset=True, colon=True)

    def _spelled(self, offset: bool, colon: bool = False) -> str:
        """The date and time with a space in the middle, pandas' printed form.

        Returns:
            The text.
        """
        # The year is not padded to four digits, because pandas does not pad it
        # and `Timestamp("0001-01-01")` prints as `1-01-01` there. That only
        # shows up before the year 1000, which a microsecond moment can reach.
        base = (
            f"{self.year}-{self.month:02d}-{self.day:02d} "
            f"{self.hour:02d}:{self.minute:02d}:{self.second:02d}"
        )
        # Nine digits when there is anything below the microsecond and six when
        # there is not, and never trimmed. The unit does not come into it, which
        # is the surprising half: a moment quoted in nanoseconds still prints six
        # digits when its last three are zeros.
        fraction = self.microsecond * 1_000 + self.nanosecond
        if self.nanosecond:
            base += f".{fraction:09d}"
        elif self.microsecond:
            base += f".{self.microsecond:06d}"
        if offset and self.tzinfo is not None:
            found = self.utcoffset()
            if found is not None:
                total = int(found.total_seconds())
                sign = "-" if total < 0 else "+"
                hours, rest = divmod(abs(total), 3600)
                gap = ":" if colon else ""
                minutes, seconds = divmod(rest, 60)
                base += f"{sign}{hours:02d}{gap}{minutes:02d}"
                # Seconds in an offset are not a curiosity from the far past
                # alone, they are what every zone had before the railways, and
                # `Europe/Paris` carries nine minutes and twenty one seconds of
                # them until 1911. Dropping them prints a moment that is not the
                # moment, so they go in when they are there and stay out when
                # they are not.
                if seconds:
                    base += f"{gap}{seconds:02d}"
        return base

    def __hash__(self) -> int:
        """The hash, which agrees with equality on both sides of the nanosecond.

        A moment on a whole microsecond is equal to the plain `datetime` holding
        it, so it hashes like one. A moment with nanoseconds in it is equal to no
        `datetime`, so it hashes on the nanosecond count. This is pandas' split
        and it is what keeps a dictionary keyed on datetimes working when a
        firepanda column drops one of these into it.

        Returns:
            The hash.
        """
        if self._nanosecond == 0:
            return _datetime.datetime.__hash__(self)
        return hash(self._total)

    def _finer(self, other: Any) -> str:
        """Picks the unit an answer built out of this and another thing is quoted at.

        Returns:
            The finer of the two units, which is the one that loses nothing.
        """
        theirs = other.unit if isinstance(other, Timestamp | Timedelta) else "us"
        return min(self._unit, theirs, key=lambda name: _UNITS[name])

    def __add__(self, other: Any) -> Any:
        """A moment plus an elapsed time is a moment.

        Returns:
            The moved moment, or NotImplemented.
        """
        if isinstance(other, _datetime.timedelta):
            nanos = other._nanos if isinstance(other, Timedelta) else Timedelta(other)._nanos
            return type(self)._from_nanos(self._total + nanos, self._finer(other), self.tzinfo)
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other: Any) -> Any:
        """A moment minus a moment is an elapsed time, minus an elapsed time is a moment.

        Returns:
            The difference, or NotImplemented.
        """
        if isinstance(other, _datetime.datetime):
            mine = self._total
            theirs = other._total if isinstance(other, Timestamp) else self._epoch(other)
            # The finer of the two units, because a difference is only as coarse
            # as the coarser of the things it came from is wrong: subtracting a
            # date from a nanosecond reading has nanoseconds in it.
            return Timedelta._from_nanos(mine - theirs, self._finer(other))
        if isinstance(other, _datetime.timedelta):
            nanos = other._nanos if isinstance(other, Timedelta) else Timedelta(other)._nanos
            return type(self)._from_nanos(self._total - nanos, self._finer(other), self.tzinfo)
        return NotImplemented

    def __rsub__(self, other: Any) -> Any:
        """A plain datetime minus a moment is an elapsed time.

        Returns:
            The difference, or NotImplemented.
        """
        if isinstance(other, _datetime.datetime):
            return Timedelta._from_nanos(self._epoch(other) - self._total, self._finer(other))
        return NotImplemented


Timestamp.min = Timestamp._from_nanos(-9_223_372_036_854_775_807, "ns", None)
"""The earliest moment a nanosecond count reaches, which is in 1677."""

Timestamp.max = Timestamp._from_nanos(9_223_372_036_854_775_807, "ns", None)
"""The latest moment a nanosecond count reaches, which is in 2262."""


class Timedelta(_datetime.timedelta):
    """An elapsed time, to the nanosecond, which is `pandas.Timedelta`.

    It is a `datetime.timedelta` with three more digits, and it is what
    subtracting one `Timestamp` from another produces.

    Attributes:
        value: The whole number of nanoseconds.
        unit: The resolution the value is quoted at.
    """

    __slots__ = ("_nanos", "_unit")

    _nanos: int
    _unit: str

    def __new__(cls, value: Any = _KEEP, unit: Any = None, **kwargs: Any) -> Timedelta:
        """Builds an elapsed time out of whatever names one.

        The keyword collector is named `kwargs` because that is what pandas names
        it and the conformance board compares parameter names. `fields` reads
        better and is what the helper below still calls it.

        Args:
            value: Text, a whole number, or a `timedelta`.
            unit: What a whole number counts.
            **kwargs: `days`, `hours` and the rest, when no value is given.

        Returns:
            The elapsed time.

        Raises:
            TypeError: If the input names no elapsed time.
            ValueError: If the text or the unit does not parse.
        """
        if value is _KEEP:
            return cls._from_nanos(*cls._from_fields(kwargs))
        # The named fields are dropped rather than refused when a value is given
        # too, because that is what pandas does with them: `Timedelta(1, "s",
        # days=1)` is one second there and the day goes nowhere. Refusing would
        # be the better design and it is not the one being copied.
        return cls._from_nanos(*cls._read(value, unit))

    @staticmethod
    def _from_fields(fields: dict[str, Any]) -> tuple[int, str]:
        """Adds up the named spans, in nanoseconds.

        Returns:
            The total and the unit it should be quoted at, which is nanoseconds
            only when nanoseconds were named and microseconds otherwise. That is
            pandas' rule and it is why `Timedelta(days=1).unit` is `us`.

        Raises:
            TypeError: If a name is not one of the eight.
        """
        total = 0
        for name, given in fields.items():
            if name not in _SPANS:
                raise InvalidArgumentError(
                    "cannot construct a Timedelta from the passed arguments,"
                    " allowed keywords are [weeks, days, hours, minutes,"
                    " seconds, milliseconds, microseconds, nanoseconds]"
                )
            total += round(given * _SPANS[name])
        return total, "ns" if "nanoseconds" in fields else "us"

    @classmethod
    def _read(cls, value: Any, unit: Any) -> tuple[int, str]:
        """Reads whatever was passed into nanoseconds and a unit.

        The unit that comes back is inferred rather than asked for, and the rule
        is measured out of pandas rather than reasoned about. A whole number
        with a unit lands on the coarsest of the four Arrow units that unit fits
        inside, so a count of days is seconds and a count of milliseconds is
        milliseconds. A whole number with no unit is nanoseconds. Anything with
        a fraction in it is nanoseconds, whatever the unit said, which is why
        `Timedelta(1, "D")` is seconds and `Timedelta(1.5, "D")` is not.

        Returns:
            The nanoseconds and the unit.

        Raises:
            TypeError: If nothing here knows how to read it.
        """
        if isinstance(value, Timedelta):
            return value._nanos, value.unit
        if isinstance(value, _datetime.timedelta):
            whole = (value.days * 86_400 + value.seconds) * 1_000_000_000
            return whole + value.microseconds * 1_000, "us"
        if isinstance(value, bool):
            raise InvalidArgumentError(
                "Value must be Timedelta, string, integer, float, timedelta or"
                " convertible, not bool"
            )
        if isinstance(value, int | float):
            if unit is not None and unit not in _ABBREVIATIONS:
                raise InvalidArgumentError(f"Invalid unit: {unit!r}")
            scale = _ABBREVIATIONS[unit] if unit is not None else 1
            if isinstance(value, float):
                return int(value * scale), "ns"
            return value * scale, _COARSEST[scale] if unit is not None else "ns"
        if isinstance(value, str):
            return cls._parse(value)
        raise InvalidArgumentError(
            "Value must be Timedelta, string, integer, float, timedelta or"
            f" convertible, not {_kind(value)}"
        )

    @staticmethod
    def _parse(text: str) -> tuple[int, str]:
        """Reads either of the two shapes pandas accepts in a string.

        There are two, and only the first of them is the shape a span prints as.
        `1 days 02:03:04.000005006` is the printed form and it round trips. The
        other is a run of counts with their units written next to them, as in
        `1h30min` or `1 day, 2:03:04`, and it is the shape people actually type.
        Both are here because a library whose constructor only accepts what its
        own repr produces is a library nobody can call by hand, and because the
        conformance board builds its whole `Timedelta` namespace by evaluating
        `Timedelta("1D")` and cannot ask a single question about a class it
        could not construct.

        Returns:
            The nanoseconds and the unit.

        Raises:
            ValueError: If it does not parse.
        """
        found = re.match(
            r"^\s*([+-]?)\s*(?:(\d+)\s*days?\s*,?\s*)?"
            r"(?:\+?\s*(\d+):(\d{1,2})(?::(\d{1,2})(?:[.,](\d+))?)?)?\s*$",
            text,
        )
        if found is None or (found.group(2) is None and found.group(3) is None):
            return Timedelta._parse_counts(text)
        sign, days, hours, minutes, seconds, fraction = found.groups()
        clock = int(hours or 0) * 3_600_000_000_000
        clock += int(minutes or 0) * 60_000_000_000
        clock += int(seconds or 0) * 1_000_000_000
        clock += int((fraction or "").ljust(9, "0")[:9] or 0)
        # The sign lands on the days when there are days, and on the whole span
        # when there are not. That is not an obvious reading, but it is the one
        # that makes the printed form parse back to the value it was printed
        # from: pandas prints a negative span as `-1 days +02:03:04`, which is
        # minus one day plus two hours, and reading the minus as covering both
        # halves would turn a round trip into a different number.
        whole = int(days or 0) * 86_400_000_000_000
        unit = "ns" if fraction is not None and len(fraction) > 6 else "us"
        if days is None:
            return (-clock if sign == "-" else clock), unit
        return (-whole + clock if sign == "-" else whole + clock), unit

    @staticmethod
    def _parse_counts(text: str) -> tuple[int, str]:
        """Reads the `1h30min` shape, which is counts with their units attached.

        Returns:
            The nanoseconds and the unit. The unit is nanoseconds when any count
            was written in them, when any count spelled more than six fractional
            digits, or when the total is not a whole number of microseconds, and
            microseconds otherwise. All three of those were measured rather than
            reasoned about: `Timedelta("0ns")` is quoted in nanoseconds even
            though it is zero, `Timedelta("1.000000000s")` is quoted in them
            because of the digits it wrote and not the value they came to, and
            `Timedelta("1.5us")` is quoted in them because fifteen hundred
            nanoseconds is not a whole count of microseconds.

        Raises:
            ValueError: If it does not parse, if a unit is not one of the
                thirty three, or if it is a month or a year, which have no fixed
                length and are refused rather than averaged.
        """
        body = text.strip()
        negative = body.startswith("-")
        if body[:1] in {"+", "-"}:
            body = body[1:].lstrip()
        if "+" in body or "-" in body:
            # pandas is strict about this and it is right to be. `1D-2h` reads
            # as a day less two hours to a person and as two spans to a parser,
            # and guessing which one was meant is worse than asking.
            raise InvalidArgumentError("only leading negative signs are allowed")
        count = r"(?:\d+(?:\.\d*)?|\.\d+)"
        if not body or re.fullmatch(rf"(?:{count}\s*[A-Za-z]+[\s,]*)+", body) is None:
            if re.match(r"^[A-Za-z]", body):
                raise InvalidArgumentError("unit abbreviation w/o a number")
            raise InvalidArgumentError(
                f"Could not parse {text!r} as a Timedelta. The two shapes it"
                " takes are the one it prints, `1 days 02:03:04.000005006`, and"
                " a run of counts with their units, `1h30min`"
            )
        total = 0
        nanoseconds = False
        for number, spelled in re.findall(rf"({count})\s*([A-Za-z]+)", body):
            if spelled in _AMBIGUOUS_TEXT_UNITS:
                raise InvalidArgumentError(
                    "Units 'M', 'Y' and 'y' do not represent unambiguous"
                    " timedelta values and are not supported."
                )
            scale = _TEXT_UNITS.get(spelled.lower())
            if scale is None:
                raise InvalidArgumentError(f"invalid unit abbreviation: {spelled}")
            if spelled in _DEPRECATED_TEXT_UNITS:
                instead = _DEPRECATED_TEXT_UNITS[spelled]
                warnings.warn(
                    f"'{spelled}' is deprecated and will be removed in a future"
                    f" version. Please use '{instead}' instead of '{spelled}'.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            # Every other unit in this table is matched without regard to case
            # and this one is not, which is a wart rather than a rule. `0nano`
            # and `0NANO` are both quoted in nanoseconds and `0NS` and `0Ns` are
            # quoted in microseconds, because the deprecated spelling is
            # rewritten on the way in and the rewrite is what the resolution is
            # read off. It is copied because a program that reads `.unit` gets
            # the same answer from both libraries or it gets a surprise.
            nanoseconds = nanoseconds or spelled == "ns" or spelled.lower().startswith("nano")
            whole, _, digits = number.partition(".")
            total += int(whole or 0) * scale
            if digits:
                # The whole part and the fractional part are scaled separately,
                # which is what pandas does and is not tidiness. Multiplying the
                # whole thing as one float loses the low digits of a large count
                # long before it loses anything a caller would forgive.
                #
                # The fraction is rounded to as many decimals as the unit has
                # nanoseconds in it, up to nine, and then truncated. That is two
                # rules rather than one and both were measured: `1.0009us` is a
                # thousand and one nanoseconds because the fraction rounds up to
                # a whole nanosecond first, while `1.5ns` is one nanosecond
                # because a nanosecond has no decimals to round to and the
                # truncation is all that is left. Neither is what a reader would
                # guess and the second one is not even self consistent with the
                # first, so it is copied rather than tidied.
                places = min(9, len(str(scale)) - 1)
                fraction = float(f"0.{digits}")
                total += int(round(fraction, places) * scale) if places else int(fraction * scale)
                nanoseconds = nanoseconds or len(digits) > 6
        # The sign goes on at the end, so that a negative count truncates
        # towards zero the same way a positive one does.
        nanoseconds = nanoseconds or total % 1_000 != 0
        return (-total if negative else total), ("ns" if nanoseconds else "us")

    @classmethod
    def _from_nanos(cls, nanos: int, unit: str) -> Timedelta:
        """Builds an elapsed time from a nanosecond count.

        Returns:
            The elapsed time.

        The range is checked in the unit, as it is for a moment, so a span
        quoted in microseconds reaches three hundred thousand years and only a
        span quoted in nanoseconds stops at two hundred and ninety two. That is
        what makes subtracting two moments six hundred years apart give an
        answer instead of an error.

        Raises:
            ValueError: If it does not fit in a signed 64 bit count of its own
                unit. That is a ValueError rather than firepanda's usual
                OutOfBoundsError, which is an IndexError, because pandas raises
                OutOfBoundsTimedelta here and that is a ValueError. A caller
                catching what pandas documents has to catch what pandas raises.
        """
        counted = nanos // _UNITS[unit]
        if not -9_223_372_036_854_775_808 <= counted <= 9_223_372_036_854_775_807:
            raise InvalidArgumentError(f"Cannot cast {nanos} from ns to '{unit}' without overflow.")
        whole, rest = divmod(nanos, 1_000_000_000)
        self = _datetime.timedelta.__new__(cls, seconds=whole, microseconds=rest // 1_000)
        object.__setattr__(self, "_nanos", nanos)
        object.__setattr__(self, "_unit", unit)
        return self

    @property
    def value(self) -> int:
        """The whole number of nanoseconds.

        Raises:
            OverflowError: If the span is longer than a signed 64 bit nanosecond
                count reaches, which a span quoted in a coarser unit can be.
        """
        if not -9_223_372_036_854_775_808 <= self._nanos <= 9_223_372_036_854_775_807:
            raise NumericOverflowError(
                "Cannot convert Timedelta to nanoseconds without overflow. Use"
                " `.asm8.view('i8')` to cast represent Timedelta in its own unit"
                f" (here, {self._unit})."
            )
        return self._nanos

    @property
    def unit(self) -> str:
        """The resolution this is quoted at."""
        return self._unit

    @property
    def nanoseconds(self) -> int:
        """The part below the microsecond, 0 to 999."""
        return self._parts()[6]

    @property
    def asm8(self) -> Any:
        """The same span as a numpy timedelta64."""
        return self.to_timedelta64()

    @property
    def resolution_string(self) -> str:
        """The coarsest unit this span is a whole number of.

        Returns:
            One of `D`, `h`, `min`, `s`, `ms`, `us`, `ns`.
        """
        for name, size in (
            ("D", 86_400_000_000_000),
            ("h", 3_600_000_000_000),
            ("min", 60_000_000_000),
            ("s", 1_000_000_000),
            ("ms", 1_000_000),
            ("us", 1_000),
        ):
            if self._nanos % size == 0:
                return name
        return "ns"

    @property
    def components(self) -> Components:
        """The span broken into its seven named pieces.

        Returns:
            The seven, as a named tuple, which is the shape pandas hands back.
        """
        return Components(*self._parts())

    def _parts(self) -> tuple[int, int, int, int, int, int, int]:
        """Splits the nanosecond count into the seven pieces `components` names.

        The split is a floor division all the way down, which is the same rule
        `datetime.timedelta` normalises itself by and the same one pandas breaks
        a negative span with. It means only the days can be negative and the
        other six are always at or above zero, so minus one nanosecond comes
        back as minus one day and twenty three hours and the rest, rather than
        as a day of zero and a nanosecond of minus one. Splitting the absolute
        value instead and putting the sign on the front reads more naturally and
        gives a different answer, which is why it is not what happens here.

        Returns:
            The seven. Only the first can be negative.
        """
        days, rest = divmod(self._nanos, 86_400_000_000_000)
        hours, rest = divmod(rest, 3_600_000_000_000)
        minutes, rest = divmod(rest, 60_000_000_000)
        seconds, rest = divmod(rest, 1_000_000_000)
        millis, rest = divmod(rest, 1_000_000)
        micros, nanos = divmod(rest, 1_000)
        return (days, hours, minutes, seconds, millis, micros, nanos)

    def total_seconds(self) -> float:
        """How many seconds this is, as a float and therefore not exactly.

        Returns:
            The seconds. The nanoseconds are dropped rather than added in,
            because pandas adds up the days, the seconds and the microseconds
            and stops there. That is visible from outside: one nanosecond is
            zero seconds here and so it is in pandas, and minus one nanosecond
            is about minus one microsecond in both, which is the floor split
            showing through the arithmetic.
        """
        return self.days * 86_400 + self.seconds + self.microseconds / 1_000_000

    def as_unit(self, unit: str, round_ok: bool = True) -> Timedelta:
        """The same span quoted at a different resolution.

        Args:
            unit: One of `s`, `ms`, `us`, `ns`.
            round_ok: Whether losing precision is allowed.

        Returns:
            The span at the new unit.

        Raises:
            NotImplementedError: If the unit is a real one that a scalar cannot
                be quoted at, such as a day, which is the class pandas raises
                here and not the one a bad argument usually gets.
            TypeError: If the word is not a unit at all, which is the class
                numpy's parser raises underneath pandas.
            ValueError: If it would lose precision and `round_ok` is False.
        """
        if unit not in _UNITS:
            if unit not in _NUMPY_UNITS:
                raise DTypeError(f'Invalid datetime unit in metadata string "[{unit}]"')
            raise UnsupportedError("Only resolutions 's', 'ms', 'us', 'ns' are supported.")
        size = _UNITS[unit]
        moved = self._nanos // size * size
        if moved != self._nanos and not round_ok:
            raise InvalidArgumentError(
                f"Cannot losslessly convert units from {self.unit} to {unit}"
            )
        return type(self)._from_nanos(moved, unit)

    def round(self, freq: Any) -> Timedelta:
        """The nearest multiple of a frequency, with a tie going to the even one.

        Args:
            freq: A fixed frequency.

        Returns:
            The rounded span.

        Raises:
            ValueError: If the frequency is not a fixed one.
        """
        stepped = _stepped(self._nanos, _period(freq), _ROUND_HALF_EVEN)
        return type(self)._from_nanos(stepped, self.unit)

    def floor(self, freq: Any) -> Timedelta:
        """The multiple of a frequency at or before this one.

        Args:
            freq: A fixed frequency.

        Returns:
            The floored span.

        Raises:
            ValueError: If the frequency is not a fixed one.
        """
        return type(self)._from_nanos(_stepped(self._nanos, _period(freq), _ROUND_DOWN), self.unit)

    def ceil(self, freq: Any) -> Timedelta:
        """The multiple of a frequency at or after this one.

        Args:
            freq: A fixed frequency.

        Returns:
            The raised span.

        Raises:
            ValueError: If the frequency is not a fixed one.
        """
        return type(self)._from_nanos(_stepped(self._nanos, _period(freq), _ROUND_UP), self.unit)

    def to_pytimedelta(self) -> _datetime.timedelta:
        """The plain `timedelta` underneath, which cannot carry the nanoseconds.

        The nanoseconds are rounded off rather than dropped, and the tie goes
        upwards rather than to even or away from zero, which is measured rather
        than chosen. So minus one nanosecond is nothing here where every other
        reading of the same span floors it to minus one microsecond, and 2500
        nanoseconds is three microseconds where the rounding everywhere else in
        this file would make it two.

        Returns:
            A `datetime.timedelta`, microseconds at best.
        """
        return _datetime.timedelta(microseconds=(self._nanos + 500) // 1_000)

    def to_timedelta64(self) -> Any:
        """The same span as a numpy timedelta64.

        Returns:
            A `numpy.timedelta64`.

        Raises:
            NotImplementedError: If numpy is not installed.
        """
        numpy = _numpy("Timedelta.to_timedelta64")
        return numpy.timedelta64(self._nanos // _UNITS[self._unit], self._unit)

    def to_numpy(self, dtype: Any = None, copy: bool = False) -> Any:
        """The same span as a numpy timedelta64, which is what pandas calls this.

        Args:
            dtype: Refused, since the only answer is a timedelta64.
            copy: Ignored, since a scalar is a copy already.

        Returns:
            A `numpy.timedelta64`.

        Raises:
            ValueError: If a dtype is asked for.
        """
        if dtype is not None:
            raise InvalidArgumentError("dtype and copy arguments are ignored")
        return self.to_timedelta64()

    def view(self, dtype: Any) -> Any:
        """The nanosecond count read as another type, which is a numpy idea.

        Args:
            dtype: What to read it as.

        Returns:
            The reinterpreted value.

        Raises:
            NotImplementedError: If numpy is not installed.
        """
        numpy = _numpy("Timedelta.view")
        return numpy.int64(self._nanos // _UNITS[self._unit]).view(dtype)

    def isoformat(self) -> str:
        """The ISO 8601 duration spelling.

        Returns:
            Text like `P1DT2H3M4.000005006S`.
        """
        days, hours, minutes, seconds, millis, micros, nanos = self._parts()
        # Trailing zeros come off here where `__str__` keeps them, so half a
        # second is `0.5S` in one and `00.500000` in the other. Both are pandas'
        # and the difference is that ISO 8601 spells a fraction and the printed
        # form spells a clock.
        fraction = f"{millis * 1_000_000 + micros * 1_000 + nanos:09d}".rstrip("0")
        tail = f"{seconds}.{fraction}S" if fraction else f"{seconds}S"
        return f"P{days}DT{hours}H{minutes}M{tail}"

    def __repr__(self) -> str:
        """The spelling pandas uses, which quotes the printed form.

        Returns:
            The text.
        """
        return f"Timedelta('{self}')"

    def __str__(self) -> str:
        """The `1 days 02:03:04.000005006` shape, which is what pandas prints.

        Returns:
            The text.
        """
        days, hours, minutes, seconds, millis, micros, nanos = self._parts()
        # The plus in front of the clock when the days are negative is not
        # decoration. A negative span floors to a negative day and a positive
        # remainder, so `-1 days +02:03:04` is the arithmetic written out, and
        # the sign is there to stop it being read as minus the whole thing.
        lead = "+" if days < 0 else ""
        base = f"{days} days {lead}{hours:02d}:{minutes:02d}:{seconds:02d}"
        # Nine digits or six or none, decided by the value and not by the unit,
        # which is worth saying because the unit looks like the thing that ought
        # to decide it. A span quoted in milliseconds still prints six digits,
        # and a span quoted in nanoseconds prints six of them too as soon as the
        # last three are zeros. So the question the printer asks is whether
        # there is anything below the microsecond, then whether there is
        # anything below the second, and it stops at the first no.
        fraction = millis * 1_000_000 + micros * 1_000 + nanos
        if nanos:
            base += f".{fraction:09d}"
        elif fraction:
            base += f".{fraction // 1_000:06d}"
        return base

    def _finer(self, other: Any) -> str:
        """Picks the unit the sum or difference of two spans is quoted at.

        Returns:
            The finer of the two units.
        """
        theirs = other.unit if isinstance(other, Timedelta) else "us"
        return min(self._unit, theirs, key=lambda name: _UNITS[name])

    def __bool__(self) -> bool:
        """Whether this is any time at all.

        The base class cannot answer this, because it keeps microseconds and a
        span of one nanosecond rounds to nothing in them, so the inherited
        answer for one nanosecond is False. It is a real span and pandas says
        True, so the answer comes off the nanosecond count.

        Returns:
            False only for a span of exactly zero.
        """
        return self._nanos != 0

    def __hash__(self) -> int:
        """The hash, which has to agree with equality and does so in two ways.

        A span that is a whole number of microseconds is equal to the plain
        `timedelta` holding the same span, so it has to hash like one, and it
        does. A span with nanoseconds in it is equal to no `timedelta` at all,
        so it hashes on the nanosecond count instead. Both halves are pandas'
        and the split is the reason a dictionary keyed on timedeltas keeps
        working when a firepanda column puts one of these in it.

        Returns:
            The hash.
        """
        if self._nanos % 1_000 == 0:
            return _datetime.timedelta.__hash__(self)
        return hash(self._nanos)

    def __add__(self, other: Any) -> Any:
        """Two elapsed times add.

        Returns:
            The sum, or NotImplemented.
        """
        if isinstance(other, _datetime.datetime):
            return NotImplemented
        if isinstance(other, _datetime.timedelta):
            return type(self)._from_nanos(self._nanos + Timedelta(other)._nanos, self._finer(other))
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other: Any) -> Any:
        """Two elapsed times subtract.

        Returns:
            The difference, or NotImplemented.
        """
        if isinstance(other, _datetime.timedelta):
            return type(self)._from_nanos(self._nanos - Timedelta(other)._nanos, self._finer(other))
        return NotImplemented

    def __rsub__(self, other: Any) -> Any:
        """A plain timedelta minus this one.

        Returns:
            The difference, or NotImplemented.
        """
        if isinstance(other, _datetime.timedelta):
            return type(self)._from_nanos(Timedelta(other)._nanos - self._nanos, self._finer(other))
        return NotImplemented

    def __neg__(self) -> Timedelta:
        """The same span the other way round.

        Returns:
            The negated span.
        """
        return type(self)._from_nanos(-self._nanos, self.unit)

    def __abs__(self) -> Timedelta:
        """The span without its sign.

        Returns:
            The span, never negative.
        """
        return type(self)._from_nanos(abs(self._nanos), self.unit)

    def __mul__(self, other: Any) -> Any:
        """An elapsed time times a number.

        Returns:
            The product, or NotImplemented.
        """
        if isinstance(other, bool) or not isinstance(other, int | float):
            return NotImplemented
        return type(self)._from_nanos(round(self._nanos * other), self.unit)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Any:
        """Divided by a number it is a span, by a span it is a number.

        Returns:
            The quotient, or NotImplemented.
        """
        if isinstance(other, _datetime.timedelta):
            return self._nanos / Timedelta(other)._nanos
        if isinstance(other, bool) or not isinstance(other, int | float):
            return NotImplemented
        # Truncated towards zero rather than rounded, because that is what
        # pandas does and the two disagree on every value that lands on a
        # fraction. A span of minus one day plus two hours divided by two and a
        # half is one nanosecond apart under the two rules.
        return type(self)._from_nanos(int(self._nanos / other), self.unit)

    def __floordiv__(self, other: Any) -> Any:
        """Floor divided by a number it is a span, by a span it is a whole number.

        Returns:
            The quotient, or NotImplemented.
        """
        if isinstance(other, _datetime.timedelta):
            return self._nanos // Timedelta(other)._nanos
        if isinstance(other, bool) or not isinstance(other, int | float):
            return NotImplemented
        return type(self)._from_nanos(int(self._nanos // other), self.unit)


Timedelta.min = Timedelta._from_nanos(-9_223_372_036_854_775_807, "ns")
"""The largest negative span a nanosecond count reaches, about 106752 days."""

Timedelta.max = Timedelta._from_nanos(9_223_372_036_854_775_807, "ns")
"""The largest span a nanosecond count reaches, about 106751 days."""


class _Resolution:
    """The smallest step a scalar can take, which depends on the scalar.

    This is a descriptor rather than a plain attribute because pandas answers
    the question two ways. Asked of the class, `Timestamp.resolution` is one
    nanosecond, the finest thing the type can hold. Asked of a value,
    `Timestamp("2020-01-01").resolution` is one microsecond, because that value
    is quoted in microseconds and a microsecond is as close as its neighbour
    gets. A property would lose the first answer and an attribute would lose the
    second, so it is written out.
    """

    def __get__(self, instance: Any, owner: Any = None) -> Timedelta:
        """Reads the resolution off the value, or off the type when there is no value.

        Returns:
            One unit, as an elapsed time.
        """
        if instance is None:
            return Timedelta._from_nanos(1, "ns")
        return Timedelta._from_nanos(_UNITS[instance.unit], instance.unit)


# The checker reads `resolution` as the plain elapsed time the two base classes
# declare it to be, and a descriptor is not one. Answering the question two ways
# is the whole point of the descriptor and there is no way to say that in the
# declaration, so it is said here instead.
Timedelta.resolution = _Resolution()  # type: ignore[assignment]
"""One unit of whatever the span is quoted at, and one nanosecond off the class."""

Timestamp.resolution = _Resolution()  # type: ignore[assignment]
"""One unit of whatever the moment is quoted at, and one nanosecond off the class."""
