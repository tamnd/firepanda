"""`DatetimeIndex`, which is an index whose labels are instants.

The fourth thing a pandas program holds, after a frame, a column and an index,
and the first one here that is a kind of another one rather than a new thing.
`pandas.DatetimeIndex` is `pandas.Index` with a calendar on it: every set
operation, every lookup and every slice bound is the index's own and the only
thing added is that a label can be asked what year it is in. So this subclasses
`Index` and adds the calendar, and nothing in the core knows the difference,
because a column of instants is int64 underneath and an index over it sorts,
hashes and compares by exactly the rules an int64 index already used.

The calendar members are the `dt` accessor read through another door. Document
33 argues for that at length and the short version is that the labels of an
index are a column, so `PyIndex.temporal_part` materialises them, hands them to
the same kernel `s.dt.year` calls, and wraps the answer back up. There is no
second table of field names anywhere and there is no second set of rules about
what a frequency string means.

What is deliberately absent is `freq` and the three names around it, which need
frequency inference, and `to_period`, which needs a period type. Neither is
spelled here, because document 07's rule is that a name must not resolve and
then refuse. `time`, `timetz` and `to_pydatetime` answer Python lists, the
convention document 41 set for what an index answers label by label, and
`shift` and `snap` count along calendar frequencies with the same steps
`date_range` uses.
"""

from __future__ import annotations

import datetime
import math
import zoneinfo
from typing import Any, cast

from ._frame import Index, Series
from ._pandas import (
    _NONEXISTENT,
    _NONEXISTENT_REFUSAL,
    NO_DEFAULT,
    _beyond_nanoseconds,
    _held_at,
    _instants,
    _is_default,
    _label_of,
    _naive_convert,
    _on_the_clock,
    _spelled,
    _zone_name,
)
from .errors import DTypeError, InvalidArgumentError, translate

__all__ = ["DatetimeIndex"]


def _is_temporal(dtype: Any) -> bool:
    """Whether a firepanda dtype name is one of the instant types.

    A string comparison rather than a type object, because firepanda spells a
    dtype as a string and the unit rides along inside it, so `datetime64[us]`
    and `datetime64[ns, UTC]` both have to answer yes.
    """
    return str(dtype).startswith("datetime64")


class DatetimeIndex(Index):
    """An index whose labels are instants, which is `pandas.DatetimeIndex`.

    Everything an `Index` does, plus the calendar. The labels are a timestamp
    column rather than a column of whole numbers, and every member below either
    reads a part out of them or hands the work to the index underneath.
    """

    __slots__ = ()

    def __init__(
        self,
        data: Any = None,
        freq: Any = NO_DEFAULT,
        tz: Any = NO_DEFAULT,
        ambiguous: str = "raise",
        dayfirst: bool = False,
        yearfirst: bool = False,
        dtype: Any = None,
        copy: bool | None = None,
        name: Any = None,
    ) -> None:
        """Builds an index of instants out of whatever the values are.

        The pandas signature in full, with `data` and `name` honoured and the
        rest refused by name for the reason `_refuse` gives. What reads the
        values is `to_datetime`, which is the one parser in the library, so an
        index of instants and a column of instants are built by the same rules
        and there is no second guesser to keep in step. Text is read the way
        `format="mixed"` reads it, every row with its own format, because that
        is what pandas' constructor does.

        Args:
            data: The instants. Text, whole numbers, a firepanda series of
                instants, or another index.
            freq: Refused. Frequency inference is document 33 section 6.
            tz: Refused. Attaching a clock as the values are read is
                `tz_localize` after the fact, which is written.
            ambiguous: Refused away from its default, since it only means
                something alongside `tz=`, which is refused.
            dayfirst: Refused. Only ISO 8601 is guessed and it has one order.
            yearfirst: Refused, for the same reason.
            dtype: Refused. The unit comes off the values and `as_unit` changes
                it afterwards.
            copy: Refused. There is one behaviour and it always copies.
            name: The level name. Left out, it comes off the data when the
                data is a named series or another index, and is unnamed when
                the data is a list with nobody to name it.

        Raises:
            NotImplementedError: If any of the refused arguments was passed.
        """
        if freq is not NO_DEFAULT:
            raise NotImplementedError(
                "freq= is not supported yet, because holding a frequency means"
                " inferring one and checking the labels against it, which is"
                " document 33 section 6"
            )
        if tz is not NO_DEFAULT:
            raise NotImplementedError(
                "tz= is not supported yet, because reading the values and"
                " attaching a clock at once is two operations, and the second"
                " of them is tz_localize"
            )
        _held_at(
            "ambiguous",
            ambiguous,
            "raise",
            "it only means something alongside tz=, which is tz_localize after the fact",
        )
        _held_at("dayfirst", dayfirst, False, "only ISO 8601 is guessed and it has one order")
        _held_at("yearfirst", yearfirst, False, "only ISO 8601 is guessed and it has one order")
        if dtype is not None:
            raise NotImplementedError(
                "dtype= is not supported yet, because the unit is read off the"
                " values and as_unit is how it is changed afterwards"
            )
        if copy is not None:
            raise NotImplementedError(
                "copy= is not supported yet, because there is exactly one"
                " behaviour and it always copies"
            )
        label = _label_of(data) if name is None else str(name)
        if isinstance(data, Index) and _is_temporal(data.dtype):
            # The labels are already instants, so there is nothing to read and
            # the only thing that can change is the level name. Going through
            # the parser here would read the stored whole numbers back as
            # nanoseconds and silently move a microsecond index by a factor of
            # a thousand, which is issue #348 reached from a second direction.
            try:
                self._inner = data._inner.renamed(label)
            except Exception as error:
                raise translate(error) from None
            return
        values: Any = data.tolist() if isinstance(data, Index) else data
        if not isinstance(values, Series) or not _is_temporal(values.dtype):
            values = _instants(
                values, "raise", False, False, False, "mixed", NO_DEFAULT, None, "unix"
            )
        try:
            self._inner = values._inner.to_index(label)
        except Exception as error:
            raise translate(error) from None

    def _part(self, kind: str, arg: str = "") -> Index:
        """Reads one part of the labels, and hands back a plain index.

        Every calendar field and every name that answers something other than
        instants comes through here. The answer is an `Index` and not a
        `DatetimeIndex`, which is what pandas does too, since the year of an
        instant is a number and not an instant.

        The `kind` asked for here is the kernel's name for the part and not
        always the pandas name for it. pandas spells the day of the week three
        ways and the day of the year two, and the core keeps one name for each
        because a kernel does not need three. The property resolves the spelling
        before it calls, which is the same thing `tools/bindings.py` does for the
        `dt` accessor.
        """
        try:
            return Index._wrap(self._inner.temporal_part(kind, arg))
        except Exception as error:
            raise translate(error) from None

    def _moved(self, kind: str, arg: str) -> DatetimeIndex:
        """Moves every label and hands back another index of instants."""
        try:
            return cast(DatetimeIndex, DatetimeIndex._wrap(self._inner.temporal_part(kind, arg)))
        except Exception as error:
            raise translate(error) from None

    def _placed(self, kind: str, arg: str, ambiguous: Any, nonexistent: Any) -> DatetimeIndex:
        """Moves every label back onto a clock, under pandas' two policies.

        A list of flags for `ambiguous` is answered by asking twice, and the
        two answers are put side by side as columns on the same row numbers so
        that `where` lines them up by position whatever the labels are.
        """

        def place(fold: str, gap: str, shift: int) -> DatetimeIndex:
            try:
                inner = self._inner.temporal_placed(kind, arg, fold, gap, shift)
            except Exception as error:
                raise translate(error) from None
            return cast(DatetimeIndex, DatetimeIndex._wrap(inner))

        def pick(
            flags: list[bool], if_true: DatetimeIndex, if_false: DatetimeIndex
        ) -> DatetimeIndex:
            rows = list(range(len(flags)))
            chosen = if_true.to_series(index=rows).where(flags, if_false.to_series(index=rows))
            return DatetimeIndex(chosen, name=self.name)

        return _on_the_clock(place, ambiguous, nonexistent, len(self), pick)

    def _rounded(self, kind: str, freq: Any, ambiguous: Any, nonexistent: Any) -> DatetimeIndex:
        """Moves every label to a frequency, one of three ways.

        The same argument reading `DatetimeMixin._rounded` does, and the reason
        the two policies are only read when the labels carry a zone is written
        there: pandas takes a misspelled `nonexistent` on a naive column without
        looking at it, and refusing what pandas accepts is the direction of
        difference this library does not get to have.
        """
        if not isinstance(freq, str):
            raise NotImplementedError(
                "freq has to be a string for now, because an offset object"
                " carries the whole frequency vocabulary and firepanda parses"
                " the string spelling only"
            )
        if (_is_default(ambiguous) and _is_default(nonexistent)) or self.tz is None:
            return self._moved(kind, freq)
        return self._placed(kind, freq, ambiguous, nonexistent)

    def _named(self, kind: str, locale: Any) -> Index:
        """Writes out the name of the day or the month."""
        if locale is not None and not isinstance(locale, str):
            raise TypeError(f"locale has to be a string, not {type(locale).__name__}")
        return self._part(kind, "" if locale is None else locale)

    @property
    def year(self) -> Index:
        """The year of every label."""
        return self._part("year")

    @property
    def month(self) -> Index:
        """The month of every label, one through twelve."""
        return self._part("month")

    @property
    def day(self) -> Index:
        """The day of the month of every label."""
        return self._part("day")

    @property
    def hour(self) -> Index:
        """The hour of every label, nought through twenty three."""
        return self._part("hour")

    @property
    def minute(self) -> Index:
        """The minute of every label."""
        return self._part("minute")

    @property
    def second(self) -> Index:
        """The second of every label."""
        return self._part("second")

    @property
    def microsecond(self) -> Index:
        """The microsecond of every label."""
        return self._part("microsecond")

    @property
    def nanosecond(self) -> Index:
        """The nanosecond of every label."""
        return self._part("nanosecond")

    @property
    def quarter(self) -> Index:
        """The quarter of the year of every label, one through four."""
        return self._part("quarter")

    @property
    def dayofweek(self) -> Index:
        """The day of the week of every label, with Monday as nought."""
        return self._part("dayofweek")

    @property
    def day_of_week(self) -> Index:
        """The day of the week of every label. The other spelling of `dayofweek`."""
        return self._part("dayofweek")

    @property
    def weekday(self) -> Index:
        """The day of the week of every label. The third spelling of `dayofweek`."""
        return self._part("dayofweek")

    @property
    def dayofyear(self) -> Index:
        """The day of the year of every label, one through three hundred and sixty six."""
        return self._part("dayofyear")

    @property
    def day_of_year(self) -> Index:
        """The day of the year of every label. The other spelling of `dayofyear`."""
        return self._part("dayofyear")

    @property
    def days_in_month(self) -> Index:
        """How many days are in the month every label falls in."""
        return self._part("days_in_month")

    @property
    def daysinmonth(self) -> Index:
        """How many days are in the month. The other spelling of `days_in_month`."""
        return self._part("days_in_month")

    @property
    def is_leap_year(self) -> Index:
        """Whether every label falls in a leap year."""
        return self._part("is_leap_year")

    @property
    def is_month_start(self) -> Index:
        """Whether every label is the first day of its month."""
        return self._part("is_month_start")

    @property
    def is_month_end(self) -> Index:
        """Whether every label is the last day of its month."""
        return self._part("is_month_end")

    @property
    def is_quarter_start(self) -> Index:
        """Whether every label is the first day of its quarter."""
        return self._part("is_quarter_start")

    @property
    def is_quarter_end(self) -> Index:
        """Whether every label is the last day of its quarter."""
        return self._part("is_quarter_end")

    @property
    def is_year_start(self) -> Index:
        """Whether every label is the first day of its year."""
        return self._part("is_year_start")

    @property
    def is_year_end(self) -> Index:
        """Whether every label is the last day of its year."""
        return self._part("is_year_end")

    @property
    def date(self) -> Index:
        """Every label with the time of day taken off, as a date."""
        return self._part("date")

    @property
    def asi8(self) -> list[Any]:
        """The labels as the whole numbers they are stored as.

        A list where pandas gives a numpy array, which is the divergence
        document 21 section 7 recorded for `Index.values` and is the same
        divergence wearing another name.
        """
        try:
            return list(self._inner.to_list())
        except Exception as error:
            raise translate(error) from None

    @property
    def tz(self) -> str | None:
        """The clock the labels are read against, or None when they carry none."""
        try:
            found = self._inner.temporal_word("tz")
        except Exception as error:
            raise translate(error) from None
        return found or None

    @property
    def unit(self) -> str:
        """How many of the labels' whole numbers make a second, as pandas spells it."""
        try:
            return self._inner.temporal_word("unit")
        except Exception as error:
            raise translate(error) from None

    @property
    def is_normalized(self) -> bool:
        """Whether every label is exactly midnight.

        Asked by normalising and comparing rather than by a kernel of its own,
        because the normalisation is one pass over the labels and a second
        kernel that had to agree with it would be a second chance to disagree.
        """
        try:
            flattened = self._inner.temporal_part("normalize", "").to_list()
            return list(flattened) == list(self._inner.to_list())
        except Exception as error:
            raise translate(error) from None

    def normalize(self) -> DatetimeIndex:
        """Every label with the time of day set to midnight."""
        return self._moved("normalize", "")

    def floor(self, freq: Any, ambiguous: Any = "raise", nonexistent: Any = "raise") -> Any:
        """Every label moved back to the frequency below it."""
        return self._rounded("floor", freq, ambiguous, nonexistent)

    def ceil(self, freq: Any, ambiguous: Any = "raise", nonexistent: Any = "raise") -> Any:
        """Every label moved forward to the frequency above it."""
        return self._rounded("ceil", freq, ambiguous, nonexistent)

    def round(self, freq: Any, ambiguous: Any = "raise", nonexistent: Any = "raise") -> Any:
        """Every label moved to the nearer frequency, with a half going to the even one."""
        return self._rounded("round", freq, ambiguous, nonexistent)

    def as_unit(self, unit: str, round_ok: bool = True) -> DatetimeIndex:
        """The same instants counted in another resolution."""
        _held_at(
            "round_ok",
            round_ok,
            True,
            "refusing a cast that would lose precision rather than rounding it"
            " needs the cast to look at the values first, and it looks at the"
            " types only",
        )
        try:
            return self._moved("as_unit", unit)
        except DTypeError:
            if unit == "ns":
                _beyond_nanoseconds(self.to_series())
            raise

    def day_name(self, locale: Any = None) -> Index:
        """The name of the day of the week of every label."""
        return self._named("day_name", locale)

    def month_name(self, locale: Any = None) -> Index:
        """The name of the month of every label."""
        return self._named("month_name", locale)

    def strftime(self, date_format: str) -> Index:
        """Every label written out through a format string."""
        if not isinstance(date_format, str):
            raise TypeError(f"date_format has to be a string, not {type(date_format).__name__}")
        return self._part("strftime", date_format)

    def tz_convert(self, tz: Any) -> DatetimeIndex:
        """The same instants read against another clock."""
        try:
            if tz is None:
                return self._moved("tz_convert", "UTC").tz_localize(None)
            return self._moved("tz_convert", tz if isinstance(tz, str) else _zone_name(tz))
        except DTypeError as error:
            raise _naive_convert(error) from None

    def tz_localize(
        self, tz: Any, ambiguous: Any = "raise", nonexistent: Any = "raise"
    ) -> DatetimeIndex:
        """Puts the labels on a clock, or takes them off one."""
        if tz is None:
            if not isinstance(nonexistent, datetime.timedelta):
                _spelled(nonexistent, _NONEXISTENT, _NONEXISTENT_REFUSAL)
            return self._moved("tz_localize_none", "")
        if not isinstance(tz, str):
            raise NotImplementedError(
                "tz has to be a zone name for now, because a tzinfo object is a"
                " Python object and the kernel reads the zone out of a string"
            )
        if not (_is_default(ambiguous) and _is_default(nonexistent)):
            return self._placed("tz_localize", tz, ambiguous, nonexistent)
        return self._moved("tz_localize", tz)

    def _labels(self) -> list[Any]:
        """The labels as timestamps, with None for a missing one."""
        try:
            return list(self.tolist())
        except Exception as error:
            raise translate(error) from None

    def _rebuilt(self, labels: list[Any]) -> DatetimeIndex:
        """Wall clock readings put back into an index in this one's unit, clock and name."""
        tz = self.tz
        built = DatetimeIndex(labels, name=self.name).as_unit(self.unit)
        return built.tz_localize(tz) if tz is not None else built

    def _stepped(self, freq: Any, count: int) -> DatetimeIndex:
        """Every label moved `count` steps along a calendar frequency, as pandas adds an offset.

        A label on a landing date moves `count` landings. A label between two
        of them first rolls to the next one the way it is going, which is one
        of the steps, and rolls forward for a count of nought. The time of day
        rides along.
        """
        from ._calendar_steps import calendar_step

        steps = calendar_step(freq)
        assert steps is not None
        moved: list[Any] = []
        for label in self._labels():
            if label is None:
                moved.append(None)
                continue
            day = label.date()
            if not steps.on(day):
                day = steps.roll(day, count >= 0)
                day = steps.move(day, count - 1 if count > 0 else min(count + 1, 0))
            else:
                day = steps.move(day, count)
            moved.append(_wall(label, day))
        return self._rebuilt(moved)

    def shift(self, periods: int = 1, freq: Any = None) -> DatetimeIndex:
        """Every label moved `periods` steps of `freq`.

        Raises:
            NullFrequencyError: Without a frequency. pandas uses the index's
                own frequency there, and firepanda does not keep one on an
                index, so an index made by `date_range` needs `freq` spelled.
        """
        from ._calendar_steps import calendar_step
        from ._date_range import _UNITS, _frequency
        from ._scalars import Timedelta
        from .errors import NullFrequencyError

        if freq is None:
            raise NullFrequencyError("Cannot shift with no freq")
        if calendar_step(freq) is not None:
            return self._stepped(freq, periods)
        step, _, _ = _frequency(freq)
        nanos, scale = periods * step, _UNITS[self.unit]
        by = Timedelta(nanos // scale, unit=self.unit) if nanos % scale == 0 else Timedelta(nanos)
        return DatetimeIndex(self.to_series() + by, name=self.name)

    def snap(self, freq: Any = "S") -> DatetimeIndex:
        """Every label moved to the nearer landing date of `freq`, the later one on a tie.

        A frequency of a fixed length lands everywhere, so the labels come back as they are.
        """
        from ._calendar_steps import calendar_step
        from ._date_range import _frequency

        steps = calendar_step(freq)
        if steps is None:
            _frequency(freq)
            return self.copy()
        snapped: list[Any] = []
        for label in self._labels():
            if label is None:
                raise TypeError("bad operand type for abs(): 'NaTType'")
            day = label.date()
            if not steps.on(day):
                back, forward = steps.roll(day, False), steps.roll(day, True)
                day = back if day - back < forward - day else forward
            snapped.append(_wall(label, day))
        return self._rebuilt(snapped)

    def _micros(self, zone: Any = None) -> list[int]:
        """Every label's time of day in microseconds, read off its own clock or `zone`.

        A missing label is pandas' missing instant read the same way, which
        lands at some time in the evening that depends on the unit, and so
        a range of times that wraps past midnight can take it in. That is
        pandas' answer and it is kept.
        """
        per_second = _PER_SECOND[self.unit]
        missing = (-(2**63)) % (86400 * per_second) * 10**6 // per_second
        found = []
        for label in self._labels():
            if label is None:
                found.append(missing)
                continue
            if zone is not None:
                label = label.astimezone(zone)
            found.append(
                ((label.hour * 60 + label.minute) * 60 + label.second) * 10**6 + label.microsecond
            )
        return found

    def indexer_at_time(self, time: Any, asof: bool = False) -> list[int]:
        """The positions of the labels whose time of day is `time`."""
        if asof:
            raise NotImplementedError("'asof' argument is not supported")
        if isinstance(time, str):
            time = _parsed_time(time)
        if not hasattr(time, "tzinfo"):
            raise AttributeError(f"'{type(time).__name__}' object has no attribute 'tzinfo'")
        if time.tzinfo is not None and self.tz is None:
            raise InvalidArgumentError("Index must be timezone aware.")
        wanted = _time_micros(time)
        found = self._micros(time.tzinfo)
        return [at for at, micros in enumerate(found) if micros == wanted]

    def indexer_between_time(
        self,
        start_time: Any,
        end_time: Any,
        include_start: bool = True,
        include_end: bool = True,
    ) -> list[int]:
        """The positions of the labels whose time of day is between two times.

        When the start is after the end the range wraps past midnight.
        """
        start, end = _time_micros(start_time), _time_micros(end_time)

        def after(micros: int) -> bool:
            return micros >= start if include_start else micros > start

        def before(micros: int) -> bool:
            return micros <= end if include_end else micros < end

        wraps = start > end
        return [
            at
            for at, micros in enumerate(self._micros())
            if ((after(micros) or before(micros)) if wraps else (after(micros) and before(micros)))
        ]

    def isocalendar(self) -> Any:
        """The ISO year, week and day of every label, as a frame on this index."""
        return self.to_series().dt.isocalendar().set_axis(self)

    def mean(self, *, skipna: bool = True, axis: Any = 0) -> Any:
        """The average instant, or None when there is none."""
        if axis not in (0, -1, None):
            raise IndexError("tuple index out of range")
        stamps = self.asi8
        present = [value for value in stamps if value is not None]
        if not present or (not skipna and len(present) < len(stamps)):
            return None
        return self._instant(int(sum(float(value) for value in present) / len(present)))

    def std(
        self,
        axis: Any = None,
        dtype: Any = None,
        out: Any = None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Any:
        """The spread of the instants as a span, or None when there are too few."""
        from ._scalars import Timedelta

        stamps = self.asi8
        present = [float(value) for value in stamps if value is not None]
        count = len(present)
        if count - ddof <= 0 or (not skipna and count < len(stamps)):
            return None
        average = sum(present) / count
        spread = math.sqrt(sum((average - value) ** 2 for value in present) / (count - ddof))
        return Timedelta(int(spread), unit=self.unit)

    def _instant(self, stamp: int) -> Any:
        """A whole number of the index's unit since the epoch, as a timestamp on its clock."""
        from ._scalars import Timestamp

        instant = Timestamp(stamp, unit=self.unit)
        tz = self.tz
        return instant if tz is None else instant.tz_localize("UTC").tz_convert(tz)

    def to_julian_date(self) -> Index:
        """Every label as a Julian date, read off its own clock, NaN for a missing one."""
        return Index(
            [math.nan if label is None else label.to_julian_date() for label in self._labels()],
            name=self.name,
        )

    def to_pydatetime(self) -> list[Any]:
        """Every label as a Python datetime, None for a missing one.

        A list where pandas answers a numpy array of objects, which is the
        convention document 41 set for everything an index answers position by position.
        """
        return [None if label is None else label.to_pydatetime() for label in self._labels()]

    @property
    def time(self) -> list[Any]:
        """The time of day of every label, without the clock."""
        return [None if label is None else label.time() for label in self._labels()]

    @property
    def timetz(self) -> list[Any]:
        """The time of day of every label, with the clock."""
        return [None if label is None else label.timetz() for label in self._labels()]

    @property
    def tzinfo(self) -> Any:
        """The clock the labels are read against, as a tzinfo, or None."""
        tz = self.tz
        if tz is None:
            return None
        for label in self._labels():
            if label is not None:
                return label.tzinfo
        return datetime.UTC if tz == "UTC" else zoneinfo.ZoneInfo(tz)

    @property
    def resolution(self) -> str:
        """The finest part any label uses, from `day` down to `nanosecond`."""
        finest = 0
        for label in self._labels():
            if label is None:
                continue
            finest = max(finest, _finest(label))
        return _RESOLUTIONS[finest]


_PER_SECOND = {"s": 1, "ms": 10**3, "us": 10**6, "ns": 10**9}
"""How many of each unit make a second."""

_RESOLUTIONS = ("day", "hour", "minute", "second", "millisecond", "microsecond", "nanosecond")

_TIME_FORMATS = (
    "%H:%M",
    "%H%M",
    "%I:%M%p",
    "%I%M%p",
    "%H:%M:%S",
    "%H%M%S",
    "%I:%M:%S%p",
    "%I%M%S%p",
)
"""The spellings of a time of day pandas reads after ISO, in the order it tries them."""


def _finest(label: Any) -> int:
    """The position in `_RESOLUTIONS` of the finest part one label uses."""
    nanosecond = getattr(label, "nanosecond", 0)
    parts = (
        nanosecond,
        label.microsecond % 1000,
        label.microsecond,
        label.second,
        label.minute,
        label.hour,
    )
    for at, part in enumerate(parts):
        if part:
            return len(_RESOLUTIONS) - 1 - at
    return 0


def _wall(label: Any, day: datetime.date) -> Any:
    """A label's time of day on another date, as a wall clock timestamp with no clock."""
    from ._scalars import Timestamp

    into = ((label.hour * 60 + label.minute) * 60 + label.second) * 10**9
    into += label.microsecond * 1000 + getattr(label, "nanosecond", 0)
    days = (day - datetime.date(1970, 1, 1)).days
    return Timestamp(days * 86400 * 10**9 + into, unit="ns")


def _parsed_time(text: str) -> datetime.time:
    """A time of day read from text the way `indexer_at_time` reads it.

    pandas hands the text to dateutil, which reads a bare run of digits as a
    date, so that is midnight, and takes `9am` and `9:30 pm` as well as the
    spellings `indexer_between_time` reads.

    Raises:
        InvalidArgumentError: For text that is no time, in dateutil's words.
    """
    plain = text.strip()
    if plain.isdigit():
        return datetime.time()
    try:
        return datetime.time.fromisoformat(plain)
    except ValueError:
        pass
    squeezed = plain.replace(" ", "").upper()
    for spelling in (*_TIME_FORMATS, "%I%p"):
        try:
            return datetime.datetime.strptime(squeezed, spelling).time()
        except ValueError:
            continue
    raise InvalidArgumentError(f"Unknown string format: {text}")


def _time_micros(value: Any) -> int:
    """A time of day, from a time or text, as microseconds into the day.

    Raises:
        InvalidArgumentError: For anything that is no time pandas reads, in its words.
    """
    if isinstance(value, datetime.datetime):
        value = value.time()
    if isinstance(value, str):
        try:
            value = datetime.time.fromisoformat(value)
        except ValueError:
            for spelling in _TIME_FORMATS:
                try:
                    value = datetime.datetime.strptime(value, spelling).time()
                    break
                except ValueError:
                    continue
            else:
                raise InvalidArgumentError(f"Cannot convert arg {[value]} to a time") from None
    if not isinstance(value, datetime.time):
        raise InvalidArgumentError(f"Cannot convert arg {[value]} to a time")
    return ((value.hour * 60 + value.minute) * 60 + value.second) * 10**6 + value.microsecond
