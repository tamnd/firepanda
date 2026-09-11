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

What is deliberately absent is listed in document 33 section 6. `freq` and the
three names around it need frequency inference, `time` and `timetz` need a time
of day column type, `to_period` needs a period type, and `to_pydatetime` needs a
column of Python objects. None of them is spelled here, because document 07's
rule is that a name must not resolve and then refuse.
"""

from __future__ import annotations

import datetime
from typing import Any, cast

from ._frame import Index, Series
from ._pandas import (
    _NONEXISTENT,
    _NONEXISTENT_REFUSAL,
    NO_DEFAULT,
    _held_at,
    _spelled,
    to_datetime,
)
from .errors import translate

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
        and there is no second guesser to keep in step.

        Args:
            data: The instants. Text, whole numbers, a firepanda series of
                instants, or another index.
            freq: Refused. Frequency inference is document 33 section 6.
            tz: Refused. Attaching a clock as the values are read is
                `tz_localize` after the fact, which is written.
            ambiguous: Refused away from its default, since choosing between
                the two readings of a repeated wall clock hour needs the zone's
                transition table.
            dayfirst: Refused. Only ISO 8601 is guessed and it has one order.
            yearfirst: Refused, for the same reason.
            dtype: Refused. The unit comes off the values and `as_unit` changes
                it afterwards.
            copy: Refused. There is one behaviour and it always copies.
            name: The level name, or None for unnamed.

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
            "picking which of the two readings a repeated wall clock hour means"
            " needs the zone's transition table",
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
        label = None if name is None else str(name)
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
            values = to_datetime(values)
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

    def _rounded(self, kind: str, freq: Any, ambiguous: Any, nonexistent: Any) -> DatetimeIndex:
        """Moves every label to a frequency, one of three ways.

        The same argument reading `DatetimeMixin._rounded` does, and the reason
        the two policies are only read when the labels carry a zone is written
        there: pandas takes a misspelled `nonexistent` on a naive column without
        looking at it, and refusing what pandas accepts is the direction of
        difference this library does not get to have.
        """
        if (ambiguous != "raise" or nonexistent != "raise") and self.tz is not None:
            _held_at(
                "ambiguous",
                ambiguous,
                "raise",
                "picking which of the two readings a repeated wall clock hour"
                " means needs the zone's transition table",
            )
            if not isinstance(nonexistent, datetime.timedelta):
                _spelled(nonexistent, _NONEXISTENT, _NONEXISTENT_REFUSAL)
            _held_at(
                "nonexistent",
                nonexistent,
                "raise",
                "shifting a wall clock time that a spring forward skipped needs"
                " the zone's transition table",
            )
        if not isinstance(freq, str):
            raise NotImplementedError(
                "freq has to be a string for now, because an offset object"
                " carries the whole frequency vocabulary and firepanda parses"
                " the string spelling only"
            )
        return self._moved(kind, freq)

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
        return self._moved("as_unit", unit)

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
        if tz is None:
            raise NotImplementedError(
                "tz_convert(None) moves the labels to UTC and then takes the"
                " clock off, and taking the clock off is tz_localize(None), so"
                " this is two operations pandas spells as one"
            )
        if not isinstance(tz, str):
            raise NotImplementedError(
                "tz has to be a zone name for now, because a tzinfo object is a"
                " Python object and the kernel reads the zone out of a string"
            )
        return self._moved("tz_convert", tz)

    def tz_localize(
        self, tz: Any, ambiguous: Any = "raise", nonexistent: Any = "raise"
    ) -> DatetimeIndex:
        """Puts the labels on a clock, or takes them off one."""
        _held_at(
            "ambiguous",
            ambiguous,
            "raise",
            "a wall clock hour that a fall back repeats is two instants and"
            " choosing between them needs the zone's transition table",
        )
        if not isinstance(nonexistent, datetime.timedelta):
            _spelled(nonexistent, _NONEXISTENT, _NONEXISTENT_REFUSAL)
        _held_at(
            "nonexistent",
            nonexistent,
            "raise",
            "a wall clock time that a spring forward skipped is no instant at"
            " all and shifting it needs the zone's transition table",
        )
        if tz is None:
            return self._moved("tz_localize_none", "")
        if not isinstance(tz, str):
            raise NotImplementedError(
                "tz has to be a zone name for now, because a tzinfo object is a"
                " Python object and the kernel reads the zone out of a string"
            )
        return self._moved("tz_localize", tz)
