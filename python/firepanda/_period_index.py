"""`PeriodIndex` and `period_range`, pandas' index of periods and its builder.

A `PeriodIndex` is an `Index` whose labels are period cells, the written form
Document 103 gives a period column. Everything an index does on its labels,
sorting, uniques, gaps, joins and slicing, works on the cells as text, since
they sort as their ordinals do. What is here reads the periods back: the
fields, the conversions to instants and other frequencies, the arithmetic,
lookups by a period or its text, and the repr.

An index made by the core from period cells becomes a `PeriodIndex` by
`_class_of`, so a slice, a sort or a series' labels stay one. An index that is
empty or all gaps has no cell to carry its type, so only one built here, which
keeps its type in the slot a `DatetimeIndex` keeps its frequency in, is still a
`PeriodIndex` then.
"""

from __future__ import annotations

import itertools
import numbers
import re
from typing import Any

from . import _objects, _period
from ._frame import Index
from ._period import Period, PeriodDtype
from ._scalars import NaT
from .errors import InvalidArgumentError

__all__ = ["PeriodIndex", "PeriodProperties", "period_range"]


def _kind_of(values: list[Any], freq: Any, dtype: Any) -> str:
    """The period type an index of `values` takes.

    Raises:
        ValueError: With no frequency to be had, as in pandas.
    """
    if dtype is not None:
        kind = _period.period_type(dtype).name
        if freq is not None and _period.period_type(f"period[{freq}]").name != kind:
            raise InvalidArgumentError("specified freq and dtype are different")
        return kind
    if freq is not None:
        return _period.period_type(f"period[{_text(freq)}]").name
    first = next((value for value in values if isinstance(value, Period)), None)
    if first is None:
        raise InvalidArgumentError("freq not specified and cannot be inferred")
    return f"period[{first.freqstr}]"


def _text(freq: Any) -> str:
    """A frequency as the text a period type names it by."""
    return freq if isinstance(freq, str) else _period._freq(freq).text


def _listed(data: Any) -> list[Any]:
    """The values an index is built from, as a list."""
    if data is None:
        return []
    if hasattr(data, "tolist"):
        return list(data.tolist())
    return list(data)


_RESOLUTIONS = {
    "Y": "year",
    "Q": "quarter",
    "M": "month",
    "D": "day",
    "h": "hour",
    "min": "minute",
    "s": "second",
    "ms": "millisecond",
    "us": "microsecond",
    "ns": "nanosecond",
}
"""The unit pandas names for each period frequency, which has none for a week."""


class PeriodIndex(Index):
    """An index whose labels are periods of one frequency, which is `pandas.PeriodIndex`."""

    __slots__ = ()

    def __init__(
        self,
        data: Any = None,
        freq: Any = None,
        dtype: Any = None,
        copy: bool = False,
        name: Any = None,
    ) -> None:
        """Builds an index of periods from periods, their text or instants.

        Raises:
            ValueError: With no frequency to be had, or a value that is no
                period of it.
            IncompatibleFrequency: For a period of another frequency.
        """
        if name is None:
            name = getattr(data, "name", None)
        values = _listed(data)
        kind = _kind_of(values, freq, dtype)
        cells = _objects.period_cells(_period.period_ordinals(values, kind), kind)
        Index.__init__(self, cells, name=name)
        self.__class__ = PeriodIndex
        self._freq = kind

    @classmethod
    def _of(cls, ordinals: list[int | None], kind: str, name: Any) -> PeriodIndex:
        """An index of the periods of type `kind` with these ordinals."""
        made = object.__new__(cls)
        Index.__init__(made, _objects.period_cells(ordinals, kind), name=name)
        made.__class__ = cls
        made._freq = kind
        return made

    @classmethod
    def from_ordinals(cls, ordinals: Any, *, freq: Any, name: Any = None) -> PeriodIndex:
        """The periods of a frequency with these ordinals, counted from 1970."""
        kind = _kind_of([], freq, None)
        return cls._of([int(ordinal) for ordinal in _listed(ordinals)], kind, name)

    @classmethod
    def from_fields(
        cls,
        *,
        year: Any = None,
        quarter: Any = None,
        month: Any = None,
        day: Any = None,
        hour: Any = None,
        minute: Any = None,
        second: Any = None,
        freq: Any = None,
    ) -> PeriodIndex:
        """The periods holding the moments given field by field.

        Raises:
            ValueError: For fields of different lengths.
        """
        if freq is None:
            freq = "Q" if quarter is not None else "M"
        kind = _kind_of([], freq, None)
        found = [year, month, quarter, day, hour, minute, second]
        given = [_listed(field) for field in found if field is not None]
        size = len(given[0]) if given else 0
        if any(len(field) != size for field in given):
            raise InvalidArgumentError("Mismatched Period array lengths")
        columns = [[None] * size if field is None else _listed(field) for field in found]
        at = _period.period_type(kind)._freq
        ordinals = [_period._from_fields(at, *row) for row in zip(*columns, strict=True)]
        return cls._of(list(ordinals), kind, None)

    @property
    def _type(self) -> str:
        """The period type's name, kept beside the labels or read off the first cell."""
        kind = getattr(self, "_freq", None)
        return kind or _objects.period_name_of(self._inner) or "period[D]"

    @property
    def dtype(self) -> PeriodDtype:
        """The period type."""
        return _period.period_type(self._type)

    @property
    def freq(self) -> Any:
        """The frequency, as the offset that steps by it."""
        return self.dtype.freq

    @property
    def freqstr(self) -> str:
        """The frequency's name."""
        return self._type[7:-1]

    @property
    def inferred_type(self) -> str:
        """What pandas infers the labels to be."""
        return "period"

    @property
    def asi8(self) -> Any:
        """Each period's ordinal as numpy whole numbers, the smallest int64 for a gap."""
        import numpy

        least = numpy.iinfo(numpy.int64).min
        found = [least if value is NaT else value.ordinal for value in self._periods()]
        return numpy.array(found, dtype=numpy.int64)

    @property
    def is_full(self) -> bool:
        """Whether no period is missing between the first and the last.

        Raises:
            ValueError: For labels out of order, in pandas' words.
        """
        if len(self) == 0:
            return True
        if not self.is_monotonic_increasing:
            raise InvalidArgumentError("Index is not monotonic")
        ordinals = self.asi8.tolist()
        return all(later - earlier < 2 for earlier, later in itertools.pairwise(ordinals))

    @property
    def resolution(self) -> str:
        """The name of the unit the frequency counts in.

        Raises:
            ValueError: For a weekly or business day frequency, in pandas' words.
        """
        base = self.freqstr.split("-")[0]
        if base not in _RESOLUTIONS:
            raise InvalidArgumentError(f"Invalid frequency: {self.freqstr}")
        return _RESOLUTIONS[base]

    def _periods(self) -> list[Any]:
        """The labels as periods, NaT for a gap."""
        return [NaT if value is None or value is NaT else value for value in self.tolist()]

    def _field(self, pick: Any) -> Index:
        """One field of each period as an index of whole numbers, -1 for a gap."""
        found = [-1 if value is NaT else pick(value) for value in self._periods()]
        return Index(found, name=self.name)

    @property
    def year(self) -> Index:
        """The year of each period."""
        return self._field(lambda value: value.year)

    @property
    def month(self) -> Index:
        """The month of each period."""
        return self._field(lambda value: value.month)

    @property
    def day(self) -> Index:
        """The day of the month of each period."""
        return self._field(lambda value: value.day)

    @property
    def hour(self) -> Index:
        """The hour of each period."""
        return self._field(lambda value: value.hour)

    @property
    def minute(self) -> Index:
        """The minute of each period."""
        return self._field(lambda value: value.minute)

    @property
    def second(self) -> Index:
        """The second of each period."""
        return self._field(lambda value: value.second)

    @property
    def quarter(self) -> Index:
        """The quarter of each period."""
        return self._field(lambda value: value.quarter)

    @property
    def qyear(self) -> Index:
        """The fiscal year each period's quarter belongs to."""
        return self._field(lambda value: value.qyear)

    @property
    def week(self) -> Index:
        """The week of the year of each period."""
        return self._field(lambda value: value.week)

    weekofyear = week

    @property
    def dayofweek(self) -> Index:
        """The day of the week of each period, Monday being 0."""
        return self._field(lambda value: value.dayofweek)

    day_of_week = weekday = dayofweek

    @property
    def dayofyear(self) -> Index:
        """The day of the year of each period."""
        return self._field(lambda value: value.dayofyear)

    day_of_year = dayofyear

    @property
    def days_in_month(self) -> Index:
        """How many days the month of each period has."""
        return self._field(lambda value: value.days_in_month)

    daysinmonth = days_in_month

    @property
    def is_leap_year(self) -> Any:
        """Whether each period's year is a leap year, False for a gap."""
        import numpy

        return numpy.array([value is not NaT and value.is_leap_year for value in self._periods()])

    def _instants(self, pick: Any) -> Any:
        """One instant of each period, as an index of instants."""
        from ._datetime import DatetimeIndex

        found = [NaT if value is NaT else pick(value) for value in self._periods()]
        return DatetimeIndex(found, name=self.name)

    @property
    def start_time(self) -> Any:
        """The first instant of each period."""
        return self._instants(lambda value: value.start_time)

    @property
    def end_time(self) -> Any:
        """The last instant of each period."""
        return self._instants(lambda value: value.end_time)

    def to_timestamp(self, freq: Any = None, how: str = "start") -> Any:
        """Each period as an instant, at its start or its end.

        Starts keep the frequency they are read to step at, as pandas infers
        one for them, and ends have none.
        """
        found = self._instants(lambda value: value.to_timestamp(freq=freq, how=how))
        if not _period._how(how):
            found.freq = found.inferred_freq
        return found

    def _moved(self, move: Any, kind: str | None = None) -> PeriodIndex:
        """Each period changed by `move`, as an index of type `kind`, its own by default."""
        found = [None if value is NaT else move(value) for value in self._periods()]
        if kind is None:
            kind = next(
                (f"period[{value.freqstr}]" for value in found if value is not None), self._type
            )
        return PeriodIndex._of(
            [None if value is None else value.ordinal for value in found], kind, self.name
        )

    def asfreq(self, freq: Any = None, how: str = "E") -> PeriodIndex:
        """Each period as the one of another frequency at its start or its end."""
        kind = _kind_of([], freq, None)
        return self._moved(lambda value: value.asfreq(freq, how=how), kind)

    def strftime(self, date_format: str) -> Index:
        """Each period written by a format, a gap for NaT."""
        found = [None if value is NaT else value.strftime(date_format) for value in self._periods()]
        return Index(found, name=self.name)

    def shift(self, periods: int = 1, freq: Any = None) -> PeriodIndex:
        """Each period moved by a count of its frequency."""
        if freq is not None:
            raise InvalidArgumentError("freq is not a valid argument for PeriodIndex.shift")
        return self + periods

    def __add__(self, other: Any) -> Any:
        if isinstance(other, bool) or not isinstance(other, numbers.Integral | _offset_type()):
            return NotImplemented
        return self._moved(lambda value: value + other)

    __radd__ = __add__

    def __sub__(self, other: Any) -> Any:
        if isinstance(other, Period):
            found = [NaT if value is NaT else value - other for value in self._periods()]
            return Index(found, name=self.name)
        if isinstance(other, bool) or not isinstance(other, numbers.Integral | _offset_type()):
            return NotImplemented
        return self._moved(lambda value: value - other)

    def _compared(self, other: Any, op: str) -> Any:
        import numpy

        if isinstance(other, str):
            other = Period(other, freq=self.freqstr)
        if isinstance(other, Period) and other.freqstr != self.freqstr:
            if op in ("__eq__", "__ne__"):
                return numpy.full(len(self), op == "__ne__")
            raise TypeError(f"Invalid comparison between dtype={self._type} and Period")
        found = [
            op == "__ne__" if value is NaT else bool(getattr(value, op)(other))
            for value in self._periods()
        ]
        return numpy.array(found)

    def __eq__(self, other: object) -> Any:  # type: ignore[override]
        return self._compared(other, "__eq__")

    def __ne__(self, other: object) -> Any:  # type: ignore[override]
        return self._compared(other, "__ne__")

    def __lt__(self, other: Any) -> Any:
        return self._compared(other, "__lt__")

    def __le__(self, other: Any) -> Any:
        return self._compared(other, "__le__")

    def __gt__(self, other: Any) -> Any:
        return self._compared(other, "__gt__")

    def __ge__(self, other: Any) -> Any:
        return self._compared(other, "__ge__")

    __hash__ = None  # type: ignore[assignment]

    def _cell(self, key: Any) -> Any:
        """A key as the cell of its period, or None when it is no period of this frequency."""
        try:
            period = key if isinstance(key, Period) else Period(key, freq=self.freqstr)
        except (ValueError, TypeError):
            return None
        if period is NaT or period.freqstr != self.freqstr:
            return None
        return _objects.period_cells([period.ordinal], self._type)[0]

    def get_loc(self, key: Any) -> Any:
        """Where a period, or its text, is.

        Raises:
            KeyError: For a key that is not there.
        """
        cell = self._cell(key)
        if cell is None:
            raise KeyError(key)
        try:
            return Index.get_loc(self, cell)
        except KeyError:
            raise KeyError(key) from None

    def _bound(self, key: Any) -> Any:
        """A slice bound as the cell of its period, as it is when it is no period."""
        if key is None:
            return None
        cell = self._cell(key)
        return key if cell is None else cell

    def slice_locs(self, start: Any = None, end: Any = None, step: Any = None) -> tuple[int, int]:
        """The rows from one period to another, both ends included."""
        return Index.slice_locs(self, self._bound(start), self._bound(end), step)

    def slice_indexer(self, start: Any = None, end: Any = None, step: Any = None) -> slice:
        """The rows from one period to another, as a slice."""
        return Index.slice_indexer(self, self._bound(start), self._bound(end), step)

    def isin(self, values: Any, level: Any = None) -> Any:
        """Whether each period is one of a set of periods or their text."""
        cells = [cell for value in _listed(values) if (cell := self._cell(value)) is not None]
        return Index.isin(self, cells, level) if cells else [False] * len(self)

    def __contains__(self, key: Any) -> bool:
        try:
            self.get_loc(key)
        except (KeyError, ValueError, TypeError):
            return False
        return True

    def __getitem__(self, key: Any) -> Any:
        found = Index.__getitem__(self, key)
        if isinstance(found, Index) and not isinstance(found, PeriodIndex):
            ordinals = _period.period_ordinals(found.tolist(), self._type)
            return PeriodIndex._of(ordinals, self._type, self.name)
        if isinstance(found, PeriodIndex):
            found._freq = self._type
        return NaT if found is None else found

    def _array(self) -> Any:
        import numpy

        found = numpy.empty(len(self), dtype=object)
        found[:] = self._periods()
        return found

    @property
    def values(self) -> Any:
        """The periods as a numpy array of objects, NaT for a gap."""
        return self._array()

    def to_numpy(self, dtype: Any = None, copy: bool = False, na_value: Any = None) -> Any:
        """The periods as a numpy array of objects, NaT for a gap."""
        found = self._array()
        return found if dtype is None else found.astype(dtype)

    @property
    def array(self) -> Any:
        """The periods as pandas' `PeriodArray`."""
        return self.to_series().array

    def __arrow_array__(self, type: Any = None) -> Any:
        """The periods as the Arrow array pandas exports, `pandas.period` over their ordinals."""
        import pyarrow as pa

        ordinals = [None if value is NaT else value.ordinal for value in self._periods()]
        storage = pa.array(ordinals, type=pa.int64())
        return pa.ExtensionArray.from_storage(_arrow_period(self.freqstr), storage)

    def astype(self, dtype: Any, copy: bool = True) -> Any:
        """The periods as text, or as periods of another frequency."""
        if str(dtype) in ("str", "string", "<class 'str'>"):
            found = [None if value is NaT else str(value) for value in self._periods()]
            return Index(found, name=self.name)
        if isinstance(dtype, PeriodDtype) or str(dtype).lower().startswith("period["):
            kind = _period.period_type(dtype).name
            return self._moved(lambda value: value.asfreq(kind[7:-1]), kind)
        return Index.astype(self, dtype, copy=copy)

    def __repr__(self) -> str:
        from ._config import get_option
        from ._pandas import _pprinted, _summary

        values = self._periods()
        width = get_option("display.width") or 80
        most = get_option("display.max_seq_items") or len(values)
        body = _summary(values, lambda value: f"'{value}'", True, "PeriodIndex", width, most)
        attrs = [f"dtype='{self._type}'"]
        if self.name is not None:
            attrs.append(f"name={_pprinted(self.name)}")
        if len(values) > most:
            attrs.append(f"length={len(values)}")
        return f"PeriodIndex({body}{', '.join(attrs)})"

    __str__ = __repr__


_ALIASES = {
    "M": ("MS", "BMS", "BME", "SME", "SMS", "CBME", "CBMS", "EOM"),
    "Q": ("QS", "BQS", "BQE"),
    "Y": ("YS", "BYS", "BYE"),
}
"""The offsets whose instants read as periods of another name, by that name,
which is pandas' `get_period_alias` for the ones an index can keep."""


def _period_alias(text: str) -> str:
    """The period frequency an index's own frequency reads as when it has to become one."""
    found = re.fullmatch(r"(\d*)([A-Za-z]+)(?:-(\w+))?", text)
    if found is None:
        return text
    count, prefix, suffix = found.groups()
    if prefix in ("ME", "QE", "YE"):
        return count + prefix[0] + (f"-{suffix}" if suffix else "")
    for name, prefixes in _ALIASES.items():
        if prefix in prefixes:
            return name
    return text


class PeriodProperties:
    """The `dt` accessor on a column of periods, which is `pandas.PeriodProperties`.

    `s.dt` answers one of these rather than a `DatetimeProperties` when the
    column holds periods. Every name reads the column as a `PeriodIndex` and
    hands the answer back as a column with the column's own labels and name.
    """

    __slots__ = ("_series",)
    """The series the accessor was reached from."""

    def __init__(self, data: Any) -> None:
        """Holds the series. Not a public entry point."""
        self._series = data

    def _index(self) -> PeriodIndex:
        """The column's periods as an index."""
        series = self._series
        return PeriodIndex(series.tolist(), dtype=series.dtype)

    def _wrapped(self, values: Any) -> Any:
        """Values as a column with the series' labels and name."""
        from ._frame import Series

        series = self._series
        return Series(values, index=series.index, name=series.name)

    def _read(self, name: str) -> Any:
        return self._wrapped(getattr(self._index(), name))

    year = property(lambda self: self._read("year"), doc="The year of each period.")
    month = property(lambda self: self._read("month"), doc="The month of each period.")
    day = property(lambda self: self._read("day"), doc="The day of each period.")
    hour = property(lambda self: self._read("hour"), doc="The hour of each period.")
    minute = property(lambda self: self._read("minute"), doc="The minute of each period.")
    second = property(lambda self: self._read("second"), doc="The second of each period.")
    quarter = property(lambda self: self._read("quarter"), doc="The quarter of each period.")
    qyear = property(lambda self: self._read("qyear"), doc="The fiscal year of each quarter.")
    week = weekofyear = property(
        lambda self: self._read("week"), doc="The week of the year of each period."
    )
    dayofweek = day_of_week = weekday = property(
        lambda self: self._read("dayofweek"), doc="The day of the week, Monday being 0."
    )
    dayofyear = day_of_year = property(
        lambda self: self._read("dayofyear"), doc="The day of the year of each period."
    )
    days_in_month = daysinmonth = property(
        lambda self: self._read("days_in_month"), doc="How many days each period's month has."
    )
    is_leap_year = property(
        lambda self: self._read("is_leap_year"), doc="Whether each period's year is a leap year."
    )
    start_time = property(
        lambda self: self._read("start_time"), doc="The first instant of each period."
    )
    end_time = property(lambda self: self._read("end_time"), doc="The last instant of each period.")

    @property
    def freq(self) -> Any:
        """The frequency, as the offset that steps by it."""
        return self._series.dtype.freq

    def to_timestamp(self, freq: Any = None, how: str = "start") -> Any:
        """Each period as an instant, at its start or its end."""
        return self._wrapped(self._index().to_timestamp(freq=freq, how=how))

    def asfreq(self, freq: Any = None, how: str = "E") -> Any:
        """Each period as the one of another frequency at its start or its end."""
        return self._wrapped(self._index().asfreq(freq, how=how))

    def strftime(self, date_format: str) -> Any:
        """Each period written by a format, a gap for NaT."""
        return self._wrapped(self._index().strftime(date_format))


def _offset_type() -> Any:
    """The offset class, read late since the offsets import this package."""
    from .offsets import BaseOffset

    return BaseOffset


_ARROW_TYPE: list[Any] = []


def _arrow_period(freq: str) -> Any:
    """pandas' Arrow type for periods, `pandas.period`, without registering it.

    pandas registers the name when its Arrow types are first imported, so its own
    type is used when pandas is loaded, which a reader in the same process then
    sees as pandas' type, and otherwise one of the same name and metadata is
    built, as `_interval._arrow_type` builds the one for intervals.
    """
    import json
    import sys

    import pyarrow as pa

    if not _ARROW_TYPE and "pandas" in sys.modules:
        from pandas.core.arrays.arrow.extension_types import ArrowPeriodType

        _ARROW_TYPE.append(ArrowPeriodType)
    if not _ARROW_TYPE:

        class ArrowPeriodType(pa.ExtensionType):
            def __init__(self, freq: str) -> None:
                self._freq = freq
                pa.ExtensionType.__init__(self, pa.int64(), "pandas.period")

            def __arrow_ext_serialize__(self) -> bytes:
                return json.dumps({"freq": self._freq}).encode()

            @classmethod
            def __arrow_ext_deserialize__(cls, storage: Any, serialized: bytes) -> Any:
                return cls(json.loads(serialized.decode())["freq"])

        _ARROW_TYPE.append(ArrowPeriodType)
    return _ARROW_TYPE[0](freq)


def period_range(
    start: Any = None,
    end: Any = None,
    periods: int | None = None,
    freq: Any = None,
    name: Any = None,
) -> PeriodIndex:
    """Periods of one frequency in a row, which is `pandas.period_range`.

    Two of `start`, `end` and `periods` say where the row runs. The frequency is
    a period end's own when it is not given, and a day otherwise.

    Raises:
        ValueError: Unless exactly two of `start`, `end` and `periods` are given.
    """
    if sum(part is not None for part in (start, end, periods)) != 2:
        raise InvalidArgumentError(
            "Of the three parameters: start, end, and periods, exactly two must be specified"
        )
    if freq is None:
        given = start if isinstance(start, Period) else end
        freq = given.freqstr if isinstance(given, Period) else "D"
    kind = _kind_of([], freq, None)
    text = kind[7:-1]
    step = _period.period_type(kind)._freq.n
    ends = [None if end is None else Period(end, freq=text) for end in (start, end)]
    first, last = (None if found is None else found.ordinal for found in ends)
    if first is None:
        first = last - (periods - 1) * step
    if last is None:
        last = first + (periods - 1) * step
    return PeriodIndex._of(list(range(first, last + 1, step)), kind, name)
