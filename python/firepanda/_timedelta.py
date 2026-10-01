"""`TimedeltaIndex` and `timedelta_range`, an index whose labels are spans.

The span twin of `DatetimeIndex`. Every lookup, set operation and slice bound is
the index's own, because a column of spans is whole numbers underneath, and
what is added is the fields a span is read by: `days`, `seconds`,
`microseconds`, `nanoseconds`, `components` and `total_seconds`, the same ones
the `dt` accessor answers on a column of spans.

`timedelta_range` counts along the whole numbers of a unit the way `date_range`
does and shares its reading of a frequency and its spacing of the points. The
rules below were measured against pandas 3.0.

- The unit is the finest one the ends carry. Text is `us`, or `ns` when it
  writes nanoseconds, a `Timedelta` keeps its own, a `timedelta` is `us` and a
  whole number is nanoseconds. A step finer than that unit raises the unit to
  the step's.
- `closed` keeps both ends, or drops the end it does not name.

An index holds the frequency `timedelta_range` made it at, or the one its
constructor was given, as `freq` and `freqstr`. `floor`, `ceil` and `round` need a
rounding kernel for spans, `resolution` is not written yet, and `to_pytimedelta`
needs a column of Python objects. None of them is spelled here, so none of them
resolves.
"""

from __future__ import annotations

import datetime
import re
from typing import Any

from ._date_range import _UNITS, _frequency, _points
from ._frame import DataFrame, Index, Series
from ._frequency import _conforming, _held, _hold, _offset_of
from ._held_freq import HeldFreq
from ._pandas import (
    _NO_ROUND_OK,
    NO_DEFAULT,
    _held_values,
    _label_of,
    _span_column,
    _span_frame,
    _span_parts,
    _span_type,
    to_timedelta,
)
from ._scalars import NaT, Timedelta
from .errors import InvalidArgumentError, translate

__all__ = ["TimedeltaIndex", "timedelta_range"]


def _is_span(dtype: Any) -> bool:
    """Whether a firepanda dtype name is one of the span types."""
    return str(dtype).startswith("timedelta64")


class TimedeltaIndex(HeldFreq, Index):
    """An index whose labels are spans, which is `pandas.TimedeltaIndex`."""

    __slots__ = ()

    def __init__(
        self,
        data: Any = None,
        freq: Any = NO_DEFAULT,
        dtype: Any = None,
        copy: bool | None = None,
        name: Any = None,
    ) -> None:
        """Builds an index of spans out of whatever the values are.

        The values are read by `to_timedelta`, so text, whole numbers counted in
        nanoseconds, `Timedelta` and `timedelta` objects all work, and an index
        or column that already holds spans keeps its unit.

        Args:
            data: The spans.
            freq: The frequency the index holds. Left out it comes along from
                another `TimedeltaIndex`, `infer` takes the one the labels keep,
                and anything else has to be one they keep.
            dtype: The type of the labels, `timedelta64[unit]`.
            copy: Read and ignored. The labels are always a copy.
            name: The level name, or the name the data carries.

        Raises:
            TypeError: If nothing was passed, in pandas' words.
            ValueError: For a type that is not one of spans, in pandas' words.
        """
        unit = None
        if dtype is not None:
            found = re.fullmatch(r"timedelta64\[(s|ms|us|ns)\]", str(dtype))
            if found is None:
                raise InvalidArgumentError(
                    f"dtype '{dtype}' is invalid, should be np.timedelta64 dtype"
                )
            unit = found.group(1)
        if data is None:
            raise TypeError(
                "TimedeltaIndex(...) must be called with a collection of some kind, None was passed"
            )
        label = _label_of(data) if name is None else str(name)
        if isinstance(data, Index) and _is_span(data.dtype):
            try:
                self._inner = data._inner.renamed(label)
            except Exception as error:
                raise translate(error) from None
            _hold(self, freq, data)
            if unit is not None and unit != self.unit:
                self._inner = self.as_unit(unit)._inner
            return
        values: Any = _held_values(data._inner) if isinstance(data, Index) else data
        if not isinstance(values, Series):
            # pandas counts spans read out of nothing in seconds.
            unit = unit or ("s" if hasattr(values, "__len__") and not len(values) else None)
            values = Series(values)
        if not _is_span(values.dtype):
            values = to_timedelta(values)
        if unit is not None:
            values = values.dt.as_unit(unit)
        try:
            self._inner = values._inner.to_index(label)
        except Exception as error:
            raise translate(error) from None
        _hold(self, freq, data)

    @property
    def freq(self) -> Any:
        """The frequency the index holds, as an offset, or None."""
        return getattr(self, "_freq", None)

    @freq.setter
    def freq(self, value: Any) -> None:
        held = _offset_of(value)
        if held is not None:
            _conforming(self, held)
        self._freq = held

    @property
    def freqstr(self) -> str | None:
        """The frequency the index holds, as text, or None."""
        held = self.freq
        return None if held is None else held.freqstr

    def _field(self, field: str) -> Index:
        """One field of every label, as pandas reads it off a span."""
        values = _span_column(_span_parts(_held_values(self._inner)), field)
        whole = "int64" if field == "days" else "int32"
        return Index(Series(values, name=self.name, dtype=_span_type(values, whole)))

    @property
    def days(self) -> Index:
        """The whole days of every label, floored."""
        return self._field("days")

    @property
    def seconds(self) -> Index:
        """The seconds of every label past its whole days."""
        return self._field("seconds")

    @property
    def microseconds(self) -> Index:
        """The microseconds of every label past its whole seconds."""
        return self._field("microseconds")

    @property
    def nanoseconds(self) -> Index:
        """The nanoseconds of every label past its whole microseconds."""
        return self._field("nanoseconds")

    @property
    def components(self) -> DataFrame:
        """Every label cut into days down to nanoseconds, one column each."""
        parts = _span_parts(_held_values(self._inner))
        return _span_frame(parts, list(range(len(parts))))

    def total_seconds(self) -> Index:
        """Every label as a number of seconds."""
        return Index(self._column().dt.total_seconds().rename(self.name))

    @property
    def unit(self) -> str:
        """The resolution the labels are stored in, one of s, ms, us and ns."""
        return str(self.dtype)[len("timedelta64[") : -1]

    @property
    def asi8(self) -> list[Any]:
        """Every label as the whole number it is stored as, in its own unit."""
        scale = _UNITS[self.unit]
        return [
            None if value is None else value.value // scale for value in _held_values(self._inner)
        ]

    @property
    def inferred_freq(self) -> str | None:
        """The frequency the labels keep, as `infer_freq` finds it, or None."""
        from ._frequency import infer_freq

        if len(self) < 3:
            return None
        try:
            return infer_freq(self)
        except ValueError:
            return None

    def shift(self, periods: int = 1, freq: Any = None) -> TimedeltaIndex:
        """Every span moved `periods` steps of `freq`, or of the step the index holds.

        The step has to be a fixed length, and the answer holds the frequency it
        held, as a span added to spans does.

        Raises:
            NullFrequencyError: Without a frequency given or held.
        """
        from .errors import NullFrequencyError
        from .offsets import _elapsed

        if freq is None:
            if self.freq is None:
                raise NullFrequencyError("Cannot shift with no freq")
            freq = self.freq
        return self + _elapsed(_offset_of(freq) * periods)

    def floor(self, freq: Any) -> TimedeltaIndex:
        """Every span moved down to a whole number of steps of a fixed frequency."""
        return TimedeltaIndex(self._column().dt.floor(freq), name=self.name)

    def ceil(self, freq: Any) -> TimedeltaIndex:
        """Every span moved up to a whole number of steps of a fixed frequency."""
        return TimedeltaIndex(self._column().dt.ceil(freq), name=self.name)

    def round(self, freq: Any) -> TimedeltaIndex:  # type: ignore[override]
        """Every span moved to the nearer whole step of a fixed frequency, even on a tie."""
        return TimedeltaIndex(self._column().dt.round(freq), name=self.name)

    def to_pytimedelta(self) -> list[Any]:
        """Every label as a Python timedelta, `NaT` for a missing one, as a list."""
        return [NaT if label is NaT else label.to_pytimedelta() for label in self.tolist()]

    def as_unit(self, unit: str, round_ok: bool = True) -> TimedeltaIndex:
        """The same labels stored at another resolution."""
        if not round_ok:
            raise TypeError(_NO_ROUND_OK)
        moved = TimedeltaIndex(self._column().dt.as_unit(unit), name=self.name)
        moved._freq = self.freq
        return moved

    def mean(self, *, skipna: bool = True, axis: Any = 0) -> Any:
        """The average span, or NaT when there is none."""
        return self._column().mean(skipna=skipna)

    def median(self, *, axis: Any = None, skipna: bool = True, **kwargs: Any) -> Any:
        """The middle span, or NaT when there is none."""
        return self._column().median(skipna=skipna)

    def std(
        self,
        *,
        axis: Any = None,
        dtype: Any = None,
        out: Any = None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Any:
        """The spread of the spans, or NaT when there are too few."""
        return self._column().std(ddof=ddof, skipna=skipna)

    def sum(
        self,
        *,
        axis: Any = None,
        dtype: Any = None,
        out: Any = None,
        keepdims: bool = False,
        initial: Any = None,
        skipna: bool = True,
        min_count: int = 0,
    ) -> Any:
        """The spans added up, a zero span when there are none."""
        return self._column().sum(skipna=skipna, min_count=min_count)

    def _column(self) -> Series:
        """The labels as a column of spans."""
        return (
            Series(_held_values(self._inner), name=self.name)
            .pipe(to_timedelta)
            .dt.as_unit(self.unit)
        )


_CLOSED = (None, "left", "right")


def _span_end(value: Any) -> tuple[int, str] | None:
    """One end of a range as nanoseconds and the unit it carries, or None."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise InvalidArgumentError(
            "Value must be Timedelta, string, integer, float, timedelta or convertible, not"
            f" {type(value).__name__}"
        )
    if isinstance(value, int):
        return value, "ns"
    found = Timedelta(value)
    unit = (
        "us"
        if isinstance(value, datetime.timedelta) and not isinstance(value, Timedelta)
        else found.unit
    )
    return found.value, unit


def _span_step(freq: Any) -> tuple[int | None, str]:
    """The step in nanoseconds and its offset, reading a compound step like `2D3h` too.

    `date_range`'s reading comes first, because it names the offset the way
    pandas does in its errors. Text it cannot read is tried as a span, since
    pandas reads a span's frequency the same way it reads a span.
    """
    if freq is None:
        return None, ""
    try:
        step, _, offset = _frequency(freq)
    except InvalidArgumentError:
        try:
            found = Timedelta(freq)
        except Exception:
            raise InvalidArgumentError(
                f"Invalid frequency: {freq}. Failed to parse with error message:"
                f" ValueError('Invalid frequency: {freq}.')"
            ) from None
        return found.value, f"<{found.value} * Nanos>"
    return step, offset


def timedelta_range(
    start: Any = None,
    end: Any = None,
    periods: Any = None,
    freq: Any = None,
    name: Any = None,
    closed: Any = None,
    *,
    unit: Any = None,
) -> TimedeltaIndex:
    """A fixed frequency `TimedeltaIndex`, which is `pandas.timedelta_range`.

    Args:
        start: The first span, as text, a `Timedelta`, a `timedelta` or nanoseconds.
        end: The last span, in the same forms.
        periods: How many spans.
        freq: The step, as text like `"6h"` or a timedelta. Left out it is a
            day, unless start, end and periods are all given, when the points
            are spaced evenly between the two ends.
        name: The level name.
        closed: Which end to keep when only one is wanted, or None for both.
        unit: The resolution of the answer.

    Raises:
        TypeError: When periods is not a whole number.
        ValueError: When not exactly three of start, end, periods and freq are
            given, or for an unknown closed, unit or frequency.
        NotImplementedError: For a calendar offset.
    """
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
    if closed not in _CLOSED:
        raise InvalidArgumentError("Closed has to be either 'left', 'right' or None")
    if unit is not None and unit not in _UNITS:
        raise InvalidArgumentError("'unit' must be one of 's', 'ms', 'us', 'ns'")
    step, offset = _span_step(freq)
    ends = [_span_end(start), _span_end(end)]
    if unit is None:
        units = [found[1] for found in ends if found is not None]
        unit = min(units, key=_UNITS.__getitem__)
        while step is not None and step % _UNITS[unit]:
            unit = {"s": "ms", "ms": "us", "us": "ns"}[unit]
    elif step is not None and step % _UNITS[unit]:
        raise InvalidArgumentError(
            f"freq={offset} is incompatible with unit={unit}. Use a lower freq or a higher"
            " unit instead."
        )
    scale = _UNITS[unit]
    first, last = (None if found is None else found[0] // scale for found in ends)
    counts = _points(first, last, periods, None if step is None else step // scale)
    if counts and closed == "right" and counts[0] == first:
        counts = counts[1:]
    if counts and closed == "left" and counts[-1] == last:
        counts = counts[:-1]
    # An empty column of whole numbers has no spans to read, so an empty range
    # is one span cut down to none, which keeps the unit.
    spans = to_timedelta(Series(counts or [0], dtype="int64"), unit=unit).dt.as_unit(unit)
    if not counts:
        spans = spans.iloc[:0]
    return _held(TimedeltaIndex(spans, name=name), freq)
