"""`resample` over a datetime index or column, on fixed steps of time.

A resample is a group by whose key is the bin each timestamp falls in, with one
difference that matters: every bin between the first and the last is in the
answer, empty or not. So the work here is three steps. The timestamps become
integer counts in their own unit and each count becomes a bin number, the
values are grouped by that number with the group by firepanda already has, and
the answer is reindexed onto every bin number from the first to the last, with
the value pandas gives an empty bin, before the bin numbers turn back into
timestamps for the labels.

The steps are the ones pandas calls ticks, hours down to nanoseconds, and days,
which pandas 3 does not count as a tick but which on a timestamp with no zone
is 24 hours measured from midnight. Weeks, months, quarters, years and business
days are calendar offsets whose bins are not all the same length. Their edges
are listed one by one, the way pandas lays them: from the first day rolled back
onto the offset, or a step before it when the bins close on the right, to a
step past the last day. The offsets that land on the end of a period, a month,
quarter or year end or a week, close on the right and are named by their right
end, and a bin closed on the right runs to the last moment of its end day.

The rules below were measured against pandas 3.0.

- The bins are measured from `origin`. The default, `start_day`, is midnight
  of the first day, `start` is the first timestamp, `epoch` is 1970-01-01 and
  a timestamp is itself. `end` and `end_day` count whole steps back from the
  last timestamp or the midnight after it, and make the bins close on and be
  named by their right end unless told otherwise. `offset` moves the origin
  by a span.
- A rule in days always measures from midnight of the first day, and pandas
  warns that `origin` and `offset` do nothing to it. Its bins run from
  midnight of the first day, or a step before it when they close on the
  right, to a step past midnight of the last day, so an end bin can be empty.
- With `closed="left"`, the default for every step here, a bin holds the
  timestamps from its start up to but not including its end, and with
  `closed="right"` from after its start up to and including its end.
  `label` picks which end names the bin.
- An empty bin sums to 0, multiplies to 1 and counts 0, and is NaN for every
  other reduction, so an integer column that meets one widens to float64.
- A missing timestamp is in no bin.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from .errors import InvalidArgumentError

if TYPE_CHECKING:
    from ._frame import DataFrame, Index, Series

_NANOS = {
    "D": 86_400 * 10**9,
    "h": 3_600 * 10**9,
    "min": 60 * 10**9,
    "s": 10**9,
    "ms": 10**6,
    "us": 10**3,
    "ns": 1,
}
"""The steps a rule can name, in nanoseconds."""

_CALENDAR = frozenset(
    {
        "W", "ME", "MS", "SME", "SMS", "BME", "BMS", "CBME", "CBMS", "QE", "QS", "BQE",
        "BQS", "YE", "YS", "BYE", "BYS", "B", "C", "BH", "CBH", "WOM", "LWOM",
    }
)  # fmt: skip
"""The calendar offsets pandas reads, whose bins are not all one length."""

_EMPTY = {"sum": 0, "prod": 1, "count": 0, "size": 0, "nunique": 0}
"""What pandas answers for an empty bin, where it is not NaN."""

_BIN = "__firepanda_bin__"
_VALUE = "__firepanda_value__"
_LABEL = "__firepanda_label__"


def _step(rule: Any) -> tuple[int, bool]:
    """The length of a rule in nanoseconds, and whether it is counted in days.

    Raises:
        InvalidArgumentError: For a rule pandas cannot read, with its message.
        ZeroDivisionError: For a rule of no length, as pandas raises.
        TypeError: For a rule that is neither text nor an offset, as pandas raises.
        NotImplementedError: For a calendar offset, which `_calendar` reads.
    """
    from .offsets import BaseOffset, Day, Tick

    if isinstance(rule, Tick | Day):
        # A tick or a day is one length, which its frequency text says.
        rule = rule.freqstr
    elif isinstance(rule, BaseOffset):
        raise NotImplementedError(
            f"resample: the offset {rule!r} is a calendar offset, whose bins are not all one"
            " length, and firepanda resamples on fixed steps of days and less for now"
        )
    elif not isinstance(rule, str):
        raise TypeError(
            f"Argument 'freq' has incorrect type (expected str, got {type(rule).__name__})"
        )
    found = re.fullmatch(r"\s*(\d+(?:\.\d*)?)?\s*([A-Za-z]+)(-[A-Za-z]+)?\s*", rule)
    if found is None:
        raise InvalidArgumentError(f"Invalid frequency: {rule}")
    count, unit, anchor = found.groups()
    if unit in _CALENDAR:
        raise NotImplementedError(
            f"resample: the rule {rule!r} is a calendar offset, whose bins are not all one"
            " length, and firepanda resamples on fixed steps of days and less for now"
        )
    if unit not in _NANOS or anchor is not None:
        raise InvalidArgumentError(f"Invalid frequency: {rule}")
    length = float(count) * _NANOS[unit] if count else _NANOS[unit]
    if length == 0:
        raise ZeroDivisionError("division by zero")
    if length != int(length):
        raise InvalidArgumentError(f"Invalid frequency: {rule}")
    return int(length), unit == "D"


def _calendar(rule: Any) -> Any:
    """The calendar offset a rule names, or None for a fixed step or a rule pandas refuses."""
    from . import offsets

    if isinstance(rule, offsets.Tick | offsets.Day):
        return None
    if isinstance(rule, offsets.BaseOffset):
        return rule
    if not isinstance(rule, str):
        return None
    found = re.fullmatch(r"\s*\d*\s*([A-Za-z]+)(-[A-Za-z]+)?\s*", rule)
    if found is None or found.group(1) not in _CALENDAR:
        return None
    try:
        return offsets._parsed(rule.strip())
    except ValueError:
        raise InvalidArgumentError(f"Invalid frequency: {rule}") from None


def _ends_period(offset: Any) -> bool:
    """Whether an offset lands on the end of a period, which closes and names bins on the right."""
    return offset.rule_code.split("-")[0] in ("ME", "YE", "QE", "BME", "BYE", "BQE", "W")


def _calendar_bins(offset: Any, counts: Any, unit: str, right: bool) -> tuple[list[int], Any]:
    """The labels of a calendar offset's bins and each timestamp's bin number.

    The labels are one more than the bins, each bin lying between two of them,
    and are counts in the timestamps' own unit, as the timestamps are.
    """
    from bisect import bisect_left, bisect_right

    from ._date_range import date_range
    from ._frame import Series
    from ._scalars import Timestamp

    per = _NANOS[unit]
    values = counts.tolist()
    first = Timestamp(min(values) * per).normalize()
    last = Timestamp(max(values) * per).normalize()
    first = first - offset if right else offset.rollback(first)
    marks = [int(stamp.value) // per for stamp in date_range(first, last + offset, freq=offset)]
    edges = marks
    if _ends_period(offset):
        if right:
            # A bin closed on its end day holds every moment of that day.
            edges = [mark + _NANOS["D"] // per - 1 for mark in marks]
        if len(edges) > 1 and edges[-2] > max(values):
            marks, edges = marks[:-1], edges[:-1]
    place = bisect_left if right else bisect_right
    return marks, Series([place(edges, value) - 1 for value in values], dtype="int64")


def _numeric(dtype: str) -> bool:
    """Whether pandas counts a type as a number under `numeric_only`."""
    return re.fullmatch(r"u?int\d+|float\d+|bool", dtype) is not None


def _unit(dtype: str) -> str:
    """The unit of a datetime type with no zone, or empty for any other type."""
    found = re.fullmatch(r"datetime64\[(s|ms|us|ns)\]", dtype)
    return found.group(1) if found else ""


def _moment(origin: Any) -> Any:
    """`origin` as one of pandas' words or a timestamp with no zone.

    Raises:
        InvalidArgumentError: For text that is neither, or a timestamp with a zone, in
            pandas' words.
    """
    if isinstance(origin, str) and origin in ("epoch", "start", "start_day", "end", "end_day"):
        return origin
    from ._scalars import Timestamp

    try:
        moment = Timestamp(origin)
    except (ValueError, TypeError):
        raise InvalidArgumentError(
            "'origin' should be equal to 'epoch', 'start', 'start_day', 'end', 'end_day' or"
            f" should be a Timestamp convertible type. Got '{origin}' instead."
        ) from None
    if moment.tz is not None:
        raise InvalidArgumentError("The origin must have the same timezone as the index.")
    return moment


def _shift(offset: Any) -> int | None:
    """`offset` in nanoseconds, or None when there is none.

    Raises:
        InvalidArgumentError: For a value that is not a span, in pandas' words.
    """
    if offset is None:
        return None
    from ._scalars import Timedelta

    try:
        return int(Timedelta(offset).value)
    except (ValueError, TypeError):
        raise InvalidArgumentError(
            f"'offset' should be a Timedelta convertible type. Got '{offset}' instead."
        ) from None


def _counted(nanoseconds: int | None, per: int) -> int:
    """A span in nanoseconds counted in the timestamps' own unit.

    Raises:
        NotImplementedError: For a span finer than that unit.
    """
    if not nanoseconds:
        return 0
    if nanoseconds % per:
        raise NotImplementedError(
            "resample: an origin or offset finer than the unit the timestamps are counted in"
            " is not supported, because pandas answers it on a finer unit"
        )
    return nanoseconds // per


def _unmoved(origin: Any, offset: Any) -> None:
    """Warns, as pandas does, that a rule in days is not moved by `origin` or `offset`."""
    import warnings

    for name, given in (("offset", offset is not None), ("origin", origin != "start_day")):
        if given:
            warnings.warn(
                f"The '{name}' keyword does not take effect when resampling with a 'freq' that"
                " is not Tick-like (h, m, s, ms, us, ns)",
                RuntimeWarning,
                stacklevel=4,
            )


def _anchor(origin: Any, counts: Any, step: int, day: int, per: int, right: bool) -> int:
    """The count the bins of a fixed step are measured from, before any offset.

    `end` counts whole steps back from the last timestamp and `end_day` from
    the midnight after it, as far as the first timestamp, with one step more
    when the bins close on the left so that the first one still holds it.
    """
    first, last = int(counts.min()), int(counts.max())
    if origin == "epoch":
        return 0
    if origin == "start":
        return first
    if origin == "start_day":
        return first // day * day
    if origin in ("end", "end_day"):
        end = last if origin == "end" else -(-last // day) * day
        return end - ((end - first) // step + (not right)) * step
    return _counted(int(origin.value), per)


def _label_of(how: Any) -> Any:
    """The column label pandas gives a reduction: its name, or a function's name."""
    return how if isinstance(how, str) else getattr(how, "__name__", "<lambda>")


class Resampler:
    """The bins of a resample, waiting for a reduction.

    Built by `DataFrame.resample` and `Series.resample`. Every reduction is the
    group by's own over the bin numbers, reindexed onto every bin.
    """

    def __init__(
        self,
        obj: DataFrame | Series,
        rule: Any,
        closed: Any = None,
        label: Any = None,
        on: Any = None,
        level: Any = None,
        origin: Any = "start_day",
        offset: Any = None,
        *,
        _bins: tuple[Any, ...] | None = None,
    ) -> None:
        self._obj = obj
        self._picked = False
        if _bins is not None:
            (self._times, self._codes, self._first, self._count, self._origin, self._step,
             self._time_unit, self._right_label, self._name, self._dropped,
             self._right_closed, self._chosen, self._rule, self._marks,
             self._zone, self._wall) = _bins  # fmt: skip
            return
        from ._frame import DataFrame

        if closed not in (None, "left", "right"):
            raise InvalidArgumentError(f"Unsupported value {closed} for `closed`")
        if label not in (None, "left", "right"):
            raise InvalidArgumentError(f"Unsupported value {label} for `label`")
        origin = _moment(origin)
        shift = _shift(offset)
        calendar = _calendar(rule)
        if calendar is not None and _ends_period(calendar):
            closed = closed or "right"
            label = label or "right"
        if origin in ("end", "end_day"):
            # Counting back from the last timestamp, a bin is named by and holds its end.
            closed = closed or "right"
            label = label or "right"
        if on is not None:
            if not isinstance(obj, DataFrame):
                raise KeyError(on)
            times = obj[on]
            self._obj = obj.drop(columns=[on])
            self._name = on
            kind = "Index"
        elif level is not None and obj.index.nlevels > 1:
            times = obj.index.get_level_values(level).to_series()
            self._name = times.name
            kind = "Index"
        else:
            if level not in (None, 0, obj.index.name):
                raise InvalidArgumentError(f"The level {level} is not valid")
            times = obj.index.to_series()
            self._name = obj.index.name
            kind = "RangeIndex" if obj.index._inner.is_range() else "Index"
        dtype = str(times.dtype)
        unit = _unit(dtype)
        zoned = re.fullmatch(r"datetime64\[(s|ms|us|ns), .+\]", dtype)
        self._zone = times.dt.tz if zoned else None
        if zoned:
            unit = zoned.group(1)
        if not unit:
            raise TypeError(
                "Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex, but got an"
                f" instance of '{kind}'"
            )
        length, days = (0, False) if calendar is not None else _step(rule)
        per = _NANOS[unit]
        if length % per:
            raise NotImplementedError(
                f"resample: a step of {rule!r} is finer than the {unit} the timestamps are"
                " counted in, which pandas answers on a finer unit"
            )
        times = times.reset_index(drop=True)
        missing = times.isna()
        self._dropped = bool(missing.any())
        if self._dropped:
            # A missing timestamp is in no bin, so its row is left out of every answer.
            kept = [at for at, gone in enumerate(missing.tolist()) if not gone]
            self._obj = self._obj.iloc[kept]
            times = times.iloc[kept].reset_index(drop=True)
        step = length // per
        # Days and calendar steps count a zone's own clock, whose days are not
        # all 24 hours long; a fixed step counts the instants themselves.
        self._wall = bool(days or calendar is not None)
        self._time_unit = unit
        if self._zone is None:
            counts = times.astype("int64")
        elif self._wall:
            counts = times.dt.tz_localize(None).astype("int64")
        else:
            counts = times.dt.tz_convert("UTC").dt.tz_localize(None).astype("int64")
        right = closed == "right"
        day = _NANOS["D"] // per
        self._marks: list[int] | None = None
        if days or calendar is not None:
            _unmoved(origin, offset)
        if calendar is not None and len(counts):
            self._marks, codes = _calendar_bins(calendar, counts, unit, right)
            start, step = 0, 1
        elif len(counts) == 0:
            start = 0
        elif days:
            start = int(counts.min()) // day * day
        else:
            start = _anchor(origin, counts, step, day, per, right) + _counted(shift, per)
            if self._zone is not None and origin == "start_day":
                # The first day starts at midnight on the zone's clock.
                clock = times.dt.tz_localize(None).astype("int64")
                start += self._midnight(int(clock.min()) // day * day) - counts.min() // day * day
            elif self._zone is not None and origin == "epoch":
                start += self._midnight(0)
        if self._marks is None:
            # Closed on the left a bin holds [edge, edge + step), and closed on the
            # right (edge, edge + step], which is the ceiling less one.
            codes = -((start - counts) // step) - 1 if right else (counts - start) // step
        self._times = times
        self._codes = codes
        self._first = int(codes.min()) if len(codes) else 0
        self._count = int(codes.max()) - self._first + 1 if len(codes) else 0
        if self._marks is not None:
            self._count = len(self._marks) - 1
        elif days and len(codes):
            # pandas 3 lays day bins from one step before midnight of the first
            # day when they close on the right, and up to midnight of the last
            # day plus a step, so either end can hold a bin with nothing in it.
            self._first = -1 if right else 0
            self._count = (int(counts.max()) // day * day - start) // step - self._first + 1
        self._origin = start
        self._step = step
        self._time_unit = unit
        self._right_label = label == "right"
        self._right_closed = right
        self._chosen = on is not None or (level is not None and obj.index.nlevels > 1)
        self._rule = rule

    def _state(self) -> tuple[Any, ...]:
        """The bins, handed to a resampler over a part of the same rows."""
        return (self._times, self._codes, self._first, self._count, self._origin, self._step,
                self._time_unit, self._right_label, self._name, self._dropped,
                self._right_closed, self._chosen, self._rule, self._marks,
                self._zone, self._wall)  # fmt: skip

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def __getitem__(self, key: Any) -> Resampler:
        """The same bins over one column, or over a list of columns."""
        from ._frame import Series

        if isinstance(self._obj, Series):
            raise KeyError(key)
        picked = Resampler(self._obj[key], None, _bins=self._state())
        picked._picked = True
        return picked

    def __getattr__(self, name: str) -> Resampler:
        """A column by name, as `resampler.column` reads it in pandas."""
        if name.startswith("_"):
            raise AttributeError(name)
        from ._frame import DataFrame

        if isinstance(self._obj, DataFrame) and name in list(self._obj.columns):
            return self[name]
        raise AttributeError(f"'DatetimeIndexResampler' object has no attribute {name!r}")

    @property
    def ndim(self) -> int:
        """1 over a series and 2 over a frame."""
        return self._obj.ndim

    @property
    def ngroups(self) -> int:
        """How many bins there are, the empty ones included."""
        return self._count

    @property
    def obj(self) -> DataFrame | Series:
        """What is being resampled."""
        return self._obj

    @property
    def ax(self) -> Index:
        """The timestamps being binned."""
        from ._frame import Index

        return Index(self._times.reset_index(drop=True)).rename(self._name)

    @property
    def binner(self) -> Index:
        """The edges of the bins, one more than there are bins."""
        from ._frame import Index

        edges = Index(self._stamps(range(self._first, self._first + self._count + 1), 0))
        edges = edges.rename(self._name)
        edges.freq = self._rule
        return edges

    @property
    def groups(self) -> dict[Any, int]:
        """Every bin's label and the position one past its last row, as pandas gives it."""
        sizes = self.size().tolist()
        labels = self._stamps(range(self._first, self._first + self._count)).tolist()
        answer = {}
        total = 0
        for stamp, size in zip(labels, sizes, strict=True):
            total += int(size)
            answer[stamp] = total
        return answer

    @property
    def indices(self) -> dict[Any, list[int]]:
        """The positions of the rows in every bin that has any."""
        answer: dict[Any, list[int]] = {}
        codes = self._codes.tolist()
        labels = self._stamps(range(self._first, self._first + self._count)).tolist()
        for position, code in enumerate(codes):
            if code is None or code != code:
                continue
            answer.setdefault(labels[int(code) - self._first], []).append(position)
        return answer

    def get_group(self, name: Any) -> DataFrame | Series:
        """The rows of the bin labelled `name`.

        Raises:
            KeyError: For a label that names no bin with rows in it.
        """
        from ._scalars import Timestamp

        wanted = Timestamp(name)
        for stamp, positions in self.indices.items():
            if stamp == wanted:
                return self._obj.iloc[positions]
        raise KeyError(name)

    # ------------------------------------------------------------------
    # The machinery
    # ------------------------------------------------------------------

    def _stamps(self, codes: Any, shift: int | None = None) -> Series:
        """The timestamps that label some bin numbers."""
        from ._frame import Series
        from ._pandas import to_datetime

        extra = (1 if self._right_label else 0) if shift is None else shift
        if self._marks is not None:
            counts = [self._marks[code - self._first + extra] for code in codes]
        else:
            counts = [self._origin + (code + extra) * self._step for code in codes]
        stamps = to_datetime(Series(counts, dtype="int64"), unit=self._time_unit)
        if self._zone is None:
            return stamps
        if self._wall:
            return stamps.dt.tz_localize(self._zone)
        return stamps.dt.tz_localize("UTC").dt.tz_convert(self._zone)

    def _midnight(self, clock: int) -> int:
        """The instant, counted from the epoch, of a midnight on the zone's clock."""
        from ._frame import Series
        from ._pandas import to_datetime

        stamp = to_datetime(Series([clock], dtype="int64"), unit=self._time_unit)
        stamp = stamp.dt.tz_localize(self._zone).dt.tz_convert("UTC").dt.tz_localize(None)
        return int(stamp.astype("int64").iloc[0])

    def _edges(self, shift: int | None = None) -> Index:
        """The labels of every bin, holding the rule as their frequency, as pandas' do."""
        from ._frame import Index

        codes = range(self._first, self._first + self._count)
        edges = Index(self._stamps(codes, shift)).rename(self._name)
        edges.freq = self._rule
        return edges

    def _frame(self) -> DataFrame:
        """The values on plain positions with their bin numbers as the last column."""
        from ._frame import DataFrame, Series

        plain = self._obj.reset_index(drop=True)
        if isinstance(self._obj, Series):
            plain = DataFrame({_VALUE: plain})
        return plain.assign(**{_BIN: self._codes})

    def _labelled(self, out: DataFrame, fill: dict[str, Any] | None) -> DataFrame | Series:
        """A reduction by bin number put on every bin and labelled with its timestamp."""
        from ._attrs import hold_freq
        from ._frame import Series
        from ._frequency import _offset_of

        types = {name: str(out[name].dtype) for name in out.columns}
        if len(out) < self._count:
            out = out.reindex(list(range(self._first, self._first + self._count)))
            for name in out.columns:
                if fill is not None and name in fill:
                    out = out.assign(**{name: out[name].fillna(fill[name])})
                    if types[name] != str(out[name].dtype) and not bool(out[name].isna().any()):
                        out = out.assign(**{name: out[name].astype(types[name])})
        # Setting a column as the index keeps the dates' type on an empty answer too.
        labels = self._stamps(range(self._first, self._first + self._count))
        out = out.reset_index(drop=True).assign(**{_LABEL: labels})
        out = out.set_index(_LABEL).rename_axis(self._name)
        if isinstance(self._obj, Series):
            out = out[_VALUE].rename(self._obj.name)
        # The bins step by the rule, which pandas' answer carries on its index.
        hold_freq(out, _offset_of(self._rule))
        return out

    def _reduce(self, how: str, *args: Any, **kwargs: Any) -> DataFrame | Series:
        """One of the group by's reductions over the bins."""
        from ._frame import DataFrame, DataFrameGroupBy

        work = self._frame()
        if kwargs.get("numeric_only"):
            kwargs["numeric_only"] = False
            work = work[
                [name for name in work.columns if name == _BIN or _numeric(str(work[name].dtype))]
            ]
        grouped = DataFrameGroupBy(work, [_BIN], True, True, True)
        least = kwargs.pop("min_count", 0) if how in ("sum", "prod") else 0
        out = getattr(grouped, how)(*args, **kwargs)
        if not isinstance(out, DataFrame):
            out = DataFrame({_VALUE: out})
        if least > 0:
            # A bin with fewer values than `min_count` answers NaN, which the
            # group by's own sum does not do, so the counts are taken beside it.
            counts = grouped.count()
            for name in out.columns:
                short = counts[name] < least
                if bool(short.any()):
                    widened = out[name].astype("float64").mask(short, float("nan"))
                    out = out.assign(**{name: widened})
        if how not in ("quantile", "count", "nunique"):
            # pandas keeps a float32 column float32 through every reduction that
            # answers a value of the column's own kind.
            narrow = [
                name
                for name in out.columns
                if str(work[name].dtype) == "float32" and str(out[name].dtype) == "float64"
            ]
            if narrow:
                out = out.astype(dict.fromkeys(narrow, "float32"))
        fill = None
        empty = _EMPTY.get(how)
        if empty is not None and least <= 0:
            fill = {
                name: "" if str(out[name].dtype) in ("string", "str") else empty
                for name in out.columns
            }
        return self._labelled(out, fill)

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------

    def sum(self, numeric_only: bool = False, min_count: int = 0) -> DataFrame | Series:
        """The sum of every bin, 0 for an empty one."""
        return self._reduce("sum", numeric_only=numeric_only, min_count=min_count)

    def prod(self, numeric_only: bool = False, min_count: int = 0) -> DataFrame | Series:
        """The product of every bin, 1 for an empty one."""
        return self._reduce("prod", numeric_only=numeric_only, min_count=min_count)

    def mean(self, numeric_only: bool = False) -> DataFrame | Series:
        """The mean of every bin."""
        return self._reduce("mean", numeric_only=numeric_only)

    def median(self, numeric_only: bool = False) -> DataFrame | Series:
        """The median of every bin."""
        return self._reduce("median", numeric_only=numeric_only)

    def min(self, numeric_only: bool = False, min_count: int = 0) -> DataFrame | Series:
        """The smallest value of every bin."""
        return self._reduce("min", numeric_only=numeric_only, min_count=min_count or -1)

    def max(self, numeric_only: bool = False, min_count: int = 0) -> DataFrame | Series:
        """The largest value of every bin."""
        return self._reduce("max", numeric_only=numeric_only, min_count=min_count or -1)

    def first(
        self, numeric_only: bool = False, min_count: int = 0, skipna: bool = True
    ) -> DataFrame | Series:
        """The first value of every bin."""
        return self._reduce(
            "first", numeric_only=numeric_only, min_count=min_count or -1, skipna=skipna
        )

    def last(
        self, numeric_only: bool = False, min_count: int = 0, skipna: bool = True
    ) -> DataFrame | Series:
        """The last value of every bin."""
        return self._reduce(
            "last", numeric_only=numeric_only, min_count=min_count or -1, skipna=skipna
        )

    def std(self, ddof: int = 1, numeric_only: bool = False) -> DataFrame | Series:
        """The standard deviation of every bin."""
        return self._reduce("std", ddof=ddof, numeric_only=numeric_only)

    def var(self, ddof: int = 1, numeric_only: bool = False) -> DataFrame | Series:
        """The variance of every bin."""
        return self._reduce("var", ddof=ddof, numeric_only=numeric_only)

    def sem(self, ddof: int = 1, numeric_only: bool = False) -> DataFrame | Series:
        """The standard error of the mean of every bin."""
        return self._reduce("sem", ddof=ddof, numeric_only=numeric_only)

    def quantile(self, q: Any = 0.5, **kwargs: Any) -> DataFrame | Series:
        """One quantile of every bin."""
        return self._reduce("quantile", q, **kwargs)

    def count(self) -> DataFrame | Series:
        """How many values every bin holds, missing ones left out."""
        return self._reduce("count")

    def nunique(self) -> DataFrame | Series:
        """How many distinct values every bin holds."""
        return self._reduce("nunique")

    def size(self) -> Series:
        """How many rows every bin holds."""
        from ._frame import DataFrame, DataFrameGroupBy

        grouped = DataFrameGroupBy(self._frame()[[_BIN]], [_BIN], True, True, True)
        out = DataFrame({_VALUE: grouped.size()})
        answer = self._labelled(out, {_VALUE: 0})
        if not isinstance(answer, DataFrame):
            return answer
        return answer[_VALUE].rename(None)

    def ohlc(self) -> DataFrame:
        """The first, largest, smallest and last value of every bin.

        Over a frame each column gets the four under its own name, as two
        levels of column labels.
        """
        from ._frame import DataFrame, Series

        if not isinstance(self._obj, Series):
            parts = {
                (name, how): part for name in self._obj for how, part in self[name].ohlc().items()
            }
            first = next(iter(parts.values()), None)
            if first is None:
                return self.first()
            out = DataFrame({name: part.reset_index(drop=True) for name, part in parts.items()})
            return out.set_axis(first.index)
        parts = {"open": self.first(), "high": self.max(), "low": self.min(), "close": self.last()}
        out = DataFrame({name: part.reset_index(drop=True) for name, part in parts.items()})
        return out.set_axis(parts["open"].index)

    # ------------------------------------------------------------------
    # Several at once, and the rest
    # ------------------------------------------------------------------

    def aggregate(self, func: Any = None, *args: Any, **kwargs: Any) -> Any:
        """A reduction by name or a function, a list of them, or either for each column.

        A list names a column for each reduction, under the column it reduces
        when there is one, as pandas' two levels of column labels do. A function
        is run on every bin, the empty ones too, and on each column of a frame.

        Raises:
            InvalidArgumentError: For something that is neither a name, a function,
                a list of them nor a mapping of columns to them.
        """
        from ._frame import DataFrame

        if isinstance(func, str):
            return self._by_name(func)(*args, **kwargs)
        if callable(func):
            return self._called(func, args, kwargs)
        named = isinstance(func, list) and all(
            isinstance(how, str) or callable(how) for how in func
        )
        if named and not isinstance(self._obj, DataFrame):
            parts = {_label_of(how): self._one(how) for how in func}
        elif named:
            parts = {
                (name, _label_of(how)): self[name]._one(how) for name in self._obj for how in func
            }
        elif isinstance(func, dict) and isinstance(self._obj, DataFrame):
            if all(not isinstance(how, list) for how in func.values()):
                parts = {name: self[name].aggregate(how) for name, how in func.items()}
            else:
                parts = {
                    (name, _label_of(how)): self[name]._one(how)
                    for name, hows in func.items()
                    for how in (hows if isinstance(hows, list) else [hows])
                }
        else:
            raise InvalidArgumentError(
                "aggregate takes a reduction by name, a function, a list of them, or a"
                " mapping of columns to them"
            )
        first = next(iter(parts.values()))
        out = DataFrame({name: part.reset_index(drop=True) for name, part in parts.items()})
        return out.set_axis(first.index)

    agg = aggregate

    def _by_name(self, how: str) -> Any:
        """A reduction of this resampler by the name pandas gives it."""
        if how in ("agg", "aggregate", "apply", "transform", "pipe") or how.startswith("_"):
            raise AttributeError(f"'{how}' is not a valid function for 'Resampler' object")
        method = getattr(type(self), how, None)
        if not callable(method):
            raise AttributeError(f"'{how}' is not a valid function for 'Resampler' object")
        return getattr(self, how)

    def _one(self, how: Any) -> Any:
        """One reduction, by name or a function, over every bin."""
        return self._by_name(how)() if isinstance(how, str) else self._called(how, (), {})

    def _rows(self) -> list[list[int]]:
        """The positions of the rows in every bin, the empty ones too, in order."""
        rows: list[list[int]] = [[] for _ in range(self._count)]
        for position, code in enumerate(self._codes.tolist()):
            if code is not None and code == code:
                rows[int(code) - self._first].append(position)
        return rows

    def _called(self, func: Any, args: Any, kwargs: Any) -> Any:
        """A function run on every bin, as pandas' `aggregate` and `apply` run it.

        Over a frame it is run on each column first, and on the whole of every
        bin when a column refuses it. Values come back one a bin, labelled by
        the bins; pieces with labels of their own are put end to end.
        """
        from ._frame import DataFrame, Series

        rows = self._rows()
        if isinstance(self._obj, Series):
            answers = [func(self._obj.iloc[at], *args, **kwargs) for at in rows]
            return self._gathered({_VALUE: answers})
        try:
            columns = {
                name: [func(self._obj[name].iloc[at], *args, **kwargs) for at in rows]
                for name in self._obj
            }
        except Exception:
            columns = {}
        if columns:
            return self._gathered(columns)
        answers = [func(self._obj.iloc[at], *args, **kwargs) for at in rows]
        if answers and all(isinstance(answer, Series) for answer in answers):
            labels = answers[0].index.tolist()
            table = {label: [answer[label] for answer in answers] for label in labels}
            return self._labelled(DataFrame(table), None)
        out = self._gathered({_VALUE: answers})
        return out[_VALUE].rename(None) if isinstance(out, DataFrame) else out

    def _gathered(self, columns: dict[Any, list[Any]]) -> Any:
        """Answers of a function a bin, labelled by the bins, or put end to end."""
        from ._frame import DataFrame, Series
        from ._pandas import concat

        if any(
            isinstance(answer, Series | DataFrame) for part in columns.values() for answer in part
        ):
            pieces = [answer for part in columns.values() for answer in part]
            return concat(pieces)
        return self._labelled(
            DataFrame({name: Series(part) for name, part in columns.items()}), None
        )

    def apply(self, func: Any = None, *args: Any, **kwargs: Any) -> Any:
        """A reduction by name, or a function run on every bin."""
        return self.aggregate(func, *args, **kwargs)

    def _transformed(self, func: Any, args: Any, kwargs: Any) -> Series:
        """A function run on every bin of a series, its answers put back on the rows."""
        from ._frame import DataFrame, Series

        held = self._obj
        values: list[Any] = [None] * len(held)
        for at in self._rows():
            if not at:
                continue
            answer = func(held.iloc[at], *args, **kwargs)
            if isinstance(answer, Series | DataFrame):
                answer = answer.tolist()
                if len(answer) != len(at):
                    raise InvalidArgumentError(
                        "transform must return a scalar value for each group"
                    )
            else:
                answer = [answer] * len(at)
            for position, value in zip(at, answer, strict=True):
                values[position] = value
        return Series(values, index=held.index, name=held.name)

    def transform(self, arg: Any, *args: Any, **kwargs: Any) -> DataFrame | Series:
        """A reduction by name or a function put back on every row of its bin.

        The answer is on the rows' own labels. A function is run on every bin
        with rows in it, and on each column of a frame.
        """
        from ._frame import DataFrame, DataFrameGroupBy, Series

        if self._dropped:
            raise NotImplementedError(
                "transform over rows with a missing timestamp answers NaN for those rows,"
                " which is not written yet"
            )
        if not isinstance(arg, str):
            if isinstance(self._obj, Series):
                return self._transformed(arg, args, kwargs)
            parts = {name: self[name]._transformed(arg, args, kwargs) for name in self._obj}
            return DataFrame(parts, index=self._obj.index)
        grouped = DataFrameGroupBy(self._frame(), [_BIN], True, True, True)
        return self._on_rows(grouped.transform(arg, *args, **kwargs))

    def _on_rows(self, out: DataFrame) -> DataFrame | Series:
        """An answer a row put back on the rows' own labels."""
        from ._frame import Series

        out = out.set_axis(self._obj.index)
        if isinstance(self._obj, Series):
            return out[_VALUE].rename(self._obj.name)
        return out

    def pipe(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """`func(self, *args, **kwargs)`, or with the resampler under a keyword for a pair."""
        if isinstance(func, tuple):
            func, keyword = func
            kwargs[keyword] = self
            return func(*args, **kwargs)
        return func(self, *args, **kwargs)

    def _upsampled(
        self, name: str, method: str | None, limit: int | None = None, fill_value: Any = None
    ) -> DataFrame | Series:
        """The rows reindexed onto the edges of the bins, as pandas upsamples.

        The edges are the left ones for bins closed on the left and the right
        ones for bins closed on the right, whichever side labels the bins.

        Raises:
            AttributeError: After a column is picked, where pandas has a group
                by over the bins, which has no `name`.
            ValueError: For bins read from a column with `on=`, in pandas' words.
            NotImplementedError: When a row had a missing timestamp.
        """
        if self._picked:
            kind = "SeriesGroupBy" if self._obj.ndim == 1 else "DataFrameGroupBy"
            raise AttributeError(f"'{kind}' object has no attribute {name!r}")
        if self._chosen:
            raise InvalidArgumentError(
                "Upsampling from level= or on= selection is not supported, use"
                " .set_index(...) to explicitly set index to datetime-like"
            )
        if self._dropped:
            raise NotImplementedError(
                "upsampling rows with a missing timestamp is not supported yet, because"
                " pandas reindexes them with the missing label still among the rows"
            )
        from ._attrs import hold_freq
        from ._frequency import _offset_of

        edges = self._edges(1 if self._right_closed else 0)
        out = self._obj.reindex(edges, method=method, limit=limit, fill_value=fill_value)
        # The rows step by the rule, which pandas' answer carries on its index.
        hold_freq(out, _offset_of(self._rule))
        return out

    def _filled(self, method: str, limit: int | None) -> DataFrame | Series:
        """`ffill` or `bfill`, upsampled, or within each bin after a column is picked.

        Raises:
            NotImplementedError: Within the bins when a row had a missing timestamp.
        """
        from ._frame import DataFrameGroupBy

        if not self._picked:
            return self._upsampled(method, method, limit)
        if self._dropped:
            raise NotImplementedError(
                f"{method} over rows with a missing timestamp answers NaN for those rows,"
                " which is not written yet"
            )
        grouped = DataFrameGroupBy(self._frame(), [_BIN], True, True, True)
        return self._on_rows(getattr(grouped, method)(limit=limit))

    def asfreq(self, fill_value: Any = None) -> DataFrame | Series:
        """The value at exactly every bin's edge, missing where no row is there.

        Raises:
            ValueError: When two rows share a timestamp, which pandas refuses too.
        """
        return self._upsampled("asfreq", None, fill_value=fill_value)

    def ffill(self, limit: int | None = None) -> DataFrame | Series:
        """Every bin's edge with the last value at or before it.

        After a column is picked pandas fills each bin's rows from the rows
        before them in the bin instead, on the rows' own labels, and so does this.
        """
        return self._filled("ffill", limit)

    def bfill(self, limit: int | None = None) -> DataFrame | Series:
        """Every bin's edge with the first value at or after it, or, after a column
        is picked, each bin's rows filled from the rows after them in the bin."""
        return self._filled("bfill", limit)

    def nearest(self, limit: int | None = None) -> DataFrame | Series:
        """Every bin's edge with the value nearest to it."""
        return self._upsampled("nearest", "nearest", limit)

    def interpolate(
        self,
        method: str = "linear",
        *,
        axis: Any = 0,
        limit: int | None = None,
        limit_direction: str = "forward",
        limit_area: Any = None,
        **kwargs: Any,
    ) -> DataFrame | Series:
        """The values at every bin's edge, interpolated between the rows around it.

        The rows that fall between the edges are put in among them before the
        interpolation and taken out after it, as pandas does.

        Raises:
            ValueError: For `inplace=True`, in pandas' words.
        """
        from ._pandas import concat

        if kwargs.pop("inplace", False):
            raise InvalidArgumentError("Cannot interpolate inplace on a resampled object.")
        answer = self._upsampled("interpolate", None)
        edges = answer.index
        on_edge = self._obj.index.isin(edges)
        between = [at for at, there in enumerate(on_edge) if not there]
        if between:
            answer = concat([answer, self._obj.iloc[between]]).sort_index()
        filled = answer.interpolate(
            method=method,
            axis=axis,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area,
            **kwargs,
        )
        return filled.reindex(edges)
