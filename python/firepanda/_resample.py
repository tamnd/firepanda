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
days are calendar offsets whose bins are not all the same length, and they are
refused by name.

The rules below were measured against pandas 3.0.

- The bins are measured from `origin`. The default, `start_day`, is midnight
  of the first day, `start` is the first timestamp and `epoch` is 1970-01-01.
  A rule in days always measures from midnight of the first day.
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
_LABEL = "__firepanda_label__"
_VALUE = "__firepanda_value__"


def _step(rule: Any) -> tuple[int, bool]:
    """The length of a rule in nanoseconds, and whether it is counted in days.

    Raises:
        InvalidArgumentError: For a rule pandas cannot read, with its message.
        ZeroDivisionError: For a rule of no length, as pandas raises.
        NotImplementedError: For a calendar offset or a rule that is not text.
    """
    if not isinstance(rule, str):
        raise NotImplementedError(
            "resample: a rule is read from text for now, like '6h' or 'D', because an offset"
            " object is pandas' offsets namespace, which firepanda does not have"
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


def _numeric(dtype: str) -> bool:
    """Whether pandas counts a type as a number under `numeric_only`."""
    return re.fullmatch(r"u?int\d+|float\d+|bool", dtype) is not None


def _unit(dtype: str) -> str:
    """The unit of a datetime type with no zone, or empty for any other type."""
    found = re.fullmatch(r"datetime64\[(s|ms|us|ns)\]", dtype)
    return found.group(1) if found else ""


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
        if _bins is not None:
            (self._times, self._codes, self._first, self._count, self._origin, self._step,
             self._time_unit, self._right_label, self._name, self._dropped) = _bins  # fmt: skip
            return
        from ._frame import DataFrame

        if closed not in (None, "left", "right"):
            raise InvalidArgumentError(f"Unsupported value {closed} for `closed`")
        if label not in (None, "left", "right"):
            raise InvalidArgumentError(f"Unsupported value {label} for `label`")
        if origin not in ("epoch", "start", "start_day", "end", "end_day") and isinstance(
            origin, str
        ):
            raise InvalidArgumentError(
                "'origin' should be equal to 'epoch', 'start', 'start_day', 'end', 'end_day'"
                f" or should be a Timestamp convertible type. Got {origin!r} instead."
            )
        if origin not in ("epoch", "start", "start_day"):
            raise NotImplementedError(
                f"resample: origin={origin!r} is not supported yet, because measuring the"
                " bins back from the last timestamp or from a chosen moment is not written"
            )
        if offset is not None:
            raise NotImplementedError(
                "offset= is not supported yet, because moving the bins by a span needs"
                " pandas' span parsing, which the resample does not have"
            )
        if on is not None:
            if not isinstance(obj, DataFrame):
                raise NotImplementedError("resample: on= names a column, which a series has not")
            times = obj[on]
            self._obj = obj.drop(columns=[on])
            self._name = on
            kind = "Index"
        else:
            if level not in (None, 0, obj.index.name):
                raise NotImplementedError(
                    "resample: level= picks a level of a MultiIndex, which firepanda does not have"
                )
            times = obj.index.to_series()
            self._name = obj.index.name
            kind = "RangeIndex" if obj.index._inner.is_range() else "Index"
        dtype = str(times.dtype)
        unit = _unit(dtype)
        if not unit:
            if dtype.startswith("datetime64["):
                raise NotImplementedError(
                    "resample: a timestamp with a zone is not supported yet, because a day"
                    " in a zone is not always 24 hours long"
                )
            raise TypeError(
                "Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex, but got an"
                f" instance of '{kind}'"
            )
        length, days = _step(rule)
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
        counts = times.astype("int64")
        if len(counts) == 0 or (origin == "epoch" and not days):
            start = 0
        elif origin == "start" and not days:
            start = int(counts.min())
        else:
            day = _NANOS["D"] // per
            start = int(counts.min()) // day * day
        # Closed on the left a bin holds [edge, edge + step), and closed on the
        # right (edge, edge + step], which is the ceiling less one.
        right = closed == "right"
        codes = -((start - counts) // step) - 1 if right else (counts - start) // step
        self._times = times
        self._codes = codes
        self._first = int(codes.min()) if len(codes) else 0
        self._count = int(codes.max()) - self._first + 1 if len(codes) else 0
        self._origin = start
        self._step = step
        self._time_unit = unit
        self._right_label = label == "right"

    def _state(self) -> tuple[Any, ...]:
        """The bins, handed to a resampler over a part of the same rows."""
        return (self._times, self._codes, self._first, self._count, self._origin, self._step,
                self._time_unit, self._right_label, self._name, self._dropped)  # fmt: skip

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def __getitem__(self, key: Any) -> Resampler:
        """The same bins over one column, or over a list of columns."""
        from ._frame import Series

        if isinstance(self._obj, Series):
            raise KeyError(key)
        return Resampler(self._obj[key], None, _bins=self._state())

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

        return Index(self._stamps(range(self._first, self._first + self._count + 1), 0)).rename(
            self._name
        )

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
        counts = [self._origin + (code + extra) * self._step for code in codes]
        return to_datetime(Series(counts, dtype="int64"), unit=self._time_unit)

    def _frame(self) -> DataFrame:
        """The values on plain positions with their bin numbers as the last column."""
        from ._frame import DataFrame, Series

        plain = self._obj.reset_index(drop=True)
        if isinstance(self._obj, Series):
            plain = DataFrame({_VALUE: plain})
        return plain.assign(**{_BIN: self._codes})

    def _labelled(self, out: DataFrame, fill: dict[str, Any] | None) -> DataFrame | Series:
        """A reduction by bin number put on every bin and labelled with its timestamp."""
        from ._frame import Series

        types = {name: str(out[name].dtype) for name in out.columns}
        if len(out) < self._count:
            out = out.reindex(list(range(self._first, self._first + self._count)))
            for name in out.columns:
                if fill is not None and name in fill:
                    out = out.assign(**{name: out[name].fillna(fill[name])})
                    if types[name] != str(out[name].dtype) and not bool(out[name].isna().any()):
                        out = out.assign(**{name: out[name].astype(types[name])})
        out = out.reset_index(drop=True).assign(
            **{_LABEL: self._stamps(range(self._first, self._first + self._count))}
        )
        out = out.set_index(_LABEL).rename_axis(self._name)
        if isinstance(self._obj, Series):
            answer = out[_VALUE]
            return answer.rename(self._obj.name)
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

        Raises:
            NotImplementedError: Over a frame, where pandas answers with two
                levels of column labels.
        """
        from ._frame import DataFrame, Series

        if not isinstance(self._obj, Series):
            raise NotImplementedError(
                "ohlc over a frame answers with two levels of column labels, which is"
                " pandas' MultiIndex and firepanda does not have one"
            )
        parts = {"open": self.first(), "high": self.max(), "low": self.min(), "close": self.last()}
        stamps = parts["open"].index.to_series().reset_index(drop=True)
        out = DataFrame({name: part.reset_index(drop=True) for name, part in parts.items()})
        return out.assign(**{_LABEL: stamps}).set_index(_LABEL).rename_axis(self._name)

    # ------------------------------------------------------------------
    # Several at once, and the rest
    # ------------------------------------------------------------------

    def aggregate(self, func: Any = None, *args: Any, **kwargs: Any) -> Any:
        """A reduction by name, or one by name for each column.

        Raises:
            NotImplementedError: For a function, or for a list of names, which
                pandas answers with two levels of column labels.
        """
        from ._frame import DataFrame

        if isinstance(func, str):
            return self._by_name(func)(*args, **kwargs)
        if isinstance(func, dict) and isinstance(self._obj, DataFrame):
            parts = {name: self[name].aggregate(how) for name, how in func.items()}
            if any(not isinstance(how, str) for how in func.values()):
                raise NotImplementedError(
                    "aggregate: a list of names for a column answers with two levels of"
                    " column labels, which firepanda does not have"
                )
            first = next(iter(parts.values()))
            out = DataFrame({name: part.reset_index(drop=True) for name, part in parts.items()})
            return (
                out.assign(**{_LABEL: first.index.to_series().reset_index(drop=True)})
                .set_index(_LABEL)
                .rename_axis(self._name)
            )
        raise NotImplementedError(
            "aggregate takes a reduction by name, or a mapping of columns to names, for"
            " now, because a function is Python run once a bin and a list answers with"
            " two levels of column labels"
        )

    agg = aggregate

    def _by_name(self, how: str) -> Any:
        """A reduction of this resampler by the name pandas gives it."""
        if how in ("agg", "aggregate", "apply", "transform", "pipe") or how.startswith("_"):
            raise AttributeError(f"'{how}' is not a valid function for 'Resampler' object")
        method = getattr(type(self), how, None)
        if not callable(method):
            raise AttributeError(f"'{how}' is not a valid function for 'Resampler' object")
        return getattr(self, how)

    def apply(self, func: Any = None, *args: Any, **kwargs: Any) -> Any:
        """A reduction by name; a function is refused."""
        if isinstance(func, str):
            return self.aggregate(func, *args, **kwargs)
        raise NotImplementedError(
            "apply with a function runs Python once a bin, which is not written yet"
        )

    def transform(self, arg: Any, *args: Any, **kwargs: Any) -> DataFrame | Series:
        """A reduction by name put back on every row of its bin, on the rows' own labels.

        Raises:
            NotImplementedError: For a function.
        """
        from ._frame import DataFrameGroupBy, Series

        if not isinstance(arg, str):
            raise NotImplementedError(
                "transform takes a reduction by name for now, because a function is Python"
                " run once a bin"
            )
        if self._dropped:
            raise NotImplementedError(
                "transform over rows with a missing timestamp answers NaN for those rows,"
                " which is not written yet"
            )
        grouped = DataFrameGroupBy(self._frame(), [_BIN], True, True, True)
        out = grouped.transform(arg, *args, **kwargs)
        labels = self._obj.index.to_series().reset_index(drop=True).rename(None)
        out = out.assign(**{_LABEL: labels}).set_index(_LABEL).rename_axis(self._obj.index.name)
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

    def asfreq(self, fill_value: Any = None) -> DataFrame | Series:
        """The value at exactly every bin's label, missing where no row is there.

        Raises:
            NotImplementedError: When two rows share a timestamp, which pandas
                refuses too, and for a label on the right.
        """
        from ._frame import DataFrame, DataFrameGroupBy

        if self._right_label:
            raise NotImplementedError("asfreq with label='right' is not written yet")
        frame = self._frame()
        counts = self._times.reset_index(drop=True).astype("int64")
        exact = counts == (frame[_BIN] * self._step + self._origin)
        frame = frame.loc[exact.fillna(False)]
        if bool(frame[_BIN].duplicated().any()):
            raise InvalidArgumentError("cannot reindex on an axis with duplicate labels")
        grouped = DataFrameGroupBy(frame, [_BIN], True, True, True)
        out = grouped.first()
        if not isinstance(out, DataFrame):
            out = DataFrame({_VALUE: out})
        fill = None if fill_value is None else dict.fromkeys(out.columns, fill_value)
        return self._labelled(out, fill)

    def _upsampling(self, name: str) -> Any:
        raise NotImplementedError(
            f"{name} fills the bins from the rows before or after them, which is"
            " upsampling and not written yet"
        )

    def ffill(self, limit: int | None = None) -> Any:
        """Refused: upsampling is not written yet."""
        return self._upsampling("ffill")

    def bfill(self, limit: int | None = None) -> Any:
        """Refused: upsampling is not written yet."""
        return self._upsampling("bfill")

    def nearest(self, limit: int | None = None) -> Any:
        """Refused: upsampling is not written yet."""
        return self._upsampling("nearest")

    def interpolate(
        self,
        method: str = "linear",
        *,
        axis: Any = 0,
        limit: int | None = None,
        limit_direction: str = "forward",
        limit_area: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Refused: upsampling is not written yet."""
        return self._upsampling("interpolate")
