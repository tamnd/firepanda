"""Resampling rows labelled by periods, which is pandas' `PeriodIndexResampler`.

A period index resamples by moving each period to the target frequency rather
than by cutting a line of instants into bins. Going to a coarser frequency, a
month to its quarter, each row joins the bin of the period it falls in, and
the bins run from the first row's period to the last row's with every period
between them, as the timestamps' bins do. Going to a finer frequency, a month
to its days, each row lands on the target period at its start or its end, as
`convention` says, and the rows between are the gaps an upsample leaves.

pandas decides between the two by whether the one frequency is a sub period of
the other, `is_subperiod` and `is_superperiod` in `pandas.tseries.frequencies`,
and `_sub` and `_super` here read the same tables. A reduction on a finer or
equal frequency is the upsample with nothing filled, and one between two
frequencies neither of which divides the other is refused. `closed`, `label`,
`origin` and `offset` mean nothing to periods and are set aside, as pandas
sets them aside.

The bins are numbered by the target periods' ordinals, so everything the
timestamp resampler does with bin numbers, the group by reductions, `size`,
`ohlc`, `agg` and `apply`, works here unchanged, and only the labels, the
upsample and `get_group` are written again.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from ._resample import Resampler
from .errors import IncompatibleFrequency

__all__ = ["PeriodResampler"]

_TIMES = {"h", "min", "s", "ms", "us", "ns"}
_DAYS = {"D", "C", "B"} | _TIMES
_MONTHS = {
    name: number
    for number, name in enumerate(
        ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1
    )
}


def _code(freqstr: str) -> str:
    """A frequency's name as pandas compares it, upper case but for the clock units."""
    freqstr = freqstr.lstrip("0123456789")
    return freqstr if freqstr in _TIMES else freqstr.upper()


def _annual(code: str) -> bool:
    return code == "Y" or code.startswith("Y-")


def _quarterly(code: str) -> bool:
    return code == "Q" or code.startswith("Q-")


def _weekly(code: str) -> bool:
    return code == "W" or code.startswith("W-")


def _month(code: str) -> int:
    """The month a yearly or quarterly frequency ends on."""
    return _MONTHS[code.split("-")[1]] if "-" in code else 12


def _conform(source: str, target: str) -> bool:
    return _month(source) % 3 == _month(target) % 3


def _finer(code: str) -> set[str]:
    """The frequencies strictly inside one of a day or finer, as pandas lists them."""
    if code in ("B", "C", "D"):
        return {code} | _TIMES
    order = ["h", "min", "s", "ms", "us", "ns"]
    return set(order[order.index(code) :]) if code in order else set()


def _sub(source: str, target: str) -> bool:
    """Whether `source` is a sub period of `target`, so that resampling to it gathers rows."""
    if _annual(target):
        if _quarterly(source):
            return _conform(source, target)
        return source in _DAYS | {"M"}
    if _quarterly(target):
        return source in _DAYS | {"M"}
    if target == "M":
        return source in _DAYS
    if _weekly(target):
        return source in _DAYS | {target}
    return source in _finer(target)


def _super(source: str, target: str) -> bool:
    """Whether `source` is a super period of `target`, so that resampling to it spreads rows."""
    if _annual(source):
        if _annual(target):
            return _month(source) == _month(target)
        if _quarterly(target):
            return _conform(source, target)
        return target in _DAYS | {"M"}
    if _quarterly(source) or source == "M":
        return target in _DAYS | ({"M"} if _quarterly(source) else set())
    if _weekly(source):
        return target in _DAYS | {source}
    if source in ("B", "C", "D"):
        return target in _DAYS
    return target in _finer(source)


class PeriodResampler(Resampler):
    """The bins of a resample over rows labelled by periods, waiting for a reduction."""

    def __init__(self, obj: Any, rule: Any, convention: str, group_keys: bool) -> None:
        from ._frame import Series
        from ._period_index import _kind_of
        from ._scalars import NaT

        index = obj.index
        self._obj = obj
        self._picked = False
        self._keyed = bool(group_keys)
        self._how = "start" if convention in ("start", "s") else "end"
        self._source = index.freqstr
        self._target = _kind_of([], rule, None)[7:-1]
        periods = index._periods()
        kept = [at for at, period in enumerate(periods) if period is not NaT]
        self._dropped = len(kept) < len(periods)
        if self._dropped:
            # A missing period is in no bin, so its row is left out of every answer.
            self._obj = obj.iloc[kept]
            periods = [periods[at] for at in kept]
        ordinals = [period.asfreq(self._target, how=self._how).ordinal for period in periods]
        self._ordinals = ordinals
        self._times = index.to_series().iloc[kept].reset_index(drop=True)
        # A bin of a frequency counted more than once, `2Q`, spans that many
        # ordinals, laid from the first row's.
        self._each = int(re.match(r"\d*", self._target).group() or 1)
        self._low = min(ordinals) if ordinals else 0
        codes = [(ordinal - self._low) // self._each for ordinal in ordinals]
        self._codes = Series(codes, dtype="int64")
        self._first = 0
        self._count = max(codes) + 1 if codes else 0
        self._origin, self._step, self._time_unit = 0, 1, "ns"
        self._right_label = self._right_closed = self._chosen = self._wall = False
        self._name = index.name
        # The periods carry their own frequency, so the labels hold no offset beside it.
        self._rule = None
        self._marks = None
        self._zone = None

    def __getitem__(self, key: Any) -> PeriodResampler:
        """The same bins over one column, or over a list of columns."""
        from ._frame import Series

        if isinstance(self._obj, Series):
            raise KeyError(key)
        picked = copy.copy(self)
        picked._obj = self._obj[key]
        picked._picked = True
        return picked

    def get_group(self, name: Any) -> Any:
        """The rows of the bin labelled `name`.

        Raises:
            KeyError: For a label that names no bin with rows in it.
        """
        from ._period import Period

        wanted = name if isinstance(name, Period) else Period(name, freq=self._target)
        for label, positions in self.indices.items():
            if label == wanted:
                return self._obj.iloc[positions]
        raise KeyError(name)

    def _stamps(self, codes: Any, shift: int | None = None) -> Any:
        """The target periods that label some bin numbers."""
        from ._frame import Series
        from ._period_index import PeriodIndex

        ordinals = [self._low + code * self._each for code in codes]
        return Series(PeriodIndex.from_ordinals(ordinals, freq=self._target))

    def _reduce(self, how: str, *args: Any, **kwargs: Any) -> Any:
        """A reduction over the bins going to a coarser frequency, or the upsample otherwise.

        Raises:
            IncompatibleFrequency: Between two frequencies neither of which is a
                sub period of the other, in pandas' words.
        """
        source, target = _code(self._source), _code(self._target)
        if source != target and _sub(source, target):
            return super()._reduce(how, *args, **kwargs)
        if source == target or _super(source, target):
            return self.asfreq()
        raise IncompatibleFrequency(
            f"Frequency {self._obj.index.freq!r} cannot be resampled to"
            f" {self._stamps([0]).dtype.freq!r}, as they are not sub or super periods"
        )

    def _upsampled(
        self, name: str, method: str | None, limit: int | None = None, fill_value: Any = None
    ) -> Any:
        """The rows moved to the target frequency and reindexed onto every period.

        The periods run from the first row's, at the start or the end as
        `convention` says, to the end of the last row's.

        Raises:
            AttributeError: After a column is picked, as the timestamps' resampler does.
            NotImplementedError: When a row had a missing period.
        """
        from ._period_index import PeriodIndex, period_range

        if self._picked:
            kind = "SeriesGroupBy" if self._obj.ndim == 1 else "DataFrameGroupBy"
            raise AttributeError(f"'{kind}' object has no attribute {name!r}")
        if self._dropped:
            raise NotImplementedError(
                "upsampling rows with a missing period is not supported yet, because"
                " pandas reindexes them with the missing label still among the rows"
            )
        moved = self._obj.copy()
        moved.index = PeriodIndex.from_ordinals(self._ordinals, freq=self._target, name=self._name)
        if not len(moved):
            return moved
        periods = self._times.tolist()
        first = min(periods).asfreq(self._target, how=self._how)
        last = max(periods).asfreq(self._target, how="end")
        edges = period_range(first, last, freq=self._target, name=self._name)
        return moved.reindex(edges, method=method, limit=limit, fill_value=fill_value)

    def interpolate(self, method: str = "linear", **kwargs: Any) -> Any:
        """The values at every target period, interpolated between the rows around them.

        Raises:
            ValueError: For `inplace=True`, in pandas' words.
        """
        from .errors import InvalidArgumentError

        if kwargs.pop("inplace", False):
            raise InvalidArgumentError("Cannot interpolate inplace on a resampled object.")
        return self._upsampled("interpolate", None).interpolate(method=method, **kwargs)


def is_periods(index: Any) -> bool:
    """Whether rows are labelled by periods, which resample by `PeriodResampler`."""
    from ._period_index import PeriodIndex

    return isinstance(index, PeriodIndex)
