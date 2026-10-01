"""A rolling window whose bounds a `BaseIndexer` gives, reduced one window at a time.

The native window kernel takes a width and walks it along the rows. A window
from `pandas.api.indexers` is a pair of row number arrays instead, which can
look forward, skip, or follow a calendar, so it is read here: each window is a
slice of the values, and a slice with fewer present values than `min_periods`
answers NaN, as in pandas. `count` measures the slice itself against
`min_periods` rather than its present values, which is pandas' rule too.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from .errors import InvalidArgumentError


def _present(window: list[float]) -> list[float]:
    return [value for value in window if not math.isnan(value)]


def _spread(values: list[float], ddof: int) -> float:
    if len(values) - ddof <= 0:
        return math.nan
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - ddof)


class IndexedRolling:
    """Reductions over windows a `BaseIndexer` bounds, for `rolling(indexer)`."""

    def __init__(
        self,
        data: Any,
        indexer: Any,
        min_periods: int | None,
        center: bool,
        closed: str | None,
        step: int | None,
    ) -> None:
        self._data = data
        self._indexer = indexer
        default = getattr(indexer, "window_size", 0)
        self._min_periods = default if min_periods is None else min_periods
        self._center = center
        self._closed = closed
        self._step = step

    def _bounds(self) -> list[tuple[int, int]]:
        start, end = self._indexer.get_window_bounds(
            num_values=len(self._data),
            min_periods=self._min_periods,
            center=self._center,
            closed=self._closed,
            step=self._step,
        )
        if len(start) != len(end):
            raise InvalidArgumentError(
                f"start ({len(start)}) and end ({len(end)}) bounds must be the same length"
            )
        return [(int(a), int(b)) for a, b in zip(start, end, strict=True)]

    def _each(self, reduce: Any, by_length: bool = False) -> Any:
        """`reduce` over every window of every column, NaN where too few values are present."""
        from ._frame import DataFrame, Series

        bounds = self._bounds()
        index = self._data.index[:: self._step or 1]

        def column(values: list[Any], name: Any) -> Series:
            floats = [math.nan if value is None else float(value) for value in values]
            out = []
            for start, end in bounds:
                window = floats[start:end]
                present = _present(window)
                enough = (len(window) if by_length else len(present)) >= self._min_periods
                out.append(reduce(window, present) if enough and window else math.nan)
            return Series(out, index=index, name=name, dtype="float64")

        if isinstance(self._data, DataFrame):
            return DataFrame(
                {name: column(self._data[name].tolist(), name) for name in self._data.columns},
                index=index,
            )
        return column(self._data.tolist(), self._data.name)

    def count(self, numeric_only: bool = False) -> Any:
        """How many values each window holds."""
        return self._each(lambda window, present: float(len(present)), by_length=True)

    def sum(self, numeric_only: bool = False, *args: Any, **kwargs: Any) -> Any:
        """Each window's total."""
        return self._each(lambda window, present: float(sum(present)))

    def mean(self, numeric_only: bool = False, *args: Any, **kwargs: Any) -> Any:
        """Each window's mean."""
        return self._each(lambda window, present: sum(present) / len(present))

    def median(self, numeric_only: bool = False, **kwargs: Any) -> Any:
        """Each window's median."""
        return self._each(lambda window, present: float(statistics.median(present)))

    def min(self, numeric_only: bool = False, *args: Any, **kwargs: Any) -> Any:
        """Each window's smallest value."""
        return self._each(lambda window, present: min(present))

    def max(self, numeric_only: bool = False, *args: Any, **kwargs: Any) -> Any:
        """Each window's largest value."""
        return self._each(lambda window, present: max(present))

    def var(self, ddof: int = 1, numeric_only: bool = False, **kwargs: Any) -> Any:
        """Each window's variance."""
        return self._each(lambda window, present: _spread(present, ddof))

    def std(self, ddof: int = 1, numeric_only: bool = False, **kwargs: Any) -> Any:
        """Each window's standard deviation."""
        return self._each(lambda window, present: math.sqrt(_spread(present, ddof)))

    def apply(
        self,
        func: Any,
        raw: bool = False,
        engine: Any = None,
        engine_kwargs: Any = None,
        args: Any = None,
        kwargs: Any = None,
    ) -> Any:
        """`func` called on each window, as a numpy array when `raw`, else a column."""
        import numpy

        from ._frame import Series

        extra, named = tuple(args or ()), dict(kwargs or {})

        def one(window: list[float], present: list[float]) -> float:
            values = numpy.array(window) if raw else Series(window)
            return float(func(values, *extra, **named))

        return self._each(one)

    def aggregate(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """One reduction by name or function, or a list of them on a column.

        Raises:
            AttributeError: For a name that is no reduction.
        """
        from ._frame import DataFrame

        if isinstance(func, str):
            return getattr(self, func)(*args, **kwargs)
        if callable(func):
            return self.apply(func, args=args, kwargs=kwargs)
        if isinstance(func, list) and not isinstance(self._data, DataFrame):
            names = [one if isinstance(one, str) else one.__name__ for one in func]
            done = [self.aggregate(one) for one in func]
            return DataFrame(dict(zip(names, done, strict=True)), index=done[0].index)
        raise InvalidArgumentError("aggregate takes a name, a function or a list on a column")

    agg = aggregate
