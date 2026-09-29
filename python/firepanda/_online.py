"""An exponentially weighted mean that can be fed more rows later, which is `ewm(...).online()`.

pandas computes this with a numba kernel and refuses every engine but numba.
The recurrence is short, so it is written here in plain Python: the running
mean and the weight of what came before are kept per column, and `mean(update=...)`
carries on from where the last call stopped instead of starting over. The
arithmetic follows `pandas.core.window.online` step for step, so the numbers
agree with pandas run with numba to the last bit.
"""

from __future__ import annotations

import math
from typing import Any

from ._frame import DataFrame, ExponentialMovingWindow, Series
from ._pandas import EwmMixin
from .errors import InvalidArgumentError


class OnlineExponentialMovingWindow(ExponentialMovingWindow):
    """A decay that remembers its last row, so later rows can be added to it."""

    __slots__ = ("_engine", "_engine_kwargs", "_last", "_old_weights")

    _engine: Any
    _engine_kwargs: Any
    _last: list[float] | None
    _old_weights: list[float]

    def __init__(
        self, window: ExponentialMovingWindow, engine: Any = "numba", engine_kwargs: Any = None
    ) -> None:
        for name in EwmMixin.__slots__:
            setattr(self, name, getattr(window, name))
        if engine != "numba":
            raise InvalidArgumentError("'numba' is the only supported engine")
        self._engine = engine
        self._engine_kwargs = engine_kwargs
        self.reset()

    @property
    def engine(self) -> Any:
        return self._engine

    @property
    def engine_kwargs(self) -> Any:
        return self._engine_kwargs

    def _width(self) -> int:
        data = self._data
        return len(data.columns) if isinstance(data, DataFrame) else 1

    def reset(self) -> None:
        """Forget the rows seen so far, so the next `mean` starts over."""
        self._old_weights = [1.0] * self._width()
        self._last = None

    def aggregate(self, func: Any = None, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("aggregate is not implemented.")

    agg = aggregate

    def std(self, bias: bool = False, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("std is not implemented.")

    def corr(self, other: Any = None, pairwise: Any = None, numeric_only: bool = False) -> Any:
        raise NotImplementedError("corr is not implemented.")

    def cov(
        self,
        other: Any = None,
        pairwise: Any = None,
        bias: bool = False,
        numeric_only: bool = False,
    ) -> Any:
        raise NotImplementedError("cov is not implemented.")

    def var(self, bias: bool = False, numeric_only: bool = False) -> Any:
        raise NotImplementedError("var is not implemented.")

    def mean(self, *args: Any, update: Any = None, update_times: Any = None, **kwargs: Any) -> Any:
        """The weighted mean of every row so far, or of the rows in `update` after them.

        Raises:
            NotImplementedError: If `update_times` is given, as in pandas.
            InvalidArgumentError: If `update` arrives before any plain call.
        """
        if update_times is not None:
            raise NotImplementedError("update_times is not implemented.")
        frame = isinstance(self._data, DataFrame)
        if update is not None:
            if self._last is None:
                raise InvalidArgumentError(
                    "Must call mean with update=None first before passing update"
                )
            source, skip = update, 1
            rows = [list(self._last), *_rows(update, frame)]
        else:
            source, skip = self._data, 0
            rows = _rows(self._data, frame)
        result = self._run(rows)[skip:]
        if frame:
            return DataFrame(result, index=source.index, columns=source.columns)
        return Series([row[0] for row in result], index=source.index, name=source.name)

    def _run(self, rows: list[list[float]]) -> list[list[float]]:
        """pandas' `online_ewma` over the rows, keeping the weights and the last row."""
        alpha = self._factor
        new_weight = 1.0 if self._adjust else alpha
        decay = 1.0 - alpha
        weights = self._old_weights
        average = list(rows[0])
        seen = [0 if math.isnan(value) else 1 for value in average]
        least = self._min_periods

        def shown() -> list[float]:
            return [a if n >= least else math.nan for a, n in zip(average, seen, strict=True)]

        result = [shown()]
        for row in rows[1:]:
            for j, value in enumerate(row):
                observed = not math.isnan(value)
                seen[j] += observed
                if not math.isnan(average[j]):
                    if observed or not self._ignore_na:
                        weights[j] *= decay
                        if observed:
                            if average[j] != value:
                                average[j] = (weights[j] * average[j] + new_weight * value) / (
                                    weights[j] + new_weight
                                )
                            weights[j] = weights[j] + new_weight if self._adjust else 1.0
                elif observed:
                    average[j] = value
            result.append(shown())
        self._last = result[-1]
        return result


def _rows(data: Any, frame: bool) -> list[list[float]]:
    """The data as rows of floats, with a gap as NaN."""
    if frame:
        columns = [_floats(data.iloc[:, j]) for j in range(len(data.columns))]
        return [list(row) for row in zip(*columns, strict=True)]
    return [[value] for value in _floats(data)]


def _floats(column: Any) -> list[float]:
    return [math.nan if value is None else float(value) for value in column.tolist()]
