"""An exponentially weighted mean that decays against real instants, `ewm(times=...)`.

Without `times` the weight of a row falls by the same factor at every step,
and the core runs that. With `times` the step between two rows is the time
between them divided by the half life, and the factor is raised to that
power, so a gap of three days decays three times as far as a gap of one. This
is pandas' own recurrence, run here in Python, with the steps pandas takes on a
missing value and under `adjust=False`.

pandas only answers the mean this way, and every other reduction is refused
with the sentence pandas raises.
"""

from __future__ import annotations

import datetime
import itertools
import math
from typing import Any

from .errors import InvalidArgumentError

__all__ = ["TimedWindow"]

# How many nanoseconds one step of each unit a datetime column can be held in is.
_NANOSECONDS = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}


def _center(com: Any, span: Any, alpha: Any) -> float:
    """The centre of mass the decay is given in, one when none of the three is.

    Raises:
        InvalidArgumentError: For more than one of them or one out of range, in pandas' words.
    """
    if sum(value is not None for value in (com, span, alpha)) > 1:
        raise InvalidArgumentError("comass, span, halflife, and alpha are mutually exclusive")
    if com is not None:
        if com < 0:
            raise InvalidArgumentError("comass must satisfy: comass >= 0")
        return float(com)
    if span is not None:
        if span < 1:
            raise InvalidArgumentError("span must satisfy: span >= 1")
        return (span - 1) / 2
    if alpha is not None:
        if alpha <= 0 or alpha > 1:
            raise InvalidArgumentError("alpha must satisfy: 0 < alpha <= 1")
        return 1 / alpha - 1
    return 1.0


def _steps(times: Any, halflife: Any) -> list[float]:
    """The time between each pair of rows, counted in half lives.

    Raises:
        InvalidArgumentError: For a missing instant, in pandas' words.
    """
    from ._scalars import Timedelta

    printed = str(times.dtype)
    unit = printed[printed.index("[") + 1 :].split(",")[0].rstrip("]")
    if bool(times.isna().any()):
        raise InvalidArgumentError("Cannot convert NaT values to integer")
    instants = [float(value) for value in times.astype("int64").tolist()]
    span = float(Timedelta(halflife).value) / _NANOSECONDS[unit]
    return [(later - earlier) / span for earlier, later in itertools.pairwise(instants)]


def _decayed(
    values: list[float], steps: list[float], alpha: float, adjust: bool, ignore_na: bool, least: int
) -> list[float]:
    """pandas' weighted mean down one column, with the decay raised to each step.

    Under `adjust=False` pandas gives a new value the weight alpha, except when
    the centre of mass is one, where it gives it what the decayed old weight
    left over, so a gap hands the new value more than half.
    """
    if not values:
        return []
    kept = 1.0 - alpha
    weighted = values[0]
    seen = int(weighted == weighted)
    old = 1.0
    new = 1.0 if adjust else alpha
    out = [weighted if seen >= least else math.nan]
    for at in range(1, len(values)):
        value = values[at]
        observed = value == value
        seen += observed
        if weighted == weighted:
            if observed or not ignore_na:
                old *= kept ** steps[at - 1]
            if observed:
                if weighted != value:
                    if not adjust and alpha == 0.5:
                        # pandas' kernel reweighs by the leftover only when com is one,
                        # and the new weight stays as set until it is set again.
                        new = 1.0 - old
                    weighted = (old * weighted + new * value) / (old + new)
                old = old + new if adjust else 1.0
        elif observed:
            weighted = value
        out.append(weighted if seen >= least else math.nan)
    return out


def _floats(column: Any) -> list[float]:
    """A column's values as floats, with NaN for a gap."""
    return [
        math.nan if value is None or value != value else float(value) for value in column.tolist()
    ]


def _column_mean(
    column: Any, steps: list[float], alpha: float, adjust: bool, ignore_na: bool, least: int
) -> Any:
    """One column's weighted mean as a float64 column with its labels and name."""
    from ._frame import Series

    values = _decayed(_floats(column), steps, alpha, adjust, ignore_na, least)
    return Series(values, index=column.index, name=column.name, dtype="float64")


def _mean_of(
    data: Any, steps: list[float], alpha: float, adjust: bool, ignore_na: bool, least: int
) -> Any:
    """The weighted mean of a column, or of each column of a frame."""
    from ._frame import DataFrame, Series

    if isinstance(data, Series):
        return _column_mean(data, steps, alpha, adjust, ignore_na, least)
    columns = {
        name: _column_mean(data[name], steps, alpha, adjust, ignore_na, least)
        for name in data.columns
    }
    return DataFrame(columns, index=data.index)


def unadjusted_mean(data: Any, alpha: float, least: int) -> Any:
    """The mean under `adjust=False` of rows that are evenly spaced, the core's case with gaps.

    With a centre of mass of one pandas weighs the new value by what the old
    weight left over, so a gap that decays the old weight twice gives the new
    value the rest. That is the same recurrence as a timed one with every step
    one row long.
    """
    return _mean_of(data, [1.0] * max(len(data) - 1, 0), alpha, False, False, least)


class TimedWindow:
    """What `ewm(times=...)` hands back, which answers `mean` and refuses the rest."""

    __slots__ = ("_adjust", "_alpha", "_data", "_ignore_na", "_least", "_shown", "_steps")

    def __init__(
        self,
        data: Any,
        com: Any,
        span: Any,
        halflife: Any,
        alpha: Any,
        min_periods: Any,
        adjust: bool,
        ignore_na: bool,
        times: Any,
    ) -> None:
        """Checks `times` and the half life as pandas does, in pandas' order.

        Raises:
            InvalidArgumentError: For times that are not instants, of another
                length, with a gap, or a half life that is not a duration.
            NotImplementedError: For a decay given another way beside
                `adjust=False`, which pandas refuses.
        """
        printed = str(getattr(times, "dtype", ""))
        if not printed.startswith("datetime64"):
            raise InvalidArgumentError("times must be datetime64 dtype.")
        if len(times) != len(data):
            raise InvalidArgumentError("times must be the same length as the object.")
        if not isinstance(halflife, (str, datetime.timedelta)) and (
            type(halflife).__name__ != "timedelta64"
        ):
            raise InvalidArgumentError("halflife must be a timedelta convertible object")
        self._steps = _steps(times, halflife)
        if any(value is not None for value in (com, span, alpha)) and not adjust:
            raise NotImplementedError(
                "None of com, span, or alpha can be specified if times is provided and adjust=False"
            )
        self._alpha = 1 / (1 + _center(com, span, alpha))
        self._data = data
        self._adjust = bool(adjust)
        self._ignore_na = bool(ignore_na)
        self._least = 1 if min_periods is None else max(int(min_periods), 1)
        self._shown = (
            ("com", com),
            ("span", span),
            ("halflife", halflife),
            ("alpha", alpha),
            ("min_periods", self._least),
            ("adjust", self._adjust),
            ("ignore_na", self._ignore_na),
            ("times", times),
        )

    def __repr__(self) -> str:
        """The decay as pandas prints it, with the instants it was given."""
        shown = [f"{name}={value}" for name, value in self._shown if value is not None]
        return f"ExponentialMovingWindow [{','.join(shown)},method=single]"

    def mean(
        self, numeric_only: bool = False, engine: Any = None, engine_kwargs: Any = None
    ) -> Any:
        """The weighted mean, with each row's weight decayed by the time since it."""
        data = self._data
        if numeric_only and hasattr(data, "columns"):
            data = data.select_dtypes("number")
        return _mean_of(data, self._steps, self._alpha, self._adjust, self._ignore_na, self._least)

    def sum(self, *args: Any, **kwargs: Any) -> Any:
        """Refused, as pandas refuses it."""
        raise NotImplementedError("sum is not implemented with times")

    def std(self, *args: Any, **kwargs: Any) -> Any:
        """Refused, as pandas refuses it."""
        raise NotImplementedError("std is not implemented with times")

    def var(self, *args: Any, **kwargs: Any) -> Any:
        """Refused, as pandas refuses it."""
        raise NotImplementedError("var is not implemented with times")

    def corr(self, *args: Any, **kwargs: Any) -> Any:
        """Refused, as pandas refuses it."""
        raise NotImplementedError("corr is not implemented with times")

    def cov(self, *args: Any, **kwargs: Any) -> Any:
        """Refused, as pandas refuses it."""
        raise NotImplementedError("cov is not implemented with times")
