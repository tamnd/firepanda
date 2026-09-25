"""`Interval`, a bounded span of numbers, instants or spans, which is `pandas.Interval`.

A pure Python scalar: two ends and which of them the interval holds. The rules
below were measured against pandas 3.0.

- The ends are numbers, both `Timestamp` or both `Timedelta`, and the left is
  at most the right, so a NaN end is refused as pandas refuses it.
- Two intervals are equal, ordered and hashed as the triple of left, right and
  closed.
- Adding or taking away a number or a span moves both ends, and multiplying
  or dividing by a number scales both. Two intervals do not add.

`IntervalIndex`, `interval_range` and an interval column type need a column
type that holds pairs, which firepanda does not have yet.
"""

from __future__ import annotations

import datetime
import numbers
import operator
from typing import Any

from ._scalars import Timedelta, Timestamp
from .errors import InvalidArgumentError

__all__ = ["Interval"]

_CLOSED = ("right", "left", "both", "neither")


def _endpoint(value: Any) -> None:
    """Refuses an end that is not a number, a `Timestamp` or a `Timedelta`."""
    number = isinstance(value, numbers.Real) and not isinstance(value, bool)
    if not number and not isinstance(value, (Timestamp, Timedelta)):
        raise InvalidArgumentError(
            "Only numeric, Timestamp and Timedelta endpoints are allowed when constructing"
            " an Interval."
        )


def _moves(value: Any) -> bool:
    """Whether a value can move both ends of an interval: a number or a span."""
    if isinstance(value, bool):
        return False
    return isinstance(value, (numbers.Number, datetime.timedelta)) or (
        type(value).__name__ == "timedelta64"
    )


def _scales(value: Any) -> bool:
    """Whether a value can scale both ends of an interval: a number."""
    return isinstance(value, numbers.Number) and not isinstance(value, bool)


class Interval:
    """A bounded span with its ends and which of them it holds."""

    __slots__ = ("_closed", "_left", "_right")

    def __init__(self, left: Any, right: Any, closed: str = "right") -> None:
        """Builds an interval.

        Args:
            left: The left end.
            right: The right end, at least the left.
            closed: Which ends the interval holds: right, left, both or neither.

        Raises:
            ValueError: For an unknown closed, an end of the wrong kind, or a left
                end past the right one.
            TypeError: For an instant with a zone and one without, which do not compare.
        """
        if closed not in _CLOSED:
            raise InvalidArgumentError(f"invalid option for 'closed': {closed}")
        _endpoint(left)
        _endpoint(right)
        if not left <= right:
            raise InvalidArgumentError("left side of interval must be <= right side")
        self._left = left
        self._right = right
        self._closed = closed

    @property
    def left(self) -> Any:
        """The left end."""
        return self._left

    @property
    def right(self) -> Any:
        """The right end."""
        return self._right

    @property
    def closed(self) -> str:
        """Which ends the interval holds."""
        return self._closed

    @property
    def closed_left(self) -> bool:
        """Whether the interval holds its left end."""
        return self._closed in ("left", "both")

    @property
    def closed_right(self) -> bool:
        """Whether the interval holds its right end."""
        return self._closed in ("right", "both")

    @property
    def open_left(self) -> bool:
        """Whether the interval leaves its left end out."""
        return not self.closed_left

    @property
    def open_right(self) -> bool:
        """Whether the interval leaves its right end out."""
        return not self.closed_right

    @property
    def length(self) -> Any:
        """The distance from the left end to the right one."""
        return self._right - self._left

    @property
    def mid(self) -> Any:
        """The point halfway between the ends."""
        try:
            return 0.5 * (self._left + self._right)
        except TypeError:
            return self._left + 0.5 * self.length

    @property
    def is_empty(self) -> bool:
        """Whether the interval holds no point: equal ends not both held."""
        return self._left == self._right and self._closed != "both"

    def overlaps(self, other: Any) -> bool:
        """Whether two intervals share a point.

        Raises:
            TypeError: When other is not an interval.
        """
        if not isinstance(other, Interval):
            raise TypeError(f"`other` must be an Interval, got {type(other).__name__}")
        first = operator.le if self.closed_left and other.closed_right else operator.lt
        second = operator.le if other.closed_left and self.closed_right else operator.lt
        return first(self._left, other._right) and second(other._left, self._right)

    def __contains__(self, key: Any) -> bool:
        """Whether a point, or every point of another interval, lies in the interval."""
        if isinstance(key, Interval):
            strict = self.open_left and key.closed_left
            above = self._left < key._left if strict else self._left <= key._left
            strict = self.open_right and key.closed_right
            return above and (key._right < self._right if strict else key._right <= self._right)
        above = self._left < key if self.open_left else self._left <= key
        return above and (key < self._right if self.open_right else key <= self._right)

    def _key(self) -> tuple[Any, Any, str]:
        """The triple an interval is compared and hashed by."""
        return self._left, self._right, self._closed

    def __eq__(self, other: object) -> bool:
        """Equal ends and equal closed."""
        if not isinstance(other, Interval):
            return NotImplemented
        return self._key() == other._key()

    def __ne__(self, other: object) -> bool:
        """Not equal ends, or not equal closed."""
        if not isinstance(other, Interval):
            return NotImplemented
        return self._key() != other._key()

    def __lt__(self, other: Any) -> bool:
        """Ordered by left, then right, then closed."""
        return self._compare(other, operator.lt, "<")

    def __le__(self, other: Any) -> bool:
        """Ordered by left, then right, then closed."""
        return self._compare(other, operator.le, "<=")

    def __gt__(self, other: Any) -> bool:
        """Ordered by left, then right, then closed."""
        return self._compare(other, operator.gt, ">")

    def __ge__(self, other: Any) -> bool:
        """Ordered by left, then right, then closed."""
        return self._compare(other, operator.ge, ">=")

    def _compare(self, other: Any, op: Any, symbol: str) -> bool:
        """One ordering, or pandas' TypeError against anything but an interval."""
        if not isinstance(other, Interval):
            raise TypeError(
                f"'{symbol}' not supported between instances of"
                f" 'pandas._libs.interval.Interval' and '{type(other).__name__}'"
            )
        return op(self._key(), other._key())

    def __hash__(self) -> int:
        """The hash of the triple."""
        return hash(self._key())

    def __add__(self, other: Any) -> Any:
        """Both ends moved by a number or a span."""
        if not _moves(other):
            return NotImplemented
        return Interval(self._left + other, self._right + other, self._closed)

    __radd__ = __add__

    def __sub__(self, other: Any) -> Any:
        """Both ends moved back by a number or a span."""
        if not _moves(other):
            return NotImplemented
        return Interval(self._left - other, self._right - other, self._closed)

    def __mul__(self, other: Any) -> Any:
        """Both ends scaled by a number."""
        if not _scales(other):
            return NotImplemented
        return Interval(self._left * other, self._right * other, self._closed)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Any:
        """Both ends divided by a number."""
        if not _scales(other):
            return NotImplemented
        return Interval(self._left / other, self._right / other, self._closed)

    def __floordiv__(self, other: Any) -> Any:
        """Both ends floor divided by a number."""
        if not _scales(other):
            return NotImplemented
        return Interval(self._left // other, self._right // other, self._closed)

    def __repr__(self) -> str:
        """The ends and closed as pandas writes them, an instant bare and a span as its repr."""
        left, right = (
            str(end) if isinstance(end, Timestamp) else repr(end)
            for end in (self._left, self._right)
        )
        return f"Interval({left}, {right}, closed={self._closed!r})"

    def __str__(self) -> str:
        """The interval in bracket notation, like `(0, 1]`."""
        start = "[" if self.closed_left else "("
        end = "]" if self.closed_right else ")"
        return f"{start}{self._left}, {self._right}{end}"
