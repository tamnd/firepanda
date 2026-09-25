"""`RangeIndex`, an index of evenly stepped whole numbers.

pandas keeps a `RangeIndex` as its three numbers and never writes the labels
out. firepanda writes them out as an int64 index and keeps the three numbers
beside it, so `start`, `stop`, `step` and the repr read as pandas' do, and every
other method is the index's own and answers a plain `Index`, where pandas keeps
a `RangeIndex` when it can. The rules below were measured against pandas 3.0.

- One number is the stop. Two are the start and the stop. A `range` or another
  `RangeIndex` is taken whole.
- The numbers are whole: a float that is a whole number is read as one, and a
  bool or anything else is refused with pandas' TypeError.
- `dtype` may name a signed integer type, and the labels stay int64 as in pandas.

A frame's default labels are not a `RangeIndex` here, so `isinstance` on them
answers False.
"""

from __future__ import annotations

from typing import Any

from ._frame import Index, Series
from .errors import InvalidArgumentError, translate

__all__ = ["RangeIndex"]

_SIGNED = frozenset({"int8", "int16", "int32", "int64", "int", "i8", "i4", "i2", "i1"})


def _whole(value: Any) -> int:
    """One of the three numbers as an int, or pandas' TypeError."""
    if isinstance(value, bool) or (
        not isinstance(value, (int, float)) and not hasattr(value, "__index__")
    ):
        if isinstance(value, (list, tuple, dict, set)):
            raise TypeError(f"Value needs to be a scalar value, was type {type(value).__name__}")
        raise TypeError(f"Wrong type {type(value)} for value {value}")
    try:
        found = int(value)
    except (TypeError, ValueError, OverflowError):
        raise TypeError(f"Wrong type {type(value)} for value {value}") from None
    if found != value:
        raise TypeError(f"Wrong type {type(value)} for value {value}")
    return found


def _check_dtype(dtype: Any) -> None:
    """Refuses a type that is not a signed integer, in pandas' words."""
    if dtype is None:
        return
    text = getattr(dtype, "name", None) or str(dtype)
    if text not in _SIGNED:
        if text == "float":
            text = "float64"
        raise InvalidArgumentError(
            f"Incorrect `dtype` passed: expected signed integer, received {text}"
        )


class RangeIndex(Index):
    """An index of evenly stepped whole numbers, which is `pandas.RangeIndex`."""

    __slots__ = ("_range",)

    def __init__(
        self,
        start: Any = None,
        stop: Any = None,
        step: Any = None,
        dtype: Any = None,
        copy: bool = False,
        name: Any = None,
    ) -> None:
        """Builds the labels of a range.

        Args:
            start: The first number, or the stop when it is the only one given,
                or a `range` or `RangeIndex` taken whole.
            stop: The number the labels stop before.
            step: The distance between labels, not zero.
            dtype: A signed integer type, or None.
            copy: Ignored, as in pandas, since a range holds no buffer to share.
            name: The level name.

        Raises:
            TypeError: When no number is given, or one is not whole.
            ValueError: For a step of zero or a dtype that is not a signed integer.
        """
        _check_dtype(dtype)
        if isinstance(start, RangeIndex):
            found = start._range
            name = start.name if name is None else name
        elif isinstance(start, range):
            found = start
        else:
            if start is None and stop is None and step is None:
                raise TypeError("RangeIndex(...) must be called with integers")
            first = 0 if start is None else _whole(start)
            if stop is None:
                first, last = 0, first
            else:
                last = _whole(stop)
            stride = 1 if step is None else _whole(step)
            if stride == 0:
                raise InvalidArgumentError("Step must not be zero")
            found = range(first, last, stride)
        self._range = found
        labels = Series(list(found), dtype="int64")
        try:
            self._inner = labels._inner.to_index(None if name is None else str(name))
        except Exception as error:
            raise translate(error) from None

    @classmethod
    def from_range(cls, data: Any, name: Any = None, dtype: Any = None) -> RangeIndex:
        """The labels of a Python `range`.

        Raises:
            TypeError: When data is not a `range`.
        """
        if not isinstance(data, range):
            raise TypeError(
                f"{cls.__name__}(...) must be called with object coercible to a range,"
                f" {data!r} was passed"
            )
        return cls(data, dtype=dtype, name=name)

    @property
    def start(self) -> int:
        """The first number."""
        return self._range.start

    @property
    def stop(self) -> int:
        """The number the labels stop before."""
        return self._range.stop

    @property
    def step(self) -> int:
        """The distance between labels."""
        return self._range.step

    def __repr__(self) -> str:
        """The three numbers and the name, the way pandas writes them."""
        named = "" if self.name is None else f", name={self.name!r}"
        return f"RangeIndex(start={self.start}, stop={self.stop}, step={self.step}{named})"

    __str__ = __repr__
