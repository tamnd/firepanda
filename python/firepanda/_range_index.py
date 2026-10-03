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

A frame's default labels are a `RangeIndex` too, the core holding them as a
range. Methods that answer whole numbers in an even step, slices, `take`,
`delete`, `insert`, `append`, the set operations, sorting, arithmetic by a
whole number and the like, answer a `RangeIndex` as pandas' do, and anything
else answers a plain `Index`.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

from . import _qualnames
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


def _as_range(values: list[int], step: int) -> range | None:
    """The range whole numbers in an even step stand for, as pandas finds it, or None.

    No numbers are the empty range from zero, and one number is a range of one
    stepping by `step`, the step of the range it came from.
    """
    if not values:
        return range(0)
    if len(values) == 1:
        return range(values[0], values[0] + step, step)
    diff = values[1] - values[0]
    if diff == 0:
        return None
    if all(after - before == diff for before, after in pairwise(values)):
        return range(values[0], values[-1] + diff, diff)
    return None


def _ranged(answer: Any, step: int) -> Any:
    """An index answer as a `RangeIndex` when its labels are an even step, as pandas keeps one.

    Anything that is not whole numbers, or not in an even step, is a plain index.
    """
    if not isinstance(answer, Index) or type(answer).__name__ == "MultiIndex":
        return answer
    if answer._inner.is_range():
        return Index._wrap(answer._inner)
    if str(answer.dtype) != "int64":
        return answer if type(answer) is not RangeIndex else Index._wrap(answer._inner)
    found = _as_range(answer.tolist(), step)
    if found is None:
        return answer if type(answer) is not RangeIndex else Index._wrap(answer._inner)
    return RangeIndex(found, name=answer.name)


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
        return self._range_of().start

    @property
    def stop(self) -> int:
        """The number the labels stop before."""
        return self._range_of().stop

    @property
    def step(self) -> int:
        """The distance between labels."""
        return self._range_of().step

    def __repr__(self) -> str:
        """The three numbers and the name, the way pandas writes them."""
        named = "" if self.name is None else f", name={self.name!r}"
        return f"RangeIndex(start={self.start}, stop={self.stop}, step={self.step}{named})"

    __str__ = __repr__

    def __getitem__(self, key: Any) -> Any:
        """A label, or the labels picked, a slice of the range staying a range as in pandas."""
        if isinstance(key, slice):
            return RangeIndex(self._range_of()[key], name=self.name)
        return _ranged(super().__getitem__(key), self.step)

    def sort_values(self, *args: Any, **kwargs: Any) -> Any:
        """The labels in order, a range either way round, with the positions when asked."""
        answer = super().sort_values(*args, **kwargs)
        if isinstance(answer, tuple):
            return (_ranged(answer[0], self.step), *answer[1:])
        return _ranged(answer, self.step)

    def append(self, other: Any) -> Any:
        """The labels with `other`'s after them, keeping this name past a plain index.

        pandas makes the answer from this range whenever anything appended is not
        a range, so it keeps this range's name whatever the other names are.
        """
        answer = _ranged(Index.append(self, other), self.step)
        others = other if isinstance(other, (list, tuple)) else [other]
        if all(isinstance(one, RangeIndex) for one in others):
            return answer
        return answer.set_names(self.name) if str(answer.dtype) == "int64" else answer

    def symmetric_difference(self, other: Any, result_name: Any = None, sort: Any = None) -> Any:
        """The labels in one or the other, keeping this name past a plain index as pandas does."""
        answer = _ranged(Index.symmetric_difference(self, other, result_name, sort), self.step)
        if result_name is not None or (isinstance(other, RangeIndex) and sort is None):
            return answer
        return answer.set_names(self.name) if str(answer.dtype) == "int64" else answer

    def intersection(self, other: Any, sort: Any = False) -> Any:
        """The labels in both, worked out from the two ranges when both are ranges, as pandas does.

        Two ranges meet in a range whose stop is the nearer of the two stops,
        which is how pandas writes it, rather than one step past the last label.
        """
        if not isinstance(other, RangeIndex):
            return _ranged(Index.intersection(self, other, sort=sort), self.step)
        found = _ranges_met(self._range_of(), other._range_of())
        name = self.name if self.name == other.name else None
        return RangeIndex(found, name=name)

    def repeat(self, *args: Any, **kwargs: Any) -> Any:
        """Each label repeated, a plain index as pandas answers even for one repeat."""
        answer = Index.repeat(self, *args, **kwargs)
        return Index._wrap(answer._inner) if type(answer) is RangeIndex else answer

    def min(self, *args: Any, **kwargs: Any) -> Any:
        """The smallest label, a Python int as pandas answers for a range."""
        return _scalar(super().min(*args, **kwargs))

    def max(self, *args: Any, **kwargs: Any) -> Any:
        """The largest label, a Python int as pandas answers for a range."""
        return _scalar(super().max(*args, **kwargs))

    def argmin(self, *args: Any, **kwargs: Any) -> Any:
        """Where the smallest label is, a Python int as pandas answers for a range."""
        return _scalar(super().argmin(*args, **kwargs))

    def argmax(self, *args: Any, **kwargs: Any) -> Any:
        """Where the largest label is, a Python int as pandas answers for a range."""
        return _scalar(super().argmax(*args, **kwargs))

    def all(self, *args: Any, **kwargs: Any) -> bool:
        """Whether no label is zero. pandas' range takes numpy's keywords here unread."""
        return 0 not in self._range_of()

    def any(self, *args: Any, **kwargs: Any) -> bool:
        """Whether some label is not zero, taking numpy's keywords unread as `all` does."""
        return any(self._range_of())


def _ranges_met(mine: range, theirs: range) -> range:
    """Where two ranges meet, by pandas' own arithmetic for `RangeIndex.intersection`."""
    first = mine[::-1] if mine.step < 0 else mine
    second = theirs[::-1] if theirs.step < 0 else theirs
    low = max(first.start, second.start)
    high = min(first.stop, second.stop)
    if high <= low:
        return range(0)
    gcd, factor = _extended_gcd(first.step, second.step)
    if (first.start - second.start) % gcd:
        return range(0)
    start = first.start + (second.start - first.start) * first.step // gcd * factor
    step = first.step * second.step // gcd
    start += step * -(-(low - start) // step)
    found = range(start, high, step)
    if (mine.step < 0 and theirs.step < 0) is not (found.step < 0):
        found = found[::-1]
    return found


def _extended_gcd(a: int, b: int) -> tuple[int, int]:
    """The greatest common divisor of two steps and the factor on `a` that makes it."""
    s, old_s = 0, 1
    r, old_r = b, a
    while r:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
    return old_r, old_s


def _scalar(value: Any) -> Any:
    """A numpy number as the Python one, which is what pandas answers for a range."""
    return value.item() if hasattr(value, "item") and not hasattr(value, "index") else value


def _kept(name: str) -> Any:
    """`Index.<name>` on a range, its answer kept a range when its labels are an even step."""
    # A copy, so a wrong call names the method as pandas' range does.
    shown = _qualnames._INDEX_OWN["RangeIndex"].get(name) or f"Index.{name}"
    method = _qualnames._copied(getattr(Index, name), shown)

    def kept(self: RangeIndex, *args: Any, **kwargs: Any) -> Any:
        return _ranged(method(self, *args, **kwargs), self.step)

    kept.__name__ = name
    kept.__doc__ = method.__doc__
    return kept


def _by_whole(name: str) -> Any:
    """Arithmetic `name` on a range, kept a range against a whole number, as pandas does."""
    method = getattr(Index, name)

    def kept(self: RangeIndex, other: Any = None, *args: Any) -> Any:
        answer = method(self, *args) if other is None else method(self, other, *args)
        if other is None or (isinstance(other, int) and not isinstance(other, bool)):
            return _ranged(answer, self.step)
        return answer if type(answer) is not RangeIndex else Index._wrap(answer._inner)

    kept.__name__ = name
    kept.__doc__ = method.__doc__
    return kept


for _name in (
    "take",
    "delete",
    "insert",
    "union",
    "difference",
    "drop",
    "unique",
    "drop_duplicates",
    "where",
    "astype",
    "putmask",
    "fillna",
    "dropna",
    "view",
    "copy",
    "rename",
    "set_names",
):
    if hasattr(Index, _name):
        setattr(RangeIndex, _name, _kept(_name))
for _name in (
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__floordiv__",
    "__neg__",
    "__pos__",
):
    if hasattr(Index, _name):
        setattr(RangeIndex, _name, _by_whole(_name))
