"""`Interval`, a bounded span of numbers, instants or spans, which is `pandas.Interval`.

A pure Python scalar: two ends and which of them the interval holds. The rules
below were measured against pandas 3.0.

- The ends are numbers, both `Timestamp` or both `Timedelta`, and the left is
  at most the right, so a NaN end is refused as pandas refuses it.
- Two intervals are equal, ordered and hashed as the triple of left, right and
  closed.
- Adding or taking away a number or a span moves both ends, and multiplying
  or dividing by a number scales both. Two intervals do not add.

A list of intervals of numbers that share `closed` is an interval column,
held as written cells the way document 102 describes, and `IntervalDtype` and
`IntervalIndex` are pandas' names for its type and for an index of intervals,
and `interval_range` builds one of evenly spaced breaks. Intervals of instants
or spans are held the same way, their type naming the unit, and the zone of
the instants when they have one.
"""

from __future__ import annotations

import datetime
import itertools
import numbers
import operator
from typing import Any

from ._scalars import Timedelta, Timestamp
from .errors import InvalidArgumentError

__all__ = ["Interval", "IntervalDtype", "IntervalIndex", "interval_range"]

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

    def __format__(self, spec: str) -> str:
        """The bracket notation, padded as a format spec asks."""
        return format(str(self), spec)


def _number(value: Any) -> bool:
    """Whether an end is a number an interval column holds, a bool not being one."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


_UNITS = ("s", "ms", "us", "ns")


def moment_subtype(ends: Any) -> str | None:
    """The subtype of intervals whose ends are all instants or all spans, or None.

    The unit is the finest any end has, and instants carry their zone, which all of
    them have to share.
    """
    if not ends:
        return None
    if all(isinstance(end, Timedelta) for end in ends):
        unit = max((end.unit for end in ends), key=_UNITS.index)
        return f"timedelta64[{unit}]"
    if not all(isinstance(end, Timestamp) for end in ends):
        return None
    zones = {None if end.tz is None else str(end.tz) for end in ends}
    if len(zones) > 1:
        return None
    unit = max((end.unit for end in ends), key=_UNITS.index)
    zone = zones.pop()
    return f"datetime64[{unit}]" if zone is None else f"datetime64[{unit}, {zone}]"


def _moment_kind(kind: str) -> bool:
    """Whether an interval type's ends are instants or spans."""
    return kind.startswith(("interval[datetime", "interval[timedelta"))


def interval_kind(values: Any) -> str | None:
    """The interval type a list of values makes, or None when it makes an object column.

    Every value that is not a gap has to be an interval of numbers, of instants or
    of spans, and all of them closed on the same side. The ends of numbers are whole
    numbers or floats, and one float anywhere makes every end a float, as pandas
    holds them.
    """
    from ._objects import is_gap

    found = [value for value in values if not is_gap(value)]
    if not found or not all(isinstance(value, Interval) for value in found):
        return None
    sides = {value.closed for value in found}
    ends = [end for value in found for end in (value.left, value.right)]
    if len(sides) > 1:
        return None
    moment = moment_subtype(ends)
    if moment is not None:
        return f"interval[{moment}, {sides.pop()}]"
    if not all(_number(end) for end in ends):
        return None
    whole = len(found) == len(values) and all(isinstance(end, numbers.Integral) for end in ends)
    return f"interval[{'int64' if whole else 'float64'}, {sides.pop()}]"


def interval_pairs(values: Any, kind: str) -> list[Any]:
    """The ends of each interval as the type holds them, None for a gap."""
    from ._objects import is_gap

    if _moment_kind(kind):
        return [None if is_gap(value) else (value.left, value.right) for value in values]
    cast = int if kind.startswith("interval[int") else float
    return [None if is_gap(value) else (cast(value.left), cast(value.right)) for value in values]


def _arrow_ends(subtype: str | None) -> Any:
    """The Arrow type of an interval type's ends."""
    import pyarrow as pa

    if subtype is None or subtype.startswith("float"):
        return pa.float64()
    if subtype == "int64":
        return pa.int64()
    unit, _, zone = subtype[subtype.find("[") + 1 : -1].partition(", ")
    if subtype.startswith("timedelta"):
        return pa.duration(unit)
    return pa.timestamp(unit, tz=zone or None)


def _ticks(end: Any, unit: str) -> int:
    """An instant or a span as a whole count of its unit since the epoch or from nothing."""
    return end.value // 1000 ** (3 - _UNITS.index(unit))


def _kind_parts(kind: str) -> tuple[str | None, str | None]:
    """The subtype and the side an interval type's text names, either None when absent."""
    inside = kind[len("interval[") : -1] if kind.startswith("interval[") else ""
    if not inside:
        return None, None
    subtype, _, closed = inside.rpartition(", ")
    if closed not in _CLOSED:
        # No side named, and a zone's comma is not one: `interval[datetime64[us, UTC]]`.
        return inside, None
    return subtype, closed


_ARROW_TYPE: list[Any] = []


def _arrow_type(ends: Any, closed: str) -> Any:
    """pandas' Arrow type for intervals, `pandas.interval`, without registering it.

    pandas registers the name when its Arrow types are first imported, and a second
    registration would fail, so the type is only built here. It crosses the C data
    interface with its name and metadata, and a reader that has pandas' type
    registered rebuilds it. When pandas is loaded its own type is used, which
    registers it, so a reader in the same process sees pandas' type.
    """
    import json
    import sys

    import pyarrow as pa

    if not _ARROW_TYPE and "pandas" in sys.modules:
        from pandas.core.arrays.arrow.extension_types import ArrowIntervalType

        _ARROW_TYPE.append(ArrowIntervalType)
    if not _ARROW_TYPE:

        class ArrowIntervalType(pa.ExtensionType):
            def __init__(self, ends: Any, closed: str) -> None:
                self._closed = closed
                storage = pa.struct([("left", ends), ("right", ends)])
                pa.ExtensionType.__init__(self, storage, "pandas.interval")

            def __arrow_ext_serialize__(self) -> bytes:
                subtype = self.storage_type.field(0).type
                return json.dumps({"subtype": str(subtype), "closed": self._closed}).encode()

            @classmethod
            def __arrow_ext_deserialize__(cls, storage: Any, serialized: bytes) -> Any:
                found = json.loads(serialized.decode())
                return cls(storage.field(0).type, found["closed"])

        _ARROW_TYPE.append(ArrowIntervalType)
    return _ARROW_TYPE[0](ends, closed)


def interval_arrow(values: Any, kind: str) -> Any:
    """Intervals as the Arrow array pandas exports, `pandas.interval` over left and right."""
    import pyarrow as pa

    subtype, closed = _kind_parts(kind)
    ends = _arrow_ends(subtype)
    pairs = interval_pairs(values, kind)
    if _moment_kind(kind):
        unit = ends.unit
        pairs = [
            None if pair is None else tuple(_ticks(end, unit) for end in pair) for pair in pairs
        ]
    gaps = [pair is None for pair in pairs]
    left = pa.array([None if pair is None else pair[0] for pair in pairs], type=ends)
    right = pa.array([None if pair is None else pair[1] for pair in pairs], type=ends)
    mask = pa.array(gaps, type=pa.bool_()) if any(gaps) else None
    storage = pa.StructArray.from_arrays([left, right], names=["left", "right"], mask=mask)
    return pa.ExtensionArray.from_storage(_arrow_type(ends, closed or "right"), storage)


class IntervalDtype(str):
    """pandas' type for a column of intervals, equal to its text, like `interval[int64, right]`."""

    def __new__(cls, subtype: Any = None, closed: str | None = None) -> IntervalDtype:
        text = str(subtype) if subtype is not None else None
        if text is not None and text.startswith("interval"):
            text, found = _kind_parts(text)
            if closed is not None and found is not None and closed != found:
                raise InvalidArgumentError(
                    "'closed' keyword does not match value specified in dtype string"
                )
            closed = closed or found
        if closed is not None and closed not in _CLOSED:
            raise InvalidArgumentError("closed must be one of 'right', 'left', 'both', 'neither'")
        if text is not None and not (text.startswith("datetime64[") and "," in text):
            import numpy

            text = str(numpy.dtype(text))
        name = (
            "interval"
            if text is None
            else f"interval[{text}{'' if closed is None else ', ' + closed}]"
        )
        made = super().__new__(cls, name)
        made._subtype = text
        made._closed = closed
        return made

    @property
    def subtype(self) -> Any:
        """The numpy dtype of the ends, or None, and the type's text for instants with a zone."""
        if self._subtype is None:
            return None
        if "," in self._subtype:
            return self._subtype
        import numpy

        return numpy.dtype(self._subtype)

    @property
    def closed(self) -> str | None:
        """Which ends each interval holds, or None when the type does not say."""
        return self._closed

    @property
    def name(self) -> str:
        """`interval`, whatever the subtype."""
        return "interval"

    @property
    def type(self) -> type:
        """The Python type of one value."""
        return Interval

    @property
    def kind(self) -> str:
        """numpy's letter for the values' kind, which is object."""
        return "O"

    @property
    def na_value(self) -> float:
        """What a gap reads as, which is NaN."""
        return float("nan")

    def __repr__(self) -> str:
        return str.__str__(self)

    def __hash__(self) -> int:
        return str.__hash__(self)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str) and other == "interval":
            return True
        return isinstance(other, str) and str.__eq__(self, str(other)) is True

    def __ne__(self, other: object) -> bool:
        return not self == other


def _formatted(value: Any) -> str:
    """One label of an interval index as pandas prints it, `nan` for a gap."""
    return "nan" if value is None else str(value)


class IntervalIndex:
    """An index of intervals, which is `pandas.IntervalIndex`.

    It holds its intervals as a list and answers the attributes pandas' does. An
    index of intervals under a series is a category index the extension holds,
    and this is what `cat.categories` and the constructors hand back.
    """

    def __init__(
        self,
        data: Any = (),
        closed: str | None = None,
        dtype: Any = None,
        copy: bool = False,
        name: Any = None,
        verify_integrity: bool = True,
    ) -> None:
        from ._objects import is_gap

        values = [None if is_gap(value) else value for value in data]
        if any(not isinstance(value, Interval) for value in values if value is not None):
            raise TypeError("type <class 'object'> with value is not an interval")
        kind = str(dtype) if dtype is not None else interval_kind(values)
        if kind is None:
            sides = {value.closed for value in values if value is not None}
            if len(sides) > 1:
                raise InvalidArgumentError("intervals must all be closed on the same side.")
            kind = f"interval[float64, {closed or 'right'}]" if not sides else None
            if kind is None:
                raise NotImplementedError(
                    "IntervalIndex: intervals of instants or spans are not supported yet"
                )
        subtype, side = _kind_parts(kind)
        side = closed or side or "right"
        kind = f"interval[{subtype}, {side}]"
        pairs = interval_pairs(values, kind)
        self._values = [None if pair is None else Interval(*pair, side) for pair in pairs]
        self._dtype = IntervalDtype(kind)
        self.name = name

    @classmethod
    def from_breaks(
        cls,
        breaks: Any,
        closed: str = "right",
        name: Any = None,
        copy: bool = False,
        dtype: Any = None,
    ) -> IntervalIndex:
        """Intervals between each break and the next."""
        ends = [_plain(end) for end in breaks]
        return cls._from_pairs(list(itertools.pairwise(ends)), closed, name, dtype, ends)

    @classmethod
    def from_arrays(
        cls,
        left: Any,
        right: Any,
        closed: str = "right",
        name: Any = None,
        copy: bool = False,
        dtype: Any = None,
    ) -> IntervalIndex:
        """Intervals from a list of left ends and a list of right ends."""
        lefts, rights = [_plain(end) for end in left], [_plain(end) for end in right]
        if len(lefts) != len(rights):
            raise InvalidArgumentError("left and right must have the same length")
        return cls._from_pairs(
            list(zip(lefts, rights, strict=True)), closed, name, dtype, lefts + rights
        )

    @classmethod
    def from_tuples(
        cls,
        data: Any,
        closed: str = "right",
        name: Any = None,
        copy: bool = False,
        dtype: Any = None,
    ) -> IntervalIndex:
        """Intervals from pairs of ends, None for a gap."""
        pairs = [None if pair is None else tuple(_plain(end) for end in pair) for pair in data]
        ends = [end for pair in pairs if pair is not None for end in pair]
        return cls._from_pairs(pairs, closed, name, dtype, ends + [None] * (None in pairs))

    @classmethod
    def _from_pairs(
        cls, pairs: list[Any], closed: str, name: Any, dtype: Any, ends: list[Any]
    ) -> IntervalIndex:
        from ._objects import is_gap

        whole = all(isinstance(end, numbers.Integral) and not isinstance(end, bool) for end in ends)
        gaps = any(pair is None or any(is_gap(end) for end in pair) for pair in pairs)
        kind = str(dtype) if dtype is not None else None
        moment = moment_subtype([end for end in ends if not is_gap(end)])
        if kind is None and moment is not None:
            kind = f"interval[{moment}, {closed}]"
        if kind is None:
            kind = f"interval[{'int64' if whole and not gaps else 'float64'}, {closed}]"
        values = [
            None if pair is None or any(is_gap(end) for end in pair) else Interval(*pair, closed)
            for pair in pairs
        ]
        return cls(values, closed=closed, dtype=kind, name=name)

    @property
    def dtype(self) -> IntervalDtype:
        """The interval type."""
        return self._dtype

    @property
    def closed(self) -> str:
        """Which ends each interval holds."""
        return str(self._dtype.closed)

    def _ends(self, pick: Any) -> Any:
        from ._frame import Index

        found = [None if value is None else pick(value) for value in self._values]
        if _moment_kind(str(self._dtype)):
            from ._scalars import NaT

            return Index([NaT if end is None else end for end in found])
        if None in found or str(self._dtype.subtype).startswith("float"):
            return Index([float("nan") if end is None else float(end) for end in found])
        return Index(found)

    @property
    def left(self) -> Any:
        """The left ends, as an index."""
        return self._ends(lambda value: value.left)

    @property
    def right(self) -> Any:
        """The right ends, as an index."""
        return self._ends(lambda value: value.right)

    @property
    def mid(self) -> Any:
        """The midpoints, as an index of floats, or of instants or spans."""
        from ._frame import Index

        if _moment_kind(str(self._dtype)):
            return self._ends(lambda value: value.mid)
        return Index(
            [float("nan") if value is None else float(value.mid) for value in self._values]
        )

    @property
    def length(self) -> Any:
        """Each interval's length, as an index."""
        return self._ends(lambda value: value.length)

    @property
    def is_non_overlapping_monotonic(self) -> bool:
        """Whether the intervals increase and none overlaps the next."""
        found = [value for value in self._values if value is not None]
        if len(found) < len(self._values):
            return False
        both = self.closed == "both"
        rising = all(
            a.right < b.left or (a.right == b.left and not both)
            for a, b in itertools.pairwise(found)
        )
        falling = all(
            b.right < a.left or (b.right == a.left and not both)
            for a, b in itertools.pairwise(found)
        )
        return rising or falling

    @property
    def is_empty(self) -> Any:
        """Whether each interval holds no point."""
        import numpy

        return numpy.array([value is not None and value.is_empty for value in self._values])

    def contains(self, other: Any) -> Any:
        """Whether each interval holds a point."""
        import numpy

        return numpy.array([value is not None and other in value for value in self._values])

    def tolist(self) -> list[Any]:
        """The intervals, NaN for a gap."""
        return [float("nan") if value is None else value for value in self._values]

    to_list = tolist

    def __arrow_array__(self, type: Any = None) -> Any:
        """The intervals as the Arrow array pandas exports for them."""
        return interval_arrow(self._values, str(self._dtype))

    def to_series(self, index: Any = None, name: Any = None) -> Any:
        """The intervals as an interval column, labelled by themselves unless `index` says."""
        from ._frame import Series

        index = self if index is None else index
        return Series(self, index=index, name=self.name if name is None else name)

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Any:
        return iter(self.tolist())

    @property
    def size(self) -> int:
        """How many intervals there are."""
        return len(self._values)

    @property
    def shape(self) -> tuple[int]:
        """The length, as a tuple."""
        return (len(self._values),)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, slice):
            return IntervalIndex(self._values[key], dtype=self._dtype, name=self.name)
        found = self._values[key]
        return float("nan") if found is None else found

    def equals(self, other: Any) -> bool:
        """Whether another index of intervals holds the same intervals in the same type."""
        return (
            isinstance(other, IntervalIndex)
            and self._dtype == other._dtype
            and self._values == other._values
        )

    def __repr__(self) -> str:
        from ._config import get_option
        from ._pandas import _pprinted, _summary

        width = get_option("display.width") or 80
        most = get_option("display.max_seq_items") or len(self._values)
        body = _summary(self._values, _formatted, True, "IntervalIndex", width, most)
        attrs = [f"dtype='{self._dtype}'"]
        if self.name is not None:
            attrs.append(f"name={_pprinted(self.name)}")
        if len(self._values) > most:
            attrs.append(f"length={len(self._values)}")
        return f"IntervalIndex({body}{', '.join(attrs)})"


def _plain(value: Any) -> Any:
    """A numpy number as the Python number it holds, and a numpy instant or span as ours."""
    name = type(value).__name__
    if name in ("datetime64", "timedelta64") and type(value).__module__ == "numpy":
        return (Timestamp if name == "datetime64" else Timedelta)(value)
    item = getattr(value, "item", None)
    if item is not None and type(value).__module__ == "numpy":
        return item()
    return value


def _moment(value: Any) -> bool:
    """Whether an end is an instant or a span, which pandas ranges over by dates."""
    kinds = (datetime.date, datetime.timedelta, Timestamp, Timedelta)
    return isinstance(value, kinds) or type(value).__name__ in ("datetime64", "timedelta64")


def _moment_range(
    start: Any, end: Any, periods: Any, freq: Any, name: Any, closed: str
) -> IntervalIndex:
    """Intervals between instants or spans evenly spaced, as `date_range` spaces them.

    Raises:
        InvalidArgumentError: For other than three of the four, with pandas' words.
        TypeError: For ends of two kinds.
    """
    from ._date_range import date_range
    from ._timedelta import timedelta_range

    if sum(value is not None for value in (start, end, periods, freq)) != 3:
        raise InvalidArgumentError(
            "Of the four parameters: start, end, periods, and freq, exactly three must be specified"
        )
    ends = [_plain(value) for value in (start, end) if value is not None]
    spans = [isinstance(value, datetime.timedelta) for value in ends]
    if not all(_moment(value) for value in ends) or len(set(spans)) > 1:
        raise TypeError("start, end, freq need to be type compatible")
    if periods is not None:
        periods += 1
    build = timedelta_range if spans[0] else date_range
    breaks = build(start=start, end=end, periods=periods, freq=freq)
    return IntervalIndex.from_breaks(breaks, closed=closed, name=name)


def interval_range(
    start: Any = None,
    end: Any = None,
    periods: Any = None,
    freq: Any = None,
    name: Any = None,
    closed: str = "right",
) -> IntervalIndex:
    """Evenly spaced intervals of numbers, instants or spans, which is `pandas.interval_range`.

    Three of `start`, `end`, `periods` and `freq` decide the fourth, and `freq`
    is one when only two of the others are given, or a day for instants and
    spans, whose breaks `date_range` and `timedelta_range` make. The breaks are
    whole numbers when every one of the three given is, as pandas makes them.

    Raises:
        InvalidArgumentError: For other than three of the four, and an end that
            is not a number, with pandas' words.
        TypeError: For a count that is not whole, a `freq` that is not a number,
            and ends of two kinds.
    """
    import numpy

    endpoint = start if start is not None else end
    moment = _moment(start) or _moment(end)
    if freq is None and None in (periods, start, end):
        freq = "D" if moment else 1
    if moment:
        return _moment_range(start, end, periods, freq, name, closed)
    if sum(value is not None for value in (start, end, periods, freq)) != 3:
        raise InvalidArgumentError(
            "Of the four parameters: start, end, periods, and freq, exactly three must be specified"
        )
    for side, value in (("start", start), ("end", end)):
        if value is not None and not _number(value):
            raise InvalidArgumentError(f"{side} must be numeric or datetime-like, got {value}")
    if isinstance(periods, float) and periods.is_integer():
        periods = int(periods)
    if periods is not None and not isinstance(periods, numbers.Integral):
        raise TypeError(f"periods must be an integer, got {periods}")
    if freq is not None and not _number(freq):
        raise TypeError("start, end, freq need to be type compatible")
    given = [value for value in (start, end, freq) if value is not None]
    if periods is not None:
        periods += 1
    if start is not None and end is not None and freq is not None:
        breaks = numpy.arange(start, end + (freq * 0.1), freq)
    else:
        if periods is None:
            periods = int((end - start) // freq) + 1
        elif start is None:
            start = end - (periods - 1) * freq
        elif end is None:
            end = start + (periods - 1) * freq
        breaks = numpy.linspace(start, end, periods)
    whole = all(isinstance(value, numbers.Integral) for value in given)
    if whole and _number(endpoint) and numpy.all(breaks == numpy.round(breaks)):
        breaks = breaks.astype("int64")
    return IntervalIndex.from_breaks(breaks, closed=closed, name=name)
