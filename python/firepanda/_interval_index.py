"""`IntervalIndex`, pandas' index of intervals.

An `IntervalIndex` is an `Index` whose labels are interval cells, the written
form `_objects` gives an interval column. Everything an index does on its
labels, sorting, uniques, gaps, joins and slicing, works on the cells, since
the ends are written so that they sort as the intervals do. What is here reads
the intervals back: the ends, the midpoints and lengths, lookups by an
interval or by a point one holds, and the repr.

An index made by the core from interval cells becomes an `IntervalIndex` by
`_class_of`, so a slice, a sort or a series' labels stay one. An index that is
empty or all gaps has no cell to carry its type, so only one built here, which
keeps its type in the slot a `DatetimeIndex` keeps its frequency in, is still
an `IntervalIndex` then.
"""

from __future__ import annotations

import contextlib
import itertools
import numbers
from typing import Any

from . import _objects
from ._frame import Index
from ._interval import (
    Interval,
    IntervalDtype,
    _formatted,
    _kind_parts,
    _moment_kind,
    _plain,
    interval_arrow,
    interval_kind,
    interval_pairs,
    moment_subtype,
)
from .errors import InvalidArgumentError

__all__ = ["IntervalIndex"]


class IntervalIndex(Index):
    """An index of intervals, which is `pandas.IntervalIndex`."""

    __slots__ = ()

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

        with contextlib.suppress(AttributeError):
            # Built already, by `Index.__new__` answering a list of intervals with one.
            self._inner  # noqa: B018
            return
        if isinstance(data, IntervalIndex):
            # pandas keeps the name and the type of an index it is handed, and
            # reads only `closed` off the call.
            name = data.name if name is None else name
            dtype = None
        values = [None if is_gap(value) else value for value in data]
        if any(not isinstance(value, Interval) for value in values if value is not None):
            raise TypeError("type <class 'object'> with value is not an interval")
        kind = str(dtype) if dtype is not None else interval_kind(values)
        if kind is None:
            sides = {value.closed for value in values if value is not None}
            if len(sides) > 1:
                raise InvalidArgumentError("intervals must all be closed on the same side")
            subtype = "float64" if values else "int64"
            kind = f"interval[{subtype}, {closed or 'right'}]" if not sides else None
            if kind is None:
                raise NotImplementedError(
                    "IntervalIndex: intervals of instants or spans are not supported yet"
                )
        subtype, side = _kind_parts(kind)
        side = closed or side or "right"
        kind = f"interval[{subtype}, {side}]"
        pairs = interval_pairs(values, kind)
        Index.__init__(self, _objects.interval_cells(pairs, kind), name=name)
        self.__class__ = IntervalIndex
        self._freq = kind

    @property
    def _type(self) -> str:
        """The interval type's name, kept beside the labels or read off the first cell."""
        kind = getattr(self, "_freq", None)
        return kind or _objects.interval_name_of(self._inner) or "interval[float64, right]"

    @property
    def _dtype(self) -> IntervalDtype:
        return IntervalDtype(self._type)

    @property
    def _values(self) -> list[Any]:
        """The intervals, None for a gap."""
        return [None if _objects.is_gap(value) else value for value in Index.tolist(self)]

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
        from ._objects import is_gap

        pairs = [None if is_gap(pair) else tuple(_plain(end) for end in pair) for pair in data]
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

    @property
    def closed_left(self) -> bool:
        """Whether each interval holds its left end."""
        return self.closed in ("left", "both")

    @property
    def closed_right(self) -> bool:
        """Whether each interval holds its right end."""
        return self.closed in ("right", "both")

    @property
    def open_left(self) -> bool:
        """Whether each interval leaves out its left end."""
        return not self.closed_left

    @property
    def open_right(self) -> bool:
        """Whether each interval leaves out its right end."""
        return not self.closed_right

    def _tuples(self, na_tuple: bool) -> list[Any]:
        pairs = zip(self.left.tolist(), self.right.tolist(), strict=True)
        return [
            (left, right) if value is not None or na_tuple else float("nan")
            for value, (left, right) in zip(self._values, pairs, strict=True)
        ]

    def to_tuples(self, na_tuple: bool = True) -> Index:
        """Each interval as a pair of its ends, an index of objects.

        A gap is a pair of NaN, or one NaN with `na_tuple` off.
        """
        return Index(self._tuples(na_tuple), dtype=object, tupleize_cols=False, name=self.name)

    def _ends(self, pick: Any) -> Any:
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
    def is_overlapping(self) -> bool:
        """Whether any two intervals share a point, gaps aside."""
        found = sorted(
            (value for value in self._values if value is not None),
            key=lambda value: (value.left, value.right),
        )
        both = self.closed == "both"
        furthest = None
        for value in found:
            if furthest is not None and (
                value.left < furthest or (value.left == furthest and both)
            ):
                return True
            furthest = value.right if furthest is None else max(furthest, value.right)
        return False

    def overlaps(self, other: Any) -> Any:
        """Whether each interval shares a point with the interval `other`, as numpy bools."""
        import numpy

        return numpy.array([value is not None and value.overlaps(other) for value in self._values])

    def set_closed(self, closed: str) -> IntervalIndex:
        """The same ends, each interval holding the ends `closed` names."""
        pairs = [None if value is None else (value.left, value.right) for value in self._values]
        return type(self).from_tuples(pairs, closed=closed, name=self.name)

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

    def __iter__(self) -> Any:
        return iter(self.tolist())

    def __getitem__(self, key: Any) -> Any:
        found = Index.__getitem__(self, key)
        if isinstance(found, Index):
            found.__class__ = IntervalIndex
            found._freq = self._type
            return found
        return float("nan") if _objects.is_gap(found) else found

    def _cell(self, key: Any) -> Any:
        """An interval as the cell it is held as here."""
        return _objects.interval_cells(interval_pairs([key], self._type), self._type)[0]

    def get_loc(self, key: Any) -> Any:
        """Where an interval is, or where the intervals holding a point are.

        One hit is its position. More are a slice when they sit together and a
        mask of flags when they do not, as pandas answers them.

        Raises:
            KeyError: For an interval that is not there, or a point no interval holds.
        """
        if isinstance(key, Interval):
            if key.closed != self.closed:
                raise KeyError(key)
            try:
                return Index.get_loc(self, self._cell(key))
            except (KeyError, TypeError, ValueError):
                raise KeyError(key) from None
        try:
            hits = [value is not None and key in value for value in self._values]
        except TypeError:
            raise KeyError(key) from None
        places = [at for at, hit in enumerate(hits) if hit]
        if not places:
            raise KeyError(key)
        if len(places) == 1:
            return places[0]
        if places[-1] - places[0] + 1 == len(places):
            return slice(places[0], places[-1] + 1)
        import numpy

        return numpy.array(hits)

    def __contains__(self, key: Any) -> bool:
        if _objects.is_gap(key):
            return None in self._values
        if not isinstance(key, Interval):
            return False
        try:
            Index.get_loc(self, self._cell(key))
        except (KeyError, TypeError, ValueError):
            return False
        return key.closed == self.closed

    def to_numpy(self, dtype: Any = None, copy: bool = False, na_value: Any = None) -> Any:
        """The intervals as a numpy array of objects, NaN for a gap."""
        import numpy

        found = numpy.empty(len(self), dtype=object)
        found[:] = self.tolist()
        return found if dtype is None else found.astype(dtype)

    @property
    def values(self) -> Any:
        """The intervals as pandas' `IntervalArray`."""
        return self.to_series().array

    @property
    def array(self) -> Any:
        """The intervals as pandas' `IntervalArray`."""
        return self.to_series().array

    def equals(self, other: Any) -> bool:
        """Whether another index of intervals holds the same intervals in the same type."""
        return (
            isinstance(other, IntervalIndex)
            and self._dtype == other._dtype
            and self._values == other._values
        )

    __hash__ = None  # type: ignore[assignment]

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

    __str__ = __repr__
