"""The frequency an index of instants or spans passes on, by pandas' rules.

pandas keeps the frequency on a new index when the operation cannot have broken
the step. A take keeps it when the positions make a slice, times the slice's
step, so sorting a daily index backwards answers `-1D`. A mask keeps it when
the labels it keeps are one run. Deleting or inserting keeps it only at the
ends, and inserting only a label one step past the end. A union or an
intersection of an index with itself is that index, and otherwise the answer
takes the frequency its labels keep, the way `infer_freq` reads it. Arithmetic
with a fixed length keeps a fixed step, and scaling a span scales its step.

Each rule is measured against pandas 3.0 and is written once here, for both
`DatetimeIndex` and `TimedeltaIndex`, which put this in front of `Index`.
"""

from __future__ import annotations

import datetime
import itertools
from typing import Any

__all__: list[str] = []


def _step_of(positions: Any, size: int) -> int | None:
    """The step of the slice the positions make, or None when they make none.

    This is numpy's `maybe_indices_to_slice` under pandas: every position has
    to be in range and not negative, and the steps have to be one nonzero step.
    """
    got = [int(one) for one in positions]
    if any(one < 0 or one >= size for one in got):
        return None
    if len(got) < 2:
        return 1
    step = got[1] - got[0]
    if step == 0 or any(after - before != step for before, after in itertools.pairwise(got)):
        return None
    return step


def _one_run(mask: list[bool]) -> bool:
    """Whether the kept labels sit together, which is a mask pandas reads as a slice."""
    kept = [at for at, one in enumerate(mask) if one]
    return not kept or kept[-1] - kept[0] + 1 == len(kept)


def _fixed(freq: Any) -> bool:
    """Whether the step is a fixed length or a day, which moving by a length keeps."""
    from . import offsets

    return isinstance(freq, (offsets.Tick, offsets.Day))


def _scaled(freq: Any, factor: Any) -> Any:
    """The step times a number, or None when no one step is that long."""
    from . import offsets

    if factor == 0:
        return None
    if isinstance(factor, int) and not isinstance(factor, bool):
        return freq if factor == 1 else freq * factor
    try:
        return offsets._tick_of(freq.nanos * factor)
    except (ValueError, TypeError, AttributeError):
        return None


def _inferred(index: Any) -> Any:
    """The step the labels keep, as an offset, or None for fewer than three."""
    from ._frequency import _offset_of

    if len(index) < 3:
        return None
    try:
        return _offset_of(index.inferred_freq)
    except (ValueError, TypeError):
        return None


def _is_span(value: Any) -> bool:
    from . import offsets

    return (
        isinstance(value, datetime.timedelta)
        or type(value).__name__ == "timedelta64"
        or isinstance(value, (offsets.Tick, offsets.Day))
    )


def _is_instant(value: Any) -> bool:
    return isinstance(value, datetime.datetime) or type(value).__name__ == "datetime64"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


_SCALE = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}
"""Nanoseconds in one stored step of each unit."""


def freq_shown(labels: Any, freq: Any) -> Any:
    """The frequency labels show, given one the frame or column around them carried.

    That frequency while the labels still step by it, a whole multiple of it
    when they step by that, as taking every other row does, and None otherwise.
    A fixed step is read off the gaps between the stored numbers in one pass,
    which is cheap at any length. Any other offset is checked the way the index
    constructor checks it, and only up to a length where that is quick.
    """
    from . import offsets
    from ._frequency import _conforming

    count = len(labels)
    if count < 2:
        return freq
    if labels.hasnans:
        return None
    fixed = isinstance(freq, offsets.Tick) or (
        isinstance(freq, offsets.Day) and getattr(labels, "tz", None) is None
    )
    if fixed:
        scale = _SCALE.get(labels.unit)
        if scale is None or freq.nanos % scale:
            return None
        step = freq.nanos // scale
        gaps = labels.to_series().astype("int64").diff().iloc[1:]
        first = gaps.iloc[0]
        if first == 0 or first % step or not bool((gaps == first).all()):
            return None
        return freq if first == step else freq * int(first // step)
    if count > 100_000:
        return None
    try:
        _conforming(labels, freq)
    except ValueError:
        return None
    return freq


class HeldFreq:
    """The operations of an index of instants or spans that pass the frequency on."""

    __slots__ = ()

    def _holding(self, made: Any, freq: Any) -> Any:
        """The new index, holding a frequency when it is one of instants or spans."""
        if isinstance(made, HeldFreq):
            made._freq = freq
        return made

    def _taken(self, made: Any, positions: Any) -> Any:
        freq = self.freq
        if freq is None:
            return made
        step = _step_of(positions, len(self))
        return self._holding(made, None if step is None else _scaled(freq, step))

    def unique(self, level: Any = None) -> Any:
        return self._holding(super().unique(level), self.freq)

    def drop_duplicates(self, *, keep: Any = "first") -> Any:
        return self._holding(super().drop_duplicates(keep=keep), self.freq)

    def take(
        self,
        indices: Any,
        axis: Any = 0,
        allow_fill: bool = True,
        fill_value: Any = None,
        **kwargs: Any,
    ) -> Any:
        made = super().take(indices, axis, allow_fill, fill_value, **kwargs)
        return self._taken(made, list(indices))

    def __getitem__(self, key: Any) -> Any:
        made = super().__getitem__(key)
        if self.freq is None or isinstance(key, (slice, str, int)):
            return made
        try:
            mask = list(key)
        except TypeError:
            return made
        flags = len(mask) == len(self) and all(
            type(one).__name__ in ("bool", "bool_") for one in mask
        )
        if flags and _one_run([bool(one) for one in mask]):
            return self._holding(made, self.freq)
        return made

    def sort_values(
        self,
        *,
        return_indexer: bool = False,
        ascending: bool = True,
        na_position: str = "last",
        key: Any = None,
    ) -> Any:
        made, order = super().sort_values(
            return_indexer=True, ascending=ascending, na_position=na_position, key=key
        )
        made = self._taken(made, order)
        return (made, order) if return_indexer else made

    def delete(self, loc: Any) -> Any:
        wanted = list(range(len(self)))[loc] if isinstance(loc, slice) else loc
        made = super().delete(wanted)
        freq, size = self.freq, len(self)
        if freq is None:
            return made
        if isinstance(loc, int) and not isinstance(loc, bool):
            kept = loc in (0, -size, -1, size - 1)
        elif isinstance(loc, slice):
            kept = loc.step in (1, None) and (loc.start in (0, None) or loc.stop in (size, None))
        else:
            positions = [int(one) for one in loc]
            if _step_of(positions, size) != 1:
                kept = False
            else:
                kept = not positions or positions[0] == 0 or positions[-1] + 1 == size
        return self._holding(made, freq if kept else None)

    def insert(self, loc: int, item: Any) -> Any:
        """The index with one label put in at a position, read as pandas reads it."""
        from ._pandas import _temporal_insert
        from ._scalars import NaT

        made = _temporal_insert(self, loc, item)
        freq, size = self.freq, len(self)
        if freq is None:
            return made
        kept = False
        if not size:
            label = made[0]
            kept = label is not NaT and (_fixed(freq) or freq.is_on_offset(label))
        elif loc in (0, -size) and made[0] is not NaT:
            kept = made[0] + freq == self[0]
        elif loc == size and made[size] is not NaT:
            kept = made[size] - freq == self[-1]
        return self._holding(made, freq if kept else None)

    def repeat(self, repeats: Any, axis: None = None) -> Any:
        # A label twice over breaks the step, and pandas drops it even for one repeat.
        return self._holding(super().repeat(repeats, axis), None)

    def union(self, other: Any, sort: bool | None = None) -> Any:
        from ._frequency import _conforming

        made = super().union(other, sort)
        if not isinstance(made, HeldFreq):
            return made
        others = getattr(other, "_freq", None)
        if not len(other) or self.equals(other):
            return self._holding(made, self.freq)
        if not len(self):
            return self._holding(made, others)
        if self.freq is not None and others == self.freq:
            try:
                _conforming(made, self.freq)
            except ValueError:
                pass
            else:
                return self._holding(made, self.freq)
        return self._holding(made, _inferred(made))

    def intersection(self, other: Any, sort: bool = False) -> Any:
        made = super().intersection(other, sort)
        if not isinstance(made, HeldFreq):
            return made
        freq = self.freq
        if self.equals(other):
            return self._holding(made, freq)
        fast = (
            freq is not None
            and type(other) is type(self)
            and other.freq == freq
            and freq.n == 1
            and self.is_monotonic_increasing
        )
        return self._holding(made, freq if fast else _inferred(made))

    def difference(self, other: Any, sort: bool | None = None) -> Any:
        made = super().difference(other, sort)
        freq = self.freq
        if not isinstance(made, HeldFreq) or freq is None:
            return made
        left = set(made.tolist())
        mask = [label in left for label in self.tolist()]
        ordered = sort is False or self.is_monotonic_increasing
        return self._holding(made, freq if ordered and _one_run(mask) else None)

    def symmetric_difference(
        self, other: Any, result_name: Any = None, sort: bool | None = None
    ) -> Any:
        made = super().symmetric_difference(other, result_name, sort)
        if not isinstance(made, HeldFreq) or self.freq is None:
            return made
        return self._holding(made, _inferred(made))

    def where(self, cond: Any, other: Any = None) -> Any:
        made = super().where(cond, other)
        kept = all(bool(one) for one in list(cond))
        return self._holding(made, self.freq if kept else None)

    def putmask(self, mask: Any, value: Any) -> Any:
        made = super().putmask(mask, value)
        kept = not any(bool(one) for one in list(mask))
        return self._holding(made, self.freq if kept else None)

    def fillna(self, value: Any) -> Any:
        # An index with a step holds no missing label, so there is nothing to fill.
        return self._holding(super().fillna(value), self.freq)

    def dropna(self, how: str = "any") -> Any:
        return self._holding(super().dropna(how), self.freq)

    def astype(self, dtype: Any, copy: bool = True) -> Any:
        made = super().astype(dtype, copy)
        kept = type(made) is type(self) and made.dtype == self.dtype
        return self._holding(made, self.freq if kept else None)

    def view(self, cls: Any = None) -> Any:
        return self._holding(super().view(cls), self.freq)

    def append(self, other: Any) -> Any:
        made = super().append(other)
        freq = self.freq
        if freq is None:
            return made
        pieces = [self, *(other if isinstance(other, (list, tuple)) else [other])]
        full = [piece for piece in pieces if len(piece)]
        kept = all(getattr(piece, "_freq", None) == freq for piece in full) and all(
            before[-1] + freq == after[0] for before, after in itertools.pairwise(full)
        )
        return self._holding(made, freq if kept else None)

    def reindex(
        self,
        target: Any,
        method: Any = None,
        level: Any = None,
        limit: Any = None,
        tolerance: Any = None,
    ) -> Any:
        made, indexer = super().reindex(target, method, level, limit, tolerance)
        freq = getattr(target, "_freq", None)
        kept = isinstance(target, HeldFreq)
        return self._holding(made, freq if kept else None), indexer

    def _arithmetic(self, op: str, other: Any) -> Any:
        made = super()._arithmetic(op, other)
        freq = self.freq
        if freq is None or not isinstance(made, HeldFreq):
            return made
        return self._holding(made, self._moved_freq(op, other, freq))

    def _moved_freq(self, op: str, other: Any, freq: Any) -> Any:
        """The step after an operator, which pandas keeps for a scalar that moves or scales."""
        spans = not str(self.dtype).startswith("datetime64")
        flipped = op == "__rsub__"
        kept: Any = None
        if _is_span(other) and op in ("__add__", "__radd__", "__sub__", "__rsub__"):
            kept = freq if spans or _fixed(freq) else None
        elif _is_instant(other):
            # A span plus an instant is an instant, and an instant less one a span.
            added = spans and op in ("__add__", "__radd__")
            taken = not spans and op in ("__sub__", "__rsub__") and _fixed(freq)
            kept = freq if added or taken else None
        elif spans and _is_number(other):
            if op in ("__mul__", "__rmul__"):
                return _scaled(freq, other)
            if op in ("__truediv__", "__floordiv__") and other:
                return _scaled(freq, 1 / other)
        if kept is not None and flipped:
            return -kept
        return kept

    def __neg__(self) -> Any:
        freq = self.freq
        return self._holding(super().__neg__(), None if freq is None else -freq)

    def __pos__(self) -> Any:
        return self._holding(super().__pos__(), self.freq)
