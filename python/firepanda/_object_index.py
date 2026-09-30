"""An index of objects, the labels pandas holds when no one type fits them all.

The labels are object cells, the written form `_objects` gives an object
column, so `Index(["a", 1])` and an index made from an object column keep
each label as it was written. Everything an index does on its labels as text,
taking rows, uniques, gaps and joins, works on the cells. What is here reads
them back: the type, the kind pandas infers, lookups and comparisons by a
value rather than by its cell, and the class a slice keeps.

An index made by the core from object cells becomes an `ObjectIndex` by
`_class_of`, so a slice, a sort or a series' labels stay one. The class calls
itself `Index`, which is what pandas calls an index of objects.
"""

from __future__ import annotations

from typing import Any

from . import _objects
from ._frame import Index

__all__ = ["ObjectIndex"]


def _cells_for(key: Any, spelling: str) -> list[Any]:
    """The cells a key can be held as, since pandas finds 1 by 1.0 and 1.0 by 1."""
    found = [key]
    if isinstance(key, float) and key.is_integer():
        found.append(int(key))
    elif isinstance(key, int) and not isinstance(key, bool):
        found.append(float(key))
    return [_objects.cell(each, spelling) for each in found]


def _same(value: Any, key: Any) -> bool:
    """Whether a label equals a key the way pandas' comparison of objects has it."""
    if _objects.is_gap(value) or _objects.is_gap(key):
        return False
    try:
        return bool(value == key)
    except Exception:
        return False


class ObjectIndex(Index):
    """An `Index` whose labels are objects."""

    __slots__ = ()

    def _shown_dtype(self) -> Any:
        return "object"

    @property
    def inferred_type(self) -> str:
        """What pandas calls the kind of the labels, read off the labels themselves."""
        from .api.types import infer_dtype

        return infer_dtype(self.tolist(), skipna=False)

    def _spelling(self) -> str:
        return _objects.spelling_of(self._inner) or ""

    def get_loc(self, key: Any) -> Any:
        """Where a label is, found by its value.

        Raises:
            KeyError: For a key that is not there.
        """
        if _objects.is_gap(key):
            return Index.get_loc(self, None)
        for cell in _cells_for(key, self._spelling()):
            try:
                return Index.get_loc(self, cell)
            except (KeyError, TypeError, ValueError):
                continue
        raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        try:
            self.get_loc(key)
        except (KeyError, TypeError, ValueError):
            return False
        return True

    def isin(self, values: Any, level: Any = None) -> Any:
        """Whether each label is one of a set of values."""
        wanted = list(values)
        return [any(_same(label, value) for value in wanted) for label in self.tolist()]

    def sort_values(
        self,
        *,
        return_indexer: bool = False,
        ascending: bool = True,
        na_position: str = "last",
        key: Any = None,
    ) -> Any:
        """The labels in the order Python sorts them, which raises on a mix it cannot.

        Raises:
            TypeError: For labels that cannot be put in order, as pandas raises.
        """
        if key is not None:
            return Index.sort_values(
                self,
                return_indexer=return_indexer,
                ascending=ascending,
                na_position=na_position,
                key=key,
            )
        from ._pandas import _objects_order

        order = _objects_order(self.tolist(), ascending, na_position)
        made = self.take(order)
        return (made, order) if return_indexer else made

    def _compared(self, other: Any, equal: bool) -> Any:
        if isinstance(other, (list, tuple, Index)) and not isinstance(other, str):
            others = list(other)
            if len(others) != len(self):
                raise ValueError("Lengths must match to compare")
            found = [_same(a, b) for a, b in zip(self.tolist(), others, strict=True)]
        else:
            found = [_same(label, other) for label in self.tolist()]
        return found if equal else [not each for each in found]

    def __eq__(self, other: object) -> Any:  # type: ignore[override]
        return self._compared(other, True)

    def __ne__(self, other: object) -> Any:  # type: ignore[override]
        return self._compared(other, False)

    __hash__ = None  # type: ignore[assignment]


ObjectIndex.__name__ = "Index"
ObjectIndex.__qualname__ = "Index"
