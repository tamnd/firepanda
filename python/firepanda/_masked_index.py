"""An index of a masked type, the labels of `Index([1, None], dtype="Int64")`.

The labels are the cells `_objects` writes for a masked column, so an index
made from an `Int64`, `Float64`, `boolean` or `string` column keeps its type
and its `NA`. Taking rows, uniques and joins work on the cells as they do for
any text labels. What is here reads them back: the type, lookups by a value
rather than by its cell, comparisons that answer `NA` beside a gap, and the
order of the values rather than of their cells.

An index made by the core from masked cells becomes a `MaskedIndex` by
`_class_of`, and the class calls itself `Index`, which is what pandas calls it.
"""

from __future__ import annotations

from typing import Any

from . import _masked, _objects
from ._frame import Index

__all__ = ["MaskedIndex"]


class MaskedIndex(Index):
    """An `Index` whose labels are of a masked type."""

    __slots__ = ()

    def _type(self) -> str:
        return _objects.masked_name_of(self._inner) or "Int64"

    def _shown_dtype(self) -> Any:
        return _masked.masked_dtype(self._type())

    def _cells(self, key: Any) -> list[Any]:
        """The cells a key can be held as, so 3.0 finds 3 and 3 finds 3.0."""
        name = self._type()
        lower = _masked._LOWER[name]
        found = []
        if lower == "str":
            if isinstance(key, str):
                found.append(key)
        elif lower == "bool":
            if isinstance(key, bool):
                found.append(key)
        elif isinstance(key, (int, float)) and not isinstance(key, bool):
            if lower.startswith("float"):
                found.append(float(key))
            elif float(key).is_integer():
                found.append(int(key))
        return _objects.masked_cells(found, name)

    def get_loc(self, key: Any) -> Any:
        """Where a label is, found by its value.

        Raises:
            KeyError: For a key that is not there.
        """
        if _objects.is_gap(key) or type(key).__name__ == "NAType":
            return Index.get_loc(self, None)
        for cell in self._cells(key):
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

    def _column(self) -> Any:
        from ._frame import Series

        return Series(list(self._inner.to_list()), dtype="str", name=self.name)

    def isin(self, values: Any, level: Any = None) -> Any:
        """Whether each label is one of a set of values, as a `BooleanArray` like pandas'."""
        from ._frame import Series

        wanted = list(values)
        flags = [bool(value in wanted) if value is not None else False for value in self._held()]
        return Series(flags, dtype="boolean").array

    def _held(self) -> list[Any]:
        from ._pandas import _held_values

        return _held_values(self._inner)

    def _compared(self, other: Any, op: str) -> Any:
        column = self._column()
        if isinstance(other, Index):
            other = other.tolist()
        return getattr(column, op)(other).array

    def __eq__(self, other: object) -> Any:  # type: ignore[override]
        return self._compared(other, "__eq__")

    def __ne__(self, other: object) -> Any:  # type: ignore[override]
        return self._compared(other, "__ne__")

    __hash__ = None  # type: ignore[assignment]

    def map(self, mapper: Any, na_action: Any = None) -> Any:
        """Each label through a function or a mapping, the answer in a masked type as pandas has it.

        An answer of the same kind as the labels keeps their type, so an `Int32`
        index stays `Int32`. Numbers or flags of another kind take the masked type
        of their own kind, and text from numbers is objects.
        """
        from ._pandas import _word

        answer = Index.map(self, mapper, na_action)
        source = self._type()
        lower = _masked._LOWER[source]
        kind = _word(answer.dtype)
        family = _family(kind)
        if lower == "str":
            if family != "text":
                return answer
            target = source
        elif family == _family(lower):
            target = source
        elif family in _WIDEST:
            target = _WIDEST[family]
        elif family == "text":
            return Index(answer.tolist(), dtype=object, name=answer.name)
        else:
            return answer
        try:
            return Index(_masked.as_masked(answer.to_series(), target), name=answer.name)
        except (TypeError, ValueError, OverflowError):
            return answer

    def sort_values(
        self,
        *,
        return_indexer: bool = False,
        ascending: bool = True,
        na_position: str = "last",
        key: Any = None,
    ) -> Any:
        """The labels in the order of their values, gaps at one end."""
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


_WIDEST = {"whole": "Int64", "float": "Float64", "flag": "boolean"}
"""The masked type pandas gives an answer of each kind that is not the labels' kind."""


def _family(kind: str) -> str:
    """The kind of a core type: whole numbers, floats, flags, text, or something else."""
    if kind.startswith(("int", "uint")):
        return "whole"
    if kind.startswith("float"):
        return "float"
    if kind == "bool":
        return "flag"
    if kind in ("string", "str"):
        return "text"
    return ""


MaskedIndex.__name__ = "Index"
MaskedIndex.__qualname__ = "Index"
