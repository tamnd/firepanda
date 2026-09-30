"""pandas' `FrozenList`, the list `names`, `levels` and `codes` answer.

It is a list that refuses to be written into and prints under its own name,
its items the way pandas' `pprint_thing` writes them: text quoted, and an
index or an array as the list of its values.
"""

from __future__ import annotations

from typing import Any, NoReturn

__all__ = ["FrozenList"]


def _printed(item: Any, top: int) -> str:
    """One item as `pprint_thing` writes it inside a list."""
    if isinstance(item, str):
        return repr(item)
    if isinstance(item, tuple):
        body = ", ".join(_printed(value, top) for value in item)
        return f"({body},)" if len(item) == 1 else f"({body})"
    if isinstance(item, (list, dict)) or hasattr(item, "tolist"):
        values = item.tolist() if hasattr(item, "tolist") else item
        if isinstance(values, list):
            return _sequence(values, top)
    return str(item)


def _sequence(values: list[Any], top: int) -> str:
    """A list as `pprint_thing` writes it, cut after `top` items."""
    body = ", ".join(_printed(value, top) for value in values[:top])
    return f"[{body}{', ...' if len(values) > top else ''}]"


class FrozenList(list):
    """A list that cannot be changed once made, as pandas answers for level names."""

    __slots__ = ()

    def __str__(self) -> str:
        from ._config import get_option

        top = get_option("display.max_seq_items") or len(self)
        return _sequence(list(self), top)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self})"

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(tuple(self))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (tuple, FrozenList)):
            other = list(other)
        return list.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __add__(self, other: Any) -> FrozenList:  # type: ignore[override]
        return type(self)(list(self) + list(other))

    def __radd__(self, other: Any) -> FrozenList:
        return type(self)(list(other) + list(self))

    def __mul__(self, other: Any) -> FrozenList:  # type: ignore[override]
        return type(self)(list(self) * other)

    __rmul__ = __mul__

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, slice):
            return type(self)(list.__getitem__(self, key))
        return list.__getitem__(self, key)

    def __reduce__(self) -> Any:
        return type(self), (list(self),)

    def union(self, other: Any) -> FrozenList:
        """This list and another one after it."""
        return self + other

    def difference(self, other: Any) -> FrozenList:
        """The items not in another list, in this list's order."""
        gone = set(other)
        return type(self)([item for item in self if item not in gone])

    def _refused(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError(f"'{type(self).__name__}' does not support mutable operations.")

    __setitem__ = __delitem__ = __iadd__ = __imul__ = _refused  # type: ignore[assignment]
    append = extend = insert = pop = remove = sort = reverse = clear = _refused  # type: ignore[assignment]
