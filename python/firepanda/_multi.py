"""`MultiIndex`, labels made of several levels, which is `pandas.MultiIndex`.

A firepanda `Index` holds one column of labels and cannot hold tuples, so this
is its own class rather than a kind of `Index`. It keeps what pandas keeps: a
list of levels, each the distinct values of one position of the tuples, and a
list of codes, each saying which value of its level every row holds, with -1
for a gap. The levels are held as Python lists and handed out as `Index`
objects, and the codes as lists of whole numbers.

The rules below were measured against pandas 3.0.

- A level is the distinct values in sorted order, or in the order they are
  first met when they cannot be sorted. A level of whole numbers with a gap
  holds floats, because pandas reads the gap as a NaN.
- A row is a tuple of its values, with a gap read out as NaN.
- A lookup on a sorted index answers a slice, and on an unsorted one a mask,
  unless it matches a whole row exactly once, when it answers the position.
- A set operation or `append` keeps the levels of both sides, even the values
  no row uses any more, which is why `remove_unused_levels` exists.

What is not here is what needs a frame or a column whose rows are labelled by
a `MultiIndex`, or a column that holds tuples: `to_frame` with the labels kept,
`to_series`, `to_flat_index` and `value_counts` are refused by name.
"""

from __future__ import annotations

import bisect
import itertools
from collections.abc import Callable, Iterator
from typing import Any

from ._frame import DataFrame, Index, Series
from ._pandas import NO_DEFAULT, _list_like, _missing
from .errors import InvalidArgumentError

__all__ = ["MultiIndex"]

_GAP = float("nan")


def _gap(value: Any) -> bool:
    """Whether a label is a gap, None or a NaN."""
    try:
        return _missing(value)
    except (TypeError, ValueError):
        return False


def _ordered(values: list[Any]) -> list[Any]:
    """Distinct values sorted, or in the order they were first met when they do not sort."""
    distinct = list(dict.fromkeys(values))
    try:
        return sorted(distinct)
    except TypeError:
        return distinct


def _factorized(values: list[Any]) -> tuple[list[Any], list[int]]:
    """One level and its codes out of a column of values, a gap coded as -1."""
    present = [v for v in values if not _gap(v)]
    level = _ordered(present)
    if len(present) < len(values) and all(
        isinstance(v, int) and not isinstance(v, bool) for v in present
    ):
        level = [float(v) for v in level]
    where = {v: i for i, v in enumerate(level)}
    return level, [-1 if _gap(v) else where[v] for v in values]


def _names_of(given: Any, count: int, fallback: list[Any]) -> list[Any]:
    """The level names a constructor was given, checked against the number of levels."""
    if given is NO_DEFAULT or given is None:
        return fallback
    if not _list_like(given):
        raise InvalidArgumentError("Names should be list-like for a MultiIndex")
    names = list(given)
    if len(names) != count:
        raise InvalidArgumentError("Length of names must match number of levels in MultiIndex.")
    return names


def _array(values: list[Any], dtype: str) -> Any:
    """A list as the numpy array pandas answers with, or the list when numpy is missing."""
    try:
        import numpy
    except ImportError:
        return values
    return numpy.array(values, dtype=dtype)


def _code_type(level: list[Any]) -> str:
    """The smallest whole number type that holds every code into a level, as in pandas."""
    for dtype, top in (("int8", 2**7), ("int16", 2**15), ("int32", 2**31)):
        if len(level) < top:
            return dtype
    return "int64"


def _list(values: Any) -> list[Any]:
    """A list-like as a list, reading an index or column out as its values."""
    if isinstance(values, (Index, Series)):
        return values.tolist()
    return list(values)


class _NoStr:
    """The `str` member, which pandas declares on every index and refuses on this one."""

    def __get__(self, owner: Any, kind: Any = None) -> Any:
        if owner is None:

            def str(data: Any) -> None:
                """The string accessor, which a MultiIndex does not have."""
                raise AttributeError("Can only use .str accessor with Index, not MultiIndex")

            return str
        raise AttributeError("Can only use .str accessor with Index, not MultiIndex")


class MultiIndex:
    """Labels made of several levels, which is `pandas.MultiIndex`."""

    __slots__ = ("_codes", "_levels", "_names")

    _levels: list[list[Any]]
    _codes: list[list[int]]
    _names: list[Any]

    str = _NoStr()

    def __init__(
        self,
        levels: Any = None,
        codes: Any = None,
        sortorder: Any = None,
        names: Any = None,
        copy: bool = False,
        name: Any = None,
        verify_integrity: bool = True,
    ) -> None:
        """Builds an index out of its levels and the codes into them.

        Args:
            levels: One list of distinct values per level.
            codes: One list per level saying which value every row holds, -1 for a gap.
            sortorder: How many levels the rows are sorted by. Taken and not needed,
                because sortedness is read off the rows.
            names: The level names.
            copy: Taken, since the levels and codes are always copied.
            name: Another spelling of `names`.
            verify_integrity: Whether to check the codes against the levels.

        Raises:
            TypeError: When one of levels and codes is missing.
            ValueError: For levels and codes that do not fit together, in pandas' words.
        """
        if levels is None or codes is None:
            raise TypeError("Must pass both levels and codes")
        levels = [_list(level) for level in levels]
        codes = [[int(c) for c in _list(code)] for code in codes]
        if len(levels) != len(codes):
            raise InvalidArgumentError("Length of levels and codes must be the same.")
        if not levels:
            raise InvalidArgumentError("Must pass non-zero number of levels/codes")
        self._levels = levels
        self._codes = codes
        if verify_integrity:
            self._verified()
        self._names = _names_of(names if name is None else name, len(levels), [None] * len(levels))

    @classmethod
    def _built(cls, levels: list[list[Any]], codes: list[list[int]], names: list[Any]) -> Any:
        """An index out of parts already known to fit, without checking them."""
        self = object.__new__(cls)
        self._levels = levels
        self._codes = codes
        self._names = list(names)
        return self

    def _verified(self) -> None:
        """Checks the codes against the levels the way pandas does."""
        lengths = [len(code) for code in self._codes]
        if len(set(lengths)) > 1:
            raise InvalidArgumentError(f"Unequal code lengths: {lengths}")
        for number, (level, code) in enumerate(zip(self._levels, self._codes, strict=True)):
            if code and max(code) >= len(level):
                raise InvalidArgumentError(
                    f"On level {number}, code max ({max(code)}) >= length of level"
                    f" ({len(level)}). NOTE: this index is in an inconsistent state"
                )
            if code and min(code) < -1:
                raise InvalidArgumentError(f"On level {number}, code value ({min(code)}) < -1")
            if len(set(level)) != len(level):
                raise InvalidArgumentError(
                    f"Level values must be unique: {level!r} on level {number}"
                )

    @classmethod
    def _of_columns(cls, columns: list[list[Any]], names: list[Any]) -> Any:
        """An index out of one list of values per level."""
        levels, codes = [], []
        for values in columns:
            level, code = _factorized(values)
            levels.append(level)
            codes.append(code)
        return cls._built(levels, codes, names)

    @classmethod
    def from_arrays(cls, arrays: Any, sortorder: Any = None, names: Any = NO_DEFAULT) -> Any:
        """An index out of one list-like per level, which is `MultiIndex.from_arrays`.

        Raises:
            TypeError: When arrays is not a list of list-likes.
            ValueError: For arrays of different lengths or names that do not fit.
        """
        if not _list_like(arrays) or isinstance(arrays, (Index, Series)):
            raise TypeError("Input must be a list / sequence of array-likes.")
        arrays = list(arrays)
        if not all(_list_like(array) for array in arrays):
            raise TypeError("Input must be a list / sequence of array-likes.")
        fallback = [getattr(array, "name", None) for array in arrays]
        columns = [_list(array) for array in arrays]
        if len({len(column) for column in columns}) > 1:
            raise InvalidArgumentError("all arrays must be same length")
        if not columns:
            raise InvalidArgumentError("Must pass non-zero number of levels/codes")
        return cls._of_columns(columns, _names_of(names, len(columns), fallback))

    @classmethod
    def from_tuples(cls, tuples: Any, sortorder: Any = None, names: Any = None) -> Any:
        """An index out of a list of tuples, which is `MultiIndex.from_tuples`.

        A short tuple is filled out with gaps, as in pandas.

        Raises:
            TypeError: For an empty list with no names, or an item that is not a sequence.
        """
        if not _list_like(tuples):
            raise TypeError("Input must be a list / sequence of tuple-likes.")
        rows = [row if isinstance(row, tuple) else tuple(row[: len(row)]) for row in tuples]
        if not rows:
            if names is None:
                raise TypeError("Cannot infer number of levels from empty list")
            count = len(list(names))
            return cls._built([[] for _ in range(count)], [[] for _ in range(count)], names)
        width = max(len(row) for row in rows)
        columns = [[row[i] if i < len(row) else None for row in rows] for i in range(width)]
        return cls._of_columns(columns, _names_of(names, width, [None] * width))

    @classmethod
    def from_product(cls, iterables: Any, sortorder: Any = None, names: Any = NO_DEFAULT) -> Any:
        """Every combination of the values of several list-likes, which is `from_product`.

        Raises:
            TypeError: When iterables is not a list of list-likes.
        """
        if not _list_like(iterables) or not all(_list_like(each) for each in iterables):
            raise TypeError("Input must be a list / sequence of iterables.")
        fallback = [getattr(each, "name", None) for each in iterables]
        parts = [_factorized(_list(each)) for each in iterables]
        rows = list(itertools.product(*(code for _, code in parts)))
        codes = [[row[i] for row in rows] for i in range(len(parts))]
        return cls._built(
            [level for level, _ in parts], codes, _names_of(names, len(parts), fallback)
        )

    @classmethod
    def from_frame(cls, df: Any, sortorder: Any = None, names: Any = None) -> Any:
        """An index with one level per column of a frame, which is `from_frame`.

        Raises:
            TypeError: When df is not a frame.
        """
        if not isinstance(df, DataFrame):
            raise TypeError("Input must be a DataFrame")
        labels = list(df.columns)
        columns = [df[label].tolist() for label in labels]
        return cls._of_columns(columns, _names_of(names, len(labels), labels))

    # The shape of the thing.

    @property
    def levels(self) -> list[Index]:
        """The distinct values of every level, each an index named after its level."""
        return [
            Index(list(level), name=name)
            for level, name in zip(self._levels, self._names, strict=True)
        ]

    @property
    def codes(self) -> list[Any]:
        """Which value of its level every row holds, one array per level, -1 for a gap."""
        return [_array(list(code), _code_type(level)) for code, level in self._pairs()]

    @property
    def names(self) -> list[Any]:
        """The level names."""
        return list(self._names)

    @names.setter
    def names(self, names: Any) -> None:
        if not _list_like(names) or len(list(names)) != self.nlevels:
            raise InvalidArgumentError("Length of names must match number of levels in MultiIndex.")
        self._names = list(names)

    @property
    def name(self) -> None:
        """None, since the names belong to the levels."""
        return None

    @property
    def nlevels(self) -> int:
        """How many levels."""
        return len(self._levels)

    @property
    def levshape(self) -> tuple[int, ...]:
        """How many distinct values each level holds."""
        return tuple(len(level) for level in self._levels)

    @property
    def ndim(self) -> int:
        """One, since the labels run along one axis."""
        return 1

    @property
    def size(self) -> int:
        """How many rows."""
        return len(self)

    @property
    def shape(self) -> tuple[int]:
        """How many rows, as a tuple one long."""
        return (len(self),)

    @property
    def empty(self) -> bool:
        """Whether there are no rows."""
        return len(self) == 0

    @property
    def dtype(self) -> str:
        """object, since a row is a tuple."""
        return "object"

    @property
    def dtypes(self) -> Series:
        """The type of every level, labelled by the level names."""
        return Series([str(level.dtype) for level in self.levels], index=self._names)

    @property
    def inferred_type(self) -> str:
        """mixed, which is what pandas calls a column of tuples."""
        return "mixed"

    @property
    def values(self) -> Any:
        """The rows as a numpy array of tuples."""
        return self.to_numpy()

    @property
    def array(self) -> Any:
        """Refused as pandas refuses it, since no one array holds the rows."""
        raise InvalidArgumentError(
            "MultiIndex has no single backing array. Use 'MultiIndex.to_numpy()' to get a"
            " NumPy array of tuples."
        )

    @property
    def nbytes(self) -> int:
        """Roughly how many bytes the codes and levels take."""
        return sum(8 * len(code) + 8 * len(level) for code, level in self._pairs())

    def memory_usage(self, deep: bool = False) -> int:
        """Roughly how many bytes the index takes."""
        return self.nbytes

    @property
    def T(self) -> Any:
        """The index itself, since it has one axis."""
        return self

    def transpose(self, *args: Any, **kwargs: Any) -> Any:
        """The index itself, since it has one axis."""
        return self

    def ravel(self, order: str = "C") -> Any:
        """The index itself, since it has one axis."""
        return self

    def view(self, cls: Any = None) -> Any:
        """A copy of the index."""
        return self.copy()

    def _pairs(self) -> Iterator[tuple[list[int], list[Any]]]:
        """Every level's codes with its values."""
        return zip(self._codes, self._levels, strict=True)

    # Reading rows.

    def __len__(self) -> int:
        """How many rows."""
        return len(self._codes[0]) if self._codes else 0

    def _row(self, at: int) -> tuple[Any, ...]:
        """One row as a tuple, a gap as NaN."""
        return tuple(_GAP if code[at] < 0 else level[code[at]] for code, level in self._pairs())

    def _keys(self) -> list[tuple[int, ...]]:
        """Every row as a tuple of its codes."""
        return list(zip(*self._codes, strict=True)) if self._codes else []

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        """The rows as tuples."""
        return (self._row(at) for at in range(len(self)))

    def tolist(self) -> list[tuple[Any, ...]]:
        """The rows as a list of tuples, a gap as NaN."""
        return list(self)

    to_list = tolist

    def to_numpy(
        self, dtype: Any = None, copy: bool = False, na_value: Any = NO_DEFAULT, **kwargs: Any
    ) -> Any:
        """The rows as a numpy array of tuples."""
        import numpy

        rows = numpy.empty(len(self), dtype=object)
        rows[:] = self.tolist()
        return rows

    def item(self) -> tuple[Any, ...]:
        """The one row.

        Raises:
            ValueError: Unless there is exactly one row.
        """
        if len(self) != 1:
            raise InvalidArgumentError("can only convert an array of size 1 to a Python scalar")
        return self._row(0)

    def __repr__(self) -> str:
        """The rows one per line, then the names, as pandas prints them."""
        named = any(name is not None for name in self._names)
        tail = f"names={self._names!r}" if named else ""
        if not len(self):
            return f"MultiIndex([], {tail})"
        rows = [repr(row) for row in self]
        pad = " " * len("MultiIndex([")
        body = f",\n{pad}".join(rows)
        return f"MultiIndex([{body}],\n{' ' * len('MultiIndex(')}{tail})"

    __str__ = __repr__

    def __getitem__(self, key: Any) -> Any:
        """One row as a tuple, or an index of the rows a slice, positions or a mask pick."""
        size = len(self)
        if isinstance(key, int) and not isinstance(key, bool):
            at = key + size if key < 0 else key
            if not 0 <= at < size:
                raise IndexError(f"index {key} is out of bounds for axis 0 with size {size}")
            return self._row(at)
        if isinstance(key, slice):
            return self._taken(list(range(size))[key])
        picks = _list(key)
        if picks and all(isinstance(pick, bool) for pick in picks):
            return self._taken([at for at, keep in enumerate(picks) if keep])
        return self.take(picks)

    def _taken(self, rows: list[int]) -> Any:
        """The rows at some positions, keeping the levels."""
        return self._built(
            self._levels, [[code[at] for at in rows] for code in self._codes], self._names
        )

    def take(
        self,
        indices: Any,
        axis: Any = 0,
        allow_fill: bool = True,
        fill_value: Any = None,
        **kwargs: Any,
    ) -> Any:
        """The rows at some positions, which may count from the end.

        Raises:
            IndexError: For a position past either end.
        """
        size = len(self)
        rows = []
        for pick in _list(indices):
            at = int(pick)
            if not -size <= at < size:
                raise IndexError(f"index {at} is out of bounds for axis 0 with size {size}")
            rows.append(at + size if at < 0 else at)
        return self._taken(rows)

    # Levels by number or name.

    def _level_number(self, level: Any) -> int:
        """A level's position out of its name or number, as pandas reads it.

        Raises:
            KeyError: For a name that is not a level.
            IndexError: For a number past the last level.
            ValueError: For a name several levels share.
        """
        count = self._names.count(level)
        if count > 1 and not isinstance(level, int):
            raise InvalidArgumentError(
                f"The name {level} occurs multiple times, use a level number"
            )
        if count:
            return self._names.index(level)
        if not isinstance(level, int) or isinstance(level, bool):
            raise KeyError(f"Level {level} not found")
        levels = self.nlevels
        if level < 0:
            if level + levels < 0:
                raise IndexError(
                    f"Too many levels: Index has only {levels} levels, {level} is not a valid"
                    " level number"
                )
            return level + levels
        if level >= levels:
            raise IndexError(f"Too many levels: Index has only {levels} levels, not {level + 1}")
        return level

    def _level_numbers(self, level: Any) -> list[int]:
        """Several levels' positions, or one."""
        chosen = level if isinstance(level, (list, tuple)) else [level]
        return [self._level_number(each) for each in chosen]

    def get_level_values(self, level: Any) -> Index:
        """Every row's value on one level, as an index named after the level."""
        number = self._level_number(level)
        values = self._levels[number]
        column = [None if code < 0 else values[code] for code in self._codes[number]]
        return Index(column, name=self._names[number])

    def _chosen(self, numbers: list[int]) -> Any:
        """An index of some levels in a given order."""
        return self._built(
            [self._levels[n] for n in numbers],
            [self._codes[n] for n in numbers],
            [self._names[n] for n in numbers],
        )

    def droplevel(self, level: Any = 0) -> Any:
        """The index without some levels, a plain index when one is left.

        Raises:
            ValueError: When every level would go.
        """
        gone = set(self._level_numbers(level))
        if len(gone) >= self.nlevels:
            raise InvalidArgumentError(
                f"Cannot remove {len(gone)} levels from an index with {self.nlevels} levels:"
                " at least one level must be left."
            )
        kept = [n for n in range(self.nlevels) if n not in gone]
        if len(kept) == 1:
            return self.get_level_values(kept[0])
        return self._chosen(kept)

    def swaplevel(self, i: Any = -2, j: Any = -1) -> Any:
        """The index with two levels traded places."""
        first, second = self._level_number(i), self._level_number(j)
        order = list(range(self.nlevels))
        order[first], order[second] = order[second], order[first]
        return self._chosen(order)

    def reorder_levels(self, order: Any) -> Any:
        """The index with its levels in a given order.

        Raises:
            AssertionError: When the order does not name every level, in pandas' words.
        """
        order = list(order)
        if len(order) != self.nlevels:
            raise AssertionError(
                f"Length of order must be same as number of levels ({self.nlevels}),"
                f" got {len(order)}"
            )
        return self._chosen([self._level_number(each) for each in order])

    def set_names(self, names: Any, *, level: Any = None, inplace: bool = False) -> Any:
        """The index with new level names, or None when done in place.

        Raises:
            TypeError: For one name given for every level.
            ValueError: For a number of names that does not fit.
        """
        renamed = list(self._names)
        if isinstance(names, dict):
            renamed = [names.get(name, name) for name in renamed]
        elif level is None:
            if not _list_like(names):
                raise TypeError("Must pass list-like as `names`.")
            names = list(names)
            if len(names) != self.nlevels:
                raise InvalidArgumentError(
                    "Length of names must match number of levels in MultiIndex."
                )
            renamed = names
        elif isinstance(level, (list, tuple)):
            if not _list_like(names) or len(list(names)) != len(level):
                raise InvalidArgumentError("Length of names must match length of level.")
            for each, name in zip(self._level_numbers(level), names, strict=True):
                renamed[each] = name
        else:
            renamed[self._level_number(level)] = names
        if inplace:
            self._names = renamed
            return None
        return self._built(self._levels, self._codes, renamed)

    def rename(self, names: Any, *, level: Any = None, inplace: bool = False) -> Any:
        """Another spelling of `set_names`."""
        return self.set_names(names, level=level, inplace=inplace)

    def copy(self, names: Any = None, deep: bool = False, name: Any = None) -> Any:
        """A copy, with new names when they are given."""
        given = names if names is not None else name
        chosen = self._names if given is None else _names_of(given, self.nlevels, self._names)
        return self._built([list(v) for v in self._levels], [list(c) for c in self._codes], chosen)

    def set_levels(self, levels: Any, *, level: Any = None, verify_integrity: bool = True) -> Any:
        """The index with some levels' values replaced, the codes kept.

        Raises:
            ValueError: For values that repeat or do not cover the codes.
        """
        values = [list(v) for v in self._levels]
        if level is None:
            given = [_list(each) for each in levels]
            if len(given) != self.nlevels:
                raise InvalidArgumentError("Length of levels must match number of levels.")
            values = given
        elif isinstance(level, (list, tuple)):
            for each, new in zip(self._level_numbers(level), levels, strict=True):
                values[each] = _list(new)
        else:
            values[self._level_number(level)] = _list(levels)
        built = self._built(values, self._codes, self._names)
        if verify_integrity:
            built._verified()
        return built

    def set_codes(self, codes: Any, *, level: Any = None, verify_integrity: bool = True) -> Any:
        """The index with some levels' codes replaced.

        Raises:
            ValueError: For codes that do not fit the levels.
        """
        chosen = [list(code) for code in self._codes]
        if level is None:
            given = [[int(c) for c in _list(each)] for each in codes]
            if len(given) != self.nlevels:
                raise InvalidArgumentError("Length of codes must match number of levels.")
            chosen = given
        elif isinstance(level, (list, tuple)):
            for each, new in zip(self._level_numbers(level), codes, strict=True):
                chosen[each] = [int(c) for c in _list(new)]
        else:
            chosen[self._level_number(level)] = [int(c) for c in _list(codes)]
        built = self._built(self._levels, chosen, self._names)
        if verify_integrity:
            built._verified()
        return built

    def remove_unused_levels(self) -> Any:
        """The index with the values no row holds taken out of its levels."""
        levels, codes = [], []
        for code, level in self._pairs():
            used = sorted({c for c in code if c >= 0})
            where = {old: new for new, old in enumerate(used)}
            levels.append([level[old] for old in used])
            codes.append([where.get(c, -1) for c in code])
        return self._built(levels, codes, self._names)

    # Comparing and sorting.

    def _ranks(self) -> list[list[int]]:
        """Every level's values as their sorted position, so codes compare like values."""
        ranks = []
        for level in self._levels:
            try:
                order = sorted(range(len(level)), key=level.__getitem__)
            except TypeError:
                order = list(range(len(level)))
            rank = [0] * len(level)
            for place, at in enumerate(order):
                rank[at] = place
            ranks.append(rank)
        return ranks

    def _order(self, levels: list[int], ascending: list[bool], gaps_first: bool) -> list[int]:
        """The positions of the rows sorted by some levels, stably."""
        ranks = self._ranks()
        rows = list(range(len(self)))
        for number, up in reversed(list(zip(levels, ascending, strict=True))):
            code, rank = self._codes[number], ranks[number]
            gap = -1 if gaps_first == up else len(rank)

            def key(at: int, code: list[int] = code, rank: list[int] = rank, gap: int = gap) -> int:
                return gap if code[at] < 0 else rank[code[at]]

            rows.sort(key=key, reverse=not up)
        return rows

    def sortlevel(
        self,
        level: Any = 0,
        ascending: Any = True,
        sort_remaining: bool = True,
        na_position: str = "first",
    ) -> tuple[Any, Any]:
        """The rows sorted by some levels then the rest, and the positions they came from.

        Raises:
            ValueError: For a list of directions that does not match the levels.
        """
        numbers = self._level_numbers(level)
        if isinstance(ascending, list):
            if len(ascending) != len(numbers):
                raise InvalidArgumentError("level must have same length as ascending")
            directions = [bool(each) for each in ascending]
            rest = True
        else:
            directions = [bool(ascending)] * len(numbers)
            rest = bool(ascending)
        if sort_remaining:
            others = [n for n in range(self.nlevels) if n not in numbers]
            numbers = numbers + others
            directions = directions + [rest] * len(others)
        rows = self._order(numbers, directions, na_position == "first")
        return self._taken(rows), _array(rows, "int64")

    def argsort(self, *args: Any, na_position: str = "last", **kwargs: Any) -> Any:
        """The positions that sort the rows."""
        return _array(self._ascending(na_position == "first"), "int64")

    def _ascending(self, gaps_first: bool = False) -> list[int]:
        """The positions that sort the rows, as a list."""
        levels = list(range(self.nlevels))
        return self._order(levels, [True] * len(levels), gaps_first)

    def sort_values(
        self,
        *,
        return_indexer: bool = False,
        ascending: bool = True,
        na_position: str = "last",
        key: Callable[..., Any] | None = None,
    ) -> Any:
        """The rows sorted, with the positions they came from when asked."""
        if key is not None:
            raise NotImplementedError(
                "key= is not supported yet on a MultiIndex, because it would call the key on"
                " every level as a column"
            )
        levels = list(range(self.nlevels))
        rows = self._order(levels, [bool(ascending)] * len(levels), na_position == "first")
        ordered = self._taken(rows)
        return (ordered, _array(rows, "int64")) if return_indexer else ordered

    def _sorted(self) -> bool:
        """Whether the rows are in ascending order."""
        return self.is_monotonic_increasing

    @property
    def is_monotonic_increasing(self) -> bool:
        """Whether every row is at or after the one before it."""
        try:
            return all(a <= b for a, b in itertools.pairwise(self))
        except TypeError:
            return False

    @property
    def is_monotonic_decreasing(self) -> bool:
        """Whether every row is at or before the one before it."""
        try:
            return all(a >= b for a, b in itertools.pairwise(self))
        except TypeError:
            return False

    def equals(self, other: object) -> bool:
        """Whether another index holds the same rows in the same order."""
        if not isinstance(other, MultiIndex) or len(other) != len(self):
            return False
        return all(_same(a, b) for a, b in zip(self.tolist(), other.tolist(), strict=True))

    def identical(self, other: Any) -> bool:
        """Whether another index holds the same rows and names."""
        return self.equals(other) and self._names == other._names

    def equal_levels(self, other: Any) -> bool:
        """Whether another index has the same levels holding the same values."""
        return (
            isinstance(other, MultiIndex)
            and self.nlevels == other.nlevels
            and all(a == b for a, b in zip(self._levels, other._levels, strict=True))
        )

    def is_(self, other: Any) -> bool:
        """Whether another object is this very index."""
        return self is other

    def __eq__(self, other: object) -> Any:
        """Whether every row equals the matching row of another index, as a list."""
        if isinstance(other, MultiIndex):
            if len(other) != len(self):
                raise InvalidArgumentError("Lengths must match to compare")
            return [a == b for a, b in zip(self.tolist(), other.tolist(), strict=True)]
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    # Duplicates.

    def duplicated(self, keep: Any = "first") -> Any:
        """Whether every row repeats an earlier one, or a later one, or any other.

        Raises:
            ValueError: For a keep that is not first, last or False.
        """
        if keep not in ("first", "last", False):
            raise InvalidArgumentError('keep must be either "first", "last" or False')
        keys = self._keys()
        if keep is False:
            counts: dict[tuple[int, ...], int] = {}
            for each in keys:
                counts[each] = counts.get(each, 0) + 1
            return _array([counts[each] > 1 for each in keys], "bool")
        seen: set[tuple[int, ...]] = set()
        marks = [False] * len(keys)
        order = range(len(keys)) if keep == "first" else range(len(keys) - 1, -1, -1)
        for at in order:
            marks[at] = keys[at] in seen
            seen.add(keys[at])
        return _array(marks, "bool")

    def drop_duplicates(self, *, keep: Any = "first") -> Any:
        """The index with the repeated rows taken out."""
        marks = self.duplicated(keep=keep)
        return self._taken([at for at, repeat in enumerate(marks) if not repeat])

    def unique(self, level: Any = None) -> Any:
        """The distinct rows in the order first met, or one level's distinct values."""
        if level is not None:
            return self.get_level_values(level).unique()
        return self.drop_duplicates()

    def nunique(self, dropna: bool = True) -> int:
        """How many distinct rows, leaving out rows with a gap unless asked."""
        keys = set(self._keys())
        if dropna:
            keys = {each for each in keys if -1 not in each}
        return len(keys)

    @property
    def is_unique(self) -> bool:
        """Whether no row repeats."""
        return len(set(self._keys())) == len(self)

    @property
    def has_duplicates(self) -> bool:
        """Whether some row repeats."""
        return not self.is_unique

    def factorize(self, sort: bool = False, use_na_sentinel: bool = True) -> tuple[Any, Any]:
        """A code for every row and the distinct rows the codes point into."""
        uniques = self.drop_duplicates()
        if sort:
            uniques = uniques.sort_values()
        where = {each: at for at, each in enumerate(uniques._keys())}
        codes = [where[each] for each in self._keys()]
        return _array(codes, "int64"), uniques.set_names([None] * self.nlevels)

    # Gaps, which pandas refuses to look for on this index.

    def isna(self) -> list[bool]:
        """Refused as pandas refuses it."""
        raise NotImplementedError("isna is not defined for MultiIndex")

    isnull = isna

    def notna(self) -> list[bool]:
        """Refused as pandas refuses it."""
        raise NotImplementedError("isna is not defined for MultiIndex")

    notnull = notna

    @property
    def hasnans(self) -> bool:
        """Refused as pandas refuses it."""
        raise NotImplementedError("isna is not defined for MultiIndex")

    def fillna(self, value: Any) -> Any:
        """Refused as pandas refuses it."""
        raise NotImplementedError("fillna is not defined for MultiIndex")

    def dropna(self, how: str = "any") -> Any:
        """The index without the rows with a gap in any level, or in every level.

        Raises:
            ValueError: For a how that is not any or all.
        """
        if how not in ("any", "all"):
            raise InvalidArgumentError(f"invalid how option: {how}")
        test = any if how == "any" else all
        return self._taken(
            [at for at, each in enumerate(self._keys()) if not test(c < 0 for c in each)]
        )

    # Lookups.

    def _code_of(self, number: int, value: Any) -> int | None:
        """The code of a value on one level, -1 for a gap, None when it is not there."""
        if _gap(value):
            return -1
        try:
            return self._levels[number].index(value)
        except ValueError:
            return None

    def _matches(self, key: tuple[Any, ...]) -> list[int]:
        """The positions of the rows whose first levels hold a key."""
        wanted = [self._code_of(number, value) for number, value in enumerate(key)]
        if None in wanted:
            return []
        codes = self._codes[: len(key)]
        return [
            at
            for at in range(len(self))
            if all(code[at] == want for code, want in zip(codes, wanted, strict=True))
        ]

    def _answer(self, rows: list[int], whole: bool) -> Any:
        """A position, a slice or a mask, the way pandas answers a lookup."""
        if whole and len(rows) == 1:
            return rows[0]
        if self._sorted() and rows == list(range(rows[0], rows[-1] + 1)):
            return slice(rows[0], rows[-1] + 1, None)
        chosen = set(rows)
        return _array([at in chosen for at in range(len(self))], "bool")

    def get_loc(self, key: Any) -> Any:
        """Where a row, or the rows starting with some values, are.

        Raises:
            KeyError: When no row matches, or the key is longer than a row.
        """
        whole = key if isinstance(key, tuple) else (key,)
        if len(whole) > self.nlevels:
            raise KeyError(f"Key length ({len(whole)}) exceeds index depth ({self.nlevels})")
        try:
            rows = self._matches(whole)
        except TypeError:
            rows = []
        if not rows:
            raise KeyError(key)
        return self._answer(rows, len(whole) == self.nlevels)

    def __contains__(self, key: Any) -> bool:
        """Whether some row is, or starts with, a key."""
        try:
            self.get_loc(key)
        except (KeyError, TypeError):
            return False
        return True

    def _level_mask(self, number: int, pick: Any) -> list[bool]:
        """Which rows one piece of a `get_locs` key picks on its level."""
        code = self._codes[number]
        level = self._levels[number]
        if isinstance(pick, slice):
            if pick.start is None and pick.stop is None:
                return [True] * len(self)
            return [
                c >= 0
                and (pick.start is None or level[c] >= pick.start)
                and (pick.stop is None or level[c] <= pick.stop)
                for c in code
            ]
        if _list_like(pick):
            wanted = {self._code_of(number, value) for value in pick}
            return [c in wanted for c in code]
        wanted_code = self._code_of(number, pick)
        if wanted_code is None:
            raise KeyError(pick)
        return [c == wanted_code for c in code]

    def get_locs(self, seq: Any) -> Any:
        """The positions of the rows a key of values, lists and slices picks, level by level."""
        keep = [True] * len(self)
        for number, pick in enumerate(seq):
            keep = [a and b for a, b in zip(keep, self._level_mask(number, pick), strict=True)]
        return _array([at for at, chosen in enumerate(keep) if chosen], "int64")

    def get_loc_level(self, key: Any, level: Any = 0, drop_level: bool = True) -> Any:
        """Where a value of one level is, and the index of the rows it picks.

        Raises:
            KeyError: When no row holds the value.
        """
        if isinstance(level, (list, tuple)):
            raise NotImplementedError(
                "get_loc_level over several levels is not supported yet; look up one level"
            )
        number = self._level_number(level)
        if number == 0:
            found = self.get_loc(key)
            rows = _rows_of(found, len(self))
        else:
            wanted = self._code_of(number, key)
            rows = [at for at, c in enumerate(self._codes[number]) if c == wanted]
            if wanted is None or not rows:
                raise KeyError(key)
            found = self._answer(rows, False)
            if isinstance(found, slice):
                chosen = set(rows)
                found = _array([at in chosen for at in range(len(self))], "bool")
        picked = self._taken(rows)
        if drop_level and self.nlevels > 1:
            gone = list(range(len(key))) if number == 0 and isinstance(key, tuple) else [number]
            if len(gone) < self.nlevels:
                picked = picked.droplevel(gone)
        return found, picked

    def _target_keys(self, target: Any) -> list[tuple[int, ...] | None]:
        """Every row of a target as codes into this index, None for one that cannot match."""
        rows = target.tolist() if isinstance(target, MultiIndex) else _list(target)
        keys: list[tuple[int, ...] | None] = []
        for row in rows:
            if not isinstance(row, tuple) or len(row) != self.nlevels:
                keys.append(None)
                continue
            codes = tuple(self._code_of(n, value) for n, value in enumerate(row))
            keys.append(None if None in codes else codes)  # type: ignore[arg-type]
        return keys

    def get_indexer(
        self, target: Any, method: Any = None, limit: Any = None, tolerance: Any = None
    ) -> list[int]:
        """The position of every target row in this index, -1 where it is not.

        Raises:
            InvalidIndexError: When this index repeats a row.
        """
        from .errors import InvalidIndexError

        if method is not None or limit is not None or tolerance is not None:
            raise NotImplementedError(
                "method=, limit= and tolerance= are not supported yet on a MultiIndex, because"
                " they need an ordering of the rows to fill along"
            )
        if not self.is_unique:
            raise InvalidIndexError("Reindexing only valid with uniquely valued Index objects")
        where = {each: at for at, each in enumerate(self._keys())}
        found = [-1 if each is None else where.get(each, -1) for each in self._target_keys(target)]
        return _array(found, "int64")

    def get_indexer_for(self, target: Any) -> Any:
        """The position of every target row, taking every match when this index repeats."""
        if self.is_unique:
            return self.get_indexer(target)
        return self.get_indexer_non_unique(target)[0]

    def get_indexer_non_unique(self, target: Any) -> tuple[Any, Any]:
        """Every position of every target row, and which target rows were not found."""
        where: dict[tuple[int, ...], list[int]] = {}
        for at, each in enumerate(self._keys()):
            where.setdefault(each, []).append(at)
        found, missing = [], []
        for number, each in enumerate(self._target_keys(target)):
            hits = where.get(each, []) if each is not None else []
            if hits:
                found.extend(hits)
            else:
                found.append(-1)
                missing.append(number)
        return _array(found, "int64"), _array(missing, "int64")

    def isin(self, values: Any, level: Any = None) -> Any:
        """Whether every row, or every row's value on one level, is among some values."""
        if level is not None:
            number = self._level_number(level)
            wanted = {self._code_of(number, value) for value in _list(values)}
            return _array([c in wanted for c in self._codes[number]], "bool")
        wanted_rows = set(self._target_keys(values)) - {None}
        return _array([each in wanted_rows for each in self._keys()], "bool")

    def _bound(self, label: Any, side: str) -> int:
        """Where a key would go among the sorted rows, before or after its equals."""
        key = label if isinstance(label, tuple) else (label,)
        prefixes = [row[: len(key)] for row in self]
        if not self._sorted():
            from .errors import UnsortedIndexError

            raise UnsortedIndexError(
                f"Key length ({len(key)}) was greater than MultiIndex lexsort depth (0)"
            )
        found = bisect.bisect_left if side == "left" else bisect.bisect_right
        return found(prefixes, key)

    def get_slice_bound(self, label: Any, side: str) -> int:
        """Where a key would go among the sorted rows, on one side of its equals."""
        if side not in ("left", "right"):
            raise InvalidArgumentError(
                f"Invalid value for side kwarg, must be either 'left' or 'right': {side}"
            )
        return self._bound(label, side)

    def slice_locs(self, start: Any = None, end: Any = None, step: Any = None) -> tuple[int, int]:
        """Where the rows from a start key through an end key begin and stop."""
        first = 0 if start is None else self._bound(start, "left")
        last = len(self) if end is None else self._bound(end, "right")
        return first, last

    def slice_indexer(self, start: Any = None, end: Any = None, step: Any = None) -> slice:
        """A slice over the rows between two keys."""
        first, last = self.slice_locs(start, end, step)
        return slice(first, last, step)

    def truncate(self, before: Any = None, after: Any = None) -> Any:
        """The rows between two keys.

        Raises:
            ValueError: When after comes before before.
        """
        if before is not None and after is not None and after < before:
            raise InvalidArgumentError("after < before")
        first, last = self.slice_locs(before, after)
        return self._taken(list(range(first, last)))

    def searchsorted(self, value: Any, side: str = "left", sorter: Any = None) -> int:
        """Where a row would go to keep the rows sorted."""
        rows = self.tolist()
        if sorter is not None:
            rows = [rows[at] for at in _list(sorter)]
        found = bisect.bisect_left if side == "left" else bisect.bisect_right
        return found(rows, value)

    def asof(self, label: Any) -> Any:
        """The label itself when a row holds it."""
        if label in self:
            return label
        raise NotImplementedError(
            "asof on a MultiIndex for a label no row holds is not supported yet"
        )

    def asof_locs(self, where: Any, mask: Any) -> Any:
        """Refused, since it needs rows ordered as instants."""
        raise NotImplementedError("asof_locs is not supported on a MultiIndex yet")

    # Building new indexes.

    def _merged(self, others: list[Any]) -> tuple[list[list[Any]], list[Any]]:
        """Levels holding the values of this index and others, and the rows of all."""
        levels = []
        for number in range(self.nlevels):
            values = list(self._levels[number])
            for other in others:
                values += [v for v in other._levels[number] if v not in values]
            levels.append(_ordered(values))
        rows = self.tolist()
        for other in others:
            rows += other.tolist()
        return levels, rows

    def _on_levels(self, levels: list[list[Any]], rows: list[tuple[Any, ...]], names: Any) -> Any:
        """An index of some rows coded against given levels."""
        codes = []
        for number, level in enumerate(levels):
            where = {v: at for at, v in enumerate(level)}
            codes.append([-1 if _gap(row[number]) else where[row[number]] for row in rows])
        return self._built(levels, codes, names)

    def _other(self, other: Any) -> Any:
        """Another index for a set operation, read as rows.

        Raises:
            TypeError: For anything but another MultiIndex or a list of tuples.
        """
        if isinstance(other, MultiIndex):
            return other
        rows = _list(other) if _list_like(other) else None
        if rows is None or not all(isinstance(row, tuple) for row in rows):
            raise TypeError("other must be a MultiIndex or a list of tuples")
        if not rows:
            return self._taken([])
        return MultiIndex.from_tuples(rows, names=self._names)

    def _names_with(self, other: Any) -> list[Any]:
        """The names both sides share, None where they differ."""
        return [a if a == b else None for a, b in zip(self._names, other._names, strict=True)]

    def _sorted_or_not(self, result: Any, sort: Any) -> Any:
        """A result sorted the way pandas sorts a set operation when it can."""
        if sort is False:
            return result
        try:
            return result.sort_values()
        except TypeError:
            return result

    def append(self, other: Any) -> Any:
        """This index's rows then another's, or several others'.

        Raises:
            NotImplementedError: For something that is not a MultiIndex, since the
                answer would be a plain index of tuples.
        """
        others = list(other) if isinstance(other, (list, tuple)) else [other]
        if not all(isinstance(each, MultiIndex) for each in others):
            raise NotImplementedError(
                "appending something other than a MultiIndex is not supported yet, because"
                " the answer is an index of tuples"
            )
        levels, rows = self._merged(others)
        names = self._names
        for each in others:
            names = [a if a == b else None for a, b in zip(names, each._names, strict=True)]
        return self._on_levels(levels, rows, names)

    def insert(self, loc: int, item: Any) -> Any:
        """The index with one row put in at a position.

        Raises:
            ValueError: For a row with the wrong number of values.
        """
        if not isinstance(item, tuple) or len(item) != self.nlevels:
            raise InvalidArgumentError("Item must have length equal to number of levels.")
        levels = [list(level) for level in self._levels]
        codes = [list(code) for code in self._codes]
        for number, value in enumerate(item):
            code = self._code_of(number, value)
            if code is None:
                levels[number].append(value)
                code = len(levels[number]) - 1
            codes[number].insert(loc, code)
        return self._built(levels, codes, self._names)

    def delete(self, loc: Any) -> Any:
        """The index without the rows at some positions."""
        size = len(self)
        gone = {at + size if at < 0 else at for at in (_list(loc) if _list_like(loc) else [loc])}
        return self._taken([at for at in range(size) if at not in gone])

    def repeat(self, repeats: Any, axis: Any = None) -> Any:
        """Every row repeated some number of times."""
        counts = _list(repeats) if _list_like(repeats) else [repeats] * len(self)
        return self._taken([at for at, count in enumerate(counts) for _ in range(int(count))])

    def drop(self, codes: Any, level: Any = None, errors: str = "raise") -> Any:
        """The index without some rows, named whole or by their first values or one level.

        Raises:
            KeyError: For a key no row holds, unless errors is ignore.
        """
        keys = _list(codes) if _list_like(codes) and not isinstance(codes, tuple) else [codes]
        if level is not None:
            number = self._level_number(level)
            gone_codes = {self._code_of(number, key) for key in keys} - {None}
            if not gone_codes and errors == "raise":
                raise KeyError(f"labels {keys} not found in level")
            return self._taken(
                [at for at, c in enumerate(self._codes[number]) if c not in gone_codes]
            )
        gone: set[int] = set()
        for key in keys:
            try:
                gone.update(_rows_of(self.get_loc(key), len(self)))
            except KeyError:
                if errors == "raise":
                    raise
        return self._taken([at for at in range(len(self)) if at not in gone])

    def union(self, other: Any, sort: Any = None) -> Any:
        """The rows of either index, each once."""
        other = self._other(other)
        levels, rows = self._merged([other])
        merged = self._on_levels(levels, rows, self._names_with(other)).drop_duplicates()
        return self._sorted_or_not(merged, sort)

    def intersection(self, other: Any, sort: bool = False) -> Any:
        """The rows of both indexes, each once, in this index's order."""
        other = self._other(other)
        theirs = set(map(_hashed, other.tolist()))
        rows = [at for at, row in enumerate(self.tolist()) if _hashed(row) in theirs]
        kept = self._taken(rows).drop_duplicates().set_names(self._names_with(other))
        return self._sorted_or_not(kept, sort if sort else False)

    def difference(self, other: Any, sort: Any = None) -> Any:
        """The rows of this index that are not in another, each once."""
        other = self._other(other)
        theirs = set(map(_hashed, other.tolist()))
        rows = [at for at, row in enumerate(self.tolist()) if _hashed(row) not in theirs]
        return self._sorted_or_not(self._taken(rows).drop_duplicates(), sort)

    def symmetric_difference(self, other: Any, result_name: Any = None, sort: Any = None) -> Any:
        """The rows of exactly one of the two indexes, each once."""
        other = self._other(other)
        mine = set(map(_hashed, self.tolist()))
        theirs = set(map(_hashed, other.tolist()))
        levels, _ = self._merged([other])
        rows = [row for row in self.tolist() if _hashed(row) not in theirs]
        rows += [row for row in other.tolist() if _hashed(row) not in mine]
        names = self._names_with(other) if result_name is None else result_name
        return self._sorted_or_not(self._on_levels(levels, rows, names).drop_duplicates(), sort)

    def join(
        self,
        other: Any,
        *,
        how: str = "left",
        level: Any = None,
        return_indexers: bool = False,
        sort: bool = False,
    ) -> Any:
        """The rows two indexes are joined on, the way `Index.join` picks them."""
        if level is not None or return_indexers:
            raise NotImplementedError(
                "level= and return_indexers= are not supported yet on a MultiIndex join"
            )
        other = self._other(other)
        chosen = {
            "left": lambda: self,
            "right": lambda: other,
            "inner": lambda: self.intersection(other, sort=False),
            "outer": lambda: self.union(other),
        }
        if how not in chosen:
            raise InvalidArgumentError(f"do not recognize join method {how}")
        joined = chosen[how]()
        return joined.sort_values() if sort else joined

    def reindex(
        self,
        target: Any,
        method: Any = None,
        level: Any = None,
        limit: Any = None,
        tolerance: Any = None,
    ) -> tuple[Any, Any]:
        """The target as an index, and where each of its rows is in this one."""
        if level is not None:
            raise NotImplementedError("level= is not supported yet on a MultiIndex reindex")
        wanted = self._other(target)
        wanted = wanted.set_names(self._names) if wanted._names == [None] * self.nlevels else wanted
        if wanted.equals(self):
            return wanted, None
        return wanted, self.get_indexer(wanted, method=method, limit=limit, tolerance=tolerance)

    def putmask(self, mask: Any, value: Any) -> Any:
        """The index with the rows a mask picks taken from another index at the same place."""
        other = self._other(value)
        picks = _list(mask)
        rows = self.tolist()
        theirs = other.tolist()
        mixed = [
            theirs[at % len(theirs)] if keep else row
            for at, (row, keep) in enumerate(zip(rows, picks, strict=True))
        ]
        levels = [list(level) for level in self._levels]
        for number, level in enumerate(levels):
            level += [
                v
                for v in dict.fromkeys(row[number] for row in mixed)
                if not _gap(v) and v not in level
            ]
        return self._on_levels([_ordered(level) for level in levels], mixed, self._names)

    def where(self, cond: Any, other: Any = None) -> Any:
        """Refused as pandas refuses it."""
        raise NotImplementedError(".where is not supported for MultiIndex operations")

    def map(self, mapper: Any, na_action: Any = None) -> Any:
        """Every row through a function or a dict, a MultiIndex when every answer is a tuple."""
        call = mapper.get if isinstance(mapper, dict) else mapper
        answers = [call(row) for row in self]
        if answers and all(isinstance(each, tuple) for each in answers):
            return MultiIndex.from_tuples(answers)
        return Index(answers)

    def groupby(self, values: Any) -> dict[Any, Any]:
        """The rows grouped by a list of keys, one index per key."""
        groups: dict[Any, list[int]] = {}
        for at, key in enumerate(_list(values)):
            groups.setdefault(key, []).append(at)
        return {key: self._taken(groups[key]) for key in _ordered(list(groups))}

    def astype(self, dtype: Any, copy: bool = True) -> Any:
        """A copy when asked for object, since a row is a tuple.

        Raises:
            TypeError: For any other type.
        """
        if str(dtype) not in ("object", "O") and dtype is not object:
            raise TypeError(
                "Setting a MultiIndex dtype to anything other than object is not supported"
            )
        return self.copy()

    def infer_objects(self, copy: bool = True) -> Any:
        """Refused as pandas refuses it."""
        raise NotImplementedError(
            "infer_objects is not implemented for MultiIndex. Use index.to_frame().infer_objects()"
            " instead."
        )

    def to_frame(
        self, index: bool = True, name: Any = NO_DEFAULT, allow_duplicates: bool = False
    ) -> DataFrame:
        """A frame with one column per level.

        Raises:
            ValueError: For names that do not fit the levels or repeat.
            NotImplementedError: For index=True, which labels the rows with this index.
        """
        if index:
            raise NotImplementedError(
                "to_frame(index=True) is not supported yet, because it labels the rows of a"
                " frame with a MultiIndex; pass index=False"
            )
        if name is NO_DEFAULT:
            labels = [n if n is not None else at for at, n in enumerate(self._names)]
        else:
            if not _list_like(name) or len(list(name)) != self.nlevels:
                raise InvalidArgumentError(
                    "'name' should have same length as number of levels on index."
                )
            labels = list(name)
        if len(set(labels)) != len(labels):
            raise InvalidArgumentError(
                "Cannot create duplicate column labels if allow_duplicates is False"
            )
        return DataFrame({label: self.get_level_values(at) for at, label in enumerate(labels)})

    def to_series(self, index: Any = None, name: Any = None) -> Series:
        """Refused, since a column cannot hold tuples."""
        raise NotImplementedError(
            "to_series on a MultiIndex is not supported yet, because a column cannot hold tuples"
        )

    def to_flat_index(self) -> Index:
        """Refused, since an index cannot hold tuples."""
        raise NotImplementedError(
            "to_flat_index is not supported yet, because an index cannot hold tuples"
        )

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins: Any = None,
        dropna: bool = True,
    ) -> Series:
        """Refused, since the answer is a column labelled by a MultiIndex."""
        raise NotImplementedError(
            "value_counts on a MultiIndex is not supported yet, because the answer is a column"
            " labelled by a MultiIndex"
        )

    # Reductions and arithmetic, which a row of tuples mostly does not have.

    def _extreme(self, last: bool) -> Any:
        """The first or last row in sorted order, NaN when there are none."""
        if not len(self):
            return _GAP
        rows = self._ascending()
        return self._row(rows[-1] if last else rows[0])

    def min(self, axis: Any = None, skipna: bool = True, *args: Any, **kwargs: Any) -> Any:
        """The smallest row."""
        return self._extreme(False)

    def max(self, axis: Any = None, skipna: bool = True, *args: Any, **kwargs: Any) -> Any:
        """The largest row."""
        return self._extreme(True)

    def argmin(self, axis: Any = None, skipna: bool = True, *args: Any, **kwargs: Any) -> int:
        """The position of the smallest row."""
        return self._ascending()[0]

    def argmax(self, axis: Any = None, skipna: bool = True, *args: Any, **kwargs: Any) -> int:
        """The position of the largest row, the first of equals."""
        rows = self._ascending()
        best = self._keys()[rows[-1]]
        return next(at for at in rows if self._keys()[at] == best)

    def all(self, *args: Any, **kwargs: Any) -> Any:
        """Refused as pandas refuses it."""
        raise TypeError("cannot perform all with MultiIndex")

    def any(self, *args: Any, **kwargs: Any) -> Any:
        """Refused as pandas refuses it."""
        raise TypeError("cannot perform any with MultiIndex")

    def round(self, decimals: int = 0) -> Any:
        """Refused as pandas refuses it, since a tuple does not round."""
        raise TypeError("type tuple doesn't define __round__ method")

    def diff(self, periods: int = 1) -> Any:
        """Refused as pandas refuses it, since tuples do not subtract."""
        raise TypeError("unsupported operand type(s) for -: 'tuple' and 'tuple'")

    def shift(self, periods: int = 1, freq: Any = None) -> Any:
        """Refused as pandas refuses it."""
        raise NotImplementedError(
            "This method is only implemented for DatetimeIndex, PeriodIndex and"
            " TimedeltaIndex; Got type MultiIndex"
        )


def _same(a: tuple[Any, ...], b: tuple[Any, ...]) -> bool:
    """Whether two rows are equal, a gap equal to a gap."""
    return len(a) == len(b) and all(
        (_gap(x) and _gap(y)) or x == y for x, y in zip(a, b, strict=True)
    )


def _hashed(row: tuple[Any, ...]) -> tuple[Any, ...]:
    """A row with its gaps made one value, so rows with gaps compare in a set."""
    return tuple(None if _gap(value) else value for value in row)


def _rows_of(found: Any, size: int) -> list[int]:
    """The positions a lookup answer names, whether a position, a slice or a mask."""
    if isinstance(found, int):
        return [found]
    if isinstance(found, slice):
        return list(range(size))[found]
    return [at for at, keep in enumerate(found) if keep]
