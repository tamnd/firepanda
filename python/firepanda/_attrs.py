"""`attrs` and `flags`, the two things pandas carries from a frame to what is made of it.

A frame or a column holds both in one slot, `_carried`, as a pair of the
`attrs` dictionary and whether duplicate labels are refused. The slot is unset
on almost every object, which is the fast path: carrying costs one slot read.

pandas carries them in `__finalize__`, which most methods call on their answer
and some do not, so which answers keep them is a list rather than a rule. The
lists below were measured on pandas 3.0 by calling every method of each class
on an object holding `attrs` and seeing which answers still held them.

- Most methods of a frame or a column keep them, as do `loc`, `iloc`, `str`
  and `dt`. `value_counts`, `dtypes`, `memory_usage`, `dot`, `align`,
  `combine`, `pivot` and an `agg` of a list or a dictionary do not.
- An arithmetic or comparison answer takes the other side's when it has some,
  and its own otherwise.
- `merge`, `join` and `concat` keep them only when every input holds the same.
- In a group by, the reductions pandas runs in its compiled code drop them,
  which is `size`, `count`, `std`, `var`, the cumulative methods and more, and
  the rest keep them. A resample drops the same kind.
- Windows and the `cat` accessor drop them.

Refusing duplicate labels is carried the same way, and every answer that
carries it checks its labels and raises `DuplicateLabelError` with pandas'
table of where each repeated label is.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import inspect
from collections.abc import Callable, Iterator
from typing import Any

from .errors import DuplicateLabelError

_BINARY = frozenset(
    [
        f"__{op}__"
        for op in [
            "add",
            "radd",
            "sub",
            "rsub",
            "mul",
            "rmul",
            "truediv",
            "rtruediv",
            "floordiv",
            "rfloordiv",
            "mod",
            "rmod",
            "pow",
            "rpow",
            "and",
            "rand",
            "or",
            "ror",
            "xor",
            "rxor",
            "eq",
            "ne",
            "lt",
            "le",
            "gt",
            "ge",
        ]
    ]
    + [
        "add",
        "radd",
        "sub",
        "rsub",
        "mul",
        "rmul",
        "truediv",
        "rtruediv",
        "div",
        "rdiv",
        "floordiv",
        "rfloordiv",
        "mod",
        "rmod",
        "pow",
        "rpow",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
    ]
)
"""The operators, whose answer takes the other side's `attrs` when it has some."""

_AGREED = frozenset(["merge", "join"])
"""Methods whose answer keeps `attrs` only when every input holds the same."""

_UNWRAPPED = frozenset(["attrs", "flags", "set_flags", "pipe", "_wrap"])
"""Names left alone: the two themselves, `set_flags`, which sets them, and
`pipe`, which answers whatever the function it is handed answers."""

_ITERATED = frozenset(["items", "iterrows", "__iter__"])
"""Methods that hand out frames or columns one at a time, each of which keeps them."""

_FRAME_DROPS = frozenset(
    [
        "value_counts",
        "dtypes",
        "memory_usage",
        "dot",
        "__matmul__",
        "__rmatmul__",
        "align",
        "combine",
        "pivot",
    ]
)

_GROUPED_DROPS = frozenset(
    [
        "all",
        "any",
        "bfill",
        "count",
        "cumcount",
        "cummax",
        "cummin",
        "cumprod",
        "cumsum",
        "ffill",
        "kurt",
        "ngroup",
        "nunique",
        "ohlc",
        "pct_change",
        "quantile",
        "rank",
        "sem",
        "size",
        "skew",
        "std",
        "var",
    ]
)

_RESAMPLED_DROPS = frozenset(["count", "nunique", "ohlc", "quantile", "sem", "size", "std", "var"])


_USED = False
"""Whether any frame or column has held anything yet. Until one does, every
wrapper below calls straight through, so a program that never sets `attrs`
or `flags` pays one global read for each call."""


_FREQ_USED = False
"""Whether any frame or column has carried an index frequency, the fast path's flag."""

_COLUMNS_USED = False
"""Whether any frame has held a name for its column axis, the fast path's flag."""

_FREQ_DROPS = frozenset(
    [
        "set_index",
        "reset_index",
        "reindex",
        "reindex_like",
        "set_axis",
        "T",
        "transpose",
        "explode",
        "stack",
        "unstack",
        "melt",
        "pivot_table",
        "asfreq",
        "resample",
        "to_period",
        "to_timestamp",
        "merge",
        "join",
        "droplevel",
        "swaplevel",
        "reorder_levels",
        "describe",
        "sort_values",
        "nlargest",
        "nsmallest",
        "mode",
        "corr",
        "cov",
        "corrwith",
        "compare",
        "sample",
        "value_counts",
        "dtypes",
        "memory_usage",
        "dot",
        "__matmul__",
        "__rmatmul__",
        "align",
        "combine",
        "pivot",
    ]
)
"""The methods whose answer has labels of its own, which do not carry the frequency."""


def row_freq(obj: Any) -> Any:
    """The frequency a frame or column holds for its row labels, or None."""
    try:
        return obj._row_freq
    except AttributeError:
        return None


def hold_freq(obj: Any, freq: Any) -> None:
    """Gives a frame or column a frequency for its row labels, or takes it away."""
    global _FREQ_USED
    if freq is None:
        with contextlib.suppress(AttributeError):
            del obj._row_freq
        return
    _FREQ_USED = True
    obj._row_freq = freq


def column_names(obj: Any) -> tuple[Any, ...] | None:
    """The names a frame holds for its column axis, one a level, or None when it holds none."""
    try:
        return obj._column_names
    except AttributeError:
        return None


def hold_columns(obj: Any, names: Any) -> None:
    """Gives a frame names for its column axis, or takes them away with None.

    Names of None are held rather than dropped, so a frame whose axis was
    cleared does not take its source's names back when a method carries them.
    """
    global _COLUMNS_USED
    if names is None:
        with contextlib.suppress(AttributeError):
            del obj._column_names
        return
    with contextlib.suppress(AttributeError):
        obj._column_names = tuple(names)
        _COLUMNS_USED = True


_COLUMNS_DROPS = frozenset(
    [
        "melt",
        "filter",
        "stack",
        "unstack",
        "pivot",
        "pivot_table",
        "value_counts",
        "memory_usage",
        "duplicated",
        "set_axis",
        "T",
        "transpose",
    ]
)
"""The methods whose answer does not keep the name of the column axis, measured on pandas 3.0."""


def _carry_columns(result: Any, source: Any, name: str) -> None:
    """Passes the name of the column axis on from a frame to what was made of it.

    A frame keeps it, as nearly every pandas method keeps the columns' index. A
    column whose labels are the frame's columns, which is a reduction or one
    row read across, takes it as the name of its labels.
    """
    if result is source or name in _COLUMNS_DROPS:
        return
    held = column_names(source)
    if held is None:
        return
    kind = type(result)
    if kind is type(source):
        if column_names(result) is None:
            result._column_names = held
        if name in ("corr", "cov") and len(held) == 1 and held[0] is not None:
            _name_labels(result, source, held[0])
    elif kind in _SLOTS and len(held) == 1 and held[0] is not None:
        _name_labels(result, source, held[0])


def _name_labels(column: Any, frame: Any, name: Any) -> None:
    """Names the labels of a column that are the columns of the frame it came from."""
    labels = column.index
    if labels.name is not None or len(labels) != frame._inner.width():
        return
    if labels.tolist() == frame.columns.tolist():
        column._inner = column.rename_axis(name)._inner


def _turning(fn: Callable[..., Any], source_of: Callable[[Any], Any]) -> Any:
    """`T` or `transpose`, whose answer swaps the names of the two axes, as in pandas."""

    @functools.wraps(fn)
    def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = fn(self, *args, **kwargs)
        source = source_of(self)
        if result is source or type(result) is not type(source) or not hasattr(source, "columns"):
            return result
        held = column_names(source)
        rows = source.index.names
        if held is not None and len(held) == result.index.nlevels and any(held):
            result._inner = result.rename_axis(list(held) if len(held) > 1 else held[0])._inner
        if any(one is not None for one in rows):
            hold_columns(result, rows)
        return result

    return method


def _carry_freq(result: Any, source: Any) -> None:
    """Passes the frequency on from a source to what was made of it.

    Whether the labels still step by it is read when they are asked for, so a
    filter that broke the run hands it on and the index shows none.
    """
    if result is source or type(result) not in _SLOTS:
        return
    held = row_freq(source)
    if held is not None and row_freq(result) is None:
        result._row_freq = held


def carried(obj: Any) -> tuple[dict[Any, Any], bool] | None:
    """The `attrs` and the refusal of duplicate labels a frame or a column holds, or None."""
    read = _SLOTS.get(type(obj))
    if read is None:
        return None
    try:
        return read(obj)
    except AttributeError:
        return None


def hold(obj: Any, attrs: dict[Any, Any] | None, unique_only: bool) -> None:
    """Sets what `obj` carries, leaving the slot empty when there is nothing to carry."""
    global _USED
    write = _WRITERS[type(obj)]
    if attrs is not None or unique_only:
        _USED = True
        write(obj, (attrs if attrs is not None else {}, unique_only))
    else:
        with contextlib.suppress(AttributeError):
            _DELETERS[type(obj)](obj)


def _stamp(result: Any, held: tuple[dict[Any, Any], bool] | None) -> None:
    """Gives a new frame or column what its source carried, as `__finalize__` does."""
    if held is None or type(result) not in _SLOTS:
        return
    attrs, unique_only = held
    mine = carried(result)
    own_attrs, own_unique = mine if mine is not None else ({}, False)
    if unique_only:
        check_unique(result)
    hold(result, copy.deepcopy(attrs) if attrs else own_attrs, unique_only or own_unique)


def _strip(result: Any, source: Any) -> None:
    """Empties what a new frame or column carries, for the methods pandas does not finalize."""
    if result is not source and type(result) in _SLOTS and carried(result) is not None:
        hold(result, None, False)


class Flags:
    """What a frame or a column allows, which is pandas' `Flags`.

    There is one flag, `allows_duplicate_labels`. Setting it to False checks
    the labels there are and raises `DuplicateLabelError` if one repeats, and
    every answer made from the object afterwards refuses a repeated label too.
    The flags an object answers write through to it, while one made by hand
    only checks.
    """

    _keys: set[str] = {"allows_duplicate_labels"}  # noqa: RUF012

    def __init__(self, obj: Any, *, allows_duplicate_labels: bool) -> None:
        self._allows_duplicate_labels = allows_duplicate_labels
        self._obj = obj
        self._bound = False

    @property
    def allows_duplicate_labels(self) -> bool:
        """Whether the labels of the object may repeat."""
        return self._allows_duplicate_labels

    @allows_duplicate_labels.setter
    def allows_duplicate_labels(self, value: bool) -> None:
        value = bool(value)
        if not value:
            check_unique(self._obj)
        self._allows_duplicate_labels = value
        if self._bound:
            held = carried(self._obj)
            hold(self._obj, held[0] if held else None, not value)

    def __getitem__(self, key: str) -> Any:
        if key not in self._keys:
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self._keys:
            raise ValueError(f"Unknown flag {key}. Must be one of {self._keys}")
        setattr(self, key, value)

    def __repr__(self) -> str:
        return f"<Flags(allows_duplicate_labels={self.allows_duplicate_labels})>"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return self.allows_duplicate_labels == other.allows_duplicate_labels
        return False

    __hash__ = None  # type: ignore[assignment]


def flags_of(obj: Any) -> Flags:
    """The flags of a frame or a column, which write through to it."""
    held = carried(obj)
    flags = Flags(obj, allows_duplicate_labels=not (held is not None and held[1]))
    flags._bound = True
    return flags


def _duplicates_table(labels: list[Any]) -> str:
    """pandas' table of each repeated label and the positions it is at."""
    from ._frame import DataFrame, Index

    positions: dict[Any, list[int]] = {}
    again: list[Any] = []
    for at, label in enumerate(labels):
        seen = positions.setdefault(label, [])
        if len(seen) == 1:
            again.append(label)
        seen.append(at)
    shown = [str(positions[label]) for label in again]
    return repr(DataFrame({"positions": shown}, index=Index(again, name="label")))


def check_unique(obj: Any) -> None:
    """Raises pandas' `DuplicateLabelError` when a label repeats on either axis.

    Raises:
        DuplicateLabelError: For the row labels first and then the columns.
    """
    axes = [obj.index.tolist()]
    if hasattr(obj, "columns"):
        axes.append(list(obj.columns))
    for labels in axes:
        if len(set(labels)) != len(labels):
            raise DuplicateLabelError("Index has duplicates.\n" + _duplicates_table(labels))


def _other_of(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """The other side of an operator or a join, however it was passed."""
    if args:
        return args[0]
    for name in ("other", "right"):
        if name in kwargs:
            return kwargs[name]
    return None


def agreed(inputs: list[Any]) -> tuple[dict[Any, Any], bool] | None:
    """What several inputs carry in common, which is what `concat` and `merge` keep.

    `attrs` are kept only when every input holds the same, and duplicate labels
    are refused when every input refuses them.
    """
    held = [carried(each) for each in inputs]
    attrs = [pair[0] if pair else {} for pair in held]
    kept = attrs[0] if attrs and attrs[0] and all(each == attrs[0] for each in attrs) else {}
    unique_only = bool(held) and all(pair is not None and pair[1] for pair in held)
    return (kept, unique_only) if kept or unique_only else None


def _keeping(
    fn: Callable[..., Any], source_of: Callable[[Any], Any], name: str, freq: bool = False
) -> Any:
    """`fn`, with its answer given what its source carries."""
    binary = name in _BINARY
    agreeing = name in _AGREED
    freq = freq and name not in _FREQ_DROPS

    @functools.wraps(fn)
    def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not _USED:
            if not (freq and _FREQ_USED) and not _COLUMNS_USED:
                return fn(self, *args, **kwargs)
            result = fn(self, *args, **kwargs)
            if freq and _FREQ_USED:
                _carry_freq(result, source_of(self))
            if _COLUMNS_USED:
                _carry_columns(result, source_of(self), name)
            return result
        result = fn(self, *args, **kwargs)
        source = source_of(self)
        if freq and _FREQ_USED:
            _carry_freq(result, source)
        if _COLUMNS_USED:
            _carry_columns(result, source, name)
        if agreeing:
            other = _other_of(args, kwargs)
            others = other if isinstance(other, list | tuple) else [other]
            held = agreed([source, *others])
            if held is None:
                _strip(result, source)
                return result
        else:
            held = carried(source)
            if binary:
                theirs = carried(_other_of(args, kwargs))
                if theirs is not None and theirs[0]:
                    unique_only = theirs[1] or (held is not None and held[1])
                    held = (theirs[0], unique_only)
        if held is not None and result is not source:
            _stamp(result, held)
        return result

    return method


def _grouping(fn: Callable[..., Any], source_of: Callable[[Any], Any]) -> Any:
    """`groupby`, or picking columns out of one, whose frame has to carry them.

    A group by keeps a frame of its own, which is often narrowed to the keys
    and the columns picked, and its answers keep what that frame carries.
    """

    @functools.wraps(fn)
    def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not _USED:
            result = fn(self, *args, **kwargs)
            if _COLUMNS_USED:
                _carry_group_columns(result, source_of(self))
            return result
        result = fn(self, *args, **kwargs)
        source = source_of(self)
        if _COLUMNS_USED:
            _carry_group_columns(result, source)
        held = carried(source)
        frame = getattr(result, "_frame", None)
        if held is not None and frame is not None and frame is not source:
            _stamp(frame, held)
        return result

    return method


def _carry_group_columns(result: Any, source: Any) -> None:
    """Gives the frame a group by keeps the name of its source's column axis."""
    frame = getattr(result, "_frame", None)
    if frame is not None and frame is not source and type(frame) is type(source):
        held = column_names(source)
        if held is not None and column_names(frame) is None:
            frame._column_names = held


def _dropping(
    fn: Callable[..., Any], source_of: Callable[[Any], Any], name: str, freq: bool = False
) -> Any:
    """`fn`, with its answer carrying nothing, for what pandas does not finalize.

    The frequency of the row labels is not part of that, since pandas keeps it
    on the index rather than the frame, so a window still passes it on.
    """

    @functools.wraps(fn)
    def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not _USED:
            if not (freq and _FREQ_USED) and not _COLUMNS_USED:
                return fn(self, *args, **kwargs)
            result = fn(self, *args, **kwargs)
            if freq and _FREQ_USED:
                _carry_freq(result, source_of(self))
            if _COLUMNS_USED:
                _carry_columns(result, source_of(self), name)
            return result
        result = fn(self, *args, **kwargs)
        _strip(result, source_of(self))
        if freq and _FREQ_USED:
            _carry_freq(result, source_of(self))
        if _COLUMNS_USED:
            _carry_columns(result, source_of(self), name)
        return result

    return method


def _aggregating(fn: Callable[..., Any], source_of: Callable[[Any], Any], grouped: bool) -> Any:
    """`agg`, which keeps them for some kinds of function and not for others.

    On a frame or a column a name or a function keeps them and a list or a
    dictionary does not. In a group by or a resample it is the other way round
    for a dictionary, a function or a list drops them, and a name keeps them
    when the method of that name does.
    """
    drops = _GROUPED_DROPS if grouped else _FRAME_DROPS

    @functools.wraps(fn)
    def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not _USED:
            result = fn(self, *args, **kwargs)
            if _COLUMNS_USED:
                _carry_columns(result, source_of(self), "agg")
            return result
        result = fn(self, *args, **kwargs)
        source = source_of(self)
        if _COLUMNS_USED:
            _carry_columns(result, source, "agg")
        func = args[0] if args else kwargs.get("func")
        named = isinstance(func, str) and func not in drops
        if grouped:
            keeps = named or isinstance(func, dict)
        else:
            keeps = named or (callable(func) and not isinstance(func, list | dict))
        held = carried(source) if keeps else None
        if held is not None and result is not source:
            _stamp(result, held)
        elif not keeps:
            _strip(result, source)
        return result

    return method


def _iterating(fn: Callable[..., Any], source_of: Callable[[Any], Any]) -> Any:
    """A method that hands out frames or columns one at a time, each keeping them."""

    @functools.wraps(fn)
    def method(self: Any, *args: Any, **kwargs: Any) -> Iterator[Any]:
        if not _USED:
            yield from fn(self, *args, **kwargs)
            return
        held = carried(source_of(self))
        for item in fn(self, *args, **kwargs):
            if held is not None:
                for part in item if isinstance(item, tuple) else (item,):
                    _stamp(part, held)
            yield item

    return method


def _install(
    cls: type,
    source_of: Callable[[Any], Any],
    drops: Any,
    drop_all: bool = False,
    grouped: bool = False,
    freq: bool = False,
) -> None:
    """Wraps every public method and property of `cls`, and the operators.

    Two names for one function, such as `agg` and `aggregate`, keep one wrapper
    between them, so they stay the same object.
    """
    made: dict[Any, Any] = {}
    for name in dir(cls):
        public = not name.startswith("_") or (name.startswith("__") and name.endswith("__"))
        wanted = public and name not in _UNWRAPPED
        if name.startswith("__") and name not in _BINARY | _ITERATED | _DUNDERS:
            wanted = False
        if not wanted:
            continue
        raw = inspect.getattr_static(cls, name)
        if isinstance(raw, property):
            if raw.fget is None:
                continue
            getter = _wrapped(raw.fget, name, source_of, drops, drop_all, grouped, freq)
            setattr(cls, name, property(getter, raw.fset, raw.fdel, raw.__doc__))
        elif inspect.isfunction(raw):
            if raw not in made:
                made[raw] = _wrapped(raw, name, source_of, drops, drop_all, grouped, freq)
            setattr(cls, name, made[raw])


def _wrapped(
    fn: Callable[..., Any],
    name: str,
    source_of: Callable[[Any], Any],
    drops: Any,
    drop_all: bool,
    grouped: bool,
    freq: bool = False,
) -> Any:
    """The right wrapper for one method of a class, by the lists above."""
    if name in _ITERATED:
        return _iterating(fn, source_of)
    if name in ("T", "transpose") and not grouped:
        return _turning(_keeping(fn, source_of, name, freq), source_of)
    if name == "groupby" or (grouped and name == "__getitem__"):
        return _grouping(fn, source_of)
    if drop_all or name in drops:
        return _dropping(fn, source_of, name, freq and name not in _FREQ_DROPS)
    if name in ("agg", "aggregate"):
        return _aggregating(fn, source_of, grouped)
    return _keeping(fn, source_of, name, freq)


def _index_names(labels: Any) -> list[Any] | None:
    """The names of an index handed in as labels, or None for none or a plain list."""
    names = getattr(labels, "names", None)
    if names is None or isinstance(labels, (list, tuple)):
        return None
    names = list(names)
    return names if any(one is not None for one in names) else None


def _holding_init(fn: Callable[..., Any]) -> Any:
    """A constructor that takes the frequency of the row labels it was given."""

    @functools.wraps(fn)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        fn(self, *args, **kwargs)
        index = kwargs["index"] if "index" in kwargs else args[1] if len(args) > 1 else None
        data = kwargs["data"] if "data" in kwargs else args[0] if args else None
        held = getattr(index, "freq", None) if index is not None else row_freq(data)
        if held is not None and not isinstance(held, str):
            hold_freq(self, held)
        columns = kwargs["columns"] if "columns" in kwargs else args[2] if len(args) > 2 else None
        named = _index_names(columns)
        if named is not None and hasattr(self, "columns"):
            hold_columns(self, named)

    return __init__


def _holding_axis(fn: Callable[..., Any]) -> Any:
    """`set_axis`, whose answer takes the frequency of new row labels."""

    @functools.wraps(fn)
    def set_axis(self: Any, labels: Any, *args: Any, **kwargs: Any) -> Any:
        result = fn(self, labels, *args, **kwargs)
        axis = kwargs.get("axis", args[0] if args else 0)
        held = getattr(labels, "freq", None)
        if axis in (0, "index", "rows") and held is not None and not isinstance(held, str):
            hold_freq(result, held)
        if axis in (1, "columns") and result is not None:
            hold_columns(result, _index_names(labels))
        return result

    return set_axis


_DUNDERS = frozenset(
    [
        "__getitem__",
        "__getattr__",
        "__neg__",
        "__pos__",
        "__abs__",
        "__invert__",
        "__round__",
        "__copy__",
        "__deepcopy__",
        "__matmul__",
        "__rmatmul__",
    ]
)
"""The special methods that answer a new frame or column."""

_SLOTS: dict[type, Callable[[Any], Any]] = {}
_WRITERS: dict[type, Callable[[Any, Any], None]] = {}
_DELETERS: dict[type, Callable[[Any], None]] = {}


def _dropping_function(fn: Callable[..., Any]) -> Any:
    """A module function whose answer carries nothing, as in pandas."""

    @functools.wraps(fn)
    def function(*args: Any, **kwargs: Any) -> Any:
        if not _USED:
            return fn(*args, **kwargs)
        result = fn(*args, **kwargs)
        _strip(result, None)
        return result

    return function


def _agreeing_function(
    fn: Callable[..., Any], inputs_of: Callable[..., list[Any]], freq: bool = False
) -> Any:
    """`concat` or `merge`, which keep what every input holds in common.

    `concat` also keeps a frequency every input's row labels share, which the
    labels drop again when they are read if the pieces did not join up.
    """

    @functools.wraps(fn)
    def function(*args: Any, **kwargs: Any) -> Any:
        if not _USED:
            if not (freq and _FREQ_USED) and not _COLUMNS_USED:
                return fn(*args, **kwargs)
            result = fn(*args, **kwargs)
            if freq and _FREQ_USED:
                _agreed_freq(result, inputs_of(*args, **kwargs), kwargs)
            if _COLUMNS_USED:
                _agreed_columns(result, inputs_of(*args, **kwargs))
            return result
        result = fn(*args, **kwargs)
        if _COLUMNS_USED:
            _agreed_columns(result, inputs_of(*args, **kwargs))
        if freq and _FREQ_USED:
            _agreed_freq(result, inputs_of(*args, **kwargs), kwargs)
        held = agreed(inputs_of(*args, **kwargs))
        if held is None:
            _strip(result, None)
        else:
            _stamp(result, held)
        return result

    return function


def _agreed_columns(result: Any, inputs: list[Any]) -> None:
    """Gives a joined frame the name of the column axis every input frame shares."""
    kind = type(result)
    if not inputs or kind not in _SLOTS or not all(type(one) is kind for one in inputs):
        return
    held = [column_names(one) for one in inputs]
    agreed = held[0] is not None and all(one == held[0] for one in held)
    if agreed and column_names(result) is None:
        result._column_names = held[0]


def _agreed_freq(result: Any, inputs: list[Any], kwargs: dict[str, Any]) -> None:
    """Gives a joined answer the frequency all of its inputs' row labels share."""
    if kwargs.get("ignore_index") or type(result) not in _SLOTS or not inputs:
        return
    held = [row_freq(one) for one in inputs]
    shared = held[0]
    if shared is not None and all(one == shared for one in held) and row_freq(result) is None:
        hold_freq(result, shared)


def _concat_inputs(objs: Any = None, *args: Any, **kwargs: Any) -> list[Any]:
    if objs is None:
        objs = kwargs.get("objs", [])
    return list(objs.values() if isinstance(objs, dict) else objs)


def _merge_inputs(left: Any = None, right: Any = None, *args: Any, **kwargs: Any) -> list[Any]:
    return [kwargs.get("left", left), kwargs.get("right", right)]


def install() -> None:
    """Wraps the classes and the module functions. Called once, on import."""
    import firepanda

    from . import _frame, _pandas, _resample

    for cls, mixin in (
        (_frame.DataFrame, _pandas.DataFrameMixin),
        (_frame.Series, _pandas.SeriesMixin),
    ):
        descriptor = mixin.__dict__["_carried"]
        _SLOTS[cls] = descriptor.__get__
        _WRITERS[cls] = descriptor.__set__
        _DELETERS[cls] = descriptor.__delete__

    def itself(obj: Any) -> Any:
        return obj

    _install(_frame.DataFrame, itself, _FRAME_DROPS, freq=True)
    _install(_frame.Series, itself, _FRAME_DROPS, freq=True)
    for cls in (_frame.DataFrame, _frame.Series):
        cls.__init__ = _holding_init(cls.__init__)  # type: ignore[method-assign]
        cls.set_axis = _holding_axis(cls.set_axis)  # type: ignore[method-assign]
    for cls in (_pandas._Positional, _pandas._Labelled, _pandas._Along):
        _install(cls, lambda selection: selection._owner, frozenset(), freq=True)
    _install(_frame.StringAccessor, lambda accessor: accessor._series, frozenset(), freq=True)
    _install(
        _frame.DatetimeProperties,
        lambda accessor: accessor._series,
        frozenset(["isocalendar", "to_pydatetime"]),
        freq=True,
    )
    _install(_frame.CategoricalAccessor, lambda accessor: accessor._series, frozenset(), True)
    _install(_frame.DataFrameGroupBy, lambda grouped: grouped._frame, _GROUPED_DROPS, grouped=True)
    _install(
        _frame.SeriesGroupBy,
        lambda grouped: grouped._frame,
        _GROUPED_DROPS | {"describe"},
        grouped=True,
    )
    _install(_resample.Resampler, lambda resampled: resampled._obj, _RESAMPLED_DROPS, grouped=True)
    for cls in (_frame.Rolling, _frame.Expanding):
        _install(cls, lambda window: window._whole, frozenset(), True, freq=True)
    _install(
        _frame.ExponentialMovingWindow, lambda window: window._data, frozenset(), True, freq=True
    )

    for name in ("melt", "pivot", "crosstab", "cut", "qcut", "to_datetime", "to_numeric"):
        for module in (_pandas, firepanda):
            if hasattr(module, name):
                setattr(module, name, _dropping_function(getattr(module, name)))
    for module in (_pandas, firepanda):
        module.concat = _agreeing_function(module.concat, _concat_inputs, freq=True)
        module.merge = _agreeing_function(module.merge, _merge_inputs)
