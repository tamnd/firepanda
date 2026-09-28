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


def _keeping(fn: Callable[..., Any], source_of: Callable[[Any], Any], name: str) -> Any:
    """`fn`, with its answer given what its source carries."""
    binary = name in _BINARY
    agreeing = name in _AGREED

    @functools.wraps(fn)
    def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not _USED:
            return fn(self, *args, **kwargs)
        result = fn(self, *args, **kwargs)
        source = source_of(self)
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
            return fn(self, *args, **kwargs)
        result = fn(self, *args, **kwargs)
        source = source_of(self)
        held = carried(source)
        frame = getattr(result, "_frame", None)
        if held is not None and frame is not None and frame is not source:
            _stamp(frame, held)
        return result

    return method


def _dropping(fn: Callable[..., Any], source_of: Callable[[Any], Any]) -> Any:
    """`fn`, with its answer carrying nothing, for what pandas does not finalize."""

    @functools.wraps(fn)
    def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not _USED:
            return fn(self, *args, **kwargs)
        result = fn(self, *args, **kwargs)
        _strip(result, source_of(self))
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
            return fn(self, *args, **kwargs)
        result = fn(self, *args, **kwargs)
        source = source_of(self)
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
            getter = _wrapped(raw.fget, name, source_of, drops, drop_all, grouped)
            setattr(cls, name, property(getter, raw.fset, raw.fdel, raw.__doc__))
        elif inspect.isfunction(raw):
            if raw not in made:
                made[raw] = _wrapped(raw, name, source_of, drops, drop_all, grouped)
            setattr(cls, name, made[raw])


def _wrapped(
    fn: Callable[..., Any],
    name: str,
    source_of: Callable[[Any], Any],
    drops: Any,
    drop_all: bool,
    grouped: bool,
) -> Any:
    """The right wrapper for one method of a class, by the lists above."""
    if name in _ITERATED:
        return _iterating(fn, source_of)
    if name == "groupby" or (grouped and name == "__getitem__"):
        return _grouping(fn, source_of)
    if drop_all or name in drops:
        return _dropping(fn, source_of)
    if name in ("agg", "aggregate"):
        return _aggregating(fn, source_of, grouped)
    return _keeping(fn, source_of, name)


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


def _agreeing_function(fn: Callable[..., Any], inputs_of: Callable[..., list[Any]]) -> Any:
    """`concat` or `merge`, which keep what every input holds in common."""

    @functools.wraps(fn)
    def function(*args: Any, **kwargs: Any) -> Any:
        if not _USED:
            return fn(*args, **kwargs)
        result = fn(*args, **kwargs)
        held = agreed(inputs_of(*args, **kwargs))
        if held is None:
            _strip(result, None)
        else:
            _stamp(result, held)
        return result

    return function


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

    _install(_frame.DataFrame, itself, _FRAME_DROPS)
    _install(_frame.Series, itself, _FRAME_DROPS)
    for cls in (_pandas._Positional, _pandas._Labelled):
        _install(cls, lambda selection: selection._owner, frozenset())
    _install(_frame.StringAccessor, lambda accessor: accessor._series, frozenset())
    _install(
        _frame.DatetimeProperties,
        lambda accessor: accessor._series,
        frozenset(["isocalendar", "to_pydatetime"]),
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
        _install(cls, lambda window: window._whole, frozenset(), True)
    _install(_frame.ExponentialMovingWindow, lambda window: window._data, frozenset(), True)

    for name in ("melt", "pivot", "crosstab", "cut", "qcut", "to_datetime", "to_numeric"):
        for module in (_pandas, firepanda):
            if hasattr(module, name):
                setattr(module, name, _dropping_function(getattr(module, name)))
    for module in (_pandas, firepanda):
        module.concat = _agreeing_function(module.concat, _concat_inputs)
        module.merge = _agreeing_function(module.merge, _merge_inputs)
