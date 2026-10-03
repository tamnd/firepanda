"""The names pandas gives its methods, which Python prints when a call is wrong.

A call with an argument the method does not take, or without one it needs, is
refused by Python itself, and the sentence starts with the function's qualified
name. pandas defines many methods once on `NDFrame` and some through numpy's
argument checks, which name the bare function, so `df.head(x=1)` reads
"NDFrame.head() got an unexpected keyword argument 'x'". The methods here live
on mixins whose names pandas does not have, so each is renamed once, at import,
to the name pandas would print. A method not in these tables is named after the
class the caller holds.

The tables were read off pandas by calling every public method with an argument
it does not take. Methods that hand their keywords on to another method are left
out, since pandas' sentence there names the other method, which this cannot copy
without doing the same.
"""

from __future__ import annotations

import inspect
import types

_SHARED = {
    "abs": "NDFrame.abs",
    "add_prefix": "NDFrame.add_prefix",
    "add_suffix": "NDFrame.add_suffix",
    "align": "NDFrame.align",
    "all": "all",
    "any": "any",
    "asfreq": "NDFrame.asfreq",
    "asof": "NDFrame.asof",
    "astype": "NDFrame.astype",
    "at_time": "NDFrame.at_time",
    "between_time": "NDFrame.between_time",
    "bfill": "NDFrame.bfill",
    "clip": "clip",
    "convert_dtypes": "NDFrame.convert_dtypes",
    "copy": "NDFrame.copy",
    "cummax": "cummax",
    "cummin": "cummin",
    "cumprod": "cumprod",
    "cumsum": "cumsum",
    "describe": "NDFrame.describe",
    "droplevel": "NDFrame.droplevel",
    "equals": "NDFrame.equals",
    "ewm": "NDFrame.ewm",
    "expanding": "NDFrame.expanding",
    "ffill": "NDFrame.ffill",
    "fillna": "NDFrame.fillna",
    "filter": "NDFrame.filter",
    "first_valid_index": "NDFrame.first_valid_index",
    "get": "NDFrame.get",
    "head": "NDFrame.head",
    "infer_objects": "NDFrame.infer_objects",
    "kurt": "kurt",
    "kurtosis": "kurt",
    "last_valid_index": "NDFrame.last_valid_index",
    "mask": "NDFrame.mask",
    "max": "max",
    "mean": "mean",
    "median": "median",
    "min": "min",
    "prod": "prod",
    "product": "prod",
    "rank": "NDFrame.rank",
    "reindex_like": "NDFrame.reindex_like",
    "replace": "NDFrame.replace",
    "resample": "NDFrame.resample",
    "rolling": "NDFrame.rolling",
    "round": "round",
    "sample": "NDFrame.sample",
    "sem": "sem",
    "set_flags": "NDFrame.set_flags",
    "skew": "skew",
    "squeeze": "NDFrame.squeeze",
    "std": "std",
    "sum": "sum",
    "tail": "NDFrame.tail",
    "to_csv": "NDFrame.to_csv",
    "to_excel": "NDFrame.to_excel",
    "to_hdf": "NDFrame.to_hdf",
    "to_json": "NDFrame.to_json",
    "to_latex": "NDFrame.to_latex",
    "to_pickle": "NDFrame.to_pickle",
    "to_sql": "NDFrame.to_sql",
    "to_xarray": "NDFrame.to_xarray",
    "truncate": "NDFrame.truncate",
    "tz_convert": "NDFrame.tz_convert",
    "tz_localize": "NDFrame.tz_localize",
    "var": "var",
    "where": "NDFrame.where",
    "xs": "NDFrame.xs",
}
_FRAME = {
    "div": "DataFrame.truediv",
    "divide": "DataFrame.truediv",
    "keys": "NDFrame.keys",
    "multiply": "DataFrame.mul",
    "rdiv": "DataFrame.rtruediv",
    "rename_axis": "NDFrame.rename_axis",
    "subtract": "DataFrame.sub",
}
_SERIES = {
    "argmax": "argmax",
    "argmin": "argmax",
    "div": "Series.truediv",
    "divide": "Series.truediv",
    "factorize": "IndexOpsMixin.factorize",
    "item": "IndexOpsMixin.item",
    "multiply": "Series.mul",
    "nunique": "IndexOpsMixin.nunique",
    "rdiv": "Series.rtruediv",
    "shift": "NDFrame.shift",
    "subtract": "Series.sub",
    "to_list": "IndexOpsMixin.tolist",
    "to_numpy": "to_numpy",
    "tolist": "IndexOpsMixin.tolist",
    "transpose": "transpose",
    "value_counts": "IndexOpsMixin.value_counts",
}

# What a group by of either kind prints, from pandas' GroupBy and BaseGroupBy.
_GROUPED = {
    "all": "GroupBy.all",
    "any": "GroupBy.any",
    "bfill": "GroupBy.bfill",
    "count": "GroupBy.count",
    "cumcount": "GroupBy.cumcount",
    "diff": "GroupBy.diff",
    "ewm": "GroupBy.ewm",
    "expanding": "GroupBy.expanding",
    "ffill": "GroupBy.ffill",
    "first": "GroupBy.first",
    "get_group": "BaseGroupBy.get_group",
    "head": "GroupBy.head",
    "last": "GroupBy.last",
    "max": "GroupBy.max",
    "mean": "GroupBy.mean",
    "median": "GroupBy.median",
    "min": "GroupBy.min",
    "ngroup": "GroupBy.ngroup",
    "ohlc": "GroupBy.ohlc",
    "pct_change": "GroupBy.pct_change",
    "pipe": "BaseGroupBy.pipe",
    "prod": "GroupBy.prod",
    "quantile": "GroupBy.quantile",
    "rank": "GroupBy.rank",
    "resample": "GroupBy.resample",
    "rolling": "GroupBy.rolling",
    "sample": "GroupBy.sample",
    "sem": "GroupBy.sem",
    "shift": "GroupBy.shift",
    "size": "GroupBy.size",
    "std": "GroupBy.std",
    "sum": "GroupBy.sum",
    "tail": "GroupBy.tail",
    "var": "GroupBy.var",
}

_GROUPED_FRAME = {"apply": "GroupBy.apply", "describe": "GroupBy.describe"}

_RESAMPLED = {"get_group": "BaseGroupBy.get_group", "quantile": "GroupBy.quantile"}


def name_as_pandas(frame: type, series: type) -> None:
    """Renames the methods of the two classes to the names pandas prints for them.

    A function both classes reach is renamed only when the two agree on the
    name, so a helper shared by a frame and a column keeps its own.

    Args:
        frame: The DataFrame class.
        series: The Series class.
    """
    wanted: dict[int, tuple[object, set[str]]] = {}
    for cls, kind, table in ((frame, "DataFrame", _FRAME), (series, "Series", _SERIES)):
        for owner in reversed(cls.__mro__):
            if not owner.__module__.startswith("firepanda"):
                continue
            for name, member in vars(owner).items():
                if name.startswith("_"):
                    continue
                function = getattr(member, "__func__", member)
                if not inspect.isfunction(function):
                    continue
                shown = table.get(name) or _SHARED.get(name) or f"{kind}.{name}"
                wanted.setdefault(id(function), (function, set()))[1].add(shown)
    for function, names in wanted.values():
        if len(names) != 1:
            continue
        _rename(function, names.pop())


# The second name pandas gives a method, which prints the first.
_ALIASES = {"agg": "aggregate"}


def name_each(*classes: tuple[type, dict[str, str]]) -> None:
    """Renames the methods of related classes, each to the name pandas prints on it.

    A method not in a class's table is named after the class, which is what
    pandas prints for the group bys and the windows. A method firepanda defines
    once on a mixin that pandas names differently on two of the classes, such as
    `Rolling.sum` and `Expanding.sum`, is copied onto each class under its own
    name, so the one function is not asked to have two.

    Args:
        classes: Each class with its table of the names that differ from the default.
    """
    wanted: dict[int, tuple[types.FunctionType, dict[tuple[type, str], str]]] = {}
    for cls, table in classes:
        for name in dir(cls):
            if name.startswith("_"):
                continue
            function = inspect.getattr_static(cls, name)
            if not inspect.isfunction(function) or not function.__module__.startswith("firepanda"):
                continue
            shown = table.get(name) or f"{cls.__name__}.{_ALIASES.get(name, name)}"
            wanted.setdefault(id(function), (function, {}))[1][cls, name] = shown
    for function, names in wanted.values():
        if len(set(names.values())) == 1:
            _rename(function, next(iter(names.values())))
            continue
        if hasattr(function, "__wrapped__"):
            continue
        # Aliases such as `agg` and `aggregate` stay one function on each class.
        copies: dict[tuple[type, str], types.FunctionType] = {}
        for (cls, name), shown in names.items():
            if (cls, shown) not in copies:
                copies[cls, shown] = _copied(function, shown)
            setattr(cls, name, copies[cls, shown])


def _rename(function: object, shown: str) -> None:
    """Names `function`, and whatever it wraps, as pandas names it."""
    # A decorated method raises from the function it wraps, so that is named too.
    while function is not None:
        function.__qualname__ = shown  # type: ignore[attr-defined]
        function = getattr(function, "__wrapped__", None)


def _copied(function: types.FunctionType, shown: str) -> types.FunctionType:
    """A second function with the same code and everything else, under another name."""
    copy = types.FunctionType(
        function.__code__,
        function.__globals__,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    copy.__kwdefaults__ = function.__kwdefaults__
    copy.__dict__.update(function.__dict__)
    copy.__doc__ = function.__doc__
    copy.__module__ = function.__module__
    copy.__annotations__ = function.__annotations__
    copy.__qualname__ = shown
    return copy
