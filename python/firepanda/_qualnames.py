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


# What each index prints for a wrong call, where it is not `Index.<name>`: the
# validators print a bare name, and the rest are where pandas defines the method.
_INDEX_COMMON = {
    "all": "all",
    "any": "any",
    "argmax": "argmax",
    "argmin": "argmin",
    "argsort": "argsort",
    "factorize": "IndexOpsMixin.factorize",
    "get_level_values": "Index._get_level_values",
    "isnull": "Index.isna",
    "item": "IndexOpsMixin.item",
    "max": "max",
    "min": "min",
    "notnull": "Index.notna",
    "nunique": "IndexOpsMixin.nunique",
    "searchsorted": "IndexOpsMixin.searchsorted",
    "to_list": "IndexOpsMixin.tolist",
    "to_numpy": "to_numpy",
    "tolist": "IndexOpsMixin.tolist",
    "transpose": "transpose",
    "value_counts": "IndexOpsMixin.value_counts",
}
# Where an index kind differs from the plain index.
_INDEX_OWN = {
    "CategoricalIndex": {
        "add_categories": "Categorical.add_categories",
        "as_ordered": "Categorical.as_ordered",
        "as_unordered": "Categorical.as_unordered",
        "equals": "CategoricalIndex.equals",
        "map": "CategoricalIndex.map",
        "reindex": "CategoricalIndex.reindex",
        "remove_categories": "Categorical.remove_categories",
        "remove_unused_categories": "Categorical.remove_unused_categories",
        "rename_categories": "Categorical.rename_categories",
        "reorder_categories": "Categorical.reorder_categories",
        "searchsorted": "NDArrayBackedExtensionArray.searchsorted",
        "set_categories": "Categorical.set_categories",
        "to_numpy": "ExtensionArray.to_numpy",
        "tolist": "ExtensionArray.tolist",
    },
    "DatetimeIndex": {
        "as_unit": "TimelikeOps.as_unit",
        "ceil": "TimelikeOps.ceil",
        "day_name": "DatetimeArray.day_name",
        "delete": "DatetimeTimedeltaMixin.delete",
        "equals": "DatetimeIndexOpsMixin.equals",
        "floor": "TimelikeOps.floor",
        "get_loc": "DatetimeIndex.get_loc",
        "indexer_at_time": "DatetimeIndex.indexer_at_time",
        "indexer_between_time": "DatetimeIndex.indexer_between_time",
        "insert": "DatetimeTimedeltaMixin.insert",
        "isocalendar": "DatetimeIndex.isocalendar",
        "mean": "DatetimeIndexOpsMixin.mean",
        "month_name": "DatetimeArray.month_name",
        "normalize": "DatetimeArray.normalize",
        "round": "TimelikeOps.round",
        "shift": "DatetimeTimedeltaMixin.shift",
        "slice_indexer": "DatetimeIndex.slice_indexer",
        "snap": "DatetimeIndex.snap",
        "std": "DatetimeArray.std",
        "strftime": "DatetimeIndex.strftime",
        "to_julian_date": "DatetimeIndex.to_julian_date",
        "to_period": "DatetimeArray.to_period",
        "to_pydatetime": "DatetimeArray.to_pydatetime",
    },
    "IntervalIndex": {
        "contains": "IntervalArray.contains",
        "get_indexer_non_unique": "IntervalIndex.get_indexer_non_unique",
        "get_loc": "IntervalIndex.get_loc",
        "memory_usage": "IntervalIndex.memory_usage",
        "overlaps": "IntervalArray.overlaps",
        "set_closed": "IntervalArray.set_closed",
        "to_numpy": "ExtensionArray.to_numpy",
        "to_tuples": "IntervalArray.to_tuples",
    },
    "MultiIndex": {
        "append": "MultiIndex.append",
        "astype": "MultiIndex.astype",
        "copy": "MultiIndex.copy",
        "delete": "MultiIndex.delete",
        "drop": "MultiIndex.drop",
        "dropna": "MultiIndex.dropna",
        "duplicated": "MultiIndex.duplicated",
        "equal_levels": "MultiIndex.equal_levels",
        "equals": "MultiIndex.equals",
        "fillna": "MultiIndex.fillna",
        "get_level_values": "MultiIndex.get_level_values",
        "get_loc": "MultiIndex.get_loc",
        "get_loc_level": "MultiIndex.get_loc_level",
        "get_locs": "MultiIndex.get_locs",
        "get_slice_bound": "MultiIndex.get_slice_bound",
        "insert": "MultiIndex.insert",
        "isin": "MultiIndex.isin",
        "memory_usage": "MultiIndex.memory_usage",
        "putmask": "MultiIndex.putmask",
        "remove_unused_levels": "MultiIndex.remove_unused_levels",
        "rename": "Index.set_names",
        "reorder_levels": "MultiIndex.reorder_levels",
        "repeat": "MultiIndex.repeat",
        "set_codes": "MultiIndex.set_codes",
        "set_levels": "MultiIndex.set_levels",
        "slice_locs": "MultiIndex.slice_locs",
        "sortlevel": "MultiIndex.sortlevel",
        "swaplevel": "MultiIndex.swaplevel",
        "to_flat_index": "MultiIndex.to_flat_index",
        "to_frame": "MultiIndex.to_frame",
        "truncate": "MultiIndex.truncate",
        "unique": "MultiIndex.unique",
        "view": "MultiIndex.view",
    },
    "PeriodIndex": {
        "asfreq": "PeriodIndex.asfreq",
        "asof_locs": "PeriodIndex.asof_locs",
        "equals": "DatetimeIndexOpsMixin.equals",
        "get_loc": "PeriodIndex.get_loc",
        "mean": "DatetimeIndexOpsMixin.mean",
        "shift": "PeriodIndex.shift",
        "strftime": "DatelikeOps.strftime",
        "to_numpy": "ExtensionArray.to_numpy",
        "to_timestamp": "PeriodIndex.to_timestamp",
    },
    "RangeIndex": {
        "copy": "RangeIndex.copy",
        "delete": "RangeIndex.delete",
        "equals": "RangeIndex.equals",
        "factorize": "RangeIndex.factorize",
        "get_loc": "RangeIndex.get_loc",
        "insert": "RangeIndex.insert",
        "memory_usage": "RangeIndex.memory_usage",
        "round": "RangeIndex.round",
        "searchsorted": "RangeIndex.searchsorted",
        "sort_values": "RangeIndex.sort_values",
        "symmetric_difference": "RangeIndex.symmetric_difference",
        "tolist": "RangeIndex.tolist",
        "value_counts": "RangeIndex.value_counts",
    },
    "TimedeltaIndex": {
        "as_unit": "DatetimeTimedeltaMixin.as_unit",
        "ceil": "TimelikeOps.ceil",
        "delete": "DatetimeTimedeltaMixin.delete",
        "equals": "DatetimeIndexOpsMixin.equals",
        "floor": "TimelikeOps.floor",
        "get_loc": "TimedeltaIndex.get_loc",
        "insert": "DatetimeTimedeltaMixin.insert",
        "mean": "DatetimeIndexOpsMixin.mean",
        "median": "median",
        "round": "TimelikeOps.round",
        "shift": "DatetimeTimedeltaMixin.shift",
        "std": "TimedeltaArray.std",
        "sum": "TimedeltaArray.sum",
        "to_pytimedelta": "TimedeltaArray.to_pytimedelta",
        "total_seconds": "TimedeltaArray.total_seconds",
    },
}


def name_indexes(*classes: type) -> None:
    """Names each index's methods as the same pandas index prints them.

    Args:
        classes: The index classes, each named after the pandas class it stands for.
    """
    tables = []
    for cls in classes:
        own = _INDEX_OWN.get(cls.__name__, {})
        names = [name for name in dir(cls) if not name.startswith("_")]
        table = {
            name: own.get(name) or _INDEX_COMMON.get(name) or f"Index.{name}" for name in names
        }
        tables.append((cls, table))
    name_each(*tables)


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
            # A wrapper is not copied, so it takes the name most of its classes print.
            shown = list(names.values())
            _rename(function, max(shown, key=shown.count))
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
