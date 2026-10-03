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
        shown = names.pop()
        # A decorated method raises from the function it wraps, so that is named too.
        while function is not None:
            function.__qualname__ = shown  # type: ignore[attr-defined]
            function = getattr(function, "__wrapped__", None)
