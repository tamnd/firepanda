"""`to_xarray`, which hands a frame or a series to xarray the way pandas does.

pandas' own method is three lines: import xarray, and call
`xarray.Dataset.from_dataframe` on a frame or `xarray.DataArray.from_series` on
a series. Those two read pandas' internals, the levels of a multi index and the
extension arrays behind a column, so they need a pandas object and not
something shaped like one. xarray depends on pandas, so wherever xarray can be
imported pandas can be too, and the frame is rebuilt as a pandas frame column
by column with every type kept, then given to xarray. That keeps the answer
xarray's own, including how it fills the gaps of a multi index and which types
it keeps as extension arrays, rather than a copy of it that drifts as xarray
changes.
"""

from __future__ import annotations

import importlib
from typing import Any

_MISSING = "Missing optional dependency '{}'.  Use pip or conda to install {}."

_MASKED = frozenset(
    ["boolean"]
    + [f"{kind}{bits}" for kind in ("Int", "UInt") for bits in (8, 16, 32, 64)]
    + [f"Float{bits}" for bits in (32, 64)]
)


def _module(name: str) -> Any:
    """Imports an optional dependency, with pandas' sentence when it is not installed."""
    try:
        return importlib.import_module(name)
    except ImportError:
        raise ImportError(_MISSING.format(name, name)) from None


def _pandas_values(pandas: Any, column: Any) -> Any:
    """The values of a firepanda series as the array pandas would hold them in.

    A category column is rebuilt from its codes so the categories and their
    order survive, text and the nullable types go through an object array with
    `None` for a missing value, which pandas reads as missing under any type,
    and a moment with a zone is moved through UTC because numpy has no zones.
    Everything else is a numpy array already.
    """
    printed = str(column.dtype)
    if printed == "category":
        return pandas.Categorical.from_codes(
            column.cat.codes.to_numpy(),
            categories=_pandas_index(pandas, column.cat.categories),
            ordered=bool(column.cat.ordered),
        )
    if printed in ("string", "str"):
        return pandas.array(column.to_numpy(dtype=object, na_value=None), dtype="str")
    if printed in _MASKED:
        return pandas.array(column.to_numpy(dtype=object, na_value=None), dtype=printed)
    if printed.startswith("datetime64[") and "," in printed:
        zone = str(column.dt.tz)
        naive = column.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
        return pandas.DatetimeIndex(naive).tz_localize("UTC").tz_convert(zone).array
    return column.to_numpy()


def _pandas_index(pandas: Any, index: Any) -> Any:
    """A firepanda index as a pandas index, a multi index level by level."""
    from ._frame import Series

    if index.nlevels > 1:
        levels = [
            _pandas_values(pandas, Series(index.get_level_values(i))) for i in range(index.nlevels)
        ]
        return pandas.MultiIndex.from_arrays(levels, names=list(index.names))
    return pandas.Index(_pandas_values(pandas, Series(index)), name=index.name)


def _pandas_frame(pandas: Any, frame: Any) -> Any:
    """A firepanda frame as a pandas frame, with repeated column names kept repeated."""
    columns = list(frame.columns)
    arrays = {i: _pandas_values(pandas, frame.iloc[:, i]) for i in range(len(columns))}
    out = pandas.DataFrame(arrays, index=_pandas_index(pandas, frame.index))
    out.columns = pandas.Index(columns)
    return out


def to_xarray(self: Any) -> Any:
    """An xarray `Dataset` of a frame or `DataArray` of a series, as pandas answers it."""
    xarray = _module("xarray")
    pandas = _module("pandas")
    if self.ndim == 1:
        series = pandas.Series(
            _pandas_values(pandas, self), index=_pandas_index(pandas, self.index), name=self.name
        )
        return xarray.DataArray.from_series(series)
    return xarray.Dataset.from_dataframe(_pandas_frame(pandas, self))
