"""`HDFStore`, `read_hdf` and `to_hdf`, the HDF5 store pandas keeps in `pandas.io.pytables`.

pandas writes a Series or a DataFrame into an HDF5 file through PyTables in one
of two layouts. The fixed layout stores the index, the column labels and each
block of same-typed columns as plain arrays, and the table layout stores the
rows in one PyTables table that can be appended to and queried with a where
clause. This module is pandas' own code for both, ported class for class, so a
file either library writes has the same nodes, attributes and types and reads
back in the other. PyTables does the file work and needs numpy, so both are
imported only when a store is opened.

pandas hands its storers blocks of values in the shapes its block manager
keeps them. The port holds the same thing in two small classes: `_Arr` is the
values of a column pandas keeps in an extension array, a text, zoned datetime,
categorical, period or nullable column, and `_Ix` is an index with the parts
the storers read from one. A frame is its column labels, its index and its
blocks, grouped and consolidated by pandas' rules so the blocks land in the
file in the order pandas writes them.
"""

from __future__ import annotations

import copy
import datetime
import functools
import importlib
import importlib.util
import itertools
import math
import os
import re
import sys
import warnings
from contextlib import suppress
from textwrap import dedent
from typing import Any, ClassVar

from ._config import get_option
from ._hdf_expr import PyTablesExpr, Term, ensure_term, is_list_like, maybe_expression
from ._optional import imported
from .errors import (
    AttributeConflictWarning,
    ClosedFileError,
    IncompatibilityWarning,
    PerformanceWarning,
    PossibleDataLossError,
)

try:
    import numpy as np
except ImportError:
    np = None

__all__ = [
    "AppendableFrameTable",
    "AppendableMultiFrameTable",
    "AppendableMultiSeriesTable",
    "AppendableSeriesTable",
    "AppendableTable",
    "BlockManagerFixed",
    "DataCol",
    "DataIndexableCol",
    "Fixed",
    "FrameFixed",
    "GenericDataIndexableCol",
    "GenericFixed",
    "GenericIndexCol",
    "GenericTable",
    "HDFStore",
    "IndexCol",
    "PyTablesExpr",
    "Selection",
    "SeriesFixed",
    "Table",
    "TableIterator",
    "Term",
    "WORMTable",
    "maybe_expression",
    "read_hdf",
    "to_hdf",
]

_version = "0.15.2"
_default_encoding = "UTF-8"
_PACKAGE = os.path.dirname(os.path.abspath(__file__))


def _fp() -> Any:
    import firepanda

    return firepanda


def _stack_level() -> int:
    """How many frames up the first caller outside firepanda is, for a warning."""
    frame: Any = sys._getframe(1)
    level = 1
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(_PACKAGE):
        frame = frame.f_back
        level += 1
    return level


def _pprint(thing: Any) -> str:
    if isinstance(thing, (list, tuple)):
        inner = ", ".join(_pprint(part) for part in thing)
        if isinstance(thing, tuple):
            return f"({inner}{',' if len(thing) == 1 else ''})"
        return f"[{inner}]"
    return str(thing)


def _ensure_encoding(encoding: str | None) -> str:
    if encoding is None:
        encoding = _default_encoding
    return encoding


def _ensure_str(name: Any) -> Any:
    if isinstance(name, str):
        name = str(name)
    return name


def _stringify_path(path: Any) -> Any:
    if isinstance(path, os.PathLike):
        return os.fspath(path)
    return path


def _all_none(*args: Any) -> bool:
    return all(arg is None for arg in args)


def _fill_missing_names(names: Any) -> list[Any]:
    return [f"level_{i}" if name is None else name for i, name in enumerate(names)]


incompatibility_doc = (
    "\nwhere criteria is being ignored as this version [%s] is too old (or\n"
    "not-defined), read the file in and write it out to a new file to upgrade (with\n"
    "the copy_to method)\n"
)
attribute_conflict_doc = (
    "\nthe [%s] attribute of the existing index is [%s] which conflicts with the new\n"
    "[%s], resetting the attribute to None\n"
)
performance_doc = (
    "\nyour performance may suffer as PyTables will pickle object types that it cannot\n"
    "map directly to c-types [inferred_type->%s,key->%s] [items->%s]\n"
)
_FORMAT_MAP = {"f": "fixed", "fixed": "fixed", "t": "table", "table": "table"}

_table_mod: Any = None
_table_file_open_policy_is_strict = False


def _tables() -> Any:
    global _table_mod
    global _table_file_open_policy_is_strict
    if _table_mod is None:
        import tables

        _table_mod = tables
        with suppress(AttributeError):
            _table_file_open_policy_is_strict = tables.file._FILE_OPEN_POLICY == "strict"
    return _table_mod


# The missing values and the kinds of values, as pandas' `isna` and
# `lib.infer_dtype` see them.


def _is_na(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if type(value).__name__ in ("NAType", "NaTType"):
        return True
    if np is not None and isinstance(value, np.floating) and np.isnan(value):
        return True
    if np is not None and isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    return False


def _isna_array(values: Any) -> Any:
    if values.dtype.kind == "f":
        return np.isnan(values)
    if values.dtype.kind in "mM":
        return np.isnat(values)
    if values.dtype.kind == "O":
        flat = [_is_na(v) for v in values.ravel()]
        return np.asarray(flat, dtype=bool).reshape(values.shape)
    return np.zeros(values.shape, dtype=bool)


def _infer(values: Any, skipna: bool = False) -> str:
    """The kind of the values, in the words of pandas' `lib.infer_dtype`."""
    if isinstance(values, _Ix):
        if isinstance(values.values, _Arr) and values.values.kind == "str":
            return "string"
        values = values.plain()
    elif isinstance(values, _Arr):
        if values.kind == "str":
            return "string"
        elif values.kind == "cat":
            return "categorical"
        elif values.kind == "tz":
            return "datetime64"
        elif values.kind == "period":
            return "period"
        else:
            values = values.data
    if np is not None and isinstance(values, np.ndarray) and values.dtype.kind != "O":
        kind = values.dtype.kind
        if values.size == 0 and kind in "iufb":
            pass
        return {
            "i": "integer",
            "u": "integer",
            "f": "floating",
            "b": "boolean",
            "M": "datetime64",
            "m": "timedelta64",
            "S": "bytes",
            "U": "string",
            "c": "complex",
        }.get(kind, "mixed")
    if np is not None and isinstance(values, np.ndarray):
        items = list(values.ravel())
    else:
        items = list(values)
    if skipna:
        items = [v for v in items if not _is_na(v)]
    if not items:
        return "empty"
    kinds = set()
    for v in items:
        if isinstance(v, bool) or (np is not None and isinstance(v, np.bool_)):
            kinds.add("boolean")
        elif isinstance(v, str):
            kinds.add("string")
        elif isinstance(v, bytes):
            kinds.add("bytes")
        elif isinstance(v, int) or (np is not None and isinstance(v, np.integer)):
            kinds.add("integer")
        elif isinstance(v, float) or (np is not None and isinstance(v, np.floating)):
            kinds.add("nan" if _is_na(v) else "floating")
        elif v is None or type(v).__name__ in ("NAType", "NaTType"):
            kinds.add("none")
        elif isinstance(v, datetime.datetime) or type(v).__name__ == "Timestamp":
            kinds.add("datetime")
        elif isinstance(v, datetime.date):
            kinds.add("date")
        elif isinstance(v, datetime.timedelta) or type(v).__name__ == "Timedelta":
            kinds.add("timedelta")
        elif isinstance(v, complex):
            kinds.add("complex")
        elif type(v).__name__ == "Decimal":
            kinds.add("decimal")
        elif type(v).__name__ == "Period":
            kinds.add("period")
        else:
            kinds.add("mixed")
    if kinds == {"nan"}:
        return "floating"
    if kinds <= {"floating", "nan"}:
        return "floating"
    if len(kinds) == 1:
        return next(iter(kinds))
    if kinds <= {"integer", "floating", "nan"}:
        return "mixed-integer-float"
    if "integer" in kinds:
        return "mixed-integer"
    if kinds <= {"datetime", "none", "nan"} and not skipna and "datetime" in kinds:
        return "datetime" if kinds <= {"datetime", "nan"} else "mixed"
    return "mixed"


def _all_str(values: Any) -> bool:
    """Whether pandas infers text: every value is text or missing and one is text."""
    seen = False
    for v in values:
        if isinstance(v, str):
            seen = True
        elif not _is_na(v):
            return False
    return seen


# The values of a column, as pandas holds them in a block.


def _ext_names(dtype_name: str) -> tuple[str, str]:
    """The array and dtype class names pandas uses for an extension type."""
    if dtype_name.startswith(("Int", "UInt")):
        return "IntegerArray", f"{dtype_name}Dtype"
    if dtype_name.startswith("Float"):
        return "FloatingArray", f"{dtype_name}Dtype"
    if dtype_name == "boolean":
        return "BooleanArray", "BooleanDtype"
    if dtype_name.startswith("period"):
        return "PeriodArray", "PeriodDtype"
    if dtype_name.startswith("interval"):
        return "IntervalArray", "IntervalDtype"
    if dtype_name.startswith("Sparse"):
        return "SparseArray", "SparseDtype"
    if dtype_name.endswith("[pyarrow]"):
        return "ArrowExtensionArray", "ArrowDtype"
    return "ExtensionArray", "ExtensionDtype"


class _Arr:
    """The values of a column pandas keeps in an extension array.

    `kind` is "str" for text, with the values in an object array and nan for a
    missing value; "tz" for zoned datetimes, as the naive UTC datetimes pandas
    stores; "cat" for a categorical, as its codes; "period" for periods, as
    their ordinals; and "ext" for any other extension type, as an object array.
    """

    def __init__(
        self,
        kind: str,
        data: Any,
        dtype_name: str,
        tz: str | None = None,
        categories: Any = None,
        ordered: Any = None,
        objects: Any = None,
    ) -> None:
        self.kind = kind
        self.data = data
        self.dtype_name = dtype_name
        self.tz = tz
        self.categories = categories
        self.ordered = ordered
        self.objects = objects

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def unit(self) -> str:
        return np.datetime_data(self.data.dtype)[0]

    def _with(self, data: Any, objects: Any = None) -> _Arr:
        return _Arr(
            self.kind, data, self.dtype_name, self.tz, self.categories, self.ordered, objects
        )

    @property
    def T(self) -> _Arr:
        return self._with(self.data.T)

    def __len__(self) -> int:
        return len(self.data)

    def take(self, locs: Any) -> _Arr:
        objects = None if self.objects is None else self.objects[locs]
        return self._with(self.data[locs], objects)

    def to_numpy(self) -> Any:
        return self.data


def _dtype_name_of(values: Any) -> str:
    if isinstance(values, _Arr):
        return values.dtype_name
    return values.dtype.name


def _values_of(series: Any) -> Any:
    """The values of a firepanda Series, as pandas would hold them."""
    dtype = str(series.dtype)
    if dtype == "category":
        cat = series.cat
        codes = np.asarray(cat.codes.to_numpy())
        categories = _plain_of(_values_of(_fp().Series(cat.categories)))
        return _Arr("cat", codes, "category", categories=categories, ordered=bool(cat.ordered))
    if dtype in ("str", "string"):
        data = series.to_numpy(dtype=object, na_value=np.nan)
        return _Arr("str", np.asarray(data, dtype=object), "str")
    if dtype.startswith("datetime64[") and "," in dtype:
        naive = series.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
        return _Arr("tz", np.asarray(naive), dtype, tz=str(series.dt.tz))
    if dtype.startswith("period"):
        objects = np.empty(len(series), dtype=object)
        objects[:] = list(series)
        ordinals = np.asarray(
            [-(2**63) if _is_na(p) else p.ordinal for p in objects], dtype=np.int64
        )
        return _Arr("period", ordinals, dtype, objects=objects)
    try:
        np_dtype = np.dtype(dtype)
    except TypeError:
        np_dtype = None
    if np_dtype is not None and dtype == np_dtype.name:
        if np_dtype.kind == "O":
            data = np.empty(len(series), dtype=object)
            data[:] = list(series.to_numpy(dtype=object))
            if len(data) and all(isinstance(v, complex) for v in data):
                # firepanda holds complex numbers as objects, pandas as complex128.
                return data.astype(np.complex128)
            return data
        return np.asarray(series.to_numpy())
    data = np.empty(len(series), dtype=object)
    data[:] = list(series)
    return _Arr("ext", data, dtype)


def _plain_of(values: Any) -> Any:
    """`np.asarray` of pandas values."""
    if isinstance(values, _Arr):
        if values.kind == "tz":
            return values.data
        if values.kind == "period":
            return values.objects
        if values.kind == "cat":
            out = np.empty(values.data.shape, dtype=object)
            cats = list(values.categories)
            out[:] = [np.nan if c < 0 else cats[c] for c in values.data]
            return out
        return values.data
    return values


def _series_of(values: Any, name: Any = None, index: Any = None, dtype: Any = None) -> Any:
    """A firepanda Series holding 1D values, as pandas builds one from them."""
    fp = _fp()
    kwargs: dict[str, Any] = {"name": name}
    if isinstance(values, _Arr):
        if values.kind == "str":
            s = fp.Series(list(values.data), dtype="str", **kwargs)
        elif values.kind == "tz":
            s = fp.Series(values.data, **kwargs).dt.tz_localize("UTC").dt.tz_convert(values.tz)
        elif values.kind == "cat":
            cats = values.categories
            if cats is None:
                cats = []
            categories = fp.Index(list(cats)) if len(cats) else fp.Index([])
            cat = fp.Categorical.from_codes(
                np.asarray(values.data, dtype=np.int8) if len(cats) < 128 else values.data,
                categories=categories,
                ordered=bool(values.ordered),
            )
            s = fp.Series(cat, **kwargs)
        elif values.kind == "period":
            s = fp.Series(values.objects, **kwargs)
        else:
            s = fp.Series(list(values.data), dtype=values.dtype_name, **kwargs)
    else:
        kind = values.dtype.kind
        if kind == "m":
            unit = np.datetime_data(values.dtype)[0]
            ints = values.view("i8")
            cells = [None if np.isnat(v) else int(i) for v, i in zip(values, ints, strict=True)]
            s = fp.to_timedelta(fp.Series(cells), unit=unit).dt.as_unit(unit)
            if name is not None:
                s = s.rename(name)
        elif kind == "O":
            cells = list(values)
            if dtype in ("str", "string") or (dtype is None and _all_str(cells)):
                s = fp.Series(cells, dtype="str", **kwargs)
            else:
                s = fp.Series(cells, dtype=object, **kwargs)
        elif kind == "S":
            s = fp.Series(list(values), dtype=object, **kwargs)
        else:
            s = fp.Series(values, **kwargs)
        if dtype in ("str", "string") and str(s.dtype) not in ("str", "string"):
            s = s.astype(dtype)
    if index is not None:
        s = s.set_axis(index)
    return s


# An index, with the parts the storers read.


def _freq_out(freq: Any) -> Any:
    """A frequency as a file stores it, a pandas offset once pandas reads it."""
    if freq is None:
        return None
    freqstr = getattr(freq, "freqstr", None)
    if freqstr is None:
        return freq
    if importlib.util.find_spec("pandas") is not None:
        return _Offset(freqstr)
    return freq


def _freq_in(freq: Any) -> Any:
    """A stored frequency as the text firepanda reads."""
    if freq is None:
        return None
    if isinstance(freq, str):
        return freq
    return getattr(freq, "freqstr", None)


class _Module:
    def __reduce__(self) -> Any:
        return (importlib.import_module, ("pandas.tseries.frequencies",))


class _Attr:
    def __call__(self, freqstr: str) -> Any:
        return importlib.import_module("pandas.tseries.frequencies").to_offset(freqstr)

    def __reduce__(self) -> Any:
        return (getattr, (_Module(), "to_offset"))


class _Offset:
    """A frequency that unpickles as pandas' offset for the same text."""

    def __init__(self, freqstr: str) -> None:
        self.freqstr = freqstr

    def __reduce__(self) -> Any:
        return (_Attr(), (self.freqstr,))

    def __eq__(self, other: object) -> bool:
        return self.freqstr == getattr(other, "freqstr", other)

    def __hash__(self) -> int:
        return hash(self.freqstr)

    def __repr__(self) -> str:
        return f"<{self.freqstr}>"


def _tz_attr(tz: str | None) -> Any:
    """A zone as pandas stores it on a node: UTC as the object, others by name."""
    if tz is None:
        return None
    if tz == "UTC":
        return datetime.UTC
    return tz


def _tz_info(tz: str | None) -> Any:
    """A zone as pandas keeps it on an index, a tzinfo."""
    if tz is None:
        return None
    if tz == "UTC":
        return datetime.UTC
    try:
        import zoneinfo

        return zoneinfo.ZoneInfo(tz)
    except (ImportError, ValueError, KeyError, OSError):
        return tz


def _tz_name(tz: Any) -> str | None:
    """A stored zone as the name firepanda reads."""
    if tz is None:
        return None
    if isinstance(tz, str):
        return tz
    if tz is datetime.UTC:
        return "UTC"
    for attr in ("key", "zone"):
        name = getattr(tz, attr, None)
        if isinstance(name, str):
            return name
    return str(tz)


class _Ix:
    """An index, as the storers read it."""

    def __init__(
        self,
        cls: str,
        values: Any,
        name: Any = None,
        freq: Any = None,
        levels: list[_Ix] | None = None,
        codes: list[Any] | None = None,
        names: list[Any] | None = None,
        labels: list[Any] | None = None,
    ) -> None:
        self.cls = cls
        self.values = values
        self.name = name
        self.freq = freq
        self.levels = levels
        self.codes = codes
        self.names = names if names is not None else [name]
        self._labels = labels

    @property
    def nlevels(self) -> int:
        return len(self.levels) if self.levels is not None else 1

    @property
    def dtype_name(self) -> str:
        if self.cls == "MultiIndex":
            return "object"
        return _dtype_name_of(self.values)

    @property
    def tz(self) -> str | None:
        return getattr(self.values, "tz", None)

    @property
    def tzinfo(self) -> Any:
        return _tz_info(self.tz)

    def plain(self) -> Any:
        if self.cls == "MultiIndex":
            out = np.empty(len(self), dtype=object)
            out[:] = self.labels()
            return out
        return _plain_of(self.values)

    def labels(self) -> list[Any]:
        if self._labels is None:
            if self.cls == "MultiIndex":
                assert self.levels is not None and self.codes is not None
                levs = [lev.labels() for lev in self.levels]
                self._labels = [
                    tuple(np.nan if c < 0 else levs[i][c] for i, c in enumerate(row))
                    for row in zip(*self.codes, strict=True)
                ]
            else:
                self._labels = [_scalar(v) for v in _series_of(self.values)]
        return self._labels

    def __len__(self) -> int:
        if self.cls == "MultiIndex":
            assert self.codes is not None
            return len(self.codes[0]) if self.codes else 0
        return len(self.values)

    def __iter__(self) -> Any:
        return iter(self.labels())

    @property
    def is_unique(self) -> bool:
        seen: list[Any] = []
        hashed = set()
        for v in self.labels():
            key = "__nan__" if _is_na(v) else v
            try:
                if key in hashed:
                    return False
                hashed.add(key)
            except TypeError:
                if key in seen:
                    return False
                seen.append(key)
        return True

    def take(self, locs: Any) -> _Ix:
        locs = list(locs)
        labels = [self.labels()[i] for i in locs]
        if self.cls == "MultiIndex":
            assert self.codes is not None
            return _Ix(
                "MultiIndex",
                None,
                levels=self.levels,
                codes=[np.asarray(c)[locs] for c in self.codes],
                names=self.names,
                labels=labels,
            )
        freq = self.freq
        steps = {b - a for a, b in itertools.pairwise(locs)}
        if steps and steps != {1}:
            freq = None
        values = self.values.take(locs) if isinstance(self.values, _Arr) else self.values[locs]
        cls = "Index" if self.cls == "RangeIndex" else self.cls
        return _Ix(cls, values, name=self.name, freq=freq, labels=labels)


def _scalar(value: Any) -> Any:
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return value


def _ix_of(index: Any) -> _Ix:
    """A firepanda index, as the storers read it."""
    fp = _fp()
    cls = type(index).__name__
    if cls == "MultiIndex":
        levels = [_ix_of(lev) for lev in index.levels]
        codes = [
            np.asarray(fp.Series(c).to_numpy() if not hasattr(c, "dtype") else c)
            for c in index.codes
        ]
        codes = [np.asarray(list(c), dtype=np.int8) if c.dtype == object else c for c in codes]
        return _Ix("MultiIndex", None, levels=levels, codes=codes, names=list(index.names))
    if cls == "RangeIndex":
        values = np.arange(index.start, index.stop, index.step, dtype=np.int64)
    else:
        values = _values_of(fp.Series(index))
    freq = None
    if cls in ("DatetimeIndex", "TimedeltaIndex", "PeriodIndex"):
        freq = _freq_out(getattr(index, "freq", None))
    labels = [_scalar(v) for v in index]
    return _Ix(cls, values, name=index.name, freq=freq, labels=labels)


def _range_ix(n: int) -> _Ix:
    return _Ix("RangeIndex", np.arange(n, dtype=np.int64))


def _labels_ix(labels: list[Any]) -> _Ix:
    """The index pandas builds from a list of labels."""
    if labels and all(isinstance(v, str) for v in labels):
        data = np.empty(len(labels), dtype=object)
        data[:] = labels
        return _Ix("Index", _Arr("str", data, "str"), labels=list(labels))
    return _ix_of(_fp().Index(labels))


# A frame and a Series, as the storers write them.


class _Block:
    def __init__(self, values: Any, locs: list[int]) -> None:
        self.values = values
        self.locs = list(locs)

    @property
    def dtype_name(self) -> str:
        return _dtype_name_of(self.values)

    @property
    def can_consolidate(self) -> bool:
        return not isinstance(self.values, _Arr)

    @property
    def is_str(self) -> bool:
        return isinstance(self.values, _Arr) and self.values.kind == "str"

    def row(self, j: int) -> Any:
        """The values of the block's j-th column, 1D."""
        v = self.values
        if isinstance(v, _Arr):
            if v.ndim == 2:
                return v._with(v.data[j])
            return v
        return v[j]


def _stack(parts: list[Any]) -> Any:
    first = parts[0]
    if isinstance(first, _Arr):
        return first._with(np.stack([p.data for p in parts]))
    return np.stack(parts)


class _Series:
    ndim = 1

    def __init__(self, values: Any, index: _Ix, name: Any = None) -> None:
        self.values = values
        self.index = index
        self.name = name

    @property
    def empty(self) -> bool:
        return len(self.index) == 0

    def to_frame(self, name: Any) -> _Frame:
        return _Frame(_labels_ix([name]), self.index, [self.values])


class _Frame:
    ndim = 2

    def __init__(
        self,
        columns: _Ix,
        index: _Ix,
        data: list[Any] | None = None,
        blocks: list[_Block] | None = None,
        src: Any = None,
    ) -> None:
        self.columns = columns
        self.index = index
        self.src = src
        if blocks is None:
            blocks = _form_blocks(data or [])
        self.blocks = blocks

    @property
    def empty(self) -> bool:
        return len(self.index) == 0 or len(self.columns) == 0

    @property
    def axes(self) -> list[_Ix]:
        return [self.index, self.columns]

    def column(self, i: int) -> Any:
        for blk in self.blocks:
            if i in blk.locs:
                return blk.row(blk.locs.index(i))
        raise IndexError(i)

    @property
    def is_consolidated(self) -> bool:
        names = [b.dtype_name for b in self.blocks if b.can_consolidate]
        return len(names) == len(set(names))

    def consolidate(self) -> _Frame:
        if self.is_consolidated:
            return self
        ordered = sorted(self.blocks, key=lambda b: (b.can_consolidate, b.dtype_name))
        blocks = []
        for (can, _), group in itertools.groupby(
            ordered, key=lambda b: (b.can_consolidate, b.dtype_name)
        ):
            group = list(group)
            if not can:
                blocks.extend(group)
                continue
            pairs = sorted(
                ((loc, (bi, j)) for bi, blk in enumerate(group) for j, loc in enumerate(blk.locs)),
                key=lambda p: p[0],
            )
            rows = [group[bi].row(j) for _, (bi, j) in pairs]
            blocks.append(_Block(_stack(rows), [loc for loc, _ in pairs]))
        return _Frame(self.columns, self.index, blocks=blocks, src=self.src)

    def take(self, positions: list[int]) -> _Frame:
        """The columns at `positions`, grouped by block as pandas' reindex groups them."""
        where = {}
        for bi, blk in enumerate(self.blocks):
            for j, loc in enumerate(blk.locs):
                where[loc] = (bi, j)
        groups: dict[int, list[tuple[int, int]]] = {}
        for new, pos in enumerate(positions):
            bi, j = where[pos]
            groups.setdefault(bi, []).append((new, j))
        blocks = []
        for bi, members in groups.items():
            blk = self.blocks[bi]
            if blk.can_consolidate:
                rows = [blk.row(j) for _, j in members]
                blocks.append(_Block(_stack(rows), [new for new, _ in members]))
            else:
                for new, j in members:
                    blocks.append(_Block(blk.row(j), [new]))
        columns = self.columns.take(positions)
        return _Frame(columns, self.index, blocks=blocks, src=self.src)

    def reindex_columns(self, labels: list[Any]) -> _Frame:
        current = self.columns.labels()
        positions = []
        for lab in labels:
            hits = [i for i, c in enumerate(current) if _same_label(c, lab)]
            if not hits:
                raise KeyError(f"{[lab]} not in index")
            positions.extend(hits)
        return self.take(positions)


def _same_label(a: Any, b: Any) -> bool:
    if _is_na(a) and _is_na(b):
        return True
    try:
        return (bool(a == b) and type(a) is not bool) or a is b
    except (TypeError, ValueError):
        return False


def _form_blocks(data: list[Any]) -> list[_Block]:
    """Blocks as pandas holds a frame's columns: runs of a numpy dtype share one."""
    blocks: list[_Block] = []
    for i, values in enumerate(data):
        if isinstance(values, _Arr):
            if values.kind == "tz":
                blocks.append(_Block(values._with(values.data.reshape(1, -1)), [i]))
            else:
                blocks.append(_Block(values, [i]))
            continue
        last = blocks[-1] if blocks else None
        if (
            last is not None
            and not isinstance(last.values, _Arr)
            and last.values.dtype == values.dtype
            and last.locs[-1] == i - 1
        ):
            last.values = np.concatenate([last.values, values.reshape(1, -1)])
            last.locs.append(i)
        else:
            blocks.append(_Block(values.reshape(1, -1), [i]))
    return blocks


def _obj_of(value: Any) -> Any:
    """A firepanda Series or DataFrame as the storers write it."""
    if isinstance(value, (_Series, _Frame)):
        return value
    if value.ndim == 1:
        return _Series(_values_of(value), _ix_of(value.index), value.name)
    columns = _ix_of(value.columns)
    data = [_values_of(value.iloc[:, i]) for i in range(value.shape[1])]
    return _Frame(columns, _ix_of(value.index), data, src=value)


def _is_value(value: Any) -> bool:
    if isinstance(value, (_Series, _Frame)):
        return True
    fp = _fp()
    return isinstance(value, (fp.Series, fp.DataFrame))


def _index_of(ix: _Ix) -> Any:
    """A firepanda index from one the port holds."""
    fp = _fp()
    if ix.cls == "MultiIndex":
        assert ix.levels is not None and ix.codes is not None
        return fp.MultiIndex(
            levels=[_index_of(lev) for lev in ix.levels], codes=ix.codes, names=ix.names
        )
    if ix.cls == "RangeIndex":
        return fp.RangeIndex(len(ix), name=ix.name)
    values = ix.values
    s = _series_of(values)
    if ix.cls == "DatetimeIndex":
        return fp.DatetimeIndex(s, freq=_freq_in(ix.freq), name=ix.name)
    if ix.cls == "TimedeltaIndex":
        return fp.TimedeltaIndex(s, freq=_freq_in(ix.freq), name=ix.name)
    if ix.cls == "PeriodIndex":
        return fp.PeriodIndex(s, name=ix.name)
    return fp.Index(s, name=ix.name)


# The module functions.


def to_hdf(
    path_or_buf: Any,
    key: str,
    value: Any,
    mode: str = "a",
    complevel: int | None = None,
    complib: str | None = None,
    append: bool = False,
    format: str | None = None,
    index: bool = True,
    min_itemsize: Any = None,
    nan_rep: Any = None,
    dropna: bool | None = None,
    data_columns: Any = None,
    errors: str = "strict",
    encoding: str = "UTF-8",
) -> None:
    if append:

        def f(store: HDFStore) -> None:
            store.append(
                key,
                value,
                format=format,
                index=index,
                min_itemsize=min_itemsize,
                nan_rep=nan_rep,
                dropna=dropna,
                data_columns=data_columns,
                errors=errors,
                encoding=encoding,
            )

    else:

        def f(store: HDFStore) -> None:
            store.put(
                key,
                value,
                format=format,
                index=index,
                min_itemsize=min_itemsize,
                nan_rep=nan_rep,
                data_columns=data_columns,
                errors=errors,
                encoding=encoding,
                dropna=dropna,
            )

    if isinstance(path_or_buf, HDFStore):
        f(path_or_buf)
    else:
        path_or_buf = _stringify_path(path_or_buf)
        with HDFStore(path_or_buf, mode=mode, complevel=complevel, complib=complib) as store:
            f(store)


def _to_hdf(
    self: Any,
    path_or_buf: Any,
    *,
    key: str,
    mode: str = "a",
    complevel: int | None = None,
    complib: str | None = None,
    append: bool = False,
    format: str | None = None,
    index: bool = True,
    min_itemsize: Any = None,
    nan_rep: Any = None,
    dropna: bool | None = None,
    data_columns: Any = None,
    errors: str = "strict",
    encoding: str = "UTF-8",
) -> None:
    """Write the object to an HDF5 file under `key`, with pandas' layout."""
    to_hdf(
        path_or_buf,
        key,
        self,
        mode=mode,
        complevel=complevel,
        complib=complib,
        append=append,
        format=format,
        index=index,
        min_itemsize=min_itemsize,
        nan_rep=nan_rep,
        dropna=dropna,
        data_columns=data_columns,
        errors=errors,
        encoding=encoding,
    )


_to_hdf.__name__ = "to_hdf"
_to_hdf.__qualname__ = "NDFrame.to_hdf"


def read_hdf(
    path_or_buf: Any,
    key: Any = None,
    mode: str = "r",
    errors: str = "strict",
    where: Any = None,
    start: int | None = None,
    stop: int | None = None,
    columns: Any = None,
    iterator: bool = False,
    chunksize: int | None = None,
    **kwargs: Any,
) -> Any:
    """Read a Series or DataFrame from an HDF5 file pandas' way."""
    if mode not in ["r", "r+", "a"]:
        raise ValueError(
            f"mode {mode} is not allowed while performing a read. Allowed modes are r, r+ and a."
        )
    if where is not None:
        where = ensure_term(where)
    if isinstance(path_or_buf, HDFStore):
        if not path_or_buf.is_open:
            raise OSError("The HDFStore must be open for reading.")
        store = path_or_buf
        auto_close = False
    else:
        path_or_buf = _stringify_path(path_or_buf)
        if not isinstance(path_or_buf, str):
            raise NotImplementedError("Support for generic buffers has not been implemented.")
        try:
            exists = os.path.exists(path_or_buf)
        except (TypeError, ValueError):
            exists = False
        if not exists:
            raise FileNotFoundError(f"File {path_or_buf} does not exist")
        store = HDFStore(path_or_buf, mode=mode, errors=errors, **kwargs)
        auto_close = True
    try:
        if key is None:
            groups = store.groups()
            if len(groups) == 0:
                raise ValueError(
                    "Dataset(s) incompatible with Pandas data types, "
                    "not table, or no datasets found in HDF5 file."
                )
            candidate_only_group = groups[0]
            for group_to_check in groups[1:]:
                if not _is_metadata_of(group_to_check, candidate_only_group):
                    raise ValueError(
                        "key must be provided when HDF5 file contains multiple datasets."
                    )
            key = candidate_only_group._v_pathname
        return store.select(
            key,
            where=where,
            start=start,
            stop=stop,
            columns=columns,
            iterator=iterator,
            chunksize=chunksize,
            auto_close=auto_close,
        )
    except (ValueError, TypeError, LookupError):
        if not isinstance(path_or_buf, HDFStore):
            with suppress(AttributeError):
                store.close()
        raise


read_hdf.__module__ = "firepanda"


def _is_metadata_of(group: Any, parent_group: Any) -> bool:
    if group._v_depth <= parent_group._v_depth:
        return False
    current = group
    while current._v_depth > 1:
        parent = current._v_parent
        if parent == parent_group and current._v_name == "meta":
            return True
        current = current._v_parent
    return False


class HDFStore:
    """A dict-like view of an HDF5 file holding pandas objects."""

    _handle: Any
    _mode: str

    def __init__(
        self,
        path: Any,
        mode: str = "a",
        complevel: int | None = None,
        complib: Any = None,
        fletcher32: bool = False,
        **kwargs: Any,
    ) -> None:
        if "format" in kwargs:
            raise ValueError("format is not a defined argument for HDFStore")
        tables = imported("tables")
        if complib is not None and complib not in tables.filters.all_complibs:
            raise ValueError(f"complib only supports {tables.filters.all_complibs} compression.")
        if complib is None and complevel is not None:
            complib = tables.filters.default_complib
        self._path = _stringify_path(path)
        if mode is None:
            mode = "a"
        self._mode = mode
        self._handle = None
        self._complevel = complevel if complevel else 0
        self._complib = complib
        self._fletcher32 = fletcher32
        self._filters = None
        self.open(mode=mode, **kwargs)

    def __fspath__(self) -> str:
        return self._path

    @property
    def root(self) -> Any:
        self._check_if_open()
        assert self._handle is not None
        return self._handle.root

    @property
    def filename(self) -> str:
        return self._path

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.put(key, value)

    def __delitem__(self, key: str) -> int | None:
        return self.remove(key)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        try:
            return self.get(name)
        except (KeyError, ClosedFileError):
            pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __contains__(self, key: str) -> bool:
        node = self.get_node(key)
        if node is not None:
            name = node._v_pathname
            if key in (name, name[1:]):
                return True
        return False

    def __len__(self) -> int:
        return len(self.groups())

    def __repr__(self) -> str:
        pstr = _pprint(self._path)
        return f"{type(self)}\nFile path: {pstr}\n"

    def __enter__(self) -> HDFStore:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def keys(self, include: str = "pandas") -> list[str]:
        if include == "pandas":
            return [n._v_pathname for n in self.groups()]
        elif include == "native":
            assert self._handle is not None
            return [n._v_pathname for n in self._handle.walk_nodes("/", classname="Table")]
        raise ValueError(f"`include` should be either 'pandas' or 'native' but is '{include}'")

    def __iter__(self) -> Any:
        return iter(self.keys())

    def items(self) -> Any:
        for g in self.groups():
            yield (g._v_pathname, g)

    def open(self, mode: str = "a", **kwargs: Any) -> None:
        tables = _tables()
        if self._mode != mode:
            if self._mode in ["a", "w"] and mode in ["r", "r+"]:
                pass
            elif mode in ["w"] and self.is_open:
                raise PossibleDataLossError(
                    f"Re-opening the file [{self._path}] with mode [{self._mode}] "
                    "will delete the current file!"
                )
            self._mode = mode
        if self.is_open:
            self.close()
        if self._complevel and self._complevel > 0:
            self._filters = _tables().Filters(
                self._complevel, self._complib, fletcher32=self._fletcher32
            )
        if _table_file_open_policy_is_strict and self.is_open:
            msg = "Cannot open HDF5 file, which is already opened, even in read-only mode."
            raise ValueError(msg)
        self._handle = tables.open_file(self._path, self._mode, **kwargs)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None

    @property
    def is_open(self) -> bool:
        if self._handle is None:
            return False
        return bool(self._handle.isopen)

    def flush(self, fsync: bool = False) -> None:
        if self._handle is not None:
            self._handle.flush()
            if fsync:
                with suppress(OSError):
                    os.fsync(self._handle.fileno())

    def get(self, key: str) -> Any:
        group = self.get_node(key)
        if group is None:
            raise KeyError(f"No object named {key} in the file")
        return self._read_group(group)

    def select(
        self,
        key: str,
        where: Any = None,
        start: Any = None,
        stop: Any = None,
        columns: Any = None,
        iterator: bool = False,
        chunksize: int | None = None,
        auto_close: bool = False,
    ) -> Any:
        group = self.get_node(key)
        if group is None:
            raise KeyError(f"No object named {key} in the file")
        where = ensure_term(where)
        s = self._create_storer(group)
        s.infer_axes()

        def func(_start: Any, _stop: Any, _where: Any) -> Any:
            return s.read(start=_start, stop=_stop, where=_where, columns=columns)

        it = TableIterator(
            self,
            s,
            func,
            where=where,
            nrows=s.nrows,
            start=start,
            stop=stop,
            iterator=iterator,
            chunksize=chunksize,
            auto_close=auto_close,
        )
        return it.get_result()

    def select_as_coordinates(
        self, key: str, where: Any = None, start: int | None = None, stop: int | None = None
    ) -> Any:
        where = ensure_term(where)
        tbl = self.get_storer(key)
        if not isinstance(tbl, Table):
            raise TypeError("can only read_coordinates with a table")
        return tbl.read_coordinates(where=where, start=start, stop=stop)

    def select_column(
        self, key: str, column: str, start: int | None = None, stop: int | None = None
    ) -> Any:
        tbl = self.get_storer(key)
        if not isinstance(tbl, Table):
            raise TypeError("can only read_column with a table")
        return tbl.read_column(column=column, start=start, stop=stop)

    def select_as_multiple(
        self,
        keys: Any,
        where: Any = None,
        selector: Any = None,
        columns: Any = None,
        start: Any = None,
        stop: Any = None,
        iterator: bool = False,
        chunksize: int | None = None,
        auto_close: bool = False,
    ) -> Any:
        where = ensure_term(where)
        if isinstance(keys, (list, tuple)) and len(keys) == 1:
            keys = keys[0]
        if isinstance(keys, str):
            return self.select(
                key=keys,
                where=where,
                columns=columns,
                start=start,
                stop=stop,
                iterator=iterator,
                chunksize=chunksize,
                auto_close=auto_close,
            )
        if not isinstance(keys, (list, tuple)):
            raise TypeError("keys must be a list/tuple")
        if not len(keys):
            raise ValueError("keys must have a non-zero length")
        if selector is None:
            selector = keys[0]
        tbls = [self.get_storer(k) for k in keys]
        s = self.get_storer(selector)
        nrows = None
        for t, k in itertools.chain([(s, selector)], zip(tbls, keys, strict=True)):
            if t is None:
                raise KeyError(f"Invalid table [{k}]")
            if not t.is_table:
                raise TypeError(
                    f"object [{t.pathname}] is not a table, and cannot be used in all "
                    "select as multiple"
                )
            if nrows is None:
                nrows = t.nrows
            elif t.nrows != nrows:
                raise ValueError("all tables must have exactly the same nrows!")
        _tbls = [x for x in tbls if isinstance(x, Table)]
        axis = {t.non_index_axes[0][0] for t in _tbls}.pop()

        def func(_start: Any, _stop: Any, _where: Any) -> Any:
            objs = [t.read(where=_where, columns=columns, start=_start, stop=_stop) for t in tbls]
            return _fp().concat(objs, axis=axis, verify_integrity=False)

        it = TableIterator(
            self,
            s,
            func,
            where=where,
            nrows=nrows,
            start=start,
            stop=stop,
            iterator=iterator,
            chunksize=chunksize,
            auto_close=auto_close,
        )
        return it.get_result(coordinates=True)

    def put(
        self,
        key: str,
        value: Any,
        format: Any = None,
        index: bool = True,
        append: bool = False,
        complib: Any = None,
        complevel: int | None = None,
        min_itemsize: Any = None,
        nan_rep: Any = None,
        data_columns: Any = None,
        encoding: Any = None,
        errors: str = "strict",
        track_times: bool = True,
        dropna: bool = False,
    ) -> None:
        if format is None:
            format = get_option("io.hdf.default_format") or "fixed"
        format = self._validate_format(format)
        self._write_to_group(
            key,
            value,
            format=format,
            index=index,
            append=append,
            complib=complib,
            complevel=complevel,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            data_columns=data_columns,
            encoding=encoding,
            errors=errors,
            track_times=track_times,
            dropna=dropna,
        )

    def remove(self, key: str, where: Any = None, start: Any = None, stop: Any = None) -> Any:
        where = ensure_term(where)
        try:
            s = self.get_storer(key)
        except KeyError:
            raise
        except AssertionError:
            raise
        except Exception as err:
            if where is not None:
                raise ValueError("trying to remove a node with a non-None where clause!") from err
            node = self.get_node(key)
            if node is not None:
                node._f_remove(recursive=True)
                return None
        if _all_none(where, start, stop):
            s.group._f_remove(recursive=True)
            return None
        if not s.is_table:
            raise ValueError("can only remove with where on objects written as tables")
        return s.delete(where=where, start=start, stop=stop)

    def append(
        self,
        key: str,
        value: Any,
        format: Any = None,
        axes: Any = None,
        index: Any = True,
        append: bool = True,
        complib: Any = None,
        complevel: int | None = None,
        columns: Any = None,
        min_itemsize: Any = None,
        nan_rep: Any = None,
        chunksize: int | None = None,
        expectedrows: Any = None,
        dropna: bool | None = None,
        data_columns: Any = None,
        encoding: Any = None,
        errors: str = "strict",
    ) -> None:
        if columns is not None:
            raise TypeError("columns is not a supported keyword in append, try data_columns")
        if dropna is None:
            dropna = get_option("io.hdf.dropna_table")
        if format is None:
            format = get_option("io.hdf.default_format") or "table"
        format = self._validate_format(format)
        self._write_to_group(
            key,
            value,
            format=format,
            axes=axes,
            index=index,
            append=append,
            complib=complib,
            complevel=complevel,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            chunksize=chunksize,
            expectedrows=expectedrows,
            dropna=dropna,
            data_columns=data_columns,
            encoding=encoding,
            errors=errors,
        )

    def append_to_multiple(
        self,
        d: dict,
        value: Any,
        selector: Any,
        data_columns: Any = None,
        axes: Any = None,
        dropna: bool = False,
        **kwargs: Any,
    ) -> None:
        if axes is not None:
            raise TypeError(
                "axes is currently not accepted as a parameter to append_to_multiple; "
                "you can create the tables independently instead"
            )
        if not isinstance(d, dict):
            raise ValueError(
                "append_to_multiple must have a dictionary specified as the way to split the value"
            )
        if selector not in d:
            raise ValueError("append_to_multiple requires a selector that is in passed dict")
        axis = 1
        remain_key = None
        remain_values: list = []
        for k, v in d.items():
            if v is None:
                if remain_key is not None:
                    raise ValueError("append_to_multiple can only have one value in d that is None")
                remain_key = k
            else:
                remain_values.extend(v)
        if remain_key is not None:
            ordered = list(value.columns)
            taken = set()
            left = []
            for c in ordered:
                if c not in remain_values and c not in taken:
                    taken.add(c)
                    left.append(c)
            d[remain_key] = left
        if data_columns is None:
            data_columns = d[selector]
        if dropna:
            keep = None
            for cols in d.values():
                rows = value[list(cols)].notna().any(axis=1)
                keep = rows if keep is None else keep & rows
            value = value.loc[keep]
        min_itemsize = kwargs.pop("min_itemsize", None)
        for k, v in d.items():
            dc = data_columns if k == selector else None
            val = value.reindex(list(v), axis=axis)
            filtered = (
                {key: value for key, value in min_itemsize.items() if key in v}
                if min_itemsize is not None
                else None
            )
            self.append(k, val, data_columns=dc, min_itemsize=filtered, **kwargs)

    def create_table_index(
        self, key: str, columns: Any = None, optlevel: int | None = None, kind: str | None = None
    ) -> None:
        _tables()
        s = self.get_storer(key)
        if s is None:
            return
        if not isinstance(s, Table):
            raise TypeError("cannot create table index on a Fixed format store")
        s.create_index(columns=columns, optlevel=optlevel, kind=kind)

    def groups(self) -> list:
        _tables()
        self._check_if_open()
        assert self._handle is not None
        assert _table_mod is not None
        return [
            g
            for g in self._handle.walk_groups()
            if not isinstance(g, _table_mod.link.Link)
            and (
                getattr(g._v_attrs, "pandas_type", None)
                or getattr(g, "table", None)
                or (isinstance(g, _table_mod.table.Table) and g._v_name != "table")
            )
        ]

    def walk(self, where: str = "/") -> Any:
        _tables()
        self._check_if_open()
        assert self._handle is not None
        assert _table_mod is not None
        for g in self._handle.walk_groups(where):
            if getattr(g._v_attrs, "pandas_type", None) is not None:
                continue
            groups = []
            leaves = []
            for child in g._v_children.values():
                pandas_type = getattr(child._v_attrs, "pandas_type", None)
                if pandas_type is None:
                    if isinstance(child, _table_mod.group.Group):
                        groups.append(child._v_name)
                else:
                    leaves.append(child._v_name)
            yield (g._v_pathname.rstrip("/"), groups, leaves)

    def get_node(self, key: str) -> Any:
        self._check_if_open()
        if not key.startswith("/"):
            key = "/" + key
        assert self._handle is not None
        assert _table_mod is not None
        try:
            node = self._handle.get_node(self.root, key)
        except _table_mod.exceptions.NoSuchNodeError:
            return None
        assert isinstance(node, _table_mod.Node), type(node)
        return node

    def get_storer(self, key: str) -> Any:
        group = self.get_node(key)
        if group is None:
            raise KeyError(f"No object named {key} in the file")
        s = self._create_storer(group)
        s.infer_axes()
        return s

    def copy(
        self,
        file: Any,
        mode: str = "w",
        propindexes: bool = True,
        keys: Any = None,
        complib: Any = None,
        complevel: int | None = None,
        fletcher32: bool = False,
        overwrite: bool = True,
    ) -> HDFStore:
        new_store = HDFStore(
            file, mode=mode, complib=complib, complevel=complevel, fletcher32=fletcher32
        )
        if keys is None:
            keys = list(self.keys())
        if not isinstance(keys, (tuple, list)):
            keys = [keys]
        for k in keys:
            s = self.get_storer(k)
            if s is not None:
                if k in new_store and overwrite:
                    new_store.remove(k)
                data = self.select(k)
                if isinstance(s, Table):
                    index: Any = False
                    if propindexes:
                        index = [a.name for a in s.axes if a.is_indexed]
                    new_store.append(
                        k,
                        data,
                        index=index,
                        data_columns=getattr(s, "data_columns", None),
                        encoding=s.encoding,
                    )
                else:
                    new_store.put(k, data, encoding=s.encoding)
        return new_store

    def info(self) -> str:
        from ._pandas import _text_adjoin

        path = _pprint(self._path)
        output = f"{type(self)}\nFile path: {path}\n"
        if self.is_open:
            lkeys = sorted(self.keys())
            if lkeys:
                keys = []
                values = []
                for k in lkeys:
                    try:
                        s = self.get_storer(k)
                        if s is not None:
                            keys.append(_pprint(s.pathname or k))
                            values.append(_pprint(s or "invalid_HDFStore node"))
                    except AssertionError:
                        raise
                    except Exception as detail:
                        keys.append(k)
                        dstr = _pprint(detail)
                        values.append(f"[invalid_HDFStore node: {dstr}]")
                output += _text_adjoin(12, keys, values)
            else:
                output += "Empty"
        else:
            output += "File is CLOSED"
        return output

    def _check_if_open(self) -> None:
        if not self.is_open:
            raise ClosedFileError(f"{self._path} file is not open!")

    def _validate_format(self, format: str) -> str:
        try:
            format = _FORMAT_MAP[format.lower()]
        except KeyError as err:
            raise TypeError(f"invalid HDFStore format specified [{format}]") from err
        return format

    def _create_storer(
        self,
        group: Any,
        format: Any = None,
        value: Any = None,
        encoding: str = "UTF-8",
        errors: str = "strict",
    ) -> Any:
        cls: Any
        if value is not None and not _is_value(value):
            raise TypeError("value must be None, Series, or DataFrame")
        pt = getattr(group._v_attrs, "pandas_type", None)
        tt = getattr(group._v_attrs, "table_type", None)
        if pt is None:
            if value is None:
                _tables()
                assert _table_mod is not None
                if getattr(group, "table", None) or isinstance(group, _table_mod.table.Table):
                    pt = "frame_table"
                    tt = "generic_table"
                else:
                    raise TypeError(
                        "cannot create a storer if the object is not existing "
                        "nor a value are passed"
                    )
            else:
                pt = "series" if value.ndim == 1 else "frame"
                if format == "table":
                    pt += "_table"
        if "table" not in pt:
            _STORER_MAP = {"series": SeriesFixed, "frame": FrameFixed}
            try:
                cls = _STORER_MAP[pt]
            except KeyError as err:
                raise TypeError(
                    "cannot properly create the storer for: [_STORER_MAP] "
                    f"[group->{group},value->{type(value)},format->{format}"
                ) from err
            return cls(self, group, encoding=encoding, errors=errors)
        if tt is None and value is not None:
            if pt == "series_table":
                index = getattr(value, "index", None)
                if index is not None:
                    if index.nlevels == 1:
                        tt = "appendable_series"
                    elif index.nlevels > 1:
                        tt = "appendable_multiseries"
            elif pt == "frame_table":
                index = getattr(value, "index", None)
                if index is not None:
                    if index.nlevels == 1:
                        tt = "appendable_frame"
                    elif index.nlevels > 1:
                        tt = "appendable_multiframe"
        _TABLE_MAP = {
            "generic_table": GenericTable,
            "appendable_series": AppendableSeriesTable,
            "appendable_multiseries": AppendableMultiSeriesTable,
            "appendable_frame": AppendableFrameTable,
            "appendable_multiframe": AppendableMultiFrameTable,
            "worm": WORMTable,
        }
        try:
            cls = _TABLE_MAP[tt]
        except KeyError as err:
            raise TypeError(
                "cannot properly create the storer for: [_TABLE_MAP] "
                f"[group->{group},value->{type(value)},format->{format}"
            ) from err
        return cls(self, group, encoding=encoding, errors=errors)

    def _write_to_group(
        self,
        key: str,
        value: Any,
        format: Any,
        axes: Any = None,
        index: Any = True,
        append: bool = False,
        complib: Any = None,
        complevel: int | None = None,
        fletcher32: Any = None,
        min_itemsize: Any = None,
        chunksize: int | None = None,
        expectedrows: Any = None,
        dropna: bool = False,
        nan_rep: Any = None,
        data_columns: Any = None,
        encoding: Any = None,
        errors: str = "strict",
        track_times: bool = True,
    ) -> None:
        if getattr(value, "empty", None) and (format == "table" or append):
            return
        group = self._identify_group(key, append)
        s = self._create_storer(group, format, value, encoding=encoding, errors=errors)
        if append:
            if not s.is_table or (s.is_table and format == "fixed" and s.is_exists):
                raise ValueError("Can only append to Tables")
            if not s.is_exists:
                s.set_object_info()
        else:
            s.set_object_info()
        if not s.is_table and complib:
            raise ValueError("Compression not supported on Fixed format stores")
        s.write(
            obj=value,
            axes=axes,
            append=append,
            complib=complib,
            complevel=complevel,
            fletcher32=fletcher32,
            min_itemsize=min_itemsize,
            chunksize=chunksize,
            expectedrows=expectedrows,
            dropna=dropna,
            nan_rep=nan_rep,
            data_columns=data_columns,
            track_times=track_times,
        )
        if isinstance(s, Table) and index:
            s.create_index(columns=index)

    def _read_group(self, group: Any) -> Any:
        s = self._create_storer(group)
        s.infer_axes()
        return s.read()

    def _identify_group(self, key: str, append: bool) -> Any:
        group = self.get_node(key)
        assert self._handle is not None
        if group is not None and not append:
            self._handle.remove_node(group, recursive=True)
            group = None
        if group is None:
            group = self._create_nodes_and_group(key)
        return group

    def _create_nodes_and_group(self, key: str) -> Any:
        assert self._handle is not None
        paths = key.split("/")
        path = "/"
        for p in paths:
            if not len(p):
                continue
            new_path = path
            if not path.endswith("/"):
                new_path += "/"
            new_path += p
            group = self.get_node(new_path)
            if group is None:
                group = self._handle.create_group(path, p)
            path = new_path
        return group


def _as_array(where: Any) -> Any:
    if isinstance(where, np.ndarray):
        return where
    if hasattr(where, "to_numpy"):
        return np.asarray(where.to_numpy())
    return np.asarray(list(where))


class TableIterator:
    """The chunks of a selection, read one at a time."""

    chunksize: int | None

    def __init__(
        self,
        store: HDFStore,
        s: Any,
        func: Any,
        where: Any,
        nrows: Any,
        start: Any = None,
        stop: Any = None,
        iterator: bool = False,
        chunksize: int | None = None,
        auto_close: bool = False,
    ) -> None:
        self.store = store
        self.s = s
        self.func = func
        self.where = where
        if self.s.is_table:
            if nrows is None:
                nrows = 0
            if start is None:
                start = 0
            if stop is None:
                stop = nrows
            stop = min(nrows, stop)
        self.nrows = nrows
        self.start = start
        self.stop = stop
        self.coordinates: Any = None
        if iterator or chunksize is not None:
            if chunksize is None:
                chunksize = 100000
            self.chunksize = int(chunksize)
        else:
            self.chunksize = None
        self.auto_close = auto_close

    def __iter__(self) -> Any:
        current = self.start
        if self.coordinates is None:
            raise ValueError("Cannot iterate until get_result is called.")
        assert self.chunksize is not None
        while current < self.stop:
            stop = min(current + self.chunksize, self.stop)
            value = self.func(None, None, self.coordinates[current:stop])
            current = stop
            if value is None or not len(value):
                continue
            yield value
        self.close()

    def close(self) -> None:
        if self.auto_close:
            self.store.close()

    def get_result(self, coordinates: bool = False) -> Any:
        if self.chunksize is not None:
            if not isinstance(self.s, Table):
                raise TypeError("can only use an iterator or chunksize on a table")
            self.coordinates = _as_array(self.s.read_coordinates(where=self.where))
            return self
        if coordinates:
            if not isinstance(self.s, Table):
                raise TypeError("can only read_coordinates on a table")
            where = self.s.read_coordinates(where=self.where, start=self.start, stop=self.stop)
        else:
            where = self.where
        results = self.func(self.start, self.stop, where)
        self.close()
        return results


def _same_info(a: Any, b: Any) -> bool:
    if hasattr(a, "freqstr") or hasattr(b, "freqstr"):
        return _freq_in(a) == _freq_in(b)
    if isinstance(a, datetime.tzinfo) or isinstance(b, datetime.tzinfo):
        return _tz_name(a) == _tz_name(b)
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


class IndexCol:
    """An indexable column of a table: its index."""

    is_an_indexable: bool = True
    is_data_indexable: bool = True
    _info_fields: tuple[str, ...] = ("freq", "tz", "index_name")

    def __init__(
        self,
        name: str,
        values: Any = None,
        kind: Any = None,
        typ: Any = None,
        cname: str | None = None,
        axis: Any = None,
        pos: Any = None,
        freq: Any = None,
        tz: Any = None,
        index_name: Any = None,
        ordered: Any = None,
        table: Any = None,
        meta: Any = None,
        metadata: Any = None,
    ) -> None:
        if not isinstance(name, str):
            raise ValueError("`name` must be a str.")
        self.values = values
        self.kind = kind
        self.typ = typ
        self.name = name
        self.cname = cname or name
        self.axis = axis
        self.pos = pos
        self.freq = freq
        self.tz = tz
        self.index_name = index_name
        self.ordered = ordered
        self.table = table
        self.meta = meta
        self.metadata = metadata
        if pos is not None:
            self.set_pos(pos)
        assert isinstance(self.name, str)
        assert isinstance(self.cname, str)

    @property
    def itemsize(self) -> int:
        return self.typ.itemsize

    @property
    def kind_attr(self) -> str:
        return f"{self.name}_kind"

    def set_pos(self, pos: int) -> None:
        self.pos = pos
        if pos is not None and self.typ is not None:
            self.typ._v_pos = pos

    def __repr__(self) -> str:
        temp = tuple(map(_pprint, (self.name, self.cname, self.axis, self.pos, self.kind)))
        return ",".join(
            [
                f"{key}->{value}"
                for key, value in zip(["name", "cname", "axis", "pos", "kind"], temp, strict=True)
            ]
        )

    def __eq__(self, other: object) -> bool:
        return all(
            getattr(self, a, None) == getattr(other, a, None)
            for a in ["name", "cname", "axis", "pos"]
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    __hash__ = None  # type: ignore[assignment]

    @property
    def is_indexed(self) -> bool:
        if not hasattr(self.table, "cols"):
            return False
        return getattr(self.table.cols, self.cname).is_indexed

    def convert(self, values: Any, nan_rep: Any, encoding: str, errors: str) -> Any:
        assert isinstance(values, np.ndarray), type(values)
        if values.dtype.fields is not None:
            values = values[self.cname].copy()
        val_kind = _decoded_kind(self.kind)
        values = _maybe_convert(values, val_kind, encoding, errors)
        index = _make_index(values, self.index_name, _freq_in(self.freq), self.freq is not None)
        tz = _tz_name(self.tz)
        if tz is not None and type(index).__name__ == "DatetimeIndex":
            index = index.tz_localize("UTC").tz_convert(tz)
        return (index, index)

    def take_data(self) -> Any:
        return self.values

    @property
    def attrs(self) -> Any:
        return self.table._v_attrs

    @property
    def description(self) -> Any:
        return self.table.description

    @property
    def col(self) -> Any:
        return getattr(self.description, self.cname, None)

    @property
    def cvalues(self) -> Any:
        return self.values

    def __iter__(self) -> Any:
        return iter(self.values)

    def maybe_set_size(self, min_itemsize: Any = None) -> None:
        if self.kind == "string":
            if isinstance(min_itemsize, dict):
                min_itemsize = min_itemsize.get(self.name)
            if min_itemsize is not None and self.typ.itemsize < min_itemsize:
                self.typ = _tables().StringCol(itemsize=min_itemsize, pos=self.pos)

    def validate_names(self) -> None:
        pass

    def validate_and_set(self, handler: AppendableTable, append: bool) -> None:
        self.table = handler.table
        self.validate_col()
        self.validate_attr(append)
        self.validate_metadata(handler)
        self.write_metadata(handler)
        self.set_attr()

    def validate_col(self, itemsize: Any = None) -> Any:
        if self.kind == "string":
            c = self.col
            if c is not None:
                if itemsize is None:
                    itemsize = self.itemsize
                if c.itemsize < itemsize:
                    raise ValueError(
                        f"Trying to store a string with len [{itemsize}] in "
                        f"[{self.cname}] column but\nthis column has a limit of "
                        f"[{c.itemsize}]!\nConsider using min_itemsize to preset the sizes "
                        "on these columns"
                    )
                return c.itemsize
        return None

    def validate_attr(self, append: bool) -> None:
        if append:
            existing_kind = getattr(self.attrs, self.kind_attr, None)
            if existing_kind is not None and existing_kind != self.kind:
                raise TypeError(f"incompatible kind in col [{existing_kind} - {self.kind}]")

    def update_info(self, info: dict) -> None:
        for key in self._info_fields:
            value = getattr(self, key, None)
            idx = info.setdefault(self.name, {})
            existing_value = idx.get(key)
            if key in idx and value is not None and not _same_info(existing_value, value):
                if key in ["freq", "index_name"]:
                    ws = attribute_conflict_doc % (key, existing_value, value)
                    warnings.warn(ws, AttributeConflictWarning, stacklevel=_stack_level())
                    idx[key] = None
                    setattr(self, key, None)
                else:
                    raise ValueError(
                        f"invalid info for [{self.name}] for [{key}], existing_value "
                        f"[{existing_value}] conflicts with new value [{value}]"
                    )
            elif value is not None or existing_value is not None:
                idx[key] = value

    def set_info(self, info: dict) -> None:
        idx = info.get(self.name)
        if idx is not None:
            self.__dict__.update(idx)

    def set_attr(self) -> None:
        setattr(self.attrs, self.kind_attr, self.kind)

    def validate_metadata(self, handler: AppendableTable) -> None:
        if self.meta == "category":
            new_metadata = self.metadata
            cur_metadata = handler.read_metadata(self.cname)
            if (
                new_metadata is not None
                and cur_metadata is not None
                and not _equivalent(list(new_metadata), list(cur_metadata))
            ):
                raise ValueError(
                    "cannot append a categorical with different categories to the existing"
                )

    def write_metadata(self, handler: AppendableTable) -> None:
        if self.metadata is not None:
            handler.write_metadata(self.cname, self.metadata)


def _decoded_kind(kind: Any) -> Any:
    if isinstance(kind, bytes):
        return kind.decode("utf-8")
    if np is not None and isinstance(kind, np.generic):
        return _decoded_kind(kind.item())
    return kind


def _equivalent(left: list[Any], right: list[Any]) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right, strict=True):
        if _is_na(a) and _is_na(b):
            continue
        try:
            if not bool(a == b):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _make_index(values: Any, name: Any, freq: Any, has_freq: bool) -> Any:
    """The index pandas' `IndexCol.convert` builds from a column of a table."""
    fp = _fp()
    if values.dtype.kind == "M":
        try:
            return fp.DatetimeIndex(fp.Series(values), freq=freq, name=name)
        except ValueError:
            return fp.DatetimeIndex(fp.Series(values), freq=None, name=name)
    if values.dtype == np.int64 and has_freq:
        offset = fp.tseries.frequencies.to_offset(freq)
        return fp.PeriodIndex.from_ordinals(values, freq=offset).rename(name)
    if has_freq:
        raise TypeError("Index.__new__() got an unexpected keyword argument 'freq'")
    return _plain_index(values, name)


def _plain_index(values: Any, name: Any) -> Any:
    """`Index(values)`, with pandas' inference for an object array."""
    fp = _fp()
    if values.dtype.kind == "O":
        cells = list(values)
        if _all_str(cells):
            return fp.Index([np.nan if _is_na(v) else v for v in cells], name=name)
        return fp.Index(cells, name=name)
    if values.dtype.kind == "m":
        return fp.TimedeltaIndex(_series_of(values), freq=None, name=name)
    if values.dtype.kind == "S":
        return fp.Index(list(values), name=name)
    return fp.Index(fp.Series(values), name=name)


class GenericIndexCol(IndexCol):
    """The index of a table pandas did not write: row numbers."""

    @property
    def is_indexed(self) -> bool:
        return False

    def convert(self, values: Any, nan_rep: Any, encoding: str, errors: str) -> Any:
        assert isinstance(values, np.ndarray), type(values)
        index = _fp().RangeIndex(len(values))
        return (index, index)

    def set_attr(self) -> None:
        pass


class DataCol(IndexCol):
    """A block of values in a table."""

    is_an_indexable = False
    is_data_indexable = False
    _info_fields = ("tz", "ordered")

    def __init__(
        self,
        name: str,
        values: Any = None,
        kind: Any = None,
        typ: Any = None,
        cname: str | None = None,
        pos: Any = None,
        tz: Any = None,
        ordered: Any = None,
        table: Any = None,
        meta: Any = None,
        metadata: Any = None,
        dtype: Any = None,
        data: Any = None,
    ) -> None:
        super().__init__(
            name=name,
            values=values,
            kind=kind,
            typ=typ,
            pos=pos,
            cname=cname,
            tz=tz,
            ordered=ordered,
            table=table,
            meta=meta,
            metadata=metadata,
        )
        self.dtype = dtype
        self.data = data

    @property
    def dtype_attr(self) -> str:
        return f"{self.name}_dtype"

    @property
    def meta_attr(self) -> str:
        return f"{self.name}_meta"

    def __repr__(self) -> str:
        temp = tuple(map(_pprint, (self.name, self.cname, self.dtype, self.kind, self.shape)))
        return ",".join(
            [
                f"{key}->{value}"
                for key, value in zip(
                    ["name", "cname", "dtype", "kind", "shape"], temp, strict=True
                )
            ]
        )

    def __eq__(self, other: object) -> bool:
        return all(
            getattr(self, a, None) == getattr(other, a, None)
            for a in ["name", "cname", "dtype", "pos"]
        )

    __hash__ = None  # type: ignore[assignment]

    def set_data(self, data: Any) -> None:
        assert data is not None
        assert self.dtype is None
        data, dtype_name = _get_data_and_dtype_name(data)
        self.data = data
        self.dtype = dtype_name
        self.kind = _dtype_to_kind(dtype_name)

    def take_data(self) -> Any:
        return self.data

    @classmethod
    def _get_atom(cls, values: Any) -> Any:
        shape = values.shape
        if values.ndim == 1:
            shape = (1, values.size)
        if isinstance(values, _Arr):
            if values.kind == "cat":
                return cls.get_atom_data(shape, kind=values.data.dtype.name)
            if values.kind == "tz":
                return cls.get_atom_datetime64(shape)
            if values.kind == "str":
                return cls.get_atom_string(shape, 8)
            return cls.get_atom_data(shape, kind=values.dtype_name)
        dtype = values.dtype
        itemsize = dtype.itemsize
        if dtype.kind == "M":
            atom = cls.get_atom_datetime64(shape)
        elif dtype.kind == "m":
            atom = cls.get_atom_timedelta64(shape)
        elif dtype.kind == "c":
            atom = _tables().ComplexCol(itemsize=itemsize, shape=shape[0])
        elif dtype.kind in "OSU":
            atom = cls.get_atom_string(shape, itemsize)
        else:
            atom = cls.get_atom_data(shape, kind=dtype.name)
        return atom

    @classmethod
    def get_atom_string(cls, shape: Any, itemsize: Any) -> Any:
        return _tables().StringCol(itemsize=itemsize, shape=shape[0])

    @classmethod
    def get_atom_coltype(cls, kind: str) -> Any:
        if kind.startswith("uint"):
            k4 = kind[4:]
            col_name = f"UInt{k4}Col"
        elif kind.startswith("period"):
            col_name = "Int64Col"
        else:
            kcap = kind.capitalize()
            col_name = f"{kcap}Col"
        return getattr(_tables(), col_name)

    @classmethod
    def get_atom_data(cls, shape: Any, kind: str) -> Any:
        return cls.get_atom_coltype(kind=kind)(shape=shape[0])

    @classmethod
    def get_atom_datetime64(cls, shape: Any) -> Any:
        return _tables().Int64Col(shape=shape[0])

    @classmethod
    def get_atom_timedelta64(cls, shape: Any) -> Any:
        return _tables().Int64Col(shape=shape[0])

    @property
    def shape(self) -> Any:
        return getattr(self.data, "shape", None)

    @property
    def cvalues(self) -> Any:
        return self.data

    def validate_attr(self, append: bool) -> None:
        if append:
            existing_fields = getattr(self.attrs, self.kind_attr, None)
            if existing_fields is not None and existing_fields != list(self.values):
                raise ValueError("appended items do not match existing items in table!")
            existing_dtype = getattr(self.attrs, self.dtype_attr, None)
            if existing_dtype is not None and existing_dtype != self.dtype:
                raise ValueError("appended items dtype do not match existing items dtype in table!")

    def convert(self, values: Any, nan_rep: Any, encoding: str, errors: str) -> Any:
        assert isinstance(values, np.ndarray), type(values)
        if values.dtype.fields is not None:
            values = values[self.cname]
        assert self.typ is not None
        if self.dtype is None:
            converted, dtype_name = _get_data_and_dtype_name(values)
            kind = _dtype_to_kind(dtype_name)
        else:
            converted = values
            dtype_name = _decoded_kind(self.dtype)
            kind = self.kind
        assert isinstance(converted, np.ndarray)
        meta = _decoded_kind(self.meta)
        metadata = self.metadata
        ordered = self.ordered
        tz = self.tz
        assert dtype_name is not None
        dtype = dtype_name
        if dtype.startswith("datetime64"):
            if dtype == "datetime64":
                dtype = "datetime64[ns]"
            converted = _set_tz(converted, tz, dtype)
        elif dtype.startswith("timedelta64"):
            if dtype == "timedelta64":
                converted = np.asarray(converted, dtype="m8[ns]")
            else:
                converted = np.asarray(converted, dtype=dtype)
        elif dtype == "date":
            try:
                converted = np.asarray(
                    [datetime.date.fromordinal(v) for v in converted], dtype=object
                )
            except ValueError:
                converted = np.asarray(
                    [datetime.date.fromtimestamp(v) for v in converted], dtype=object
                )
        elif meta == "category":
            codes = converted.ravel().copy()
            if metadata is None:
                categories: Any = np.asarray([], dtype=np.float64)
            else:
                categories = _as_objects(metadata)
                mask = np.asarray([_is_na(v) for v in categories], dtype=bool)
                if mask.any():
                    categories = categories[~mask]
                    codes[codes != -1] -= mask.astype(int).cumsum()[codes[codes != -1]]
            converted = _Arr("cat", codes, "category", categories=categories, ordered=bool(ordered))
        else:
            try:
                converted = converted.astype(dtype, copy=False)
            except TypeError:
                converted = converted.astype("O", copy=False)
        if kind == "string":
            converted = _unconvert_string_array(
                converted, nan_rep=nan_rep, encoding=encoding, errors=errors
            )
        return (self.values, converted)

    def set_attr(self) -> None:
        setattr(self.attrs, self.kind_attr, self.values)
        setattr(self.attrs, self.meta_attr, self.meta)
        assert self.dtype is not None
        setattr(self.attrs, self.dtype_attr, self.dtype)


def _as_objects(values: Any) -> Any:
    out = np.empty(len(values), dtype=object)
    out[:] = [_scalar(v) for v in (values.tolist() if hasattr(values, "tolist") else values)]
    return out


class DataIndexableCol(DataCol):
    """A data column of a table, one that a where clause can name."""

    is_data_indexable = True

    def validate_names(self) -> None:
        if not all(isinstance(v, str) for v in self.values):
            raise ValueError("cannot have non-object label DataIndexableCol")

    @classmethod
    def get_atom_string(cls, shape: Any, itemsize: Any) -> Any:
        return _tables().StringCol(itemsize=itemsize)

    @classmethod
    def get_atom_data(cls, shape: Any, kind: str) -> Any:
        return cls.get_atom_coltype(kind=kind)()

    @classmethod
    def get_atom_datetime64(cls, shape: Any) -> Any:
        return _tables().Int64Col()

    @classmethod
    def get_atom_timedelta64(cls, shape: Any) -> Any:
        return _tables().Int64Col()


class GenericDataIndexableCol(DataIndexableCol):
    """A column of a table pandas did not write."""


class Fixed:
    """A pandas object stored as plain arrays."""

    pandas_kind: str
    format_type: str = "fixed"
    obj_type: Any
    ndim: int
    parent: HDFStore
    is_table: bool = False

    def __init__(
        self, parent: HDFStore, group: Any, encoding: str | None = "UTF-8", errors: str = "strict"
    ) -> None:
        assert isinstance(parent, HDFStore), type(parent)
        assert _table_mod is not None
        assert isinstance(group, _table_mod.Node), type(group)
        self.parent = parent
        self.group = group
        self.encoding = _ensure_encoding(encoding)
        self.errors = errors

    @property
    def is_old_version(self) -> bool:
        return self.version[0] <= 0 and self.version[1] <= 10 and self.version[2] < 1

    @property
    def version(self) -> tuple[int, int, int]:
        version = getattr(self.group._v_attrs, "pandas_version", None)
        if isinstance(version, str):
            version_tup = tuple(int(x) for x in version.split("."))
            if len(version_tup) == 2:
                version_tup = (*version_tup, 0)
            assert len(version_tup) == 3
            return version_tup  # type: ignore[return-value]
        else:
            return (0, 0, 0)

    @property
    def pandas_type(self) -> Any:
        return getattr(self.group._v_attrs, "pandas_type", None)

    def __repr__(self) -> str:
        self.infer_axes()
        s = self.shape
        if s is not None:
            if isinstance(s, (list, tuple)):
                jshape = ",".join([_pprint(x) for x in s])
                s = f"[{jshape}]"
            return f"{self.pandas_type:12.12} (shape->{s})"
        return self.pandas_type

    def set_object_info(self) -> None:
        self.attrs.pandas_type = str(self.pandas_kind)
        self.attrs.pandas_version = str(_version)

    def copy(self) -> Fixed:
        new_self = copy.copy(self)
        return new_self

    @property
    def shape(self) -> Any:
        return self.nrows

    @property
    def pathname(self) -> Any:
        return self.group._v_pathname

    @property
    def _handle(self) -> Any:
        return self.parent._handle

    @property
    def _filters(self) -> Any:
        return self.parent._filters

    @property
    def _complevel(self) -> int:
        return self.parent._complevel

    @property
    def _fletcher32(self) -> bool:
        return self.parent._fletcher32

    @property
    def attrs(self) -> Any:
        return self.group._v_attrs

    def set_attrs(self) -> None:
        pass

    def get_attrs(self) -> None:
        pass

    @property
    def storable(self) -> Any:
        return self.group

    @property
    def is_exists(self) -> bool:
        return False

    @property
    def nrows(self) -> Any:
        return getattr(self.storable, "nrows", None)

    def validate(self, other: Any) -> Any:
        if other is None:
            return None
        return True

    def validate_version(self, where: Any = None) -> None:
        pass

    def infer_axes(self) -> bool:
        s = self.storable
        if s is None:
            return False
        self.get_attrs()
        return True

    def read(
        self,
        where: Any = None,
        columns: Any = None,
        start: int | None = None,
        stop: int | None = None,
    ) -> Any:
        raise NotImplementedError("cannot read on an abstract storer: subclasses should implement")

    def write(self, obj: Any, **kwargs: Any) -> None:
        raise NotImplementedError("cannot write on an abstract storer: subclasses should implement")

    def delete(self, where: Any = None, start: int | None = None, stop: int | None = None) -> Any:
        if _all_none(where, start, stop):
            self._handle.remove_node(self.group, recursive=True)
            return None
        raise TypeError("cannot delete on an abstract storer")


class GenericFixed(Fixed):
    """A fixed store with the arrays and indexes of a pandas object."""

    _index_type_map: ClassVar[dict[str, str]] = {
        "DatetimeIndex": "datetime",
        "PeriodIndex": "period",
    }
    _reverse_index_map: ClassVar[dict[str, str]] = {v: k for k, v in _index_type_map.items()}
    attributes: tuple[str, ...] = ()

    def _class_to_alias(self, cls: str) -> str:
        return self._index_type_map.get(cls, "")

    def _alias_to_class(self, alias: Any) -> str:
        if isinstance(alias, type):
            return alias.__name__
        return self._reverse_index_map.get(alias, "Index")

    def _get_index_factory(self, attrs: Any) -> tuple[Any, dict[str, Any]]:
        index_class = self._alias_to_class(getattr(attrs, "index_class", ""))
        fp = _fp()
        factory: Any
        kwargs: dict[str, Any] = {}
        if index_class == "DatetimeIndex":

            def f(values: Any, freq: Any = None, tz: Any = None) -> Any:
                result = fp.DatetimeIndex(fp.Series(values), freq=_freq_in(freq))
                if tz is not None:
                    result = result.tz_localize("UTC").tz_convert(_tz_name(tz))
                return result

            factory = f
        elif index_class == "PeriodIndex":

            def f(values: Any, freq: Any = None, tz: Any = None) -> Any:
                offset = fp.tseries.frequencies.to_offset(_freq_in(freq))
                return fp.PeriodIndex.from_ordinals(np.asarray(values), freq=offset)

            factory = f
        else:

            def f(values: Any, freq: Any = None, dtype: Any = None) -> Any:
                if freq is not None or has_freq:
                    return fp.TimedeltaIndex(
                        _series_of(np.asarray(values)), freq=_freq_in(freq), name=None
                    )
                if dtype is object:
                    return fp.Index(list(values))
                return _plain_index(np.asarray(values), None)

            factory = f
        has_freq = "freq" in attrs
        if has_freq:
            kwargs["freq"] = attrs["freq"]
        if "tz" in attrs:
            kwargs["tz"] = attrs["tz"]
            assert index_class == "DatetimeIndex"
        return (factory, kwargs)

    def validate_read(self, columns: Any, where: Any) -> None:
        if columns is not None:
            raise TypeError(
                "cannot pass a column specification when reading a Fixed format store. "
                "this store must be selected in its entirety"
            )
        if where is not None:
            raise TypeError(
                "cannot pass a where specification when reading from a Fixed format store. "
                "this store must be selected in its entirety"
            )

    @property
    def is_exists(self) -> bool:
        return True

    def set_attrs(self) -> None:
        self.attrs.encoding = self.encoding
        self.attrs.errors = self.errors

    def get_attrs(self) -> None:
        self.encoding = _ensure_encoding(getattr(self.attrs, "encoding", None))
        self.errors = getattr(self.attrs, "errors", "strict")
        for n in self.attributes:
            setattr(self, n, getattr(self.attrs, n, None))

    def write(self, obj: Any, **kwargs: Any) -> None:
        self.set_attrs()

    def read_array(self, key: str, start: int | None = None, stop: int | None = None) -> Any:
        import tables

        node = getattr(self.group, key)
        attrs = node._v_attrs
        transposed = getattr(attrs, "transposed", False)
        if isinstance(node, tables.VLArray):
            ret = node[0][start:stop]
            dtype = getattr(attrs, "value_type", None)
            if dtype is not None:
                ret = (
                    _Arr("str", np.asarray(ret, dtype=object), "str")
                    if dtype
                    in (
                        "str",
                        "string",
                    )
                    else _Arr("ext", np.asarray(ret, dtype=object), dtype)
                )
        else:
            dtype = getattr(attrs, "value_type", None)
            shape = getattr(attrs, "shape", None)
            ret = np.empty(shape, dtype=dtype) if shape is not None else node[start:stop]
            if dtype and dtype.startswith("datetime64"):
                if dtype == "datetime64":
                    dtype = "datetime64[ns]"
                tz = getattr(attrs, "tz", None)
                ret = _set_tz(ret, tz, dtype)
            elif dtype and dtype.startswith("timedelta64"):
                if dtype == "timedelta64":
                    ret = np.asarray(ret, dtype="m8[ns]")
                else:
                    ret = np.asarray(ret, dtype=dtype)
        if transposed:
            return ret.T
        else:
            return ret

    def read_index(self, key: str, start: int | None = None, stop: int | None = None) -> Any:
        variety = getattr(self.attrs, f"{key}_variety")
        if variety == "multi":
            return self.read_multi_index(key, start=start, stop=stop)
        elif variety == "regular":
            node = getattr(self.group, key)
            index = self.read_index_node(node, start=start, stop=stop)
            return index
        else:
            raise TypeError(f"unrecognized index variety: {variety}")

    def write_index(self, key: str, index: _Ix) -> None:
        if index.cls == "MultiIndex":
            setattr(self.attrs, f"{key}_variety", "multi")
            self.write_multi_index(key, index)
        else:
            setattr(self.attrs, f"{key}_variety", "regular")
            converted = _convert_index("index", index, self.encoding, self.errors)
            self.write_array(key, converted.values)
            node = getattr(self.group, key)
            node._v_attrs.kind = converted.kind
            node._v_attrs.name = index.name
            if index.cls in ("DatetimeIndex", "PeriodIndex"):
                node._v_attrs.index_class = self._class_to_alias(index.cls)
            if index.cls in ("DatetimeIndex", "PeriodIndex", "TimedeltaIndex"):
                node._v_attrs.freq = index.freq
            if index.cls == "DatetimeIndex" and index.tz is not None:
                node._v_attrs.tz = _tz_attr(index.tz)

    def write_multi_index(self, key: str, index: _Ix) -> None:
        assert index.levels is not None and index.codes is not None
        setattr(self.attrs, f"{key}_nlevels", index.nlevels)
        for i, (lev, level_codes, name) in enumerate(
            zip(index.levels, index.codes, index.names, strict=True)
        ):
            if isinstance(lev.values, _Arr) and lev.values.kind != "str":
                raise NotImplementedError(
                    "Saving a MultiIndex with an extension dtype is not supported."
                )
            level_key = f"{key}_level{i}"
            conv_level = _convert_index(level_key, lev, self.encoding, self.errors)
            self.write_array(level_key, conv_level.values)
            node = getattr(self.group, level_key)
            node._v_attrs.kind = conv_level.kind
            node._v_attrs.name = name
            setattr(node._v_attrs, f"{key}_name{name}", name)
            label_key = f"{key}_label{i}"
            self.write_array(label_key, level_codes)

    def read_multi_index(self, key: str, start: int | None = None, stop: int | None = None) -> Any:
        nlevels = getattr(self.attrs, f"{key}_nlevels")
        levels = []
        codes = []
        names: list[Any] = []
        for i in range(nlevels):
            level_key = f"{key}_level{i}"
            node = getattr(self.group, level_key)
            lev = self.read_index_node(node, start=start, stop=stop)
            levels.append(lev)
            names.append(lev.name)
            label_key = f"{key}_label{i}"
            level_codes = self.read_array(label_key, start=start, stop=stop)
            codes.append(level_codes)
        return _fp().MultiIndex(levels=levels, codes=codes, names=names)

    def read_index_node(self, node: Any, start: int | None = None, stop: int | None = None) -> Any:
        data = node[start:stop]
        if "shape" in node._v_attrs and math.prod(node._v_attrs.shape) == 0:
            data = np.empty(node._v_attrs.shape, dtype=node._v_attrs.value_type)
        kind = _decoded_kind(node._v_attrs.kind)
        name = None
        if "name" in node._v_attrs:
            name = _ensure_str(node._v_attrs.name)
        attrs = node._v_attrs
        factory, kwargs = self._get_index_factory(attrs)
        if kind in ("date", "object"):
            index = factory(
                _unconvert_index(data, kind, encoding=self.encoding, errors=self.errors),
                dtype=object,
                **kwargs,
            )
        else:
            index = factory(
                _unconvert_index(data, kind, encoding=self.encoding, errors=self.errors),
                **kwargs,
            )
        return index.rename(name)

    def write_array_empty(self, key: str, value: Any) -> None:
        arr = np.empty((1,) * value.ndim)
        self._handle.create_array(self.group, key, arr)
        node = getattr(self.group, key)
        node._v_attrs.value_type = _value_type(value)
        node._v_attrs.shape = value.shape

    def write_array(self, key: str, obj: Any, items: Any = None) -> None:
        value = obj
        if key in self.group:
            self._handle.remove_node(self.group, key)
        empty_array = value.size == 0
        transposed = False
        if isinstance(value, _Arr) and value.kind == "cat":
            raise NotImplementedError(
                "Cannot store a category dtype in an HDF5 dataset that uses "
                'format="fixed". Use format="table".'
            )
        if not empty_array:
            value = value.T
            transposed = True
        if isinstance(value, _Arr) and value.kind == "str":
            vlarr = self._handle.create_vlarray(
                self.group, key, _tables().ObjectAtom(), filters=self._filters
            )
            vlarr.append(value.to_numpy())
            node = getattr(self.group, key)
            node._v_attrs.value_type = "str"
        elif isinstance(value, _Arr) and value.kind in ("ext", "period"):
            array_name, dtype_name = _ext_names(value.dtype_name)
            if self._filters is not None:
                raise AttributeError(f"'{dtype_name}' object has no attribute 'shape'")
            if empty_array:
                self.write_array_empty(key, value)
            else:
                raise TypeError(
                    f"objects of type ``{array_name}`` are not supported in this context, "
                    "sorry; supported objects are: NumPy array, record or scalar; homogeneous "
                    "list or tuple, integer, float, complex or bytes"
                )
        elif isinstance(value, _Arr):
            self._handle.create_array(self.group, key, value.data.view("i8"))
            node = getattr(self.group, key)
            node._v_attrs.tz = _tz_attr(value.tz)
            node._v_attrs.value_type = f"datetime64[{value.unit}]"
        else:
            atom = None
            if self._filters is not None:
                with suppress(ValueError):
                    atom = _tables().Atom.from_dtype(value.dtype)
            if atom is not None:
                if not empty_array:
                    ca = self._handle.create_carray(
                        self.group, key, atom, value.shape, filters=self._filters
                    )
                    ca[:] = value
                else:
                    self.write_array_empty(key, value)
            elif value.dtype.type == np.object_:
                inferred_type = _infer(value, skipna=False)
                if empty_array or inferred_type == "string":
                    pass
                elif get_option("mode.performance_warnings"):
                    ws = performance_doc % (inferred_type, key, items)
                    warnings.warn(ws, PerformanceWarning, stacklevel=_stack_level())
                vlarr = self._handle.create_vlarray(self.group, key, _tables().ObjectAtom())
                vlarr.append(value)
            elif value.dtype.kind == "M" or value.dtype.kind == "m":
                self._handle.create_array(self.group, key, value.view("i8"))
                getattr(self.group, key)._v_attrs.value_type = str(value.dtype)
            elif empty_array:
                self.write_array_empty(key, value)
            else:
                self._handle.create_array(self.group, key, value)
        getattr(self.group, key)._v_attrs.transposed = transposed


def _value_type(value: Any) -> str:
    if isinstance(value, _Arr):
        return value.dtype_name
    return str(value.dtype)


class SeriesFixed(GenericFixed):
    """A Series stored as plain arrays."""

    pandas_kind = "series"
    attributes = ("name",)
    name: Any

    @property
    def shape(self) -> Any:
        try:
            return (len(self.group.values),)
        except (TypeError, AttributeError):
            return None

    def read(
        self,
        where: Any = None,
        columns: Any = None,
        start: int | None = None,
        stop: int | None = None,
    ) -> Any:
        self.validate_read(columns, where)
        index = self.read_index("index", start=start, stop=stop)
        values = self.read_array("values", start=start, stop=stop)
        return _series_of(values, name=self.name, index=index)

    def write(self, obj: Any, **kwargs: Any) -> None:
        super().write(obj, **kwargs)
        obj = _obj_of(obj)
        self.write_index("index", obj.index)
        self.write_array("values", obj.values)
        self.attrs.name = obj.name


def _frame_of(columns: list[tuple[Any, Any]], index: Any, labels: Any) -> Any:
    """A firepanda frame of 1D values, labelled as pandas labels one."""
    fp = _fp()
    if not columns:
        return fp.DataFrame(index=index, columns=labels)
    frame = fp.DataFrame({i: _series_of(v, dtype=d) for i, (v, d) in enumerate(columns)})
    frame = frame.set_axis(labels, axis=1)
    return frame.set_axis(index, axis=0)


class BlockManagerFixed(GenericFixed):
    """A frame stored as its axes and its blocks."""

    attributes = ("ndim", "nblocks")
    nblocks: int

    @property
    def shape(self) -> Any:
        try:
            ndim = self.ndim
            items = 0
            for i in range(self.nblocks):
                node = getattr(self.group, f"block{i}_items")
                shape = getattr(node, "shape", None)
                if shape is not None:
                    items += shape[0]
            node = self.group.block0_values
            shape = getattr(node, "shape", None)
            shape = list(shape[0 : ndim - 1]) if shape is not None else []
            shape.append(items)
            return shape
        except AttributeError:
            return None

    def read(
        self,
        where: Any = None,
        columns: Any = None,
        start: int | None = None,
        stop: int | None = None,
    ) -> Any:
        self.validate_read(columns, where)
        select_axis = 1
        axes = []
        for i in range(self.ndim):
            _start, _stop = (start, stop) if i == select_axis else (None, None)
            ax = self.read_index(f"axis{i}", start=_start, stop=_stop)
            axes.append(ax)
        items = axes[0]
        labels = [_scalar(v) for v in items]
        found: dict[int, tuple[Any, Any]] = {}
        for i in range(self.nblocks):
            blk_items = [_scalar(v) for v in self.read_index(f"block{i}_items")]
            values = self.read_array(f"block{i}_values", start=_start, stop=_stop)
            dtype = None
            if (
                isinstance(values, np.ndarray)
                and values.dtype.kind == "O"
                and values.size
                and all(isinstance(v, str) or _is_na(v) for v in values.ravel())
            ):
                dtype = "str"
            for j, label in enumerate(blk_items):
                pos = next(k for k, lab in enumerate(labels) if _same_label(lab, label))
                if isinstance(values, _Arr) and values.ndim == 1:
                    col = values
                elif isinstance(values, _Arr):
                    col = values._with(values.data[j])
                elif values.ndim == 1:
                    col = values
                else:
                    col = values[j]
                found[pos] = (col, dtype)
        if found:
            return _frame_of([found[k] for k in sorted(found)], axes[1], items)
        return _fp().DataFrame(columns=axes[0], index=axes[1])

    def write(self, obj: Any, **kwargs: Any) -> None:
        super().write(obj, **kwargs)
        data = _obj_of(obj)
        if not data.is_consolidated:
            data = data.consolidate()
        self.attrs.ndim = 2
        for i, ax in enumerate([data.columns, data.index]):
            if i == 0 and not ax.is_unique:
                raise ValueError("Columns index has to be unique for fixed format")
            self.write_index(f"axis{i}", ax)
        self.attrs.nblocks = len(data.blocks)
        for i, blk in enumerate(data.blocks):
            blk_items = data.columns.take(blk.locs)
            self.write_array(f"block{i}_values", blk.values, items=_ItemsText(blk_items))
            self.write_index(f"block{i}_items", blk_items)


class _ItemsText:
    """Block items as pandas prints an index in a warning."""

    def __init__(self, ix: _Ix) -> None:
        self.ix = ix

    def __str__(self) -> str:
        fp = _fp()
        try:
            return str(_index_of(self.ix))
        except Exception:
            return str(fp.Index(self.ix.labels()))


class FrameFixed(BlockManagerFixed):
    """A DataFrame stored as plain arrays."""

    pandas_kind = "frame"

    @property
    def obj_type(self) -> Any:
        return _fp().DataFrame


class Table(Fixed):
    """A pandas object stored as a PyTables table."""

    pandas_kind = "wide_table"
    format_type: str = "table"
    table_type: str
    levels: Any = 1
    is_table = True
    metadata: list

    def __init__(
        self,
        parent: HDFStore,
        group: Any,
        encoding: str | None = None,
        errors: str = "strict",
        index_axes: list[IndexCol] | None = None,
        non_index_axes: list[tuple[int, Any]] | None = None,
        values_axes: list[DataCol] | None = None,
        data_columns: list | None = None,
        info: dict | None = None,
        nan_rep: Any = None,
    ) -> None:
        super().__init__(parent, group, encoding=encoding, errors=errors)
        self.index_axes = index_axes or []
        self.non_index_axes = non_index_axes or []
        self.values_axes = values_axes or []
        self.data_columns = data_columns or []
        self.info = info or {}
        self.nan_rep = nan_rep

    @property
    def table_type_short(self) -> str:
        return self.table_type.split("_")[0]

    def __repr__(self) -> str:
        self.infer_axes()
        jdc = ",".join(self.data_columns) if len(self.data_columns) else ""
        dc = f",dc->[{jdc}]"
        ver = ""
        if self.is_old_version:
            jver = ".".join([str(x) for x in self.version])
            ver = f"[{jver}]"
        jindex_axes = ",".join([a.name for a in self.index_axes])
        return (
            f"{self.pandas_type:12.12}{ver} (typ->{self.table_type_short},nrows->{self.nrows},"
            f"ncols->{self.ncols},indexers->[{jindex_axes}]{dc})"
        )

    def __getitem__(self, c: str) -> Any:
        for a in self.axes:
            if c == a.name:
                return a
        return None

    def validate(self, other: Any) -> None:
        if other is None:
            return
        if other.table_type != self.table_type:
            raise TypeError(
                f"incompatible table_type with existing [{other.table_type} - {self.table_type}]"
            )
        for c in ["index_axes", "non_index_axes", "values_axes"]:
            sv = getattr(self, c, None)
            ov = getattr(other, c, None)
            if sv != ov:
                for i, sax in enumerate(sv):
                    oax = ov[i]
                    if sax != oax:
                        if c == "values_axes" and sax.kind != oax.kind:
                            raise ValueError(
                                f"Cannot serialize the column [{oax.values[0]}] because its "
                                f"data contents are not [{sax.kind}] but [{oax.kind}] object "
                                "dtype"
                            )
                        raise ValueError(
                            f"invalid combination of [{c}] on appending data [{sax}] vs "
                            f"current table [{oax}]"
                        )
                raise Exception(
                    f"invalid combination of [{c}] on appending data [{sv}] vs current table [{ov}]"
                )

    @property
    def is_multi_index(self) -> bool:
        return isinstance(self.levels, list)

    def validate_multiindex(self, obj: Any) -> tuple[Any, list[Any]]:
        levels = _fill_missing_names(obj.index.names)
        try:
            reset_obj = obj.reset_index()
        except ValueError as err:
            raise ValueError(
                "duplicate names/columns in the multi-index when storing as a table"
            ) from err
        return (reset_obj, levels)

    @property
    def nrows_expected(self) -> int:
        return math.prod([i.cvalues.shape[0] for i in self.index_axes])

    @property
    def is_exists(self) -> bool:
        return "table" in self.group

    @property
    def storable(self) -> Any:
        return getattr(self.group, "table", None)

    @property
    def table(self) -> Any:
        return self.storable

    @property
    def dtype(self) -> Any:
        return self.table.dtype

    @property
    def description(self) -> Any:
        return self.table.description

    @property
    def axes(self) -> Any:
        return itertools.chain(self.index_axes, self.values_axes)

    @property
    def ncols(self) -> int:
        return sum(len(a.values) for a in self.values_axes)

    @property
    def is_transposed(self) -> bool:
        return False

    @property
    def data_orientation(self) -> tuple[int, ...]:
        return tuple(
            itertools.chain(
                [int(a[0]) for a in self.non_index_axes], [int(a.axis) for a in self.index_axes]
            )
        )

    def queryables(self) -> dict[str, Any]:
        axis_names = {0: "index", 1: "columns"}
        d1 = [(a.cname, a) for a in self.index_axes]
        d2 = [(axis_names[axis], None) for axis, values in self.non_index_axes]
        d3 = [(v.cname, v) for v in self.values_axes if v.name in set(self.data_columns)]
        return dict(d1 + d2 + d3)

    def index_cols(self) -> list[tuple[Any, Any]]:
        return [(i.axis, i.cname) for i in self.index_axes]

    def values_cols(self) -> list[str]:
        return [i.cname for i in self.values_axes]

    def _get_metadata_path(self, key: str) -> str:
        group = self.group._v_pathname
        return f"{group}/meta/{key}/meta"

    def write_metadata(self, key: str, values: Any) -> None:
        self.parent.put(
            self._get_metadata_path(key),
            _Series(_meta_values(values), _range_ix(len(values))),
            format="table",
            encoding=self.encoding,
            errors=self.errors,
            nan_rep=self.nan_rep,
        )

    def read_metadata(self, key: str) -> Any:
        if getattr(getattr(self.group, "meta", None), key, None) is not None:
            return self.parent.select(self._get_metadata_path(key))
        return None

    def set_attrs(self) -> None:
        self.attrs.table_type = str(self.table_type)
        self.attrs.index_cols = self.index_cols()
        self.attrs.values_cols = self.values_cols()
        self.attrs.non_index_axes = self.non_index_axes
        self.attrs.data_columns = self.data_columns
        self.attrs.nan_rep = self.nan_rep
        self.attrs.encoding = self.encoding
        self.attrs.errors = self.errors
        self.attrs.levels = self.levels
        self.attrs.info = self.info

    def get_attrs(self) -> None:
        self.non_index_axes = getattr(self.attrs, "non_index_axes", None) or []
        self.data_columns = getattr(self.attrs, "data_columns", None) or []
        self.info = getattr(self.attrs, "info", None) or {}
        self.nan_rep = getattr(self.attrs, "nan_rep", None)
        self.encoding = _ensure_encoding(getattr(self.attrs, "encoding", None))
        self.errors = getattr(self.attrs, "errors", "strict")
        self.levels = getattr(self.attrs, "levels", None) or []
        self.index_axes = [a for a in self.indexables if a.is_an_indexable]
        self.values_axes = [a for a in self.indexables if not a.is_an_indexable]

    def validate_version(self, where: Any = None) -> None:
        if where is not None and self.is_old_version:
            ws = incompatibility_doc % ".".join([str(x) for x in self.version])
            warnings.warn(ws, IncompatibilityWarning, stacklevel=_stack_level())

    def validate_min_itemsize(self, min_itemsize: Any) -> None:
        if min_itemsize is None:
            return
        if not isinstance(min_itemsize, dict):
            return
        q = self.queryables()
        for k in min_itemsize:
            if k == "values":
                continue
            if k not in q:
                raise ValueError(
                    f"min_itemsize has the key [{k}] which is not an axis or data_column"
                )

    @functools.cached_property
    def indexables(self) -> list[Any]:
        _indexables: list[Any] = []
        desc = self.description
        table_attrs = self.table.attrs
        for i, (axis, name) in enumerate(self.attrs.index_cols):
            atom = getattr(desc, name)
            md = self.read_metadata(name)
            meta = "category" if md is not None else None
            kind_attr = f"{name}_kind"
            kind = getattr(table_attrs, kind_attr, None)
            index_col = IndexCol(
                name=name,
                axis=axis,
                pos=i,
                kind=kind,
                typ=atom,
                table=self.table,
                meta=meta,
                metadata=md,
            )
            _indexables.append(index_col)
        dc = set(self.data_columns)
        base_pos = len(_indexables)

        def f(i: int, c: str) -> DataCol:
            assert isinstance(c, str)
            klass: Any = DataCol
            if c in dc:
                klass = DataIndexableCol
            atom = getattr(desc, c)
            adj_name = _maybe_adjust_name(c, self.version)
            values = getattr(table_attrs, f"{adj_name}_kind", None)
            dtype = getattr(table_attrs, f"{adj_name}_dtype", None)
            kind = _dtype_to_kind(dtype)
            md = self.read_metadata(c)
            meta = getattr(table_attrs, f"{adj_name}_meta", None)
            obj = klass(
                name=adj_name,
                cname=c,
                values=values,
                kind=kind,
                pos=base_pos + i,
                typ=atom,
                table=self.table,
                meta=meta,
                metadata=md,
                dtype=dtype,
            )
            return obj

        _indexables.extend([f(i, c) for i, c in enumerate(self.attrs.values_cols)])
        return _indexables

    def create_index(
        self, columns: Any = None, optlevel: Any = None, kind: str | None = None
    ) -> None:
        if not self.infer_axes():
            return
        if columns is False:
            return
        if columns is None or columns is True:
            columns = [a.cname for a in self.axes if a.is_data_indexable]
        if not isinstance(columns, (tuple, list)):
            columns = [columns]
        kw: dict[str, Any] = {}
        if optlevel is not None:
            kw["optlevel"] = optlevel
        if kind is not None:
            kw["kind"] = kind
        table = self.table
        for c in columns:
            v = getattr(table.cols, c, None)
            if v is not None:
                if v.is_indexed:
                    index = v.index
                    cur_optlevel = index.optlevel
                    cur_kind = index.kind
                    if kind is not None and cur_kind != kind:
                        v.remove_index()
                    else:
                        kw["kind"] = cur_kind
                    if optlevel is not None and cur_optlevel != optlevel:
                        v.remove_index()
                    else:
                        kw["optlevel"] = cur_optlevel
                if not v.is_indexed:
                    if v.type.startswith("complex"):
                        raise TypeError(
                            "Columns containing complex values can be stored but cannot be "
                            "indexed when using table format. Either use fixed format, set "
                            "index=False, or do not include the columns containing complex "
                            "values to data_columns when initializing the table."
                        )
                    v.create_index(**kw)
            elif c in self.non_index_axes[0][1]:
                raise AttributeError(
                    f"column {c} is not a data_column.\nIn order to read column {c} you must "
                    f"reload the dataframe \ninto HDFStore and include {c} with the "
                    "data_columns argument."
                )

    def _read_axes(self, where: Any, start: int | None = None, stop: int | None = None) -> list:
        selection = Selection(self, where=where, start=start, stop=stop)
        values = selection.select()
        results = []
        for a in self.axes:
            a.set_info(self.info)
            res = a.convert(
                values, nan_rep=self.nan_rep, encoding=self.encoding, errors=self.errors
            )
            results.append(res)
        return results

    @classmethod
    def get_object(cls, obj: Any, transposed: bool) -> Any:
        return obj

    def validate_data_columns(
        self, data_columns: Any, min_itemsize: Any, non_index_axes: Any
    ) -> list:
        if not len(non_index_axes):
            return []
        axis, axis_labels = non_index_axes[0]
        info = self.info.get(axis, {})
        if info.get("type") == "MultiIndex" and data_columns:
            raise ValueError(
                f"cannot use a multi-index on axis [{axis}] with data_columns {data_columns}"
            )
        if data_columns is True:
            data_columns = list(axis_labels)
        elif data_columns is None:
            data_columns = []
        if isinstance(min_itemsize, dict):
            existing_data_columns = set(data_columns)
            data_columns = list(data_columns)
            data_columns.extend(
                [k for k in min_itemsize if k != "values" and k not in existing_data_columns]
            )
        return [c for c in data_columns if c in axis_labels]

    def _create_axes(
        self,
        axes: Any,
        obj: Any,
        validate: bool = True,
        nan_rep: Any = None,
        data_columns: Any = None,
        min_itemsize: Any = None,
    ) -> Table:
        if not isinstance(obj, _Frame):
            group = self.group._v_name
            raise TypeError(
                f"cannot properly create the storer for: [group->{group},value->{type(obj)}]"
            )
        if axes is None:
            axes = [0]
        axes = [_axis_number(a) for a in axes]
        if self.infer_axes():
            table_exists = True
            axes = [a.axis for a in self.index_axes]
            data_columns = list(self.data_columns)
            nan_rep = self.nan_rep
        else:
            table_exists = False
        new_info = self.info
        assert self.ndim == 2
        if len(axes) != self.ndim - 1:
            raise ValueError("currently only support ndim-1 indexers in an AppendableTable")
        new_non_index_axes: list = []
        if nan_rep is None:
            nan_rep = "nan"
        idx = next(x for x in [0, 1] if x not in axes)
        a = obj.axes[idx]
        append_axis = list(a)
        if table_exists:
            indexer = len(new_non_index_axes)
            exist_axis = self.non_index_axes[indexer][1]
            if not _equivalent(append_axis, list(exist_axis)) and _equivalent(
                sorted(append_axis), sorted(exist_axis)
            ):
                append_axis = exist_axis
        info = new_info.setdefault(idx, {})
        info["names"] = list(a.names)
        info["type"] = a.cls
        new_non_index_axes.append((idx, append_axis))
        idx = axes[0]
        a = obj.axes[idx]
        axis_name = ["index", "columns"][idx]
        new_index = _convert_index(axis_name, a, self.encoding, self.errors)
        new_index.axis = idx
        new_index.set_pos(0)
        new_index.update_info(new_info)
        new_index.maybe_set_size(min_itemsize)
        new_index_axes = [new_index]
        j = len(new_index_axes)
        assert j == 1
        assert len(new_non_index_axes) == 1
        for a in new_non_index_axes:
            obj = _reindex_axis(obj, a[0], a[1])
        transposed = new_index.axis == 1
        data_columns = self.validate_data_columns(data_columns, min_itemsize, new_non_index_axes)
        frame = self.get_object(obj, transposed).consolidate()
        blocks, blk_items = self._get_blocks_and_items(
            frame, table_exists, new_non_index_axes, self.values_axes, data_columns
        )
        vaxes = []
        for i, (blk, b_items) in enumerate(zip(blocks, blk_items, strict=True)):
            klass: Any = DataCol
            name = None
            if data_columns and len(b_items) == 1 and b_items[0] in data_columns:
                klass = DataIndexableCol
                name = b_items[0]
                if not (name is None or isinstance(name, str)):
                    raise ValueError("cannot have non-object label DataIndexableCol")
            existing_col: DataCol | None
            if table_exists and validate:
                try:
                    existing_col = self.values_axes[i]
                except (IndexError, KeyError) as err:
                    raise ValueError(
                        f"Incompatible appended table [{blocks}]with existing table "
                        f"[{self.values_axes}]"
                    ) from err
            else:
                existing_col = None
            new_name = name or f"values_block_{i}"
            data_converted = _maybe_convert_for_string_atom(
                new_name,
                blk.values,
                existing_col=existing_col,
                min_itemsize=min_itemsize,
                nan_rep=nan_rep,
                encoding=self.encoding,
                errors=self.errors,
                columns=b_items,
            )
            adj_name = _maybe_adjust_name(new_name, self.version)
            typ = klass._get_atom(data_converted)
            kind = _dtype_to_kind(_dtype_name_of(data_converted))
            tz = None
            if getattr(data_converted, "tz", None) is not None:
                tz = _tz_attr(data_converted.tz)
            meta = metadata = ordered = None
            if isinstance(data_converted, _Arr) and data_converted.kind == "cat":
                ordered = data_converted.ordered
                meta = "category"
                metadata = np.asarray(data_converted.categories).ravel()
            elif blk.is_str:
                meta = "str"
            data, dtype_name = _get_data_and_dtype_name(data_converted)
            col = klass(
                name=adj_name,
                cname=new_name,
                values=list(b_items),
                typ=typ,
                pos=j,
                kind=kind,
                tz=tz,
                ordered=ordered,
                meta=meta,
                metadata=metadata,
                dtype=dtype_name,
                data=data,
            )
            col.update_info(new_info)
            vaxes.append(col)
            j += 1
        dcs = [col.name for col in vaxes if col.is_data_indexable]
        new_table = type(self)(
            parent=self.parent,
            group=self.group,
            encoding=self.encoding,
            errors=self.errors,
            index_axes=new_index_axes,
            non_index_axes=new_non_index_axes,
            values_axes=vaxes,
            data_columns=dcs,
            info=new_info,
            nan_rep=nan_rep,
        )
        if hasattr(self, "levels"):
            new_table.levels = self.levels
        new_table.validate_min_itemsize(min_itemsize)
        if validate and table_exists:
            new_table.validate(self)
        return new_table

    @staticmethod
    def _get_blocks_and_items(
        frame: _Frame,
        table_exists: bool,
        new_non_index_axes: Any,
        values_axes: Any,
        data_columns: Any,
    ) -> tuple[list[_Block], list[list[Any]]]:
        def get_blk_items(mgr: _Frame) -> list[list[Any]]:
            return [[mgr.columns.labels()[loc] for loc in blk.locs] for blk in mgr.blocks]

        mgr = frame
        blocks: list[_Block] = list(mgr.blocks)
        blk_items = get_blk_items(mgr)
        if len(data_columns):
            _axis, axis_labels = new_non_index_axes[0]
            new_labels = _difference(list(axis_labels), list(data_columns))
            mgr = frame.reindex_columns(new_labels)
            blocks = list(mgr.blocks)
            blk_items = get_blk_items(mgr)
            for c in data_columns:
                mgr = frame.reindex_columns([c])
                blocks.extend(mgr.blocks)
                blk_items.extend(get_blk_items(mgr))
        if table_exists:
            by_items = {
                tuple(b_items): (b, b_items) for b, b_items in zip(blocks, blk_items, strict=True)
            }
            new_blocks: list[_Block] = []
            new_blk_items = []
            for ea in values_axes:
                items = tuple(ea.values)
                try:
                    b, b_items = by_items.pop(items)
                    new_blocks.append(b)
                    new_blk_items.append(b_items)
                except (IndexError, KeyError) as err:
                    jitems = ",".join([_pprint(item) for item in items])
                    raise ValueError(
                        f"cannot match existing table structure for [{jitems}] on appending data"
                    ) from err
            blocks = new_blocks
            blk_items = new_blk_items
        return (blocks, blk_items)

    def process_axes(self, obj: Any, selection: Selection, columns: Any = None) -> Any:
        if columns is not None:
            columns = list(columns)
        if columns is not None and self.is_multi_index:
            assert isinstance(self.levels, list)
            for n in self.levels:
                if n not in columns:
                    columns.insert(0, n)
        for _axis, labels in self.non_index_axes:
            obj = _reindex_frame(obj, labels, columns)

        def process_filter(field: Any, filt: Any, op: Any) -> Any:
            for axis_name in ("index", "columns"):
                axis_number = 0 if axis_name == "index" else 1
                axis_values = [_scalar(v) for v in getattr(obj, axis_name)]
                if field == axis_name:
                    if self.is_multi_index:
                        filt = _union(list(filt), list(self.levels))
                    takers = op(axis_values, filt)
                    return _take_mask(obj, axis_number, takers)
                elif field in axis_values:
                    values = [_scalar(v) for v in obj[field].tolist()]
                    takers = op(values, filt)
                    return _take_mask(obj, 1 - axis_number, takers)
            raise ValueError(f"cannot find the field [{field}] for filtering!")

        if selection.filter is not None:
            for field, op, filt in selection.filter.format():
                obj = process_filter(field, filt, op)
        return obj

    def create_description(
        self, complib: Any, complevel: int | None, fletcher32: bool, expectedrows: int | None
    ) -> dict[str, Any]:
        if expectedrows is None:
            expectedrows = max(self.nrows_expected, 10000)
        d: dict[str, Any] = {"name": "table", "expectedrows": expectedrows}
        d["description"] = {a.cname: a.typ for a in self.axes}
        if complib:
            if complevel is None:
                complevel = self._complevel or 9
            filters = _tables().Filters(
                complevel=complevel, complib=complib, fletcher32=fletcher32 or self._fletcher32
            )
            d["filters"] = filters
        elif self._filters is not None:
            d["filters"] = self._filters
        return d

    def read_coordinates(
        self, where: Any = None, start: int | None = None, stop: int | None = None
    ) -> Any:
        self.validate_version(where)
        if not self.infer_axes():
            return False
        selection = Selection(self, where=where, start=start, stop=stop)
        coords = selection.select_coords()
        if selection.filter is not None:
            for field, op, filt in selection.filter.format():
                data = self.read_column(field, start=coords.min(), stop=coords.max() + 1)
                cells = np.empty(len(data), dtype=object)
                cells[:] = [_scalar(v) for v in data.tolist()]
                coords = coords[np.asarray(op(list(cells[coords - coords.min()]), filt), bool)]
        return _fp().Index(_fp().Series(np.asarray(coords, dtype=np.int64)))

    def read_column(
        self, column: str, where: Any = None, start: int | None = None, stop: int | None = None
    ) -> Any:
        self.validate_version()
        if not self.infer_axes():
            return False
        if where is not None:
            raise TypeError("read_column does not currently accept a where clause")
        for a in self.axes:
            if column == a.name:
                if not a.is_data_indexable:
                    raise ValueError(
                        f"column [{column}] can not be extracted individually; it is not data "
                        "indexable"
                    )
                c = getattr(self.table.cols, column)
                a.set_info(self.info)
                col_values = a.convert(
                    c[start:stop], nan_rep=self.nan_rep, encoding=self.encoding, errors=self.errors
                )
                cvs = col_values[1]
                dtype = getattr(self.table.attrs, f"{column}_meta", None)
                if not isinstance(cvs, (np.ndarray, _Arr)):
                    s = _fp().Series(cvs, name=column)
                    return s.astype(dtype) if dtype in ("str", "string") else s
                return _series_of(cvs, name=column, dtype=dtype)
        raise KeyError(f"column [{column}] not found in the table")


def _meta_values(values: Any) -> Any:
    """Category labels as pandas' `Series(values)` holds them."""
    values = np.asarray(values)
    if values.dtype.kind == "O" and _all_str(list(values)):
        return _Arr("str", values, "str")
    return values


def _axis_number(axis: Any) -> int:
    if axis in (0, "index", "rows"):
        return 0
    if axis in (1, "columns"):
        return 1
    raise ValueError(f"No axis named {axis} for object type DataFrame")


def _difference(labels: list[Any], drop: list[Any]) -> list[Any]:
    """`Index(labels).difference(Index(drop))`: unique, sorted when they sort."""
    out: list[Any] = []
    for v in labels:
        if any(_same_label(v, d) for d in drop) or any(_same_label(v, o) for o in out):
            continue
        out.append(v)
    with suppress(TypeError):
        out = sorted(out)
    return out


def _union(left: list[Any], right: list[Any]) -> list[Any]:
    out = list(left)
    for v in right:
        if not any(_same_label(v, o) for o in out):
            out.append(v)
    with suppress(TypeError):
        out = sorted(out)
    return out


def _take_mask(obj: Any, axis: int, takers: list[bool]) -> Any:
    positions = [i for i, keep in enumerate(takers) if keep]
    if axis == 0:
        return obj.iloc[positions]
    return obj.iloc[:, positions]


def _reindex_axis(obj: _Frame, axis: int, labels: Any, other: Any = None) -> _Frame:
    ax = obj.axes[axis].labels()
    labels = list(labels)
    if _equivalent(labels, ax):
        return obj
    unique: list[Any] = []
    for v in labels:
        if not any(_same_label(v, u) for u in unique):
            unique.append(v)
    if not _equivalent(unique, ax):
        obj = obj.reindex_columns(unique)
    return obj


def _reindex_frame(obj: Any, labels: Any, other: Any = None) -> Any:
    """pandas' `_reindex_axis` on the columns of a frame read from a table."""
    ax = [_scalar(v) for v in obj.columns]
    labels = list(labels)
    if other is not None:
        other = list(other)
    if (other is None or _equivalent(labels, other)) and _equivalent(labels, ax):
        return obj
    unique: list[Any] = []
    for v in labels:
        if not any(_same_label(v, u) for u in unique):
            unique.append(v)
    if other is not None:
        picked: list[Any] = []
        for v in other:
            if any(_same_label(v, u) for u in unique) and not any(
                _same_label(v, p) for p in picked
            ):
                picked.append(v)
        unique = picked
    if not _equivalent(unique, ax):
        positions = []
        for v in unique:
            positions.extend(i for i, c in enumerate(ax) if _same_label(c, v))
        obj = obj.iloc[:, positions]
    return obj


class WORMTable(Table):
    """A write-once table, which pandas never finished."""

    table_type = "worm"

    def read(
        self,
        where: Any = None,
        columns: Any = None,
        start: int | None = None,
        stop: int | None = None,
    ) -> Any:
        raise NotImplementedError("WORMTable needs to implement read")

    def write(self, obj: Any, **kwargs: Any) -> None:
        raise NotImplementedError("WORMTable needs to implement write")


class AppendableTable(Table):
    """A table rows can be appended to."""

    table_type = "appendable"

    def write(
        self,
        obj: Any,
        axes: Any = None,
        append: bool = False,
        complib: Any = None,
        complevel: Any = None,
        fletcher32: Any = None,
        min_itemsize: Any = None,
        chunksize: int | None = None,
        expectedrows: Any = None,
        dropna: bool = False,
        nan_rep: Any = None,
        data_columns: Any = None,
        track_times: bool = True,
    ) -> None:
        if not append and self.is_exists:
            self._handle.remove_node(self.group, "table")
        table = self._create_axes(
            axes=axes,
            obj=_obj_of(obj),
            validate=append,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            data_columns=data_columns,
        )
        for a in table.axes:
            a.validate_names()
        if not table.is_exists:
            options = table.create_description(
                complib=complib,
                complevel=complevel,
                fletcher32=fletcher32,
                expectedrows=expectedrows,
            )
            table.set_attrs()
            options["track_times"] = track_times
            table._handle.create_table(table.group, **options)
        table.attrs.info = table.info
        for a in table.axes:
            a.validate_and_set(table, append)
        table.write_data(chunksize, dropna=dropna)

    def write_data(self, chunksize: int | None, dropna: bool = False) -> None:
        names = self.dtype.names
        nrows = self.nrows_expected
        masks = []
        if dropna:
            for a in self.values_axes:
                if isinstance(a.data, np.ndarray) and a.data.ndim == 2:
                    mask = _isna_array(a.data).all(axis=0)
                    masks.append(mask.astype("u1", copy=False))
        if masks:
            mask = masks[0]
            for m in masks[1:]:
                mask = mask & m
            mask = mask.ravel()
        else:
            mask = None
        indexes = [a.cvalues for a in self.index_axes]
        nindexes = len(indexes)
        assert nindexes == 1, nindexes
        values = [a.take_data() for a in self.values_axes]
        values = [v.transpose(np.roll(np.arange(v.ndim), v.ndim - 1)) for v in values]
        bvalues = []
        for i, v in enumerate(values):
            new_shape = (nrows, *self.dtype[names[nindexes + i]].shape)
            bvalues.append(v.reshape(new_shape))
        if chunksize is None:
            chunksize = 100000
        rows = np.empty(min(chunksize, nrows), dtype=self.dtype)
        chunks = nrows // chunksize + 1
        for i in range(chunks):
            start_i = i * chunksize
            end_i = min((i + 1) * chunksize, nrows)
            if start_i >= end_i:
                break
            self.write_data_chunk(
                rows,
                indexes=[a[start_i:end_i] for a in indexes],
                mask=mask[start_i:end_i] if mask is not None else None,
                values=[v[start_i:end_i] for v in bvalues],
            )

    def write_data_chunk(self, rows: Any, indexes: list[Any], mask: Any, values: list[Any]) -> None:
        for v in values:
            if not math.prod(v.shape):
                return
        nrows = indexes[0].shape[0]
        if nrows != len(rows):
            rows = np.empty(nrows, dtype=self.dtype)
        names = self.dtype.names
        nindexes = len(indexes)
        for i, idx in enumerate(indexes):
            rows[names[i]] = idx
        for i, v in enumerate(values):
            rows[names[i + nindexes]] = v
        if mask is not None:
            m = ~mask.ravel().astype(bool, copy=False)
            if not m.all():
                rows = rows[m]
        if len(rows):
            self.table.append(rows)
            self.table.flush()

    def delete(self, where: Any = None, start: int | None = None, stop: int | None = None) -> Any:
        if where is None or not len(where):
            if start is None and stop is None:
                nrows = self.nrows
                self._handle.remove_node(self.group, recursive=True)
            else:
                if stop is None:
                    stop = self.nrows
                nrows = self.table.remove_rows(start=start, stop=stop)
                self.table.flush()
            return nrows
        if not self.infer_axes():
            return None
        table = self.table
        selection = Selection(self, where, start=start, stop=stop)
        values = np.sort(np.asarray(selection.select_coords()))
        ln = len(values)
        if ln:
            groups = [i for i in range(1, ln) if values[i] - values[i - 1] > 1]
            if not groups:
                groups = [0]
            if groups[-1] != ln:
                groups.append(ln)
            if groups[0] != 0:
                groups.insert(0, 0)
            pg = groups.pop()
            for g in reversed(groups):
                rows = values[g:pg]
                table.remove_rows(start=int(rows[0]), stop=int(rows[-1]) + 1)
                pg = g
            self.table.flush()
        return ln


class AppendableFrameTable(AppendableTable):
    """A DataFrame stored as a table."""

    pandas_kind = "frame_table"
    table_type = "appendable_frame"
    ndim = 2

    @property
    def obj_type(self) -> Any:
        return _fp().DataFrame

    @property
    def is_transposed(self) -> bool:
        return self.index_axes[0].axis == 1

    @classmethod
    def get_object(cls, obj: Any, transposed: bool) -> Any:
        if transposed:
            if obj.src is None:
                raise NotImplementedError("a transposed table needs a DataFrame")
            obj = _obj_of(obj.src.T)
        return obj

    def read(
        self,
        where: Any = None,
        columns: Any = None,
        start: int | None = None,
        stop: int | None = None,
    ) -> Any:
        self.validate_version(where)
        if not self.infer_axes():
            return None
        result = self._read_axes(where=where, start=start, stop=stop)
        info = self.info.get(self.non_index_axes[0][0], {}) if len(self.non_index_axes) else {}
        inds = [i for i, ax in enumerate(self.axes) if ax is self.index_axes[0]]
        assert len(inds) == 1
        ind = inds[0]
        index = result[ind][0]
        cells: list[tuple[Any, Any]] = []
        labels: list[Any] = []
        for i, a in enumerate(self.axes):
            if a not in self.values_axes:
                continue
            index_vals, cvalues = result[i]
            index_vals = list(index_vals)
            for j, label in enumerate(index_vals):
                if isinstance(cvalues, _Arr):
                    col = cvalues if cvalues.ndim == 1 else cvalues._with(cvalues.data[:, j])
                elif cvalues.ndim == 1:
                    col = cvalues
                else:
                    col = cvalues[:, j]
                meta = getattr(self.table.attrs, f"{label}_meta", None)
                dtype = meta if meta in ("str", "string") else None
                labels.append(label)
                cells.append((col, dtype))
        names = info.get("names")
        fp = _fp()
        if info.get("type") != "MultiIndex":
            cols = _labels_index(labels)
            if names is not None:
                cols = cols.set_names(names)
        else:
            cols = fp.MultiIndex.from_tuples(labels, names=names)
        if self.is_transposed:
            raise NotImplementedError("reading a transposed table is not supported")
        df = _frame_of(cells, index, cols)
        selection = Selection(self, where=where, start=start, stop=stop)
        df = self.process_axes(df, selection=selection, columns=columns)
        return df


def _labels_index(labels: list[Any]) -> Any:
    fp = _fp()
    if not labels:
        return fp.RangeIndex(0)
    return fp.Index(labels)


class AppendableSeriesTable(AppendableFrameTable):
    """A Series stored as a table."""

    pandas_kind = "series_table"
    table_type = "appendable_series"
    ndim = 2

    @property
    def obj_type(self) -> Any:
        return _fp().Series

    @property
    def is_transposed(self) -> bool:
        return False

    @classmethod
    def get_object(cls, obj: Any, transposed: bool) -> Any:
        return obj

    def write(self, obj: Any, data_columns: Any = None, **kwargs: Any) -> None:
        obj = _obj_of(obj)
        if not isinstance(obj, _Frame):
            name = obj.name or "values"
            obj = obj.to_frame(name)
        super().write(obj=obj, data_columns=obj.columns.labels(), **kwargs)

    def read(
        self,
        where: Any = None,
        columns: Any = None,
        start: int | None = None,
        stop: int | None = None,
    ) -> Any:
        is_multi_index = self.is_multi_index
        if columns is not None and is_multi_index:
            assert isinstance(self.levels, list)
            for n in self.levels:
                if n not in columns:
                    columns.insert(0, n)
        s = super().read(where=where, columns=columns, start=start, stop=stop)
        if is_multi_index:
            s = s.set_index(self.levels)
        s = s.iloc[:, 0]
        if s.name == "values":
            s = s.rename(None)
        return s


class AppendableMultiSeriesTable(AppendableSeriesTable):
    """A Series with a MultiIndex stored as a table."""

    pandas_kind = "series_table"
    table_type = "appendable_multiseries"

    def write(self, obj: Any, **kwargs: Any) -> None:
        name = obj.name or "values"
        newobj, self.levels = self.validate_multiindex(obj)
        assert isinstance(self.levels, list)
        cols = list(self.levels)
        cols.append(name)
        frame = _obj_of(newobj)
        frame.columns = _labels_ix(cols)
        super().write(obj=frame, **kwargs)


class GenericTable(AppendableFrameTable):
    """A PyTables table pandas did not write, read as a frame."""

    pandas_kind = "frame_table"
    table_type = "generic_table"
    ndim = 2
    levels: list[Any]

    @property
    def pandas_type(self) -> str:
        return self.pandas_kind

    @property
    def storable(self) -> Any:
        return getattr(self.group, "table", None) or self.group

    def get_attrs(self) -> None:
        self.non_index_axes = []
        self.nan_rep = None
        self.levels = []
        self.index_axes = [a for a in self.indexables if a.is_an_indexable]
        self.values_axes = [a for a in self.indexables if not a.is_an_indexable]
        self.data_columns = [a.name for a in self.values_axes]

    @functools.cached_property
    def indexables(self) -> list[Any]:
        d = self.description
        md = self.read_metadata("index")
        meta = "category" if md is not None else None
        index_col = GenericIndexCol(name="index", axis=0, table=self.table, meta=meta, metadata=md)
        _indexables: list[Any] = [index_col]
        for i, n in enumerate(d._v_names):
            assert isinstance(n, str)
            atom = getattr(d, n)
            md = self.read_metadata(n)
            meta = "category" if md is not None else None
            dc = GenericDataIndexableCol(
                name=n, pos=i, values=[n], typ=atom, table=self.table, meta=meta, metadata=md
            )
            _indexables.append(dc)
        return _indexables

    def write(self, **kwargs: Any) -> None:  # type: ignore[override]
        raise NotImplementedError("cannot write on a generic table")


class AppendableMultiFrameTable(AppendableFrameTable):
    """A DataFrame with a MultiIndex stored as a table."""

    table_type = "appendable_multiframe"
    ndim = 2
    _re_levels = re.compile(r"^level_\d+$")

    @property
    def table_type_short(self) -> str:
        return "appendable_multi"

    def write(self, obj: Any, data_columns: Any = None, **kwargs: Any) -> None:
        if data_columns is None:
            data_columns = []
        elif data_columns is True:
            data_columns = list(obj.columns)
        obj, self.levels = self.validate_multiindex(obj)
        assert isinstance(self.levels, list)
        for n in self.levels:
            if n not in data_columns:
                data_columns.insert(0, n)
        super().write(obj=obj, data_columns=data_columns, **kwargs)

    def read(
        self,
        where: Any = None,
        columns: Any = None,
        start: int | None = None,
        stop: int | None = None,
    ) -> Any:
        df = super().read(where=where, columns=columns, start=start, stop=stop)
        df = df.set_index(self.levels)
        names = [
            None if isinstance(name, str) and self._re_levels.search(name) else name
            for name in df.index.names
        ]
        df = df.rename_axis(names)
        return df


def _set_tz(values: Any, tz: Any, datetime64_dtype: str) -> Any:
    assert values.dtype == "i8", values.dtype
    data = values.view(datetime64_dtype)
    name = _tz_name(tz)
    if name is None:
        return data
    unit = np.datetime_data(data.dtype)[0]
    return _Arr("tz", data, f"datetime64[{unit}, {name}]", tz=name)


def _convert_index(name: str, index: _Ix, encoding: str, errors: str) -> IndexCol:
    assert isinstance(name, str)
    index_name = index.name
    converted, dtype_name = _get_data_and_dtype_name(index)
    kind = _dtype_to_kind(dtype_name)
    atom = DataIndexableCol._get_atom(converted)
    values = index.values
    np_kind = values.dtype.kind if isinstance(values, np.ndarray) else None
    needs_i8 = np_kind in ("M", "m") or (
        isinstance(values, _Arr) and values.kind in ("tz", "period")
    )
    if np_kind in ("i", "u", "b") or needs_i8:
        return IndexCol(
            name,
            values=converted,
            kind=kind,
            typ=atom,
            freq=index.freq,
            tz=index.tzinfo,
            index_name=index_name,
        )
    if index.cls == "MultiIndex":
        raise TypeError("MultiIndex not supported here!")
    inferred_type = _infer(index, skipna=False)
    values = np.asarray(index.plain())
    if inferred_type == "date":
        converted = np.asarray([v.toordinal() for v in values], dtype=np.int32)
        return IndexCol(name, converted, "date", _tables().Time32Col(), index_name=index_name)
    elif inferred_type == "string":
        converted = _convert_string_array(values, encoding, errors)
        itemsize = converted.dtype.itemsize
        return IndexCol(
            name, converted, "string", _tables().StringCol(itemsize), index_name=index_name
        )
    elif inferred_type in ["integer", "floating"]:
        return IndexCol(name, values=converted, kind=kind, typ=atom, index_name=index_name)
    else:
        assert isinstance(converted, np.ndarray) and converted.dtype == object
        assert kind == "object", kind
        atom = _tables().ObjectAtom()
        return IndexCol(name, converted, kind, atom, index_name=index_name)


def _unconvert_index(data: Any, kind: str, encoding: str, errors: str) -> Any:
    if kind.startswith("datetime64"):
        index = np.asarray(data, dtype="M8[ns]") if kind == "datetime64" else data.view(kind)
    elif kind.startswith("timedelta64"):
        index = np.asarray(data, dtype="m8[ns]") if kind == "timedelta64" else data.view(kind)
    elif kind == "date":
        try:
            index = np.asarray([datetime.date.fromordinal(v) for v in data], dtype=object)
        except ValueError:
            index = np.asarray([datetime.date.fromtimestamp(v) for v in data], dtype=object)
    elif kind in ("integer", "float", "bool"):
        index = np.asarray(data)
    elif kind in "string":
        index = _unconvert_string_array(data, nan_rep=None, encoding=encoding, errors=errors)
    elif kind == "object":
        index = np.asarray(data[0])
    else:
        raise ValueError(f"unrecognized index type {kind}")
    return index


def _maybe_convert_for_string_atom(
    name: str,
    bvalues: Any,
    existing_col: Any,
    min_itemsize: Any,
    nan_rep: Any,
    encoding: Any,
    errors: Any,
    columns: list[Any],
) -> Any:
    if isinstance(bvalues, _Arr) and bvalues.kind == "str":
        bvalues = bvalues.to_numpy()
    if isinstance(bvalues, _Arr) or bvalues.dtype != object:
        return bvalues
    dtype_name = bvalues.dtype.name
    inferred_type = _infer(bvalues, skipna=False)
    if inferred_type == "date":
        raise TypeError("[date] is not implemented as a table column")
    if inferred_type == "datetime":
        raise TypeError("too many timezones in this block, create separate data columns")
    if not (inferred_type == "string" or dtype_name == "object"):
        return bvalues
    mask = _isna_array(bvalues)
    data = bvalues.copy()
    data[mask] = nan_rep
    if existing_col and mask.any() and len(nan_rep) > existing_col.itemsize:
        raise ValueError("NaN representation is too large for existing column size")
    inferred_type = _infer(data, skipna=False)
    if inferred_type != "string":
        for i in range(data.shape[0]):
            col = data[i]
            inferred_type = _infer(col if isinstance(col, np.ndarray) else [col], skipna=False)
            if inferred_type != "string":
                error_column_label = columns[i] if len(columns) > i else f"No.{i}"
                raise TypeError(
                    f"Cannot serialize the column [{error_column_label}]\nbecause its data "
                    f"contents are not [string] but [{inferred_type}] object dtype"
                )
    data_converted = _convert_string_array(data, encoding, errors).reshape(data.shape)
    itemsize = data_converted.itemsize
    if isinstance(min_itemsize, dict):
        min_itemsize = int(min_itemsize.get(name) or min_itemsize.get("values") or 0)
    itemsize = max(min_itemsize or 0, itemsize)
    if existing_col is not None:
        eci = existing_col.validate_col(itemsize)
        if eci is not None and eci > itemsize:
            itemsize = eci
    data_converted = data_converted.astype(f"|S{itemsize}", copy=False)
    return data_converted


def _max_len(values: Any) -> int:
    return max((len(v) for v in values if isinstance(v, (str, bytes))), default=0)


def _convert_string_array(data: Any, encoding: str, errors: str) -> Any:
    if len(data):
        flat = [v.encode(encoding, errors) if isinstance(v, str) else np.nan for v in data.ravel()]
        out = np.empty(len(flat), dtype=object)
        out[:] = flat
        data = out.reshape(data.shape)
    itemsize = max(1, _max_len(np.asarray(data, dtype=object).ravel()))
    data = np.asarray(data, dtype=f"S{itemsize}")
    return data


def _unconvert_string_array(data: Any, nan_rep: Any, encoding: str, errors: str) -> Any:
    shape = data.shape
    data = np.asarray(data.ravel(), dtype=object)
    if len(data):
        itemsize = _max_len(data)
        dtype = f"U{itemsize}"
        if isinstance(data[0], bytes):
            decoded = [
                v.decode(encoding, errors=errors) if isinstance(v, bytes) else np.nan for v in data
            ]
            data = np.empty(len(decoded), dtype=object)
            data[:] = decoded
        else:
            data = data.astype(dtype, copy=False).astype(object, copy=False)
    if nan_rep is None:
        nan_rep = "nan"
    for i, v in enumerate(data):
        if isinstance(v, str) and v == nan_rep:
            data[i] = np.nan
    return data.reshape(shape)


def _maybe_convert(values: Any, val_kind: str, encoding: str, errors: str) -> Any:
    assert isinstance(val_kind, str), type(val_kind)
    if _need_convert(val_kind):
        conv = _get_converter(val_kind, encoding, errors)
        values = conv(values)
    return values


def _get_converter(kind: str, encoding: str, errors: str) -> Any:
    if kind == "datetime64":
        return lambda x: np.asarray(x, dtype="M8[ns]")
    elif "datetime64" in kind:
        return lambda x: np.asarray(x, dtype=kind)
    elif kind == "string":
        return lambda x: _unconvert_string_array(x, nan_rep=None, encoding=encoding, errors=errors)
    else:
        raise ValueError(f"invalid kind {kind}")


def _need_convert(kind: str) -> bool:
    return bool(kind in ("datetime64", "string") or "datetime64" in kind)


def _maybe_adjust_name(name: str, version: Any) -> str:
    if isinstance(version, str) or len(version) < 3:
        raise ValueError("Version is incorrect, expected sequence of 3 integers.")
    if version[0] == 0 and version[1] <= 10 and version[2] == 0:
        m = re.search(r"values_block_(\d+)", name)
        if m:
            grp = m.groups()[0]
            name = f"values_{grp}"
    return name


def _dtype_to_kind(dtype_str: Any) -> str:
    dtype_str = _decoded_kind(dtype_str)
    if dtype_str is None:
        raise AttributeError("'NoneType' object has no attribute 'startswith'")
    if dtype_str.startswith(("string", "bytes")):
        kind = "string"
    elif dtype_str.startswith("float"):
        kind = "float"
    elif dtype_str.startswith("complex"):
        kind = "complex"
    elif dtype_str.startswith(("int", "uint")):
        kind = "integer"
    elif dtype_str.startswith("datetime64") or dtype_str.startswith("timedelta"):
        kind = dtype_str
    elif dtype_str.startswith("bool"):
        kind = "bool"
    elif dtype_str.startswith("category"):
        kind = "category"
    elif dtype_str.startswith("period"):
        kind = "integer"
    elif dtype_str == "object":
        kind = "object"
    elif dtype_str == "str":
        kind = "str"
    else:
        raise ValueError(f"cannot interpret dtype of [{dtype_str}]")
    return kind


def _get_data_and_dtype_name(data: Any) -> tuple[Any, str]:
    if isinstance(data, _Ix):
        values = data.values
        if data.cls == "MultiIndex":
            return data.plain(), "object"
        if isinstance(values, _Arr):
            if values.kind == "tz":
                return np.asarray(values.data.view("i8")), f"datetime64[{values.unit}]"
            if values.kind == "period":
                return np.asarray(values.data), values.dtype_name
            if values.kind == "cat":
                return np.asarray(values.data), values.data.dtype.name
            return np.asarray(values.data), values.dtype_name
        data = values
    if isinstance(data, _Arr):
        if data.kind == "cat":
            return np.asarray(data.data), data.data.dtype.name
        if data.kind == "tz":
            return np.asarray(data.data.view("i8")), f"datetime64[{data.unit}]"
        if data.kind == "period":
            return np.asarray(data.objects), data.dtype_name
        return np.asarray(data.data), data.dtype_name
    dtype_name = data.dtype.name
    if data.dtype.kind in "mM":
        data = np.asarray(data.view("i8"))
    data = np.asarray(data)
    return (data, dtype_name)


class Selection:
    """The rows of a table a where clause, a list of rows or a range picks."""

    def __init__(
        self, table: Table, where: Any = None, start: int | None = None, stop: int | None = None
    ) -> None:
        self.table = table
        self.where = where
        self.start = start
        self.stop = stop
        self.condition: Any = None
        self.filter: Any = None
        self.terms: Any = None
        self.coordinates: Any = None
        if is_list_like(where) and not isinstance(where, PyTablesExpr):
            with suppress(ValueError):
                items = _as_array(where) if hasattr(where, "to_numpy") else where
                inferred = _infer(items, skipna=False)
                if inferred in ("integer", "boolean"):
                    where = np.asarray(items)
                    if where.dtype == np.bool_:
                        start, stop = (self.start, self.stop)
                        if start is None:
                            start = 0
                        if stop is None:
                            stop = self.table.nrows
                        self.coordinates = np.arange(start, stop)[where]
                    elif issubclass(where.dtype.type, np.integer):
                        if (self.start is not None and (where < self.start).any()) or (
                            self.stop is not None and (where >= self.stop).any()
                        ):
                            raise ValueError("where must have index locations >= start and < stop")
                        self.coordinates = where
        if self.coordinates is None:
            self.terms = self.generate(where)
            if self.terms is not None:
                self.condition, self.filter = self.terms.evaluate()

    def generate(self, where: Any) -> Any:
        if where is None:
            return None
        q = self.table.queryables()
        try:
            return PyTablesExpr(where, queryables=q, encoding=self.table.encoding)
        except NameError as err:
            qkeys = ",".join(q.keys())
            msg = dedent(
                f"""\
                The passed where expression: {where}
                            contains an invalid variable reference
                            all of the variable references must be a reference to
                            an axis (e.g. 'index' or 'columns'), or a data_column
                            The currently defined references are: {qkeys}
                """
            )
            raise ValueError(msg) from err

    def select(self) -> Any:
        if self.condition is not None:
            return self.table.table.read_where(
                self.condition.format(), start=self.start, stop=self.stop
            )
        elif self.coordinates is not None:
            return self.table.table.read_coordinates(self.coordinates)
        return self.table.table.read(start=self.start, stop=self.stop)

    def select_coords(self) -> Any:
        start, stop = (self.start, self.stop)
        nrows = self.table.nrows
        if start is None:
            start = 0
        elif start < 0:
            start += nrows
        if stop is None:
            stop = nrows
        elif stop < 0:
            stop += nrows
        if self.condition is not None:
            return self.table.table.get_where_list(
                self.condition.format(), start=start, stop=stop, sort=True
            )
        elif self.coordinates is not None:
            return self.coordinates
        return np.arange(start, stop)


for _cls in (
    AppendableFrameTable,
    AppendableMultiFrameTable,
    AppendableMultiSeriesTable,
    AppendableSeriesTable,
    AppendableTable,
    BlockManagerFixed,
    DataCol,
    DataIndexableCol,
    Fixed,
    FrameFixed,
    GenericDataIndexableCol,
    GenericFixed,
    GenericIndexCol,
    GenericTable,
    IndexCol,
    Selection,
    SeriesFixed,
    Table,
    TableIterator,
    WORMTable,
):
    _cls.__module__ = "firepanda.io.pytables"
to_hdf.__module__ = "firepanda.io.pytables"
HDFStore.__module__ = "firepanda"
