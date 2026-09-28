"""`json_normalize`, which flattens nested records into a frame.

A port of pandas' `pandas/io/json/_normalize.py`. The work is all on Python
dicts and lists before a frame exists, so it is copied step for step, and the
frame is built at the end from the flat records with `DataFrame`. The one
change is the metadata columns: pandas repeats them through an object array,
and here they are repeated as a list, which the frame types by its values.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def _gap(value: Any) -> bool:
    """Whether a lone value is missing, the way `pd.isna` reads a scalar."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def nested_to_record(
    ds: Any, prefix: str = "", sep: str = ".", level: int = 0, max_level: int | None = None
) -> Any:
    """A dict, or a list of them, with nested dicts flattened into dotted keys."""
    singleton = isinstance(ds, dict)
    if singleton:
        ds = [ds]
    new_ds = []
    for d in ds:
        new_d = copy.deepcopy(d)
        for k, v in d.items():
            key = k if isinstance(k, str) else str(k)
            newkey = key if level == 0 else prefix + sep + key
            if not isinstance(v, dict) or (max_level is not None and level >= max_level):
                if level != 0:
                    new_d[newkey] = new_d.pop(k)
                continue
            new_d.pop(k)
            new_d.update(nested_to_record(v, newkey, sep, level + 1, max_level))
        new_ds.append(new_d)
    return new_ds[0] if singleton else new_ds


def _normalize_json(data: Any, key_string: str, normalized: dict[str, Any], sep: str) -> dict:
    """Walks a value to its leaves, writing each under its joined key."""
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{key_string}{sep}{key}"
            if not key_string:
                new_key = new_key.removeprefix(sep)
            _normalize_json(value, new_key, normalized, sep)
    else:
        normalized[key_string] = data
    return normalized


def _simple_json_normalize(ds: Any, sep: str = ".") -> Any:
    """The fast path pandas takes when only `sep` is given: the flat keys first."""
    if isinstance(ds, list):
        return [_simple_json_normalize(row, sep=sep) for row in ds]
    if isinstance(ds, dict):
        top = {k: v for k, v in ds.items() if not isinstance(v, dict)}
        nested = {k: v for k, v in ds.items() if isinstance(v, dict)}
        return {**top, **_normalize_json(nested, "", {}, sep)}
    return {}


def _validate_meta(meta: Any) -> None:
    """Refuses a `meta` holding anything but names and lists of names, as pandas does."""
    if meta is None or isinstance(meta, str):
        return
    for item in meta:
        if isinstance(item, list):
            for subitem in item:
                if not isinstance(subitem, str):
                    raise TypeError(
                        "All elements in nested meta paths must be strings. "
                        f"Found {type(subitem).__name__}: {subitem!r}"
                    )
        elif not isinstance(item, str):
            raise TypeError(
                "All elements in 'meta' must be strings or lists of strings. "
                f"Found {type(item).__name__}: {item!r}"
            )


def json_normalize(
    data: Any,
    record_path: Any = None,
    meta: Any = None,
    meta_prefix: str | None = None,
    record_prefix: str | None = None,
    errors: str = "raise",
    sep: str = ".",
    max_level: int | None = None,
) -> Any:
    """Normalizes semi-structured JSON data into a flat table, which is `pandas.json_normalize`.

    Args:
        data: A dict, a list of dicts, or a column of them.
        record_path: The path to the list of records in each object, when the rows
            are nested inside it.
        meta: Fields of the outer objects to repeat on every record, each a name or
            a path.
        meta_prefix: Put before the name of every metadata column.
        record_prefix: Put before the name of every record column.
        errors: `ignore` fills a missing metadata field with NaN, `raise` refuses it.
        sep: What joins the keys of a nested dict into a column name.
        max_level: How many levels of nested dicts to flatten, and all when None.

    Returns:
        The frame.
    """
    from ._frame import DataFrame, Series

    _validate_meta(meta)

    def pull_field(js: Any, spec: Any, extract_record: bool = False) -> Any:
        result = js
        try:
            if isinstance(spec, list):
                for field in spec:
                    if result is None:
                        raise KeyError(field)
                    result = result[field]
            else:
                result = result[spec]
        except KeyError as e:
            if extract_record:
                raise KeyError(
                    f"Key {e} not found. If specifying a record_path, all elements of "
                    f"data should have the path."
                ) from e
            if errors == "ignore":
                return math.nan
            raise KeyError(
                f"Key {e} not found. To replace missing values of {e} with "
                f"np.nan, pass in errors='ignore'"
            ) from e
        return result

    def pull_records(js: Any, spec: Any) -> list:
        result = pull_field(js, spec, extract_record=True)
        if not isinstance(result, list):
            if _gap(result):
                return []
            raise TypeError(
                f"Path must contain list or null, but got {type(result).__name__} at {spec!r}"
            )
        return result

    index = list(data.index) if isinstance(data, Series) else None
    if isinstance(data, Series):
        data = data.tolist()
    if isinstance(data, list) and not data:
        return DataFrame()
    if isinstance(data, dict):
        data = [data]
    elif isinstance(data, Iterable) and not isinstance(data, str):
        data = list(data)
        for i, item in enumerate(data):
            if isinstance(item, dict):
                continue
            if _gap(item):
                data[i] = {}
            else:
                raise TypeError(
                    "All items in data must be of type dict or NA-like, "
                    f"found {type(item).__name__}"
                )
    else:
        raise NotImplementedError

    if (
        record_path is None
        and meta is None
        and meta_prefix is None
        and record_prefix is None
        and max_level is None
    ):
        return DataFrame(_simple_json_normalize(data, sep=sep), index=index)

    if record_path is None:
        if any(isinstance(x, dict) for y in data for x in y.values()):
            data = nested_to_record(data, sep=sep, max_level=max_level)
        result = DataFrame(data, index=index)
        if record_prefix is not None:
            result = result.rename(columns=lambda x: f"{record_prefix}{x}")
        return result
    if not isinstance(record_path, list):
        record_path = [record_path]

    if meta is None:
        meta = []
    elif not isinstance(meta, list):
        meta = [meta]
    paths = [m if isinstance(m, list) else [m] for m in meta]

    records: list = []
    lengths: list[int] = []
    meta_vals: defaultdict[str, list] = defaultdict(list)
    meta_keys = [sep.join(val) for val in paths]

    def recursive_extract(data: Any, path: list, seen_meta: dict, level: int = 0) -> None:
        if isinstance(data, dict):
            data = [data]
        if len(path) > 1:
            for obj in data:
                for val, key in zip(paths, meta_keys, strict=True):
                    if level + 1 == len(val):
                        seen_meta[key] = pull_field(obj, val[-1])
                recursive_extract(obj[path[0]], path[1:], seen_meta, level=level + 1)
            return
        for obj in data:
            recs = [
                nested_to_record(r, sep=sep, max_level=max_level) if isinstance(r, dict) else r
                for r in pull_records(obj, path[0])
            ]
            lengths.append(len(recs))
            for val, key in zip(paths, meta_keys, strict=True):
                if level + 1 > len(val):
                    meta_vals[key].append(seen_meta[key])
                else:
                    meta_vals[key].append(pull_field(obj, val[level:]))
            records.extend(recs)

    recursive_extract(data, record_path, {}, level=0)

    if records and not isinstance(records[0], dict):
        records = [{0: r} for r in records]
    result = DataFrame(records)
    if record_prefix is not None:
        result = result.rename(columns=lambda x: f"{record_prefix}{x}")
    for k, v in meta_vals.items():
        name = k if meta_prefix is None else meta_prefix + k
        if name in result:
            raise ValueError(f"Conflicting metadata name {name}, need distinguishing prefix ")
        repeated = [value for value, n in zip(v, lengths, strict=True) for _ in range(n)]
        result = result.assign(**{name: repeated})
    if index is not None:
        repeated = [label for label, n in zip(index, lengths, strict=True) for _ in range(n)]
        result = result.set_axis(repeated)
    return result
