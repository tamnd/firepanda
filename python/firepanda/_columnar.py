"""Parquet, Feather and ORC, read and written through pyarrow when it is installed.

firepanda has no runtime dependencies, and these three formats are pyarrow's to
read and write, so pyarrow is imported only when one of them is asked for. A
frame already speaks the Arrow stream protocol, so it goes to pyarrow as it is,
with no Python object made per value.

What makes a file one pandas reads back as it wrote it is the `pandas` key in
the schema's metadata, which pyarrow writes from a pandas frame and reads when
it hands a table back to pandas. It says which columns are the row labels, or
that the labels are a range with a start, stop and step, what each column's
pandas type was, and since pandas 3 the frame's `attrs`. This module writes that
key in the layout pyarrow 25 writes for pandas 3.0.3 and reads it back, so a
file firepanda writes is read by pandas with its labels and types, and the
other way round.
"""

from __future__ import annotations

import errno
import importlib
import io
import json
import os
from typing import Any

from . import _optional
from ._pandas import NO_DEFAULT

_LEVEL = "__index_level_{}__"
_ATTRS = b"PANDAS_ATTRS"
# The pandas whose layout of the `pandas` metadata key this writes.
_PANDAS_FORMAT = "3.0.3"
_FLOATS = {"halffloat": "float16", "float": "float32", "double": "float64"}


def _imported(name: str) -> Any:
    """A pyarrow module, or pandas' ImportError when pyarrow is not installed."""
    try:
        return importlib.import_module(name)
    except ImportError:
        raise _optional.missing("pyarrow") from None


def _numpy_type(kind: Any) -> str:
    """The name pandas gives the numpy side of an Arrow type, as pyarrow writes it."""
    pa = _imported("pyarrow")
    text = str(kind)
    if pa.types.is_integer(kind) or pa.types.is_boolean(kind):
        return text
    if pa.types.is_floating(kind):
        return _FLOATS.get(text, text)
    if pa.types.is_timestamp(kind):
        return f"datetime64[{kind.unit}]"
    if pa.types.is_duration(kind):
        return f"timedelta64[{kind.unit}]"
    if pa.types.is_dictionary(kind):
        return str(kind.index_type)
    if pa.types.is_string(kind) or pa.types.is_large_string(kind):
        return "str"
    return "object"


def _described(table: Any, field: str, name: Any) -> dict[str, Any]:
    """One entry of the metadata's `columns`, for the field of that name."""
    pa = _imported("pyarrow")
    from pyarrow.pandas_compat import get_logical_type

    kind = table.schema.field(field).type
    extra: dict[str, Any] | None = None
    if pa.types.is_timestamp(kind) and kind.tz is not None:
        extra = {"timezone": kind.tz}
    elif pa.types.is_dictionary(kind):
        chunks = table.column(field).chunks
        extra = {
            "num_categories": len(chunks[0].dictionary) if chunks else 0,
            "ordered": kind.ordered,
        }
    elif pa.types.is_decimal(kind):
        extra = {"precision": kind.precision, "scale": kind.scale}
    return {
        "name": name,
        "field_name": field,
        "pandas_type": get_logical_type(kind),
        "numpy_type": _numpy_type(kind),
        "metadata": extra,
    }


def _wide_type(pa: Any, kind: Any) -> Any:
    if pa.types.is_string_view(kind):
        return pa.large_string()
    if pa.types.is_binary_view(kind):
        return pa.large_binary()
    return kind


def _widened(table: Any) -> Any:
    """The table with its view columns as large ones, which every reader knows.

    A category column holds its categories in a dictionary, and pyarrow cannot
    cast a dictionary of views, so each chunk is rebuilt from its codes and its
    categories cast on their own.
    """
    pa = _imported("pyarrow")
    for position, field in enumerate(table.schema):
        kind = field.type
        if pa.types.is_dictionary(kind):
            wide = _wide_type(pa, kind.value_type)
            if wide == kind.value_type:
                continue
            chunks = [
                pa.DictionaryArray.from_arrays(
                    chunk.indices, chunk.dictionary.cast(wide), ordered=kind.ordered
                )
                for chunk in table.column(position).chunks
            ]
            column = pa.chunked_array(chunks, pa.dictionary(kind.index_type, wide, kind.ordered))
        else:
            wide = _wide_type(pa, kind)
            if wide == kind:
                continue
            column = table.column(position).cast(wide)
        table = table.set_column(position, field.with_type(column.type), column)
    return table


def _range_of(labels: Any) -> tuple[int, int, int] | None:
    """The start, stop and step of labels that are a range of integers, or None."""
    if not str(labels.dtype).startswith(("int", "uint")):
        return None
    count = len(labels)
    if count == 0:
        return (0, 0, 1)
    first = int(labels[0])
    if count == 1:
        return (first, first + 1, 1)
    gaps = labels.to_series().diff().iloc[1:]
    step = int(gaps.iloc[0])
    if step == 0 or gaps.min() != gaps.max():
        return None
    return (first, first + step * count, step)


def table_of(frame: Any, index: bool | None, schema: Any = None) -> Any:
    """A pyarrow table of the frame, with the `pandas` metadata pandas writes.

    Args:
        frame: The frame.
        index: True to store the row labels as columns, False to drop them, and
            None to store a range of integers as metadata alone and anything
            else as columns, as pandas does.
        schema: A pyarrow schema the table is cast to, as `to_parquet` takes.

    Returns:
        The table.
    """
    pa = _imported("pyarrow")
    labels = frame.index
    names = list(labels.names)
    ranged = _range_of(labels) if index is None and len(names) == 1 else None
    fields: list[str] = []
    ranges: list[dict[str, Any]] = []
    if index is False or ranged is not None:
        flat = frame.reset_index(drop=True)
        if ranged is not None:
            start, stop, step = ranged
            ranges.append(
                {"kind": "range", "name": names[0], "start": start, "stop": stop, "step": step}
            )
    else:
        fields = [_LEVEL.format(k) if n is None else str(n) for k, n in enumerate(names)]
        flat = frame.rename_axis(fields).reset_index()
        flat = flat[[*frame.columns, *fields]]
    table = _widened(pa.table(flat))
    if schema is not None:
        table = table.select(schema.names).cast(schema)
    present = set(table.column_names)
    stored = [field for field in fields if field in present]
    level_names = dict(zip(fields, names, strict=True)) if fields else {}
    described = [
        _described(table, field, level_names.get(field, field)) for field in table.column_names
    ]
    attrs = dict(frame.attrs)
    metadata = {
        "index_columns": stored + ranges,
        "column_indexes": [
            {
                "name": None,
                "field_name": None,
                "pandas_type": "unicode",
                "numpy_type": "str",
                "metadata": {"encoding": "UTF-8"},
            }
        ],
        "columns": described,
        "attributes": attrs,
        "creator": {"library": "firepanda", "version": _version()},
        "pandas_version": _PANDAS_FORMAT,
    }
    merged = dict(table.schema.metadata or {})
    merged[b"pandas"] = json.dumps(metadata).encode()
    if attrs:
        merged[_ATTRS] = json.dumps(attrs).encode()
    return table.replace_schema_metadata(merged)


def _version() -> str:
    from . import __version__

    return __version__


def frame_of(table: Any, dtype_backend: Any = NO_DEFAULT) -> Any:
    """A frame of a pyarrow table, with the labels and `attrs` its metadata names.

    Args:
        table: The table, as pyarrow read it.
        dtype_backend: pandas' default types when not given, as
            `DataFrame.from_arrow` reads a table, and Arrow's types as
            `firepanda.from_arrow` keeps them for `"numpy_nullable"` and
            `"pyarrow"`, which are the two that keep a missing integer an integer.

    Returns:
        The frame.
    """
    from ._frame import DataFrame, from_arrow
    from ._pandas import _backend
    from ._range_index import RangeIndex

    raw = table.schema.metadata or {}
    metadata = json.loads(raw[b"pandas"]) if b"pandas" in raw else {}
    present = set(table.column_names)
    fields = [c for c in metadata.get("index_columns", []) if isinstance(c, str) and c in present]
    ranges = [c for c in metadata.get("index_columns", []) if isinstance(c, dict)]
    names = {c.get("field_name"): c.get("name") for c in metadata.get("columns", [])}
    table = table.replace_schema_metadata(None)
    if dtype_backend is NO_DEFAULT:
        frame = DataFrame.from_arrow(table)
    else:
        _backend(dtype_backend)
        frame = from_arrow(table)
    if fields:
        frame = frame.set_index(fields).rename_axis([names.get(f) for f in fields])
    elif len(ranges) == 1 and ranges[0].get("kind") == "range":
        spec = ranges[0]
        labels = RangeIndex(spec["start"], spec["stop"], spec["step"], name=spec.get("name"))
        if len(labels) == len(frame):
            frame = frame.set_axis(labels)
    attrs = metadata.get("attributes") or {}
    if _ATTRS in raw:
        attrs = json.loads(raw[_ATTRS])
    if attrs:
        frame.attrs = attrs
    return frame


def _local(path: Any, reading: bool = False) -> Any:
    """A path as pyarrow is handed it: a handle as it is, a local path expanded.

    pandas opens a local file with Python's `open`, so a missing one raises
    Python's `FileNotFoundError` rather than pyarrow's, and so does this. A
    folder is left to pyarrow, which reads it as a partitioned dataset.
    """
    if hasattr(path, "read") or hasattr(path, "write"):
        return path
    text = os.fspath(path)
    if "://" in str(text):
        return text
    text = os.path.expanduser(text)
    if reading and not os.path.exists(text):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), text)
    return text


# What pandas says about storage options it cannot use: through its general file
# opener, and through the Parquet one, which words it differently.
_HANDLE_STORAGE = "storage_options passed with file object or non-fsspec file path"
_PARQUET_STORAGE = "storage_options passed with buffer, or non-supported URL"


def _storage(storage_options: Any, path: Any, message: str = _HANDLE_STORAGE) -> None:
    """Refuses storage options the way pandas does for a local file or a handle."""
    if storage_options is None:
        return
    if isinstance(path, (str, os.PathLike)) and "://" in os.fspath(path):
        raise NotImplementedError(
            "storage_options is not supported yet, because firepanda hands a URL to"
            " pyarrow's own file systems rather than to fsspec"
        )
    raise ValueError(message)


def _engine(engine: Any) -> None:
    if engine not in ("auto", "pyarrow", "fastparquet"):
        raise ValueError("engine must be one of 'pyarrow', 'fastparquet'")
    if engine == "fastparquet":
        raise NotImplementedError(
            "engine='fastparquet' is not supported, because firepanda reads and writes"
            " Parquet through pyarrow"
        )


def _is_frame(frame: Any, what: str) -> None:
    from ._pandas import DataFrameMixin

    if not isinstance(frame, DataFrameMixin):
        raise ValueError(what)


def to_parquet(
    frame: Any,
    path: Any = None,
    engine: Any = "auto",
    compression: Any = "snappy",
    index: bool | None = None,
    partition_cols: Any = None,
    storage_options: Any = None,
    filesystem: Any = None,
    **kwargs: Any,
) -> bytes | None:
    """Writes the frame as Parquet, as `DataFrame.to_parquet` does.

    Returns:
        The file's bytes when no path is given, and None otherwise.
    """
    _engine(engine)
    _is_frame(frame, "to_parquet only supports IO with DataFrames")
    _storage(storage_options, path, _PARQUET_STORAGE)
    parquet = _imported("pyarrow.parquet")
    table = table_of(frame, index, kwargs.pop("schema", None))
    target = io.BytesIO() if path is None else _local(path)
    if partition_cols is not None:
        parquet.write_to_dataset(
            table,
            target,
            compression=compression,
            partition_cols=partition_cols,
            filesystem=filesystem,
            **kwargs,
        )
    else:
        parquet.write_table(table, target, compression=compression, filesystem=filesystem, **kwargs)
    return target.getvalue() if path is None else None


def read_parquet(
    path: Any,
    engine: str = "auto",
    columns: Any = None,
    storage_options: Any = None,
    dtype_backend: Any = NO_DEFAULT,
    filesystem: Any = None,
    filters: Any = None,
    to_pandas_kwargs: Any = None,
    **kwargs: Any,
) -> Any:
    """Reads a Parquet file, or a folder of them, into a frame.

    The row labels, the types and `attrs` come back from the `pandas` metadata
    when a file has it, which pandas and firepanda both write.

    Args:
        path: A path, a folder, a URL pyarrow knows, or a binary handle.
        engine: `"auto"` or `"pyarrow"`.
        columns: The columns to read, all of them when None.
        storage_options: Refused, as pandas refuses them for a local file.
        dtype_backend: See `frame_of`.
        filesystem: A pyarrow file system, handed to pyarrow.
        filters: Row filters, handed to pyarrow.
        to_pandas_kwargs: Refused, since there is no pandas frame to make.
        **kwargs: Handed to `pyarrow.parquet.read_table`.

    Returns:
        The frame.

    Raises:
        ValueError: For an engine pandas does not know.
        NotImplementedError: For `fastparquet` and `to_pandas_kwargs`.
    """
    _engine(engine)
    _storage(storage_options, path, _PARQUET_STORAGE)
    if to_pandas_kwargs is not None:
        raise NotImplementedError(
            "to_pandas_kwargs is not supported, because it is handed to pyarrow's"
            " to_pandas and firepanda reads the table without making a pandas frame"
        )
    parquet = _imported("pyarrow.parquet")
    kwargs["use_pandas_metadata"] = True
    table = parquet.read_table(
        _local(path, reading=True),
        columns=columns,
        filesystem=filesystem,
        filters=filters,
        **kwargs,
    )
    return frame_of(table, dtype_backend)


def to_feather(frame: Any, path: Any, **kwargs: Any) -> None:
    """Writes the frame as Feather, which is the Arrow IPC file format."""
    _is_frame(frame, "feather only support IO with DataFrames")
    _storage(kwargs.pop("storage_options", None), path)
    feather = _imported("pyarrow.feather")
    feather.write_feather(table_of(frame, None), _local(path), **kwargs)


def read_feather(
    path: Any,
    columns: Any = None,
    use_threads: bool = True,
    storage_options: Any = None,
    dtype_backend: Any = NO_DEFAULT,
) -> Any:
    """Reads a Feather file into a frame, with the labels its metadata names."""
    _storage(storage_options, path)
    feather = _imported("pyarrow.feather")
    table = feather.read_table(_local(path, reading=True), columns=columns, use_threads=use_threads)
    return frame_of(table, dtype_backend)


def to_orc(
    frame: Any,
    path: Any = None,
    engine: Any = "pyarrow",
    index: bool | None = None,
    engine_kwargs: Any = None,
) -> bytes | None:
    """Writes the frame as ORC, which keeps no row labels, as `DataFrame.to_orc` does.

    Returns:
        The file's bytes when no path is given, and None otherwise.

    Raises:
        ValueError: For row labels other than the default range, or a named one,
            and an engine other than pyarrow, in pandas' words.
        NotImplementedError: For a column type ORC cannot hold.
    """
    labels = frame.index
    if _range_of(labels) != (0, len(frame), 1) or len(labels.names) != 1:
        raise ValueError(
            "orc does not support serializing a non-default index for the index; "
            "you can .reset_index() to make the index into column(s)"
        )
    if labels.name is not None:
        raise ValueError("orc does not serialize index meta-data on a default index")
    if engine != "pyarrow":
        raise ValueError("engine must be 'pyarrow'")
    pa = _imported("pyarrow")
    orc = _imported("pyarrow.orc")
    target = io.BytesIO() if path is None else _local(path)
    try:
        orc.write_table(table_of(frame, index), target, **(engine_kwargs or {}))
    except (TypeError, pa.ArrowNotImplementedError) as error:
        raise NotImplementedError(
            "The dtype of one or more columns is not supported yet."
        ) from error
    return target.getvalue() if path is None else None


def read_orc(
    path: Any,
    columns: Any = None,
    dtype_backend: Any = NO_DEFAULT,
    filesystem: Any = None,
    **kwargs: Any,
) -> Any:
    """Reads an ORC file into a frame."""
    orc = _imported("pyarrow.orc")
    table = orc.read_table(
        _local(path, reading=True), columns=columns, filesystem=filesystem, **kwargs
    )
    return frame_of(table, dtype_backend)
