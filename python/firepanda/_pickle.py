"""Pickling a frame, a column or an index, and `to_pickle` and `read_pickle`.

A frame's columns live in Arrow buffers the extension owns, which pickle
cannot reach. They do leave through the Arrow C data interface, which every
frame already speaks for `from_arrow`, so a frame pickles as the bytes of that
export: each Arrow schema's format, name and metadata, and each array's
length, offsets and buffers, copied out with `ctypes`. Unpickling lays the
same structures out again in memory and hands them to `from_arrow`, so the
columns come back with their exact Arrow types, gaps and categories, without
going through Python values and without pyarrow.

The row labels go into the same export as leading columns, the way
`reset_index` puts them, and come back out with `set_index`. The column
labels are the Arrow names, as a firepanda label is text. The names of the row
label levels, `attrs` and the duplicate labels flag are kept beside the export
as ordinary Python values.

pandas can read nothing firepanda writes and firepanda can read nothing
pandas writes, as each pickle names the classes of the library that wrote it.
"""

from __future__ import annotations

import ctypes
import os
import pickle
from typing import Any

_VERSION = 1
"""The layout of the state below, so a later layout can still read this one."""

_LEVEL = "__firepanda_level_{}__"
"""The temporary name of a row label level while it travels as a column."""

_VALUE = "__firepanda_value__"
"""The temporary name of a column while it travels as a frame."""


class _Schema(ctypes.Structure):
    pass


_Schema._fields_ = [
    ("format", ctypes.c_char_p),
    ("name", ctypes.c_char_p),
    ("metadata", ctypes.c_void_p),
    ("flags", ctypes.c_int64),
    ("n_children", ctypes.c_int64),
    ("children", ctypes.POINTER(ctypes.POINTER(_Schema))),
    ("dictionary", ctypes.POINTER(_Schema)),
    ("release", ctypes.c_void_p),
    ("private_data", ctypes.c_void_p),
]


class _Array(ctypes.Structure):
    pass


_Array._fields_ = [
    ("length", ctypes.c_int64),
    ("null_count", ctypes.c_int64),
    ("offset", ctypes.c_int64),
    ("n_buffers", ctypes.c_int64),
    ("n_children", ctypes.c_int64),
    ("buffers", ctypes.POINTER(ctypes.c_void_p)),
    ("children", ctypes.POINTER(ctypes.POINTER(_Array))),
    ("dictionary", ctypes.POINTER(_Array)),
    ("release", ctypes.c_void_p),
    ("private_data", ctypes.c_void_p),
]

_SCHEMA_NAME = b"arrow_schema"
_ARRAY_NAME = b"arrow_array"

_capsule_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_capsule_pointer.restype = ctypes.c_void_p
_capsule_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_capsule_new = ctypes.pythonapi.PyCapsule_New
_capsule_new.restype = ctypes.py_object
_capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

_FIXED = {"c": 1, "C": 1, "s": 2, "S": 2, "i": 4, "I": 4, "l": 8, "L": 8, "e": 2, "f": 4, "g": 8}
"""The bytes of one value of each primitive Arrow format."""

_TEMPORAL = {
    "tdD": 4,
    "tdm": 8,
    "tts": 4,
    "ttm": 4,
    "ttu": 8,
    "ttn": 8,
    "tiM": 4,
    "tiD": 8,
    "tin": 16,
}
"""The bytes of one value of each date, time and interval format; instants and spans are 8."""


def _bits(count: int) -> int:
    return (count + 7) // 8


def _read(address: int, size: int) -> bytes:
    return ctypes.string_at(address, size)


def _integers(address: int, width: int, count: int) -> list[int]:
    kind = ctypes.c_int32 if width == 4 else ctypes.c_int64
    return list((kind * count).from_address(address))


def _metadata(address: int | None) -> bytes | None:
    """The metadata block as bytes: a count, then each key and value after its length."""
    if not address:
        return None
    count = ctypes.c_int32.from_address(address).value
    size = 4
    for _ in range(2 * count):
        size += 4 + ctypes.c_int32.from_address(address + size).value
    return _read(address, size)


def _widths(fmt: str, array: _Array) -> list[int]:
    """The bytes of each buffer of an array of this format, which the C interface leaves out."""
    count = array.offset + array.length
    pointers = [array.buffers[k] for k in range(array.n_buffers)]
    if fmt == "n" or fmt.startswith("+r"):
        return []
    if fmt == "b":
        return [_bits(count), _bits(count)]
    if fmt in _FIXED:
        return [_bits(count), _FIXED[fmt] * count]
    if fmt[:3] in _TEMPORAL:
        return [_bits(count), _TEMPORAL[fmt[:3]] * count]
    if fmt.startswith(("ts", "tD")):
        return [_bits(count), 8 * count]
    if fmt.startswith("w:"):
        return [_bits(count), int(fmt[2:]) * count]
    if fmt.startswith("d:"):
        parts = fmt[2:].split(",")
        return [_bits(count), (int(parts[2]) if len(parts) > 2 else 128) // 8 * count]
    if fmt in ("u", "z", "U", "Z"):
        width = 4 if fmt in ("u", "z") else 8
        end = _integers(pointers[1], width, count + 1)[-1] if pointers[1] else 0
        return [_bits(count), width * (count + 1), end]
    if fmt in ("vu", "vz"):
        variadic = array.n_buffers - 3
        sizes = _integers(pointers[-1], 8, variadic) if variadic else []
        return [_bits(count), 16 * count, *sizes, 8 * variadic]
    if fmt in ("+l", "+m"):
        return [_bits(count), 4 * (count + 1)]
    if fmt == "+L":
        return [_bits(count), 8 * (count + 1)]
    if fmt in ("+vl", "+vL"):
        width = 4 if fmt == "+vl" else 8
        return [_bits(count), width * count, width * count]
    if fmt == "+s" or fmt.startswith("+w:"):
        return [_bits(count)]
    if fmt.startswith("+ud:"):
        return [count, 4 * count]
    if fmt.startswith("+us:"):
        return [count]
    raise TypeError(f"cannot pickle a column of Arrow format {fmt!r}")


def _schema_state(schema: _Schema) -> dict[str, Any]:
    return {
        "format": schema.format,
        "name": schema.name,
        "metadata": _metadata(schema.metadata),
        "flags": schema.flags,
        "children": [_schema_state(schema.children[k][0]) for k in range(schema.n_children)],
        "dictionary": _schema_state(schema.dictionary[0]) if schema.dictionary else None,
    }


def _array_state(array: _Array, schema: _Schema) -> dict[str, Any]:
    fmt = schema.format.decode()
    widths = _widths(fmt, array)
    buffers = [
        _read(array.buffers[k], widths[k]) if array.buffers[k] else None
        for k in range(array.n_buffers)
    ]
    return {
        "length": array.length,
        "null_count": array.null_count,
        "offset": array.offset,
        "buffers": buffers,
        "children": [
            _array_state(array.children[k][0], schema.children[k][0])
            for k in range(array.n_children)
        ],
        "dictionary": (
            _array_state(array.dictionary[0], schema.dictionary[0]) if array.dictionary else None
        ),
    }


def _exported(capsules: list[Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The schema and the array behind a pair of Arrow capsules, as plain values."""
    schema = _Schema.from_address(_capsule_pointer(capsules[0], _SCHEMA_NAME))
    array = _Array.from_address(_capsule_pointer(capsules[1], _ARRAY_NAME))
    return _schema_state(schema), _array_state(array, schema)


_HELD: dict[int, list[Any]] = {}
"""What each rebuilt export keeps alive, by the number in its private data, until released."""

_RELEASE_SCHEMA = ctypes.CFUNCTYPE(None, ctypes.POINTER(_Schema))
_RELEASE_ARRAY = ctypes.CFUNCTYPE(None, ctypes.POINTER(_Array))


def _released(pointer: Any) -> None:
    """The release callback: forgets what the export held and marks it released."""
    held = pointer.contents
    if held.private_data:
        _HELD.pop(held.private_data, None)
    held.release = None


_release_schema = _RELEASE_SCHEMA(_released)
_release_array = _RELEASE_ARRAY(_released)
_RELEASE_SCHEMA_ADDRESS = ctypes.cast(_release_schema, ctypes.c_void_p).value
_RELEASE_ARRAY_ADDRESS = ctypes.cast(_release_array, ctypes.c_void_p).value


def _memory(data: bytes, keep: list[Any]) -> int:
    """The address of a copy of the bytes, aligned to eight and kept alive by `keep`."""
    block = (ctypes.c_uint64 * max(1, (len(data) + 7) // 8))()
    ctypes.memmove(block, data, len(data))
    keep.append(block)
    return ctypes.addressof(block)


def _schema_built(state: dict[str, Any], keep: list[Any]) -> _Schema:
    schema = _Schema()
    keep.append(schema)
    schema.format = state["format"]
    schema.name = state["name"]
    if state["metadata"] is not None:
        schema.metadata = _memory(state["metadata"], keep)
    schema.flags = state["flags"]
    children = [_schema_built(child, keep) for child in state["children"]]
    schema.n_children = len(children)
    if children:
        pointers = (ctypes.POINTER(_Schema) * len(children))(*map(ctypes.pointer, children))
        keep.append(pointers)
        schema.children = pointers
    if state["dictionary"] is not None:
        schema.dictionary = ctypes.pointer(_schema_built(state["dictionary"], keep))
    schema.release = _RELEASE_SCHEMA_ADDRESS
    return schema


def _array_built(state: dict[str, Any], keep: list[Any]) -> _Array:
    array = _Array()
    keep.append(array)
    array.length = state["length"]
    array.null_count = state["null_count"]
    array.offset = state["offset"]
    buffers = state["buffers"]
    array.n_buffers = len(buffers)
    if buffers:
        pointers = (ctypes.c_void_p * len(buffers))(
            *(None if data is None else _memory(data, keep) for data in buffers)
        )
        keep.append(pointers)
        array.buffers = pointers
    children = [_array_built(child, keep) for child in state["children"]]
    array.n_children = len(children)
    if children:
        links = (ctypes.POINTER(_Array) * len(children))(*map(ctypes.pointer, children))
        keep.append(links)
        array.children = links
    if state["dictionary"] is not None:
        array.dictionary = ctypes.pointer(_array_built(state["dictionary"], keep))
    array.release = _RELEASE_ARRAY_ADDRESS
    return array


class _Export:
    """A rebuilt export, which `from_arrow` reads through `__arrow_c_array__`."""

    def __init__(self, schema: dict[str, Any], array: dict[str, Any]) -> None:
        self._schema = schema
        self._array = array

    def __arrow_c_array__(self, requested_schema: Any = None) -> tuple[Any, Any]:
        capsules = []
        for built, state, name in (
            (_schema_built, self._schema, _SCHEMA_NAME),
            (_array_built, self._array, _ARRAY_NAME),
        ):
            keep: list[Any] = []
            root = built(state, keep)
            key = id(keep)
            _HELD[key] = keep
            root.private_data = key
            capsules.append(_capsule_new(ctypes.addressof(root), name, None))
        return capsules[0], capsules[1]


def _carried(obj: Any) -> tuple[dict[Any, Any], bool]:
    return dict(obj.attrs), obj.flags.allows_duplicate_labels


def _carry(obj: Any, carried: tuple[dict[Any, Any], bool]) -> Any:
    attrs, allows = carried
    if attrs:
        obj.attrs = attrs
    if not allows:
        obj.flags.allows_duplicate_labels = False
    return obj


def frame_state(frame: Any) -> dict[str, Any]:
    """Everything a frame is, as values pickle can write."""
    names = list(frame.index.names)
    levels = [_LEVEL.format(k) for k in range(len(names))]
    flat = frame.rename_axis(levels).reset_index()
    schema, array = _exported(flat._inner.arrow_c_array(None))
    return {
        "version": _VERSION,
        "schema": schema,
        "array": array,
        "levels": levels,
        "index_names": names,
        "carried": _carried(frame),
        "plain_columns": bool(getattr(frame, "_plain_columns", False)),
    }


def frame_rebuilt(state: dict[str, Any]) -> Any:
    """The frame `frame_state` describes."""
    from ._frame import from_arrow

    frame = from_arrow(_Export(state["schema"], state["array"])).set_index(state["levels"])
    frame = frame.rename_axis(state["index_names"])
    if state.get("plain_columns"):
        from ._attrs import hold_plain_columns

        hold_plain_columns(frame, True)
    return _carry(frame, state["carried"])


def column_state(column: Any) -> dict[str, Any]:
    """Everything a column is, as values pickle can write."""
    return {
        "frame": frame_state(column.to_frame(_VALUE)),
        "name": column.name,
        "carried": _carried(column),
    }


def column_rebuilt(state: dict[str, Any]) -> Any:
    """The column `column_state` describes."""
    column = frame_rebuilt(state["frame"]).iloc[:, 0].rename(state["name"])
    return _carry(column, state["carried"])


def index_state(index: Any) -> dict[str, Any]:
    """Everything an index is, as values pickle can write."""
    step = getattr(index, "step", None)
    if step is not None:
        return {"range": (index.start, index.stop, step), "name": index.name}
    return {
        "frame": frame_state(index.to_frame(index=False, name=_VALUE)),
        "name": index.name,
        "kind": type(index),
    }


def index_rebuilt(state: dict[str, Any]) -> Any:
    """The index `index_state` describes."""
    from ._frame import Index
    from ._range_index import RangeIndex

    if "range" in state:
        return RangeIndex(*state["range"], name=state["name"])
    labels = Index(frame_rebuilt(state["frame"]).iloc[:, 0])
    if state["kind"] is not Index:
        labels = state["kind"](labels)
    return labels.rename(state["name"])


_BY_END = {
    ".gz": "gzip",
    ".bz2": "bz2",
    ".zip": "zip",
    ".xz": "xz",
    ".zst": "zstd",
    ".tar": "tar",
    ".tar.gz": "tar",
    ".tgz": "tar",
    ".tar.bz2": "tar",
    ".tar.xz": "tar",
}
"""The compression each file name ending means, as pandas infers it."""

_TAR_BY_END = {".tar.gz": "gz", ".tgz": "gz", ".tar.bz2": "bz2", ".tar.xz": "xz"}
"""How a tar archive is itself compressed, by the name's ending."""


def _method(path: Any, compression: Any) -> tuple[str | None, dict[str, Any]]:
    options = dict(compression) if isinstance(compression, dict) else {"method": compression}
    method = options.pop("method", None)
    if method == "infer":
        name = str(path).lower() if isinstance(path, (str, os.PathLike)) else ""
        method = next((kind for end, kind in _BY_END.items() if name.endswith(end)), None)
    known = (None, "gzip", "bz2", "zip", "xz", "zstd", "tar")
    if method not in known:
        raise ValueError(
            f"Unrecognized compression type: {method}\n"
            "Valid compression types are ['infer', None, 'bz2', 'gzip', 'tar', 'xz', 'zip', 'zstd']"
        )
    return method, options


def _zstd() -> Any:
    try:
        from compression import zstd  # type: ignore[import-not-found]
    except ImportError:
        try:
            import zstandard as zstd  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError(
                "`Import zstandard` failed.  Use pip or conda to install the zstandard package."
            ) from None
    return zstd


def _packed(
    data: bytes, method: str | None, options: dict[str, Any], name: str, file: str
) -> bytes:
    """The pickle's bytes compressed the way `to_pickle` was asked to.

    `name` is what a zip or a tar calls the pickle inside it, and `file` is
    the name of the file written, empty for a handle.
    """
    import io

    if method is None:
        return data
    if method in ("gzip", "bz2", "xz"):
        import bz2
        import gzip
        import lzma

        if method == "gzip":
            # pandas hands gzip the path, so the header keeps the file's name.
            out = io.BytesIO()
            with gzip.GzipFile(
                filename=file,
                mode="wb",
                fileobj=out,
                compresslevel=options.get("compresslevel", 9),
                mtime=options.get("mtime", 0),
            ) as packing:
                packing.write(data)
            return out.getvalue()
        return (bz2 if method == "bz2" else lzma).compress(data)
    if method == "zstd":
        return _zstd().compress(data)
    out = io.BytesIO()
    inner = options.get("archive_name") or name
    if method == "zip":
        import zipfile

        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(inner, data)
    else:
        import tarfile

        with tarfile.open(name=file or None, fileobj=out, mode=options.get("mode", "w")) as archive:
            member = tarfile.TarInfo(inner)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return out.getvalue()


def _unpacked(data: bytes, method: str | None) -> bytes:
    """The pickle's bytes out of whatever `read_pickle` was handed."""
    import io

    if method is None:
        return data
    if method in ("gzip", "bz2", "xz"):
        import bz2
        import gzip
        import lzma

        return {"gzip": gzip, "bz2": bz2, "xz": lzma}[method].decompress(data)
    if method == "zstd":
        return _zstd().decompress(data)
    if method == "zip":
        import zipfile

        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
            if len(names) != 1:
                raise ValueError(
                    f"Multiple files found in ZIP file. Only one file per ZIP: {names}"
                )
            return archive.read(names[0])
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        files = [member for member in archive.getmembers() if member.isfile()]
        if len(files) != 1:
            raise ValueError(
                "Multiple files found in TAR archive. Only one file per TAR archive: "
                f"{[member.name for member in files]}"
            )
        handle = archive.extractfile(files[0])
        assert handle is not None
        return handle.read()


def _inner_name(path: Any) -> str:
    name = os.path.basename(os.fspath(path)) if isinstance(path, (str, os.PathLike)) else "data"
    for end in (".zip", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".tar"):
        if name.lower().endswith(end):
            return name[: -len(end)]
    return name


def _refuse_storage(storage_options: Any, path: Any) -> None:
    if storage_options is not None:
        raise ValueError("storage_options passed with file object or non-fsspec file path")


def to_pickle(
    obj: Any,
    filepath_or_buffer: Any,
    compression: Any = "infer",
    protocol: int = 5,
    storage_options: Any = None,
) -> None:
    """Pickles an object to a file or a binary handle, compressed as the name says.

    Args:
        obj: Anything pickle takes, a frame, a column or an index included.
        filepath_or_buffer: A path, or a handle with `write`.
        compression: `"infer"` to go by the name's ending, None, or one of
            `"gzip"`, `"bz2"`, `"zip"`, `"xz"`, `"zstd"` and `"tar"`, alone or
            as the `"method"` of a dictionary of options.
        protocol: The pickle protocol, the highest when negative.
        storage_options: Options for a remote file, which is not supported.

    Raises:
        ValueError: For a compression pandas does not know, or storage options
            on a local file.
    """
    if protocol < 0:
        protocol = pickle.HIGHEST_PROTOCOL
    _bytes_written(
        pickle.dumps(obj, protocol=protocol), filepath_or_buffer, compression, storage_options
    )


def _bytes_written(data: bytes, target: Any, compression: Any, storage_options: Any) -> None:
    """Writes bytes to a path or a binary handle, compressed as the name or `compression` says.

    Raises:
        ValueError: For a compression pandas does not know, or storage options
            on a local file.
    """
    _refuse_storage(storage_options, target)
    method, options = _method(target, compression)
    if method == "tar" and "mode" not in options:
        name = str(target).lower()
        packing = next((kind for end, kind in _TAR_BY_END.items() if name.endswith(end)), "")
        options["mode"] = f"w:{packing}"
    data = _packed(
        data,
        method,
        options,
        _inner_name(target),
        "" if hasattr(target, "write") else os.path.basename(os.fspath(target)),
    )
    if hasattr(target, "write"):
        target.write(data)
        return
    with open(os.path.expanduser(os.fspath(target)), "wb") as handle:
        handle.write(data)


def read_pickle(
    filepath_or_buffer: Any, compression: Any = "infer", storage_options: Any = None
) -> Any:
    """Reads back what `to_pickle` wrote, or any other pickle.

    Reading a pickle runs whatever code it names, so read only pickles from
    a source you trust.

    Args:
        filepath_or_buffer: A path, or a handle with `read`.
        compression: As for `to_pickle`, where `"infer"` goes by the name.
        storage_options: Options for a remote file, which is not supported.

    Returns:
        The object.

    Raises:
        FileNotFoundError: For a path with no file.
    """
    _refuse_storage(storage_options, filepath_or_buffer)
    method, _ = _method(filepath_or_buffer, compression)
    if hasattr(filepath_or_buffer, "read"):
        data = filepath_or_buffer.read()
    else:
        with open(os.path.expanduser(os.fspath(filepath_or_buffer)), "rb") as handle:
            data = handle.read()
    return pickle.loads(_unpacked(data, method))
