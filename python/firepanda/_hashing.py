"""Row hashes, which is `pandas.util.hash_pandas_object` and `pandas.util.hash_array`.

pandas hashes each value to a 64 bit number and then mixes the bits. A number
is hashed as its own bits, a date or a span as its count of ticks, and text as
the SipHash-2-4 of its encoded bytes under a 16 byte key. With `categorize`
the distinct texts are hashed once, and a gap hashes to all ones. The column
hashes of a row are folded together with the index hash the way pandas folds
them, so every number here equals the one pandas gives.
"""

from __future__ import annotations

import math
import struct
from typing import Any

from .errors import InvalidArgumentError

_KEY = "0123456789123456"
_MASK = (1 << 64) - 1
_GAP = _MASK
_MASKED_GAP = 2**61 - 1
_NAT = -(2**63) & _MASK
_TICKS = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}


def _rotated(x: int, bits: int) -> int:
    return ((x << bits) | (x >> (64 - bits))) & _MASK


def _siphash(key: bytes, data: bytes) -> int:
    """SipHash-2-4 of the bytes under the 16 byte key, as pandas computes it."""
    k0 = int.from_bytes(key[:8], "little")
    k1 = int.from_bytes(key[8:16], "little")
    v0 = k0 ^ 0x736F6D6570736575
    v1 = k1 ^ 0x646F72616E646F6D
    v2 = k0 ^ 0x6C7967656E657261
    v3 = k1 ^ 0x7465646279746573

    def rounds(v0: int, v1: int, v2: int, v3: int, count: int) -> tuple[int, int, int, int]:
        for _ in range(count):
            v0 = (v0 + v1) & _MASK
            v1 = _rotated(v1, 13) ^ v0
            v0 = _rotated(v0, 32)
            v2 = (v2 + v3) & _MASK
            v3 = _rotated(v3, 16) ^ v2
            v0 = (v0 + v3) & _MASK
            v3 = _rotated(v3, 21) ^ v0
            v2 = (v2 + v1) & _MASK
            v1 = _rotated(v1, 17) ^ v2
            v2 = _rotated(v2, 32)
        return v0, v1, v2, v3

    size = len(data)
    end = size - size % 8
    for start in range(0, end, 8):
        word = int.from_bytes(data[start : start + 8], "little")
        v3 ^= word
        v0, v1, v2, v3 = rounds(v0, v1, v2, v3, 2)
        v0 ^= word
    last = ((size & 0xFF) << 56) | int.from_bytes(data[end:], "little")
    v3 ^= last
    v0, v1, v2, v3 = rounds(v0, v1, v2, v3, 2)
    v0 ^= last
    v2 ^= 0xFF
    v0, v1, v2, v3 = rounds(v0, v1, v2, v3, 4)
    return v0 ^ v1 ^ v2 ^ v3


def _mixed(value: int) -> int:
    """The bit mixing pandas applies to every hash that is not a gap."""
    value ^= value >> 30
    value = (value * 13787848793156543929) & _MASK
    value ^= value >> 27
    value = (value * 10723151780598845931) & _MASK
    return value ^ (value >> 31)


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return type(value).__name__ in ("NaTType", "NAType")


def _key_bytes(hash_key: str, encoding: str) -> bytes:
    key = hash_key.encode(encoding)
    if len(key) != 16:
        raise InvalidArgumentError(
            f"key should be a 16-byte string encoded, got {key!r} (len {len(key)})"
        )
    return key


def _text_hashes(values: list[Any], encoding: str, hash_key: str) -> list[int]:
    """Each object's SipHash, with a gap or a tuple hashed as its printed form."""
    key = _key_bytes(hash_key, encoding)
    data = []
    for value in values:
        if isinstance(value, bytes):
            data.append(value)
        elif isinstance(value, str):
            data.append(value.encode(encoding))
        elif value is None or (isinstance(value, float) and math.isnan(value)):
            data.append(str(value).encode(encoding))
        elif isinstance(value, tuple):
            hash(value)
            data.append(str(value).encode(encoding))
        else:
            data = [str(item).encode(encoding) for item in values]
            break
    return [_mixed(_siphash(key, item)) for item in data]


def _float_bits(value: Any, size: int) -> int:
    code = {2: ("e", "H"), 4: ("f", "I"), 8: ("d", "Q")}[size]
    return struct.unpack(code[1], struct.pack(code[0], float(value)))[0]


def _number_bits(dtype: str, value: Any) -> int:
    """The unsigned bits of a number the way numpy stores it in a column of the dtype."""
    name = dtype.lower()
    if name in ("bool", "boolean"):
        return int(bool(value))
    if name.startswith("float"):
        return _float_bits(value, int(name[5:] or 64) // 8)
    bits = int(name.lstrip("uint") or 64)
    return int(value) & ((1 << bits) - 1)


def _ticks(value: Any, unit: str) -> int:
    if _missing(value):
        return _NAT
    return (value.value // _TICKS[unit]) & _MASK


def _unit(dtype: str) -> str:
    return dtype[dtype.index("[") + 1 :].split(",")[0].rstrip("]").strip()


def _factorized(values: list[Any]) -> tuple[list[int], list[Any]]:
    """The code of each value and the distinct values in the order they first appear."""
    seen: dict[Any, int] = {}
    uniques: list[Any] = []
    codes = []
    for value in values:
        if _missing(value):
            codes.append(-1)
            continue
        found = (type(value), value) if isinstance(value, (bool, float)) else value
        code = seen.get(found)
        if code is None:
            code = seen[found] = len(uniques)
            uniques.append(value)
        codes.append(code)
    return codes, uniques


def _taken(hashed: list[int], codes: list[int]) -> list[int]:
    return [_GAP if code < 0 else hashed[code] for code in codes]


def _values_hashes(
    dtype: str, values: list[Any], encoding: str, hash_key: str, categorize: bool
) -> list[int]:
    """The hash of each value in a column of the dtype, before any folding."""
    name = dtype.lower()
    if name in _MASKED and (dtype[:1].isupper() or name == "boolean"):
        return [
            _MASKED_GAP if _missing(value) else _mixed(_number_bits(_MASKED[name], value))
            for value in values
        ]
    if name == "bool" or name.startswith(("int", "uint", "float")):
        return [_mixed(_number_bits(name, value)) for value in values]
    if name.startswith(("datetime64", "timedelta64")):
        unit = _unit(dtype)
        return [_mixed(_ticks(value, unit)) for value in values]
    if name.startswith("period"):
        return [_mixed(_NAT if _missing(v) else v.ordinal & _MASK) for v in values]
    if name.startswith("complex"):
        return _complex_hashes(name, values)
    if name == "object" and values and all(isinstance(v, complex) for v in values):
        return _complex_hashes("complex128", values)
    if categorize:
        codes, uniques = _factorized(values)
        return _taken(_text_hashes(uniques, encoding, hash_key), codes)
    if name in ("str", "string"):
        values = [None if _missing(value) else value for value in values]
    return _text_hashes(values, encoding, hash_key)


_MASKED = {
    **{f"int{bits}": f"int{bits}" for bits in (8, 16, 32, 64)},
    **{f"uint{bits}": f"uint{bits}" for bits in (8, 16, 32, 64)},
    "float32": "float32",
    "float64": "float64",
    "boolean": "bool",
}
"""The nullable dtypes, by lower case name, and the numpy dtype that holds their data."""


def _complex_hashes(name: str, values: list[Any]) -> list[int]:
    if name == "complex64":
        return [
            _mixed(_float_bits(value.real, 4) | (_float_bits(value.imag, 4) << 32))
            for value in values
        ]
    return [
        (_mixed(_float_bits(value.real, 8)) + 23 * _mixed(_float_bits(value.imag, 8))) & _MASK
        for value in values
    ]


def _categorical_hashes(categories: Any, codes: list[int], encoding: str, hash_key: str) -> list:
    hashed = _values_hashes(
        str(categories.dtype), categories.tolist(), encoding, hash_key, categorize=False
    )
    return _taken(hashed, codes) if hashed else [0] * len(codes)


def _column_hashes(column: Any, encoding: str, hash_key: str, categorize: bool) -> list[int]:
    """The hash of each value of a series or an index."""
    dtype = str(column.dtype)
    if dtype == "category":
        holder = column.cat if hasattr(column, "cat") else column
        codes = [int(code) for code in holder.codes.tolist()]
        return _categorical_hashes(holder.categories, codes, encoding, hash_key)
    return _values_hashes(dtype, column.tolist(), encoding, hash_key, categorize)


def _combined(arrays: list[list[int]], count: int) -> list[int]:
    """The column hashes folded into one hash per row, as pandas folds them."""
    if not arrays:
        return []
    out = [3430008] * len(arrays[0])
    factor = 1000003
    for position, array in enumerate(arrays):
        reverse = count - position
        out = [((row ^ value) * factor) & _MASK for row, value in zip(out, array, strict=True)]
        factor = (factor + 82520 + reverse + reverse) & _MASK
    return [(row + 97531) & _MASK for row in out]


def _multi_hashes(index: Any, encoding: str, hash_key: str) -> list[int]:
    arrays = [
        _categorical_hashes(level, [int(code) for code in codes.tolist()], encoding, hash_key)
        for level, codes in zip(index.levels, index.codes, strict=True)
    ]
    return _combined(arrays, len(arrays))


def _index_hashes(index: Any, encoding: str, hash_key: str, categorize: bool) -> list[int]:
    from ._multi import MultiIndex

    if isinstance(index, MultiIndex):
        return _multi_hashes(index, encoding, hash_key)
    return _column_hashes(index, encoding, hash_key, categorize)


def _unsigned(hashed: list[int], index: Any) -> Any:
    """The hashes as a uint64 series, passed in as their signed twins and cast back."""
    from . import Series

    signed = [value - (1 << 64) if value >> 63 else value for value in hashed]
    return Series(signed, index=index, dtype="int64").astype("uint64")


def hash_pandas_object(
    obj: Any,
    index: bool = True,
    encoding: str = "utf8",
    hash_key: str | None = _KEY,
    categorize: bool = True,
) -> Any:
    """A uint64 hash for each row of a series, a frame or an index, equal to pandas' hash.

    Raises:
        TypeError: When `obj` is not a series, a frame or an index.
        ValueError: When the key does not encode to 16 bytes.
    """
    from . import DataFrame, Index, Series
    from ._multi import MultiIndex

    if hash_key is None:
        hash_key = _KEY
    if isinstance(obj, MultiIndex):
        return _unsigned(_multi_hashes(obj, encoding, hash_key), None)
    if isinstance(obj, Index):
        hashed = _column_hashes(obj, encoding, hash_key, categorize)
        return _unsigned(hashed, obj)
    if isinstance(obj, Series):
        hashed = _column_hashes(obj, encoding, hash_key, categorize)
        if index:
            hashed = _combined(
                [hashed, _index_hashes(obj.index, encoding, hash_key, categorize)], 2
            )
        return _unsigned(hashed, obj.index)
    if isinstance(obj, DataFrame):
        arrays = [
            _column_hashes(obj.iloc[:, position], encoding, hash_key, categorize)
            for position in range(obj.shape[1])
        ]
        if index:
            arrays.append(_index_hashes(obj.index, encoding, hash_key, categorize))
        hashed = _combined(arrays, len(arrays))
        if len(hashed) != len(obj.index):
            raise InvalidArgumentError(
                f"Length of values ({len(hashed)}) does not match length of index"
                f" ({len(obj.index)})"
            )
        return _unsigned(hashed, obj.index)
    raise TypeError(f"Unexpected type for hashing {type(obj)}")


def hash_array(
    vals: Any, encoding: str = "utf8", hash_key: str = _KEY, categorize: bool = True
) -> Any:
    """A uint64 hash for each value of a numpy array or a firepanda array, as a numpy array.

    Raises:
        TypeError: When `vals` is not an array.
        ValueError: When the key does not encode to 16 bytes.
    """
    from ._array import FirepandaArray

    if not hasattr(vals, "dtype"):
        raise TypeError("must pass an ndarray-like")
    import numpy as np

    if isinstance(vals, FirepandaArray):
        hashed = _column_hashes(vals, encoding, hash_key, categorize)
    elif isinstance(vals, np.ndarray):
        values = vals.ravel().tolist()
        dtype = vals.dtype
        if dtype.kind in "mM":
            hashed = [_mixed(int(tick) & _MASK) for tick in vals.ravel().view("i8").tolist()]
        elif dtype.kind in "biufc":
            hashed = _values_hashes(str(dtype), values, encoding, hash_key, categorize)
        else:
            hashed = _values_hashes("object", values, encoding, hash_key, categorize)
        return np.array(hashed, dtype=np.uint64).reshape(vals.shape)
    else:
        raise TypeError(
            f"hash_array requires np.ndarray or ExtensionArray, not {type(vals).__name__}."
            " Use hash_pandas_object instead."
        )
    return np.array(hashed, dtype=np.uint64)
