"""`read_stata` and `StataReader`, a port of the reading half of pandas' `io/stata.py`.

A Stata file is a header, a table of fixed width records and a table of value
labels. pandas reads the records with numpy's structured types, and this
reads them with `struct`, which needs no numpy. Every step after that is
pandas' own, in pandas' order: the column selection, the text, the long
strings, the missing values, the dates, the value labels, the widening and
the index. Document 118 of the compat notes describes the design.
"""

from __future__ import annotations

import collections.abc
import datetime as dt
import io
import math
import struct
import warnings
from typing import Any, ClassVar

from .errors import CategoricalConversionWarning

_VERSION_ERROR = (
    "Version of given Stata file is {version}. pandas supports importing "
    "versions 102, 103, 104, 105, 108, 110 (Stata 7), 111 (Stata 7SE),  "
    "113 (Stata 8/9), 114 (Stata 10/11), 115 (Stata 12), 117 (Stata 13), "
    "118 (Stata 14/15/16), and 119 (Stata 15/16, over 32,767 variables)."
)

_DATE_FORMATS = ["%tc", "%tC", "%td", "%d", "%tw", "%tm", "%tq", "%th", "%ty"]

_STATA_EPOCH = dt.datetime(1960, 1, 1)

_CATEGORICAL_CONVERSION_WARNING = """
One or more series with value labels are not fully labeled. Reading this
dataset with an iterator results in categorical variable with different
categories. This occurs since it is not possible to know all possible values
until the entire dataset has been read. To avoid this warning, you can either
read dataset without an iterator, or manually convert categorical data by
``convert_categoricals`` to False and then accessing the variable labels
through the value_labels method of the reader.
"""


def _float32(raw: bytes) -> float:
    return struct.unpack("<f", raw)[0]


def _float64(raw: bytes) -> float:
    return struct.unpack("<d", raw)[0]


class StataMissingValue:
    """One of Stata's 27 missing values, `.` and `.a` to `.z`, as pandas hands it out."""

    MISSING_VALUES: ClassVar[dict[float, str]] = {}
    bases = (101, 32741, 2147483621)
    for _base in bases:
        MISSING_VALUES[_base] = "."
        for _i in range(1, 27):
            MISSING_VALUES[_i + _base] = "." + chr(96 + _i)
    _bits32 = struct.unpack("<i", b"\x00\x00\x00\x7f")[0]
    for _i in range(27):
        _key = _float32(struct.pack("<i", _bits32))
        MISSING_VALUES[_key] = "." + (chr(96 + _i) if _i else "")
        _bits32 += struct.unpack("<i", b"\x00\x08\x00\x00")[0]
    _bits64 = struct.unpack("<q", b"\x00\x00\x00\x00\x00\x00\xe0\x7f")[0]
    for _i in range(27):
        _key = _float64(struct.pack("<q", _bits64))
        MISSING_VALUES[_key] = "." + (chr(96 + _i) if _i else "")
        _bits64 += struct.unpack("<q", b"\x00\x00\x00\x00\x00\x01\x00\x00")[0]
    BASE_MISSING_VALUES: ClassVar[dict[str, float]] = {
        "int8": 101,
        "int16": 32741,
        "int32": 2147483621,
        "float32": _float32(b"\x00\x00\x00\x7f"),
        "float64": _float64(b"\x00\x00\x00\x00\x00\x00\xe0\x7f"),
    }
    del _base, _i, _key, _bits32, _bits64

    def __init__(self, value: float) -> None:
        self._value = value
        value = int(value) if value < 2147483648 else float(value)
        self._str = self.MISSING_VALUES[value]

    @property
    def string(self) -> str:
        """The missing value as Stata writes it, such as `.` or `.a`."""
        return self._str

    @property
    def value(self) -> float:
        """The number the file holds for it."""
        return self._value

    def __str__(self) -> str:
        return self.string

    def __repr__(self) -> str:
        return f"{type(self)}({self})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.string == other.string
            and self.value == other.value
        )

    def __hash__(self) -> int:
        return hash((self.string, self.value))

    @classmethod
    def get_base_missing_value(cls, dtype: Any) -> float:
        """The plain missing value `.` for a column of the named type.

        Raises:
            ValueError: For a type Stata has no missing value for.
        """
        name = str(dtype)
        if name not in cls.BASE_MISSING_VALUES:
            raise ValueError("Unsupported dtype")
        return cls.BASE_MISSING_VALUES[name]


StataMissingValue.__module__ = "firepanda.io.stata"

_TYPE_MAP: list[Any] = [*range(251), *"bhlfd"]
_TYPE_MAP_XML = {32768: "Q", 65526: "d", 65527: "f", 65528: "l", 65529: "h", 65530: "b"}
_OLD_TYPE_MAPPING = {98: 251, 105: 252, 108: 253, 102: 254, 100: 255}
_FLOAT_RANGE = {
    "f": (_float32(b"\xff\xff\xff\xfe"), _float32(b"\xff\xff\xff\x7e")),
    "d": (
        _float64(b"\xff\xff\xff\xff\xff\xff\xef\xff"),
        _float64(b"\xff\xff\xff\xff\xff\xff\xdf\x7f"),
    ),
}
_VALID_RANGE: dict[str, tuple[float, float]] = {
    "b": (-127, 100),
    "h": (-32767, 32740),
    "l": (-2147483647, 2147483620),
    **_FLOAT_RANGE,
}
_OLD_VALID_RANGE: dict[str, tuple[float, float]] = {
    "b": (-128, 126),
    "h": (-32768, 32766),
    "l": (-2147483648, 2147483646),
    **_FLOAT_RANGE,
}
_MISSING_VALUES: dict[str, float] = {
    "b": 101,
    "h": 32741,
    "l": 2147483621,
    "f": _float32(b"\x00\x00\x00\x7f"),
    "d": _float64(b"\x00\x00\x00\x00\x00\x00\xe0\x7f"),
}
_KINDS = {"b": "int8", "h": "int16", "l": "int32", "f": "float32", "d": "float64", "Q": "uint64"}
_WIDER = {"int8": "int64", "int16": "int64", "int32": "int64", "float32": "float64"}


def _source(path_or_buf: Any, compression: Any, storage_options: Any) -> io.BytesIO:
    """The whole file, unpacked, as a buffer that can seek."""
    from ._excel import _opened
    from ._pickle import _method, _unpacked

    method, _ = _method(path_or_buf, compression)
    handle, mine = _opened(path_or_buf, storage_options)
    try:
        data = handle.read()
    finally:
        if mine:
            handle.close()
    return io.BytesIO(_unpacked(data, method))


class _Column:
    """One column on its way to the frame: its values and the type it will have."""

    def __init__(self, values: list[Any], kind: str) -> None:
        self.values = values
        self.kind = kind


class StataReader(collections.abc.Iterator):
    """Reads a Stata dta file, whole or in chunks, as pandas' `StataReader` does.

    Use it as a context manager. The file is opened at the first read or the
    first question about it, and closed when the `with` block ends.
    """

    def __init__(
        self,
        path_or_buf: Any,
        convert_dates: bool = True,
        convert_categoricals: bool = True,
        index_col: str | None = None,
        convert_missing: bool = False,
        preserve_dtypes: bool = True,
        columns: Any = None,
        order_categoricals: bool = True,
        chunksize: int | None = None,
        compression: Any = "infer",
        storage_options: Any = None,
    ) -> None:
        self._convert_dates = convert_dates
        self._convert_categoricals = convert_categoricals
        self._index_col = index_col
        self._convert_missing = convert_missing
        self._preserve_dtypes = preserve_dtypes
        self._columns = columns
        self._order_categoricals = order_categoricals
        self._original_path_or_buf = path_or_buf
        self._compression = compression
        self._storage_options = storage_options
        self._encoding = ""
        self._chunksize = chunksize
        self._using_iterator = False
        self._entered = False
        if self._chunksize is None:
            self._chunksize = 1
        elif not isinstance(chunksize, int) or isinstance(chunksize, bool) or chunksize <= 0:
            raise ValueError("chunksize must be a positive integer when set.")
        self._column_selector_set = False
        self._value_label_dict: dict[str, dict[int, str]] = {}
        self._value_labels_read = False
        self._struct: struct.Struct | None = None
        self._lines_read = 0

    def _ensure_open(self) -> None:
        if not hasattr(self, "_path_or_buf"):
            self._open_file()

    def _open_file(self) -> None:
        if not self._entered:
            warnings.warn(
                "StataReader is being used without using a context manager. "
                "Using StataReader as a context manager is the only supported method.",
                ResourceWarning,
                stacklevel=3,
            )
        self._path_or_buf = _source(
            self._original_path_or_buf, self._compression, self._storage_options
        )
        self._read_header()
        self._setup_dtype()

    def __enter__(self) -> StataReader:
        self._entered = True
        return self

    def __exit__(self, *args: object) -> None:
        if hasattr(self, "_path_or_buf"):
            self._path_or_buf.close()

    def _set_encoding(self) -> None:
        self._encoding = "latin-1" if self._format_version < 118 else "utf-8"

    def _unpack(self, code: str, size: int) -> Any:
        return struct.unpack(f"{self._byteorder}{code}", self._path_or_buf.read(size))[0]

    def _read_int8(self) -> int:
        return struct.unpack("b", self._path_or_buf.read(1))[0]

    def _read_uint8(self) -> int:
        return struct.unpack("B", self._path_or_buf.read(1))[0]

    def _read_uint16(self) -> int:
        return self._unpack("H", 2)

    def _read_uint32(self) -> int:
        return self._unpack("I", 4)

    def _read_uint64(self) -> int:
        return self._unpack("Q", 8)

    def _read_int16(self) -> int:
        return self._unpack("h", 2)

    def _read_int32(self) -> int:
        return self._unpack("i", 4)

    def _read_int64(self) -> int:
        return self._unpack("q", 8)

    def _read_int16_count(self, count: int) -> tuple[int, ...]:
        return struct.unpack(f"{self._byteorder}{'h' * count}", self._path_or_buf.read(2 * count))

    def _read_header(self) -> None:
        first_char = self._path_or_buf.read(1)
        if first_char == b"<":
            self._read_new_header()
        else:
            self._read_old_header(first_char)

    def _read_new_header(self) -> None:
        buf = self._path_or_buf
        buf.read(27)  # stata_dta><header><release>
        self._format_version = int(buf.read(3))
        if self._format_version not in (117, 118, 119):
            raise ValueError(_VERSION_ERROR.format(version=self._format_version))
        self._set_encoding()
        buf.read(21)  # </release><byteorder>
        self._byteorder = ">" if buf.read(3) == b"MSF" else "<"
        buf.read(15)  # </byteorder><K>
        self._nvar = self._read_uint16() if self._format_version <= 118 else self._read_uint32()
        buf.read(7)  # </K><N>
        self._nobs = self._get_nobs()
        buf.read(11)  # </N><label>
        self._data_label = self._get_data_label()
        buf.read(19)  # </label><timestamp>
        self._time_stamp = self._get_time_stamp()
        buf.read(26)  # </timestamp></header><map>
        buf.read(8)  # 0x0000000000000000
        buf.read(8)  # position of <map>
        self._seek_vartypes = self._read_int64() + 16
        self._seek_varnames = self._read_int64() + 10
        self._seek_sortlist = self._read_int64() + 10
        self._seek_formats = self._read_int64() + 9
        self._seek_value_label_names = self._read_int64() + 19
        self._seek_variable_labels = self._get_seek_variable_labels()
        buf.read(8)  # <characteristics>
        self._data_location = self._read_int64() + 6
        self._seek_strls = self._read_int64() + 7
        self._seek_value_labels = self._read_int64() + 14
        self._typlist = self._get_dtypes(self._seek_vartypes)
        buf.seek(self._seek_varnames)
        self._varlist = self._get_varlist()
        buf.seek(self._seek_sortlist)
        self._srtlist = self._read_int16_count(self._nvar + 1)[:-1]
        buf.seek(self._seek_formats)
        self._fmtlist = self._get_fmtlist()
        buf.seek(self._seek_value_label_names)
        self._lbllist = self._get_lbllist()
        buf.seek(self._seek_variable_labels)
        self._variable_labels = self._get_variable_labels()

    def _get_dtypes(self, seek_vartypes: int) -> list[Any]:
        self._path_or_buf.seek(seek_vartypes)
        typlist: list[Any] = []
        for _ in range(self._nvar):
            typ = self._read_uint16()
            if typ <= 2045:
                typlist.append(typ)
            elif typ in _TYPE_MAP_XML:
                typlist.append(_TYPE_MAP_XML[typ])
            else:
                raise ValueError(f"cannot convert stata types [{typ}]")
        return typlist

    def _read_texts(self, width: int) -> list[str]:
        return [self._decode(self._path_or_buf.read(width)) for _ in range(self._nvar)]

    def _get_varlist(self) -> list[str]:
        return self._read_texts(33 if self._format_version < 118 else 129)

    def _get_fmtlist(self) -> list[str]:
        version = self._format_version
        width = 57 if version >= 118 else 49 if version > 113 else 12 if version > 104 else 7
        return self._read_texts(width)

    def _get_lbllist(self) -> list[str]:
        version = self._format_version
        return self._read_texts(129 if version >= 118 else 33 if version > 108 else 9)

    def _get_variable_labels(self) -> list[str]:
        version = self._format_version
        return self._read_texts(321 if version >= 118 else 81 if version > 105 else 32)

    def _get_nobs(self) -> int:
        if self._format_version >= 118:
            return self._read_uint64()
        if self._format_version >= 103:
            return self._read_uint32()
        return self._read_uint16()

    def _get_data_label(self) -> str:
        if self._format_version >= 118:
            return self._decode(self._path_or_buf.read(self._read_uint16()))
        if self._format_version == 117:
            return self._decode(self._path_or_buf.read(self._read_int8()))
        if self._format_version > 105:
            return self._decode(self._path_or_buf.read(81))
        return self._decode(self._path_or_buf.read(32))

    def _get_time_stamp(self) -> str:
        if self._format_version >= 118:
            return self._path_or_buf.read(self._read_int8()).decode("utf-8")
        if self._format_version == 117:
            return self._decode(self._path_or_buf.read(self._read_int8()))
        if self._format_version > 104:
            return self._decode(self._path_or_buf.read(18))
        raise ValueError

    def _get_seek_variable_labels(self) -> int:
        if self._format_version == 117:
            self._path_or_buf.read(8)  # <variable_labels>, thrown away
            return self._seek_value_label_names + (33 * self._nvar) + 20 + 17
        if self._format_version >= 118:
            return self._read_int64() + 17
        raise ValueError

    def _read_old_header(self, first_char: bytes) -> None:
        buf = self._path_or_buf
        self._format_version = first_char[0]
        if self._format_version not in (102, 103, 104, 105, 108, 110, 111, 113, 114, 115):
            raise ValueError(_VERSION_ERROR.format(version=self._format_version))
        self._set_encoding()
        self._byteorder = ">" if self._read_int8() == 0x1 else "<"
        self._filetype = self._read_int8()
        buf.read(1)  # unused
        self._nvar = self._read_uint16()
        self._nobs = self._get_nobs()
        self._data_label = self._get_data_label()
        if self._format_version >= 105:
            self._time_stamp = self._get_time_stamp()
        if self._format_version >= 111:
            typlist = list(buf.read(self._nvar))
        else:
            typlist = [_OLD_TYPE_MAPPING.get(tp, tp - 127) for tp in buf.read(self._nvar)]
        self._typlist = [_TYPE_MAP[typ] for typ in typlist]
        width = 33 if self._format_version > 108 else 9
        self._varlist = self._read_texts(width)
        self._srtlist = self._read_int16_count(self._nvar + 1)[:-1]
        self._fmtlist = self._get_fmtlist()
        self._lbllist = self._get_lbllist()
        self._variable_labels = self._get_variable_labels()
        if self._format_version > 104:
            while True:
                data_type = self._read_int8()
                data_len = self._read_int32() if self._format_version > 108 else self._read_int16()
                if data_type == 0:
                    break
                buf.read(data_len)
        self._data_location = buf.tell()

    def _setup_dtype(self) -> struct.Struct:
        if self._struct is None:
            self._record_types = list(self._typlist)
            codes = [typ if isinstance(typ, str) else f"{typ}s" for typ in self._typlist]
            self._struct = struct.Struct(self._byteorder + "".join(codes))
        return self._struct

    def _decode(self, s: bytes) -> str:
        s = s.partition(b"\0")[0]
        try:
            return s.decode(self._encoding)
        except UnicodeDecodeError:
            encoding = self._encoding
            msg = f"""
One or more strings in the dta file could not be decoded using {encoding}, and
so the fallback encoding of latin-1 is being used.  This can happen when a file
has been incorrectly encoded by Stata or some other software. You should verify
the string values returned are correct."""
            warnings.warn(msg, UnicodeWarning, stacklevel=4)
            return s.decode("latin-1")

    def _read_new_value_labels(self) -> None:
        buf = self._path_or_buf
        if self._format_version >= 117:
            buf.seek(self._seek_value_labels)
        else:
            buf.seek(self._data_location + self._nobs * self._setup_dtype().size)
        while True:
            if self._format_version >= 117 and buf.read(5) == b"</val":  # <lbl>
                break  # the end of the value label table
            slength = buf.read(4)
            if not slength:
                break  # the end of the value label table before 117, or of the file
            if self._format_version == 108:
                labname = self._decode(buf.read(9))
            elif self._format_version <= 117:
                labname = self._decode(buf.read(33))
            else:
                labname = self._decode(buf.read(129))
            buf.read(3)  # padding
            n = self._read_uint32()
            txtlen = self._read_uint32()
            off = struct.unpack(f"{self._byteorder}{n}i", buf.read(4 * n))
            val = struct.unpack(f"{self._byteorder}{n}i", buf.read(4 * n))
            order = sorted(range(n), key=lambda i: off[i])
            off = tuple(off[i] for i in order)
            val = tuple(val[i] for i in order)
            txt = buf.read(txtlen)
            labels: dict[int, str] = {}
            for i in range(n):
                end = off[i + 1] if i < n - 1 else txtlen
                labels[val[i]] = self._decode(txt[off[i] : end])
            self._value_label_dict[labname] = labels
            if self._format_version >= 117:
                buf.read(6)  # </lbl>

    def _read_old_value_labels(self) -> None:
        buf = self._path_or_buf
        buf.seek(self._data_location + self._nobs * self._setup_dtype().size)
        while True:
            if not buf.read(2):
                break
            buf.seek(-2, io.SEEK_CUR)
            n = self._read_uint16()
            labname = self._decode(buf.read(9))
            buf.read(1)  # padding
            codes = struct.unpack(f"{self._byteorder}{n}h", buf.read(2 * n))
            self._value_label_dict[labname] = {code: self._decode(buf.read(8)) for code in codes}

    def _read_value_labels(self) -> None:
        self._ensure_open()
        if self._value_labels_read:
            return
        if self._format_version >= 108:
            self._read_new_value_labels()
        else:
            self._read_old_value_labels()
        self._value_labels_read = True

    def _read_strls(self) -> None:
        buf = self._path_or_buf
        buf.seek(self._seek_strls)
        self.GSO = {"0": ""}
        while True:
            if buf.read(3) != b"GSO":
                break
            if self._format_version == 117:
                v_o = self._read_uint64()
            else:
                raw = buf.read(12)
                v_size = 2 if self._format_version == 118 else 3
                if self._byteorder == "<":
                    raw = raw[0:v_size] + raw[4 : (12 - v_size)]
                else:
                    raw = raw[4 - v_size : 4] + raw[(4 + v_size) :]
                v_o = struct.unpack(f"{self._byteorder}Q", raw)[0]
            typ = self._read_uint8()
            length = self._read_uint32()
            va = buf.read(length)
            self.GSO[str(v_o)] = va[0:-1].decode(self._encoding) if typ == 130 else str(va)

    def __next__(self) -> Any:
        self._using_iterator = True
        return self.read(nrows=self._chunksize)

    def get_chunk(self, size: int | None = None) -> Any:
        """The next `size` rows, or the next chunk when no size is given."""
        if size is None:
            size = self._chunksize
        return self.read(nrows=size)

    def read(
        self,
        nrows: int | None = None,
        convert_dates: bool | None = None,
        convert_categoricals: bool | None = None,
        index_col: str | None = None,
        convert_missing: bool | None = None,
        preserve_dtypes: bool | None = None,
        columns: Any = None,
        order_categoricals: bool | None = None,
    ) -> Any:
        """The next `nrows` rows as a frame, every row left when `nrows` is None.

        An argument left as None takes the value the reader was made with.

        Raises:
            StopIteration: When every row has been read.
            ValueError: For columns that repeat or are not in the file, or value
                labels that repeat, in pandas' words.
        """
        from ._frame import DataFrame

        self._ensure_open()
        if convert_dates is None:
            convert_dates = self._convert_dates
        if convert_categoricals is None:
            convert_categoricals = self._convert_categoricals
        if convert_missing is None:
            convert_missing = self._convert_missing
        if preserve_dtypes is None:
            preserve_dtypes = self._preserve_dtypes
        if columns is None:
            columns = self._columns
        if order_categoricals is None:
            order_categoricals = self._order_categoricals
        if index_col is None:
            index_col = self._index_col
        if nrows is None:
            nrows = self._nobs
        if self._nobs == 0 and nrows == 0:
            return self._empty(columns)
        if self._format_version >= 117 and not self._value_labels_read:
            self._read_strls()
        record = self._setup_dtype()
        max_read_len = (self._nobs - self._lines_read) * record.size
        read_len = min(nrows * record.size, max_read_len)
        if read_len <= 0:
            if convert_categoricals:
                self._read_value_labels()
            raise StopIteration
        self._path_or_buf.seek(self._data_location + self._lines_read * record.size)
        read_lines = min(nrows, self._nobs - self._lines_read)
        rows = list(record.iter_unpack(self._path_or_buf.read(read_lines * record.size)))
        self._lines_read += read_lines
        if convert_categoricals:
            self._read_value_labels()
        names = list(self._varlist)
        data = {
            name: _Column(list(values), self._kind(typ))
            for name, values, typ in zip(
                names, zip(*rows, strict=True), self._record_types, strict=True
            )
        }
        if columns is not None:
            data = self._do_select_columns(data, columns)
        for column, typ in zip(data.values(), self._typlist, strict=False):
            if isinstance(typ, int):
                column.values = [self._decode(v) for v in column.values]
        self._insert_strls(data)
        self._do_convert_missing(data, convert_missing)
        if convert_dates:
            for column, fmt in zip(data.values(), self._fmtlist, strict=False):
                if any(fmt.startswith(date_fmt) for date_fmt in _DATE_FORMATS):
                    _elapsed_dates(column, fmt)
        if convert_categoricals:
            self._do_convert_categoricals(
                data, self._value_label_dict, self._lbllist, order_categoricals
            )
        if not preserve_dtypes:
            for column in data.values():
                column.kind = _WIDER.get(column.kind, column.kind)
        start = self._lines_read - read_lines
        frame = DataFrame({name: _series(column) for name, column in data.items()})
        if index_col is None:
            from ._range_index import RangeIndex

            frame = frame.set_axis(RangeIndex(start, self._lines_read))
        else:
            frame = frame.set_index(index_col)
            # pandas builds the labels from the column's values, so a narrow integer is int64.
            if str(frame.index.dtype) in ("int8", "int16", "int32"):
                frame.index = frame.index.astype("int64")
        return frame

    def _kind(self, typ: Any) -> str:
        return "str" if isinstance(typ, int) else _KINDS[typ]

    def _empty(self, columns: Any) -> Any:
        """A frame with the file's columns and no rows, typed as pandas types it."""
        from ._frame import DataFrame, Series

        data = {}
        for name, typ in zip(self._varlist, self._typlist, strict=True):
            kind = "object" if isinstance(typ, int) else _KINDS[typ]
            if kind == "uint64" and self._format_version >= 117:
                kind = "uint8"
            data[name] = Series([], dtype=kind)
        frame = DataFrame(data)
        if columns is not None:
            chosen = self._do_select_columns(dict.fromkeys(self._varlist), columns)
            frame = frame[list(chosen)]
        return frame

    def _do_convert_missing(self, data: dict[str, _Column], convert_missing: bool) -> None:
        old_missingdouble = float.fromhex("0x1.0p333")
        for column, fmt in zip(data.values(), self._typlist, strict=False):
            if self._format_version <= 105 and fmt == "d":
                column.values = [
                    _MISSING_VALUES["d"] if v == old_missingdouble else v for v in column.values
                ]
            ranges = _OLD_VALID_RANGE if self._format_version <= 111 else _VALID_RANGE
            if fmt not in ranges:
                continue
            nmin, nmax = ranges[fmt]
            missing = [v < nmin or v > nmax for v in column.values]
            if not any(missing):
                continue
            if convert_missing:
                replaced = []
                for value, gap in zip(column.values, missing, strict=True):
                    if not gap:
                        replaced.append(value)
                    elif self._format_version <= 111:
                        replaced.append(StataMissingValue(float(_MISSING_VALUES[fmt])))
                    else:
                        replaced.append(StataMissingValue(value))
                column.values = replaced
                column.kind = "object"
            else:
                column.values = [
                    math.nan if gap else value
                    for value, gap in zip(column.values, missing, strict=True)
                ]
                if column.kind not in ("float32", "float64"):
                    column.kind = "float64"

    def _insert_strls(self, data: dict[str, _Column]) -> None:
        if not hasattr(self, "GSO") or len(self.GSO) == 0:
            return
        for column, typ in zip(data.values(), self._typlist, strict=False):
            if typ == "Q":
                column.values = [self.GSO[str(k)] for k in column.values]
                column.kind = "str"

    def _do_select_columns(self, data: dict[str, Any], columns: Any) -> dict[str, Any]:
        if not self._column_selector_set:
            column_set = set(columns)
            if len(column_set) != len(columns):
                raise ValueError("columns contains duplicate entries")
            unmatched = column_set.difference(data)
            if unmatched:
                joined = ", ".join(list(unmatched))
                raise ValueError(
                    f"The following columns were not found in the Stata data set: {joined}"
                )
            at = [self._varlist.index(col) for col in columns]
            self._typlist = [self._typlist[i] for i in at]
            self._fmtlist = [self._fmtlist[i] for i in at]
            self._lbllist = [self._lbllist[i] for i in at]
            self._column_selector_set = True
        return {col: data[col] for col in columns}

    def _do_convert_categoricals(
        self,
        data: dict[str, _Column],
        value_label_dict: dict[str, dict[int, str]],
        lbllist: list[str],
        order_categoricals: bool,
    ) -> None:
        if not value_label_dict:
            return
        for (col, column), label in zip(data.items(), lbllist, strict=False):
            if label not in value_label_dict:
                continue
            vl = value_label_dict[label]
            keys = list(vl)
            present = [v for v in column.values if not _is_gap(v)]
            if (
                self._using_iterator
                and len(present) == len(column.values)
                and set(present) <= set(keys)
            ):
                categories = keys
                names = list(vl.values())
            else:
                if self._using_iterator:
                    warnings.warn(
                        _CATEGORICAL_CONVERSION_WARNING, CategoricalConversionWarning, stacklevel=3
                    )
                categories = sorted(set(present))
                names = [vl.get(category, category) for category in categories]
            if len(set(names)) != len(names):
                counts = collections.Counter(names)
                repeated = sorted(
                    (name for name in counts if counts[name] > 1), key=lambda name: -counts[name]
                )
                repeats = "-" * 80 + "\n" + "\n".join(str(name) for name in repeated)
                msg = f"""
Value labels for column {col} are not unique. These cannot be converted to
pandas categoricals.

Either read the file with `convert_categoricals` set to False or use the
low level interface in `StataReader` to separately read the values and the
value_labels.

The repeated labels are:
{repeats}
"""
                raise ValueError(msg)
            where = {category: code for code, category in enumerate(categories)}
            codes = [-1 if _is_gap(v) else where[v] for v in column.values]
            column.values = [codes, names, order_categoricals]
            column.kind = "category"

    @property
    def data_label(self) -> str:
        """The label of the data set."""
        self._ensure_open()
        return self._data_label

    @property
    def time_stamp(self) -> str:
        """When the file was written."""
        self._ensure_open()
        return self._time_stamp

    def variable_labels(self) -> dict[str, str]:
        """Each column's label, by the column's name."""
        self._ensure_open()
        return dict(zip(self._varlist, self._variable_labels, strict=True))

    def value_labels(self) -> dict[str, dict[int, str]]:
        """Each value label set by its name, as a mapping from number to label."""
        if not self._value_labels_read:
            self._read_value_labels()
        return self._value_label_dict


StataReader.__module__ = "firepanda.io.stata"


def _is_gap(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _series(column: _Column) -> Any:
    """A column as the series pandas would hold it."""
    from ._categorical import Categorical
    from ._frame import Series

    kind = column.kind
    if kind == "category":
        codes, names, ordered = column.values
        return Series(Categorical.from_codes(codes, names, ordered=ordered))
    if kind == "str":
        return Series(column.values, dtype="str")
    if kind.startswith("datetime64"):
        return Series(column.values).astype(kind)
    return Series(column.values, dtype=kind)


def _moment(value: Any, make: Any) -> Any:
    return None if _is_gap(value) else make(value)


def _months_on(months: int) -> dt.datetime:
    year, month = divmod(months, 12)
    return dt.datetime(1970 + year, month + 1, 1)


def _elapsed_dates(column: _Column, fmt: str) -> None:
    """Stata's elapsed dates as moments, as `_stata_elapsed_date_to_datetime_vec` reads them."""
    from ._scalars import NaT

    values = column.values
    if fmt.startswith(("%tc", "tc")):
        column.values = [
            _moment(v, lambda v: _STATA_EPOCH + dt.timedelta(milliseconds=int(v))) for v in values
        ]
        column.kind = "datetime64[ms]"
        return
    if fmt.startswith(("%td", "td", "%d", "d")):
        column.values = [
            _moment(v, lambda v: _STATA_EPOCH + dt.timedelta(days=int(v))) for v in values
        ]
        column.kind = "datetime64[s]"
        return
    steps = {"tm": (-120, 1), "tq": (-40, 3), "th": (-20, 6)}
    for name, (shift, months) in steps.items():
        if fmt.startswith((f"%{name}", name)):
            column.values = [
                _moment(v, lambda v, s=shift, m=months: _months_on((int(v + s)) * m))
                for v in values
            ]
            column.kind = "datetime64[s]"
            return
    if fmt.startswith(("%ty", "ty")):
        column.values = [_moment(v, lambda v: dt.datetime(int(v), 1, 1)) for v in values]
        column.kind = "datetime64[s]"
        return
    bad = [_is_gap(v) for v in values]
    whole = [1 if gap else int(v) for v, gap in zip(values, bad, strict=True)]
    if fmt.startswith(("%tC", "tC")):
        warnings.warn("Encountered %tC format. Leaving in Stata Internal Format.", stacklevel=4)
        column.values = [NaT if gap else v for v, gap in zip(whole, bad, strict=True)]
        column.kind = "object"
        return
    if fmt.startswith(("%tw", "tw")):
        moments = []
        for v, gap in zip(whole, bad, strict=True):
            year = _STATA_EPOCH.year + v // 52
            moments.append(
                None if gap else dt.datetime(year, 1, 1) + dt.timedelta(days=(v % 52) * 7)
            )
        column.values = moments
        column.kind = "datetime64[s]"
        return
    raise ValueError(f"Date fmt {fmt} not understood")


def read_stata(
    filepath_or_buffer: Any,
    *,
    convert_dates: bool = True,
    convert_categoricals: bool = True,
    index_col: str | None = None,
    convert_missing: bool = False,
    preserve_dtypes: bool = True,
    columns: Any = None,
    order_categoricals: bool = True,
    chunksize: int | None = None,
    iterator: bool = False,
    compression: Any = "infer",
    storage_options: Any = None,
) -> Any:
    """Reads a Stata dta file into a frame, or hands back a reader for chunks.

    Returns:
        The frame, or a `StataReader` when `iterator` is true or `chunksize` is
        given.
    """
    reader = StataReader(
        filepath_or_buffer,
        convert_dates=convert_dates,
        convert_categoricals=convert_categoricals,
        index_col=index_col,
        convert_missing=convert_missing,
        preserve_dtypes=preserve_dtypes,
        columns=columns,
        order_categoricals=order_categoricals,
        chunksize=chunksize,
        storage_options=storage_options,
        compression=compression,
    )
    if iterator or chunksize:
        return reader
    with reader:
        return reader.read()
