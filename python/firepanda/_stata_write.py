"""`DataFrame.to_stata` and the Stata writers, a port of the writing half of pandas' `io/stata.py`.

pandas prepares the frame in numpy and writes it as a record array. This
works on plain lists, one per column, and packs each row with `struct`, so
the bytes on disk are pandas' bytes. Each step keeps pandas' order, warnings
and error sentences. Document 119 of the compat notes describes the design.
"""

from __future__ import annotations

import datetime as dt
import io
import math
import struct
import sys
import warnings
from typing import Any

from ._stata import _MISSING_VALUES, StataMissingValue
from .errors import InvalidColumnName, PossiblePrecisionLoss, ValueLabelTypeMismatch

_EXCESSIVE_STRING_LENGTH_ERROR = """
Fixed width strings in Stata .dta files are limited to 244 (or fewer)
characters.  Column '{0}' does not satisfy this restriction. Use the
'version=117' parameter to write the newer (Stata 13 and later) format.
"""

_PRECISION_LOSS_DOC = """
Column converted from {0} to {1}, and some data are outside of the lossless
conversion range. This may result in a loss of precision in the saved data.
"""

_VALUE_LABEL_MISMATCH_DOC = """
Stata value labels (pandas categories) must be strings. Column {0} contains
non-string labels which will be converted to strings.  Please check that the
Stata data file created has not lost information due to duplicate labels.
"""

_INVALID_NAME_DOC = """
Not all pandas column names were valid Stata variable names.
The following replacements have been made:

    {0}

If this is not what you expect, please make sure you have Stata-compliant
column names in your DataFrame (strings only, max 32 characters, only
alphanumerics and underscores, no Stata reserved words)
"""

_RESERVED_TEXT = """
aggregate array boolean break byte case catch class colvector complex const continue
default delegate delete do double else eltypedef end enum explicit export external float
for friend function global goto if inline int local long NULL pragma protected quad
rowvector short typedef typename virtual _all _N _skip _b _pi str# in _pred strL _coef _rc
using _cons _se with _n
"""
_RESERVED_WORDS = set(_RESERVED_TEXT.split())

_DATE_FORMATS = {"tc", "%tc", "td", "%td", "tw", "%tw", "tm", "%tm"}
_DATE_FORMATS |= {"tq", "%tq", "th", "%th", "ty", "%ty"}
_STATA_EPOCH = dt.datetime(1960, 1, 1)
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_FLOAT32_MAX = struct.unpack("<f", b"\xff\xff\xff\x7e")[0]
_FLOAT64_MAX = struct.unpack("<d", b"\xff\xff\xff\xff\xff\xff\xdf\x7f")[0]
_INT_RANGES = {
    "int8": (-128, 127),
    "int16": (-32768, 32767),
    "int32": (-2147483648, 2147483647),
    "int64": (-(2**63), 2**63 - 1),
}
_CODES = {"int8": "b", "int16": "h", "int32": "l", "float32": "f", "float64": "d"}
_NULLABLE = {
    "Int8": "int8",
    "Int16": "int16",
    "Int32": "int32",
    "Int64": "int64",
    "UInt8": "uint8",
    "UInt16": "uint16",
    "UInt32": "uint32",
    "UInt64": "uint64",
    "boolean": "bool",
}
_NUMERIC = {"int8", "int16", "int32", "int64", "float32", "float64"}


def _is_missing(value: Any) -> bool:
    from ._na import NA
    from ._scalars import NaT

    if value is None or value is NaT or value is NA:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    try:
        return bool(value != value)
    except Exception:
        return False


def _set_endianness(endianness: str) -> str:
    if endianness.lower() in ["<", "little"]:
        return "<"
    if endianness.lower() in [">", "big"]:
        return ">"
    raise ValueError(f"Endianness {endianness} not understood")


def _pad_bytes(name: Any, length: int) -> Any:
    if isinstance(name, bytes):
        return name + b"\x00" * (length - len(name))
    return name + "\x00" * (length - len(name))


def _pad_bytes_new(name: str | bytes, length: int) -> bytes:
    if isinstance(name, str):
        name = bytes(name, "utf-8")
    return name + b"\x00" * (length - len(name))


def _max_len(values: list[Any]) -> int:
    """The longest text in a column, as `max_len_string_array` measures it."""
    return max((len(v) for v in values if isinstance(v, (str, bytes))), default=0)


def _stamp(time_stamp: Any) -> str:
    if time_stamp is None:
        time_stamp = dt.datetime.now()
    elif not isinstance(time_stamp, dt.datetime):
        raise ValueError("time_stamp should be datetime type")
    return (
        time_stamp.strftime("%d ")
        + _MONTHS[time_stamp.month - 1]
        + time_stamp.strftime(" %Y %H:%M")
    )


class _Column:
    """One column on its way to the file: its values and the type pandas would give it."""

    def __init__(self, values: list[Any], kind: str, nullable: bool = False) -> None:
        self.values = values
        self.kind = kind
        self.nullable = nullable
        self.categories: list[Any] | None = None


def _columns_of(data: Any) -> tuple[list[Any], list[_Column]]:
    """The labels and columns of a frame, typed the way `_cast_to_stata_types` starts from."""
    names = list(data.columns)
    columns = []
    for i in range(len(names)):
        series = data.iloc[:, i]
        kind = str(series.dtype)
        if kind == "category":
            column = _Column(series.cat.codes.tolist(), "category")
            column.categories = series.cat.categories.tolist()
        elif kind in ("str", "string", "object"):
            values = series.tolist()
            column = _Column([None if _is_missing(v) else v for v in values], "object")
        elif kind in _NULLABLE:
            column = _Column(series.tolist(), _NULLABLE[kind], True)
        elif kind in ("Float32", "Float64"):
            values = [math.nan if _is_missing(v) else v for v in series.tolist()]
            column = _Column(values, kind.lower())
        else:
            column = _Column(series.tolist(), kind)
        columns.append(column)
    return names, columns


class StataValueLabel:
    """The value labels of one categorical column, packed as Stata stores them."""

    def __init__(self, catarray: Any, encoding: str = "latin-1") -> None:
        if encoding not in ("latin-1", "utf-8"):
            raise ValueError("Only latin-1 and utf-8 are supported.")
        self.labname = catarray.name
        self._encoding = encoding
        categories = catarray.cat.categories
        self.value_labels = list(enumerate(categories))
        self._prepare_value_labels()

    def _prepare_value_labels(self) -> None:
        self.text_len = 0
        self.txt: list[bytes] = []
        self.n = 0
        offsets: list[int] = []
        values: list[int] = []
        for vl in self.value_labels:
            category: Any = vl[1]
            if not isinstance(category, str):
                category = str(category)
                warnings.warn(
                    _VALUE_LABEL_MISMATCH_DOC.format(self.labname),
                    ValueLabelTypeMismatch,
                    stacklevel=4,
                )
            category = category.encode(self._encoding)
            offsets.append(self.text_len)
            self.text_len += len(category) + 1
            values.append(int(vl[0]))
            self.txt.append(category)
            self.n += 1
        self.off = offsets
        self.val = values
        self.len = 4 + 4 + 4 * self.n + 4 * self.n + self.text_len

    def generate_value_label(self, byteorder: str) -> bytes:
        """The label table in the file's byte order."""
        encoding = self._encoding
        bio = io.BytesIO()
        bio.write(struct.pack(byteorder + "i", self.len))
        labname = str(self.labname)[:32].encode(encoding)
        lab_len = 32 if encoding not in ("utf-8", "utf8") else 128
        bio.write(_pad_bytes(labname, lab_len + 1))
        bio.write(b"\x00" * 3)
        bio.write(struct.pack(byteorder + "i", self.n))
        bio.write(struct.pack(byteorder + "i", self.text_len))
        for offset in self.off:
            bio.write(struct.pack(byteorder + "i", offset))
        for value in self.val:
            bio.write(struct.pack(byteorder + "i", value))
        for text in self.txt:
            bio.write(text + b"\x00")
        return bio.getvalue()


class StataNonCatValueLabel(StataValueLabel):
    """Value labels given for a numeric column, as `value_labels` in `to_stata`."""

    def __init__(
        self, labname: str, value_labels: dict[float, str], encoding: str = "latin-1"
    ) -> None:
        if encoding not in ("latin-1", "utf-8"):
            raise ValueError("Only latin-1 and utf-8 are supported.")
        self.labname = labname
        self._encoding = encoding
        self.value_labels = sorted(value_labels.items(), key=lambda x: x[0])
        self._prepare_value_labels()


StataValueLabel.__module__ = "firepanda.io.stata"
StataNonCatValueLabel.__module__ = "firepanda.io.stata"

def _cast_to_stata_types(names: list[str], columns: list[_Column]) -> None:
    """Narrows or widens each column to a type Stata has, as pandas does."""
    ws = ""
    conversion_data = (
        ("bool", "int8", "int8"),
        ("uint8", "int8", "int16"),
        ("uint16", "int16", "int32"),
        ("uint32", "int32", "int64"),
        ("uint64", "int64", "float64"),
    )
    for col, column in zip(names, columns, strict=True):
        orig_missing = [_is_missing(v) for v in column.values] if column.nullable else []
        if column.nullable:
            column.values = [
                0 if gap else int(v) for v, gap in zip(column.values, orig_missing, strict=True)
            ]
        kind = column.kind
        values = column.values
        empty_df = len(values) == 0
        for c_data in conversion_data:
            if kind == c_data[0]:
                top = max(values) if values else 0
                fits = empty_df or top <= _INT_RANGES[c_data[1]][1]
                kind = c_data[1] if fits else c_data[2]
                if c_data[2] == "int64" and top >= 2**53:
                    ws = _PRECISION_LOSS_DOC.format("uint64", "float64")
                values = [float(v) if kind == "float64" else int(v) for v in values]
        if kind == "int8" and not empty_df:
            if max(values) > 100 or min(values) < -127:
                kind = "int16"
        elif kind == "int16" and not empty_df:
            if max(values) > 32740 or min(values) < -32767:
                kind = "int32"
        elif kind == "int64":
            if empty_df or (max(values) <= 2147483620 and min(values) >= -2147483647):
                kind = "int32"
            else:
                kind = "float64"
                values = [float(v) for v in values]
                if max(values) >= 2**53 or min(values) <= -(2**53):
                    ws = _PRECISION_LOSS_DOC.format("int64", "float64")
        elif kind in ("float32", "float64"):
            if any(math.isinf(v) for v in values):
                raise ValueError(
                    f"Column {col} contains infinity or -infinity"
                    "which is outside the range supported by Stata."
                )
            present = [v for v in values if not math.isnan(v)]
            value = max(present) if present else math.nan
            if kind == "float32" and value > _FLOAT32_MAX:
                kind = "float64"
            elif kind == "float64" and value > _FLOAT64_MAX:
                raise ValueError(
                    f"Column {col} has a maximum value ({value}) outside the range "
                    f"supported by Stata ({_FLOAT64_MAX})"
                )
        column.kind = kind
        column.values = values
        if column.nullable and any(orig_missing):
            sentinel = StataMissingValue.BASE_MISSING_VALUES[kind]
            column.values = [
                sentinel if gap else v for v, gap in zip(column.values, orig_missing, strict=True)
            ]
    if ws:
        warnings.warn(ws, PossiblePrecisionLoss, stacklevel=5)


def _elapsed(values: list[Any], fmt: str) -> list[float]:
    """Moments as Stata's elapsed dates, as `_datetime_to_stata_elapsed_vec` makes them."""
    moments = []
    for value in values:
        if _is_missing(value):
            moments.append(None)
        elif isinstance(value, dt.datetime) and value.tzinfo is None:
            moments.append(value)
        else:
            raise ValueError(
                "Columns containing dates must contain either datetime64, datetime or null values."
            )
    out = []
    for moment in moments:
        if moment is None:
            out.append(_MISSING_VALUES["d"])
            continue
        plain = dt.datetime(
            moment.year,
            moment.month,
            moment.day,
            moment.hour,
            moment.minute,
            moment.second,
            moment.microsecond,
        )
        delta = plain - _STATA_EPOCH
        ms = (delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds) // 1000
        if fmt in ("%tc", "tc"):
            out.append(float(ms))
        elif fmt in ("%td", "td"):
            out.append(float(ms // 86_400_000))
        elif fmt in ("%tw", "tw"):
            days = (plain - dt.datetime(plain.year, 1, 1)).days
            out.append(float(52 * (plain.year - 1960) + days // 7))
        elif fmt in ("%tm", "tm"):
            out.append(float(12 * (plain.year - 1960) + plain.month - 1))
        elif fmt in ("%tq", "tq"):
            out.append(float(4 * (plain.year - 1960) + (plain.month - 1) // 3))
        elif fmt in ("%th", "th"):
            out.append(float(2 * (plain.year - 1960) + int(plain.month > 6)))
        elif fmt in ("%ty", "ty"):
            out.append(float(plain.year))
        else:
            raise ValueError(f"Format {fmt} is not a known Stata date format")
    return out


def _codes_kind(count: int) -> str:
    """The type pandas gives the codes of a categorical with this many categories."""
    if count < 127:
        return "int8"
    if count < 32767:
        return "int16"
    if count < 2147483647:
        return "int32"
    return "int64"


class StataWriter:
    """Writes a frame as a Stata 114 file, the format pandas writes by default."""

    _max_string_length = 244
    _encoding = "latin-1"
    _dta_version = 114

    def __init__(
        self,
        fname: Any,
        data: Any,
        convert_dates: dict[Any, str] | None = None,
        write_index: bool = True,
        byteorder: str | None = None,
        time_stamp: dt.datetime | None = None,
        data_label: str | None = None,
        variable_labels: dict[Any, str] | None = None,
        compression: Any = "infer",
        storage_options: Any = None,
        *,
        value_labels: dict[Any, dict[float, str]] | None = None,
    ) -> None:
        self.data = data
        self._convert_dates: dict[Any, str] = {} if convert_dates is None else dict(convert_dates)
        self._write_index = write_index
        self._time_stamp = time_stamp
        self._data_label = data_label
        self._variable_labels = variable_labels
        self._non_cat_value_labels = value_labels
        self._value_labels: list[StataValueLabel] = []
        self._compression = compression
        self._converted_names: dict[Any, str] = {}
        self._prepare_pandas(data)
        self.storage_options = storage_options
        if byteorder is None:
            byteorder = sys.byteorder
        self._byteorder = _set_endianness(byteorder)
        self._fname = fname

    def _write(self, to_write: str) -> None:
        self._handle.write(to_write.encode(self._encoding))

    def _write_bytes(self, value: bytes) -> None:
        self._handle.write(value)

    def _prepare_non_cat_value_labels(self) -> list[StataNonCatValueLabel]:
        non_cat_value_labels: list[StataNonCatValueLabel] = []
        if self._non_cat_value_labels is None:
            return non_cat_value_labels
        for labname, labels in self._non_cat_value_labels.items():
            if labname in self._converted_names:
                colname = self._converted_names[labname]
            elif labname in self._names:
                colname = str(labname)
            else:
                raise KeyError(
                    f"Can't create value labels for {labname}, it wasn't found in the dataset."
                )
            if self._columns[self._names.index(colname)].kind not in _NUMERIC:
                raise ValueError(
                    f"Can't create value labels for {labname}, value labels "
                    "can only be applied to numeric columns."
                )
            non_cat_value_labels.append(StataNonCatValueLabel(colname, labels, self._encoding))
        return non_cat_value_labels

    def _prepare_categoricals(self) -> None:
        get_base_missing_value = StataMissingValue.get_base_missing_value
        for i, (col, column) in enumerate(zip(self._names, self._columns, strict=True)):
            if column.kind != "category":
                continue
            self._has_value_labels[i] = True
            categories = column.categories or []
            self._value_labels.append(_CategoryLabels(col, categories, self._encoding))
            kind = _codes_kind(len(categories))
            if kind == "int64":
                raise ValueError(
                    "It is not possible to export int64-based categorical data to Stata."
                )
            if not column.values:
                raise ValueError(
                    "zero-size array to reduction operation maximum which has no identity"
                )
            if max(column.values) >= get_base_missing_value(kind):
                kind = {"int8": "int16", "int16": "int32"}.get(kind, "float64")
            missing = get_base_missing_value(kind)
            column.values = [missing if v == -1 else v for v in column.values]
            if kind == "float64":
                column.values = [float(v) for v in column.values]
            column.kind = kind

    def _replace_nans(self) -> None:
        for column in self._columns:
            if column.kind in ("float32", "float64"):
                replacement = _MISSING_VALUES["f" if column.kind == "float32" else "d"]
                column.values = [replacement if _is_missing(v) else v for v in column.values]

    def _update_strl_names(self) -> None:
        pass

    def _validate_variable_name(self, name: str) -> str:
        for c in name:
            if (c < "A" or c > "Z") and (c < "a" or c > "z") and (c < "0" or c > "9") and c != "_":
                name = name.replace(c, "_")
        return name

    def _check_column_names(self, names: list[Any]) -> list[str]:
        converted_names: dict[Any, str] = {}
        columns = list(names)
        original_columns = columns[:]
        duplicate_var_id = 0
        for j, name in enumerate(columns):
            orig_name = name
            if not isinstance(name, str):
                name = str(name)
            name = self._validate_variable_name(name)
            if name in _RESERVED_WORDS:
                name = "_" + name
            if "0" <= name[0] <= "9":
                name = "_" + name
            name = name[: min(len(name), 32)]
            if name != orig_name:
                while columns.count(name) > 0:
                    name = "_" + str(duplicate_var_id) + name
                    name = name[: min(len(name), 32)]
                    duplicate_var_id += 1
                converted_names[orig_name] = name
            columns[j] = name
        if self._convert_dates:
            for c, o in zip(columns, original_columns, strict=True):
                if c != o:
                    self._convert_dates[c] = self._convert_dates[o]
                    del self._convert_dates[o]
        if converted_names:
            conversion_warning = [f"{orig}   ->   {name}" for orig, name in converted_names.items()]
            ws = _INVALID_NAME_DOC.format("\n    ".join(conversion_warning))
            warnings.warn(ws, InvalidColumnName, stacklevel=5)
        self._converted_names = converted_names
        self._update_strl_names()
        return columns

    def _prepare_pandas(self, data: Any) -> None:
        if self._write_index:
            data = data.reset_index()
        names, self._columns = _columns_of(data)
        self._names = self._check_column_names(names)
        _cast_to_stata_types(self._names, self._columns)
        self._replace_nans()
        self._has_value_labels = [False] * len(self._names)
        non_cat_value_labels = self._prepare_non_cat_value_labels()
        non_cat_columns = [svl.labname for svl in non_cat_value_labels]
        for i, name in enumerate(self._names):
            if name in non_cat_columns:
                self._has_value_labels[i] = True
        self._value_labels.extend(non_cat_value_labels)
        self._prepare_categoricals()
        self.nobs = len(data)
        self.nvar = len(self._names)
        self.varlist = list(self._names)
        kinds = [column.kind for column in self._columns]
        for col, column in zip(self._names, self._columns, strict=True):
            if col in self._convert_dates:
                continue
            if column.kind.startswith("datetime64[") and "," not in column.kind:
                self._convert_dates[col] = "tc"
        converted: dict[int, str] = {}
        for key, value in self._convert_dates.items():
            if not value.startswith("%"):
                value = "%" + value
            if key in self.varlist:
                converted[self.varlist.index(key)] = value
            else:
                if not isinstance(key, int):
                    raise ValueError("convert_dates key must be a column or an integer")
                converted[key] = value
        self._convert_dates = converted
        for key, fmt in self._convert_dates.items():
            if fmt not in _DATE_FORMATS:
                raise NotImplementedError(f"Format {fmt} not implemented")
            kinds[key] = "float64"
        self._encode_strings()
        self._set_formats_and_types(kinds)
        for key, fmt in self._convert_dates.items():
            self.fmtlist[key] = fmt

    def _encode_strings(self) -> None:
        convert_strl = getattr(self, "_convert_strl", [])
        for i, (col, column) in enumerate(zip(self._names, self._columns, strict=True)):
            if i in self._convert_dates or col in convert_strl:
                continue
            if column.kind != "object":
                continue
            present = [v for v in column.values if v is not None]
            if not ((present and all(isinstance(v, str) for v in present)) or not column.values):
                raise ValueError(
                    f"Column `{col}` cannot be exported.\n\nOnly string-like object arrays\n"
                    "containing all strings or a mix of strings and None can be exported.\n"
                    "Object arrays containing only null values are prohibited. Other object\n"
                    "types cannot be exported and must first be converted to one of the\n"
                    "supported types."
                )
            encoded = [None if v is None else v.encode(self._encoding) for v in column.values]
            if _max_len(column.values) <= self._max_string_length:
                column.values = encoded

    def _set_formats_and_types(self, kinds: list[str]) -> None:
        self.fmtlist: list[str] = []
        self.typlist: list[int] = []
        for kind, column, col in zip(kinds, self._columns, self._names, strict=True):
            self.fmtlist.append(_default_fmt(kind, column, col, self._dta_version, False))
            self.typlist.append(_stata_type(kind, column))

    def write_file(self) -> None:
        """Writes the file.

        Raises:
            ValueError: For labels or dates Stata cannot hold, in pandas' words.
        """
        from ._pickle import _bytes_written

        self._handle = io.BytesIO()
        self._write_header(data_label=self._data_label, time_stamp=self._time_stamp)
        self._write_map()
        self._write_variable_types()
        self._write_varnames()
        self._write_sortlist()
        self._write_formats()
        self._write_value_label_names()
        self._write_variable_labels()
        self._write_expansion_fields()
        self._write_characteristics()
        records = self._prepare_data()
        self._write_data(records)
        self._write_strls()
        self._write_value_labels()
        self._write_file_close_tag()
        self._write_map()
        _bytes_written(
            self._handle.getvalue(), self._fname, self._compression, self.storage_options
        )

    def _write_map(self) -> None:
        pass

    def _write_file_close_tag(self) -> None:
        pass

    def _write_characteristics(self) -> None:
        pass

    def _write_strls(self) -> None:
        pass

    def _write_expansion_fields(self) -> None:
        self._write(_pad_bytes("", 5))

    def _write_value_labels(self) -> None:
        for vl in self._value_labels:
            self._write_bytes(vl.generate_value_label(self._byteorder))

    def _write_header(self, data_label: str | None = None, time_stamp: Any = None) -> None:
        byteorder = self._byteorder
        self._write_bytes(struct.pack("b", 114))
        self._write((byteorder == ">" and "\x01") or "\x02")
        self._write("\x01")
        self._write("\x00")
        self._write_bytes(struct.pack(byteorder + "h", self.nvar)[:2])
        self._write_bytes(struct.pack(byteorder + "i", self.nobs)[:4])
        if data_label is None:
            self._write_bytes(self._null_terminate_bytes(_pad_bytes("", 80)))
        else:
            self._write_bytes(self._null_terminate_bytes(_pad_bytes(data_label[:80], 80)))
        self._write_bytes(self._null_terminate_bytes(_stamp(time_stamp)))

    def _write_variable_types(self) -> None:
        for typ in self.typlist:
            self._write_bytes(struct.pack("B", typ))

    def _write_varnames(self) -> None:
        for name in self.varlist:
            self._write(_pad_bytes(self._null_terminate_str(name)[:32], 33))

    def _write_sortlist(self) -> None:
        self._write(_pad_bytes("", 2 * (self.nvar + 1)))

    def _write_formats(self) -> None:
        for fmt in self.fmtlist:
            self._write(_pad_bytes(fmt, 49))

    def _write_value_label_names(self) -> None:
        for i in range(self.nvar):
            if self._has_value_labels[i]:
                name = self._null_terminate_str(self.varlist[i])
                self._write(_pad_bytes(name[:32], 33))
            else:
                self._write(_pad_bytes("", 33))

    def _write_variable_labels(self) -> None:
        blank = _pad_bytes("", 81)
        if self._variable_labels is None:
            for _ in range(self.nvar):
                self._write(blank)
            return
        for col in self._names:
            if col in self._variable_labels:
                label = self._variable_labels[col]
                if len(label) > 80:
                    raise ValueError("Variable labels must be 80 characters or fewer")
                if not all(ord(c) < 256 for c in label):
                    raise ValueError(
                        "Variable labels must contain only characters that "
                        "can be encoded in Latin-1"
                    )
                self._write(_pad_bytes(label, 81))
            else:
                self._write(blank)

    def _convert_strls(self) -> None:
        pass

    def _prepare_data(self) -> list[tuple[str, list[Any]]]:
        for i, column in enumerate(self._columns):
            if i in self._convert_dates:
                column.values = _elapsed(column.values, self.fmtlist[i])
                column.kind = "float64"
        self._convert_strls()
        records = []
        for typ, column in zip(self.typlist, self._columns, strict=True):
            if typ <= self._max_string_length:
                values = [_pad_bytes(b"" if v is None else v, typ) for v in column.values]
                records.append((f"{typ}s", [bytes(v)[:typ] for v in values]))
            elif typ == 32768:
                records.append(("Q", column.values))
            else:
                records.append((_CODES[column.kind], column.values))
        return records

    def _write_data(self, records: list[tuple[str, list[Any]]]) -> None:
        self._write_bytes(self._packed(records))

    def _packed(self, records: list[tuple[str, list[Any]]]) -> bytes:
        row = struct.Struct(self._byteorder + "".join(code for code, _ in records))
        columns = [values for _, values in records]
        return b"".join(row.pack(*values) for values in zip(*columns, strict=True))

    @staticmethod
    def _null_terminate_str(s: str) -> str:
        return s + "\x00"

    def _null_terminate_bytes(self, s: str) -> bytes:
        return self._null_terminate_str(s).encode(self._encoding)


StataWriter.__module__ = "firepanda.io.stata"


class _CategoryLabels(StataValueLabel):
    """The labels of a categorical column, from its name and categories."""

    def __init__(self, labname: str, categories: list[Any], encoding: str) -> None:
        self.labname = labname
        self._encoding = encoding
        self.value_labels = list(enumerate(categories))
        self._prepare_value_labels()


def _default_fmt(kind: str, column: _Column, col: Any, version: int, force_strl: bool) -> str:
    """The display format pandas gives a column, as `_dtype_to_default_stata_fmt`."""
    if version < 117:
        max_str_len = 244
    else:
        max_str_len = 2045
        if force_strl:
            return "%9s"
    if kind == "object":
        itemsize = _max_len(column.values)
        if itemsize > max_str_len:
            if version >= 117:
                return "%9s"
            raise ValueError(_EXCESSIVE_STRING_LENGTH_ERROR.format(col))
        return "%" + str(max(itemsize, 1)) + "s"
    formats = {"float64": "%10.0g", "float32": "%9.0g", "int32": "%12.0g"}
    if kind in formats:
        return formats[kind]
    if kind in ("int8", "int16"):
        return "%8.0g"
    raise NotImplementedError(f"Data type {kind} not supported.")


def _stata_type(kind: str, column: _Column) -> int:
    if kind == "object":
        return max(_max_len(column.values), 1)
    types = {"float64": 255, "float32": 254, "int32": 253, "int16": 252, "int8": 251}
    if kind in types:
        return types[kind]
    raise NotImplementedError(f"Data type {kind} not supported.")


def _stata_type_117(kind: str, column: _Column, force_strl: bool) -> int:
    if force_strl:
        return 32768
    if kind == "object":
        itemsize = max(_max_len(column.values), 1)
        return itemsize if itemsize <= 2045 else 32768
    types = {"float64": 65526, "float32": 65527, "int32": 65528, "int16": 65529, "int8": 65530}
    if kind in types:
        return types[kind]
    raise NotImplementedError(f"Data type {kind} not supported.")


class StataStrLWriter:
    """Builds the table of long strings for versions 117 to 119.

    `df` is a frame, or a mapping from column name to a list of values.
    """

    def __init__(
        self, df: Any, columns: Any, version: int = 117, byteorder: str | None = None
    ) -> None:
        if version not in (117, 118, 119):
            raise ValueError("Only dta versions 117, 118 and 119 supported")
        self._dta_ver = version
        self.df = df
        self.columns = columns
        self._gso_table: dict[Any, tuple[int, int]] = {"": (0, 0)}
        if byteorder is None:
            byteorder = sys.byteorder
        self._byteorder = _set_endianness(byteorder)
        self._native_byteorder = self._byteorder == _set_endianness(sys.byteorder)
        gso_v_type = "I"
        gso_o_type = "Q"
        self._encoding = "utf-8"
        if version == 117:
            o_size = 4
            gso_o_type = "I"
            self._encoding = "latin-1"
        elif version == 118:
            o_size = 6
        else:
            o_size = 5
        if self._native_byteorder:
            self._o_offet = 2 ** (8 * (8 - o_size))
        else:
            self._o_offet = 2 ** (8 * o_size)
        self._gso_o_type = gso_o_type
        self._gso_v_type = gso_v_type

    def _convert_key(self, key: tuple[int, int]) -> int:
        v, o = key
        if self._native_byteorder:
            return v + self._o_offet * o
        return o + self._o_offet * v

    def generate_table(self) -> tuple[dict[Any, tuple[int, int]], Any]:
        """The table of long strings, and the data with each string swapped for its key."""
        gso_table = self._gso_table
        plain = isinstance(self.df, dict)
        data = self.df if plain else {c: self.df[c].tolist() for c in self.df.columns}
        columns = list(data)
        col_index = [(col, columns.index(col)) for col in self.columns]
        nobs = len(data[columns[0]]) if columns else 0
        keys: dict[Any, list[int]] = {col: [] for col in self.columns}
        for o in range(nobs):
            for col, v in col_index:
                val = data[col][o]
                val = "" if _is_missing(val) else val
                key = gso_table.get(val, None)
                if key is None:
                    key = (v + 1, o + 1)
                    gso_table[val] = key
                keys[col].append(self._convert_key(key))
        new = dict(data)
        new.update(keys)
        if plain:
            return gso_table, new
        from ._frame import DataFrame, Series

        return gso_table, DataFrame(
            {c: Series(v, dtype="uint64") if c in keys else self.df[c] for c, v in new.items()}
        )

    def generate_blob(self, gso_table: dict[Any, tuple[int, int]]) -> bytes:
        """The strL section of the file."""
        bio = io.BytesIO()
        gso_type = struct.pack(self._byteorder + "B", 130)
        v_type = self._byteorder + self._gso_v_type
        o_type = self._byteorder + self._gso_o_type
        for strl, vo in gso_table.items():
            if vo == (0, 0):
                continue
            v, o = vo
            bio.write(b"GSO")
            bio.write(struct.pack(v_type, v))
            bio.write(struct.pack(o_type, o))
            bio.write(gso_type)
            strl_convert = bytes(strl, "utf-8") if isinstance(strl, str) else strl
            bio.write(struct.pack(self._byteorder + "I", len(strl_convert) + 1))
            bio.write(strl_convert)
            bio.write(b"\x00")
        return bio.getvalue()


StataStrLWriter.__module__ = "firepanda.io.stata"


class StataWriter117(StataWriter):
    """Writes a Stata 117 file, which adds long strings to the format."""

    _max_string_length = 2045
    _dta_version = 117

    def __init__(
        self,
        fname: Any,
        data: Any,
        convert_dates: dict[Any, str] | None = None,
        write_index: bool = True,
        byteorder: str | None = None,
        time_stamp: dt.datetime | None = None,
        data_label: str | None = None,
        variable_labels: dict[Any, str] | None = None,
        convert_strl: Any = None,
        compression: Any = "infer",
        storage_options: Any = None,
        *,
        value_labels: dict[Any, dict[float, str]] | None = None,
    ) -> None:
        self._convert_strl: list[Any] = []
        if convert_strl is not None:
            self._convert_strl.extend(convert_strl)
        super().__init__(
            fname,
            data,
            convert_dates,
            write_index,
            byteorder=byteorder,
            time_stamp=time_stamp,
            data_label=data_label,
            variable_labels=variable_labels,
            value_labels=value_labels,
            compression=compression,
            storage_options=storage_options,
        )
        self._map: dict[str, int] = {}
        self._strl_blob = b""

    @staticmethod
    def _tag(val: str | bytes, tag: str) -> bytes:
        if isinstance(val, str):
            val = bytes(val, "utf-8")
        return bytes("<" + tag + ">", "utf-8") + val + bytes("</" + tag + ">", "utf-8")

    def _update_map(self, tag: str) -> None:
        self._map[tag] = self._handle.tell()

    def _write_header(self, data_label: str | None = None, time_stamp: Any = None) -> None:
        byteorder = self._byteorder
        self._write_bytes(bytes("<stata_dta>", "utf-8"))
        bio = io.BytesIO()
        bio.write(self._tag(bytes(str(self._dta_version), "utf-8"), "release"))
        bio.write(self._tag((byteorder == ">" and "MSF") or "LSF", "byteorder"))
        nvar_type = "H" if self._dta_version <= 118 else "I"
        bio.write(self._tag(struct.pack(byteorder + nvar_type, self.nvar), "K"))
        nobs_size = "I" if self._dta_version == 117 else "Q"
        bio.write(self._tag(struct.pack(byteorder + nobs_size, self.nobs), "N"))
        label = data_label[:80] if data_label is not None else ""
        encoded_label = label.encode(self._encoding)
        label_size = "B" if self._dta_version == 117 else "H"
        label_len = struct.pack(byteorder + label_size, len(encoded_label))
        bio.write(self._tag(label_len + encoded_label, "label"))
        stata_ts = b"\x11" + bytes(_stamp(time_stamp), "utf-8")
        bio.write(self._tag(stata_ts, "timestamp"))
        self._write_bytes(self._tag(bio.getvalue(), "header"))

    def _write_map(self) -> None:
        if not self._map:
            self._map = dict.fromkeys(
                [
                    "stata_data",
                    "map",
                    "variable_types",
                    "varnames",
                    "sortlist",
                    "formats",
                    "value_label_names",
                    "variable_labels",
                    "characteristics",
                    "data",
                    "strls",
                    "value_labels",
                    "stata_data_close",
                    "end-of-file",
                ],
                0,
            )
            self._map["map"] = self._handle.tell()
        self._handle.seek(self._map["map"])
        bio = io.BytesIO()
        for val in self._map.values():
            bio.write(struct.pack(self._byteorder + "Q", val))
        self._write_bytes(self._tag(bio.getvalue(), "map"))

    def _write_variable_types(self) -> None:
        self._update_map("variable_types")
        bio = io.BytesIO()
        for typ in self.typlist:
            bio.write(struct.pack(self._byteorder + "H", typ))
        self._write_bytes(self._tag(bio.getvalue(), "variable_types"))

    def _write_varnames(self) -> None:
        self._update_map("varnames")
        bio = io.BytesIO()
        vn_len = 32 if self._dta_version == 117 else 128
        for name in self.varlist:
            name = self._null_terminate_str(name)
            bio.write(_pad_bytes_new(name[:32].encode(self._encoding), vn_len + 1))
        self._write_bytes(self._tag(bio.getvalue(), "varnames"))

    def _write_sortlist(self) -> None:
        self._update_map("sortlist")
        sort_size = 2 if self._dta_version < 119 else 4
        self._write_bytes(self._tag(b"\x00" * sort_size * (self.nvar + 1), "sortlist"))

    def _write_formats(self) -> None:
        self._update_map("formats")
        bio = io.BytesIO()
        fmt_len = 49 if self._dta_version == 117 else 57
        for fmt in self.fmtlist:
            bio.write(_pad_bytes_new(fmt.encode(self._encoding), fmt_len))
        self._write_bytes(self._tag(bio.getvalue(), "formats"))

    def _write_value_label_names(self) -> None:
        self._update_map("value_label_names")
        bio = io.BytesIO()
        vl_len = 32 if self._dta_version == 117 else 128
        for i in range(self.nvar):
            name = self.varlist[i] if self._has_value_labels[i] else ""
            name = self._null_terminate_str(name)
            bio.write(_pad_bytes_new(name[:32].encode(self._encoding), vl_len + 1))
        self._write_bytes(self._tag(bio.getvalue(), "value_label_names"))

    def _write_variable_labels(self) -> None:
        self._update_map("variable_labels")
        bio = io.BytesIO()
        vl_len = 80 if self._dta_version == 117 else 320
        blank = _pad_bytes_new("", vl_len + 1)
        if self._variable_labels is None:
            for _ in range(self.nvar):
                bio.write(blank)
            self._write_bytes(self._tag(bio.getvalue(), "variable_labels"))
            return
        for col in self._names:
            if col in self._variable_labels:
                label = self._variable_labels[col]
                if len(label) > 80:
                    raise ValueError("Variable labels must be 80 characters or fewer")
                try:
                    encoded = label.encode(self._encoding)
                except UnicodeEncodeError as err:
                    raise ValueError(
                        "Variable labels must contain only characters that "
                        f"can be encoded in {self._encoding}"
                    ) from err
                bio.write(_pad_bytes_new(encoded, vl_len + 1))
            else:
                bio.write(blank)
        self._write_bytes(self._tag(bio.getvalue(), "variable_labels"))

    def _write_characteristics(self) -> None:
        self._update_map("characteristics")
        self._write_bytes(self._tag(b"", "characteristics"))

    def _write_data(self, records: list[tuple[str, list[Any]]]) -> None:
        self._update_map("data")
        self._write_bytes(b"<data>")
        self._write_bytes(self._packed(records))
        self._write_bytes(b"</data>")

    def _write_strls(self) -> None:
        self._update_map("strls")
        self._write_bytes(self._tag(self._strl_blob, "strls"))

    def _write_expansion_fields(self) -> None:
        pass

    def _write_value_labels(self) -> None:
        self._update_map("value_labels")
        bio = io.BytesIO()
        for vl in self._value_labels:
            bio.write(self._tag(vl.generate_value_label(self._byteorder), "lbl"))
        self._write_bytes(self._tag(bio.getvalue(), "value_labels"))

    def _write_file_close_tag(self) -> None:
        self._update_map("stata_data_close")
        self._write_bytes(bytes("</stata_dta>", "utf-8"))
        self._update_map("end-of-file")

    def _update_strl_names(self) -> None:
        for orig, new in self._converted_names.items():
            if orig in self._convert_strl:
                idx = self._convert_strl.index(orig)
                self._convert_strl[idx] = new

    def _convert_strls(self) -> None:
        convert_cols = [
            col
            for i, col in enumerate(self._names)
            if self.typlist[i] == 32768 or col in self._convert_strl
        ]
        if convert_cols:
            data = {
                col: column.values for col, column in zip(self._names, self._columns, strict=True)
            }
            ssw = StataStrLWriter(
                data, convert_cols, version=self._dta_version, byteorder=self._byteorder
            )
            tab, new_data = ssw.generate_table()
            for col, column in zip(self._names, self._columns, strict=True):
                column.values = new_data[col]
            self._strl_blob = ssw.generate_blob(tab)

    def _set_formats_and_types(self, kinds: list[str]) -> None:
        self.typlist = []
        self.fmtlist = []
        for kind, column, col in zip(kinds, self._columns, self._names, strict=True):
            force_strl = col in self._convert_strl
            self.fmtlist.append(_default_fmt(kind, column, col, self._dta_version, force_strl))
            self.typlist.append(_stata_type_117(kind, column, force_strl))


StataWriter117.__module__ = "firepanda.io.stata"


class StataWriterUTF8(StataWriter117):
    """Writes a Stata 118 or 119 file, whose text is utf-8."""

    _encoding = "utf-8"

    def __init__(
        self,
        fname: Any,
        data: Any,
        convert_dates: dict[Any, str] | None = None,
        write_index: bool = True,
        byteorder: str | None = None,
        time_stamp: dt.datetime | None = None,
        data_label: str | None = None,
        variable_labels: dict[Any, str] | None = None,
        convert_strl: Any = None,
        version: int | None = None,
        compression: Any = "infer",
        storage_options: Any = None,
        *,
        value_labels: dict[Any, dict[float, str]] | None = None,
    ) -> None:
        if version is None:
            version = 118 if data.shape[1] <= 32767 else 119
        elif version not in (118, 119):
            raise ValueError("version must be either 118 or 119.")
        elif version == 118 and data.shape[1] > 32767:
            raise ValueError(
                "You must use version 119 for data sets containing more than32,767 variables"
            )
        super().__init__(
            fname,
            data,
            convert_dates=convert_dates,
            write_index=write_index,
            byteorder=byteorder,
            time_stamp=time_stamp,
            data_label=data_label,
            variable_labels=variable_labels,
            value_labels=value_labels,
            convert_strl=convert_strl,
            compression=compression,
            storage_options=storage_options,
        )
        self._dta_version = version

    def _validate_variable_name(self, name: str) -> str:
        for c in name:
            if (
                (
                    ord(c) < 128
                    and (c < "A" or c > "Z")
                    and (c < "a" or c > "z")
                    and (c < "0" or c > "9")
                    and c != "_"
                )
                or 128 <= ord(c) < 192
                or c in {"\u00d7", "\u00f7"}
            ):
                name = name.replace(c, "_")
        return name


StataWriterUTF8.__module__ = "firepanda.io.stata"


def to_stata(
    self: Any,
    path: Any,
    *,
    convert_dates: dict[Any, str] | None = None,
    write_index: bool = True,
    byteorder: str | None = None,
    time_stamp: dt.datetime | None = None,
    data_label: str | None = None,
    variable_labels: dict[Any, str] | None = None,
    version: int | None = 114,
    convert_strl: Any = None,
    compression: Any = "infer",
    storage_options: Any = None,
    value_labels: dict[Any, dict[float, str]] | None = None,
) -> None:
    """Writes the frame as a Stata dta file, byte for byte as pandas writes it.

    Raises:
        ValueError: For a version pandas does not write, or data Stata cannot
            hold, in pandas' words.
    """
    writer_class: Any
    if version not in (114, 117, 118, 119, None):
        raise ValueError("Only formats 114, 117, 118 and 119 are supported.")
    if version == 114:
        if convert_strl is not None:
            raise ValueError("strl is not supported in format 114")
        writer_class = StataWriter
    elif version == 117:
        writer_class = StataWriter117
    else:
        writer_class = StataWriterUTF8
    kwargs: dict[str, Any] = {}
    if version is None or version >= 117:
        kwargs["convert_strl"] = convert_strl
    if version is None or version >= 118:
        kwargs["version"] = version
    writer = writer_class(
        path,
        self,
        convert_dates=convert_dates,
        byteorder=byteorder,
        time_stamp=time_stamp,
        data_label=data_label,
        write_index=write_index,
        variable_labels=variable_labels,
        compression=compression,
        storage_options=storage_options,
        value_labels=value_labels,
        **kwargs,
    )
    writer.write_file()
