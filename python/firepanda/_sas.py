"""`read_sas` and its readers, a port of pandas' `io/sas` package.

pandas reads SAS files in two formats. The XPORT reader is Python over numpy
record arrays, and the SAS7BDAT reader is Python for the metadata with a
Cython parser for the rows and the two decompressors. This reads both with
`struct` and plain bytes, which needs no numpy, and keeps pandas' steps, their
order and their messages. Document 120 of the compat notes describes the design.
"""

from __future__ import annotations

import codecs
import collections.abc
import contextlib
import datetime as dt
import math
import struct
import sys
import warnings
from abc import abstractmethod
from typing import Any

_SAS_EPOCH = dt.datetime(1960, 1, 1)

# The 1960 origin as seconds before the Unix one.
_SAS_OFFSET = -315619200


class SASReader(collections.abc.Iterator):
    """The base of the SAS readers, which read a file a chunk at a time."""

    @abstractmethod
    def read(self, nrows: int | None = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> Any:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


SASReader.__module__ = "firepanda.io.sas.sasreader"


def _handle(filepath_or_buffer: Any, compression: Any) -> Any:
    from ._stata import _source

    return _source(filepath_or_buffer, compression, None)


def _frame(data: dict[str, Any], start: int, stop: int) -> Any:
    """The frame pandas builds from the columns, on a range index."""
    from ._frame import DataFrame
    from ._range_index import RangeIndex

    return DataFrame(data).set_axis(RangeIndex(start, stop), axis=0)


def _moments(values: list[Any], kind: str) -> Any:
    """Moments as a datetime series of the given unit, None for a gap."""
    from ._frame import Series

    if not values:
        # An empty column cannot take a datetime type directly, so one gap is cut away.
        return Series([None]).astype(kind).iloc[:0]
    return Series(values).astype(kind)


# XPORT

_CORRECT_LINE1 = "HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!000000000000000000000000000000  "
_CORRECT_HEADER1 = "HEADER RECORD*******MEMBER  HEADER RECORD!!!!!!!000000000000000001600000000"
_CORRECT_HEADER2 = (
    "HEADER RECORD*******DSCRPTR HEADER RECORD!!!!!!!000000000000000000000000000000  "
)
_CORRECT_OBS_HEADER = (
    "HEADER RECORD*******OBS     HEADER RECORD!!!!!!!000000000000000000000000000000  "
)
_FIELDKEYS = [
    "ntype",
    "nhfun",
    "field_length",
    "nvar0",
    "name",
    "label",
    "nform",
    "nfl",
    "num_decimals",
    "nfj",
    "nfill",
    "niform",
    "nifl",
    "nifd",
    "npos",
    "_",
]
_FIELD = struct.Struct(">hhhh8s40s8shhh2s8shhl52s")
_IBM = struct.Struct(">II")
_IEEE = struct.Struct(">d")
_MISSING_LEAD = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ_.")


def _parse_date(datestr: str) -> Any:
    from ._scalars import NaT

    try:
        return dt.datetime.strptime(datestr, "%d%b%y:%H:%M:%S")
    except ValueError:
        return NaT


def _split_line(s: str, parts: list[list[Any]]) -> dict[str, Any]:
    out = {}
    start = 0
    for name, length in parts:
        out[name] = s[start : start + length].strip()
        start += length
    del out["_"]
    return out


def _ibm_float(raw: bytes) -> float:
    """An IBM double as an IEEE one, with the bit steps of `_parse_float_vec`."""
    x1, x2 = _IBM.unpack(raw)
    ieee1 = x1 & 0xFFFFFF
    shift = 0
    if x1 & 0x200000:
        shift = 1
    if x1 & 0x400000:
        shift = 2
    if x1 & 0x800000:
        shift = 3
    ieee1 >>= shift
    ieee2 = ((x2 >> shift) | ((x1 & 7) << (32 - shift))) & 0xFFFFFFFF
    ieee1 &= 0xFFEFFFFF
    ieee1 |= (((((x1 >> 24) & 0x7F) - 65) << 2) + shift + 1023) << 20 | (x1 & 0x80000000)
    ieee1 &= 0xFFFFFFFF
    return _IEEE.unpack(_IBM.pack(ieee1, ieee2))[0]


class XportReader(SASReader):
    """Class for reading SAS Xport files.

    Parameters
    ----------
    filepath_or_buffer : str or file-like object
        Path to SAS file or object implementing binary read method.
    index : identifier of index column
        Identifier of column that should be used as index of the DataFrame.
    encoding : str
        Encoding for text data.
    chunksize : int
        Read file `chunksize` lines at a time, returns iterator.

    Attributes
    ----------
    member_info : list
        Contains information about the file
    fields : list
        Contains information about the variables in the file
    """

    def __init__(
        self,
        filepath_or_buffer: Any,
        index: Any = None,
        encoding: str | None = "ISO-8859-1",
        chunksize: int | None = None,
        compression: Any = "infer",
    ) -> None:
        self._encoding = encoding
        self._lines_read = 0
        self._index = index
        self._chunksize = chunksize
        codecs.lookup(encoding or "utf-8")
        self.filepath_or_buffer = _handle(filepath_or_buffer, compression)
        try:
            self._read_header()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self.filepath_or_buffer.close()

    def _get_row(self) -> str:
        return self.filepath_or_buffer.read(80).decode()

    def _read_header(self) -> None:
        self.filepath_or_buffer.seek(0)
        line1 = self._get_row()
        if line1 != _CORRECT_LINE1:
            if "**COMPRESSED**" in line1:
                raise ValueError("Header record indicates a CPORT file, which is not readable.")
            raise ValueError("Header record is not an XPORT file.")
        line2 = self._get_row()
        fif = [["prefix", 24], ["version", 8], ["OS", 8], ["_", 24], ["created", 16]]
        file_info = _split_line(line2, fif)
        if file_info["prefix"] != "SAS     SAS     SASLIB":
            raise ValueError("Header record has invalid prefix.")
        file_info["created"] = _parse_date(file_info["created"])
        self.file_info = file_info
        line3 = self._get_row()
        file_info["modified"] = _parse_date(line3[:16])
        header1 = self._get_row()
        header2 = self._get_row()
        if not (header1.startswith(_CORRECT_HEADER1) and header2 == _CORRECT_HEADER2):
            raise ValueError("Member header not found")
        fieldnamelength = int(header1[-5:-2])
        mem = [
            ["prefix", 8],
            ["set_name", 8],
            ["sasdata", 8],
            ["version", 8],
            ["OS", 8],
            ["_", 24],
            ["created", 16],
        ]
        member_info = _split_line(self._get_row(), mem)
        mem = [["modified", 16], ["_", 16], ["label", 40], ["type", 8]]
        member_info.update(_split_line(self._get_row(), mem))
        member_info["modified"] = _parse_date(member_info["modified"])
        member_info["created"] = _parse_date(member_info["created"])
        self.member_info = member_info
        types = {1: "numeric", 2: "char"}
        fieldcount = int(self._get_row()[54:58])
        datalength = fieldnamelength * fieldcount
        if datalength % 80:
            datalength += 80 - datalength % 80
        fielddata = self.filepath_or_buffer.read(datalength)
        fields = []
        obs_length = 0
        while len(fielddata) >= fieldnamelength:
            fieldbytes, fielddata = (fielddata[:fieldnamelength], fielddata[fieldnamelength:])
            fieldbytes = fieldbytes.ljust(140)
            field = dict(zip(_FIELDKEYS, _FIELD.unpack(fieldbytes), strict=True))
            del field["_"]
            field["ntype"] = types[field["ntype"]]
            fl = field["field_length"]
            if field["ntype"] == "numeric" and (fl < 2 or fl > 8):
                raise TypeError(f"Floating field width {fl} is not between 2 and 8.")
            for k, v in field.items():
                with contextlib.suppress(AttributeError):
                    field[k] = v.strip()
            obs_length += field["field_length"]
            fields += [field]
        header = self._get_row()
        if header != _CORRECT_OBS_HEADER:
            raise ValueError("Observation header not found.")
        self.fields = fields
        self.record_length = obs_length
        self.record_start = self.filepath_or_buffer.tell()
        self.nobs = self._record_count()
        self.columns = [x["name"].decode() for x in self.fields]

    def __next__(self) -> Any:
        return self.read(nrows=self._chunksize or 1)

    def _record_count(self) -> int:
        self.filepath_or_buffer.seek(0, 2)
        total_records_length = self.filepath_or_buffer.tell() - self.record_start
        if total_records_length % 80 != 0:
            warnings.warn("xport file may be corrupted.", stacklevel=6)
        if self.record_length > 80:
            self.filepath_or_buffer.seek(self.record_start)
            return total_records_length // self.record_length
        self.filepath_or_buffer.seek(-80, 2)
        last_card = self.filepath_or_buffer.read(80)
        blank = b" " * 8
        tail_pad = 8 * sum(last_card[i : i + 8] == blank for i in range(0, 80, 8))
        self.filepath_or_buffer.seek(self.record_start)
        return (total_records_length - tail_pad) // self.record_length

    def get_chunk(self, size: int | None = None) -> Any:
        """Reads lines from Xport file and returns as dataframe."""
        if size is None:
            size = self._chunksize
        return self.read(nrows=size)

    def read(self, nrows: int | None = None) -> Any:
        """Read observations from SAS Xport file, returning as data frame."""
        from ._frame import Series

        if nrows is None:
            nrows = self.nobs
        read_lines = min(nrows, self.nobs - self._lines_read)
        read_len = read_lines * self.record_length
        if read_len <= 0:
            self.close()
            raise StopIteration
        raw = self.filepath_or_buffer.read(read_len)
        widths = [field["field_length"] for field in self.fields]
        size = self.record_length
        rows = [raw[i * size : (i + 1) * size] for i in range(read_lines)]
        df_data: dict[str, Any] = {}
        start = 0
        for j, name in enumerate(self.columns):
            width = widths[j]
            cells = [row[start : start + width] for row in rows]
            start += width
            if self.fields[j]["ntype"] == "numeric":
                values = []
                for cell in cells:
                    cell = cell.ljust(8, b"\x00")
                    if cell[0] in _MISSING_LEAD and not any(cell[1:]):
                        values.append(math.nan)
                    else:
                        values.append(_ibm_float(cell))
                df_data[name] = Series(values, dtype="float64")
            else:
                texts = [cell.rstrip(b"\x00").rstrip() for cell in cells]
                if self._encoding is not None:
                    decoded = [text.decode(self._encoding) for text in texts]
                    df_data[name] = Series(decoded, dtype="str")
                else:
                    df_data[name] = Series(texts, dtype="object")
        df = _frame(df_data, self._lines_read, self._lines_read + read_lines)
        if self._index is not None:
            df = df.set_index(self._index)
        self._lines_read += read_lines
        return df


XportReader.__module__ = "firepanda.io.sas.sas_xport"


# SAS7BDAT

_MAGIC = (
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc2\xea\x81`"
    b"\xb3\x14\x11\xcf\xbd\x92\x08\x00\t\xc71\x8c\x18\x1f\x10\x11"
)
_PAGE_META_TYPES = (0, 16384)
_PAGE_DATA_TYPE = 256
_PAGE_MIX_TYPE = 512
_PAGE_AMD_TYPE = 1024
_PAGE_TYPE_MASK2 = 61440 | 3840
_SUBHEADER_POINTERS_OFFSET = 8
_RLE_COMPRESSION = b"SASYZCRL"
_RDC_COMPRESSION = b"SASYZCR2"
_ENCODING_NAMES = {
    20: "utf-8",
    29: "latin1",
    30: "latin2",
    31: "latin3",
    32: "latin4",
    33: "cyrillic",
    34: "arabic",
    35: "greek",
    36: "hebrew",
    37: "latin5",
    38: "latin6",
    39: "cp874",
    40: "latin9",
    41: "cp437",
    42: "cp850",
    43: "cp852",
    44: "cp857",
    45: "cp858",
    46: "cp862",
    47: "cp864",
    48: "cp865",
    49: "cp866",
    50: "cp869",
    51: "cp874",
    55: "cp720",
    56: "cp737",
    57: "cp775",
    58: "cp860",
    59: "cp863",
    60: "cp1250",
    61: "cp1251",
    62: "cp1252",
    63: "cp1253",
    64: "cp1254",
    65: "cp1255",
    66: "cp1256",
    67: "cp1257",
    68: "cp1258",
    118: "cp950",
    123: "big5",
    125: "gb2312",
    126: "cp936",
    134: "euc_jp",
    136: "cp932",
    138: "shift_jis",
    140: "euc-kr",
    141: "cp949",
    227: "latin8",
}

# The subheader signatures, as the index of the processor that reads each.
_SIGNATURES = {
    b"\xf7\xf7\xf7\xf7": 0,
    b"\x00\x00\x00\x00\xf7\xf7\xf7\xf7": 0,
    b"\xf7\xf7\xf7\xf7\x00\x00\x00\x00": 0,
    b"\xf7\xf7\xf7\xf7\xff\xff\xfb\xfe": 0,
    b"\xf6\xf6\xf6\xf6": 1,
    b"\x00\x00\x00\x00\xf6\xf6\xf6\xf6": 1,
    b"\xf6\xf6\xf6\xf6\x00\x00\x00\x00": 1,
    b"\xf6\xf6\xf6\xf6\xff\xff\xfb\xfe": 1,
    b"\x00\xfc\xff\xff": 2,
    b"\xff\xff\xfc\x00": 2,
    b"\x00\xfc\xff\xff\xff\xff\xff\xff": 2,
    b"\xff\xff\xff\xff\xff\xff\xfc\x00": 2,
    b"\xfd\xff\xff\xff": 3,
    b"\xff\xff\xff\xfd": 3,
    b"\xfd\xff\xff\xff\xff\xff\xff\xff": 3,
    b"\xff\xff\xff\xff\xff\xff\xff\xfd": 3,
    b"\xff\xff\xff\xff": 4,
    b"\xff\xff\xff\xff\xff\xff\xff\xff": 4,
    b"\xfc\xff\xff\xff": 5,
    b"\xff\xff\xff\xfc": 5,
    b"\xfc\xff\xff\xff\xff\xff\xff\xff": 5,
    b"\xff\xff\xff\xff\xff\xff\xff\xfc": 5,
    b"\xfe\xfb\xff\xff": 6,
    b"\xff\xff\xfb\xfe": 6,
    b"\xfe\xfb\xff\xff\xff\xff\xff\xff": 6,
    b"\xff\xff\xff\xff\xff\xff\xfb\xfe": 6,
    b"\xfe\xff\xff\xff": 7,
    b"\xff\xff\xff\xfe": 7,
    b"\xfe\xff\xff\xff\xff\xff\xff\xff": 7,
    b"\xff\xff\xff\xff\xff\xff\xff\xfe": 7,
}
_DATA_SUBHEADER_INDEX = 8

_DATE_TEXT = """
DATE DAY DDMMYY DOWNAME JULDAY JULIAN MMDDYY MMYY MMYYC MMYYD MMYYP MMYYS MMYYN MONNAME MONTH
MONYY QTR QTRR NENGO WEEKDATE WEEKDATX WEEKDAY WEEKV WORDDATE WORDDATX YEAR YYMM YYMMC YYMMD
YYMMP YYMMS YYMMN YYMON YYMMDD YYQ YYQC YYQD YYQP YYQS YYQN YYQR YYQRC YYQRD YYQRP YYQRS YYQRN
YYMMDDP YYMMDDC E8601DA YYMMDDN MMDDYYC MMDDYYS MMDDYYD YYMMDDS B8601DA DDMMYYN YYMMDDD DDMMYYB
DDMMYYP MMDDYYP YYMMDDB MMDDYYN DDMMYYC DDMMYYD DDMMYYS MINGUO
"""
_DATETIME_TEXT = """
DATETIME DTWKDATX B8601DN B8601DT B8601DX B8601DZ B8601LX E8601DN E8601DT E8601DX E8601DZ E8601LX
DATEAMPM DTDATE DTMONYY DTYEAR TOD MDYAMPM
"""
_DATE_FORMATS = frozenset(_DATE_TEXT.split())
_DATETIME_FORMATS = frozenset(_DATETIME_TEXT.split())


def _rle_decompress(inbuff: bytes) -> bytearray:
    """Run length decoding, as `rle_decompress` in pandas' `_sas.pyx`."""
    out = bytearray()
    ipos = 0
    length = len(inbuff)
    while ipos < length:
        control_byte = inbuff[ipos] & 0xF0
        end_of_first_byte = inbuff[ipos] & 0x0F
        ipos += 1
        if control_byte == 0x00:
            nbytes = inbuff[ipos] + 64 + end_of_first_byte * 256
            ipos += 1
            out += inbuff[ipos : ipos + nbytes]
            ipos += nbytes
        elif control_byte == 0x40:
            nbytes = inbuff[ipos] + 18 + end_of_first_byte * 256
            ipos += 1
            out += bytes((inbuff[ipos],)) * nbytes
            ipos += 1
        elif control_byte == 0x60:
            nbytes = end_of_first_byte * 256 + inbuff[ipos] + 17
            ipos += 1
            out += b" " * nbytes
        elif control_byte == 0x70:
            nbytes = end_of_first_byte * 256 + inbuff[ipos] + 17
            ipos += 1
            out += b"\x00" * nbytes
        elif control_byte in (0x80, 0x90, 0xA0, 0xB0):
            nbytes = end_of_first_byte + {0x80: 1, 0x90: 17, 0xA0: 33, 0xB0: 49}[control_byte]
            out += inbuff[ipos : ipos + nbytes]
            ipos += nbytes
        elif control_byte == 0xC0:
            nbytes = end_of_first_byte + 3
            out += bytes((inbuff[ipos],)) * nbytes
            ipos += 1
        elif control_byte == 0xD0:
            out += b"@" * (end_of_first_byte + 2)
        elif control_byte == 0xE0:
            out += b" " * (end_of_first_byte + 2)
        elif control_byte == 0xF0:
            out += b"\x00" * (end_of_first_byte + 2)
        else:
            raise ValueError(f"unknown control byte: {control_byte}")
    return out


def _rdc_decompress(inbuff: bytes) -> bytearray:
    """Ross Data Compression decoding, as `rdc_decompress` in pandas' `_sas.pyx`."""
    out = bytearray()
    ctrl_bits = 0
    ctrl_mask = 0
    ipos = 0
    length = len(inbuff)
    while ipos < length:
        ctrl_mask >>= 1
        if ctrl_mask == 0:
            ctrl_bits = (inbuff[ipos] << 8) + inbuff[ipos + 1]
            ipos += 2
            ctrl_mask = 0x8000
        if ctrl_bits & ctrl_mask == 0:
            out.append(inbuff[ipos])
            ipos += 1
            continue
        cmd = (inbuff[ipos] >> 4) & 0x0F
        cnt = inbuff[ipos] & 0x0F
        ipos += 1
        if cmd == 0:
            out += bytes((inbuff[ipos],)) * (cnt + 3)
            ipos += 1
        elif cmd == 1:
            cnt += (inbuff[ipos] << 4) + 19
            ipos += 1
            out += bytes((inbuff[ipos],)) * cnt
            ipos += 1
        elif cmd == 2:
            ofs = cnt + 3 + (inbuff[ipos] << 4)
            ipos += 1
            cnt = inbuff[ipos] + 16
            ipos += 1
            _copy_back(out, ofs, cnt)
        else:
            ofs = cnt + 3 + (inbuff[ipos] << 4)
            ipos += 1
            _copy_back(out, ofs, cmd)
    return out


def _copy_back(out: bytearray, ofs: int, cnt: int) -> None:
    """Append `cnt` bytes copied from `ofs` bytes back, which may overlap what is added."""
    start = len(out) - ofs
    if start < 0:
        raise AssertionError("Out of bounds read")
    for k in range(cnt):
        out.append(out[start + k])


class _Column:
    def __init__(
        self, col_id: int, name: Any, label: Any, format: Any, ctype: bytes, length: int
    ) -> None:
        self.col_id = col_id
        self.name = name
        self.label = label
        self.format = format
        self.ctype = ctype
        self.length = length


class SAS7BDATReader(SASReader):
    """Read SAS files in SAS7BDAT format.

    Parameters
    ----------
    path_or_buf : path name or buffer
        Name of SAS file or file-like object pointing to SAS file
        contents.
    index : column identifier, defaults to None
        Column to use as index.
    convert_dates : bool, defaults to True
        Attempt to convert dates to Pandas datetime values.  Note that
        some rarely used SAS date formats may be unsupported.
    blank_missing : bool, defaults to True
        Convert empty strings to missing values (SAS uses blanks to
        indicate missing character variables).
    chunksize : int, defaults to None
        Return SAS7BDATReader object for iterations, returns chunks
        with given number of lines.
    encoding : str, 'infer', defaults to None
        String encoding acc. to Python standard encodings,
        encoding='infer' tries to detect the encoding from the file header,
        encoding=None will leave the data in binary format.
    convert_text : bool, defaults to True
        If False, text variables are left as raw bytes.
    convert_header_text : bool, defaults to True
        If False, header text, including column names, are left as raw
        bytes.
    """

    def __init__(
        self,
        path_or_buf: Any,
        index: Any = None,
        convert_dates: bool = True,
        blank_missing: bool = True,
        chunksize: int | None = None,
        encoding: str | None = None,
        convert_text: bool = True,
        convert_header_text: bool = True,
        compression: Any = "infer",
    ) -> None:
        self.index = index
        self.convert_dates = convert_dates
        self.blank_missing = blank_missing
        self.chunksize = chunksize
        self.encoding = encoding
        self.convert_text = convert_text
        self.convert_header_text = convert_header_text
        self.default_encoding = "latin-1"
        self.compression = b""
        self.column_names_raw: list[bytes] = []
        self.column_names: list[Any] = []
        self.column_formats: list[Any] = []
        self.columns: list[_Column] = []
        self._current_page_data_subheader_pointers: list[tuple[int, int]] = []
        self._cached_page: bytes | None = None
        self._column_data_lengths: list[int] = []
        self._column_data_offsets: list[int] = []
        self._column_types: list[bytes] = []
        self._current_row_in_file_index = 0
        self._current_row_on_page_index = 0
        self._path_or_buf = _handle(path_or_buf, compression)
        self._subheader_processors = [
            self._process_rowsize_subheader,
            self._process_columnsize_subheader,
            self._process_subheader_counts,
            self._process_columntext_subheader,
            self._process_columnname_subheader,
            self._process_columnattributes_subheader,
            self._process_format_subheader,
            self._process_columnlist_subheader,
            None,
        ]
        try:
            self._get_properties()
            self._parse_metadata()
        except Exception:
            self.close()
            raise

    def column_data_lengths(self) -> list[int]:
        """Return a list of column data lengths."""
        return list(self._column_data_lengths)

    def column_data_offsets(self) -> list[int]:
        """Return a list of column data offsets."""
        return list(self._column_data_offsets)

    def column_types(self) -> list[bytes]:
        """Returns a list of the column types."""
        return list(self._column_types)

    def close(self) -> None:
        self._path_or_buf.close()

    def _get_properties(self) -> None:
        from ._pandas import to_timedelta

        self._path_or_buf.seek(0)
        self._cached_page = self._path_or_buf.read(288)
        if self._cached_page[0 : len(_MAGIC)] != _MAGIC:
            raise ValueError("magic number mismatch (not a SAS file?)")
        if self._read_bytes(32, 1) == b"3":
            self.U64 = True
            self._int_length = 8
            self._page_bit_offset = 32
            self._subheader_pointer_length = 24
        else:
            self.U64 = False
            self._page_bit_offset = 16
            self._subheader_pointer_length = 12
            self._int_length = 4
        align1 = 4 if self._read_bytes(35, 1) == b"3" else 0
        if self._read_bytes(37, 1) == b"\x01":
            self.byte_order = "<"
            self.need_byteswap = sys.byteorder == "big"
        else:
            self.byte_order = ">"
            self.need_byteswap = sys.byteorder == "little"
        code = self._read_bytes(70, 1)[0]
        if code in _ENCODING_NAMES:
            self.inferred_encoding = _ENCODING_NAMES[code]
            if self.encoding == "infer":
                self.encoding = self.inferred_encoding
        else:
            self.inferred_encoding = f"unknown (code={code})"
        x = self._read_float(164 + align1, 8)
        self.date_created = _SAS_EPOCH + to_timedelta(x, unit="s")
        x = self._read_float(172 + align1, 8)
        self.date_modified = _SAS_EPOCH + to_timedelta(x, unit="s")
        self.header_length = self._read_uint(196 + align1, 4)
        self._cached_page += self._path_or_buf.read(self.header_length - 288)
        if len(self._cached_page) != self.header_length:
            raise ValueError("The SAS7BDAT file appears to be truncated.")
        self._page_length = self._read_uint(200 + align1, 4)

    def __next__(self) -> Any:
        da = self.read(nrows=self.chunksize or 1)
        if da.empty:
            self.close()
            raise StopIteration
        return da

    def _read_float(self, offset: int, width: int) -> float:
        if width == 4:
            return struct.unpack(self.byte_order + "f", self._read_bytes(offset, 4))[0]
        if width == 8:
            return struct.unpack(self.byte_order + "d", self._read_bytes(offset, 8))[0]
        self.close()
        raise ValueError("invalid float width")

    def _read_uint(self, offset: int, width: int) -> int:
        if width == 1:
            return self._read_bytes(offset, 1)[0]
        codes = {2: "H", 4: "I", 8: "Q"}
        if width in codes:
            return struct.unpack(self.byte_order + codes[width], self._read_bytes(offset, width))[0]
        self.close()
        raise ValueError("invalid int width")

    def _read_bytes(self, offset: int, length: int) -> bytes:
        assert self._cached_page is not None
        if offset + length > len(self._cached_page):
            self.close()
            raise ValueError("The cached page is too small.")
        return self._cached_page[offset : offset + length]

    def _parse_metadata(self) -> None:
        done = False
        while not done:
            self._cached_page = self._path_or_buf.read(self._page_length)
            if len(self._cached_page) <= 0:
                break
            if len(self._cached_page) != self._page_length:
                raise ValueError("Failed to read a meta data page from the SAS file.")
            done = self._process_page_meta()

    def _process_page_meta(self) -> bool:
        self._read_page_header()
        if self._current_page_type in (*_PAGE_META_TYPES, _PAGE_AMD_TYPE, _PAGE_MIX_TYPE):
            self._process_page_metadata()
        is_data_page = self._current_page_type == _PAGE_DATA_TYPE
        is_mix_page = self._current_page_type == _PAGE_MIX_TYPE
        return bool(is_data_page or is_mix_page or self._current_page_data_subheader_pointers)

    def _read_page_header(self) -> None:
        bit_offset = self._page_bit_offset
        self._current_page_type = self._read_uint(bit_offset, 2) & _PAGE_TYPE_MASK2
        self._current_page_block_count = self._read_uint(2 + bit_offset, 2)
        self._current_page_subheaders_count = self._read_uint(4 + bit_offset, 2)

    def _process_page_metadata(self) -> None:
        bit_offset = self._page_bit_offset
        for i in range(self._current_page_subheaders_count):
            total_offset = _SUBHEADER_POINTERS_OFFSET + bit_offset
            total_offset += self._subheader_pointer_length * i
            subheader_offset = self._read_uint(total_offset, self._int_length)
            total_offset += self._int_length
            subheader_length = self._read_uint(total_offset, self._int_length)
            total_offset += self._int_length
            subheader_compression = self._read_uint(total_offset, 1)
            total_offset += 1
            subheader_type = self._read_uint(total_offset, 1)
            if subheader_length == 0 or subheader_compression == 1:
                continue
            subheader_signature = self._read_bytes(subheader_offset, self._int_length)
            subheader_index = _SIGNATURES.get(subheader_signature, _DATA_SUBHEADER_INDEX)
            subheader_processor = self._subheader_processors[subheader_index]
            if subheader_processor is None:
                f1 = subheader_compression in (4, 0)
                f2 = subheader_type == 1
                if self.compression and f1 and f2:
                    self._current_page_data_subheader_pointers.append(
                        (subheader_offset, subheader_length)
                    )
                else:
                    self.close()
                    raise ValueError(f"Unknown subheader signature {subheader_signature}")
            else:
                subheader_processor(subheader_offset, subheader_length)

    def _process_rowsize_subheader(self, offset: int, length: int) -> None:
        int_len = self._int_length
        if self.U64:
            lcs_offset = offset + 682
            lcp_offset = offset + 706
        else:
            lcs_offset = offset + 354
            lcp_offset = offset + 378
        self.row_length = self._read_uint(offset + 5 * int_len, int_len)
        self.row_count = self._read_uint(offset + 6 * int_len, int_len)
        self.col_count_p1 = self._read_uint(offset + 9 * int_len, int_len)
        self.col_count_p2 = self._read_uint(offset + 10 * int_len, int_len)
        self._mix_page_row_count = self._read_uint(offset + 15 * int_len, int_len)
        self._lcs = self._read_uint(lcs_offset, 2)
        self._lcp = self._read_uint(lcp_offset, 2)

    def _process_columnsize_subheader(self, offset: int, length: int) -> None:
        int_len = self._int_length
        offset += int_len
        self.column_count = self._read_uint(offset, int_len)
        if self.col_count_p1 + self.col_count_p2 != self.column_count:
            print(
                f"Warning: column count mismatch ({self.col_count_p1} + "
                f"{self.col_count_p2} != {self.column_count})\n"
            )

    def _process_subheader_counts(self, offset: int, length: int) -> None:
        pass

    def _process_columntext_subheader(self, offset: int, length: int) -> None:
        offset += self._int_length
        text_block_size = self._read_uint(offset, 2)
        buf = self._read_bytes(offset, text_block_size)
        cname_raw = buf[0:text_block_size].rstrip(b"\x00 ")
        self.column_names_raw.append(cname_raw)
        if len(self.column_names_raw) != 1:
            return
        compression_literal = b""
        for cl in (_RLE_COMPRESSION, _RDC_COMPRESSION):
            if cl in cname_raw:
                compression_literal = cl
        self.compression = compression_literal
        offset -= self._int_length
        wide = 4 if self.U64 else 0
        buf = self._read_bytes(offset + 16 + wide, self._lcp)
        compression_literal = buf.rstrip(b"\x00")
        if compression_literal == b"":
            self._lcs = 0
            buf = self._read_bytes(offset + 32 + wide, self._lcp)
            self.creator_proc = buf[0 : self._lcp]
        elif compression_literal == _RLE_COMPRESSION:
            buf = self._read_bytes(offset + 40 + wide, self._lcp)
            self.creator_proc = buf[0 : self._lcp]
        elif self._lcs > 0:
            self._lcp = 0
            buf = self._read_bytes(offset + 16 + wide, self._lcs)
            self.creator_proc = buf[0 : self._lcp]
        if hasattr(self, "creator_proc"):
            self.creator_proc = self._convert_header_text(self.creator_proc)

    def _process_columnname_subheader(self, offset: int, length: int) -> None:
        int_len = self._int_length
        offset += int_len
        column_name_pointers_count = (length - 2 * int_len - 12) // 8
        for i in range(column_name_pointers_count):
            base = offset + 8 * (i + 1)
            idx = self._read_uint(base, 2)
            col_offset = self._read_uint(base + 2, 2)
            col_len = self._read_uint(base + 4, 2)
            name_raw = self.column_names_raw[idx]
            cname = name_raw[col_offset : col_offset + col_len]
            self.column_names.append(self._convert_header_text(cname))

    def _process_columnattributes_subheader(self, offset: int, length: int) -> None:
        int_len = self._int_length
        column_attributes_vectors_count = (length - 2 * int_len - 12) // (int_len + 8)
        for i in range(column_attributes_vectors_count):
            step = i * (int_len + 8)
            x = self._read_uint(offset + int_len + 8 + step, int_len)
            self._column_data_offsets.append(x)
            x = self._read_uint(offset + 2 * int_len + 8 + step, 4)
            self._column_data_lengths.append(x)
            x = self._read_uint(offset + 2 * int_len + 14 + step, 1)
            self._column_types.append(b"d" if x == 1 else b"s")

    def _process_columnlist_subheader(self, offset: int, length: int) -> None:
        pass

    def _process_format_subheader(self, offset: int, length: int) -> None:
        base = offset + 3 * self._int_length
        x = self._read_uint(base + 22, 2)
        format_idx = min(x, len(self.column_names_raw) - 1)
        format_start = self._read_uint(base + 24, 2)
        format_len = self._read_uint(base + 26, 2)
        label_idx = self._read_uint(base + 28, 2)
        label_idx = min(label_idx, len(self.column_names_raw) - 1)
        label_start = self._read_uint(base + 30, 2)
        label_len = self._read_uint(base + 32, 2)
        label_names = self.column_names_raw[label_idx]
        column_label = self._convert_header_text(label_names[label_start : label_start + label_len])
        format_names = self.column_names_raw[format_idx]
        column_format = self._convert_header_text(
            format_names[format_start : format_start + format_len]
        )
        current_column_number = len(self.columns)
        col = _Column(
            current_column_number,
            self.column_names[current_column_number],
            column_label,
            column_format,
            self._column_types[current_column_number],
            self._column_data_lengths[current_column_number],
        )
        self.column_formats.append(column_format)
        self.columns.append(col)

    def read(self, nrows: int | None = None) -> Any:
        from ._frame import DataFrame

        if nrows is None and self.chunksize is not None:
            nrows = self.chunksize
        elif nrows is None:
            nrows = self.row_count
        if len(self._column_types) == 0:
            from .errors import EmptyDataError

            self.close()
            raise EmptyDataError("No columns to parse from file")
        if nrows > 0 and self._current_row_in_file_index >= self.row_count:
            return DataFrame()
        nrows = min(nrows, self.row_count - self._current_row_in_file_index)
        self._numbers: list[list[float]] = [[] for t in self._column_types if t == b"d"]
        self._strings: list[list[Any]] = [[] for t in self._column_types if t == b"s"]
        self._current_row_in_chunk_index = 0
        self._parse_rows(nrows)
        rslt = self._chunk_to_dataframe()
        if self.index is not None:
            rslt = rslt.set_index(self.index)
        return rslt

    def _read_next_page(self) -> bool:
        self._current_page_data_subheader_pointers = []
        self._cached_page = self._path_or_buf.read(self._page_length)
        if len(self._cached_page) <= 0:
            return True
        if len(self._cached_page) != self._page_length:
            self.close()
            raise ValueError(
                f"failed to read complete page from file (read {len(self._cached_page):d} "
                f"of {self._page_length:d} bytes)"
            )
        self._read_page_header()
        if self._current_page_type in _PAGE_META_TYPES:
            self._process_page_metadata()
        if self._current_page_type not in (*_PAGE_META_TYPES, _PAGE_DATA_TYPE, _PAGE_MIX_TYPE):
            return self._read_next_page()
        self._current_row_on_page_index = 0
        return False

    def _parse_rows(self, nrows: int) -> None:
        """The row loop of pandas' Cython `Parser.read`."""
        if self.compression == _RLE_COMPRESSION:
            self._decompress = _rle_decompress
        elif self.compression == _RDC_COMPRESSION:
            self._decompress = _rdc_decompress
        else:
            self._decompress = None
        self._layout = list(
            zip(
                self._column_data_lengths,
                self._column_data_offsets,
                self._column_types,
                strict=True,
            )
        )
        for _ in range(nrows):
            if self._readline():
                break

    def _readline(self) -> bool:
        bit_offset = self._page_bit_offset
        while True:
            page_type = self._current_page_type
            if page_type in _PAGE_META_TYPES:
                pointers = self._current_page_data_subheader_pointers
                if self._current_row_on_page_index >= len(pointers):
                    if self._read_next_page():
                        return True
                    continue
                offset, length = pointers[self._current_row_on_page_index]
                self._process_byte_array_with_data(offset, length)
                return False
            if page_type == _PAGE_MIX_TYPE:
                offset = bit_offset + _SUBHEADER_POINTERS_OFFSET
                offset += self._current_page_subheaders_count * self._subheader_pointer_length
                offset += offset % 8
                offset += self._current_row_on_page_index * self.row_length
                self._process_byte_array_with_data(offset, self.row_length)
                mn = min(self.row_count, self._mix_page_row_count)
                return self._current_row_on_page_index == mn and self._read_next_page()
            if page_type == _PAGE_DATA_TYPE:
                offset = bit_offset + _SUBHEADER_POINTERS_OFFSET
                offset += self._current_row_on_page_index * self.row_length
                self._process_byte_array_with_data(offset, self.row_length)
                block_count = self._current_page_block_count
                return self._current_row_on_page_index == block_count and self._read_next_page()
            raise ValueError(f"unknown page type: {page_type}")

    def _process_byte_array_with_data(self, offset: int, length: int) -> None:
        page = self._cached_page
        assert page is not None
        assert offset + length <= len(page), "Out of bounds read"
        source = page[offset : offset + length]
        if self._decompress is not None and length < self.row_length:
            source = self._decompress(source)
            if len(source) != self.row_length:
                raise ValueError(
                    f"Expected decompressed line of length {self.row_length} bytes "
                    f"but decompressed {len(source)} bytes"
                )
        little = self.byte_order == "<"
        jb = js = 0
        filled_numbers = filled_strings = 0
        for lngt, start, ctype in self._layout:
            if lngt == 0:
                break
            if ctype == b"d":
                raw = source[start : start + lngt]
                raw = raw.rjust(8, b"\x00") if little else raw.ljust(8, b"\x00")
                self._numbers[jb].append(struct.unpack(self.byte_order + "d", raw)[0])
                jb += 1
            else:
                text = bytes(source[start : start + lngt]).rstrip(b"\x00 ")
                if not text and self.blank_missing:
                    self._strings[js].append(math.nan)
                else:
                    self._strings[js].append(text)
                js += 1
        filled_numbers, filled_strings = jb, js
        for column in self._numbers[filled_numbers:]:
            column.append(0.0)
        for column in self._strings[filled_strings:]:
            column.append(None)
        self._current_row_on_page_index += 1
        self._current_row_in_chunk_index += 1
        self._current_row_in_file_index += 1

    def _chunk_to_dataframe(self) -> Any:
        from ._frame import Series

        n = self._current_row_in_chunk_index
        m = self._current_row_in_file_index
        if len(set(self.column_names)) != len(self.column_names):
            self.close()
            raise NotImplementedError(
                "read_sas: the column names repeat, and a firepanda frame names each column once"
            )
        rslt: dict[Any, Any] = {}
        js = jb = 0
        for j in range(self.column_count):
            name = self.column_names[j]
            if self._column_types[j] == b"d":
                values = self._numbers[jb]
                fmt = self.column_formats[j]
                if self.convert_dates and fmt in _DATE_FORMATS:
                    rslt[name] = _moments(_sas_dates(values), "datetime64[s]")
                elif self.convert_dates and fmt in _DATETIME_FORMATS:
                    rslt[name] = _moments(_sas_datetimes(values), "datetime64[ms]")
                else:
                    rslt[name] = Series(values, dtype="float64")
                jb += 1
            elif self._column_types[j] == b"s":
                values = self._strings[js]
                if self.convert_text and self.encoding is not None:
                    encoding = self.encoding or self.default_encoding
                    decoded = [
                        v.decode(encoding) if isinstance(v, bytes) else math.nan for v in values
                    ]
                    rslt[name] = Series(decoded, dtype="str")
                else:
                    rslt[name] = Series(values, dtype="object")
                js += 1
            else:
                self.close()
                raise ValueError(f"unknown column type {self._column_types[j]!r}")
        return _frame(rslt, m - n, m)

    def _decode_string(self, b: bytes) -> str:
        return b.decode(self.encoding or self.default_encoding)

    def _convert_header_text(self, b: bytes) -> Any:
        if self.convert_header_text:
            return self._decode_string(b)
        return b


SAS7BDATReader.__module__ = "firepanda.io.sas.sas7bdat"


def _sas_dates(values: list[float]) -> list[Any]:
    """Days from 1960 as moments, cast toward zero as numpy casts floats to days."""
    return [None if math.isnan(v) else _SAS_EPOCH + dt.timedelta(days=int(v)) for v in values]


def _sas_datetimes(values: list[float]) -> list[Any]:
    """Seconds from 1960 as moments to the millisecond, as `cast_from_unit_vectorized`."""
    out = []
    for v in values:
        if math.isnan(v):
            out.append(None)
            continue
        base = int(v)
        frac = round((v - base) * 1000.0) / 1000.0
        millis = base * 1000 + int(frac * 1000.0)
        out.append(_SAS_EPOCH + dt.timedelta(milliseconds=millis))
    return out


def read_sas(
    filepath_or_buffer: Any,
    *,
    format: str | None = None,
    index: Any = None,
    encoding: str | None = None,
    chunksize: int | None = None,
    iterator: bool = False,
    compression: Any = "infer",
) -> Any:
    """Read SAS files stored as either XPORT or SAS7BDAT format files.

    Parameters
    ----------
    filepath_or_buffer : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``read()`` function. The string could be a URL.
    format : str {'xport', 'sas7bdat'} or None
        If None, file format is inferred from file extension. If 'xport' or
        'sas7bdat', uses the corresponding format.
    index : identifier of index column, defaults to None
        Identifier of column that should be used as index of the DataFrame.
    encoding : str, default is None
        Encoding for text data.  If None, text data are stored as raw bytes.
    chunksize : int
        Read file `chunksize` lines at a time, returns iterator.
    iterator : bool, defaults to False
        If True, returns an iterator for reading the file incrementally.
    compression : str or dict, default 'infer'
        For on-the-fly decompression of on-disk data.

    Returns
    -------
    DataFrame, SAS7BDATReader, or XportReader
        DataFrame if iterator=False and chunksize=None, else SAS7BDATReader
        or XportReader, file format is inferred from file extension.
    """
    import os

    if format is None:
        buffer_error_msg = (
            "If this is a buffer object rather than a string name, you must specify a format string"
        )
        if isinstance(filepath_or_buffer, os.PathLike):
            filepath_or_buffer = os.fspath(filepath_or_buffer)
        if not isinstance(filepath_or_buffer, str):
            raise ValueError(buffer_error_msg)
        fname = filepath_or_buffer.lower()
        if ".xpt" in fname:
            format = "xport"
        elif ".sas7bdat" in fname:
            format = "sas7bdat"
        else:
            raise ValueError(f"unable to infer format of SAS file from filename: {fname!r}")
    reader: SASReader
    if format.lower() == "xport":
        reader = XportReader(
            filepath_or_buffer,
            index=index,
            encoding=encoding,
            chunksize=chunksize,
            compression=compression,
        )
    elif format.lower() == "sas7bdat":
        reader = SAS7BDATReader(
            filepath_or_buffer,
            index=index,
            encoding=encoding,
            chunksize=chunksize,
            compression=compression,
        )
    else:
        raise ValueError("unknown SAS format")
    if iterator or chunksize:
        return reader
    with reader:
        return reader.read()
