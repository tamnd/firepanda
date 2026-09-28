"""pandas' text readers, for every call the core CSV reader cannot take.

The core reader maps a file by its path and reads it one way, which is fast and is
pandas' answer for a plain file. This module is the rest of `read_csv`, and all of
`read_table` and `read_fwf`. It reads the text itself, from a path, a handle, bytes or a
URL, compressed or not and in any encoding. Then it splits the text with the standard
library's `csv` module where it can, and with a tokeniser of its own where pandas' C
parser does something `csv` cannot: a comment character, a line terminator of the
caller's choosing, or runs of whitespace as the separator.

The rows then go through pandas' steps in pandas' order. Skipped rows and blank lines
come first, then the header, the names and their mangling, and a leading field with no
name, which becomes the row labels. After those come the rows with too many fields,
`usecols` and `index_col`. Each column's type is picked the way the C parser picks it.
An integer is tried first, and a gap makes it float64. A float is tried next, then the
booleans, and anything else is text.

The columns reach firepanda as Arrow data rather than as lists of Python objects. That
is how an unsigned column and a gap in an integer column arrive intact, and how
`DataFrame.from_arrow` gets to apply pandas' widening while `firepanda.from_arrow` keeps
Arrow's types for `dtype_backend`.
"""

from __future__ import annotations

import csv
import inspect
import io
import math
import os
import re
import warnings
from array import array
from collections.abc import Callable
from typing import Any

from .errors import EmptyDataError, ParserError, ParserWarning

DEFAULT_NA = frozenset(
    {
        "",
        "#N/A",
        "#N/A N/A",
        "#NA",
        "-1.#IND",
        "-1.#QNAN",
        "-NaN",
        "-nan",
        "1.#IND",
        "1.#QNAN",
        "<NA>",
        "N/A",
        "NA",
        "NULL",
        "NaN",
        "None",
        "n/a",
        "nan",
        "null",
    }
)
"""The text pandas reads as missing unless `keep_default_na=False`."""

_TRUE = frozenset({"True", "TRUE", "true"})
_FALSE = frozenset({"False", "FALSE", "false"})
_NAN = frozenset({"nan", "+nan", "-nan"})
_INT64 = (-(1 << 63), (1 << 63) - 1)
_UINT64 = (1 << 64) - 1
_LINES = re.compile(r"\r\n|\r|\n")
_FIELD_LIMIT = (1 << 31) - 1

_WHITESPACE = r"\s+"
_REGEX_SEPARATOR = (
    "the 'c' engine does not support regex separators (separators > 1 char and"
    " different from '\\s+' are interpreted as regex)"
)
_HEADER_BOOL = (
    "Passing a bool to header is invalid. Use header=None for no header or header=int or"
    " list-like of ints to specify the row(s) making up the column names"
)
_PARSE_DATES = "Only booleans and lists are accepted for the 'parse_dates' parameter"
_STORAGE = "storage_options passed with file object or non-fsspec file path"
_PRECISIONS = (None, "high", "legacy", "round_trip")
_INT_NAMES = frozenset({"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"})
_FLOAT_NAMES = frozenset({"float16", "float32", "float64"})
_TEXT_NAMES = frozenset({"str", "string", "object", "O", "U", "unicode"})
_UTF8 = ("utf-8", "utf8", "utf_8")


class _Options:
    """The reader's arguments as attributes, with what is derived from them."""

    def __init__(self, values: dict[str, Any]) -> None:
        self.__dict__.update(values)


def _defaults() -> dict[str, Any]:
    from ._pandas import read_csv

    return {
        name: parameter.default
        for name, parameter in inspect.signature(read_csv).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def _warn_fallback(why: str) -> None:
    warnings.warn(
        f"Falling back to the 'python' engine because {why}; you can avoid this warning by"
        " specifying engine='python'.",
        ParserWarning,
        stacklevel=4,
    )


def _checked(values: dict[str, Any], separator: str | None) -> _Options:
    """The arguments checked as pandas checks them, in pandas' order."""
    from ._pandas import NO_DEFAULT

    o = _Options(values)
    o.python = o.engine == "python"
    if o.dialect is not None:
        dialect = csv.get_dialect(o.dialect) if isinstance(o.dialect, str) else o.dialect
        separator = dialect.delimiter
        o.quotechar = dialect.quotechar
        o.doublequote = dialect.doublequote
        o.escapechar = dialect.escapechar
        o.skipinitialspace = dialect.skipinitialspace
        o.quoting = dialect.quoting
    o.sep = separator
    o.whitespace = separator == _WHITESPACE
    o.regex = separator is not None and len(separator) > 1 and not o.whitespace
    if o.regex:
        if o.engine == "c":
            raise ValueError(_REGEX_SEPARATOR)
        if o.engine is None:
            _warn_fallback(_REGEX_SEPARATOR)
        o.python = True
    if o.skipfooter:
        if o.engine == "c":
            raise ValueError("the 'c' engine does not support skipfooter")
        if o.engine is None and not o.regex:
            _warn_fallback("the 'c' engine does not support skipfooter")
        o.python = True
        if o.nrows is not None:
            raise ValueError("'skipfooter' not supported with 'nrows'")
    if o.nrows is not None and (
        not isinstance(o.nrows, int) or isinstance(o.nrows, bool) or o.nrows < 0
    ):
        raise ValueError("'nrows' must be an integer >=0")
    if o.chunksize is not None and (
        not isinstance(o.chunksize, int) or isinstance(o.chunksize, bool) or o.chunksize < 1
    ):
        raise ValueError("'chunksize' must be an integer >=1")
    if len(o.decimal) != 1:
        raise ValueError("Only length-1 decimal markers supported")
    if o.thousands is not None and len(o.thousands) != 1:
        raise ValueError("Only length-1 thousands markers supported")
    if o.lineterminator is not None and len(o.lineterminator) != 1:
        raise ValueError("Only length-1 line terminators supported")
    if o.comment is not None and len(o.comment) != 1:
        raise ValueError("Only length-1 comment characters supported")
    if o.float_precision not in _PRECISIONS:
        raise ValueError(f"Unrecognized float_precision option: {o.float_precision}")
    bad = o.on_bad_lines
    if callable(bad):
        if o.engine not in ("python", "pyarrow"):
            raise ValueError(
                "on_bad_line can only be a callable function if engine='python' or 'pyarrow'"
            )
    elif bad not in ("error", "warn", "skip"):
        raise ValueError(f"Argument {bad} is invalid for on_bad_lines")
    header = o.header
    if isinstance(header, bool):
        raise TypeError(_HEADER_BOOL)
    if isinstance(header, (list, tuple)):
        if len(header) != 1:
            raise NotImplementedError(
                "a header over more than one row is not supported yet, because the"
                " columns would be a MultiIndex and a frame cannot hold one yet"
            )
        header = header[0]
    if isinstance(header, int) and header < 0:
        raise ValueError(
            "Passing negative integer to header is invalid. For no header, use header=None instead"
        )
    o.header = header
    if o.index_col is True:
        raise ValueError("The value of index_col couldn't be 'True'")
    names = None if o.names is NO_DEFAULT else o.names
    if names is not None:
        if isinstance(names, (set, dict, str)):
            raise ValueError("Names should be an ordered collection.")
        names = list(names)
        if len(set(names)) != len(names):
            raise ValueError("Duplicate names are not allowed.")
        names = [str(name) for name in names]
    o.names = names
    dates = o.parse_dates
    if dates is None or dates is False:
        o.dates, o.date_index = [], False
    elif dates is True:
        o.dates, o.date_index = [], True
    elif isinstance(dates, (list, tuple)):
        if any(isinstance(one, (list, tuple)) for one in dates):
            raise TypeError("list indices must be integers or slices, not list")
        o.dates, o.date_index = list(dates), False
    else:
        raise TypeError(_PARSE_DATES)
    for dtype in _dtypes(o.dtype):
        name = str(dtype)
        if name.startswith(("datetime64", "timedelta64", "<M8", "<m8", "M8", "m8")):
            raise TypeError(
                f"the dtype {name} is not supported for parsing, pass this column using"
                " parse_dates instead"
            )
    o.backend = o.dtype_backend is not NO_DEFAULT
    return o


def _dtypes(dtype: Any) -> list[Any]:
    if dtype is None:
        return []
    if isinstance(dtype, dict):
        found = list(dtype.values())
        factory = getattr(dtype, "default_factory", None)
        return found + ([factory()] if factory is not None else [])
    return [dtype]


def _local(path: str) -> str:
    return os.path.expanduser(path)


def _text(source: Any, o: _Options) -> str:
    """The whole text of the source, decompressed and decoded."""
    from ._pickle import _method, _unpacked

    if isinstance(source, bytes):
        source = source.decode()
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        if isinstance(path, bytes):
            path = path.decode()
        method, _ = _method(path, o.compression)
        if "://" in path:
            data = _fetched(path, o.storage_options)
        else:
            if o.storage_options is not None:
                raise ValueError(_STORAGE)
            with open(_local(path), "rb") as handle:
                data = handle.read()
    elif hasattr(source, "read"):
        if o.storage_options is not None:
            raise ValueError(_STORAGE)
        method, _ = _method(None, o.compression)
        data = source.read()
    else:
        raise ValueError(f"Invalid file path or buffer object type: {type(source)}")
    if isinstance(data, str):
        return data[1:] if data.startswith("﻿") else data
    data = _unpacked(data, method)
    encoding = o.encoding or "utf-8"
    if str(encoding).lower().replace("-", "_") in ("utf_8", "utf8"):
        encoding = "utf-8-sig"
    return data.decode(encoding, o.encoding_errors)


def _fetched(url: str, storage_options: Any) -> bytes:
    import urllib.request

    scheme = url.split("://", 1)[0].lower()
    if scheme not in ("http", "https", "ftp", "file"):
        raise ImportError(
            "`Import fsspec` failed.  Use pip or conda to install the fsspec package."
        )
    request = urllib.request.Request(url, headers=dict(storage_options or {}))
    with urllib.request.urlopen(request) as response:
        return response.read()


Record = tuple[int, list[str], bool]
"""A record's last line, its fields, and whether it was nothing but a comment."""


def _records(text: str, o: _Options) -> list[Record]:
    """The text split into records, before any row is skipped.

    Each record is numbered by its place among the records, which is the line number
    pandas' C parser puts in its messages, so a quoted field over two lines counts once.
    """
    return [(at, fields, commented) for at, (_, fields, commented) in enumerate(_split(text, o), 1)]


def _split(text: str, o: _Options) -> list[Record]:
    if o.regex:
        return _split_by_pattern(text, o)
    if o.whitespace or o.comment is not None or o.lineterminator is not None:
        return _tokenised(text, o)
    separator = o.sep
    if separator is None:
        separator = _sniffed(text, o)
    csv.field_size_limit(max(csv.field_size_limit(), _FIELD_LIMIT))
    quoting = csv.QUOTE_NONE if o.quoting == csv.QUOTE_NONE else csv.QUOTE_MINIMAL
    reader = csv.reader(
        io.StringIO(text, newline=""),
        delimiter=separator,
        quotechar=o.quotechar or '"',
        quoting=quoting,
        doublequote=o.doublequote,
        escapechar=o.escapechar,
        skipinitialspace=o.skipinitialspace,
        strict=False,
    )
    return [(reader.line_num, row, False) for row in reader]


def _sniffed(text: str, o: _Options) -> str:
    for line in _LINES.split(text):
        if line.strip():
            return csv.Sniffer().sniff(line).delimiter
    raise EmptyDataError("No columns to parse from file")


def _split_by_pattern(text: str, o: _Options) -> list[Record]:
    """The python engine's reading with a regex separator, which ignores quotes."""
    pattern = re.compile(o.sep)
    lines = _LINES.split(text)
    if lines and lines[-1] == "":
        lines.pop()
    out: list[Record] = []
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        fields = pattern.split(stripped) if stripped else []
        commented = False
        if o.comment is not None:
            for at, field in enumerate(fields):
                if o.comment in field:
                    fields = [*fields[:at], field[: field.index(o.comment)]]
                    break
            if fields == [""]:
                fields, commented = [], True
        out.append((number, fields, commented))
    return out


_START_RECORD, _START_FIELD, _IN_FIELD, _IN_QUOTED, _QUOTE_IN_QUOTED = range(5)
_ESCAPED, _ESCAPED_QUOTED, _COMMENT = range(5, 8)


def _tokenised(text: str, o: _Options) -> list[Record]:
    """pandas' C tokeniser, for a comment, a line terminator or whitespace separators.

    It keeps the states the C tokeniser keeps. A quote opens a quoted field only at the
    start of a field, a doubled quote inside one is a quote, an escape character takes
    the next character as it is, and a comment character outside quotes ends the
    record. A record that is nothing but a comment is marked, because pandas drops it
    even when blank lines are kept.
    """
    whitespace = o.whitespace
    delimiter = None if whitespace else o.sep
    quote = None if o.quoting == csv.QUOTE_NONE else o.quotechar
    escape = o.escapechar
    comment = o.comment
    terminator = o.lineterminator
    doubled = o.doublequote
    skip_space = o.skipinitialspace
    out: list[Record] = []
    fields: list[str] = []
    field: list[str] = []
    state = _START_RECORD
    line = 1
    at = 0
    size = len(text)
    while at < size:
        char = text[at]
        at += 1
        if terminator is not None:
            end = char == terminator
        else:
            end = char in "\r\n"
            if char == "\r" and at < size and text[at] == "\n":
                char = "\r\n"
                at += 1
        if state == _COMMENT:
            if end:
                out.append((line, fields, not fields))
                fields, field, state = [], [], _START_RECORD
                line += 1
            continue
        if state == _ESCAPED:
            field.append(char)
            state = _IN_FIELD
            line += end
            continue
        if state == _ESCAPED_QUOTED:
            field.append(char)
            state = _IN_QUOTED
            line += end
            continue
        if state == _IN_QUOTED:
            if escape is not None and char == escape:
                state = _ESCAPED_QUOTED
            elif char == quote:
                state = _QUOTE_IN_QUOTED
            else:
                field.append(char)
                line += end
            continue
        if state == _QUOTE_IN_QUOTED:
            if doubled and char == quote:
                field.append(char)
                state = _IN_QUOTED
                continue
            state = _IN_FIELD
        if end:
            if state == _START_RECORD:
                out.append((line, [], False))
            else:
                if not (whitespace and state == _START_FIELD):
                    fields.append("".join(field))
                out.append((line, fields, False))
            fields, field, state = [], [], _START_RECORD
            line += 1
            continue
        if comment is not None and char == comment:
            if state != _START_RECORD and not (whitespace and state == _START_FIELD):
                fields.append("".join(field))
            field = []
            state = _COMMENT
            continue
        if whitespace and char in " \t":
            if state == _IN_FIELD:
                fields.append("".join(field))
                field = []
                state = _START_FIELD
            continue
        if char == delimiter:
            fields.append("".join(field))
            field = []
            state = _START_FIELD
            continue
        if state in (_START_RECORD, _START_FIELD):
            if skip_space and char == " ":
                state = _START_FIELD
                continue
            if quote is not None and char == quote:
                state = _IN_QUOTED
                continue
            if escape is not None and char == escape:
                state = _ESCAPED
                continue
            field.append(char)
            state = _IN_FIELD
            continue
        if escape is not None and char == escape:
            state = _ESCAPED
            continue
        field.append(char)
    if state == _COMMENT:
        out.append((line, fields, not fields))
    elif state != _START_RECORD:
        if not (whitespace and state == _START_FIELD):
            fields.append("".join(field))
        out.append((line, fields, False))
    return out


def _fixed_records(text: str, o: _Options, colspecs: Any, widths: Any, rows: int) -> list[Record]:
    """The lines cut at fixed positions, as `read_fwf` cuts them."""
    filler = "\r\n" + o.fill if o.fill else "\n\r\t "
    lines = _LINES.split(text)
    if lines and lines[-1] == "":
        lines.pop()
    if widths is not None:
        if colspecs not in ("infer", None):
            raise ValueError("You must specify only one of 'widths' and 'colspecs'")
        colspecs, start = [], 0
        for width in widths:
            colspecs.append((start, start + width))
            start += width
    elif colspecs == "infer":
        colspecs = _inferred_colspecs(lines, o, filler, rows)
    out: list[Record] = []
    for number, line in enumerate(lines, 1):
        if o.comment is not None and o.comment in line:
            line = line.partition(o.comment)[0]
            if not line.strip(filler):
                out.append((number, [], True))
                continue
        if not line.strip(filler):
            out.append((number, [], False))
            continue
        out.append((number, [line[start:stop].strip(filler) for start, stop in colspecs], False))
    return out


def _inferred_colspecs(lines: list[str], o: _Options, filler: str, rows: int) -> list[Any]:
    skip = o.skiprows
    kept = []
    for number, line in enumerate(lines):
        if len(kept) >= rows:
            break
        if skip is not None and _skipped(skip, number):
            continue
        kept.append(line)
    if not kept:
        raise EmptyDataError("No rows from which to infer column width")
    if o.comment is not None:
        kept = [line.partition(o.comment)[0] for line in kept]
    pattern = re.compile("[^" + "".join("\\" + char for char in filler) + "]+")
    mask = [False] * (max(map(len, kept)) + 1)
    for line in kept:
        for match in pattern.finditer(line):
            for at in range(match.start(), match.end()):
                mask[at] = True
    edges = [at for at in range(len(mask)) if mask[at] != (mask[at - 1] if at else False)]
    return list(zip(edges[::2], edges[1::2], strict=False))


def _skipped(skip: Any, number: int) -> bool:
    if callable(skip):
        return bool(skip(number))
    if isinstance(skip, int) and not isinstance(skip, bool):
        return number < skip
    return number in skip


def _blank(fields: list[str]) -> bool:
    return not fields or (len(fields) == 1 and fields[0] != "" and not fields[0].strip(" \t"))


def _mangled(names: list[str]) -> list[str]:
    """Repeated names numbered as pandas' C parser numbers them.

    The second `a` becomes `a.1`, unless the header already has an `a.1`, in which
    case it moves on to `a.2`, so `a,a,a.1` reads as `a`, `a.2` and `a.1`.
    """
    present = set(names)
    counts: dict[str, int] = {}
    out = []
    for name in names:
        original = name
        count = counts.get(name, 0)
        while count > 0:
            counts[original] = count + 1
            name = f"{original}.{count}"
            count = count + 1 if name in present else counts.get(name, 0)
        out.append(name)
        counts[name] = count + 1
    return out


class _Layout:
    """What the header decided: the names, the rows under them and the row labels."""

    def __init__(self, names: list[str], rows: list[list[str]], leading: int) -> None:
        self.names = names
        self.rows = rows
        self.leading = leading
        # Names counted from 0 because the file gave none, which pandas holds as numbers.
        self.numbered = False


def _laid_out(records: list[Record], o: _Options) -> _Layout:
    """Skipped rows, blank lines, the header, the names and the rows that do not fit."""
    if o.skiprows is not None:
        records = [record for at, record in enumerate(records) if not _skipped(o.skiprows, at)]
    rows: list[tuple[int, list[str]]] = []
    for line, fields, commented in records:
        if commented or (o.skip_blank_lines and _blank(fields)):
            continue
        rows.append((line, fields))
    header = o.header
    if header == "infer":
        header = 0 if o.names is None else None
    file_names = None
    if header is not None:
        if header >= len(rows):
            if not rows and o.names is None:
                raise EmptyDataError("No columns to parse from file")
            if rows:
                raise ParserError(f"Passed header={header} but only {len(rows)} lines in file")
        else:
            file_names = rows[header][1]
            rows = rows[header + 1 :]
    if o.names is not None:
        names = list(o.names)
    elif file_names is not None:
        names = _mangled(
            [name if name != "" else f"Unnamed: {at}" for at, name in enumerate(file_names)]
        )
    else:
        if not rows:
            raise EmptyDataError("No columns to parse from file")
        names = [str(at) for at in range(len(rows[0][1]))]
    if o.skipfooter:
        rows = rows[: -o.skipfooter] if o.skipfooter < len(rows) else []
    if o.nrows is not None:
        rows = rows[: o.nrows]
    leading = 0
    if o.index_col is not False and rows and len(rows[0][1]) > len(names):
        leading = len(rows[0][1]) - len(names)
    expected = len(names) + leading
    kept = []
    truncated = False
    for line, fields in rows:
        if len(fields) > expected:
            if o.index_col is False:
                fields = fields[:expected]
                truncated = True
            else:
                fields = _bad(line, fields, expected, o)
                if fields is None:
                    continue
        kept.append(fields)
    if truncated:
        warnings.warn(
            "Length of header or names does not match length of data. This leads to a loss"
            " of data with index_col=False.",
            ParserWarning,
            stacklevel=4,
        )
    layout = _Layout(names, kept, leading)
    layout.numbered = o.names is None and file_names is None
    return layout


def _bad(line: int, fields: list[str], expected: int, o: _Options) -> list[str] | None:
    bad = o.on_bad_lines
    if callable(bad):
        fixed = bad(fields)
        return None if fixed is None else list(fixed)[:expected]
    if bad == "skip":
        return None
    if bad == "warn":
        warnings.warn(
            f"Skipping line {line}: expected {expected} fields, saw {len(fields)}\n",
            ParserWarning,
            stacklevel=5,
        )
        return None
    if o.python:
        message = f"Expected {expected} fields in line {line}, saw {len(fields)}"
        if o.regex and o.quoting != csv.QUOTE_NONE:
            message += (
                ". Error could possibly be due to quotes being ignored when a multi-char"
                " delimiter is used."
            )
        raise ParserError(message)
    raise ParserError(
        f"Error tokenizing data. C error: Expected {expected} fields in line {line},"
        f" saw {len(fields)}\n"
    )


def _na_rules(o: _Options) -> Callable[[str, int], tuple[frozenset[str], frozenset[float]]]:
    """Which text is missing in a column, and which numbers, as pandas cleans them."""
    if not o.na_filter:
        empty: tuple[frozenset[str], frozenset[float]] = (frozenset(), frozenset())
        return lambda name, at: empty
    values = o.na_values
    keep = o.keep_default_na
    if values is None:
        texts = DEFAULT_NA if keep else frozenset()
        fixed = (texts, frozenset())
        return lambda name, at: fixed
    if isinstance(values, dict):
        by_column = {}
        for key, given in values.items():
            listed = list(given) if _listed(given) else [given]
            texts = _stringified(listed) | (DEFAULT_NA if keep else frozenset())
            by_column[key] = (frozenset(texts), _floats(texts))
        other = (DEFAULT_NA if keep else frozenset(), frozenset())

        def rule(name: str, at: int) -> tuple[frozenset[str], frozenset[float]]:
            if name in by_column:
                return by_column[name]
            return by_column.get(at, other)

        return rule
    listed = list(values) if _listed(values) else [values]
    texts = _stringified(listed) | (DEFAULT_NA if keep else frozenset())
    fixed = (frozenset(texts), _floats(texts))
    return lambda name, at: fixed


def _listed(value: Any) -> bool:
    return not isinstance(value, (str, bytes)) and hasattr(value, "__iter__")


def _stringified(values: list[Any]) -> set[str]:
    out = set()
    for value in values:
        out.add(str(value))
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isnan(number) and not math.isinf(number) and number == int(number):
            out.add(f"{int(number)}.0")
            out.add(str(int(number)))
    return out


def _floats(texts: Any) -> frozenset[float]:
    out = set()
    for text in texts:
        try:
            number = float(text)
        except (TypeError, ValueError):
            continue
        if not math.isnan(number):
            out.add(number)
    return frozenset(out)


class _Numbers:
    """pandas' C parser's reading of an integer and a float, with the separators."""

    def __init__(self, o: _Options) -> None:
        self.thousands = o.thousands
        self.decimal = o.decimal
        self.plain = o.thousands is None and o.decimal == "."
        self.nonnumeric = o.quoting == csv.QUOTE_NONNUMERIC
        if not self.plain:
            group = re.escape(o.thousands) if o.thousands else None
            digits = f"[0-9]+(?:{group}[0-9]+)*" if group else "[0-9]+"
            point = re.escape(o.decimal)
            self.integer = re.compile(rf"\s*[+-]?{digits}\s*")
            self.real = re.compile(
                rf"\s*[+-]?(?:{digits}(?:{point}[0-9]*)?|{point}[0-9]+)(?:[eE][+-]?[0-9]+)?\s*"
            )
        self.infinity = re.compile(r"\s*[+-]?inf(?:inity)?\s*", re.IGNORECASE)

    def to_int(self, text: str) -> int:
        if self.plain:
            if not text.isascii() or "_" in text:
                raise ValueError(text)
            return int(text)
        if not self.integer.fullmatch(text):
            raise ValueError(text)
        return int(text.replace(self.thousands, "") if self.thousands else text)

    def to_float(self, text: str) -> float:
        if self.plain:
            if not text.isascii() or "_" in text or text.strip().lower() in _NAN:
                raise ValueError(text)
            return float(text)
        if self.infinity.fullmatch(text):
            return float(text)
        if not self.real.fullmatch(text):
            raise ValueError(text)
        if self.thousands:
            text = text.replace(self.thousands, "")
        return float(text.replace(self.decimal, "."))


class _Column:
    """A typed column on its way to Arrow, and what is done to it once it is a column."""

    def __init__(self, fmt: str, values: list[Any], after: Any = None) -> None:
        self.fmt = fmt
        self.values = values
        self.after = after


def _kind(dtype: Any) -> str:
    if dtype in (str, object):
        return "text"
    if dtype is int:
        return "int64"
    if dtype is float:
        return "float64"
    if dtype is bool:
        return "bool"
    name = str(dtype)
    if name in _TEXT_NAMES or name.startswith("<U"):
        return "text"
    if name.startswith("category") or type(dtype).__name__ == "CategoricalDtype":
        return "category"
    if name in _INT_NAMES:
        return name
    if name in _FLOAT_NAMES:
        return name
    if name == "bool":
        return "bool"
    return "other"


def _typed(
    raw: list[str | None],
    at: int,
    na: frozenset[str],
    fna: frozenset[float],
    converter: Any,
    dtype: Any,
    numbers: _Numbers,
    truths: tuple[frozenset[str], frozenset[str]],
) -> _Column:
    """One column typed the way pandas' C parser types it."""
    if converter is not None:
        return _objects([converter("" if value is None else value) for value in raw])
    values = [None if value is None or value in na else value for value in raw]
    if dtype is not None:
        kind = _kind(dtype)
        if kind == "text":
            return _Column("u", values)
        if kind == "category":
            return _Column("u", values, dtype)
        if kind.startswith(("int", "uint")):
            if any(value is None for value in values):
                raise ValueError(f"Integer column has NA values in column {at}")
            parsed = []
            for value in values:
                try:
                    parsed.append(numbers.to_int(value))  # type: ignore[arg-type]
                except ValueError:
                    raise ValueError(f"invalid literal for int() with base 10: {value!r}") from None
            fmt = "L" if kind == "uint64" else "l"
            return _Column(fmt, parsed, None if kind in ("int64", "uint64") else kind)
        if kind.startswith("float"):
            floats = []
            for value in values:
                if value is None:
                    floats.append(None)
                    continue
                try:
                    floats.append(numbers.to_float(value))
                except ValueError:
                    raise ValueError(f"could not convert string to float: {value!r}") from None
            return _Column("g", floats, None if kind == "float64" else kind)
        if kind == "bool":
            return _Column("b", [None if value is None else value in truths[0] for value in values])
        column = _inferred(values, fna, numbers, truths)
        column.after = dtype
        return column
    return _inferred(values, fna, numbers, truths)


def _inferred(
    values: list[str | None],
    fna: frozenset[float],
    numbers: _Numbers,
    truths: tuple[frozenset[str], frozenset[str]],
) -> _Column:
    present = [value for value in values if value is not None]
    if not values:
        return _Column("u", [])
    if not present:
        return _Column("g", values)
    try:
        integers = [None if value is None else numbers.to_int(value) for value in values]
    except ValueError:
        integers = None
    if integers is not None:
        if fna:
            integers = [
                None if value is None or float(value) in fna else value for value in integers
            ]
        if any(value is None for value in integers):
            return _Column("g", [None if value is None else float(value) for value in integers])
        if numbers.nonnumeric:
            return _Column("g", [float(value) for value in integers])  # type: ignore[arg-type]
        low, high = min(integers), max(integers)  # type: ignore[type-var]
        if _INT64[0] <= low and high <= _INT64[1]:
            return _Column("l", integers)
        if low >= 0 and high <= _UINT64:
            return _Column("L", integers)
        return _Column("u", values)
    try:
        floats = [None if value is None else numbers.to_float(value) for value in values]
    except ValueError:
        floats = None
    if floats is not None:
        if fna:
            floats = [None if value in fna else value for value in floats]
        return _Column("g", floats)
    truth, falsity = truths
    if all(value in truth or value in falsity for value in present):
        return _Column("b", [None if value is None else value in truth for value in values])
    return _Column("u", values)


def _objects(values: list[Any]) -> _Column:
    """A converter's answers typed by what they are."""
    kept = [
        None if value is None or (isinstance(value, float) and math.isnan(value)) else value
        for value in values
    ]
    present = [value for value in kept if value is not None]
    if not present:
        return _Column("g" if kept else "u", kept)
    if all(isinstance(value, bool) for value in present):
        return _Column("b", kept)
    if all(isinstance(value, int) and not isinstance(value, bool) for value in present):
        if len(present) == len(kept) and all(_INT64[0] <= value <= _INT64[1] for value in present):
            return _Column("l", kept)
        return _Column("g", [None if value is None else float(value) for value in kept])
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in present):
        return _Column("g", [None if value is None else float(value) for value in kept])
    return _Column("u", [None if value is None else str(value) for value in kept])


def _bitmap(flags: list[bool]) -> bytes:
    out = bytearray((len(flags) + 7) // 8)
    for at, flag in enumerate(flags):
        if flag:
            out[at >> 3] |= 1 << (at & 7)
    return bytes(out)


def _arrow(fmt: str, values: list[Any]) -> tuple[str, dict[str, Any]]:
    """One column's Arrow format and array, built from Python values."""
    nulls = sum(value is None for value in values)
    validity = _bitmap([value is not None for value in values]) if nulls else None
    if fmt == "u":
        encoded = [b"" if value is None else value.encode("utf-8", "replace") for value in values]
        total = sum(map(len, encoded))
        large = total > _FIELD_LIMIT
        offsets = array("q" if large else "i", [0])
        running = 0
        for data in encoded:
            running += len(data)
            offsets.append(running)
        buffers = [validity, offsets.tobytes(), b"".join(encoded)]
        fmt = "U" if large else "u"
    elif fmt == "b":
        buffers = [validity, _bitmap([bool(value) for value in values])]
    else:
        code = {"l": "q", "L": "Q", "g": "d"}[fmt]
        zero = 0.0 if fmt == "g" else 0
        buffers = [
            validity,
            array(code, [zero if value is None else value for value in values]).tobytes(),
        ]
    return fmt, {
        "length": len(values),
        "null_count": nulls,
        "offset": 0,
        "buffers": buffers,
        "children": [],
        "dictionary": None,
    }


def _exported(names: list[str], columns: list[_Column], length: int) -> Any:
    from ._pickle import _Export

    schemas, arrays = [], []
    for name, column in zip(names, columns, strict=True):
        fmt, built = _arrow(column.fmt, column.values)
        schemas.append(
            {
                "format": fmt.encode(),
                "name": name.encode(),
                "metadata": None,
                "flags": 2,
                "children": [],
                "dictionary": None,
            }
        )
        arrays.append(built)
    schema = {
        "format": b"+s",
        "name": b"",
        "metadata": None,
        "flags": 0,
        "children": schemas,
        "dictionary": None,
    }
    table = {
        "length": length,
        "null_count": 0,
        "offset": 0,
        "buffers": [None],
        "children": arrays,
        "dictionary": None,
    }
    return _Export(schema, table)


def _keyed(mapping: Any, name: str, at: int) -> Any:
    if not isinstance(mapping, dict):
        return mapping
    if name in mapping:
        return mapping[name]
    if at in mapping:
        return mapping[at]
    factory = getattr(mapping, "default_factory", None)
    return factory() if factory is not None else None


_IMPLICIT = "__firepanda_index_{}__"


class _Plan:
    """Which columns are kept, which become the row labels, and how each is typed."""

    def __init__(self, layout: _Layout, o: _Options) -> None:
        from ._pandas import _usecols

        names = layout.names
        self.leading = layout.leading
        implicit = [_IMPLICIT.format(k) for k in range(layout.leading)]
        used = list(names) if o.usecols is None else _usecols(names, o.usecols)
        self.positions = list(range(layout.leading)) + [
            layout.leading + names.index(name) for name in used
        ]
        self.names = implicit + used
        self.numbered = layout.numbered
        self.original = list(range(layout.leading)) + [names.index(name) for name in used]
        if implicit:
            self.index = list(implicit)
        elif o.index_col is None or o.index_col is False:
            self.index = []
        else:
            wanted = o.index_col if isinstance(o.index_col, (list, tuple)) else [o.index_col]
            self.index = []
            for one in wanted:
                if isinstance(one, int) and not isinstance(one, bool):
                    if not 0 <= one < len(used):
                        raise IndexError("list index out of range")
                    self.index.append(used[one])
                elif one in used:
                    self.index.append(one)
                else:
                    raise ValueError(f"Index {one} invalid")
        if len(self.index) > 1:
            raise NotImplementedError(
                "index_col with more than one column is not supported yet, because the"
                " result is a MultiIndex and a frame cannot hold one yet"
            )
        dates = set()
        for one in o.dates:
            if isinstance(one, int) and not isinstance(one, bool):
                if 0 <= one < len(used):
                    dates.add(used[one])
            elif one in self.names:
                dates.add(one)
            else:
                raise ValueError(f"Missing column provided to 'parse_dates': '{one}'")
        if o.date_index:
            dates.update(self.index)
        self.dates = dates


def _built(layout_rows: list[list[str]], plan: _Plan, o: _Options, start: int) -> Any:
    """The frame for some of the rows, typed as pandas types them."""
    from ._frame import DataFrame, Series, from_arrow
    from ._pandas import to_datetime
    from ._range_index import RangeIndex

    width = max(plan.positions, default=-1) + 1
    length = len(layout_rows)
    padded = [row + [None] * (width - len(row)) if len(row) < width else row for row in layout_rows]
    by_column = list(zip(*padded, strict=False)) if padded else [() for _ in range(width)]
    na_rule = _na_rules(o)
    numbers = _Numbers(o)
    truths = (
        _TRUE | frozenset(o.true_values or ()),
        _FALSE | frozenset(o.false_values or ()),
    )
    columns = []
    dated: dict[str, Any] = {}
    for name, position, original in zip(plan.names, plan.positions, plan.original, strict=True):
        raw = list(by_column[position]) if position < len(by_column) else [None] * length
        na, fna = na_rule(name, original)
        converter = _keyed(o.converters, name, original) if o.converters else None
        dtype = _keyed(o.dtype, name, original) if o.dtype is not None else None
        if converter is not None and dtype is not None:
            warnings.warn(
                f"Both a converter and dtype were specified for column {name} - only the"
                " converter will be used.",
                ParserWarning,
                stacklevel=4,
            )
            dtype = None
        if name in plan.dates and converter is None:
            text = [None if value is None or value in na else value for value in raw]
            fmt = _keyed(o.date_format, name, original) if o.date_format is not None else None
            try:
                dated[name] = to_datetime(
                    Series(text, dtype="str"), format=fmt, dayfirst=o.dayfirst
                )
            except Exception:
                dated.pop(name, None)
        columns.append(_typed(raw, original, na, fna, converter, dtype, numbers, truths))
    if not plan.names:
        frame = DataFrame()
    else:
        export = _exported(plan.names, columns, length)
        frame = from_arrow(export) if o.backend else DataFrame.from_arrow(export)
    for name, column in zip(plan.names, columns, strict=True):
        if column.after is not None:
            frame[name] = frame[name].astype(column.after)
    for name, converted in dated.items():
        frame[name] = converted
    if plan.index:
        label = plan.index[0]
        frame = frame.set_index(label)
        if label.startswith("__firepanda_index_") or "Unnamed" in label:
            frame = frame.rename_axis(None)
    elif start and plan.names:
        frame = frame.set_axis(RangeIndex(start, start + length))
    if plan.numbered:
        frame = _numbered_names(frame, plan)
    return frame


def _numbered_names(frame: Any, plan: _Plan) -> Any:
    """The frame with the names the file did not give turned from text into numbers."""
    frame = frame.rename(columns={name: int(name) for name in plan.names if name.isdigit()})
    if plan.index and plan.index[0].isdigit():
        frame = frame.rename_axis(int(plan.index[0]))
    return frame


class TextFileReader:
    """What `read_csv` hands back for `iterator=True` or a `chunksize`, as pandas does.

    Each chunk's types are picked from its own rows, which is what pandas' C parser
    does, so a column that turns to text late in a file is numbers in the chunks before
    it. The row labels carry on from one chunk to the next.
    """

    def __init__(self, rows: list[list[str]], plan: _Plan, o: _Options) -> None:
        self._rows = rows
        self._plan = plan
        self._options = o
        self._at = 0
        self.chunksize = o.chunksize

    def __iter__(self) -> TextFileReader:
        return self

    def __next__(self) -> Any:
        if self._at >= len(self._rows):
            raise StopIteration
        return self.get_chunk()

    def read(self, nrows: int | None = None) -> Any:
        """The next `nrows` rows as a frame, or all that are left."""
        if self._at >= len(self._rows) and self._at:
            raise StopIteration
        stop = len(self._rows) if nrows is None else min(len(self._rows), self._at + nrows)
        frame = _built(self._rows[self._at : stop], self._plan, self._options, self._at)
        self._at = stop
        return frame

    def get_chunk(self, size: int | None = None) -> Any:
        """The next chunk, of `size` rows or the chunk size."""
        return self.read(self.chunksize if size is None else size)

    def close(self) -> None:
        """Nothing is held open, so this only ends the reading."""
        self._at = len(self._rows)

    def __enter__(self) -> TextFileReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _finished(records: list[Record], o: _Options) -> Any:
    layout = _laid_out(records, o)
    plan = _Plan(layout, o)
    if o.iterator or o.chunksize is not None:
        return TextFileReader(layout.rows, plan, o)
    return _built(layout.rows, plan, o, 0)


def _plain(values: dict[str, Any], separator: Any) -> bool:
    """Whether the core reader gives pandas' answer for these arguments."""
    defaults = _defaults()
    free = {
        "usecols",
        "index_col",
        "engine",
        "cache_dates",
        "low_memory",
        "memory_map",
        "float_precision",
        "encoding",
        "sep",
        "delimiter",
        "compression",
    }
    for name, default in defaults.items():
        if name in free:
            continue
        if values[name] is not default and values[name] != default:
            return False
    if separator != "," or callable(values["usecols"]):
        return False
    encoding = values["encoding"]
    return encoding is None or str(encoding).lower() in _UTF8


def _core(source: Any, values: dict[str, Any], separator: Any) -> Any:
    """The core reader's frame when it is pandas' frame, or None to read it here."""
    from ._frame import _read_csv
    from ._pickle import _method
    from .errors import ReaderError

    if not isinstance(source, (str, os.PathLike)) or not _plain(values, separator):
        return None
    path = os.fspath(source)
    if not isinstance(path, str) or "://" in path:
        return None
    path = _local(path)
    if not os.path.isfile(path) or _method(path, values["compression"])[0] is not None:
        return None
    with open(path, "rb") as handle:
        head = handle.read(1 << 16)
    if not head.strip() or head.startswith(b"\xef\xbb\xbf"):
        return None
    first = next(csv.reader(io.StringIO(head.decode("utf-8", "replace"), newline="")), [])
    if not first or "" in first or len(set(first)) != len(first):
        return None
    try:
        frame = _read_csv(path)
    except ReaderError:
        return None
    for name in list(frame.columns):
        column = frame[name]
        if not len(column):
            continue
        if str(column.dtype) == "float64":
            if not column.isna().any() and column.abs().max() >= 2.0**63:
                return None
            continue
        if str(column.dtype) != "string":
            continue
        if column.isna().all():
            frame[name] = column.astype("float64")
            continue
        filled = column.fillna("")
        if (filled.str.strip() != filled).any():
            return None
    return frame


def read_csv(source: Any, values: dict[str, Any], separator: Any) -> Any:
    """`pandas.read_csv` once `_pandas.read_csv` has checked what it checks first."""
    from ._frame import DataFrame
    from ._pandas import _usecols

    frame = _core(source, values, separator)
    if frame is not None:
        usecols, index_col = values["usecols"], values["index_col"]
        if usecols is not None:
            frame = frame[_usecols(frame.columns, usecols)]
        if index_col is not None and index_col is not False:
            if isinstance(index_col, (list, tuple)):
                if len(index_col) != 1:
                    raise NotImplementedError(
                        "index_col with more than one column is not supported yet, because"
                        " the result is a MultiIndex and a frame cannot hold one yet"
                    )
                index_col = index_col[0]
            if isinstance(index_col, int) and not isinstance(index_col, bool):
                index_col = list(frame.columns)[index_col]
            frame = frame.set_index(index_col)
        assert isinstance(frame, DataFrame)
        return frame
    o = _checked(values, separator)
    return _finished(_records(_text(source, o), o), o)


def read_fwf(
    filepath_or_buffer: Any,
    *,
    colspecs: Any = "infer",
    widths: Any = None,
    infer_nrows: int = 100,
    iterator: bool = False,
    chunksize: Any = None,
    **kwds: Any,
) -> Any:
    """Reads fixed width columns into a frame, which is `pandas.read_fwf`.

    Each line is cut at the column positions, given as `colspecs` pairs or as `widths`,
    or found from the first `infer_nrows` lines as the spans that are never blank. Each
    piece has the filler characters stripped from both ends, spaces and tabs unless
    `delimiter` names others. The pieces then go through every step `read_csv` takes, so
    the header, the names, the missing values and the types are read the same way.

    Args:
        filepath_or_buffer: A path, a URL, or a handle open for text or bytes.
        colspecs: Pairs of start and stop positions, a stop of None meaning the end of
            the line, or `infer`.
        widths: The width of each column, instead of `colspecs`.
        infer_nrows: How many lines `infer` looks at.
        iterator: Hand back a `TextFileReader` instead of a frame.
        chunksize: Hand back a `TextFileReader` that reads this many rows at a time.
        **kwds: Any other `read_csv` argument. Anything else is ignored, as pandas
            ignores it.

    Returns:
        The frame, or a `TextFileReader`.

    Raises:
        ValueError: For both `widths` and `colspecs`.
        EmptyDataError: When there are no lines to infer the columns from.
    """
    values = _defaults()
    values.update((name, value) for name, value in kwds.items() if name in values)
    values["iterator"] = iterator
    values["chunksize"] = chunksize
    fill = values.pop("delimiter")
    values["engine"] = "python"
    values.pop("sep")
    o = _checked(values, ",")
    o.fill = fill
    text = _text(filepath_or_buffer, o)
    return _finished(_fixed_records(text, o, colspecs, widths, infer_nrows), o)


def read_table(filepath_or_buffer: Any, **kwargs: Any) -> Any:
    """Reads a file of tab separated values into a frame, which is `pandas.read_table`.

    This is `read_csv` with a tab as the default separator, and it takes every argument
    `read_csv` takes, with the same defaults otherwise.

    Args:
        filepath_or_buffer: A path, a URL, or a handle open for text or bytes.
        **kwargs: Any `read_csv` argument.

    Returns:
        The frame, or a `TextFileReader`.
    """
    from ._pandas import NO_DEFAULT, read_csv

    if kwargs.get("sep", NO_DEFAULT) is NO_DEFAULT and kwargs.get("delimiter") is None:
        kwargs["sep"] = "\t"
    return read_csv(filepath_or_buffer, **kwargs)


def _signed_like_read_csv() -> None:
    from ._pandas import read_csv

    read_table.__signature__ = inspect.signature(read_csv)  # type: ignore[attr-defined]


_signed_like_read_csv()
