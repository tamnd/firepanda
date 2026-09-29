"""`to_sql`, `read_sql`, `read_sql_query` and `read_sql_table` over a sqlite3 connection.

pandas talks to a database through SQLAlchemy when it is installed and falls
back to a plain sqlite3 connection when it is not, writing the SQL itself.
This is that fallback, statement for statement: the same `CREATE TABLE` text
with the same column types, the same index on the row labels, the same
`INSERT` with one `?` per value, and on the way back the same reading of each
column's values into a type. A connection of any other kind is refused, since
without SQLAlchemy pandas refuses it too or only guesses at it.
"""

from __future__ import annotations

import datetime
import sqlite3
import warnings
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from ._pandas import NO_DEFAULT
from .errors import DatabaseError, InvalidArgumentError

if TYPE_CHECKING:
    from ._frame import DataFrame, Series

_SQL_TYPES = {
    "string": "TEXT",
    "floating": "REAL",
    "integer": "INTEGER",
    "datetime": "TIMESTAMP",
    "date": "DATE",
    "time": "TIME",
    "boolean": "INTEGER",
}
"""The SQLite type pandas writes for each kind of column, and TEXT for any other."""

_UNITS = ("D", "d", "h", "m", "s", "ms", "us", "ns")

_NANOSECONDS_PER = {"s": 10**9, "ms": 10**6, "us": 1000, "ns": 1}
"""The nanoseconds in each unit a span column has, since pandas writes a span as its count."""

_SPAN_GAP = -(2**63)
"""What pandas writes for a missing span, its gap's count, since its gap mask skips spans."""

_IF_EXISTS = ("fail", "replace", "append", "delete_rows")


def _adapt_time(value: datetime.time) -> str:
    return f"{value.hour:02d}:{value.minute:02d}:{value.second:02d}.{value.microsecond:06d}"


def _register_date_adapters() -> None:
    """The adapters and converters pandas registers with sqlite3 whenever it opens one."""
    sqlite3.register_adapter(datetime.time, _adapt_time)
    sqlite3.register_adapter(datetime.date, lambda value: value.isoformat())
    sqlite3.register_adapter(datetime.datetime, lambda value: value.isoformat(" "))
    sqlite3.register_converter("date", lambda value: datetime.date.fromisoformat(value.decode()))
    sqlite3.register_converter(
        "timestamp", lambda value: datetime.datetime.fromisoformat(value.decode())
    )


def _connection(con: Any) -> Any:
    """The connection to talk to, which must be sqlite3's.

    Raises:
        ImportError: For a URI, which needs SQLAlchemy, in pandas' words.
        NotImplementedError: For any other connection, which pandas reads
            through SQLAlchemy or ADBC.
    """
    if isinstance(con, sqlite3.Connection):
        _register_date_adapters()
        return con
    if isinstance(con, str):
        raise ImportError(
            "Using URI string without version '2.0.36' or newer of 'sqlalchemy' installed."
        )
    raise NotImplementedError(
        "only a sqlite3 connection is supported, since SQLAlchemy and ADBC connections"
        " go through libraries firepanda does not use"
    )


def _quoted(name: object) -> str:
    """A name as SQLite quotes an identifier, checked the way pandas checks it."""
    text = str(name)
    if not text:
        raise InvalidArgumentError("Empty table or column name specified")
    if "\x00" in text:
        raise InvalidArgumentError("SQLite identifier cannot contain NULs")
    return '"' + text.replace('"', '""') + '"'


def _execute(con: Any, sql: Any, params: Any = None) -> Any:
    """Runs one statement and answers its cursor, rolling back on pandas' error."""
    if not isinstance(sql, str):
        raise TypeError("Query must be a string unless using sqlalchemy.")
    cursor = con.cursor()
    try:
        cursor.execute(sql, *([] if params is None else [params]))
    except sqlite3.Error as error:
        try:
            con.rollback()
        except sqlite3.Error as inner:
            raise DatabaseError(
                f"Execution failed on sql: {sql}\n{error}\nunable to rollback"
            ) from inner
        raise DatabaseError(f"Execution failed on sql '{sql}': {error}") from error
    return cursor


def _has_table(con: Any, name: str) -> bool:
    query = "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name=?;"
    return len(_execute(con, query, [name]).fetchall()) > 0


def _missing(value: Any) -> bool:
    from ._na import NA
    from ._scalars import NaT

    return (
        value is None
        or value is NA
        or value is NaT
        or (isinstance(value, float) and value != value)
    )


def _inferred(dtype: str, values: list[Any]) -> str:
    """What pandas' `infer_dtype` calls a column, skipping its gaps."""
    for prefix, kind in (
        ("datetime64", "datetime64"),
        ("timedelta64", "timedelta64"),
        ("bool", "boolean"),
        ("boolean", "boolean"),
        ("int", "integer"),
        ("Int", "integer"),
        ("uint", "integer"),
        ("UInt", "integer"),
        ("float", "floating"),
        ("Float", "floating"),
        ("str", "string"),
        ("category", "categorical"),
    ):
        if dtype.startswith(prefix) and not dtype.startswith("interval"):
            return kind
    present = [value for value in values if not _missing(value)]
    if not present:
        return "empty"
    kinds = {type(value) for value in present}
    if kinds == {bool}:
        return "boolean"
    if all(isinstance(value, int) and not isinstance(value, bool) for value in present):
        return "integer"
    if all(isinstance(value, float) for value in present):
        return "floating"
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in present):
        return "mixed-integer-float"
    if all(isinstance(value, complex) for value in present):
        return "complex"
    if all(isinstance(value, str) for value in present):
        return "string"
    if all(isinstance(value, datetime.datetime) for value in present):
        return "datetime"
    if all(isinstance(value, datetime.date) for value in present):
        return "date"
    if all(isinstance(value, datetime.time) for value in present):
        return "time"
    return "mixed"


def _sql_type(name: Any, dtype: str, values: list[Any], chosen: dict[Any, str] | None) -> str:
    """The SQLite type pandas writes a column as, or the one `dtype=` chose for it."""
    if chosen and name in chosen:
        return chosen[name]
    kind = _inferred(dtype, values)
    if kind == "timedelta64":
        warnings.warn(
            "the 'timedelta' type is not supported, and will be written as integer values"
            " (ns frequency) to the database.",
            UserWarning,
            stacklevel=4,
        )
        kind = "integer"
    elif kind == "datetime64":
        kind = "datetime"
    elif kind == "empty":
        kind = "string"
    elif kind == "complex":
        raise InvalidArgumentError("Complex datatypes not supported")
    return _SQL_TYPES.get(kind, "TEXT")


def _written(dtype: str, values: list[Any]) -> list[Any]:
    """Each value as pandas hands it to sqlite3: None for a gap, text for an instant."""
    out: list[Any] = []
    spans = dtype.startswith("timedelta64")
    unit = dtype[dtype.find("[") + 1 : -1] if spans and "[" in dtype else "ns"
    per = _NANOSECONDS_PER.get(unit, 1)
    for value in values:
        if spans and _missing(value):
            out.append(_SPAN_GAP)
        elif _missing(value):
            out.append(None)
        elif spans and isinstance(value, datetime.timedelta):
            nanoseconds = getattr(value, "value", None)
            if nanoseconds is None:
                nanoseconds = value // datetime.timedelta(microseconds=1) * 1000
            out.append(int(nanoseconds) // per)
        elif isinstance(value, datetime.datetime):
            plain = value.to_pydatetime() if hasattr(value, "to_pydatetime") else value
            out.append(plain.isoformat(" "))
        elif isinstance(value, datetime.date):
            out.append(value.isoformat())
        elif isinstance(value, datetime.time):
            out.append(_adapt_time(value))
        else:
            out.append(value)
    return out


def _index_labels(frame: DataFrame, index_label: Any) -> list[Any]:
    """The names the row labels are written under, as pandas picks them."""
    index = frame.index
    levels = index.nlevels
    if index_label is not None:
        labels = index_label if isinstance(index_label, list) else [index_label]
        if len(labels) != levels:
            raise InvalidArgumentError(
                f"Length of 'index_label' should match number of levels, which is {levels}"
            )
        return labels
    if levels == 1 and "index" not in list(frame.columns) and index.name is None:
        return ["index"]
    names = list(index.names)
    return [f"level_{n}" if name is None else name for n, name in enumerate(names)]


def to_sql(
    frame: DataFrame | Series,
    name: str,
    con: Any,
    schema: str | None = None,
    if_exists: str = "fail",
    index: bool = True,
    index_label: Any = None,
    chunksize: int | None = None,
    dtype: Any = None,
    method: Any = None,
) -> int | None:
    """Writes a frame or a column to a table, as pandas' sqlite3 fallback does.

    Returns:
        The number of rows sqlite3 counts as written.

    Raises:
        ValueError: For a table that is there under `if_exists="fail"`, and the
            other mistakes pandas names, in its words.
        NotImplementedError: For a callable `method`, which pandas hands its
            own table object.
    """
    from ._frame import DataFrame, Series

    if if_exists not in _IF_EXISTS:
        raise InvalidArgumentError(f"'{if_exists}' is not valid for if_exists")
    if isinstance(frame, Series):
        frame = frame.to_frame()
    elif not isinstance(frame, DataFrame):
        raise NotImplementedError("'frame' argument should be either a Series or a DataFrame")
    con = _connection(con)
    chosen: dict[Any, str] | None = None
    if dtype:
        chosen = dict(dtype) if isinstance(dtype, dict) else dict.fromkeys(frame.columns, dtype)
        for column, kind in chosen.items():
            if not isinstance(kind, str):
                raise InvalidArgumentError(f"{column} ({kind}) not a string")
    if method is not None and method != "multi":
        if callable(method):
            raise NotImplementedError(
                "a callable method is handed pandas' own SQLTable object, which firepanda"
                " does not have"
            )
        raise InvalidArgumentError(f"Invalid parameter `method`: {method}")

    columns: list[tuple[str, str, list[Any], bool]] = []
    if index:
        labels = _index_labels(frame, index_label)
        for number, label in enumerate(labels):
            level = frame.index.get_level_values(number) if frame.index.nlevels > 1 else frame.index
            values = level.tolist()
            kind = _sql_type(level.name, str(level.dtype), values, chosen)
            columns.append((str(label), kind, _written(str(level.dtype), values), True))
    for column in frame.columns:
        series = frame[column]
        values = series.tolist()
        kind = _sql_type(column, str(series.dtype), values, chosen)
        columns.append((str(column), kind, _written(str(series.dtype), values), False))

    if _has_table(con, name):
        if if_exists == "fail":
            raise InvalidArgumentError(f"Table '{name}' already exists.")
        if if_exists == "replace":
            _execute(con, f"DROP TABLE {_quoted(name)}").close()
            _create(con, name, schema, columns)
        elif if_exists == "delete_rows":
            _execute(con, f"DELETE FROM {_quoted(name)}").close()
    else:
        _create(con, name, schema, columns)
    held = [column for column, _, _, is_index in columns if not is_index]
    for label in [column for column, _, _, is_index in columns if is_index]:
        if label in held:
            raise InvalidArgumentError(
                f"duplicate name in index/columns: cannot insert {label}, already exists"
            )
    return _insert(con, name, columns, len(frame), chunksize, method)


def _create(
    con: Any, name: str, schema: str | None, columns: list[tuple[str, str, list[Any], bool]]
) -> None:
    """The `CREATE TABLE` and `CREATE INDEX` statements pandas writes, run in order."""
    body = ",\n  ".join(f"{_quoted(column)} {kind}" for column, kind, _, _ in columns)
    prefix = f"{schema}." if schema else ""
    statements = [f"CREATE TABLE {prefix}{_quoted(name)} (\n{body}\n)"]
    keyed = [column for column, _, _, is_index in columns if is_index]
    if keyed:
        listed = ",".join(_quoted(column) for column in keyed)
        index_name = _quoted("ix_" + name + "_" + "_".join(keyed))
        statements.append(f"CREATE INDEX {index_name}ON {_quoted(name)} ({listed})")
    cursor = con.cursor()
    try:
        for statement in statements:
            cursor.execute(statement)
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        cursor.close()


def _insert(
    con: Any,
    name: str,
    columns: list[tuple[str, str, list[Any], bool]],
    rows: int,
    chunksize: int | None,
    method: Any,
) -> int:
    """Inserts the rows in batches inside one transaction and answers sqlite3's count."""
    if rows == 0:
        return 0
    if chunksize is None:
        chunksize = rows
    elif chunksize == 0:
        raise InvalidArgumentError("chunksize argument should be non-zero")
    names = ",".join(_quoted(column) for column, _, _, _ in columns)
    one = "(" + ",".join("?" * len(columns)) + ")"
    data = list(zip(*(values for _, _, values, _ in columns), strict=True))
    total = 0
    cursor = con.cursor()
    try:
        for start in range(0, rows, chunksize):
            batch = data[start : start + chunksize]
            if method == "multi":
                marks = ",".join([one] * len(batch))
                cursor.execute(
                    f"INSERT INTO {_quoted(name)} ({names}) VALUES {marks}",
                    [value for row in batch for value in row],
                )
            else:
                try:
                    cursor.executemany(f"INSERT INTO {_quoted(name)} ({names}) VALUES {one}", batch)
                except sqlite3.Error as error:
                    raise DatabaseError("Execution failed") from error
            total += cursor.rowcount
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        cursor.close()
    return total


def _column(values: list[Any]) -> Series:
    """One column of a result, typed the way pandas reads a column of Python values.

    Whole numbers are int64, or float64 with NaN in a gap, numbers of both kinds
    are float64, text is `str` with NaN in a gap, and anything else, or a
    column of nothing but gaps, is objects.
    """
    from ._frame import Series

    present = [value for value in values if value is not None]
    gaps = len(present) != len(values)
    if present and all(isinstance(value, bool) for value in present):
        return Series(values, dtype=object if gaps else "bool")
    if present and all(
        isinstance(value, (int, float)) and not isinstance(value, bool) for value in present
    ):
        if not gaps and all(isinstance(value, int) for value in present):
            if all(-(2**63) <= value < 2**63 for value in present):
                return Series(values, dtype="int64")
            if all(0 <= value < 2**64 for value in present):
                return Series(values, dtype="uint64")
            return Series(values, dtype=object)
        return Series(
            [float("nan") if value is None else float(value) for value in values], dtype="float64"
        )
    if present and all(isinstance(value, str) for value in present):
        return Series(values, dtype="str")
    return Series(values, dtype=object)


def _framed(rows: list[Any], names: list[str]) -> DataFrame:
    """The rows of a result as a frame, one column at a time."""
    from ._frame import DataFrame, Series

    if len(set(names)) != len(names):
        raise NotImplementedError(
            "a result with two columns of one name, which a firepanda frame cannot hold"
        )
    if not rows:
        return DataFrame({name: Series([], dtype=object) for name in names})
    return DataFrame({name: _column([row[n] for row in rows]) for n, name in enumerate(names)})


def _dated(frame: DataFrame, parse_dates: Any) -> DataFrame:
    """The columns `parse_dates` names read as instants, as pandas reads them."""
    from ._pandas import to_datetime

    if parse_dates is True or parse_dates is None or parse_dates is False:
        parse_dates = []
    elif not hasattr(parse_dates, "__iter__"):
        parse_dates = [parse_dates]
    for name in list(frame.columns):
        if name not in parse_dates:
            continue
        try:
            how = parse_dates[name]
        except (KeyError, TypeError):
            how = None
        column = frame[name]
        if isinstance(how, dict):
            frame[name] = to_datetime(column, **how)
            continue
        if how is None and str(column.dtype).startswith(("int", "uint", "float")):
            how = "s"
        if how in _UNITS:
            frame[name] = to_datetime(column, errors="coerce", unit=how)
        else:
            frame[name] = to_datetime(column, errors="coerce", format=how)
    return frame


def _wrapped(
    rows: list[Any], names: list[str], index_col: Any, parse_dates: Any, dtype: Any
) -> DataFrame:
    frame = _framed(rows, names)
    if dtype:
        frame = frame.astype(dtype)
    frame = _dated(frame, parse_dates)
    if index_col is not None:
        frame = frame.set_index(index_col)
    return frame


def _chunks(
    cursor: Any,
    chunksize: int,
    names: list[str],
    index_col: Any,
    parse_dates: Any,
    dtype: Any,
) -> Iterator[DataFrame]:
    read = False
    while True:
        rows = cursor.fetchmany(chunksize)
        if not rows:
            cursor.close()
            if not read:
                empty = _framed([], names)
                yield empty.astype(dtype) if dtype else empty
            return
        read = True
        yield _wrapped(rows, names, index_col, parse_dates, dtype)


def _checked_backend(dtype_backend: Any) -> None:
    from ._pandas import _backend

    if dtype_backend is NO_DEFAULT:
        return
    _backend(dtype_backend)
    raise NotImplementedError(
        f"dtype_backend={dtype_backend!r} reads each column into a masked or Arrow type,"
        " which the SQL reader does not write yet"
    )


def read_sql_query(
    sql: Any,
    con: Any,
    index_col: Any = None,
    coerce_float: bool = True,
    params: Any = None,
    parse_dates: Any = None,
    chunksize: int | None = None,
    dtype: Any = None,
    dtype_backend: Any = NO_DEFAULT,
) -> DataFrame | Iterator[DataFrame]:
    """The rows a query answers, as a frame, or as an iterator of frames with `chunksize`.

    Each column is typed the way pandas reads the Python values sqlite3 hands
    back, `dtype` casts after that, `parse_dates` reads columns as instants
    and `index_col` moves columns into the row labels.
    """
    _checked_backend(dtype_backend)
    con = _connection(con)
    cursor = _execute(con, sql, params)
    names = [column[0] for column in cursor.description]
    if chunksize is not None:
        return _chunks(cursor, chunksize, names, index_col, parse_dates, dtype)
    rows = cursor.fetchall()
    cursor.close()
    return _wrapped(rows, names, index_col, parse_dates, dtype)


def read_sql(
    sql: Any,
    con: Any,
    index_col: Any = None,
    coerce_float: bool = True,
    params: Any = None,
    parse_dates: Any = None,
    columns: Any = None,
    chunksize: int | None = None,
    dtype_backend: Any = NO_DEFAULT,
    dtype: Any = None,
) -> DataFrame | Iterator[DataFrame]:
    """What `read_sql_query` answers, since over sqlite3 pandas reads `sql` as a query.

    `columns` picks the columns of a table read by name, which a sqlite3
    connection never does, so it is not used, as in pandas.
    """
    return read_sql_query(
        sql,
        con,
        index_col=index_col,
        coerce_float=coerce_float,
        params=params,
        parse_dates=parse_dates,
        chunksize=chunksize,
        dtype=dtype,
        dtype_backend=dtype_backend,
    )


def read_sql_table(
    table_name: str,
    con: Any,
    schema: str | None = None,
    index_col: Any = None,
    coerce_float: bool = True,
    parse_dates: Any = None,
    columns: Any = None,
    chunksize: int | None = None,
    dtype_backend: Any = NO_DEFAULT,
) -> DataFrame | Iterator[DataFrame]:
    """A table read by name, which pandas does only through SQLAlchemy.

    Over sqlite3 pandas checks that the table is there and then raises a
    NotImplementedError with no message, and this does the same.

    Raises:
        ValueError: For a table that is not there, in pandas' words.
        NotImplementedError: For a table that is, as pandas raises it.
    """
    _checked_backend(dtype_backend)
    con = _connection(con)
    if not _has_table(con, table_name):
        raise InvalidArgumentError(f"Table {table_name} not found")
    raise NotImplementedError
