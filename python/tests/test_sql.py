"""`to_sql` and `read_sql` over a sqlite3 connection against pandas, compared by repr."""

from __future__ import annotations

import datetime
import sqlite3
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")


def sample(lib: ModuleType) -> Any:
    return lib.DataFrame(
        {
            "a": [1, 2],
            "b": [1.5, None],
            "c": ["x", None],
            "d": [True, False],
            "t": lib.to_datetime(["2026-01-01 00:00:00", "2026-01-02 03:04:05"]),
        }
    )


def written(lib: ModuleType, **options: Any) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    sample(lib).to_sql("t", con, **options)
    return con


def schema(con: sqlite3.Connection) -> list[Any]:
    return con.execute("select sql from sqlite_master order by name").fetchall()


def rows(con: sqlite3.Connection, table: str = "t") -> list[Any]:
    return con.execute(f"select * from {table}").fetchall()


def twice(lib: ModuleType, **options: Any) -> Any:
    con = written(lib)
    count = sample(lib).to_sql("t", con, **options)
    return count, schema(con), rows(con)


def dtypes(frame: Any) -> list[str]:
    """The column types, with text spelled `str` as pandas 3 spells it."""
    return [str(dtype).replace("string", "str") for dtype in frame.dtypes]


def spans(lib: ModuleType) -> Any:
    con = sqlite3.connect(":memory:")
    lib.DataFrame({"x": lib.to_timedelta(["1s", None])}).to_sql("s", con)
    return schema(con), rows(con, "s")


def clocks(lib: ModuleType) -> Any:
    con = sqlite3.connect(":memory:")
    frame = lib.DataFrame({"x": [datetime.date(2026, 1, 2)], "y": [datetime.time(1, 2)]})
    frame.to_sql("c", con, index=False)
    return schema(con), rows(con, "c")


def named(lib: ModuleType) -> Any:
    con = sqlite3.connect(":memory:")
    lib.Series([1], index=lib.Index(["p"], name="n"), name="v").to_sql("s", con)
    return schema(con)


def column(lib: ModuleType) -> Any:
    con = sqlite3.connect(":memory:")
    count = sample(lib)["b"].to_sql("s", con)
    return count, schema(con), rows(con, "s")


def read(lib: ModuleType, sql: str, **options: Any) -> Any:
    return lib.read_sql(sql, written(lib), **options)


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "count": lambda lib: sample(lib).to_sql("t", sqlite3.connect(":memory:")),
    "schema": lambda lib: schema(written(lib)),
    "rows": lambda lib: rows(written(lib)),
    "no-index": lambda lib: (schema(written(lib, index=False)), rows(written(lib, index=False))),
    "index-label": lambda lib: schema(written(lib, index_label="k")),
    "index-label-length": lambda lib: written(lib, index_label=["k", "j"]),
    "named-index": named,
    "series": column,
    "fail": lambda lib: twice(lib),
    "append": lambda lib: twice(lib, if_exists="append"),
    "replace": lambda lib: twice(lib, if_exists="replace"),
    "delete-rows": lambda lib: twice(lib, if_exists="delete_rows"),
    "if-exists-unknown": lambda lib: written(lib, if_exists="x"),
    "chunks": lambda lib: rows(written(lib, chunksize=1)),
    "chunks-zero": lambda lib: written(lib, chunksize=0),
    "multi": lambda lib: rows(written(lib, method="multi")),
    "method-unknown": lambda lib: written(lib, method="x"),
    "dtype": lambda lib: schema(written(lib, dtype={"a": "TEXT"})),
    "dtype-scalar": lambda lib: schema(written(lib, dtype="BLOB")),
    "dtype-not-text": lambda lib: written(lib, dtype={"a": int}),
    "duplicate": lambda lib: (
        sample(lib).set_index("a", drop=False).to_sql("x", sqlite3.connect(":memory:"))
    ),
    "spans": spans,
    "clocks": clocks,
    "uri": lambda lib: sample(lib).to_sql("t", "sqlite://"),
    "read": lambda lib: read(lib, "select * from t"),
    "read-dtypes": lambda lib: dtypes(read(lib, "select * from t")),
    "read-params": lambda lib: read(lib, "select a, b from t where a < ?", params=(2,)),
    "read-index": lambda lib: read(lib, "select a, c from t", index_col="a"),
    "read-dates": lambda lib: read(lib, "select t from t", parse_dates=["t"]),
    "read-epoch": lambda lib: read(lib, "select a from t", parse_dates={"a": "s"}),
    "read-dtype": lambda lib: read(lib, "select a, b from t", dtype={"a": "float64"}),
    "read-chunks": lambda lib: list(read(lib, "select a from t", chunksize=1)),
    "read-chunks-empty": lambda lib: list(read(lib, "select a from t where a > 9", chunksize=1)),
    "read-empty": lambda lib: read(lib, "select a from t where a > 9").shape,
    "read-query": lambda lib: lib.read_sql_query("select a from t", written(lib)),
    "read-big": lambda lib: read(lib, "select 9223372036854775808 as u, 1 as i"),
    "read-bad": lambda lib: read(lib, "select zz from t"),
    "read-not-text": lambda lib: lib.read_sql(1, written(lib)),
    "read-table": lambda lib: lib.read_sql_table("t", written(lib)),
    "read-table-missing": lambda lib: lib.read_sql_table("nope", written(lib)),
    "read-uri": lambda lib: lib.read_sql("select 1", "sqlite://"),
}


def mistake(error: Exception) -> str:
    """A mistake as its builtin class and message, since each library raises its own subclass."""
    kind = next(kind.__name__ for kind in type(error).__mro__ if kind.__module__ == "builtins")
    return f"{kind}: {error}"


def outcome(build: Callable[[], Any]) -> str:
    try:
        return repr(build())
    except Exception as error:
        return mistake(error)


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_answers_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))


def test_a_column_of_nothing_but_nulls_reads_as_gaps() -> None:
    """pandas reads it as objects holding None, and firepanda cannot mark such a column object.

    An object column is known by its first written cell, and a column of
    nothing but gaps, or of no rows, has none, so its gaps read as NaN.
    """
    con = sqlite3.connect(":memory:")
    assert fp.read_sql("select null as n", con)["n"].isna().tolist() == [True]
    assert fp.read_sql("select 1 as n where 0", con).shape == (0, 1)
