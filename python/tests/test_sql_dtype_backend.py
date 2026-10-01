"""read_sql reads into the types a dtype_backend names, and Arrow columns cast to plain types."""

import sqlite3

import pyarrow as pa

import firepanda as fp


def _connection() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute("create table t (i integer, f real, s text)")
    con.execute("insert into t values (1, 1.5, 'x'), (null, null, null), (3, 2.0, 'z')")
    return con


def test_nullable_backend_keeps_whole_numbers_whole():
    got = fp.read_sql_query("select * from t", _connection(), dtype_backend="numpy_nullable")
    assert [str(t) for t in got.dtypes] == ["Int64", "Float64", "string"]
    assert got["i"].isna().tolist() == [False, True, False]


def test_arrow_backend():
    got = fp.read_sql("select * from t", _connection(), dtype_backend="pyarrow")
    assert [str(t) for t in got.dtypes] == ["int64[pyarrow]", "double[pyarrow]", "string[pyarrow]"]


def test_backend_before_dtype_and_dates():
    con = _connection()
    got = fp.read_sql_query("select * from t", con, dtype_backend="pyarrow", dtype={"i": "float64"})
    assert str(got["i"].dtype) == "float64"
    assert got["i"].isna().tolist() == [False, True, False]
    con.execute("create table d (t text)")
    con.execute("insert into d values ('2024-01-02'), (null)")
    got = fp.read_sql_query("select * from d", con, dtype_backend="pyarrow", parse_dates=["t"])
    assert str(got["t"].dtype) == "datetime64[us]"


def test_chunks_take_the_backend():
    con = _connection()
    chunks = fp.read_sql_query("select * from t", con, dtype_backend="pyarrow", chunksize=2)
    assert [str(c["i"].dtype) for c in chunks] == ["int64[pyarrow]", "int64[pyarrow]"]


def test_arrow_column_casts_to_plain_types():
    column = fp.Series([1, None, 3], dtype=fp.ArrowDtype(pa.int64()))
    assert str(column.astype("float64").dtype) == "float64"
    assert column.astype(str).tolist()[0] == "1"
    assert str(fp.DataFrame({"a": column}).astype({"a": "float64"})["a"].dtype) == "float64"
    text = fp.Series(["2024-01-02", None], dtype=fp.ArrowDtype(pa.string()))
    assert str(text.astype("str").dtype) == "str"
    assert str(fp.to_datetime(text).dtype) == "datetime64[us]"
    assert str(fp.to_datetime(fp.Series(["2024-01-02", None], dtype="string")).dtype) == (
        "datetime64[us]"
    )
