"""read_parquet and read_feather read into the types a dtype_backend names."""

import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq

import firepanda as fp


def _table() -> pa.Table:
    return pa.table(
        {
            "i": pa.array([1.0, None, 3.0]),
            "j": pa.array([1, 2, 3], pa.int32()),
            "s": pa.array(["x", None, "z"]),
            "b": pa.array([True, False, None]),
            "gap": pa.array([None, None, None], pa.float64()),
        }
    )


def test_parquet_numpy_nullable(tmp_path):
    path = tmp_path / "f.parquet"
    pq.write_table(_table(), path)
    got = fp.read_parquet(path, dtype_backend="numpy_nullable")
    assert [str(t) for t in got.dtypes] == ["Float64", "Int32", "string", "boolean", "float64"]
    assert got["i"].isna().tolist() == [False, True, False]


def test_parquet_pyarrow(tmp_path):
    path = tmp_path / "f.parquet"
    pq.write_table(_table(), path)
    got = fp.read_parquet(path, dtype_backend="pyarrow", columns=["i", "j", "s", "b"])
    assert [str(t) for t in got.dtypes] == [
        "double[pyarrow]",
        "int32[pyarrow]",
        "string[pyarrow]",
        "bool[pyarrow]",
    ]


def test_feather_backend(tmp_path):
    path = tmp_path / "f.feather"
    feather.write_feather(_table(), path)
    got = fp.read_feather(path, dtype_backend="numpy_nullable")
    assert str(got["j"].dtype) == "Int32"
    got = fp.read_feather(path, dtype_backend="pyarrow")
    assert str(got["s"].dtype) == "string[pyarrow]"


def test_empty_to_pandas_kwargs(tmp_path):
    path = tmp_path / "f.parquet"
    pq.write_table(_table(), path)
    got = fp.read_parquet(path, to_pandas_kwargs={})
    assert str(got["j"].dtype) == "int32"
