"""`read_csv` and `read_json` read into the nullable or the Arrow types `dtype_backend` names."""

import io

import pytest

import firepanda as fp

TEXT = "a,b,c,d,e\n1,x,2.0,True,1.5\n,y,3.0,False,\n"


@pytest.mark.parametrize(
    ("backend", "dtypes"),
    [
        ("numpy_nullable", ["Int64", "string", "Float64", "boolean", "Float64"]),
        (
            "pyarrow",
            [
                "int64[pyarrow]",
                "string[pyarrow]",
                "double[pyarrow]",
                "bool[pyarrow]",
                "double[pyarrow]",
            ],
        ),
    ],
)
def test_columns_take_the_backend_types(backend, dtypes):
    frame = fp.read_csv(io.StringIO(TEXT), dtype_backend=backend)
    assert [str(dtype) for dtype in frame.dtypes] == dtypes
    assert frame["a"].tolist() == [1, fp.NA]
    assert frame["b"].tolist() == ["x", "y"]
    assert frame["e"].isna().tolist() == [False, True]


def test_dates_keep_their_numpy_type():
    text = "when,n\n2024-01-02,1\n2024-01-03,2\n"
    for backend in ("numpy_nullable", "pyarrow"):
        frame = fp.read_csv(io.StringIO(text), dtype_backend=backend, parse_dates=["when"])
        assert str(frame["when"].dtype).startswith("datetime64")


def test_no_backend_keeps_numpy_types():
    frame = fp.read_csv(io.StringIO(TEXT))
    expected = ["float64", "str", "float64", "bool", "float64"]
    assert [str(dtype) for dtype in frame.dtypes] == expected


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("int64[pyarrow]", [1, 2]),
        ("double[pyarrow]", [1.5, 2.5]),
        ("bool[pyarrow]", [True, False]),
    ],
)
def test_astype_reads_an_arrow_name(name, values):
    column = fp.Series(values).astype(name)
    assert str(column.dtype) == name
    assert column.tolist() == values
    frame = fp.DataFrame({"v": values}).astype({"v": name})
    assert str(frame["v"].dtype) == name


@pytest.mark.parametrize(
    ("backend", "dtypes"),
    [
        ("numpy_nullable", ["Int64", "string", "Float64", "boolean"]),
        ("pyarrow", ["int64[pyarrow]", "string[pyarrow]", "double[pyarrow]", "bool[pyarrow]"]),
    ],
)
def test_read_json_takes_the_backend_types(backend, dtypes):
    text = '[{"a":1,"b":"x","c":1.5,"d":true},{"a":null,"b":null,"c":null,"d":false}]'
    frame = fp.read_json(io.StringIO(text), dtype_backend=backend)
    assert [str(dtype) for dtype in frame.dtypes] == dtypes
    assert frame["a"].tolist() == [1, fp.NA]


def test_read_json_series_takes_the_backend_type():
    column = fp.read_json(io.StringIO('{"0":1,"1":null}'), typ="series", dtype_backend="pyarrow")
    assert str(column.dtype) == "int64[pyarrow]"


def test_read_json_refuses_an_unknown_backend():
    with pytest.raises(ValueError, match="only 'numpy_nullable' and 'pyarrow' are allowed"):
        fp.read_json(io.StringIO("[]"), dtype_backend="numpy")
