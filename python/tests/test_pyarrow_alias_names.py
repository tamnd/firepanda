"""Constructors take `int64[pyarrow]` names, and `read_csv` reads `string` as the string type."""

import io

import firepanda as pd


def test_series_takes_pyarrow_name():
    assert str(pd.Series([1, 2], dtype="int64[pyarrow]").dtype) == "int64[pyarrow]"
    assert str(pd.Series([1.0, None], dtype="double[pyarrow]").dtype) == "double[pyarrow]"


def test_array_takes_pyarrow_name():
    assert str(pd.array([1, 2], dtype="int64[pyarrow]").dtype) == "int64[pyarrow]"


def test_frame_takes_pyarrow_name():
    frame = pd.DataFrame({"a": [1, 2]}, dtype="int64[pyarrow]")
    assert frame.dtypes.astype(str).tolist() == ["int64[pyarrow]"]


def test_string_pyarrow_stays_string():
    assert str(pd.Series(["a"], dtype="string[pyarrow]").dtype) == "string"


def test_read_csv_string_dtype():
    text = "a,b\nx,1\n,2\n"
    frame = pd.read_csv(io.StringIO(text), dtype={"a": "string"})
    assert frame.dtypes.astype(str).tolist() == ["string", "int64"]
    assert frame["a"].isna().tolist() == [False, True]
    whole = pd.read_csv(io.StringIO(text), dtype=pd.StringDtype())
    assert whole.dtypes.astype(str).tolist() == ["string", "string"]


def test_read_xml_string_dtype():
    xml = "<data><row><a>x</a><b>1</b></row><row><a>y</a><b>2</b></row></data>"
    frame = pd.read_xml(io.StringIO(xml), parser="etree", dtype={"a": "string"})
    assert frame.dtypes.astype(str).tolist() == ["string", "int64"]


def test_double_pyarrow_header_pads():
    frame = pd.DataFrame({"longname": pd.Series([4.0, None], dtype="double[pyarrow]")})
    assert repr(frame).splitlines()[0] == "   longname"
