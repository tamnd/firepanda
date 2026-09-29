"""`read_excel` and `ExcelFile`, compared with pandas.

The checks that need no engine run everywhere: the signatures, telling a
workbook's kind from its first bytes, and the sentence for a missing engine.
The reading tests need the engine's package, write one workbook, read it with
both libraries under the same options, and compare the types, the labels and
the values.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import inspect
import io
import sys
import zipfile
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)
needs_openpyxl = pytest.mark.skipif(
    importlib.util.find_spec("openpyxl") is None, reason="openpyxl is not installed"
)


def zipped(*names: str) -> bytes:
    """A zip holding empty files, which is enough to tell a workbook's kind."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name in names:
            archive.writestr(name, "")
    return buf.getvalue()


@needs_pandas
@pytest.mark.parametrize("path", ["read_excel", "ExcelFile.__init__", "ExcelFile.parse"])
def test_the_signature_is_pandas(path: str) -> None:
    import pandas

    def found(lib: ModuleType) -> Any:
        target: Any = lib
        for part in path.split("."):
            target = getattr(target, part)
        return target

    ours = inspect.signature(found(fp))
    theirs = inspect.signature(found(pandas))
    assert list(ours.parameters) == list(theirs.parameters)


@needs_pandas
@pytest.mark.parametrize(
    "call",
    [
        lambda lib: lib.read_excel(io.BytesIO(b"not a workbook at all")),
        lambda lib: lib.read_excel(io.BytesIO(zipped("a.txt")), engine="nope"),
        lambda lib: lib.read_excel(io.BytesIO(zipped("a.txt")), dtype_backend="x"),
        lambda lib: lib.ExcelFile(io.BytesIO(b"\x00\x01")),
    ],
)
def test_a_mistake_before_reading_is_pandas_mistake(call: Callable[[ModuleType], Any]) -> None:
    import pandas

    with pytest.raises(Exception) as theirs:
        call(pandas)
    with pytest.raises(Exception) as ours:
        call(fp)
    assert isinstance(ours.value, type(theirs.value))
    assert str(ours.value) == str(theirs.value)


@needs_pandas
@pytest.mark.parametrize(
    ("names", "module"),
    [
        (("xl/workbook.xml",), "openpyxl"),
        (("xl/workbook.bin",), "pyxlsb"),
        (("content.xml",), "odf"),
    ],
)
def test_a_missing_engine_is_pandas_sentence(
    monkeypatch: Any, names: tuple[str, ...], module: str
) -> None:
    import pandas

    monkeypatch.setitem(sys.modules, module, None)
    data = zipped(*names)
    with pytest.raises(ImportError) as theirs:
        pandas.read_excel(io.BytesIO(data))
    with pytest.raises(ImportError) as ours:
        fp.read_excel(io.BytesIO(data))
    assert str(ours.value) == str(theirs.value)


def test_a_missing_xlrd_names_the_version(monkeypatch: Any) -> None:
    monkeypatch.setitem(sys.modules, "xlrd", None)
    data = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + bytes(64)
    with pytest.raises(ImportError, match=r"Install xlrd >= 2\.0\.1 for xls Excel support"):
        fp.read_excel(io.BytesIO(data))


ROWS: list[list[Any]] = [
    ["a", "b", "c", "d", "e"],
    [1, 1.5, "x", True, dt.datetime(2020, 1, 1)],
    [2, None, "y", False, dt.datetime(2020, 1, 2)],
    [3, 2.0, None, True, None],
    [4, 3.25, "NA", None, dt.datetime(2020, 1, 4)],
]


def workbook(rows: list[list[Any]], sheets: tuple[str, ...] = ("S1",)) -> bytes:
    import openpyxl

    book = openpyxl.Workbook()
    book.remove(book.active)
    for name in sheets:
        sheet = book.create_sheet(name)
        for row in rows:
            sheet.append(row)
    buf = io.BytesIO()
    book.save(buf)
    return buf.getvalue()


GAPS = frozenset(["None", "nan", "NaT", "<NA>"])


def shown(frame: Any) -> Any:
    """What a frame holds, in plain values that compare equal across the two libraries."""
    columns = []
    for i in range(frame.shape[1]):
        values = frame.iloc[:, i].tolist()
        # A gap prints as nan, NaT or <NA> depending on the type, and all read as missing here.
        columns.append([None if str(v) in GAPS else str(v) for v in values])
    # firepanda spells pandas' text type `string`.
    dtypes = [str(t).replace("string", "str") for t in frame.dtypes]
    return dtypes, [str(c) for c in frame.columns], [str(i) for i in frame.index], columns


def outcome(lib: ModuleType, data: bytes, options: dict[str, Any]) -> Any:
    try:
        out = lib.read_excel(io.BytesIO(data), **options)
    except Exception as err:
        return "raise", isinstance(err, ValueError), str(err)
    if isinstance(out, dict):
        return {key: shown(frame) for key, frame in out.items()}
    return shown(out)


READ: list[dict[str, Any]] = [
    {},
    {"header": None},
    {"index_col": 0},
    {"index_col": [0, 1]},
    {"usecols": "A:C"},
    {"usecols": "A,C:D"},
    {"usecols": [0, 2]},
    {"usecols": ["a", "e"]},
    {"usecols": lambda name: name != "b"},
    {"nrows": 2},
    {"na_values": ["x"]},
    {"na_values": {"c": ["y"]}},
    {"keep_default_na": False},
    {"na_filter": False},
    {"names": ["p", "q", "r", "s", "t"]},
    {"dtype": {"a": float}},
    {"dtype": {"c": str}},
    {"converters": {"a": str}},
    {"header": [0, 1], "usecols": "A:D"},
    {"skipfooter": 1},
    {"skiprows": [2]},
    {"skiprows": lambda i: i == 3},
    {"true_values": ["x"], "false_values": ["y"]},
    {"sheet_name": None},
    {"sheet_name": [0, "S1"]},
    {"sheet_name": "nope"},
    {"sheet_name": 3},
    {"sheet_name": []},
    {"usecols": 3},
    {"usecols": ["zz"]},
    {"nrows": -1},
    {"header": -1},
    {"dtype": {"c": int}},
    {"names": list("abcdefg")},
    {"dtype_backend": "numpy_nullable", "usecols": [0, 1, 3, 4]},
]


@needs_pandas
@needs_openpyxl
@pytest.mark.parametrize("options", READ)
def test_a_sheet_reads_as_pandas_reads_it(options: dict[str, Any]) -> None:
    import pandas

    data = workbook(ROWS)
    assert outcome(fp, data, options) == outcome(pandas, data, options)


TYPED: list[list[list[Any]]] = [
    # Text that looks like a number reads as one, with a comma for thousands.
    [["n"], ["1"], ["2.5"], [" 3 "], ["1e3"]],
    # Flags with a gap read as floats, and text flags as flags.
    [["f", "t"], [True, "True"], [None, "false"], [False, "TRUE"]],
    # Repeated and missing names.
    [["a", "a", "", "a"], [1, 2, 3, 4]],
    # A column of nothing but gaps, and a ragged row.
    [["a", "b"], [1], [2, None], [3]],
    # Numbers and text in one column make an object column.
    [["m"], [1], ["a"], [2.5]],
    # Moments with a gap stay moments.
    [["t"], [dt.datetime(2021, 5, 1, 12)], [None], [dt.datetime(2021, 5, 2)]],
]


@needs_pandas
@needs_openpyxl
@pytest.mark.parametrize("rows", TYPED)
@pytest.mark.parametrize("options", [{}, {"thousands": ","}, {"index_col": 0}])
def test_a_column_is_typed_as_pandas_types_it(
    rows: list[list[Any]], options: dict[str, Any]
) -> None:
    import pandas

    data = workbook(rows)
    assert outcome(fp, data, options) == outcome(pandas, data, options)


@needs_pandas
@needs_openpyxl
def test_an_excel_file_reads_several_sheets() -> None:
    import pandas

    data = workbook(ROWS, sheets=("S1", "S2"))
    with fp.ExcelFile(io.BytesIO(data)) as ours, pandas.ExcelFile(io.BytesIO(data)) as theirs:
        assert ours.engine == theirs.engine == "openpyxl"
        assert ours.sheet_names == theirs.sheet_names
        assert shown(ours.parse("S2", usecols="B:C")) == shown(theirs.parse("S2", usecols="B:C"))
        assert shown(fp.read_excel(ours, sheet_name=1)) == shown(
            pandas.read_excel(theirs, sheet_name=1)
        )
        with pytest.raises(ValueError, match="ExcelFile already has the engine set"):
            fp.read_excel(ours, engine="odf")


@needs_openpyxl
def test_a_path_reads_like_a_handle(tmp_path: Any) -> None:
    path = tmp_path / "book.xlsx"
    path.write_bytes(workbook(ROWS))
    assert shown(fp.read_excel(path)) == shown(fp.read_excel(io.BytesIO(path.read_bytes())))
    assert shown(fp.read_excel(str(path))) == shown(fp.read_excel(path))


@pytest.mark.skipif(
    importlib.util.find_spec("odf") is None or importlib.util.find_spec("pandas") is None,
    reason="odfpy or pandas is not installed",
)
@pytest.mark.parametrize("options", [{}, {"index_col": 0}, {"header": None}, {"nrows": 1}])
def test_an_open_document_sheet_reads_as_pandas_reads_it(options: dict[str, Any]) -> None:
    import pandas

    frame = pandas.DataFrame({"a": [1, 2], "b": [1.5, None], "c": ["x", None], "d": [True, False]})
    buf = io.BytesIO()
    frame.to_excel(buf, engine="odf", index=False)
    data = buf.getvalue()
    assert outcome(fp, data, options) == outcome(pandas, data, options)


@pytest.mark.skipif(
    importlib.util.find_spec("python_calamine") is None
    or importlib.util.find_spec("openpyxl") is None
    or importlib.util.find_spec("pandas") is None,
    reason="python-calamine, openpyxl or pandas is not installed",
)
@pytest.mark.parametrize("options", [{}, {"index_col": 0}, {"nrows": 2}])
def test_calamine_reads_as_pandas_reads_it(options: dict[str, Any]) -> None:
    import pandas

    data = workbook(ROWS)
    options = {"engine": "calamine", **options}
    assert outcome(fp, data, options) == outcome(pandas, data, options)
