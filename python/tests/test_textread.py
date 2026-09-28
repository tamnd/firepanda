"""`read_csv`, `read_table` and `read_fwf`, compared with pandas.

Each case reads the same text with both libraries and compares the names, each column's
type, the row labels and every value. The mistakes compare the exception's type and its
message, and the warnings compare their messages, because code written against pandas
catches both by what they say.
"""

from __future__ import annotations

import bz2
import csv
import gzip
import importlib.util
import io
import math
import warnings
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is needed"
)


def plain(value: Any) -> Any:
    """A value as Python has it, with every kind of missing as None."""
    if hasattr(value, "isoformat"):
        return str(value)
    if hasattr(value, "item"):
        value = value.item()
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if type(value).__name__ in ("NAType", "NaTType"):
        return None
    return value


def shown(result: Any) -> Any:
    """What is compared: the names, the types, the row labels and the values."""
    if not hasattr(result, "columns"):
        return [shown(chunk) for chunk in result]
    return (
        [str(name) for name in result.columns],
        [str(kind).replace("string", "str") for kind in result.dtypes],
        [plain(label) for label in result.index],
        result.index.name,
        [[plain(value) for value in result[name].tolist()] for name in result.columns],
    )


def outcome(read: Callable[[], Any]) -> tuple[Any, list[str]]:
    """What a read gives, or the mistake it makes, with the warnings it raises."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            got = shown(read())
        except Exception as error:
            got = (type(error).__name__, str(error))
    said = [
        f"{type(one.message).__name__}: {one.message}"
        for one in caught
        if "Pandas4Warning" not in type(one.message).__name__
    ]
    return got, said


def same(firepanda: ModuleType, read: Callable[[ModuleType], Any]) -> None:
    import pandas as pd

    assert outcome(lambda: read(firepanda)) == outcome(lambda: read(pd))


CASES: dict[str, tuple[str, dict[str, Any]]] = {
    "plain": ("a,b\n1,2\n3,4\n", {}),
    "padded numbers": ("a,b\n 1, 2\n3,4\n", {}),
    "a column with nothing in it": ("a,b\n,1\n,2\n", {}),
    "a gap in numbers": ("a,b\n1,x\n,y\n", {}),
    "short rows": ("a,b,c\n1,2\n3,4,5\n", {}),
    "a leading field with no name": ("a,b\n1,2,3\n4,5,6\n", {}),
    "a leading field on the first row only": ("a,b\n1,2,3\n4,5\n", {}),
    "blank and repeated names": ("a,,a,a\n1,2,3,4\n", {}),
    "a repeat that clashes": ("a,a,a.1,a\n1,2,3,4\n", {}),
    "a byte order mark": ("﻿a,b\n1,2\n", {}),
    "a trailing comma": ("a,b,\n1,2,\n", {}),
    "semicolons": ("a;b\n1;2\n", {"sep": ";"}),
    "whitespace": ("a b\n1   2\n", {"sep": r"\s+"}),
    "a pattern": ("a::b\n1::2\n", {"sep": "::", "engine": "python"}),
    "the pattern warns": ("a::b\n1::2\n", {"sep": "::"}),
    "a sniffed separator": ("a b\n1 2\n", {"sep": None, "engine": "python"}),
    "skipfooter": ("a,b\n1,2\n3,4\n5,6\n", {"skipfooter": 1, "engine": "python"}),
    "skipfooter warns": ("a,b\n1,2\n3,4\n5,6\n", {"skipfooter": 1}),
    "nrows": ("a,b\n1,2\n3,4\n5,6\n", {"nrows": 2}),
    "skiprows as a list": ("a,b\n1,2\n3,4\n5,6\n", {"skiprows": [1]}),
    "skiprows as a count": ("x\na,b\n1,2\n", {"skiprows": 1}),
    "skiprows as a callable": ("x\n1\n2\n3\n", {"skiprows": lambda at: at == 2}),
    "no header": ("a,b\n1,2\n3,4\n", {"header": None}),
    "names": ("a,b\n1,2\n3,4\n", {"names": ["x", "y"]}),
    "names over the header": ("a,b\n1,2\n3,4\n", {"names": ["x", "y"], "header": 0}),
    "the header further down": ("x,y\na,b\n1,2\n", {"header": 1}),
    "index_col by position": ("a,b\n1,2\n3,4\n", {"index_col": 0}),
    "index_col by name": ("a,b\n1,2\n3,4\n", {"index_col": "b"}),
    "index_col in a list": ("a,b\n1,2\n", {"index_col": [0]}),
    "usecols with index_col": ("a,b\nx,1\ny,2\n", {"index_col": "a", "usecols": ["a", "b"]}),
    "usecols as a callable": ("a,b,c\n1,2,3\n", {"usecols": lambda name: name != "b"}),
    "usecols as positions": ("a,b,c\n1,2,3\n", {"usecols": [2, 0]}),
    "every column as text": ("a,b\n1,2\n3,4\n", {"dtype": str}),
    "leading zeros kept": ("a,b\n007,2\n", {"dtype": {"a": str}}),
    "a float column": ("a,b\n1,2\n3,4\n", {"dtype": {"a": "float64"}}),
    "a narrow integer": ("a,b\n1,2\n3,4\n", {"dtype": {"a": "int32"}}),
    "a type by position": ("a,b\n1,2\n", {"dtype": {1: "float64"}}),
    "categories": ("a,b\n1,x\n3,y\n", {"dtype": {"b": "category"}}),
    "converters": ("a,b\n1,2\n3,4\n", {"converters": {"a": lambda text: int(text) * 10}}),
    "a converter over a type": (
        "a,b\n1,2\n",
        {"dtype": {"a": "float64"}, "converters": {"a": str}},
    ),
    "true and false values": (
        "a,b\nyes,no\nno,yes\n",
        {"true_values": ["yes"], "false_values": ["no"]},
    ),
    "booleans": ("a,b\nTrue,1\nFalse,2\n", {}),
    "no default missing text": ("a,b\nfoo,1\nNA,2\n", {"keep_default_na": False}),
    "no missing text at all": ("a,b\nfoo,1\nNA,2\n", {"na_filter": False}),
    "more missing text": ("a,b\nfoo,1\nbar,2\n", {"na_values": ["foo"]}),
    "missing numbers by column": ("a,b\n1,1\n2,2\n", {"na_values": {"a": [1]}}),
    "blank lines kept": ("a,b\n1,2\n\n3,4\n", {"skip_blank_lines": False}),
    "blank lines skipped": ("\n\na,b\n\n1,2\n", {}),
    "a line of spaces": ("a\n1\n \n2\n", {}),
    "a quoted empty field": ('a,b\n"",1\n', {}),
    "dates": ("a,b\n2021-01-01,1\n2021-01-02,2\n", {"parse_dates": ["a"]}),
    "dated row labels": ("a,b\n2021-01-01,1\n", {"parse_dates": True, "index_col": 0}),
    "day first": ("a,b\n01/02/2021,1\n", {"parse_dates": ["a"], "dayfirst": True}),
    "a date format": ("a\n01/02/2021\n", {"parse_dates": ["a"], "date_format": "%d/%m/%Y"}),
    "dates that are not": ("a\nfoo\n", {"parse_dates": ["a"]}),
    "thousands": ('a,b\n"1,000",2\n"2,000",3\n', {"thousands": ","}),
    "a decimal comma": ("a;b\n1,5;2\n", {"sep": ";", "decimal": ","}),
    "both separators": ("a\n1.000,5\n", {"thousands": ".", "decimal": ","}),
    "a line terminator": ("a,b|1,2|3,4|", {"lineterminator": "|"}),
    "another quote": ("a,b\n'x,y',1\n", {"quotechar": "'"}),
    "a doubled quote": ('a,b\n"x""y",1\n', {}),
    "an escape": ("a,b\nx\\,y,1\n", {"escapechar": "\\"}),
    "comments": ("a,b\n1,2 # hi\n#full\n3,4\n", {"comment": "#"}),
    "comments with whitespace": ("a b\n1 2 # c\n", {"sep": r"\s+", "comment": "#"}),
    "no quoting": ("a,b\n1,2\n", {"quoting": csv.QUOTE_NONE}),
    "quote everything else": ('a,b\n"1",x\n', {"quoting": csv.QUOTE_NONNUMERIC}),
    "skipinitialspace": ("a, b\n1, x\n", {"skipinitialspace": True}),
    "a dialect": ("a,b\n1,2\n", {"dialect": "excel-tab"}),
    "signs": ("a,b\n-1,2\n+3,4\n", {}),
    "underscores are text": ("a,b\n1_0,2\n", {}),
    "infinity": ("a,b\n1.5,2\ninf,3\n", {}),
    "nan": ("a,b\nnan,2\n1.5,3\n", {}),
    "exponents": ("a,b\n1e3,2\n", {}),
    "past int64": ("a,b\n18446744073709551615,1\n0,2\n", {}),
    "carriage returns": ("a,b\r\n1,2\r\n3,4\r\n", {}),
    "extra fields cut": ("a,b\n1,2,3\n4,5\n", {"index_col": False}),
    "bad lines skipped": ("a,b\n1,2\n3,4,5\n6,7\n", {"on_bad_lines": "skip"}),
    "bad lines warned": ("a,b\n1,2\n3,4,5\n6,7\n", {"on_bad_lines": "warn"}),
    "bad lines handed over": (
        "a,b\n1,2\n3,4,5\n6,7\n",
        {"on_bad_lines": lambda fields: fields[:2], "engine": "python"},
    ),
    "chunks": ("a,b\n1,2\n3,4\n5,6\n", {"chunksize": 2}),
    "chunks typed alone": ("a\n1\nx\n3\n", {"chunksize": 1}),
    "latin 1": ("a,b\n1,2\n", {"encoding": "latin-1"}),
}


@pytest.mark.parametrize("name", list(CASES))
def test_read_csv_reads_as_pandas_does(firepanda: ModuleType, name: str) -> None:
    text, options = CASES[name]
    same(firepanda, lambda m: m.read_csv(io.StringIO(text), **options))


MISTAKES: dict[str, tuple[str, dict[str, Any]]] = {
    "a bad line": ("a,b\n1,2\n3,4,5\n6,7\n", {}),
    "a bad line under python": ("a,b\n1,2\n3,4,5\n6,7\n", {"engine": "python"}),
    "a bad line after a quoted newline": ('a,b\n"x\ny",1\n2,3,4\n', {}),
    "nothing in the file": ("", {}),
    "a header past the end": ("a,b\n1,2\n", {"header": 5}),
    "a header of True": ("a,b\n1,2\n", {"header": True}),
    "a negative header": ("a,b\n1,2\n", {"header": -1}),
    "negative nrows": ("a,b\n1,2\n", {"nrows": -1}),
    "repeated names": ("a,b\n1,2\n", {"names": ["x", "x"]}),
    "skipfooter on c": ("a,b\n1,2\n", {"skipfooter": 1, "engine": "c"}),
    "skipfooter and nrows": ("a,b\n1,2\n", {"skipfooter": 1, "nrows": 1, "engine": "python"}),
    "a pattern on c": ("a::b\n1::2\n", {"sep": "::", "engine": "c"}),
    "parse_dates as a dict": ("a,b\n1,2\n", {"parse_dates": {"a": 1}}),
    "parse_dates nested": ("a,b\n1,2\n", {"parse_dates": [["a", "b"]]}),
    "parse_dates missing": ("a\n1\n", {"parse_dates": ["z"]}),
    "a date type": ("a,b\n1,2\n", {"dtype": {"a": "datetime64[ns]"}}),
    "a float precision": ("a,b\n1,2\n", {"float_precision": "x"}),
    "an on_bad_lines": ("a,b\n1,2\n", {"on_bad_lines": "x"}),
    "a callable on c": ("a,b\n1,2\n", {"on_bad_lines": print}),
    "index_col of True": ("a,b\n1,2\n", {"index_col": True}),
    "usecols not there": ("a,b\n1,2\n", {"usecols": ["z"]}),
    "an engine": ("a,b\n1,2\n", {"engine": "zzz"}),
    "a dtype_backend": ("a,b\n1,2\n", {"dtype_backend": "zzz"}),
    "sep and delimiter": ("a,b\n1,2\n", {"sep": ",", "delimiter": ","}),
    "storage options": ("a,b\n1,2\n", {"storage_options": {"x": 1}}),
    "a gap in an integer type": ("a,b\n1,\n3,4\n", {"dtype": {"b": "int64"}}),
    "text in an integer type": ("a\nx\n", {"dtype": {"a": "int64"}}),
    "text in a float type": ("a\nx\n", {"dtype": "float64"}),
    "a zero chunk size": ("x\n1\n", {"chunksize": 0}),
    "a long decimal": ("x\n1\n", {"decimal": ".."}),
    "a long thousands": ("x\n1\n", {"thousands": ".."}),
}


@pytest.mark.parametrize("name", list(MISTAKES))
def test_a_mistake_is_pandas_mistake(firepanda: ModuleType, name: str) -> None:
    text, options = MISTAKES[name]
    same(firepanda, lambda m: m.read_csv(io.StringIO(text), **options))


FILES = {
    "plain": "a,b\n1,2\n3,4\n",
    "padded": "a,b\n 1, 2\n3,4\n",
    "all missing": "a,b\n,1\n,2\n",
    "a leading field": "a,b\n1,2,3\n4,5,6\n",
    "short rows": "a,b,c\n1,2\n3,4,5\n",
    "a blank name": "a,,c\n1,2,3\n",
    "repeated names": "a,a,a\n1,2,3\n",
    "empty": "",
    "a byte order mark": "﻿a,b\n1,2\n",
    "a trailing comma": "a,b,\n1,2,\n",
    "text with a gap": "a,b\nx,1\ny,\n",
    "a line of spaces": "a\n1\n \n2\n",
    "past int64": "v,u\n1,0\n2,18446744073709551615\n",
}


@pytest.mark.parametrize("index_col", [None, 0])
@pytest.mark.parametrize("name", list(FILES))
def test_a_file_on_disk_reads_as_pandas_reads_it(
    firepanda: ModuleType, tmp_path: Path, name: str, index_col: Any
) -> None:
    path = tmp_path / "frame.csv"
    path.write_text(FILES[name], encoding="utf-8")
    same(firepanda, lambda m: m.read_csv(str(path), index_col=index_col))
    same(firepanda, lambda m: m.read_csv(path, usecols=lambda column: True))


@pytest.mark.parametrize(("suffix", "opener"), [(".gz", gzip.open), (".bz2", bz2.open), ("", open)])
def test_compressed_files_are_read(
    firepanda: ModuleType, tmp_path: Path, suffix: str, opener: Any
) -> None:
    path = tmp_path / f"frame.csv{suffix}"
    with opener(path, "wt") as handle:
        handle.write("a,b\n1,x\n2,y\n")
    same(firepanda, lambda m: m.read_csv(path))
    method = {".gz": "gzip", ".bz2": "bz2", "": None}[suffix]
    same(firepanda, lambda m: m.read_csv(io.BytesIO(path.read_bytes()), compression=method))


def test_bytes_and_encodings(firepanda: ModuleType) -> None:
    same(firepanda, lambda m: m.read_csv(io.BytesIO(b"a,b\n1,2\n")))
    latin = "a,b\n\xe9,2\n".encode("latin-1")
    same(firepanda, lambda m: m.read_csv(io.BytesIO(latin), encoding="latin-1"))
    same(firepanda, lambda m: m.read_csv(io.BytesIO(b"a,b\n\xff,2\n"), encoding_errors="replace"))
    same(firepanda, lambda m: m.read_csv(io.BytesIO(b"\xef\xbb\xbfa,b\n1,2\n")))


def test_a_missing_file_and_a_url_scheme(firepanda: ModuleType, tmp_path: Path) -> None:
    same(firepanda, lambda m: m.read_csv(tmp_path / "nope.csv"))
    same(firepanda, lambda m: m.read_csv(str(tmp_path / "nope.csv"), sep=";"))
    same(firepanda, lambda m: m.read_csv("s3://bucket/frame.csv"))


def test_read_table_splits_on_tabs(firepanda: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "frame.tsv"
    path.write_text("a\tb\n1\t2\n", encoding="utf-8")
    same(firepanda, lambda m: m.read_table(path))
    same(firepanda, lambda m: m.read_table(path, sep=","))
    same(firepanda, lambda m: m.read_table(io.StringIO("a b\n1 2\n"), sep=" "))
    same(firepanda, lambda m: m.read_table(io.StringIO("a\tb\n1\t2\n"), usecols=["b"]))


FIXED = "  a    b  c\n  1  2.5  x\n 10  3.0  y\n"

FWF: dict[str, dict[str, Any]] = {
    "inferred": {},
    "widths": {"widths": [3, 5, 3]},
    "colspecs": {"colspecs": [(0, 3), (3, 8)]},
    "an open end": {"colspecs": [(0, 3), (8, None)]},
    "no header": {"header": None},
    "names": {"names": ["x", "y", "z"], "skiprows": 1},
    "row labels": {"index_col": 0},
    "both": {"colspecs": [(0, 3)], "widths": [3]},
    "an unknown keyword": {"zzz": 1},
    "chunks": {"chunksize": 1},
}


@pytest.mark.parametrize("name", list(FWF))
def test_read_fwf_cuts_as_pandas_cuts(firepanda: ModuleType, name: str) -> None:
    same(firepanda, lambda m: m.read_fwf(io.StringIO(FIXED), **FWF[name]))


def test_read_fwf_with_nothing_to_infer_from(firepanda: ModuleType) -> None:
    same(firepanda, lambda m: m.read_fwf(io.StringIO("")))


def test_the_reader_hands_out_chunks(firepanda: ModuleType) -> None:
    def steps(m: ModuleType) -> list[Any]:
        reader = m.read_csv(io.StringIO("a\n1\n2\n3\n4\n"), iterator=True)
        first = reader.get_chunk(1)
        second = reader.read(2)
        rest = reader.read()
        return [first, second, rest]

    same(firepanda, steps)

    def within(m: ModuleType) -> list[Any]:
        with m.read_csv(io.StringIO("a\n1\n2\n3\n"), chunksize=2) as reader:
            return list(reader)

    same(firepanda, within)


def test_the_signatures_are_pandas(firepanda: ModuleType) -> None:
    import inspect

    import pandas as pd

    for name in ("read_csv", "read_table", "read_fwf"):
        ours = inspect.signature(getattr(firepanda, name)).parameters
        theirs = inspect.signature(getattr(pd, name)).parameters
        assert list(ours) == list(theirs), name


def test_where_firepanda_differs(firepanda: ModuleType) -> None:
    """The differences that stay, because firepanda has no object dtype."""
    frame = firepanda.read_csv(io.StringIO("a,b\nTrue,99999999999999999999999\n,1\n"))
    assert [str(kind) for kind in frame.dtypes] == ["bool", "string"]
    assert frame["a"].tolist() == [True, None]
    frame = firepanda.read_csv(io.StringIO("a;b\n1;2\n"), sep=None)
    assert list(frame.columns) == ["a", "b"]
