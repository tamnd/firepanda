"""`DataFrame.to_stata` and the Stata writers, compared with pandas byte for byte.

Each test writes the same frame with both libraries under the same options
and a fixed time stamp, then compares the bytes of the files and the warnings
raised on the way, or the type and words of the error. The writer needs no
package beyond firepanda, so only the comparison needs pandas.
"""

from __future__ import annotations

import datetime as dt
import gzip
import importlib.util
import inspect
import io
import warnings
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

VERSIONS = [114, 117, 118, 119]
STAMP = dt.datetime(2024, 3, 4, 5, 6)


@needs_pandas
@pytest.mark.parametrize(
    "path",
    [
        "DataFrame.to_stata",
        "io.stata.StataWriter",
        "io.stata.StataWriter117",
        "io.stata.StataWriterUTF8",
        "io.stata.StataStrLWriter",
        "io.stata.StataValueLabel",
        "io.stata.StataNonCatValueLabel",
    ],
)
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
    assert [p.kind for p in ours.parameters.values()] == [
        p.kind for p in theirs.parameters.values()
    ]


def outcome(m: ModuleType, build: Callable[[Any], Any], **options: Any) -> Any:
    """The bytes and warnings of a write, or the type and words of its error."""
    buf = io.BytesIO()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build(m).to_stata(buf, time_stamp=STAMP, **options)
    except Exception as err:
        return type(err).__name__, str(err)
    # A file another test left open can be collected mid write and warn here.
    said = [w for w in caught if not issubclass(w.category, ResourceWarning)]
    return buf.getvalue(), [(type(w.message).__name__, str(w.message)) for w in said]


def mixed(m: Any) -> Any:
    return m.DataFrame(
        {
            "g": m.Series([1, 2, None], dtype="float64"),
            "b": m.Series([1, 2, 3], dtype="int8"),
            "h": m.Series([1, 300, 3], dtype="int16"),
            "l": m.Series([1, 70000, 3], dtype="int32"),
            "q": [1, 2, 3],
            "big": [1, 2**40, 3],
            "f": m.Series([1.5, None, 3], dtype="float32"),
            "d": [1.5, 2.5, None],
            "s": ["x", "yy", None],
            "bo": [True, False, True],
            "t": m.to_datetime(["2020-01-01 00:00:00", None, "2021-05-06 07:08:09"]),
            "c": m.Categorical(["a", "b", None]),
            "ni": m.Series([1, None, 3], dtype="Int64"),
            "u8": m.Series([1, 200, 3], dtype="uint8"),
            "bad name!": [1, 2, 3],
            "in": [1, 2, 3],
        }
    )


def dates(m: Any) -> Any:
    moments = m.to_datetime(["1959-03-04", None, "2021-11-30"])
    return m.DataFrame({k: moments for k in ("tc", "td", "tw", "tm", "tq", "th", "ty")})


def one(m: Any) -> Any:
    return m.DataFrame({"a": [1]})


WRITES: list[tuple[Callable[[Any], Any], dict[str, Any]]] = [
    (mixed, {}),
    (mixed, {"write_index": False}),
    (mixed, {"byteorder": ">"}),
    (mixed, {"byteorder": "little", "data_label": "lab", "variable_labels": {"b": "Bee"}}),
    (mixed, {"value_labels": {"b": {1: "one", 2: "two"}, "q": {3: "three"}}}),
    (dates, {"convert_dates": {k: k for k in ("tc", "td", "tw", "tm", "tq", "th", "ty")}}),
    (lambda m: m.DataFrame({"a": ["x" * 3000, "short", ""], "b": ["p", "é", "q"]}), {}),
    (lambda m: m.DataFrame({"a": m.Series([1, 2**60], dtype="uint64")}), {}),
    (lambda m: m.DataFrame({"a": [1, 2**60]}), {}),
    (lambda m: m.DataFrame({"a": m.Categorical([1, 2, 1])}), {}),
    (lambda m: m.DataFrame({"a": m.Series([True, None], dtype="boolean")}), {}),
    (lambda m: m.DataFrame({"a": m.Series([1.5, None], dtype="Float64")}), {}),
    (
        lambda m: m.DataFrame({"a": m.Series([], dtype="int64"), "b": m.Series([], dtype="str")}),
        {"write_index": False},
    ),
    (lambda m: m.DataFrame({"a": [1, 2]}, index=m.Index([5, 6], name="ix")), {}),
    (lambda m: m.DataFrame({"é": ["ü"]}), {}),
    (lambda m: m.DataFrame({"a" * 40: [1], "a" * 33: [2]}), {}),
    (lambda m: m.DataFrame({"a": [float("inf")]}), {}),
    (lambda m: m.DataFrame({"a": ["x" * 300]}), {}),
    (lambda m: m.DataFrame({"a": m.to_datetime(["2020-01-01"]).tz_localize("UTC")}), {}),
    (one, {"version": 115}),
    (one, {"value_labels": {"zz": {1: "a"}}}),
    (one, {"value_labels": {"a": {1: 2}}}),
    (lambda m: m.DataFrame({"a": ["x"]}), {"value_labels": {"a": {1: "a"}}}),
    (one, {"convert_dates": {"a": "tz"}}),
    (one, {"convert_dates": {0: "td"}}),
    (one, {"convert_dates": {"zz": "td"}}),
    (one, {"variable_labels": {"a": "x" * 81}}),
    (one, {"data_label": "y" * 90}),
    (one, {"byteorder": "middle"}),
    (one, {"time_stamp": "x"}),
]


@needs_pandas
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize(("build", "options"), WRITES)
def test_a_file_is_the_bytes_pandas_writes(
    version: int, build: Callable[[Any], Any], options: dict[str, Any]
) -> None:
    import pandas

    options = {"version": version, **options}
    stamp = options.pop("time_stamp", STAMP)

    def write(m: ModuleType) -> Any:
        return outcome(m, build, **options) if stamp is STAMP else _stamped(m, build, stamp)

    assert write(fp) == write(pandas)


def _stamped(m: ModuleType, build: Callable[[Any], Any], stamp: Any) -> Any:
    try:
        build(m).to_stata(io.BytesIO(), time_stamp=stamp)
    except Exception as err:
        return type(err).__name__, str(err)
    return None


@needs_pandas
@pytest.mark.parametrize("version", [117, 118, 119, None])
@pytest.mark.parametrize("byteorder", ["<", ">"])
def test_long_strings_are_the_bytes_pandas_writes(version: int | None, byteorder: str) -> None:
    import pandas

    def build(m: Any) -> Any:
        return m.DataFrame({"a": ["x" * 300, "s", "", None], "b": ["p", "q", "p", "r"]})

    options = {"version": version, "byteorder": byteorder, "convert_strl": ["b"]}
    assert outcome(fp, build, **options) == outcome(pandas, build, **options)


@needs_pandas
def test_format_114_refuses_long_strings_as_pandas_does() -> None:
    import pandas

    options = {"version": 114, "convert_strl": ["a"]}
    assert outcome(fp, one, **options) == outcome(pandas, one, **options)


@needs_pandas
def test_a_path_and_a_compressed_file_are_what_pandas_writes(tmp_path: Path) -> None:
    import pandas

    for m in (fp, pandas):
        frame = mixed(m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame.to_stata(tmp_path / f"{m.__name__}.dta", time_stamp=STAMP)
            frame.to_stata(str(tmp_path / f"{m.__name__}.dta.gz"), time_stamp=STAMP)
    assert (tmp_path / "firepanda.dta").read_bytes() == (tmp_path / "pandas.dta").read_bytes()
    ours = gzip.decompress((tmp_path / "firepanda.dta.gz").read_bytes())
    assert ours == gzip.decompress((tmp_path / "pandas.dta.gz").read_bytes())
    with pytest.raises(ValueError) as theirs:
        pandas.DataFrame({"a": [1]}).to_stata(tmp_path / "x.dta", storage_options={"a": 1})
    with pytest.raises(ValueError, match=str(theirs.value)):
        fp.DataFrame({"a": [1]}).to_stata(tmp_path / "x.dta", storage_options={"a": 1})


@needs_pandas
def test_a_written_file_reads_back() -> None:
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mixed(fp).to_stata(buf, version=118, write_index=False)
        back = fp.read_stata(io.BytesIO(buf.getvalue()))
    assert back["s"].tolist() == ["x", "yy", ""]
    assert back["c"].tolist()[:2] == ["a", "b"]
    assert back["ni"].tolist()[0] == 1


@needs_pandas
def test_a_writer_writes_what_to_stata_writes() -> None:
    import pandas

    def written(m: ModuleType) -> bytes:
        buf = io.BytesIO()
        frame = m.DataFrame({"a": [1, 2], "s": ["x", "y" * 3000]})
        writer = m.io.stata.StataWriterUTF8(buf, frame, time_stamp=STAMP, convert_strl=["s"])
        writer.write_file()
        return buf.getvalue()

    assert written(fp) == written(pandas)
