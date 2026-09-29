"""`read_stata` and `StataReader`, compared with pandas on files pandas writes.

Each test writes a dta file with pandas, in every version pandas can write,
then reads it with both libraries under the same options and compares the
types, the labels and the values of the frames, or the type and words of the
error. The reader needs no package beyond firepanda, so only the comparison
needs pandas.
"""

from __future__ import annotations

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


@needs_pandas
@pytest.mark.parametrize("path", ["read_stata", "io.stata.StataReader"])
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


def shown(frame: Any) -> Any:
    """The types, labels and values of a frame, and the categories of each categorical."""
    dtypes = [str(t).replace("string", "str") for t in frame.dtypes]
    values = [
        [
            str(v) if isinstance(v, float) else repr(v).replace("firepanda.io", "pandas.io")
            for v in frame.iloc[:, i].tolist()
        ]
        for i in range(frame.shape[1])
    ]
    cats = [
        (frame[c].cat.categories.tolist(), frame[c].cat.ordered)
        for c in frame.columns
        if str(frame[c].dtype) == "category"
    ]
    return dtypes, [str(c) for c in frame.columns], [str(i) for i in frame.index], values, cats


def outcome(call: Callable[[], Any]) -> Any:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return shown(call())
    except Exception as err:
        return type(err).__name__, str(err)


def written(version: int, build: Callable[[Any], Any], **options: Any) -> bytes:
    import pandas

    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build(pandas).to_stata(buf, version=version, write_index=False, **options)
    return buf.getvalue()


def mixed(m: Any) -> Any:
    return m.DataFrame(
        {
            "i8": m.Series([1, 2, None], dtype="float64"),
            "b": m.Series([1, 2, 3], dtype="int8"),
            "h": m.Series([1, 300, 3], dtype="int16"),
            "l": m.Series([1, 70000, 3], dtype="int32"),
            "f": m.Series([1.5, None, 3], dtype="float32"),
            "d": [1.5, 2.5, None],
            "s": ["x", "yy", ""],
            "t": m.to_datetime(["2020-01-01 00:00:00", None, "2021-05-06 07:08:09"]),
            "c": m.Categorical(["a", "b", "a"]),
        }
    )


READS: list[dict[str, Any]] = [
    {},
    {"convert_missing": True, "convert_dates": False},
    {"convert_categoricals": False},
    {"preserve_dtypes": False},
    {"columns": ["d", "s"]},
    {"columns": ["c", "b"]},
    {"columns": ["d", "d"]},
    {"columns": ["zz"]},
    {"index_col": "s"},
    {"order_categoricals": False},
    {"convert_dates": False},
    {"chunksize": 0},
    {"chunksize": 1.5},
]


@needs_pandas
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("options", READS)
def test_a_file_reads_as_pandas_reads_it(version: int, options: dict[str, Any]) -> None:
    import pandas

    data = written(version, mixed, convert_dates={"t": "tc"})

    def read(m: ModuleType) -> Any:
        return outcome(lambda: m.read_stata(io.BytesIO(data), **options))

    assert read(fp) == read(pandas)


def dates(m: Any) -> Any:
    moments = m.to_datetime(["1959-03-04", None, "2021-11-30"])
    return m.DataFrame({k: moments for k in ("tc", "td", "tw", "tm", "tq", "th", "ty")})


def long_text(m: Any) -> Any:
    return m.DataFrame({"a": ["x" * 3000, "short", ""], "b": ["p", "é", "q"]})


def labelled(m: Any) -> Any:
    return m.DataFrame({"x": [1, 2, 3, 1], "y": [1, 1, 2, 2], "z": [1.5, 2.5, None, 3.5]})


def empty(m: Any) -> Any:
    return m.DataFrame({"a": m.Series([], dtype="int32"), "f": m.Series([], dtype="float32")})


FILES: list[tuple[Callable[[Any], Any], dict[str, Any]]] = [
    (dates, {"convert_dates": {k: k for k in ("tc", "td", "tw", "tm", "tq", "th", "ty")}}),
    (long_text, {}),
    (long_text, {"convert_strl": ["a"]}),
    (labelled, {"value_labels": {"x": {1: "one", 2: "two", 3: "three"}, "y": {1: "a", 2: "b"}}}),
    (labelled, {"value_labels": {"y": {1: "same", 2: "same"}}}),
    (empty, {}),
]


@needs_pandas
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize(("build", "writing"), FILES)
@pytest.mark.parametrize("options", [{}, {"convert_dates": False}, {"preserve_dtypes": False}])
def test_every_kind_of_column_reads_as_pandas_reads_it(
    version: int, build: Callable[[Any], Any], writing: dict[str, Any], options: dict[str, Any]
) -> None:
    import pandas

    if version < 117 and build is long_text:
        pytest.skip("text over 244 characters needs version 117 or later")
    data = written(version, build, **writing)

    def read(m: ModuleType) -> Any:
        return outcome(lambda: m.read_stata(io.BytesIO(data), **options))

    assert read(fp) == read(pandas)


def chunks(m: ModuleType, data: bytes, **options: Any) -> Any:
    """Everything a reader says about a file, and every chunk it reads in turn."""
    with m.read_stata(io.BytesIO(data), **options) as reader:
        about = [
            reader.data_label,
            reader.variable_labels(),
            {k: {int(a): b for a, b in v.items()} for k, v in reader.value_labels().items()},
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            about.append([shown(chunk) for chunk in reader])
            about.append(outcome(lambda: reader.get_chunk(1)))
        about.append([(type(w.message).__name__, str(w.message)) for w in caught])
    return about


@needs_pandas
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize(
    ("build", "writing"),
    [
        (mixed, {"convert_dates": {"t": "tc"}, "data_label": "lab"}),
        (labelled, {"value_labels": {"x": {3: "three", 1: "one", 2: "two"}}}),
        (labelled, {"value_labels": {"y": {1: "a", 2: "b"}}, "variable_labels": {"y": "Why"}}),
    ],
)
@pytest.mark.parametrize("options", [{"chunksize": 2}, {"iterator": True, "chunksize": 3}])
def test_a_reader_reads_in_chunks_as_pandas_does(
    version: int, build: Callable[[Any], Any], writing: dict[str, Any], options: dict[str, Any]
) -> None:
    import pandas

    data = written(version, build, **writing)
    assert chunks(fp, data, **options) == chunks(pandas, data, **options)


@needs_pandas
def test_a_reader_reads_rows_on_request() -> None:
    import pandas

    data = written(118, mixed, convert_dates={"t": "tc"})

    def read(m: ModuleType) -> Any:
        with m.read_stata(io.BytesIO(data), iterator=True) as reader:
            return [
                shown(reader.read(1)),
                shown(reader.read(1, convert_dates=False, columns=["t", "b"])),
                shown(reader.read(columns=["t", "b"])),
                outcome(lambda: reader.read(1)),
            ]

    assert read(fp) == read(pandas)


@needs_pandas
def test_a_path_and_a_compressed_file_read(tmp_path: Path) -> None:
    import pandas

    data = written(117, mixed, convert_dates={"t": "tc"})
    plain = tmp_path / "x.dta"
    plain.write_bytes(data)
    packed = tmp_path / "x.dta.gz"
    packed.write_bytes(gzip.compress(data))
    for path in (plain, str(plain), packed):
        assert outcome(lambda p=path: fp.read_stata(p)) == outcome(
            lambda p=path: pandas.read_stata(p)
        )
    with pytest.raises(ValueError) as theirs:
        pandas.read_stata(plain, storage_options={"a": 1})
    with pytest.raises(ValueError, match=str(theirs.value)):
        fp.read_stata(plain, storage_options={"a": 1})


def test_a_reader_outside_a_with_block_warns_as_pandas_does() -> None:
    data = io.BytesIO(b"<stata_dta><header><release>116</release>" + bytes(200))
    reader = fp.read_stata(data, iterator=True)
    with (
        pytest.warns(ResourceWarning, match="without using a context manager"),
        pytest.raises(ValueError, match="Version of given Stata file"),
    ):
        reader.read()


def test_an_unknown_version_is_pandas_sentence() -> None:
    with pytest.raises(ValueError, match="Version of given Stata file is 99"):
        fp.read_stata(io.BytesIO(bytes([99]) + bytes(200)))


def test_a_missing_value_is_pandas_missing_value() -> None:
    missing = fp.io.stata.StataMissingValue(101)
    assert (missing.string, missing.value, str(missing)) == (".", 101, ".")
    assert repr(missing) == "<class 'firepanda.io.stata.StataMissingValue'>(.)"
    assert fp.io.stata.StataMissingValue(127).string == ".z"
    assert missing == fp.io.stata.StataMissingValue(101)
    assert fp.io.stata.StataMissingValue.get_base_missing_value("int16") == 32741
    with pytest.raises(ValueError, match="Unsupported dtype"):
        fp.io.stata.StataMissingValue.get_base_missing_value("uint8")
