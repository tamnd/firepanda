"""`read_sas` and the SAS readers, compared with pandas on pandas' own test files.

The files under `data/sas` are the SAS7BDAT and XPORT files pandas tests its
readers with, gzipped, plus one XPORT file with text columns that pyreadstat
wrote. Each test unpacks a file under its own name, reads it with both
libraries under the same options, and compares the types, labels and values of
the frames, or the type and words of the error. The readers need no package
beyond firepanda, so only the comparison needs pandas.
"""

from __future__ import annotations

import gzip
import importlib
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

DATA = Path(__file__).parent / "data" / "sas"
FILES = sorted(p.name.removesuffix(".gz") for p in DATA.glob("*.gz"))


def unpacked(name: str, where: Path) -> Path:
    path = where / name
    path.write_bytes(gzip.decompress((DATA / f"{name}.gz").read_bytes()))
    return path


def found(lib: ModuleType, path: str) -> Any:
    module, _, name = f"{lib.__name__}.{path}".rpartition(".")
    return getattr(importlib.import_module(module), name)


@needs_pandas
@pytest.mark.parametrize(
    "path",
    [
        "read_sas",
        "io.sas.read_sas",
        "io.sas.sasreader.SASReader",
        "io.sas.sas_xport.XportReader",
        "io.sas.sas7bdat.SAS7BDATReader",
    ],
)
def test_the_signature_is_pandas(path: str) -> None:
    import pandas

    ours = inspect.signature(found(fp, path))
    theirs = inspect.signature(found(pandas, path))
    assert list(ours.parameters) == list(theirs.parameters)
    assert [p.kind for p in ours.parameters.values()] == [
        p.kind for p in theirs.parameters.values()
    ]
    assert [p.default for p in ours.parameters.values()] == [
        p.default for p in theirs.parameters.values()
    ]


def gap(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value) or str(value) == "NaT"


def shown(frame: Any) -> Any:
    """The types, labels and values of a frame.

    firepanda cannot hold a column of object dtype that is empty or whose values
    are all missing and gives it the str dtype, so such a column shows as
    missing either way.
    """
    dtypes = []
    values = []
    for i in range(frame.shape[1]):
        cells = frame.iloc[:, i].tolist()
        kind = str(frame.dtypes.iloc[i]).replace("string", "str")
        if all(gap(v) for v in cells) and kind in ("str", "object"):
            kind = "missing"
        dtypes.append(kind)
        values.append(["nan" if gap(v) else repr(v) for v in cells])
    return dtypes, [str(c) for c in frame.columns], [str(i) for i in frame.index], values


def outcome(call: Callable[[], Any]) -> Any:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = call()
            if isinstance(got, list):
                return [shown(chunk) for chunk in got]
            return shown(got)
    except Exception as err:
        return type(err).__name__, str(err)


def chunks(reader: Any) -> list[Any]:
    with reader:
        return list(reader)


READS: list[dict[str, Any]] = [
    {},
    {"encoding": "infer"},
    {"encoding": "latin-1"},
    {"encoding": "utf-8"},
]


@needs_pandas
@pytest.mark.parametrize("options", READS, ids=str)
@pytest.mark.parametrize("name", FILES)
def test_a_file_reads_as_pandas_reads_it(
    name: str, options: dict[str, Any], tmp_path: Path
) -> None:
    import pandas

    path = unpacked(name, tmp_path)
    fmt = {"format": "xport"} if name.endswith(".cpt") else {}

    def read(m: ModuleType) -> Any:
        return outcome(lambda: m.read_sas(path, **fmt, **options))

    assert read(fp) == read(pandas)


@needs_pandas
@pytest.mark.parametrize("size", [10, 1000])
@pytest.mark.parametrize("name", FILES)
def test_the_chunks_are_pandas_chunks(name: str, size: int, tmp_path: Path) -> None:
    import pandas

    path = unpacked(name, tmp_path)
    fmt = {"format": "xport"} if name.endswith(".cpt") else {}

    def read(m: ModuleType) -> Any:
        return outcome(lambda: chunks(m.read_sas(path, chunksize=size, **fmt)))

    assert read(fp) == read(pandas)


@needs_pandas
@pytest.mark.parametrize(
    ("name", "index"),
    [("airline.sas7bdat", "YEAR"), ("SSHSV1_A.xpt", "SEQN"), ("chars.xpt", "txt")],
)
def test_an_index_column_is_pandas(name: str, index: str, tmp_path: Path) -> None:
    import pandas

    path = unpacked(name, tmp_path)

    def read(m: ModuleType) -> Any:
        return outcome(lambda: m.read_sas(path, index=index, encoding="utf-8"))

    assert read(fp) == read(pandas)


@needs_pandas
@pytest.mark.parametrize("name", ["test1.sas7bdat", "paxraw_d_short.xpt"])
def test_an_iterator_reads_as_pandas(name: str, tmp_path: Path) -> None:
    import pandas

    path = unpacked(name, tmp_path)

    def read(m: ModuleType) -> Any:
        def steps() -> list[Any]:
            with m.read_sas(path, iterator=True) as reader:
                out = [reader.read(3), reader.read(0), reader.read()]
                if hasattr(reader, "get_chunk"):
                    return out
                return [*out, reader.read(2)]

        return outcome(steps)

    assert read(fp) == read(pandas)


@needs_pandas
def test_get_chunk_is_pandas(tmp_path: Path) -> None:
    import pandas

    path = unpacked("paxraw_d_short.xpt", tmp_path)

    def read(m: ModuleType) -> Any:
        def steps() -> list[Any]:
            with m.read_sas(path, chunksize=30) as reader:
                return [reader.get_chunk(), reader.get_chunk(5), next(reader)]

        return outcome(steps)

    assert read(fp) == read(pandas)


@needs_pandas
def test_a_buffer_and_a_gzip_file_read_as_pandas(tmp_path: Path) -> None:
    import pandas

    packed = tmp_path / "airline.sas7bdat.gz"
    packed.write_bytes((DATA / "airline.sas7bdat.gz").read_bytes())
    raw = gzip.decompress(packed.read_bytes())
    calls: list[Callable[[ModuleType], Any]] = [
        lambda m: m.read_sas(packed),
        lambda m: m.read_sas(str(packed), compression="gzip"),
        lambda m: m.read_sas(io.BytesIO(raw), format="sas7bdat"),
        lambda m: m.read_sas(io.BytesIO(raw), format="SAS7BDAT"),
        lambda m: m.read_sas(io.BytesIO(raw)),
    ]
    for call in calls:
        assert outcome(lambda c=call: c(fp)) == outcome(lambda c=call: c(pandas))


@needs_pandas
def test_the_errors_are_pandas(tmp_path: Path) -> None:
    import pandas

    sas = unpacked("airline.sas7bdat", tmp_path)
    xpt = unpacked("SSHSV1_A.xpt", tmp_path)
    other = tmp_path / "airline.csv"
    other.write_bytes(sas.read_bytes())
    calls: list[Callable[[ModuleType], Any]] = [
        lambda m: m.read_sas(sas, format="csv"),
        lambda m: m.read_sas(other),
        lambda m: m.read_sas(tmp_path / "missing.xpt"),
        lambda m: m.read_sas(xpt, format="sas7bdat"),
        lambda m: m.read_sas(sas, format="xport"),
        lambda m: m.read_sas(xpt, encoding="infer"),
        lambda m: m.read_sas(sas, encoding="no-such-codec"),
        lambda m: m.read_sas(io.BytesIO(b""), format="xport"),
        lambda m: m.read_sas(io.BytesIO(b"x" * 400), format="sas7bdat"),
    ]
    for call in calls:
        assert outcome(lambda c=call: c(fp)) == outcome(lambda c=call: c(pandas))


@needs_pandas
def test_the_file_details_are_pandas(tmp_path: Path) -> None:
    import pandas

    sas = unpacked("productsales.sas7bdat", tmp_path)
    xpt = unpacked("paxraw_d_short.xpt", tmp_path)

    def details(m: ModuleType) -> Any:
        with m.read_sas(sas, iterator=True) as reader:
            found = [
                reader.column_names,
                reader.column_formats,
                [(c.col_id, c.name, c.label, c.format, c.ctype, c.length) for c in reader.columns],
                reader.inferred_encoding,
                str(reader.date_created),
                str(reader.date_modified),
                reader.creator_proc,
                reader.row_count,
                reader.U64,
                reader.byte_order,
                reader.compression,
                list(reader.column_data_lengths()),
                list(reader.column_data_offsets()),
                list(reader.column_types()),
            ]
        with m.read_sas(xpt, iterator=True) as reader:
            found += [
                reader.columns,
                reader.nobs,
                reader.record_length,
                reader.fields,
                reader.member_info,
                reader.file_info,
            ]
        return found

    assert details(fp) == details(pandas)


def test_the_readers_are_where_pandas_keeps_them() -> None:
    assert fp.io.sas.sas_xport.XportReader.__module__ == "firepanda.io.sas.sas_xport"
    assert fp.io.sas.sas7bdat.SAS7BDATReader.__module__ == "firepanda.io.sas.sas7bdat"
    assert issubclass(fp.io.sas.sas_xport.XportReader, fp.io.sas.sasreader.SASReader)
    assert fp.io.api.read_sas is fp.read_sas


def test_repeated_column_names_are_refused(tmp_path: Path) -> None:
    """A frame names each column once, so a file that repeats names cannot be read."""
    path = unpacked("test1.sas7bdat", tmp_path)
    reader = fp.read_sas(path, iterator=True)
    reader.column_names[1] = reader.column_names[0]
    with pytest.raises(NotImplementedError, match="column names repeat"):
        reader.read()
