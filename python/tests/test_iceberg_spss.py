"""`read_iceberg`, `DataFrame.to_iceberg` and `read_spss`, compared with pandas.

Iceberg runs against a SQL catalog in a temporary SQLite file, which pyiceberg
ships, so a table is written and read back by both libraries without a
server. SPSS reads a file pyreadstat writes. The signatures and the sentences
for a missing package are checked everywhere.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def installed(*names: str) -> bool:
    try:
        return all(importlib.util.find_spec(name) is not None for name in names)
    except ModuleNotFoundError:
        return False


needs_iceberg = pytest.mark.skipif(
    not installed("pandas", "pyarrow", "pyiceberg.catalog.sql", "sqlalchemy"),
    reason="pandas, pyarrow, pyiceberg or SQLAlchemy is not installed",
)
needs_pyreadstat = pytest.mark.skipif(
    not installed("pandas", "pyarrow", "pyreadstat"),
    reason="pandas, pyarrow or pyreadstat is not installed",
)


@needs_pandas
@pytest.mark.parametrize("path", ["read_iceberg", "DataFrame.to_iceberg", "read_spss"])
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


@pytest.mark.parametrize(
    ("module", "call"),
    [
        ("pyiceberg", lambda: fp.read_iceberg("ns.t")),
        ("pyiceberg", lambda: fp.DataFrame({"a": [1]}).to_iceberg("ns.t")),
        ("pyreadstat", lambda: fp.read_spss("x.sav")),
    ],
)
def test_a_missing_package_is_pandas_sentence(
    monkeypatch: Any, module: str, call: Callable[[], Any]
) -> None:
    for name in list(sys.modules):
        if name == module or name.startswith(module + "."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError, match="Use pip or conda to install"):
        call()


def shown(frame: Any) -> Any:
    """The types, labels and values of a frame, with firepanda's `string` read as `str`."""
    dtypes = [str(t).replace("string", "str") for t in frame.dtypes]
    values = [[str(v) for v in frame.iloc[:, i].tolist()] for i in range(frame.shape[1])]
    return dtypes, [str(c) for c in frame.columns], [str(i) for i in frame.index], values


def catalog(tmp_path: Path, name: str) -> dict[str, str]:
    import pyiceberg.catalog

    props = {
        "type": "sql",
        "uri": f"sqlite:///{tmp_path}/{name}.db",
        "warehouse": f"file://{tmp_path}",
    }
    pyiceberg.catalog.load_catalog(name, **props).create_namespace("ns")
    return props


def frame(m: ModuleType) -> Any:
    return m.DataFrame(
        {
            "a": [1, 2],
            "s": ["x", None],
            "f": [1.5, None],
            "t": m.to_datetime(["2020-01-01", None]),
            "b": [True, False],
        }
    )


READS: list[dict[str, Any]] = [
    {},
    {"columns": ["a", "s"]},
    {"row_filter": "a > 1"},
    {"limit": 1},
    {"case_sensitive": False, "columns": ["A"]},
]


@needs_iceberg
@pytest.mark.parametrize("options", READS)
def test_a_table_reads_back_as_pandas_reads_it(tmp_path: Path, options: dict[str, Any]) -> None:
    import pandas

    def made(m: ModuleType) -> Any:
        props = catalog(tmp_path, m.__name__)
        frame(m).to_iceberg("ns.t", catalog_name=m.__name__, catalog_properties=props)
        frame(m).to_iceberg("ns.t", catalog_name=m.__name__, catalog_properties=props, append=True)
        read = m.read_iceberg("ns.t", catalog_name=m.__name__, catalog_properties=props, **options)
        return shown(read)

    assert made(fp) == made(pandas)


@needs_iceberg
def test_labels_are_written_as_a_column(tmp_path: Path) -> None:
    import pandas

    def made(m: ModuleType) -> Any:
        props = catalog(tmp_path, m.__name__)
        labelled = m.DataFrame({"a": [1, 2]}, index=["p", "q"])
        labelled.to_iceberg("ns.t", catalog_name=m.__name__, catalog_properties=props)
        labelled.to_iceberg("ns.t", catalog_name=m.__name__, catalog_properties=props)
        return shown(m.read_iceberg("ns.t", catalog_name=m.__name__, catalog_properties=props))

    assert made(fp) == made(pandas)


@pytest.fixture
def sav(tmp_path: Path) -> Path:
    import pandas
    import pyreadstat

    path = tmp_path / "x.sav"
    pyreadstat.write_sav(
        pandas.DataFrame(
            {
                "a": [1.0, 2.0, None],
                "s": ["x", "y", ""],
                "t": pandas.to_datetime(["2020-01-01", None, "2021-01-01"]),
            }
        ),
        str(path),
        variable_value_labels={"a": {1.0: "one", 2.0: "two"}},
        column_labels=["A", "S", "T"],
    )
    return path


SPSS: list[dict[str, Any]] = [
    {},
    {"convert_categoricals": False},
    {"usecols": ["s"]},
    {"usecols": "s"},
    {"dtype_backend": "numpy_nullable"},
    {"dtype_backend": "pyarrow"},
    {"dtype_backend": "x"},
]


@needs_pyreadstat
@pytest.mark.parametrize("options", SPSS)
def test_an_spss_file_reads_as_pandas_reads_it(sav: Path, options: dict[str, Any]) -> None:
    import pandas

    def read(m: ModuleType) -> Any:
        try:
            got = m.read_spss(sav, **options)
        except Exception as err:
            return isinstance(err, ValueError), isinstance(err, TypeError), str(err)
        return shown(got), got.attrs["column_labels"], sorted(got.attrs)

    assert read(fp) == read(pandas)
