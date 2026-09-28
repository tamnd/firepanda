"""Parquet, Feather and ORC, compared with pandas in both directions.

Every file is written by one library and read by the other as well as by
itself, and what is read has to print as what pandas reads back from the file
pandas wrote. That is what the `pandas` metadata key is for: the row labels,
each column's type and `attrs` travel in it, so a file crosses between the two
libraries without either noticing which one wrote it.
"""

from __future__ import annotations

import importlib.util
import io
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None or importlib.util.find_spec("pyarrow") is None,
    reason="pandas and pyarrow are both needed",
)


def categories(m: ModuleType, frame: Any) -> Any:
    frame["c"] = frame["c"].astype(m.CategoricalDtype(["b", "a", "z"], ordered=True))
    return frame


FRAMES: dict[str, Callable[[ModuleType], Any]] = {
    "numbers": lambda m: m.DataFrame({"a": [1, 2, 3], "b": [1.5, float("nan"), -2.0]}),
    "text with a gap": lambda m: m.DataFrame({"s": ["x", None, "zz"], "n": [1, 2, 3]}),
    "booleans": lambda m: m.DataFrame({"b": [True, False, True]}),
    "instants": lambda m: m.DataFrame(
        {"t": m.to_datetime(["2024-01-01", "2024-02-03 04:05:06", None], format="ISO8601")}
    ),
    "zoned instants": lambda m: m.DataFrame(
        {"t": m.to_datetime(["2024-01-01", None]).tz_localize("Asia/Tokyo")}
    ),
    "spans": lambda m: m.DataFrame({"d": m.to_timedelta(["1 days", None, "2 hours"])}),
    "categories": lambda m: categories(m, m.DataFrame({"c": ["b", "a", "b"], "n": [1, 2, 3]})),
    "small integers": lambda m: m.DataFrame({"a": [1, 2, 3]}).astype("int8"),
    "labelled rows": lambda m: m.DataFrame(
        {"a": [1.0, 2.0]}, index=m.Index(["r", "s"], name="row")
    ),
    "unnamed labels": lambda m: m.DataFrame({"a": [1, 2]}, index=m.Index(["r", "s"])),
    "integer labels": lambda m: m.DataFrame({"a": [1, 2, 3]}, index=m.Index([5, 1, 9])),
    "a range with a name": lambda m: m.DataFrame({"a": [1, 2]}).rename_axis("r"),
    "instant rows": lambda m: m.DataFrame(
        {"a": [1, 2, 3]}, index=m.date_range("2024-01-01", periods=3, tz="UTC", name="when")
    ),
    "a slice": lambda m: m.DataFrame({"a": list(range(20)), "s": [str(k) for k in range(20)]}).iloc[
        7:12
    ],
    "no rows": lambda m: m.DataFrame({"a": [1, 2]}).iloc[:0],
}

ENGINES = ("firepanda", "pandas")


def engine(firepanda: ModuleType, name: str) -> ModuleType:
    import pandas as pd

    return firepanda if name == "firepanda" else pd


def shown(frame: Any) -> tuple[str, list[str], Any]:
    """What is compared: the printed frame, its column types and its row label names."""
    kinds = [str(kind).replace("string", "str") for kind in frame.dtypes]
    return repr(frame), kinds, list(frame.index.names)


@pytest.mark.parametrize("reader", ENGINES)
@pytest.mark.parametrize("writer", ENGINES)
@pytest.mark.parametrize("name", list(FRAMES))
def test_parquet_crosses_between_the_libraries(
    firepanda: ModuleType, name: str, writer: str, reader: str
) -> None:
    """A file either library writes reads back in either as pandas reads its own."""
    import pandas as pd

    data = FRAMES[name](engine(firepanda, writer)).to_parquet()
    back = engine(firepanda, reader).read_parquet(io.BytesIO(data))
    expected = pd.read_parquet(io.BytesIO(FRAMES[name](pd).to_parquet()))
    assert shown(back) == shown(expected)


@pytest.mark.parametrize("reader", ENGINES)
@pytest.mark.parametrize("writer", ENGINES)
@pytest.mark.parametrize("name", list(FRAMES))
def test_feather_crosses_between_the_libraries(
    firepanda: ModuleType, name: str, writer: str, reader: str, tmp_path: Path
) -> None:
    """The same for Feather, which pyarrow writes with the same metadata."""
    import pandas as pd

    def written(m: ModuleType, file: str) -> Path:
        target = tmp_path / file
        FRAMES[name](m).to_feather(target)
        return target

    back = engine(firepanda, reader).read_feather(written(engine(firepanda, writer), "x.feather"))
    expected = pd.read_feather(written(pd, "y.feather"))
    assert shown(back) == shown(expected)


@pytest.mark.parametrize("index", [True, False, None])
@pytest.mark.parametrize("name", ["numbers", "labelled rows", "a slice", "integer labels"])
def test_the_index_parameter_stores_labels_as_pandas_does(
    firepanda: ModuleType, name: str, index: Any
) -> None:
    """The fields and the metadata each writes, and what pandas reads back."""
    import json

    import pandas as pd
    import pyarrow.parquet as pq

    def written(m: ModuleType) -> tuple[list[str], Any, str]:
        data = FRAMES[name](m).to_parquet(index=index)
        table = pq.read_table(io.BytesIO(data))
        stored = json.loads(table.schema.metadata[b"pandas"])["index_columns"]
        return table.column_names, stored, repr(pd.read_parquet(io.BytesIO(data)))

    assert written(firepanda) == written(pd)


@pytest.mark.parametrize("compression", ["snappy", "gzip", "brotli", "zstd", "lz4", None])
def test_each_codec_writes_what_pandas_writes(firepanda: ModuleType, compression: Any) -> None:
    """The codec is pyarrow's, so the column chunks say the same one."""
    import pandas as pd
    import pyarrow.parquet as pq

    def codec(m: ModuleType) -> tuple[str, str]:
        data = FRAMES["text with a gap"](m).to_parquet(compression=compression)
        chunk = pq.ParquetFile(io.BytesIO(data)).metadata.row_group(0).column(0)
        return chunk.compression, repr(m.read_parquet(io.BytesIO(data)))

    assert codec(firepanda) == codec(pd)


def test_a_path_a_handle_and_bytes(firepanda: ModuleType, tmp_path: Path) -> None:
    """`to_parquet` writes to a path, a text path or a handle, or hands back bytes."""
    frame = FRAMES["labelled rows"](firepanda)
    frame.to_parquet(tmp_path / "a.parquet")
    frame.to_parquet(str(tmp_path / "b.parquet"))
    handle = io.BytesIO()
    assert frame.to_parquet(handle) is None
    data = frame.to_parquet()
    assert isinstance(data, bytes)
    for source in (tmp_path / "a.parquet", str(tmp_path / "b.parquet"), io.BytesIO(data)):
        assert firepanda.read_parquet(source).equals(frame)
    handle.seek(0)
    assert firepanda.read_parquet(handle).equals(frame)


@pytest.mark.parametrize(
    "options",
    [
        {"columns": ["a"]},
        {"columns": ["s", "a"]},
        {"filters": [("a", ">", 5)]},
        {"filters": [("s", "in", ["3", "4"])], "columns": ["s"]},
    ],
    ids=str,
)
def test_reading_part_of_a_file(firepanda: ModuleType, options: dict[str, Any]) -> None:
    """`columns` and `filters` pick what is read, and the labels come along as in pandas."""
    import pandas as pd

    def read(m: ModuleType) -> tuple[str, list[str], Any]:
        frame = m.DataFrame(
            {"a": list(range(10)), "s": [str(k) for k in range(10)]},
            index=m.Index([f"r{k}" for k in range(10)], name="k"),
        )
        return shown(m.read_parquet(io.BytesIO(frame.to_parquet()), **options))

    assert read(firepanda) == read(pd)


def test_partitioned_folders_read_back(firepanda: ModuleType, tmp_path: Path) -> None:
    """`partition_cols` splits the rows into folders, and reading the folder puts them back."""
    import pandas as pd

    def read(m: ModuleType) -> tuple[list[str], list[int]]:
        folder = tmp_path / m.__name__
        frame = m.DataFrame({"g": ["x", "y", "x", "y"], "v": [1, 2, 3, 4]})
        frame.to_parquet(folder, partition_cols=["g"], index=False)
        back = m.read_parquet(folder)
        parts = sorted(path.name for path in folder.iterdir())
        return parts, sorted(back["v"].tolist())

    assert read(firepanda) == read(pd)


def test_attrs_travel_in_every_direction(firepanda: ModuleType) -> None:
    """`attrs` goes in the metadata, and each library reads the other's."""
    import pandas as pd

    for writer, reader in ((firepanda, pd), (pd, firepanda), (firepanda, firepanda)):
        frame = writer.DataFrame({"a": [1]})
        frame.attrs = {"source": "sensor", "rate": 2}
        back = reader.read_parquet(io.BytesIO(frame.to_parquet()))
        assert back.attrs == {"source": "sensor", "rate": 2}


def test_arrow_types_when_asked(firepanda: ModuleType) -> None:
    """A `dtype_backend` keeps a missing integer an integer, rather than a float."""
    frame = firepanda.DataFrame({"a": [1, None, 3]})
    data = frame.to_parquet()
    assert str(firepanda.read_parquet(io.BytesIO(data)).dtypes["a"]) == "float64"
    for backend in ("numpy_nullable", "pyarrow"):
        back = firepanda.read_parquet(io.BytesIO(data), dtype_backend=backend)
        assert str(back.dtypes["a"]) == "int64"
        assert back["a"].isna().tolist() == [False, True, False]


@pytest.mark.parametrize("name", ["numbers", "text with a gap", "instants", "booleans", "no rows"])
def test_orc_reads_back_as_pandas_reads_it(firepanda: ModuleType, name: str) -> None:
    """ORC keeps no labels, and a file written by either reads the same in both."""
    import pandas as pd

    expected = pd.read_orc(io.BytesIO(FRAMES[name](pd).to_orc()))
    for writer in (firepanda, pd):
        data = FRAMES[name](writer).to_orc()
        for reader in (firepanda, pd):
            assert shown(reader.read_orc(io.BytesIO(data))) == shown(expected)


MISTAKES: dict[str, Callable[[ModuleType, Path], Any]] = {
    "an unknown engine": lambda m, p: m.DataFrame({"a": [1]}).to_parquet(engine="x"),
    "reading with an unknown engine": lambda m, p: m.read_parquet(p / "x.pq", engine="x"),
    "storage options on a local file": lambda m, p: m.DataFrame({"a": [1]}).to_parquet(
        p / "x.pq", storage_options={"k": 1}
    ),
    "an unknown backend": lambda m, p: m.read_parquet(
        io.BytesIO(m.DataFrame({"a": [1]}).to_parquet()), dtype_backend="numpy"
    ),
    "orc with labels": lambda m, p: m.DataFrame({"a": [1]}, index=m.Index(["x"])).to_orc(),
    "orc with a named range": lambda m, p: m.DataFrame({"a": [1]}).rename_axis("r").to_orc(),
    "orc with another engine": lambda m, p: m.DataFrame({"a": [1]}).to_orc(engine="x"),
    "orc with categories": lambda m, p: categories(
        m, m.DataFrame({"c": ["a", "b"], "n": [1, 2]})
    ).to_orc(),
    "a feather file that is not there": lambda m, p: m.read_feather(p / "missing.feather"),
    "a parquet file that is not there": lambda m, p: m.read_parquet(str(p / "missing.pq")),
    "an orc file that is not there": lambda m, p: m.read_orc(p / "missing.orc"),
    "storage options on a feather file": lambda m, p: m.DataFrame({"a": [1]}).to_feather(
        p / "x.feather", storage_options={"k": 1}
    ),
    "storage options reading feather": lambda m, p: m.read_feather(
        p / "x.feather", storage_options={"k": 1}
    ),
}


@pytest.mark.parametrize("name", list(MISTAKES))
def test_a_mistake_raises_what_pandas_raises(
    firepanda: ModuleType, name: str, tmp_path: Path
) -> None:
    """The same kind of error with the same words."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        MISTAKES[name](pd, tmp_path)
    with pytest.raises(Exception) as mine:
        MISTAKES[name](firepanda, tmp_path)
    assert isinstance(mine.value, type(theirs.value))
    assert str(mine.value) == str(theirs.value)


def test_the_signatures_are_pandas(firepanda: ModuleType) -> None:
    """The three writers and the three readers take pandas' parameters."""
    import inspect

    import pandas as pd

    def names(fn: Any) -> list[tuple[str, Any]]:
        return [(p.name, p.kind) for p in inspect.signature(fn).parameters.values()]

    for path in (
        "DataFrame.to_parquet",
        "DataFrame.to_feather",
        "DataFrame.to_orc",
        "read_parquet",
        "read_feather",
        "read_orc",
    ):
        mine, theirs = firepanda, pd
        for part in path.split("."):
            mine, theirs = getattr(mine, part), getattr(theirs, part)
        assert names(mine) == names(theirs), path
