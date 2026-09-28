"""Pickling frames, columns and indexes, and `to_pickle` and `read_pickle`, compared with pandas.

A firepanda pickle names firepanda's classes, so pandas cannot read it and
the other way round. What is compared is what each library hands back from
its own round trip, printed, and the files each writes: the compression they
are in and the mistakes each refuses.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import io
import pickle
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def dated(m: ModuleType) -> Any:
    return m.to_datetime(["2024-01-01", "2024-02-03 04:05:06", None], format="ISO8601")


FRAMES: dict[str, Callable[[ModuleType], Any]] = {
    "numbers": lambda m: m.DataFrame({"a": [1, 2, 3], "b": [1.5, float("nan"), -2.0]}),
    "text with a gap": lambda m: m.DataFrame({"s": ["x", None, "zz"], "n": [1, 2, 3]}),
    "booleans": lambda m: m.DataFrame({"b": [True, False, True]}),
    "instants": lambda m: m.DataFrame({"t": dated(m)}),
    "zoned instants": lambda m: m.DataFrame({"t": dated(m).tz_localize("Asia/Tokyo")}),
    "spans": lambda m: m.DataFrame({"d": m.to_timedelta(["1 days", None, "2 hours"])}),
    "categories": lambda m: m.DataFrame(
        {"c": m.Series(["b", "a", "b"]).astype(m.CategoricalDtype(["b", "a", "z"], ordered=True))}
    ),
    "small integers": lambda m: m.DataFrame({"a": [1, 2, 3]}).astype("int8"),
    "labelled rows": lambda m: m.DataFrame(
        {"a": [1.0, 2.0]}, index=m.Index(["r", "s"], name="row")
    ),
    "instant rows": lambda m: m.DataFrame(
        {"a": [1, 2, 3]}, index=m.date_range("2024-01-01", periods=3, tz="UTC", name="when")
    ),
    "a slice": lambda m: m.DataFrame({"a": list(range(50)), "s": [str(k) for k in range(50)]}).iloc[
        17:23
    ],
    "no rows": lambda m: m.DataFrame({"a": [1, 2]}).iloc[:0],
    "nothing": lambda m: m.DataFrame(),
}


def unpickled(obj: Any, protocol: int = 5) -> Any:
    return pickle.loads(pickle.dumps(obj, protocol=protocol))


@needs_pandas
@pytest.mark.parametrize("name", list(FRAMES))
def test_a_frame_comes_back_as_pandas_frame_does(firepanda: ModuleType, name: str) -> None:
    """The frame each library reads back prints the same."""
    import pandas as pd

    def shown(m: ModuleType) -> str:
        return repr(unpickled(FRAMES[name](m)))

    assert shown(firepanda) == shown(pd)


@pytest.mark.parametrize("name", list(FRAMES))
@pytest.mark.parametrize("protocol", [2, 4, 5])
def test_a_frame_comes_back_equal(firepanda: ModuleType, name: str, protocol: int) -> None:
    """Every column, type, gap and row label is what was written, at each protocol."""
    frame = FRAMES[name](firepanda)
    back = unpickled(frame, protocol)
    assert back.equals(frame)
    assert list(back.dtypes) == list(frame.dtypes)
    assert list(back.columns) == list(frame.columns)
    assert back.index.equals(frame.index)
    assert list(back.index.names) == list(frame.index.names)


COLUMNS: dict[str, Callable[[ModuleType], Any]] = {
    "named": lambda m: m.Series([1.5, 2.5], name="x"),
    "unnamed": lambda m: m.Series(["a", None]),
    "named by a number": lambda m: m.Series([1, 2], name=0),
    "named by a tuple": lambda m: m.Series([True, False], name=("a", 1)),
    "instants": lambda m: m.Series(dated(m), name="t"),
    "labelled": lambda m: m.Series([1, 2], index=m.Index(["p", "q"], name="k")),
}


@needs_pandas
@pytest.mark.parametrize("name", list(COLUMNS))
def test_a_column_comes_back_as_pandas_column_does(firepanda: ModuleType, name: str) -> None:
    """The column each reads back prints the same, its name included."""
    import pandas as pd

    def shown(m: ModuleType) -> tuple[str, Any]:
        back = unpickled(COLUMNS[name](m))
        return repr(back), back.name

    assert shown(firepanda) == shown(pd)


INDEXES: dict[str, Callable[[ModuleType], Any]] = {
    "integers": lambda m: m.Index([5, 6, 7], name="k"),
    "text": lambda m: m.Index(["a", "b"]),
    "a range": lambda m: m.RangeIndex(0, 10, 3, name="r"),
    "instants": lambda m: m.DatetimeIndex(["2024-01-01", "2024-01-05"], name="d"),
    "spans": lambda m: m.TimedeltaIndex(["1 days", "3 hours"]),
    "pairs": lambda m: m.MultiIndex.from_tuples([(1, "a"), (2, "b")], names=["n", None]),
}


@needs_pandas
@pytest.mark.parametrize("name", list(INDEXES))
def test_an_index_comes_back_as_pandas_index_does(firepanda: ModuleType, name: str) -> None:
    """The index each reads back prints the same and is of the class that was written."""
    import pandas as pd

    def shown(m: ModuleType) -> tuple[str, str]:
        index = INDEXES[name](m)
        back = unpickled(index)
        return repr(back), type(back).__name__ == type(index).__name__

    assert shown(firepanda) == shown(pd)


def test_attrs_and_flags_come_back(firepanda: ModuleType) -> None:
    """`attrs` and a refusal of repeated labels travel with the frame and the column."""
    frame = firepanda.DataFrame({"a": [1, 2]}).set_flags(allows_duplicate_labels=False)
    frame.attrs["source"] = {"file": "x.csv"}
    back = unpickled(frame)
    assert back.attrs == {"source": {"file": "x.csv"}}
    assert not back.flags.allows_duplicate_labels
    column = frame["a"]
    assert unpickled(column).attrs == {"source": {"file": "x.csv"}}
    assert unpickled(firepanda.DataFrame({"a": [1]})).attrs == {}


def test_a_large_frame_comes_back_equal(firepanda: ModuleType) -> None:
    """Many rows of numbers and text, which travel as Arrow buffers rather than values."""
    count = 200_000
    frame = firepanda.DataFrame(
        {"a": list(range(count)), "s": [f"v{k}" for k in range(count)], "f": [0.5] * count}
    )
    assert unpickled(frame).equals(frame)


def test_copies_still_answer_with_copy(firepanda: ModuleType) -> None:
    """`copy.copy` and `copy.deepcopy` keep their own methods rather than pickling."""
    frame = firepanda.DataFrame({"a": [1, 2]})
    assert copy.deepcopy(frame).equals(frame)
    assert copy.copy(frame["a"]).equals(frame["a"])


ENDINGS = [
    "frame.pkl",
    "frame.pkl.gz",
    "frame.pkl.bz2",
    "frame.pkl.xz",
    "frame.pkl.zip",
    "frame.pkl.tar",
    "frame.pkl.tar.gz",
    "frame.pkl.zst",
]


def importable(*names: str) -> bool:
    for name in names:
        try:
            importlib.import_module(name)
        except ImportError:
            continue
        return True
    return False


@needs_pandas
@pytest.mark.parametrize("ending", ENDINGS)
def test_to_pickle_compresses_as_the_name_says(
    firepanda: ModuleType, ending: str, tmp_path: Path
) -> None:
    """The file starts as pandas' does for the same name, and reads back equal."""
    import pandas as pd

    if ending.endswith(".zst") and not importable("zstandard"):
        pytest.skip("pandas reads zstd only through the zstandard package")

    def written(m: ModuleType) -> tuple[bytes, Any]:
        target = tmp_path / m.__name__ / ending
        target.parent.mkdir(exist_ok=True)
        frame = FRAMES["text with a gap"](m)
        frame.to_pickle(target)
        # The magic number, the method and the flags; a bare pickle's length follows.
        start = target.read_bytes()[: 2 if ending.endswith(".pkl") else 4]
        return start, repr(m.read_pickle(target))

    mine, theirs = written(firepanda), written(pd)
    if ending.endswith(".tar"):
        assert mine[1] == theirs[1]
    else:
        assert mine == theirs


def test_zstd_needs_only_the_standard_library(firepanda: ModuleType, tmp_path: Path) -> None:
    """Where Python has `compression.zstd`, firepanda writes zstd without `zstandard`."""
    if not importable("compression.zstd", "zstandard"):
        pytest.skip("no zstd")
    target = tmp_path / "frame.pkl.zst"
    frame = FRAMES["numbers"](firepanda)
    frame.to_pickle(target)
    assert target.read_bytes()[:4] == b"\x28\xb5\x2f\xfd"
    assert firepanda.read_pickle(target).equals(frame)


@needs_pandas
@pytest.mark.parametrize(
    "compression",
    ["gzip", "bz2", "xz", "zip", None, {"method": "gzip", "compresslevel": 1}],
    ids=str,
)
def test_a_named_compression_on_a_handle(firepanda: ModuleType, compression: Any) -> None:
    """A binary handle takes a compression by name, and `read_pickle` reads it back."""
    import pandas as pd

    def written(m: ModuleType) -> tuple[bytes, str]:
        handle = io.BytesIO()
        column = m.Series([1.5, None], name="x")
        m.to_pickle(column, handle, compression=compression)
        start = handle.getvalue()[:2]
        handle.seek(0)
        return start, repr(m.read_pickle(handle, compression=compression))

    assert written(firepanda) == written(pd)


MISTAKES: dict[str, Callable[[ModuleType, Path], Any]] = {
    "an unknown compression": lambda m, p: m.DataFrame({"a": [1]}).to_pickle(
        p / "x.pkl", compression="foo"
    ),
    "storage options on a local file": lambda m, p: m.DataFrame({"a": [1]}).to_pickle(
        io.BytesIO(), storage_options={"k": 1}
    ),
    "a file that is not there": lambda m, p: m.read_pickle(p / "missing.pkl"),
    "reading with an unknown compression": lambda m, p: m.read_pickle(
        p / "x.pkl", compression="foo"
    ),
}


@needs_pandas
@pytest.mark.parametrize("name", list(MISTAKES))
def test_a_mistake_raises_what_pandas_raises(
    firepanda: ModuleType, name: str, tmp_path: Path
) -> None:
    """The same kind of error with the same words, apart from the path in them."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        MISTAKES[name](pd, tmp_path)
    with pytest.raises(Exception) as mine:
        MISTAKES[name](firepanda, tmp_path)
    assert type(mine.value) is type(theirs.value)
    assert str(mine.value) == str(theirs.value)


@needs_pandas
def test_the_signatures_are_pandas(firepanda: ModuleType) -> None:
    """`to_pickle` on a frame and a column, and the two functions, take pandas' parameters."""
    import inspect

    import pandas as pd

    def names(fn: Any) -> list[tuple[str, Any, Any]]:
        return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]

    for path in ("DataFrame.to_pickle", "Series.to_pickle", "to_pickle", "read_pickle"):
        mine, theirs = firepanda, pd
        for part in path.split("."):
            mine, theirs = getattr(mine, part), getattr(theirs, part)
        assert names(mine) == names(theirs), path
