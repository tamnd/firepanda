"""`read_csv` with pandas' signature, checked against a running pandas.

The reader underneath takes a path and nothing else, and pandas declares forty
nine parameters. What these tests hold is the split `read_csv` in `_pandas.py`
makes between them: the two that are applied after the read give pandas'
answer, the ones that do not change the answer are accepted, and everything
else is refused by name rather than ignored.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import ModuleType

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

TEXT = "a,b,c\n1,x,2.5\n2,y,3.5\n3,z,4.5\n"
"""Three columns of three types, so a column that went missing is noticed."""


@pytest.fixture
def csv(tmp_path: Path) -> Path:
    """The file, written once per test."""
    path = tmp_path / "three.csv"
    path.write_text(TEXT)
    return path


@needs_pandas
def test_the_signature_is_pandas_signature(firepanda: ModuleType) -> None:
    """Every name, in order, of the same kind, which is what the board compares."""
    import pandas as pd

    def shape(function: object) -> list[tuple[str, object]]:
        parameters = inspect.signature(function).parameters.values()  # type: ignore[arg-type]
        return [(one.name, one.kind) for one in parameters]

    assert shape(firepanda.read_csv) == shape(pd.read_csv)


def test_a_path_object_reads_like_a_string(firepanda: ModuleType, csv: Path) -> None:
    """pandas takes anything `os.fspath` takes, and so does this."""
    assert list(firepanda.read_csv(csv).columns) == ["a", "b", "c"]


@needs_pandas
@pytest.mark.parametrize("usecols", [["c", "a"], [2, 0], ["b"], "b"])
def test_usecols_keeps_the_file_order(firepanda: ModuleType, csv: Path, usecols: object) -> None:
    """pandas reads the file's order rather than the list's, and so does this."""
    import pandas as pd

    if isinstance(usecols, str):
        # pandas refuses a bare string, and saying so is the answer it gives.
        with pytest.raises(ValueError, match="usecols"):
            pd.read_csv(csv, usecols=usecols)
        return
    mine = firepanda.read_csv(csv, usecols=usecols)
    theirs = pd.read_csv(csv, usecols=usecols)
    assert list(mine.columns) == list(theirs.columns)
    for name in theirs.columns:
        assert mine[name].tolist() == theirs[name].tolist()


@needs_pandas
def test_usecols_naming_a_missing_column_is_pandas_error(firepanda: ModuleType, csv: Path) -> None:
    import pandas as pd

    for module in (firepanda, pd):
        with pytest.raises(ValueError, match="columns expected but not found"):
            module.read_csv(csv, usecols=["a", "zz"])


@needs_pandas
@pytest.mark.parametrize("index_col", [0, "b"])
def test_index_col_moves_a_column_into_the_labels(
    firepanda: ModuleType, csv: Path, index_col: object
) -> None:
    import pandas as pd

    mine = firepanda.read_csv(csv, index_col=index_col)
    theirs = pd.read_csv(csv, index_col=index_col)
    assert list(mine.columns) == list(theirs.columns)
    assert list(mine.index) == list(theirs.index)


@pytest.mark.parametrize(
    "arguments",
    [
        {"engine": "pyarrow"},
        {"engine": "python"},
        {"low_memory": False},
        {"memory_map": True},
        {"cache_dates": False},
        {"sep": ","},
        {"delimiter": ","},
        {"encoding": "UTF-8"},
        {"header": 0},
        {"index_col": False},
    ],
)
def test_an_argument_that_does_not_change_the_answer_is_accepted(
    firepanda: ModuleType, csv: Path, arguments: dict[str, object]
) -> None:
    assert list(firepanda.read_csv(csv, **arguments).columns) == ["a", "b", "c"]


def gapped(values: list[object]) -> list[object]:
    """The values with every kind of missing as None."""
    return [None if value is None or value != value else value for value in values]


@needs_pandas
@pytest.mark.parametrize(
    "arguments",
    [
        {"sep": ";"},
        {"nrows": 2},
        {"dtype": {"a": "int32"}},
        {"skiprows": 1},
        {"na_values": ["x"]},
        {"names": ["p", "q", "r"]},
        {"header": None},
        {"skipfooter": 1, "engine": "python"},
        {"encoding": "latin-1"},
        {"usecols": lambda name: name != "b"},
    ],
)
def test_an_argument_that_changes_the_answer_changes_it_as_in_pandas(
    firepanda: ModuleType, csv: Path, arguments: dict[str, object]
) -> None:
    import pandas as pd

    ours = firepanda.read_csv(csv, **arguments)
    theirs = pd.read_csv(csv, **arguments)
    assert list(ours.columns) == list(theirs.columns)
    for mine, other in zip(ours.columns, theirs.columns, strict=True):
        assert gapped(ours[mine].tolist()) == gapped(theirs[other].tolist())


@needs_pandas
@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"sep": ",", "delimiter": ","}, "Specified a sep and a delimiter"),
        ({"engine": "rust"}, "Unknown engine"),
        ({"dtype_backend": "numpy"}, "dtype_backend numpy is invalid"),
    ],
)
def test_a_mistake_is_pandas_value_error(
    firepanda: ModuleType, csv: Path, arguments: dict[str, object], expected: str
) -> None:
    import pandas as pd

    for module in (firepanda, pd):
        with pytest.raises(ValueError, match=expected):
            module.read_csv(csv, **arguments)


def test_a_buffer_is_read_as_the_file_would_be(firepanda: ModuleType, csv: Path) -> None:
    import io

    from_buffer = firepanda.read_csv(io.StringIO(TEXT))
    from_path = firepanda.read_csv(csv)
    assert list(from_buffer.columns) == list(from_path.columns)
    for name in from_path.columns:
        assert from_buffer[name].tolist() == from_path[name].tolist()
