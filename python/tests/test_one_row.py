"""One row of a frame read as a series, and names that are numbers, checked against pandas.

`df.iloc[0]` and `df.loc[label]` answer a series labelled by the column names
and named by the row's label, holding the one type every column fits. pandas
answers a mix of a number and a flag or text with an object column, which
firepanda does not have, so those are refused. `quantile` names its answer by
the fraction and `rename` takes a number, and both hand the number back.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

NAN = float("nan")
TABLES: dict[str, dict[str, Any]] = {
    "whole": {"a": [3, 1, 2], "b": [4, 5, 6]},
    "mixed numbers": {"a": [3, 1, 2], "b": [1.5, NAN, 2.5]},
    "flags": {"a": [True, False, True], "b": [False, False, True]},
    "text": {"a": ["x", "y", "z"], "b": ["p", None, "r"]},
}


def plain(values: list[Any]) -> list[Any]:
    """The values with every spelling of missing as one word."""
    return ["missing" if value is None or value != value else value for value in values]


def agrees(got: Any, want: Any) -> None:
    """The same values, labels, name and type, text being spelled either way."""
    assert plain(got.tolist()) == plain(want.tolist())
    assert list(got.index) == list(want.index)
    assert got.name == want.name
    assert type(got.name) is type(getattr(want.name, "item", lambda: want.name)())
    assert str(got.dtype).replace("string", "str") == str(want.dtype)


def frames(m: ModuleType, table: dict[str, Any]) -> Any:
    """The frame, sorted so a label and a position are different rows."""
    return m.DataFrame(table).sort_values("a")


READS: list[Callable[[Any], Any]] = [
    lambda df: df.iloc[0],
    lambda df: df.iloc[-1],
    lambda df: df.loc[1],
    lambda df: df.iloc[1, [1, 0]],
    lambda df: df.loc[2, ["b"]],
]


@pytest.mark.parametrize("read", READS)
@pytest.mark.parametrize("table", list(TABLES))
def test_a_row_is_pandas_row(firepanda: ModuleType, table: str, read: Callable[[Any], Any]) -> None:
    """Whole numbers, numbers with a gap, flags and text, by position and by label."""
    import pandas as pd

    agrees(read(frames(firepanda, TABLES[table])), read(frames(pd, TABLES[table])))


def test_widths_meet_the_way_numpy_does(firepanda: ModuleType) -> None:
    """int32 next to int8 is int32 and float32 next to int64 is float64."""
    import pyarrow as pa

    table = pa.table(
        {
            "a": pa.array([1], pa.int32()),
            "b": pa.array([2], pa.int8()),
            "c": pa.array([0.5], pa.float32()),
            "d": pa.array([7], pa.int64()),
        }
    )
    mine, theirs = firepanda.from_arrow(table), table.to_pandas()
    for columns in (["a", "b"], ["c", "d"], ["a", "c"]):
        agrees(mine[columns].iloc[0], theirs[columns].iloc[0])


@pytest.mark.parametrize(
    "table", [{"a": [1], "b": [True]}, {"a": [1], "b": ["x"]}, {"a": [1.5], "b": ["x"]}]
)
def test_a_mix_pandas_holds_as_object_is_refused(
    firepanda: ModuleType, table: dict[str, Any]
) -> None:
    """A number next to a flag or to text is an object column in pandas."""
    import pandas as pd

    assert pd.DataFrame(table).iloc[0].dtype == object
    with pytest.raises(NotImplementedError, match="object column"):
        firepanda.DataFrame(table).iloc[0]


@pytest.mark.parametrize("position", [3, -4, 9999])
def test_a_position_past_the_end_is_pandas_mistake(firepanda: ModuleType, position: int) -> None:
    """IndexError with pandas' words, from either end."""
    import pandas as pd

    table = TABLES["whole"]
    with pytest.raises(IndexError) as theirs:
        pd.DataFrame(table).iloc[position]
    with pytest.raises(IndexError) as mine:
        firepanda.DataFrame(table).iloc[position]
    assert str(mine.value) == str(theirs.value)


NAMES: list[Callable[[Any], Any]] = [
    lambda m: m.DataFrame({"a": [1, 2, 4], "b": [1.0, 3.0, 9.0]}).quantile(0.5),
    lambda m: m.DataFrame({"a": [1, 2, 4], "b": [1.0, 3.0, 9.0]}).quantile(1),
    lambda m: m.DataFrame({"a": [1, 2, 4]}).quantile(0.25, interpolation="lower"),
    lambda m: m.Series([1, 2]).rename(5),
    lambda m: m.Series([1, 2]).rename(2.5),
    lambda m: m.Series([1, 2]).rename(5).rename("x"),
    lambda m: m.Series([1, 2]).rename(5).rename(None),
]


@pytest.mark.parametrize("build", NAMES)
def test_a_name_that_is_a_number_stays_one(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """A quantile's fraction, a float even for `quantile(1)`, and `rename` with a number."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


def test_a_rename_in_place_keeps_the_number(firepanda: ModuleType) -> None:
    """The column put back answers the number, and a later text name replaces it."""
    column = firepanda.Series([1, 2])
    column.rename(7, inplace=True)
    assert column.name == 7
    column.rename("q", inplace=True)
    assert column.name == "q"
