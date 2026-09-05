"""The column a frame hands back, and the expression that asks for it.

`df["a"]` is the most written line in pandas and until there was a bound series
type there was no answer to it at all. So these tests are less about the methods
on `Series` than about the shape of the thing: that a column comes out under its
own name, with its own type, with its missing values still missing, and that the
key you cannot use says so rather than doing something else.

The values are read back with `tolist` rather than compared against a repr,
because a repr test passes for a column that is right and for one that is
formatted right.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

CSV = "region,qty,price,ok\nnorth,10,1.5,true\nsouth,,2.5,false\neast,25,,true\n"


def _frame(firepanda: ModuleType, tmp_path: Path) -> object:
    """Reads the four column fixture every test below works on."""
    path = tmp_path / "t.csv"
    path.write_text(CSV)
    return firepanda.read_csv(str(path))


def test_a_column_comes_out_as_a_series(firepanda: ModuleType, tmp_path: Path) -> None:
    """The whole point of the type, stated once."""
    series = _frame(firepanda, tmp_path)["qty"]
    assert isinstance(series, firepanda.Series)
    assert series.name == "qty"
    assert len(series) == 3


def test_a_series_reports_its_shape_the_pandas_way(firepanda: ModuleType, tmp_path: Path) -> None:
    """`shape` is a one long tuple on a series and a two long one on a frame.

    Worth a test because the obvious mistake is to give a series the frame's
    shape, which is a tuple of the right kind and the wrong length, and every
    caller that unpacks it breaks somewhere else.
    """
    series = _frame(firepanda, tmp_path)["qty"]
    assert series.shape == (3,)
    assert series.size == 3


def test_the_values_come_back_with_the_holes_still_in_them(
    firepanda: ModuleType, tmp_path: Path
) -> None:
    """A missing integer stays missing rather than becoming a float.

    This is a real difference from pandas and it is the one worth being loud
    about. pandas would have widened this column to float64 and put a NaN in the
    gap, because a numpy int64 array has nowhere to record absence. A firepanda
    column is Arrow and has a validity bitmap, so the value is missing rather
    than approximated, and `None` is what says so. pyarrow and Polars both answer
    the same way.
    """
    series = _frame(firepanda, tmp_path)["qty"]
    assert series.tolist() == [10, None, 25]
    assert series.dtype == "int64"


def test_every_kind_of_column_reads_back(firepanda: ModuleType, tmp_path: Path) -> None:
    """Text, floats and booleans, because each takes a different path out.

    Text is not dispatched over the numeric dtypes, since a string column is
    physically uint8 and would otherwise come back as a list of bytes, and a
    boolean is a bit rather than a byte. Only the integer case would pass if the
    other two were forgotten.
    """
    frame = _frame(firepanda, tmp_path)
    assert frame["region"].tolist() == ["north", "south", "east"]
    assert frame["price"].tolist() == [1.5, 2.5, None]
    assert frame["ok"].tolist() == [True, False, True]
    assert frame["region"].dtype == "string"
    assert frame["price"].dtype == "float64"
    assert frame["ok"].dtype == "bool"


def test_a_series_counts_what_is_there_and_what_is_not(
    firepanda: ModuleType, tmp_path: Path
) -> None:
    """`count` is the non missing count in pandas, which is the trap in the name."""
    series = _frame(firepanda, tmp_path)["qty"]
    assert series.count() == 2
    assert series.hasnans
    assert not _frame(firepanda, tmp_path)["region"].hasnans


def test_head_and_tail_stay_series(firepanda: ModuleType, tmp_path: Path) -> None:
    """A slice of a series is a series, not a frame and not a list."""
    series = _frame(firepanda, tmp_path)["qty"]
    assert isinstance(series.head(2), firepanda.Series)
    assert series.head(2).tolist() == [10, None]
    assert series.tail(1).tolist() == [25]
    assert series.head(2).name == "qty"


def test_a_list_of_names_gives_a_frame_back(firepanda: ModuleType, tmp_path: Path) -> None:
    """The other half of `df[...]`, which is a different operation under one name."""
    frame = _frame(firepanda, tmp_path)[["region", "qty"]]
    assert isinstance(frame, firepanda.DataFrame)
    assert frame.columns == ["region", "qty"]
    assert frame.shape == (3, 2)


def test_a_name_that_is_not_there_raises_key_error(firepanda: ModuleType, tmp_path: Path) -> None:
    """The class matters, not the message. `except KeyError` is what callers wrote."""
    with pytest.raises(KeyError):
        _frame(firepanda, tmp_path)["nope"]
    with pytest.raises(KeyError):
        _frame(firepanda, tmp_path)[["region", "nope"]]


def test_a_key_that_is_not_read_yet_says_so(firepanda: ModuleType, tmp_path: Path) -> None:
    """pandas reads masks, slices and callables here and firepanda does not.

    Refused rather than approximated. A key that quietly means something else is
    worse than one that does not work, because the second is found immediately.
    """
    with pytest.raises(TypeError) as caught:
        _frame(firepanda, tmp_path)[3]
    assert "column name" in str(caught.value)


def test_an_empty_constructor_gives_an_empty_one(firepanda: ModuleType) -> None:
    """`firepanda.Series()` and `firepanda.DataFrame()` are what pandas says they are.

    This used to be a refusal, because the generated wrapper took the extension
    object as its only argument and there was no conversion behind it. Both
    halves of that are gone now.
    """
    assert len(firepanda.Series()) == 0
    assert firepanda.DataFrame().shape == (0, 0)


def test_a_series_renders(firepanda: ModuleType, tmp_path: Path) -> None:
    """`repr` and `str` agree, which is what pandas does."""
    series = _frame(firepanda, tmp_path)["qty"]
    assert repr(series) == str(series)
    assert "qty" in repr(series)
