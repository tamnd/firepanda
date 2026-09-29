"""`to_xarray` on a frame and a series, compared with pandas.

The conversion runs everywhere xarray is installed, and each test builds the
same object in both libraries and asks xarray whether the two answers are
identical, which compares the dimensions, the coordinates, the values, the
types and the names.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)
needs_xarray = pytest.mark.skipif(
    importlib.util.find_spec("xarray") is None, reason="xarray is not installed"
)


@needs_pandas
@pytest.mark.parametrize("owner", ["DataFrame", "Series"])
def test_the_signature_is_pandas(owner: str) -> None:
    import pandas

    ours = inspect.signature(getattr(fp, owner).to_xarray)
    theirs = inspect.signature(getattr(pandas, owner).to_xarray)
    assert list(ours.parameters) == list(theirs.parameters)


def test_no_xarray_is_pandas_sentence(monkeypatch: Any) -> None:
    monkeypatch.setitem(sys.modules, "xarray", None)
    with pytest.raises(ImportError) as raised:
        fp.DataFrame({"a": [1]}).to_xarray()
    assert str(raised.value) == (
        "`Import xarray` failed.  Use pip or conda to install the xarray package."
    )


CONVERTED: list[Callable[[ModuleType], Any]] = [
    lambda m: m.DataFrame({"a": [1, 2, 3], "b": [1.5, None, 3.0]}, index=["p", "q", "r"]),
    lambda m: m.DataFrame({"s": ["x", "y", None], "f": [True, False, True]}),
    lambda m: m.DataFrame({"k": [5, 6], "a": [1, 2]}).set_index("k"),
    lambda m: m.DataFrame({"k": ["x", "x", "y"], "j": [1, 2, 1], "v": [1.0, 2.0, 3.0]}).set_index(
        ["k", "j"]
    ),
    lambda m: m.DataFrame({"c": m.Series(["a", "b", "a"]).astype("category")}),
    lambda m: m.DataFrame(
        {
            "i": m.Series([1, None, 3]).astype("Int64"),
            # No gap here, because xarray finds a boolean column with a gap
            # different from itself, pandas' own included.
            "b": m.Series([True, False, False], dtype="boolean"),
        }
    ),
    lambda m: m.DataFrame(
        {
            "t": m.to_datetime(["2020-01-01", None]),
            "z": m.Series(m.to_datetime(["2020-01-01", "2020-01-02"])).dt.tz_localize("UTC"),
            "d": m.to_timedelta([1, 2], unit="s"),
        }
    ),
    lambda m: m.Series([1.0, 2.0], index=["u", "v"], name="n"),
    lambda m: m.Series([1, 2]),
    lambda m: m.DataFrame({"k": ["x", "y"], "j": [1, 2], "v": [1, 2]}).set_index(["k", "j"])["v"],
]


@needs_pandas
@needs_xarray
@pytest.mark.parametrize("build", CONVERTED)
def test_the_answer_is_pandas_answer(build: Callable[[ModuleType], Any]) -> None:
    import pandas

    ours = build(fp).to_xarray()
    theirs = build(pandas).to_xarray()
    assert type(ours) is type(theirs)
    assert ours.identical(theirs)


@needs_pandas
@needs_xarray
def test_a_repeated_label_is_xarray_refusal() -> None:
    import pandas

    def build(m: ModuleType) -> Any:
        frame = m.DataFrame({"k": ["x", "x"], "j": [1, 1], "v": [1, 2]})
        return frame.set_index(["k", "j"]).to_xarray()

    with pytest.raises(ValueError) as theirs:
        build(pandas)
    with pytest.raises(ValueError) as ours:
        build(fp)
    assert str(ours.value) == str(theirs.value)
