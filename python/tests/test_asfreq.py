"""`asfreq` on a frame and a column, compared with pandas.

Each case builds the same rows in both libraries, labelled by instants, and asks
both for them at a fixed frequency. The answer is compared as values, labels,
label name and label dtype.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")


def hours(lib: ModuleType) -> Any:
    """A column at midnight, three and five, with a gap at three."""
    index = lib.DatetimeIndex(
        ["2024-01-01 00:00", "2024-01-01 03:00", "2024-01-01 05:00"], name="t"
    )
    return lib.Series([1.0, None, 3.0], index=index)


def outcome(call: Callable[[ModuleType], Any], lib: ModuleType) -> Any:
    got = call(lib)
    if hasattr(got, "columns"):
        values: Any = {name: got[name].tolist() for name in got.columns}
    else:
        values = got.tolist()
    labels = [str(one) for one in got.index]
    return repr(values), labels, got.index.name, str(got.index.dtype)


CASES: list[Callable[[ModuleType], Any]] = [
    lambda lib: hours(lib).asfreq("h"),
    lambda lib: hours(lib).asfreq("2h", method="ffill"),
    lambda lib: hours(lib).asfreq("h", method="bfill"),
    lambda lib: hours(lib).asfreq("h", fill_value=9.0),
    lambda lib: hours(lib).asfreq("D", normalize=True),
    lambda lib: hours(lib).asfreq("h", how="start"),
    lambda lib: hours(lib).to_frame("v").asfreq("30min"),
    lambda lib: hours(lib).to_frame("v").assign(w=[1, 2, 3]).asfreq("h", method="pad"),
    lambda lib: hours(lib)[:0].asfreq("h"),
    lambda lib: hours(lib).tz_localize("UTC").asfreq("2h"),
    lambda lib: lib.Series(
        [2.0, 1.0], index=lib.DatetimeIndex(["2024-01-02", "2024-01-01"])
    ).asfreq("D"),
]


@pytest.mark.parametrize("call", CASES)
def test_the_rows_are_pandas_rows(call: Callable[[ModuleType], Any]) -> None:
    assert outcome(call, fp) == outcome(call, pd)


def test_labels_that_are_not_instants_are_refused() -> None:
    with pytest.raises(TypeError):
        fp.Series([1], index=[0])[:0].asfreq("h")


def test_the_signatures_are_pandas_signatures() -> None:
    for owner in ("DataFrame", "Series"):
        ours = inspect.signature(getattr(fp, owner).asfreq)
        theirs = inspect.signature(getattr(pd, owner).asfreq)
        assert list(ours.parameters) == list(theirs.parameters)
