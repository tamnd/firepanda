"""`to_numpy` and `values` on a frame, a column and an index, checked against pandas.

pandas hands back a numpy array in the type numpy has for the values: numbers
keep theirs, whole numbers with a gap become floats, instants with no zone stay
instants, and text, categories and instants with a zone are objects. A frame's
columns share one type, the wider number or objects.
"""

from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def plain(values: Any) -> Any:
    """The values with every spelling of missing as one word, nested lists too."""
    if isinstance(values, list):
        return [plain(value) for value in values]
    if values is None or values != values:
        return "missing"
    return str(values) if type(values).__name__ in ("Timestamp", "datetime64") else values


def agrees(got: Any, want: Any) -> None:
    """The same numpy type, shape and values."""
    import numpy as np

    assert isinstance(got, np.ndarray)
    assert got.dtype == want.dtype
    assert got.shape == want.shape
    assert plain(got.tolist()) == plain(want.tolist())


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.Series([1, 2, 3]).to_numpy(),
    lambda m: m.Series([1.5, None]).to_numpy(),
    lambda m: m.Series([1, None]).to_numpy(),
    lambda m: m.Series(["a", None, "c"]).to_numpy(),
    lambda m: m.Series([True, False]).to_numpy(),
    lambda m: m.Series([1, 2], dtype="int32").to_numpy(),
    lambda m: m.Series([1.5, None]).to_numpy(na_value=0.0),
    lambda m: m.Series(["a", None]).to_numpy(na_value="z"),
    lambda m: m.Series([1, 2]).to_numpy(dtype="float32"),
    lambda m: m.Series([1, 2]).values,
    lambda m: m.Series([1.5, None]).values,
    lambda m: m.Series([], dtype="int64").to_numpy(),
    lambda m: m.Series(["b", "a", "b"], dtype="category").to_numpy(),
    lambda m: m.to_datetime(m.Series(["2024-01-01"])).dt.tz_localize("UTC").to_numpy(),
    lambda m: m.DataFrame({"a": [1, 2], "b": [1.5, 2.5]}).to_numpy(),
    lambda m: m.DataFrame({"a": [1, 2], "b": [3, 4]}).to_numpy(),
    lambda m: m.DataFrame({"a": [1, 2], "b": ["x", None]}).to_numpy(),
    lambda m: m.DataFrame({"a": [1.0, None]}).to_numpy(na_value=-1.0),
    lambda m: m.DataFrame({"a": [1, 2], "b": [3, 4]}).to_numpy(dtype="float64"),
    lambda m: m.DataFrame({"a": [1, 2], "b": [1.5, 2.5]}).values,
    lambda m: m.Index([3, 1, 2]).to_numpy(),
    lambda m: m.Index(["a", "b"]).to_numpy(),
    lambda m: m.Index([1.5, 2.5]).to_numpy(dtype="int64"),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Each type, gaps, a fill, a cast, a frame's shared type and an index."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


def test_instants_with_no_zone_stay_instants(firepanda: ModuleType) -> None:
    """numpy holds them as counts of the unit, with NaT in a gap."""
    import pandas as pd

    def build(m: ModuleType) -> Any:
        return m.to_datetime(m.Series(["2024-01-01", None])).to_numpy()

    got, want = build(firepanda), build(pd)
    assert got.dtype == want.dtype
    assert [str(value) for value in got] == [str(value) for value in want]


def test_text_values_hold_their_type(firepanda: ModuleType) -> None:
    """pandas answers an extension array for text, and firepanda its own array."""
    answer = firepanda.Series(["a", None]).values
    assert type(answer).__name__ == "FirepandaArray"
    assert answer.tolist() == ["a", None]


@pytest.mark.parametrize(
    ("owner", "name"),
    [("Series", "to_numpy"), ("DataFrame", "to_numpy"), ("Index", "to_numpy")],
)
def test_the_signature_is_pandas_signature(firepanda: ModuleType, owner: str, name: str) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    ours = inspect.signature(getattr(getattr(firepanda, owner), name)).parameters
    yours = inspect.signature(getattr(getattr(pd, owner), name)).parameters
    assert [(p.name, p.kind, repr(p.default)) for p in ours.values()] == [
        (p.name, p.kind, repr(p.default)) for p in yours.values()
    ]
