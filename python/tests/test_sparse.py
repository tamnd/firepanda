"""`SparseDtype` and the `.sparse` accessors, compared with pandas live.

A sparse column is held as a column of tagged cells, so what these check is
what a user sees: the dtype and its name, the fill value each operation
answers with, the values, the accessor's numbers and pandas' errors. numpy
scalars are read as Python numbers first, since firepanda answers with those.
"""

from __future__ import annotations

import importlib.util
import math
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def number(value: Any) -> Any:
    # pandas writes an int fill value into a column of floats, so 0 and 0.0 read the same.
    value = value.item() if hasattr(value, "item") else value
    if isinstance(value, float):
        return "nan" if math.isnan(value) else int(value) if value.is_integer() else value
    return value


def outcome(call: Any) -> str:
    try:
        got = call()
    except Exception as error:
        kind = next(k for k in type(error).__mro__ if k.__module__ == "builtins")
        return f"{kind.__name__}: {error}"
    if hasattr(got, "dtype") and hasattr(got, "__len__"):
        return f"{got.dtype} {[number(value) for value in got.tolist()]}"
    if isinstance(got, tuple):
        return repr(tuple(number(value) for value in got))
    return repr(number(got))


def s(pd: ModuleType) -> Any:
    return pd.Series([0, 0, 1, 0, 2], dtype="Sparse[int64]")


def f(pd: ModuleType) -> Any:
    return pd.Series([1.0, None, 0.0, 3.0], dtype=pd.SparseDtype("float64"))


def b(pd: ModuleType) -> Any:
    return pd.Series([True, False, False], dtype=pd.SparseDtype(bool))


CALLS: list[Any] = [
    lambda pd: str(pd.SparseDtype()),
    lambda pd: str(pd.SparseDtype("int64")),
    lambda pd: str(pd.SparseDtype(int, 5)),
    lambda pd: str(pd.SparseDtype(bool, True)),
    lambda pd: str(pd.SparseDtype(str)),
    lambda pd: str(pd.SparseDtype("float64", 0.5)),
    lambda pd: pd.SparseDtype("int64", 1.5),
    lambda pd: pd.SparseDtype("int64", "a"),
    lambda pd: pd.SparseDtype("float64") == "Sparse[float64, nan]",
    lambda pd: pd.SparseDtype("int64") == pd.SparseDtype("int64", 0),
    lambda pd: pd.SparseDtype("int64") == pd.SparseDtype("float64", 0),
    lambda pd: hash(pd.SparseDtype("int64")) == hash(pd.SparseDtype("int64")),
    lambda pd: str(pd.SparseDtype.construct_from_string("Sparse[int64, 3]")),
    lambda pd: pd.SparseDtype.construct_from_string("int64"),
    lambda pd: pd.SparseDtype.construct_from_string("Sparse[int64, x]"),
    lambda pd: pd.SparseDtype.is_dtype("Sparse[float64]"),
    lambda pd: pd.SparseDtype.is_dtype("float64"),
    lambda pd: str(pd.SparseDtype("int64", 3).update_dtype("float64")),
    lambda pd: pd.SparseDtype("float64").kind,
    lambda pd: number(pd.SparseDtype("float64").fill_value),
    lambda pd: pd.SparseDtype("int64")._is_na_fill_value,
    lambda pd: s(pd),
    lambda pd: f(pd),
    lambda pd: b(pd),
    lambda pd: s(pd) + 1,
    lambda pd: s(pd) * 2,
    lambda pd: s(pd) / 2,
    lambda pd: s(pd) + 1.5,
    lambda pd: s(pd) + s(pd),
    lambda pd: s(pd) + pd.Series([5, 6, 7, 8, 9]),
    lambda pd: s(pd) == 0,
    lambda pd: s(pd) > 0,
    lambda pd: -s(pd),
    lambda pd: abs(s(pd)),
    lambda pd: ~b(pd),
    lambda pd: b(pd) & b(pd),
    lambda pd: f(pd) + 1,
    lambda pd: f(pd).isna(),
    lambda pd: f(pd).fillna(0.5),
    lambda pd: f(pd).dropna(),
    lambda pd: s(pd).shift(),
    lambda pd: s(pd).diff(),
    lambda pd: s(pd).where(s(pd) > 0),
    lambda pd: s(pd).where(s(pd) > 0, 0),
    lambda pd: s(pd).mask(s(pd) > 0),
    lambda pd: s(pd).clip(0, 1),
    lambda pd: s(pd).map(lambda x: x + 1),
    lambda pd: s(pd).apply(lambda x: x * 2),
    lambda pd: s(pd).astype("Sparse[float64]"),
    lambda pd: s(pd).astype("Sparse[float64, 0]"),
    lambda pd: s(pd).astype("float64"),
    lambda pd: s(pd).sort_values(),
    lambda pd: s(pd).reindex([0, 1, 9]),
    lambda pd: s(pd).nunique(),
    lambda pd: s(pd).sum(),
    lambda pd: s(pd).mean(),
    lambda pd: s(pd).max(),
    lambda pd: s(pd).count(),
    lambda pd: f(pd).sum(),
    lambda pd: s(pd).idxmax(),
    lambda pd: s(pd).std(),
    lambda pd: s(pd).median(),
    lambda pd: s(pd).prod(),
    lambda pd: s(pd).cumsum(),
    lambda pd: s(pd).describe(),
    lambda pd: s(pd).rolling(2).sum(),
    lambda pd: s(pd).value_counts().sort_index(),
    lambda pd: pd.concat([s(pd), pd.Series([5, 6])]),
    lambda pd: s(pd).sparse.fill_value,
    lambda pd: s(pd).sparse.npoints,
    lambda pd: s(pd).sparse.density,
    lambda pd: f(pd).sparse.density,
    lambda pd: s(pd).sparse.to_dense(),
    lambda pd: pd.Series([1, 2]).sparse,
    lambda pd: pd.DataFrame({"a": s(pd), "b": s(pd) + 1}).sparse.density,
    lambda pd: pd.DataFrame({"a": s(pd), "b": s(pd) + 1}).sparse.to_dense()["b"],
    lambda pd: pd.DataFrame({"a": [1]}).sparse,
    lambda pd: pd.DataFrame({"a": [0, 1]}, dtype="Sparse[int64]")["a"],
    lambda pd: pd.DataFrame({"a": [0, 1]}).astype("Sparse[float64]")["a"],
]


@needs_pandas
@pytest.mark.parametrize("call", CALLS)
def test_it_matches_pandas(firepanda: ModuleType, call: Any) -> None:
    import pandas

    assert outcome(lambda: call(firepanda)) == outcome(lambda: call(pandas))


def test_the_dtype_is_a_string(firepanda: ModuleType) -> None:
    dtype = firepanda.SparseDtype("int64")
    assert dtype == "Sparse[int64, 0]"
    assert dtype.subtype == "int64"
    assert firepanda.Series([0, 1], dtype=dtype).dtype == dtype


def test_the_dtype_pickles(firepanda: ModuleType) -> None:
    import pickle

    dtype = firepanda.SparseDtype("float64", 0.5)
    assert pickle.loads(pickle.dumps(dtype)) == dtype
