"""Group reductions over columns of a masked type against pandas, compared by repr and types."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")


def frame(lib: ModuleType) -> Any:
    return lib.DataFrame(
        {
            "g": ["a", "a", "b", "b"],
            "i": lib.Series([1, None, 3, 4], dtype="Int64"),
            "w": lib.Series([1, 2, None, 4], dtype="Int32"),
            "u": lib.Series([1, 2, 3, 4], dtype="UInt8"),
            "f": lib.Series([1.5, 2.5, None, 1.0], dtype="Float64"),
            "h": lib.Series([1.5, None, 2.0, 3.0], dtype="Float32"),
            "b": lib.Series([True, None, False, True], dtype="boolean"),
        }
    )


def texts(lib: ModuleType) -> Any:
    return lib.DataFrame({"g": ["a", "a", "b"], "s": lib.Series(["x", None, "z"], dtype="string")})


OPS = [
    "sum",
    "prod",
    "min",
    "max",
    "first",
    "last",
    "mean",
    "median",
    "count",
    "cumsum",
    "cumprod",
    "cummax",
    "cummin",
    "shift",
    "ffill",
    "bfill",
    "head",
    "tail",
    "nunique",
    "cumcount",
    "ngroup",
    "any",
    "all",
    "rank",
    "quantile",
    "idxmax",
]

CASES: dict[str, Callable[[ModuleType], Any]] = {
    **{f"frame-{op}": lambda lib, op=op: getattr(frame(lib).groupby("g"), op)() for op in OPS},
    **{f"column-{op}": lambda lib, op=op: getattr(frame(lib).groupby("g").i, op)() for op in OPS},
    "flags-sum": lambda lib: frame(lib).groupby("g").b.sum(),
    "flags-mean": lambda lib: frame(lib).groupby("g").b.mean(),
    "narrow-sum": lambda lib: frame(lib).groupby("g").w.sum(),
    "narrow-float-mean": lambda lib: frame(lib).groupby("g").h.mean(),
    "size": lambda lib: frame(lib).groupby("g").i.size(),
    "frame-size": lambda lib: frame(lib).groupby("g").size(),
    "std": lambda lib: frame(lib).groupby("g").u.std(),
    "agg-dict": lambda lib: frame(lib).groupby("g").agg({"i": "sum", "h": ["mean", "min"]}),
    "agg-list": lambda lib: frame(lib).groupby("g").agg(["sum", "mean"]),
    "agg-name": lambda lib: frame(lib).groupby("g").agg("max"),
    "agg-function": lambda lib: frame(lib).groupby("g").w.agg(lambda s: s.sum()),
    "column-agg-list": lambda lib: frame(lib).groupby("g").i.agg(["sum", "mean"]),
    "apply": lambda lib: frame(lib).groupby("g").w.apply(lambda s: s.sum()),
    "transform-function": lambda lib: frame(lib).groupby("g").w.transform(lambda s: s * 2),
    "transform-name": lambda lib: frame(lib).groupby("g").transform("mean"),
    "as-column": lambda lib: frame(lib).groupby("g", as_index=False).sum(),
    "series-groupby": lambda lib: frame(lib).i.groupby(frame(lib).g).sum(),
    "series-groupby-mean": lambda lib: frame(lib).i.groupby(frame(lib).g).mean(),
    "filter": lambda lib: frame(lib).groupby("g").filter(lambda x: len(x) > 1),
    "get-group": lambda lib: frame(lib).groupby("g").get_group("a"),
    "groups": lambda lib: [rows for _, rows in frame(lib).groupby("g")],
    "two-keys": lambda lib: frame(lib).assign(k=[1, 1, 2, 2]).groupby(["g", "k"]).i.sum(),
    "text-min": lambda lib: texts(lib).groupby("g").s.min(),
    "text-count": lambda lib: texts(lib).groupby("g").s.count(),
}


def mistake(error: Exception) -> str:
    kind = next(kind.__name__ for kind in type(error).__mro__ if kind.__module__ == "builtins")
    return f"{kind}: {error}"


def outcome(build: Callable[[], Any]) -> str:
    try:
        found = build()
    except Exception as error:
        return mistake(error)
    if isinstance(found, list):
        return "\n".join(f"{dict(each.dtypes.astype(str))}\n{each!r}" for each in found)
    types = dict(found.dtypes.astype(str)) if found.ndim == 2 else str(found.dtype)
    return f"{types}\n{found!r}"


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_answers_in_the_type_pandas_gives(case: Callable[[ModuleType], Any]) -> None:
    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        ("Int8", [1, None]),
        ("Int32", [1, None]),
        ("UInt16", [1, None]),
        ("Float32", [1.5, None]),
        ("boolean", [True, None]),
        ("string", ["a", None]),
    ],
)
def test_a_masked_column_goes_to_arrow_at_its_width(dtype: str, values: list[Any]) -> None:
    pa = pytest.importorskip("pyarrow")
    ours = pa.table(fp.DataFrame({"x": fp.Series(values, dtype=dtype)}))
    theirs = pa.table(pd.DataFrame({"x": pd.Series(values, dtype=dtype)}))
    assert ours.schema.field("x").type == theirs.schema.field("x").type
    assert ours.column("x").to_pylist() == theirs.column("x").to_pylist()
