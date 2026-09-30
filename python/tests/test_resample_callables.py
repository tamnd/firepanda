"""Functions run on every bin of a resample, `ohlc` over a frame, and `level=`.

pandas runs a function handed to `aggregate` or `apply` on every bin, the
empty ones too, and on each column of a frame, falling back to the whole of
each bin when a column refuses it. `transform` runs it on every bin with rows
and puts the answers back on the rows. Over a frame `ohlc` gives each column
four under its own name, and `level=` bins by one level of a MultiIndex. Each
test here runs the same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

STAMPS = ["2024-01-01", "2024-01-02", "2024-01-07", "2024-01-08"]


def series(lib: Any) -> Any:
    return lib.Series([1, 2, 3, 4], index=lib.to_datetime(STAMPS), name="v")


def frame(lib: Any) -> Any:
    return lib.DataFrame(
        {"a": [1, 2, 3, 4], "b": [1.5, 2.5, 3.5, 4.5]}, index=lib.to_datetime(STAMPS)
    )


def summed(x: Any) -> Any:
    return x.sum() if len(x) else -1


def keyed(lib: Any) -> Any:
    out = frame(lib).assign(k=["x", "y", "x", "y"])
    return out.set_index("k", append=True)


BUILDS = {
    "series agg": lambda lib: series(lib).resample("2D").agg(summed),
    "series apply": lambda lib: series(lib).resample("2D").apply(summed),
    "series agg arguments": lambda lib: series(lib).resample("2D").agg(lambda x, n: x.sum() * n, 3),
    "frame agg": lambda lib: frame(lib).resample("2D").agg(summed),
    "frame apply": lambda lib: frame(lib).resample("2D").apply(summed),
    "frame apply whole": lambda lib: frame(lib).resample("2D").apply(lambda g: g["a"].sum() * 10),
    "frame apply rows": lambda lib: frame(lib).resample("2D").apply(lambda g: g.sum()),
    "mapping": lambda lib: frame(lib).resample("2D").agg({"a": summed, "b": "max"}),
    "mapping list": lambda lib: frame(lib).resample("2D").agg({"a": summed, "b": ["min", "max"]}),
    "list": lambda lib: frame(lib).resample("2D").agg([summed, "max"]),
    "series list": lambda lib: series(lib).resample("2D").agg(["min", summed]),
    "timestamps": lambda lib: (
        series(lib).resample("2D").agg(lambda x: x.index[0] if len(x) else None)
    ),
    "pieces": lambda lib: series(lib).resample("2D").apply(lambda x: x * 2),
    "transform frame": lambda lib: frame(lib).resample("2D").transform(lambda x: x - x.mean()),
    "transform series": lambda lib: series(lib).resample("2D").transform(lambda x: x.cumsum()),
    "transform arguments": lambda lib: series(lib).resample("2D").transform(lambda x, n: x * n, 3),
    "ohlc frame": lambda lib: frame(lib).resample("2D").ohlc(),
    "ohlc columns": lambda lib: frame(lib).resample("3D")[["a", "b"]].ohlc(),
    "level": lambda lib: keyed(lib).resample("2D", level=0).sum(),
    "level function": lambda lib: keyed(lib).resample("3D", level=0)["a"].agg(lambda x: x.max()),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_resample_callables_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_refusals_are_pandas_refusals(firepanda: Any) -> None:
    with pytest.raises(KeyError, match="x"):
        series(firepanda).resample("2D", on="x").sum()
    with pytest.raises(ValueError, match="The level zz is not valid"):
        frame(firepanda).resample("2D", level="zz").sum()
    with pytest.raises(ValueError, match="Upsampling from level= or on= selection"):
        keyed(firepanda).resample("12h", level=0).asfreq()
