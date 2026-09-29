"""Period columns against pandas, compared by repr."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")


def four(lib: ModuleType) -> Any:
    return [
        lib.Period("2026-01", "M"),
        lib.Period("2026-03", "M"),
        None,
        lib.Period("2025-12", "M"),
    ]


def five(lib: ModuleType) -> Any:
    return [*four(lib), lib.Period("2026-01", "M")]


def series(lib: ModuleType) -> Any:
    return lib.Series(five(lib), name="p")


def frame(lib: ModuleType) -> Any:
    return lib.DataFrame({"p": five(lib), "v": [1, 2, 3, 4, 5]})


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "frame-astype": lambda lib: lib.DataFrame({"a": ["2026-01", "2026-02"]}).astype("period[M]"),
    "frame-astype-dict": lambda lib: (
        lib.DataFrame({"a": ["2026-01", "2026-02"]}).astype({"a": "period[D]"}).a.tolist()
    ),
    "is-period": lambda lib: lib.api.types.is_period_dtype(lib.Series(four(lib))),
    "make": lambda lib: lib.Series(four(lib)),
    "dtype": lambda lib: lib.Series(four(lib)).dtype,
    "list": lambda lib: lib.Series(four(lib)).tolist(),
    "isna": lambda lib: lib.Series(four(lib)).isna().tolist(),
    "sort": lambda lib: lib.Series(four(lib)).sort_values(),
    "mixed": lambda lib: lib.Series([lib.Period("2026-01", "M"), lib.Period("2026-01-01", "D")]),
    "nat": lambda lib: lib.Series([lib.Period("2026-01", "M"), lib.NaT]),
    "frame": lambda lib: lib.DataFrame({"p": four(lib), "v": [1, 2, 3, 4]}),
    "unique": lambda lib: lib.Series(four(lib)).unique(),
    "nunique": lambda lib: lib.Series(four(lib)).nunique(),
    "eq": lambda lib: (lib.Series(four(lib)) == lib.Period("2026-01", "M")).tolist(),
    "min": lambda lib: lib.Series(four(lib)).min(),
    "max": lambda lib: lib.Series(four(lib)).max(),
    "iloc": lambda lib: lib.Series(four(lib)).iloc[1],
    "getgap": lambda lib: lib.Series(four(lib)).iloc[2],
    "astype-str": lambda lib: lib.Series(four(lib)).astype(str),
    "dtype-arg": lambda lib: lib.Series(["2026-01", "2026-02"], dtype="period[M]"),
    "dtype-arg2": lambda lib: lib.Series(["2026-01", None], dtype=lib.PeriodDtype("M")),
    "copy": lambda lib: lib.Series(four(lib)).copy(),
    "shift": lambda lib: lib.Series(four(lib)).shift(1),
    "fillna": lambda lib: lib.Series(four(lib)).fillna(lib.Period("2000-01", "M")),
    "add": lambda lib: lib.Series(four(lib)) + 1,
    "sub": lambda lib: lib.Series(four(lib)) - lib.Period("2025-01", "M"),
    "concat": lambda lib: lib.concat([lib.Series(four(lib)), lib.Series(four(lib))]),
    "group": lambda lib: lib.DataFrame({"p": four(lib), "v": [1, 2, 3, 4]}).groupby("p").sum(),
    "csv": lambda lib: lib.DataFrame({"p": four(lib)}).to_csv(),
    "dict": lambda lib: lib.DataFrame({"p": four(lib)}).to_dict(),
    "str-rep": lambda lib: str(lib.Series([lib.Period("2026-03-15 10:00", "h")])),
    "isin": lambda lib: lib.Series(four(lib)).isin([lib.Period("2026-01", "M")]).tolist(),
    "where": lambda lib: lib.Series(four(lib)).where([True, False, True, True]),
    "to_numpy": lambda lib: lib.Series(four(lib)).to_numpy(),
    "values": lambda lib: lib.Series(four(lib)).values,
    "array": lambda lib: lib.Series(four(lib)).array,
    "sort-desc": lambda lib: series(lib).sort_values(ascending=False),
    "dedup": lambda lib: series(lib).drop_duplicates(),
    "dup": lambda lib: series(lib).duplicated().tolist(),
    "head": lambda lib: series(lib).head(2),
    "iloc-slice": lambda lib: series(lib).iloc[1:3],
    "mask": lambda lib: series(lib)[series(lib).notna()],
    "lt": lambda lib: (series(lib) < lib.Period("2026-02", "M")).tolist(),
    "lt-other": lambda lib: (series(lib) < lib.Period("2026-02-01", "D")).tolist(),
    "eq-series": lambda lib: (series(lib) == series(lib)).tolist(),
    "asfreq-D": lambda lib: series(lib).astype("period[D]"),
    "from-str": lambda lib: lib.Series(["2026-01", None]).astype("period[M]"),
    "wrong-freq": lambda lib: lib.Series([lib.Period("2026-01-01", "D")], dtype="period[M]"),
    "from-stamp": lambda lib: lib.Series([lib.Timestamp("2026-01-15")], dtype="period[M]"),
    "frame-sort": lambda lib: frame(lib).sort_values("p"),
    "frame-min": lambda lib: frame(lib)["p"].min(),
    "gb-min": lambda lib: frame(lib).groupby("v")["p"].min(),
    "gb-first": lambda lib: frame(lib).groupby("v")["p"].first(),
    "records": lambda lib: frame(lib).to_dict("records")[:2],
    "merge": lambda lib: frame(lib).merge(
        lib.DataFrame({"p": [lib.Period("2026-01", "M")], "w": [9]}), on="p"
    ),
    "reverse-sub": lambda lib: lib.Period("2027-01", "M") - series(lib),
    "radd": lambda lib: 2 + series(lib),
    "sub-int": lambda lib: series(lib) - 1,
    "mul": lambda lib: series(lib) * 2,
    "tolist-drop": lambda lib: series(lib).dropna().tolist(),
    "replace": lambda lib: series(lib).replace(
        lib.Period("2026-01", "M"), lib.Period("2000-01", "M")
    ),
    "map-str": lambda lib: series(lib).map(str),
    "str-dtype": lambda lib: str(series(lib).dtype),
    "dtype-eq": lambda lib: series(lib).dtype == "period[M]",
    "reindex": lambda lib: series(lib).reindex([0, 9]),
    "to_frame": lambda lib: series(lib).to_frame(),
    "describe": lambda lib: series(lib).describe(),
    "hourly": lambda lib: lib.Series(
        [lib.Period("2026-01-01 10:00", "h"), lib.Period("2026-01-01 13:00", "h")]
    ),
}


def mistake(error: Exception) -> str:
    kind = next(kind.__name__ for kind in type(error).__mro__ if kind.__module__ == "builtins")
    return f"{kind}: {error}"


def outcome(build: Callable[[], Any]) -> str:
    try:
        return repr(build())
    except Exception as error:
        return mistake(error)


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_answers_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))


def test_the_dtype_is_the_period_type() -> None:
    assert fp.Series(four(fp)).dtype == fp.PeriodDtype("M")
