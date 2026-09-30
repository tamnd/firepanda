"""pandas' masked text type, `string`, which is a different type from the plain text `str`.

pandas 3 reads text as `str` by default, and a gap in it prints as NaN. Asking for `string`,
by name or as `StringDtype()`, gives the older masked type instead: a gap is `NA`, a
comparison answers flags with a gap in them, and the `str` accessor answers in masked types,
`Int64` for a length and `boolean` for a test. firepanda holds the type the way it holds
`Int64`, as cells over the lower case column, and these cases check each answer against pandas.

Two answers differ on purpose and are checked for their values only. pandas answers a
comparison as `bool[pyarrow]` and `value_counts` as `int64[pyarrow]`, and firepanda answers the
masked `boolean` and `Int64` types, which hold the same values.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def column(pd: ModuleType) -> Any:
    """The column every case starts from, with a gap and a repeat in it."""
    return pd.Series(["ab", None, "c", "ab"], dtype="string")


def outcome(run: Callable[[], Any]) -> Any:
    """What a case answers, as types and printed values that can be compared across libraries."""
    try:
        answer = run()
    except Exception as error:
        return type(error).__name__
    if hasattr(answer, "dtypes") and not hasattr(answer, "dtype"):
        return [str(dtype) for dtype in answer.dtypes], answer.to_string()
    if hasattr(answer, "dtype"):
        return str(answer.dtype), repr(answer)
    return repr(answer)


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "construct": column,
    "dtype": lambda pd: pd.Series(["x", None], dtype=pd.StringDtype()),
    "astype": lambda pd: pd.Series(["x", None]).astype("string"),
    "back to str": lambda pd: column(pd).astype(str),
    "to object": lambda pd: column(pd).astype(object).tolist(),
    "isna": lambda pd: column(pd).isna(),
    "add": lambda pd: column(pd) + "!",
    "len": lambda pd: column(pd).str.len(),
    "find": lambda pd: column(pd).str.find("b"),
    "contains": lambda pd: column(pd).str.contains("a"),
    "startswith": lambda pd: column(pd).str.startswith("a"),
    "upper": lambda pd: column(pd).str.upper(),
    "strip": lambda pd: column(pd).str.strip(),
    "split": lambda pd: column(pd).str.split("b", expand=True),
    "extract": lambda pd: column(pd).str.extract("(a)"),
    "cat": lambda pd: column(pd).str.cat(sep="-"),
    "fillna": lambda pd: column(pd).fillna("q"),
    "dropna": lambda pd: column(pd).dropna(),
    "sort": lambda pd: column(pd).sort_values(),
    "where": lambda pd: column(pd).where(column(pd) != "c"),
    "unique": lambda pd: column(pd).unique(),
    "isin": lambda pd: column(pd).isin(["ab"]),
    "replace": lambda pd: column(pd).replace("ab", "z"),
    "map": lambda pd: column(pd).map(lambda value: value),
    "item": lambda pd: column(pd)[1],
    "max": lambda pd: column(pd).max(),
    "nunique": lambda pd: column(pd).nunique(),
    "concat": lambda pd: pd.concat([column(pd), column(pd)], ignore_index=True),
    "groupby": lambda pd: pd.DataFrame({"k": column(pd), "v": [1, 2, 3, 4]}).groupby("k").v.sum(),
    "frame": lambda pd: pd.DataFrame({"k": column(pd), "v": [1, 2, 3, 4]}),
    "csv": lambda pd: pd.DataFrame({"k": column(pd)}).to_csv(),
    "convert": lambda pd: pd.DataFrame({"a": ["x", None], "b": [1, 2]}).convert_dtypes(),
    "convert no text": lambda pd: pd.Series(["x", None]).convert_dtypes(convert_string=False),
}


@needs_pandas
@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_answers_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    import pandas as pd

    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))


@needs_pandas
@pytest.mark.parametrize(
    "case",
    [
        lambda pd: column(pd) == "ab",
        lambda pd: column(pd) < "b",
        lambda pd: column(pd).value_counts(),
    ],
    ids=["eq", "lt", "value_counts"],
)
def test_arrow_answers_hold_the_same_values(case: Callable[[ModuleType], Any]) -> None:
    import pandas as pd

    ours, theirs = case(fp), case(pd)
    assert [str(value) for value in ours.tolist()] == [str(value) for value in theirs.tolist()]
    assert list(ours.index) == list(theirs.index)


def test_the_accessor_puts_a_gap_back_where_the_row_was_one() -> None:
    answer = column(fp).str.len()
    assert str(answer.dtype) == "Int64"
    assert answer.isna().tolist() == [False, True, False, False]


def test_a_replace_in_place_changes_the_column() -> None:
    text = column(fp)
    text.replace("ab", "z", inplace=True)
    assert str(text.dtype) == "string"
    assert text.tolist()[0] == "z"
