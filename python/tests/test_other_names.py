"""Column names and series names that are not text, compared with pandas.

Each case runs in both libraries and the answers are compared by their repr,
or for a mistake by its class name.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")


def numbered(lib: ModuleType) -> Any:
    return lib.DataFrame({0: [2, 1], 1: [3, 4]})


def other(lib: ModuleType) -> Any:
    return lib.DataFrame({0: [1, 2], 5: [7.5, 8.5]})


def outcome(call: Callable[[ModuleType], Any], lib: ModuleType) -> Any:
    try:
        got = call(lib)
    except Exception as error:
        return "error", type(error).__name__
    return repr(got)


CASES: list[Callable[[ModuleType], Any]] = [
    lambda lib: numbered(lib),
    lambda lib: numbered(lib).columns,
    lambda lib: lib.DataFrame([[1, 2], [3, 4]]),
    lambda lib: lib.DataFrame([[1, 2], [3, 4]]).columns,
    lambda lib: lib.DataFrame([1, 2]),
    lambda lib: lib.DataFrame({0.5: [1], 2.5: [2]}).columns,
    lambda lib: numbered(lib)[0],
    lambda lib: numbered(lib)[[1, 0]],
    lambda lib: numbered(lib)[0].name,
    lambda lib: 0 in numbered(lib),
    lambda lib: list(numbered(lib)),
    lambda lib: numbered(lib).drop(columns=0),
    lambda lib: numbered(lib).rename(columns={0: 7}),
    lambda lib: numbered(lib).rename(columns=str).columns,
    lambda lib: numbered(lib).sort_values(by=0),
    lambda lib: numbered(lib).groupby(0).sum(),
    lambda lib: numbered(lib).groupby(0)[1].mean(),
    lambda lib: numbered(lib).groupby(0, as_index=False).sum(),
    lambda lib: numbered(lib).set_index(0),
    lambda lib: numbered(lib).loc[:, 1],
    lambda lib: numbered(lib).loc[:, [1]],
    lambda lib: numbered(lib).iloc[:, 0].name,
    lambda lib: numbered(lib).astype({0: "float64"}),
    lambda lib: numbered(lib).reset_index(),
    lambda lib: lib.Series([1, 2], name=3).reset_index(),
    lambda lib: lib.Series([1, 2], name=3).to_frame(),
    lambda lib: numbered(lib).sum(),
    lambda lib: numbered(lib).mean().index,
    lambda lib: numbered(lib).describe(),
    lambda lib: numbered(lib).melt(),
    lambda lib: numbered(lib).melt(id_vars=[0]),
    lambda lib: numbered(lib).add_prefix("c"),
    lambda lib: numbered(lib).pop(0).name,
    lambda lib: numbered(lib).merge(other(lib), on=0),
    lambda lib: numbered(lib).merge(other(lib), left_index=True, right_index=True),
    lambda lib: lib.concat([numbered(lib), other(lib)]),
    lambda lib: lib.concat([numbered(lib), other(lib)], axis=1, ignore_index=True),
    lambda lib: numbered(lib).fillna({0: 1}),
    lambda lib: numbered(lib) + other(lib),
    lambda lib: numbered(lib).to_dict(),
    lambda lib: numbered(lib).to_csv(),
    lambda lib: numbered(lib).drop_duplicates(subset=[0]),
    lambda lib: numbered(lib).nlargest(1, 0),
    lambda lib: numbered(lib).reindex(columns=[1, 0]),
    lambda lib: numbered(lib).round({0: 1}),
    lambda lib: numbered(lib).T,
    lambda lib: numbered(lib).T.T,
    lambda lib: lib.DataFrame({"a": [1, 2], "b": [3, 4]}).T,
    lambda lib: lib.DataFrame({"a": [1]}, index=lib.to_datetime(["2020-01-01"])).T,
    lambda lib: pa.table(numbered(lib)).column_names,
]


@pytest.mark.parametrize("call", CASES)
def test_the_answer_is_pandas_answer(call: Callable[[ModuleType], Any]) -> None:
    assert outcome(call, fp) == outcome(call, pd)


def test_names_of_mixed_kinds_have_no_index_yet() -> None:
    """pandas holds names of mixed kinds in an index of objects, which firepanda lacks."""
    frame = fp.DataFrame({0: [1], "a": [2]})
    assert frame["a"].tolist() == [2]
    assert repr(frame) == repr(pd.DataFrame({0: [1], "a": [2]}))
    with pytest.raises(NotImplementedError, match="object index"):
        _ = frame.columns


def test_text_names_are_held_as_they_are() -> None:
    frame = fp.DataFrame({"a": [1], "\x1d": [2]})
    assert frame._inner.names() == ["a", "\x1d"]
    assert list(frame) == ["a", "\x1d"]
