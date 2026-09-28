"""Text joined and repeated by operators, lists beside a frame, and spans beside NaT.

Each case runs in both libraries. A series is compared by its type, its name,
its labels and its index, a frame by its cells and types, and a mistake by its
class name and message, where firepanda's own classes stand in for pandas' by
being subclasses of them. A missing text cell is None in firepanda and NaN in
pandas, so both are read as None.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


def missing_as_none(values: list[Any]) -> list[Any]:
    return [None if value is None or value != value else value for value in values]


def outcome(call: Callable[[ModuleType], Any], lib: ModuleType) -> Any:
    try:
        got = call(lib)
    except Exception as error:
        return "error", error
    if isinstance(got, (fp.DataFrame, pd.DataFrame)):
        cells = {name: missing_as_none(got[name].tolist()) for name in got.columns}
        return cells, got.index.tolist(), [str(one) for one in got.dtypes]
    if isinstance(got, (fp.Series, pd.Series)):
        return got.name, missing_as_none(got.tolist()), got.index.tolist()
    return str(got)


def same(call: Callable[[ModuleType], Any]) -> None:
    ours, theirs = outcome(call, fp), outcome(call, pd)
    if theirs[0] == "error":
        assert ours[0] == "error"
        assert isinstance(ours[1], type(theirs[1])) or (
            type(ours[1]).__name__ == type(theirs[1]).__name__
        )
        assert str(ours[1]) == str(theirs[1])
    else:
        assert ours == theirs


def text(lib: ModuleType, *values: Any, **options: Any) -> Any:
    return lib.Series(list(values), **options)


def frame(lib: ModuleType) -> Any:
    return lib.DataFrame({"a": [1, 2], "b": [3, 4]})


def spans(lib: ModuleType) -> Any:
    return lib.Series(lib.to_timedelta(["1h", None]), name="s")


TEXT: list[Callable[[ModuleType], Any]] = [
    lambda lib: text(lib, "a", "b", name="n") + "x",
    lambda lib: "x" + text(lib, "a", None, name="n"),
    lambda lib: text(lib, "a", None) + text(lib, "x", "y"),
    lambda lib: text(lib, "a", "b") + text(lib, "x", "y", index=[1, 2]),
    lambda lib: text(lib, "a", "b") + ["x", "y"],  # noqa: RUF005
    lambda lib: text(lib, "a", name="x") + text(lib, "b", name="y"),
    lambda lib: text(lib, "a", name="x") + text(lib, "b", name="x"),
    lambda lib: text(lib, "a").add("z"),
    lambda lib: text(lib, "a").radd("z"),
    lambda lib: text(lib, "a", "b", name="n") * text(lib, 2, 0),
    lambda lib: text(lib, "a", "b", name="n") * -1,
    lambda lib: 3 * text(lib, "a", None, name="n"),
    lambda lib: text(lib, "ab", "c").mul(2),
    lambda lib: text(lib, "a") * True,
    lambda lib: text(lib, "a") * 1.5,
    lambda lib: text(lib, "a") + 1,
    lambda lib: 1 + text(lib, "a"),
    lambda lib: text(lib, "a") - "x",
    lambda lib: "x" - text(lib, "a"),
    lambda lib: text(lib, "a") / 2,
    lambda lib: text(lib, "a") ** 2,
    lambda lib: text(lib, "a") % "x",
    lambda lib: text(lib, "a") - text(lib, "b"),
    lambda lib: text(lib, "a") + text(lib, 1),
    lambda lib: text(lib, "a") + 1.5,
    lambda lib: text(lib, "a", None, name="n") == 1,
    lambda lib: text(lib, "a", None, name="n") != 1.5,
    lambda lib: 1 == text(lib, "a", name="n"),  # noqa: SIM300
    lambda lib: text(lib, "a").eq(1),
    lambda lib: text(lib, "a") < 1,
    lambda lib: text(lib, "a").ge(True),
    lambda lib: 1 < text(lib, "a"),  # noqa: SIM300
    lambda lib: text(lib, "a") == "a",
    lambda lib: text(lib, "a") < text(lib, "b"),
]


@pytest.mark.parametrize("call", TEXT)
def test_text_operators_are_pandas_operators(call: Callable[[ModuleType], Any]) -> None:
    same(call)


LISTED: list[Callable[[ModuleType], Any]] = [
    lambda lib: frame(lib) + [1, 10],  # noqa: RUF005
    lambda lib: frame(lib) + [1, 2, 3],  # noqa: RUF005
    lambda lib: frame(lib).add([1, 2], axis=0),
    lambda lib: frame(lib).add([1, 2, 3], axis=0),
    lambda lib: frame(lib).sub([1, 2], axis="index"),
    lambda lib: frame(lib) == [1, 3],
    lambda lib: frame(lib) == [1, 3, 4],
    lambda lib: frame(lib) * (1, 2),
    lambda lib: frame(lib) - np.array([1, 2]),
    lambda lib: [1, 2] - frame(lib),
    lambda lib: frame(lib) / [2, 4],
]


@pytest.mark.parametrize("call", LISTED)
def test_a_list_beside_a_frame_is_pandas_list(call: Callable[[ModuleType], Any]) -> None:
    same(call)


NAT: list[Callable[[ModuleType], Any]] = [
    lambda lib: spans(lib) / lib.NaT,
    lambda lib: spans(lib) // lib.NaT,
    lambda lib: spans(lib) * lib.NaT,
    lambda lib: spans(lib) % lib.NaT,
    lambda lib: spans(lib) ** lib.NaT,
    lambda lib: lib.NaT / spans(lib),
    lambda lib: lib.NaT // spans(lib),
    lambda lib: lib.NaT * spans(lib),
    lambda lib: spans(lib).truediv(lib.NaT),
    lambda lib: spans(lib) + lib.NaT,
    lambda lib: spans(lib) - lib.NaT,
]


@pytest.mark.parametrize("call", NAT)
def test_spans_beside_nat_are_pandas_spans(call: Callable[[ModuleType], Any]) -> None:
    same(call)
