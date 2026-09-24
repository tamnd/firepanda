"""`&`, `|` and `^` between two boolean operands, checked against pandas.

pandas lines two labelled operands up before it applies any of the three, and
the rule it follows over a gap is not symmetric: a row the right side is
missing reads the right side as False, and a row the left side is missing is a
False answer whatever the operator. So `a | b` and `b | a` differ, and the
cases below hold firepanda to both orders.
"""

from __future__ import annotations

import importlib.util
import math
import operator
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

OPERATORS = [operator.and_, operator.or_, operator.xor]
A = [True, False, True, False]
B = [True, True, False, False]


def labelled(m: Any, values: list[bool], labels: list[str]) -> Any:
    """A boolean series with string labels, built the one way both libraries take."""
    frame = m.DataFrame({"k": labels, "v": values}).set_index("k")
    return frame["v"].rename(None).rename_axis(None)


def plain(values: list[Any]) -> list[Any]:
    """NaN as None, so two lists with NaN in the same places compare equal."""
    return [None if isinstance(v, float) and math.isnan(v) else v for v in values]


def same_series(got: Any, want: Any) -> None:
    assert got.dtype == str(want.dtype)
    assert got.tolist() == want.tolist()
    assert got.index.tolist() == want.index.tolist()


def same_frame(got: Any, want: Any) -> None:
    assert list(got.columns) == list(want.columns)
    assert got.index.tolist() == want.index.tolist()
    for name in want.columns:
        assert got[name].dtype == str(want[name].dtype)
        assert plain(got[name].tolist()) == plain(want[name].tolist())


SERIES: list[Callable[[Any], tuple[Any, Any]]] = [
    lambda m: (m.Series(A), m.Series(B)),
    lambda m: (m.Series(A[:3]), m.Series(B).iloc[1:]),
    lambda m: (m.Series(B).iloc[1:], m.Series(A[:3])),
    lambda m: (labelled(m, A, ["d", "a", "c", "b"]), labelled(m, B[:3], ["a", "e", "d"])),
]


@pytest.mark.parametrize("op", OPERATORS)
@pytest.mark.parametrize("build", SERIES)
def test_two_series(firepanda: ModuleType, op: Callable[..., Any], build: Any) -> None:
    """Labelled the same, and with a gap on either side."""
    import pandas as pd

    same_series(op(*build(firepanda)), op(*build(pd)))


@pytest.mark.parametrize("op", OPERATORS)
@pytest.mark.parametrize("constant", [True, False])
def test_a_series_and_a_constant(firepanda: ModuleType, op: Any, constant: bool) -> None:
    """Either side of the operator."""
    import pandas as pd

    same_series(op(firepanda.Series(A), constant), op(pd.Series(A), constant))
    same_series(op(constant, firepanda.Series(A)), op(constant, pd.Series(A)))


FRAMES: list[Callable[[Any], tuple[Any, Any]]] = [
    lambda m: (m.DataFrame({"a": A, "b": B}), m.DataFrame({"a": B, "b": A})),
    lambda m: (m.DataFrame({"a": A, "b": B}), True),
    lambda m: (m.DataFrame({"a": A, "b": B}), m.DataFrame({"a": B, "c": A}).iloc[1:]),
    lambda m: (m.DataFrame({"a": B, "c": A}).iloc[1:], m.DataFrame({"a": A, "b": B})),
    lambda m: (m.DataFrame({"a": A, "b": B}), labelled(m, [True, False], ["a", "b"])),
    lambda m: (m.DataFrame({"a": A, "b": B}), labelled(m, [True, False], ["a", "c"])),
    lambda m: (m.DataFrame({"b": A, "a": B}), labelled(m, [True], ["a"])),
    lambda m: (labelled(m, [True, False], ["a", "b"]), m.DataFrame({"a": A, "b": B})),
]


@pytest.mark.parametrize("op", OPERATORS)
@pytest.mark.parametrize("build", FRAMES)
def test_a_frame(firepanda: ModuleType, op: Callable[..., Any], build: Any) -> None:
    """Against a frame, a constant and a series lined up with the columns."""
    import pandas as pd

    same_frame(op(*build(firepanda)), op(*build(pd)))


def test_a_mask_built_from_two_comparisons(firepanda: ModuleType) -> None:
    """The case the operators are mostly used for."""
    import pandas as pd

    answers = []
    for m in (firepanda, pd):
        s = m.Series([1, 2, 3, 4])
        answers.append(((s > 1) & (s < 4)) | (s == 4))
    same_series(*answers)


@pytest.mark.parametrize("constant", [1.5, None, "x"])
def test_a_constant_that_is_not_a_bool_is_a_type_error(
    firepanda: ModuleType, constant: Any
) -> None:
    """pandas raises a `TypeError` for it."""
    with pytest.raises(TypeError, match="Cannot perform 'and_'"):
        firepanda.Series(A) & constant


def test_two_integer_columns_are_refused(firepanda: ModuleType) -> None:
    """pandas answers the bitwise operation, which firepanda has not written."""
    with pytest.raises(NotImplementedError):
        firepanda.Series([1, 2]) & firepanda.Series([3, 1])
