"""`corr`, `cov` and `autocorr` on a column and a frame, checked against pandas.

pandas pairs two columns over the rows where both hold a value, so a gap in a
third column does not change a pair's answer, and Spearman ranks those rows
before correlating them. `DataFrame.cov` takes numpy's road when there is no gap
and the pairwise one when there is, and the two treat `ddof` differently. Every
answer here is compared with pandas' to nine significant figures.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

GAPS: dict[str, list[Any]] = {
    "a": [1, 2, 3, 4, 7],
    "b": [2.0, None, 1.0, 8.0, 3.0],
    "f": [True, False, True, True, False],
    "s": ["x", None, "z", "w", "v"],
}
FULL: dict[str, list[Any]] = {
    "a": [1, 2, 3, 4, 7],
    "b": [2.0, 5.0, 1.0, 8.0, 3.0],
    "c": [3, 3, 3, 3, 3],
}


def close(got: Any, want: Any) -> bool:
    """Equal, NaN equal to NaN, floats to nine significant figures."""
    if isinstance(got, float) or isinstance(want, float):
        if got != got or want != want:
            return got != got and want != want
        return math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12)
    return bool(got == want)


def agrees(got: Any, want: Any) -> None:
    """The same number, or the same square frame with the same labels both ways."""
    if not hasattr(want, "columns"):
        assert close(got, want)
        return
    assert list(got.columns) == list(want.columns)
    assert list(got.index) == list(want.index)
    for name in want.columns:
        assert str(got[name].dtype) == "float64"
        for mine, theirs in zip(got[name].tolist(), want[name].tolist(), strict=True):
            assert close(mine, theirs)


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.DataFrame(GAPS).corr(numeric_only=True),
    lambda m: m.DataFrame(GAPS).corr(numeric_only=True, method="spearman"),
    lambda m: m.DataFrame(GAPS).corr(numeric_only=True, min_periods=5),
    lambda m: m.DataFrame(GAPS).cov(numeric_only=True),
    lambda m: m.DataFrame(GAPS).cov(numeric_only=True, ddof=0),
    lambda m: m.DataFrame(GAPS).cov(numeric_only=True, min_periods=4),
    lambda m: m.DataFrame(FULL).corr(),
    lambda m: m.DataFrame(FULL).corr(method="spearman"),
    lambda m: m.DataFrame(FULL).cov(),
    lambda m: m.DataFrame(FULL).cov(ddof=0),
    lambda m: m.DataFrame(FULL).cov(min_periods=9),
    lambda m: m.DataFrame({"a": [1, 2]}).corr(),
    lambda m: m.DataFrame(GAPS)["a"].corr(m.DataFrame(GAPS)["b"]),
    lambda m: m.DataFrame(GAPS)["a"].corr(m.DataFrame(GAPS)["b"], min_periods=5),
    lambda m: m.DataFrame(GAPS)["a"].cov(m.DataFrame(GAPS)["b"]),
    lambda m: m.DataFrame(GAPS)["a"].cov(m.DataFrame(GAPS)["f"], ddof=0),
    lambda m: m.Series([1.0, 2, 3]).corr(m.Series([1.0, 2, 4], index=[1, 2, 3])),
    lambda m: m.Series([1.0, 2]).corr(m.Series([1.0, 2], index=[5, 6])),
    lambda m: m.Series([1.0, 2, 3]).corr(m.Series([5.0, 5, 5])),
    lambda m: m.Series([1.0]).cov(m.Series([1.0])),
    lambda m: m.Series([1.0, 2.0]).cov(m.Series([1.0, 3.0]), ddof=2),
    lambda m: m.Series([1.0, 2, 3, 5, 4]).autocorr(),
    lambda m: m.Series([1.0, 2, 3, 5, 4]).autocorr(2),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Pairwise gaps, flags, a constant column, labels that only partly meet, and ddof."""
    import warnings

    import pandas as pd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = build(pd)
    agrees(build(firepanda), want)


def test_spearman_on_a_column_is_the_rank_correlation(firepanda: ModuleType) -> None:
    """pandas asks scipy for this one, so it is checked against the ranks by hand.

    The shared rows are 1, 3, 4, 7 against 2, 1, 8, 3, whose ranks are 1, 2, 3, 4
    and 2, 1, 4, 3, and Pearson's r of those is 0.6.
    """
    frame = firepanda.DataFrame(GAPS)
    assert math.isclose(frame["a"].corr(frame["b"], method="spearman"), 0.6)


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: m.DataFrame(GAPS).corr(),
    lambda m: m.DataFrame(GAPS).cov(),
    lambda m: m.DataFrame(FULL).corr(method="nope"),
    lambda m: m.DataFrame(GAPS)["a"].corr(m.DataFrame(GAPS)["s"]),
    lambda m: m.DataFrame(GAPS)["a"].corr(m.DataFrame(GAPS)["b"], method="nope"),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_a_mistake_is_pandas_mistake(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Text is refused with numpy's words and an unknown method with pandas'."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(Exception) as mine:
        build(firepanda)
    assert isinstance(mine.value, type(theirs.value))
    assert str(mine.value) == str(theirs.value)


def test_kendall_is_refused(firepanda: ModuleType) -> None:
    """It counts pairs of rows that agree in order, which is not written yet."""
    with pytest.raises(NotImplementedError):
        firepanda.DataFrame(FULL).corr(method="kendall")


@pytest.mark.parametrize(
    ("owner", "name"),
    [
        ("DataFrame", "corr"),
        ("DataFrame", "cov"),
        ("Series", "corr"),
        ("Series", "cov"),
        ("Series", "autocorr"),
    ],
)
def test_the_signature_is_pandas_signature(firepanda: ModuleType, owner: str, name: str) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    ours = inspect.signature(getattr(getattr(firepanda, owner), name)).parameters
    yours = inspect.signature(getattr(getattr(pd, owner), name)).parameters
    assert [(p.name, p.kind, p.default) for p in ours.values()] == [
        (p.name, p.kind, p.default) for p in yours.values()
    ]
