"""Wrong calls on a group by or a window, which name the method as pandas names it.

Python words the refusal itself, starting with the function's qualified name,
so each method carries the name pandas' own definition has: `GroupBy.sum` for
both kinds of group by, `Rolling.sum` and `Expanding.sum` for the one function
firepanda shares between the windows.
"""

from __future__ import annotations

from types import ModuleType

import pytest

_UNKNOWN = r"^{}\(\) got an unexpected keyword argument 'zz'$"


def _frame(firepanda: ModuleType) -> object:
    return firepanda.DataFrame({"a": [1, 1, 2], "b": [1.5, 2.5, 3.5]})


@pytest.mark.parametrize(
    ("method", "shown"),
    [
        ("sum", "GroupBy.sum"),
        ("head", "GroupBy.head"),
        ("rank", "GroupBy.rank"),
        ("get_group", "BaseGroupBy.get_group"),
        ("idxmax", "{kind}.idxmax"),
        ("value_counts", "{kind}.value_counts"),
        ("skew", "group_skew"),
        ("kurt", "group_kurt"),
        ("nth", "GroupByNthSelector.__call__"),
    ],
)
def test_a_group_by_names_the_method_pandas_defines(
    firepanda: ModuleType, method: str, shown: str
) -> None:
    """A method of pandas' GroupBy reads the same on a frame's and a column's."""
    grouped = _frame(firepanda).groupby("a")
    for owner, kind in ((grouped, "DataFrameGroupBy"), (grouped["b"], "SeriesGroupBy")):
        with pytest.raises(TypeError, match=_UNKNOWN.format(shown.format(kind=kind))):
            getattr(owner, method)(zz=1)


def test_apply_and_describe_differ_by_kind(firepanda: ModuleType) -> None:
    """pandas defines these again for a column, so the two kinds print different names."""
    grouped = _frame(firepanda).groupby("a")
    with pytest.raises(TypeError, match=_UNKNOWN.format("GroupBy.describe")):
        grouped.describe(zz=1)
    with pytest.raises(TypeError, match=_UNKNOWN.format("SeriesGroupBy.describe")):
        grouped["b"].describe(zz=1)


def test_each_window_names_its_own_method(firepanda: ModuleType) -> None:
    """A reduction shared by the windows is named after the window it is called on."""
    frame = _frame(firepanda)
    for window, kind in (
        (frame.rolling(2), "Rolling"),
        (frame.expanding(), "Expanding"),
        (frame.groupby("a").rolling(2), "Rolling"),
        (frame.ewm(com=1), "ExponentialMovingWindow"),
    ):
        for method in ("sum", "mean", "std"):
            with pytest.raises(TypeError, match=_UNKNOWN.format(f"{kind}.{method}")):
                getattr(window, method)(zz=1)
    assert frame.rolling(2).sum()["a"].tolist()[1:] == [2.0, 3.0]
    assert frame.expanding().sum()["a"].tolist() == [1.0, 2.0, 4.0]


def test_a_resampler_names_what_it_takes_from_a_group_by(firepanda: ModuleType) -> None:
    """`quantile` is GroupBy's in pandas, and the rest are the resampler's own."""
    frame = _frame(firepanda).set_index(
        firepanda.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
    )
    with pytest.raises(TypeError, match=_UNKNOWN.format("GroupBy.quantile")):
        frame.resample("D").quantile(zz=1)
    with pytest.raises(TypeError, match=_UNKNOWN.format("Resampler.sum")):
        frame.resample("D").sum(zz=1)


def test_agg_with_nothing_to_do(firepanda: ModuleType) -> None:
    """Each kind of owner has its own sentence for an `agg` with no function."""
    frame = _frame(firepanda)
    for window in (frame.rolling(2), frame.expanding(), frame.ewm(com=1)):
        with pytest.raises(TypeError, match=r"^Must provide 'func' or tuples of"):
            window.agg()
        with pytest.raises(TypeError, match=r"^Must provide 'func' or tuples of"):
            window.agg(zz=1)
    column = frame.groupby("a")["b"]
    with pytest.raises(TypeError, match=r"^Must provide 'func' or named aggregation \*\*kwargs\.$"):
        column.agg()
    with pytest.raises(TypeError, match=r"^func is expected but received int in \*\*kwargs\.$"):
        column.agg(zz="sum", yy=1)
    assert column.agg(zz="sum")["zz"].tolist() == [4.0, 3.5]


def test_a_window_takes_named_aggregation(firepanda: ModuleType) -> None:
    """`name=(column, reduction)` gives a frame with a column a name, as in pandas."""
    frame = firepanda.DataFrame({"a": [1, 1, 2], "b": [1.5, 2.5, 3.5]}, index=[5, 6, 7])
    got = frame.rolling(2).agg(x=("a", "sum"), y=("b", "max"))
    assert list(got.columns) == ["x", "y"]
    assert got.index.tolist() == [5, 6, 7]
    assert got["x"].tolist()[1:] == [2.0, 3.0]
    assert got["y"].tolist()[1:] == [2.5, 3.5]
    assert frame.expanding().agg(x=("b", "mean"))["x"].tolist() == [1.5, 2.0, 2.5]
    with pytest.raises(KeyError, match=r"Label\(s\) \['zz'\] do not exist"):
        frame.rolling(2).agg(x=("zz", "sum"))
    with pytest.raises(firepanda.errors.SpecificationError, match="nested renamer"):
        frame["a"].rolling(2).agg(x=("a", "sum"))


def test_a_range_reads_no_numpy_keyword_on_all_and_any(firepanda: ModuleType) -> None:
    """pandas' range answers `all` and `any` without looking at numpy's keywords."""
    labels = firepanda.RangeIndex(3)
    assert labels.all(dtype=1) is False
    assert labels.any(out=1) is True
    assert firepanda.RangeIndex(1, 3).all() is True
    with pytest.raises(ValueError, match="the 'keepdims' parameter is not supported"):
        labels.min(keepdims=True)
