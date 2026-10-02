"""Positions off the end and missing group columns are refused in pandas' words."""

import pytest

import firepanda as pd


def _frame():
    return pd.DataFrame({"a": [1, 2], "b": [3, 4]})


@pytest.mark.parametrize(
    ("pick", "message"),
    [
        (lambda f: f.iloc[:, 5], "single positional indexer is out-of-bounds"),
        (lambda f: f.iloc[:, [0, 5]], "positional indexers are out-of-bounds"),
        (lambda f: f.iloc[0, 5], "index 5 is out of bounds for axis 0 with size 2"),
        (lambda f: f.iloc[[0, 9]], "positional indexers are out-of-bounds"),
        (lambda f: f.iloc[9, 0], "index 9 is out of bounds for axis 0 with size 2"),
        (lambda f: f.take([9]), "indices are out-of-bounds"),
        (lambda f: f.take([5], axis=1), "indices are out-of-bounds"),
        (lambda f: f["a"].take([9]), "indices are out-of-bounds"),
        (lambda f: f["a"].iloc[[0, 9]], "positional indexers are out-of-bounds"),
    ],
)
def test_position_off_the_end(pick, message):
    with pytest.raises(IndexError) as caught:
        pick(_frame())
    assert str(caught.value) == message


def test_group_column_missing():
    grouped = _frame().groupby("a")
    with pytest.raises(KeyError) as caught:
        grouped["zz"]
    assert str(caught.value) == "'Column not found: zz'"
    with pytest.raises(KeyError) as caught:
        grouped[["zz", "b", "yy"]]
    assert str(caught.value) == "\"Columns not found: 'yy', 'zz'\""


def test_fillna_needs_a_value():
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'value'"):
        _frame().fillna()
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'value'"):
        _frame()["a"].fillna()
