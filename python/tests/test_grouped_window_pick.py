"""A window over groups picks columns, labels by `on`, and leaves `win_type` unused as pandas."""

import math

import pytest

import firepanda as fp


def _frame():
    return fp.DataFrame(
        {
            "g": ["a", "b", "a", "b"],
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [1, 2, 3, 4],
            "t": fp.date_range("2024-01-01", periods=4, freq="D"),
        }
    )


def test_picked_column_of_a_rolling_window():
    out = _frame().groupby("g").rolling(2)["y"].mean()
    assert out.index.tolist() == [("a", 0), ("a", 2), ("b", 1), ("b", 3)]
    values = out.tolist()
    assert math.isnan(values[0])
    assert math.isnan(values[2])
    assert [values[1], values[3]] == [2.0, 3.0]


def test_picked_columns_and_attribute():
    grouped = _frame().groupby("g")
    assert grouped.rolling(2)[["x", "y"]].max().columns.tolist() == ["x", "y"]
    assert grouped.rolling(2).y.sum().tolist()[1] == 4.0
    assert grouped.expanding()["y"].sum().tolist() == [1.0, 4.0, 2.0, 6.0]


def test_picked_column_of_an_ewm():
    out = _frame().groupby("g").ewm(com=1)["y"].mean()
    assert out.tolist()[0] == 1.0
    assert out.tolist()[1] == pytest.approx(7 / 3)


def test_on_labels_a_picked_answer_by_its_instants():
    out = _frame().groupby("g").rolling("2D", on="t", closed="both")["x"].sum()
    assert out.index.names == ["g", "t"]
    assert out.index.tolist()[0] == ("a", fp.Timestamp("2024-01-01"))
    assert out.tolist() == [1.0, 4.0, 2.0, 6.0]


def test_missing_column_is_a_key_error():
    with pytest.raises(KeyError, match="zz"):
        _frame().groupby("g").rolling(2)["zz"]


def test_win_type_is_unused_over_groups():
    grouped = _frame().groupby("g")
    weighted = grouped["x"].rolling(2, win_type="triang").sum().tolist()
    plain = grouped["x"].rolling(2).sum().tolist()
    assert weighted[1] == plain[1] == 4.0
