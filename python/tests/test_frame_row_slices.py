"""df[slice] picks rows, and label slices read backwards with a negative step."""

import firepanda as fp


def _frame():
    return fp.DataFrame({"a": [1, 2, 3, 4]}, index=list("wxyz"))


def test_whole_number_slices_pick_rows_by_position():
    assert _frame()[::-1].index.tolist() == ["z", "y", "x", "w"]
    assert _frame()[:2]["a"].tolist() == [1, 2]
    assert _frame()[1::2].index.tolist() == ["x", "z"]
    assert _frame()[5:].index.tolist() == []
    numbered = fp.DataFrame({"a": [1, 2, 3]}, index=[10, 20, 30])
    assert numbered[0:2].index.tolist() == [10, 20]
    assert numbered[-1:].index.tolist() == [30]
    floats = fp.DataFrame({"a": [1, 2, 3]}, index=[1.0, 2.0, 3.0])
    assert floats[1:2].index.tolist() == [2.0]


def test_label_slices_pick_rows_by_label():
    assert _frame()["x":"y"].index.tolist() == ["x", "y"]
    assert _frame()["y":"w":-1]["a"].tolist() == [3, 2, 1]
    dated = fp.DataFrame({"a": [1, 2, 3]}, index=fp.date_range("2024-01-01", periods=3))
    assert dated["2024-01-02":]["a"].tolist() == [2, 3]


def test_backward_label_slices_through_loc():
    frame = _frame()
    assert frame.loc["y":"w":-1].index.tolist() == ["y", "x", "w"]
    assert frame.loc["z"::-2].index.tolist() == ["z", "x"]
    assert frame["a"].loc[:"x":-1].index.tolist() == ["z", "y", "x"]
    assert frame["a"]["y":"w":-1].tolist() == [3, 2, 1]
    assert fp.Index(list("wxyz")).slice_indexer("y", "w", -1) == slice(2, -5, -1)
    dated = fp.Series([1, 2, 3], index=fp.date_range("2024-01-01", periods=3))
    assert dated.loc["2024-01-03":"2024-01-01":-1].tolist() == [3, 2, 1]
