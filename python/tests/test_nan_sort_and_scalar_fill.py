"""NaN in a float sort key is missing, and a fill value meets a single operand."""

import math

import firepanda as fp


def _shown(values):
    return ["nan" if isinstance(v, float) and math.isnan(v) else v for v in values]


def test_a_nan_sorts_last_both_ways_and_first_when_asked():
    s = fp.Series([float("nan"), 1, 3, 2], index=list("abcd"))
    assert _shown(s.sort_values(ascending=False).tolist()) == [3.0, 2.0, 1.0, "nan"]
    assert s.sort_values(ascending=False).index.tolist() == ["c", "d", "b", "a"]
    assert _shown(s.sort_values(na_position="first").tolist()) == ["nan", 1.0, 2.0, 3.0]
    assert s.sort_values(ascending=False, ignore_index=True).index.tolist() == [0, 1, 2, 3]


def test_a_frame_sorted_by_a_float_column_with_a_nan():
    frame = fp.DataFrame({"a": [float("nan"), 1, 3, 1], "b": [1, 2, 3, 4]}, index=[5, 6, 7, 8])
    assert frame.sort_values("a", ascending=False).index.tolist() == [7, 6, 8, 5]
    out = frame.sort_values(["a", "b"], ascending=[False, True], na_position="first")
    assert out.index.tolist() == [5, 7, 6, 8]


def test_an_index_and_row_labels_with_a_nan():
    index = fp.Index([float("nan"), 1.0, 3.0])
    assert _shown(index.sort_values(ascending=False).tolist()) == [3.0, 1.0, "nan"]
    assert list(index.sort_values(ascending=False, return_indexer=True)[1]) == [2, 1, 0]
    s = fp.Series([1, 2, 3, 4], index=[float("nan"), 2.0, 1.0, 3.0])
    assert s.sort_index(ascending=False).tolist() == [4, 2, 3, 1]


def test_a_fill_value_fills_the_series_before_a_single_value():
    s = fp.Series([1, 1, 1, float("nan")], index=list("abcd"))
    assert s.mul(5, fill_value=0).tolist() == [5.0, 5.0, 5.0, 0.0]
    assert s.add(5, fill_value=0).tolist() == [6.0, 6.0, 6.0, 5.0]
    assert s.lt(5, fill_value=0).tolist() == [True, True, True, True]
    assert _shown(s.add(float("nan"), fill_value=0).tolist()) == [1.0, 1.0, 1.0, "nan"]


def test_a_fill_value_fills_the_frame_before_a_single_value():
    frame = fp.DataFrame({"x": [1.0, float("nan")]})
    assert frame.add(1, fill_value=10)["x"].tolist() == [2.0, 11.0]
    assert frame.mul(2, fill_value=3)["x"].tolist() == [2.0, 6.0]


def test_whole_number_intervals_beside_a_gap_print_as_floats():
    bins = fp.IntervalIndex.from_tuples([(0, 1), (2, 3), (4, 5)])
    shown = repr(fp.cut([0, 0.5, 2.5], bins)).splitlines()
    assert shown[0] == "[NaN, (0.0, 1.0], (2.0, 3.0]]"
    assert shown[1].startswith("Categories (3, interval[int64, right]): [(0, 1]")
    assert repr(fp.cut([0.5], bins)).splitlines()[0] == "[(0, 1]]"
