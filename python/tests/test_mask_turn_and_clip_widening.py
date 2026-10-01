"""mask's turn before alignment, regex replace by column, empty reductions, clip and crosstab."""

import math

import firepanda as fp


def test_mask_with_a_short_condition_replaces_the_rows_it_does_not_name():
    s = fp.Series(range(5))
    flags = fp.Series([True, False])
    assert s.mask(flags, 99).tolist() == [99, 1, 99, 99, 99]
    assert s.where(flags, 99).tolist() == [0, 99, 99, 99, 99]
    frame = fp.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    out = frame.mask(fp.DataFrame({"a": [True, False]}), 0)
    assert out["a"].tolist() == [0, 2, 0]
    assert out["b"].tolist() == [0, 0, 0]


def test_regex_replace_with_a_pattern_and_a_text_under_one_column_name():
    frame = fp.DataFrame({"A": ["bat", "foo", "bait"], "B": ["abc", "bar", "xyz"]})
    out = frame.replace({"A": r"^ba.$"}, {"A": "new"}, regex=True)
    assert out["A"].tolist() == ["new", "foo", "bait"]
    assert out["B"].tolist() == ["abc", "bar", "xyz"]


def test_a_frame_of_no_columns_answers_flags_and_counts():
    assert str(fp.DataFrame([]).any().dtype) == "bool"
    assert str(fp.DataFrame([]).all().dtype) == "bool"
    assert str(fp.DataFrame([]).count().dtype) == "int64"


def test_clipping_whole_numbers_by_bounds_with_a_gap_makes_floats():
    s = fp.Series([9, -3, 0, -1, 5])
    out = s.clip(fp.Series([2, -4, float("nan"), 6, 3]))
    assert str(out.dtype) == "float64"
    assert out.tolist() == [9.0, -3.0, 0.0, 6.0, 5.0]
    assert str(s.clip(fp.Series([2.0, -4, 1, 6, 3])).dtype) == "int64"
    assert str(s.clip(fp.Series([-20, -40, float("nan"), -60, -30])).dtype) == "int64"
    frame = fp.DataFrame({"a": [9, -3], "b": [1, 2]})
    out = frame.clip(fp.Series([2, float("nan")]), axis=0)
    assert [str(d) for d in out.dtypes.tolist()] == ["int64", "float64"]
    assert out["b"].tolist() == [2.0, 2.0]


def test_crosstab_without_dropna_shows_every_category():
    foo = fp.Categorical(["a", "b"], categories=["a", "b", "c"])
    bar = fp.Categorical(["d", "e"], categories=["d", "e", "f"])
    out = fp.crosstab(foo, bar, dropna=False)
    assert out.index.tolist() == ["a", "b", "c"]
    assert out.columns.tolist() == ["d", "e", "f"]
    assert out["f"].tolist() == [0, 0, 0]
    assert fp.crosstab(foo, bar).columns.tolist() == ["d", "e"]
    assert not math.isnan(out["d"].tolist()[2])
