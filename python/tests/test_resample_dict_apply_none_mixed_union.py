"""A column resampler's agg by a mapping, apply's None groups, text beside numbers in set ops."""

import pytest
from firepanda.errors import SpecificationError

import firepanda as fp


def resampler():
    labels = fp.date_range("20130101", periods=5, freq="s")
    return fp.Series([1, 2, 3, 4, 5], index=labels).resample("2s")


def test_a_column_resampler_names_columns_after_the_mapping():
    out = resampler().agg({"total": "sum", "top": "max"})
    assert out.columns.tolist() == ["total", "top"]
    assert out["total"].tolist() == [3, 7, 5]
    assert out["top"].tolist() == [2, 4, 5]
    with pytest.raises(SpecificationError, match="nested renamer"):
        resampler().agg({"a": ["max"]})


def test_apply_leaves_out_the_groups_that_answer_none():
    frame = fp.DataFrame({"A": ["a", "a", "b"], "B": [1, 2, 3], "C": [4, 6, 5]})
    out = frame.groupby("A", group_keys=False).apply(lambda x: None if x.iloc[0, 0] == 3 else x)
    assert out.index.tolist() == [0, 1]
    assert out["B"].tolist() == [1, 2]


def test_text_beside_numbers_in_set_operations_is_objects():
    words, counts = fp.Index(["a", "b", "c", "d"]), fp.Index([1, 2, 3, 4])
    assert words.union(counts).tolist() == ["a", "b", "c", "d", 1, 2, 3, 4]
    assert str(words.union(counts).dtype) == "object"
    assert counts.union(words).tolist() == [1, 2, 3, 4, "a", "b", "c", "d"]
    assert fp.Index(["b", "a", "a"]).union(fp.Index([2, 1, 2])).tolist() == [1, 2, 2, "a", "a", "b"]
    assert words.difference(counts).tolist() == ["a", "b", "c", "d"]
    assert words.symmetric_difference(counts).tolist() == [1, 2, 3, 4, "a", "b", "c", "d"]
    assert len(words.intersection(counts)) == 0
