"""Reductions of a frame made from nothing, and crosstab rows labelled by categories."""

import firepanda as fp


def test_a_frame_made_from_nothing_reduces_over_an_empty_range_index():
    frame = fp.DataFrame([])
    for answer in (frame.any(), frame.all(), frame.count()):
        assert repr(answer.index).startswith("RangeIndex")
        assert len(answer) == 0
    assert str(frame.count().dtype) == "int64"
    assert str(frame.any().dtype) == "bool"


def test_crosstab_rows_by_a_category_key_are_a_categorical_index():
    keys = fp.Categorical(["a", "b"], categories=["a", "b", "c"])
    every = fp.crosstab(keys, ["d", "e"], dropna=False)
    assert type(every.index).__name__ == "CategoricalIndex"
    assert every.index.tolist() == ["a", "b", "c"]
    assert list(every.index.categories) == ["a", "b", "c"]
    seen = fp.crosstab(keys, ["d", "e"])
    assert seen.index.tolist() == ["a", "b"]
    assert list(seen.index.categories) == ["a", "b", "c"]


def test_crosstab_rows_kept_follow_the_order_of_the_categories():
    keys = fp.Categorical(["a", "b"], categories=["c", "b", "a"], ordered=True)
    table = fp.crosstab(keys, ["d", "e"])
    assert table.index.tolist() == ["b", "a"]
    assert table.index.ordered
    assert table["d"].tolist() == [0, 1]
