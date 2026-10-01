"""union_categoricals over categoricals, category columns and category indexes."""

import pytest
from firepanda.errors import DTypeError

import firepanda as fp


def _union(*parts, **options):
    out = fp.api.types.union_categoricals(list(parts), **options)
    return out.tolist(), out.categories.tolist(), out.ordered


def test_different_categories_come_in_order_of_appearance():
    left = fp.Categorical(["b"], categories=["b", "z"])
    assert _union(left, fp.Categorical(["a"])) == (["b", "a"], ["b", "z", "a"], False)
    assert _union(fp.Categorical(["b", "a"]), fp.Categorical(["c", "a"]))[1] == ["a", "b", "c"]
    sorted_labels = _union(fp.Categorical(["b"]), fp.Categorical(["a"]), sort_categories=True)
    assert sorted_labels == (["b", "a"], ["a", "b"], False)


def test_same_categories_keep_the_first_order():
    first = fp.Categorical(["a", "b"], categories=["b", "a"])
    assert _union(first, fp.Categorical(["a"], categories=["a", "b"])) == (
        ["a", "b", "a"],
        ["b", "a"],
        False,
    )
    left = fp.Categorical(["a"], categories=["b", "a"], ordered=True)
    right = fp.Categorical(["b"], categories=["b", "a"], ordered=True)
    assert _union(left, right) == (["a", "b"], ["b", "a"], True)
    assert _union(left, right, ignore_order=True)[2] is False


def test_columns_indexes_gaps_and_numbers():
    columns = [fp.Series(["b"], dtype="category"), fp.Series(["a", "b"], dtype="category")]
    assert _union(*columns) == (["b", "a", "b"], ["b", "a"], False)
    indexes = [fp.CategoricalIndex(["x"]), fp.CategoricalIndex(["y"])]
    assert _union(*indexes)[1] == ["x", "y"]
    values, labels, _ = _union(fp.Categorical(["a", None]), fp.Categorical(["b"]))
    assert values[0] == "a"
    assert values[1] != values[1]
    assert labels == ["a", "b"]
    assert _union(fp.Categorical([3, 1]), fp.Categorical([2])) == ([3, 1, 2], [1, 3, 2], False)


def test_refusals_follow_pandas():
    ordered = fp.Categorical(["a"], ordered=True)
    with pytest.raises(DTypeError, match="all categories must be the same"):
        _union(ordered, fp.Categorical(["b"], ordered=True))
    assert _union(ordered, fp.Categorical(["b"], ordered=True), ignore_order=True)[2] is False
    with pytest.raises(DTypeError, match=r"Categorical\.ordered must be the same"):
        _union(ordered, fp.Categorical(["b"]))
    with pytest.raises(DTypeError, match="sort_categories=True with ordered"):
        _union(ordered, ordered, sort_categories=True)
    with pytest.raises(DTypeError, match="dtype of categories must be the same"):
        _union(fp.Categorical(["a"]), fp.Categorical([1]))
    with pytest.raises(DTypeError, match="must be Categorical"):
        _union(fp.Categorical(["a"]), fp.Series(["b"]))
