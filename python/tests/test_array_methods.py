"""Array methods: positions, picks, map, shift, Categorical order and setting, intervals."""

import numpy as np
import pytest
from firepanda.errors import DTypeError, InvalidArgumentError

import firepanda as fp


def _numbers():
    return fp.array([3, 1, None, 4, 1], dtype="Int64")


def _letters():
    return fp.Categorical(["b", "b", "a", "c", None], categories=["c", "b", "a"], ordered=True)


def test_positions_of_the_largest_smallest_and_order():
    arr = _numbers()
    assert int(arr.argmax()) == 3
    assert int(arr.argmin()) == 1
    assert arr.argsort().tolist() == [1, 4, 0, 3, 2]
    assert arr.argsort(ascending=False).tolist() == [3, 0, 1, 4, 2]
    assert arr.argsort(na_position="first").tolist() == [2, 1, 4, 0, 3]
    assert _letters().argsort().tolist() == [3, 0, 1, 2, 4]
    with pytest.raises(InvalidArgumentError, match="Encountered an NA value with skipna=False"):
        arr.argmax(skipna=False)


def test_duplicated_isin_and_item():
    arr = _numbers()
    assert arr.duplicated().tolist() == [False, False, False, False, True]
    assert arr.duplicated(keep=False).tolist() == [False, True, False, False, True]
    found = arr.isin([1, 4])
    assert type(found).__name__ == "BooleanArray"
    assert found.tolist() == [False, True, False, True, True]
    assert _letters().isin(["a", "c"]).tolist() == [False, False, True, True, False]
    assert fp.array([7], dtype="Int64").item() == 7
    with pytest.raises(InvalidArgumentError, match="size 1"):
        arr.item()


def test_map_nbytes_searchsorted_shift_and_view():
    mapped = _numbers().map(lambda value: value * 2)
    assert np.isnan(mapped[2])
    assert mapped[[0, 1, 3, 4]].tolist() == [6.0, 2.0, 8.0, 2.0]
    upper = _letters().map(lambda value: value.upper())
    assert upper.categories.tolist() == ["C", "B", "A"]
    assert _numbers().nbytes == 45
    assert fp.array([1.5, 2.5]).nbytes == 18
    ordered = fp.array([1, 2, 3, 5], dtype="Int64")
    assert ordered.searchsorted([4]).tolist() == [3]
    assert int(ordered.searchsorted(2, side="right")) == 2
    shifted = _numbers().shift(-1, fill_value=0)
    assert type(shifted).__name__ == "IntegerArray"
    assert shifted.tolist()[-1] == 0
    assert _letters().shift(1).tolist()[1:] == ["b", "b", "a", "c"]
    assert _numbers().view().tolist()[:2] == [3, 1]


def test_categorical_sort_and_set():
    assert _letters().sort_values().tolist()[:4] == ["c", "b", "b", "a"]
    assert _letters().sort_values(ascending=False, na_position="first").tolist()[1:] == [
        "a",
        "b",
        "b",
        "c",
    ]
    cat = fp.Categorical(["a", "b", "c"])
    cat[2] = "a"
    cat[0:2] = ["c", "c"]
    assert cat.tolist() == ["c", "c", "a"]
    assert cat.categories.tolist() == ["a", "b", "c"]
    with pytest.raises(DTypeError, match=r"new category \(z\)"):
        cat[0] = "z"


def test_interval_array_makers_and_questions():
    made = fp.arrays.IntervalArray.from_tuples([(0, 1), (1, 3), (2, 4)])
    assert made.contains(2).tolist() == [False, True, False]
    assert made.overlaps(fp.Interval(0.5, 1.5)).tolist() == [True, True, False]
    breaks = fp.arrays.IntervalArray.from_breaks(range(4))
    assert breaks.left.tolist() == [0, 1, 2]
    assert breaks.is_non_overlapping_monotonic
    assert breaks.set_closed("both").closed == "both"
    assert fp.arrays.IntervalArray.from_arrays([0, 1], [1, 2], closed="left").closed == "left"
    assert fp.arrays.IntervalArray([fp.Interval(0, 1), fp.Interval(1, 5)]).closed == "right"


def test_interval_index_is_overlapping():
    assert fp.IntervalIndex.from_tuples([(0, 2), (1, 3), (4, 5)]).is_overlapping
    assert not fp.IntervalIndex.from_breaks([0, 1, 2]).is_overlapping
    assert fp.IntervalIndex.from_breaks([0, 1, 2], closed="both").is_overlapping
    assert fp.IntervalIndex.from_tuples([(0, 10), (1, 2), (3, 4)]).is_overlapping


def test_numpy_array_wraps_and_interpolates():
    arr = fp.arrays.NumpyExtensionArray(np.array([0, 1, np.nan, 3]))
    filled = arr.interpolate(
        method="linear",
        axis=0,
        index=fp.Index([1, 2, 3, 4]),
        limit=3,
        limit_direction="forward",
        limit_area="inside",
        copy=False,
    )
    assert filled.tolist() == [0.0, 1.0, 2.0, 3.0]
