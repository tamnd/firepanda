"""Functions named by a string that does not exist are refused in pandas' words."""

import pytest

import firepanda as pd


def test_transform_by_unknown_name():
    with pytest.raises(ValueError, match=r"^Transform function failed$"):
        pd.Series([1, 2]).transform("nope")
    with pytest.raises(ValueError, match=r"^Transform function failed$"):
        pd.DataFrame({"a": [1, 2]}).transform("nope")


def test_apply_by_unknown_name():
    with pytest.raises(
        AttributeError, match=r"^'nope' is not a valid function for 'Series' object$"
    ):
        pd.Series([1, 2]).apply("nope")


def test_apply_by_known_name():
    assert pd.Series([1, 2]).apply("sum") == 3


def test_concat_axis_is_read_as_a_frame_axis():
    series = pd.Series([1])
    with pytest.raises(ValueError, match=r"^No axis named 5 for object type DataFrame$"):
        pd.concat([series, series], axis=5)
