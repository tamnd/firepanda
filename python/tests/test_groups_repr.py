"""A groupby's `groups` prints its labels as lists over one key, as pandas prints it."""

import firepanda as pd


def _frame():
    return pd.DataFrame({"k": list("abab"), "v": [1, 2, 3, 4]}, index=list("wxyz"))


def test_one_key_prints_lists():
    groups = _frame().groupby("k").groups
    assert repr(groups) == "{'a': ['w', 'y'], 'b': ['x', 'z']}"
    assert isinstance(groups, dict)
    assert isinstance(groups["a"], pd.Index)


def test_several_keys_are_a_plain_dict():
    groups = _frame().groupby(["k", "v"]).groups
    assert type(groups) is dict
    assert groups[("a", 1)].tolist() == ["w"]


def test_long_group_is_cut():
    shown = repr(pd.DataFrame({"k": [1] * 120}).groupby("k").groups)
    assert shown.endswith("98, 99, ...]}")


def test_empty_frame():
    assert repr(pd.DataFrame({"k": []}).groupby("k").groups) == "{}"
