"""`&`, `|` and `^` between an integer and a boolean column work bit by bit, as in pandas."""

import firepanda as pd


def test_int_and_bool_bit_by_bit():
    answer = pd.Series([2, 1, 3]) & pd.Series([True, True, False])
    assert answer.dtype == "bool"
    assert answer.tolist() == [False, True, False]


def test_bool_or_int():
    answer = pd.Series([True, False, True]) | pd.Series([2, 0, 0])
    assert answer.tolist() == [True, False, True]


def test_bool_and_two_is_false():
    assert (pd.Series([True]) & pd.Series([2])).tolist() == [False]


def test_xor_and_frames():
    assert (pd.Series([2, 1]) ^ pd.Series([True, True])).tolist() == [True, False]
    frame = pd.DataFrame({"a": [2, 1]}) & pd.DataFrame({"a": [True, True]})
    assert frame["a"].tolist() == [False, True]
