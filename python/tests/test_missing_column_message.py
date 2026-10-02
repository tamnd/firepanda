"""A column the frame lacks is refused with pandas' KeyError message."""

import pytest

import firepanda as pd


def _frame():
    return pd.DataFrame({"a": [1, 2], 3: [1, 2]})


def test_one_missing_column():
    with pytest.raises(KeyError) as caught:
        _frame()["b"]
    assert str(caught.value) == "'b'"


def test_some_of_a_list_missing():
    with pytest.raises(KeyError, match=r"^\"\['b'\] not in index\"$"):
        _frame()[["a", "b"]]


def test_all_of_a_list_missing():
    with pytest.raises(KeyError, match=r"None of \[Index\(\['b', 'c'\]"):
        _frame()[["b", "c"]]


def test_sort_by_missing_column():
    with pytest.raises(KeyError) as caught:
        _frame().sort_values(4)
    assert str(caught.value) == "4"


def test_set_index_on_missing_column():
    with pytest.raises(KeyError, match=r"None of \['b'\] are in the columns"):
        _frame().set_index("b")
