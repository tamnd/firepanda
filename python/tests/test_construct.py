"""Building a frame out of Python values, checked against pandas doing the same.

Document 18 is the reasoning. The tests that matter most here are the ones that
build the same literal on both sides and compare, because the whole claim of this
file is that the first line of a pandas program means the same thing.

Where firepanda deliberately disagrees, the test says so and asserts the
firepanda answer rather than skipping. There are three: a column with nothing in
it, a mixture pandas resolves with an object column, and the difference between a
None and a NaN.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def test_a_dict_of_lists_is_a_frame(firepanda: ModuleType) -> None:
    """The first line of every pandas tutorial."""
    frame = firepanda.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert frame.shape == (3, 2)
    assert frame.columns == ["a", "b"]
    assert frame["a"].tolist() == [1, 2, 3]
    assert frame["b"].tolist() == ["x", "y", "z"]


def test_the_columns_come_out_in_the_order_they_went_in(firepanda: ModuleType) -> None:
    """Which is the mapping's order, and is why dictionaries being ordered matters."""
    frame = firepanda.DataFrame({"z": [1], "a": [2], "m": [3]})
    assert frame.columns == ["z", "a", "m"]


def test_each_type_is_inferred(firepanda: ModuleType) -> None:
    """One pass over the values decides one of four concrete types."""
    frame = firepanda.DataFrame(
        {
            "i": [1, 2],
            "f": [1.5, 2.5],
            "b": [True, False],
            "s": ["one", "two"],
            "mixed": [1, 2.5],
        }
    )
    assert frame["i"].dtype == "int64"
    assert frame["f"].dtype == "float64"
    assert frame["b"].dtype == "bool"
    assert frame["s"].dtype == "string"
    assert frame["mixed"].dtype == "float64"
    assert frame["mixed"].tolist() == [1.0, 2.5]


def test_a_none_is_a_hole_rather_than_a_value(firepanda: ModuleType) -> None:
    """The validity bitmap is what a None becomes, in every type that has one."""
    frame = firepanda.DataFrame(
        {"i": [1, None, 3], "f": [1.5, None, 3.5], "b": [True, None, False], "s": ["a", None, "c"]}
    )
    for name in ("i", "f", "b", "s"):
        assert frame[name].tolist()[1] is None
        assert frame[name].count() == 2
        assert frame[name].hasnans


def test_a_nan_is_a_value_and_a_none_is_not(firepanda: ModuleType) -> None:
    """The divergence document 18 section 3 is about, asserted rather than hidden.

    Two halves of one answer. The storage keeps them apart, because an Arrow
    column has a validity bitmap and a pandas float column does not, so a NaN
    stays a NaN with its bit set and a None is a cleared bit. Read the values
    back and they are different things.

    What counts them agrees with pandas. `null_count` on a float column counts
    the cleared bits and the NaNs, which is the behaviour issue #170 argues for
    and the core is careful about, so `count` and `hasnans` say what a pandas
    user expects. Keeping the two apart in memory and together in the reductions
    is the whole of the position.
    """
    frame = firepanda.DataFrame({"a": [1.0, float("nan"), None]})
    values = frame["a"].tolist()
    assert values[1] != values[1]
    assert values[2] is None
    assert frame["a"].count() == 1
    assert frame["a"].hasnans


def test_a_tuple_a_range_and_a_generator_all_work(firepanda: ModuleType) -> None:
    """Everything goes through `list` first, so there is one shape to handle."""
    frame = firepanda.DataFrame({"a": (1, 2, 3), "b": range(3), "c": (i * 2 for i in range(3))})
    assert frame["a"].tolist() == [1, 2, 3]
    assert frame["b"].tolist() == [0, 1, 2]
    assert frame["c"].tolist() == [0, 2, 4]


def test_a_column_of_nothing_is_float64_and_all_missing(firepanda: ModuleType) -> None:
    """The second deliberate divergence, and the one with no good answer.

    pandas says object and pyarrow says the Arrow null type. firepanda has no
    object column by design and no column that carries the null type at run time,
    so the choice is float64 or a refusal, and refusing to build a frame because
    one column is empty would be absurd.
    """
    holes = firepanda.DataFrame({"a": [None, None]})
    assert holes["a"].dtype == "float64"
    assert holes["a"].tolist() == [None, None]
    nothing = firepanda.DataFrame({"b": []})
    assert nothing["b"].dtype == "float64"
    assert len(nothing["b"]) == 0


def test_a_series_is_built_from_a_sequence(firepanda: ModuleType) -> None:
    """The other constructor, with the one parameter that is honoured."""
    series = firepanda.Series([1, 2, 3], name="qty")
    assert series.name == "qty"
    assert series.tolist() == [1, 2, 3]
    assert series.dtype == "int64"
    assert firepanda.Series(["a", "b"]).name == ""


def test_an_empty_constructor_is_empty(firepanda: ModuleType) -> None:
    """Which is what pandas does, and used to be a refusal here."""
    assert firepanda.DataFrame().shape == (0, 0)
    assert len(firepanda.Series()) == 0


def test_columns_of_different_lengths_are_refused(firepanda: ModuleType) -> None:
    """A frame is rectangular and there is nothing defensible to do with the short one."""
    with pytest.raises(ValueError, match="same length"):
        firepanda.DataFrame({"a": [1, 2, 3], "b": [1]})


def test_a_mixture_pandas_would_call_object_is_refused(firepanda: ModuleType) -> None:
    """The third divergence, and the one that is a refusal rather than an answer.

    `[True, 1]` is a list somebody made by accident far more often than one they
    meant, and there is no column type here that holds both without deciding
    which of the two they wanted.
    """
    with pytest.raises(TypeError, match="mixes"):
        firepanda.DataFrame({"a": [True, 1]})
    with pytest.raises(TypeError, match="mixes"):
        firepanda.DataFrame({"a": [1, "two"]})


def test_a_type_with_no_column_to_put_it_in_names_the_row(firepanda: ModuleType) -> None:
    """The message has to say where, because a long list is not searchable by eye."""
    with pytest.raises(TypeError) as caught:
        firepanda.DataFrame({"a": [1, 2, {"nested": True}]})
    assert "row 2" in str(caught.value)
    assert "dict" in str(caught.value)


def test_an_integer_too_large_for_int64_is_refused(firepanda: ModuleType) -> None:
    """pandas widens to object and keeps it, which is not available here."""
    with pytest.raises(ValueError, match="int64"):
        firepanda.DataFrame({"a": [1, 2**70]})


def test_a_bare_string_is_a_scalar_and_is_refused(firepanda: ModuleType) -> None:
    """Iterating it would give a column of characters, which is never the intent."""
    with pytest.raises(ValueError, match="broadcast"):
        firepanda.DataFrame({"a": "abc"})


def test_a_shape_that_is_not_written_says_which_one(firepanda: ModuleType) -> None:
    """Records and rows are real pandas inputs and are not read yet."""
    with pytest.raises(ValueError, match="mapping"):
        firepanda.DataFrame([{"a": 1}, {"a": 2}])


def test_the_declared_pandas_parameters_are_refused_by_name(firepanda: ModuleType) -> None:
    """Declared and not honoured, which is only honest if it says so.

    The signature is the pandas one in full so that the parity test compares five
    parameters rather than one. A parameter that is silently ignored would be
    worse than one that is missing, so each of them raises with its own name in
    the message.
    """
    for keyword in ("index", "columns", "dtype", "copy"):
        with pytest.raises(NotImplementedError, match=keyword):
            firepanda.DataFrame({"a": [1]}, **{keyword: object()})
    for keyword in ("index", "dtype", "copy"):
        with pytest.raises(NotImplementedError, match=keyword):
            firepanda.Series([1], **{keyword: object()})


@needs_pandas
def test_the_same_literal_means_the_same_thing_in_both(firepanda: ModuleType) -> None:
    """The claim reduced to a comparison, over every type inference handles."""
    import pandas as pd

    data = {
        "i": [1, 2, 3],
        "f": [1.5, 2.5, 3.5],
        "b": [True, False, True],
        "s": ["one", "two", "three"],
    }
    ours = firepanda.DataFrame(data)
    theirs = pd.DataFrame(data)
    assert ours.shape == theirs.shape
    assert ours.columns == list(theirs.columns)
    for name in data:
        assert ours[name].tolist() == theirs[name].tolist()
        # pandas 3.0 calls a text column `str` and firepanda calls it `string`,
        # which is the Arrow name and is what pyarrow and Polars both use.
        theirs_dtype = str(theirs[name].dtype)
        assert str(ours[name].dtype) == ("string" if theirs_dtype == "str" else theirs_dtype)
