"""numpy's keywords on a reduction, checked to say nothing as pandas checks them.

pandas takes `out`, `dtype`, `keepdims` and the rest so that numpy can call a
reduction with its own keywords, and refuses any of them that holds something
other than its default. A keyword numpy does not have is a TypeError.
"""

from __future__ import annotations

from types import ModuleType

import pytest

_NOT_TAKEN = r"^the '{}' parameter is not supported in the pandas implementation of {}\(\)$"
_UNKNOWN = r"^{}\(\) got an unexpected keyword argument '{}'$"


def _owners(firepanda: ModuleType) -> list[object]:
    frame = firepanda.DataFrame({"a": [1, 2], "b": [1.5, 2.5]})
    return [frame, frame["a"]]


@pytest.mark.parametrize(
    ("method", "word"),
    [
        ("sum", "sum"),
        ("prod", "prod"),
        ("product", "prod"),
        ("mean", "mean"),
        ("median", "median"),
        ("min", "min"),
        ("max", "max"),
        ("std", "std"),
        ("var", "var"),
        ("sem", "sem"),
        ("skew", "skew"),
        ("kurt", "kurt"),
        ("kurtosis", "kurt"),
        ("all", "all"),
        ("any", "any"),
        ("cumsum", "cumsum"),
        ("cumprod", "cumprod"),
        ("cummax", "cummax"),
        ("cummin", "cummin"),
    ],
)
def test_a_numpy_keyword_that_says_something(firepanda: ModuleType, method: str, word: str) -> None:
    """`out` must stay empty and an unknown keyword is refused by name."""
    for owner in _owners(firepanda):
        with pytest.raises(ValueError, match=_NOT_TAKEN.format("out", word)):
            getattr(owner, method)(out=1)
        with pytest.raises(TypeError, match=_UNKNOWN.format(word, "zz")):
            getattr(owner, method)(zz=1)
        getattr(owner, method)(out=None)


def test_each_reduction_has_its_own_list(firepanda: ModuleType) -> None:
    """`all` takes no dtype, the running folds no keepdims, and `std` no initial."""
    column = firepanda.Series([1, 2])
    with pytest.raises(TypeError, match=_UNKNOWN.format("all", "dtype")):
        column.all(dtype="f8")
    with pytest.raises(TypeError, match=_UNKNOWN.format("cumsum", "keepdims")):
        column.cumsum(keepdims=True)
    with pytest.raises(TypeError, match=_UNKNOWN.format("std", "initial")):
        column.std(initial=None)
    with pytest.raises(ValueError, match=_NOT_TAKEN.format("initial", "sum")):
        column.sum(initial=0)
    with pytest.raises(ValueError, match=_NOT_TAKEN.format("overwrite_input", "median")):
        column.median(overwrite_input=True)
    with pytest.raises(ValueError, match=_NOT_TAKEN.format("dtype", "mean")):
        column.mean(dtype="f8")


def test_a_default_is_compared_by_value(firepanda: ModuleType) -> None:
    """`keepdims=0` equals False, so it is let through as pandas lets it through."""
    column = firepanda.Series([1, 2])
    assert column.max(keepdims=0) == 2
    assert column.sum(keepdims=False, initial=None, dtype=None) == 3
    with pytest.raises(ValueError, match=_NOT_TAKEN.format("keepdims", "max")):
        column.max(keepdims=True)


def test_an_unknown_keyword_comes_before_a_value(firepanda: ModuleType) -> None:
    """pandas looks for a name it does not know first, then at the values."""
    column = firepanda.Series([1, 2])
    with pytest.raises(TypeError, match=_UNKNOWN.format("sum", "zz")):
        column.sum(out=1, zz=1)


def test_a_value_by_position_is_read_as_the_keyword_in_its_place(firepanda: ModuleType) -> None:
    """Past the named parameters, the next value is numpy's first keyword."""
    column = firepanda.Series([1, 2])
    with pytest.raises(ValueError, match=_NOT_TAKEN.format("dtype", "cumsum")):
        column.cumsum(0, True, 1)
    assert column.cumsum(0, True, None).tolist() == [1, 3]
    with pytest.raises(TypeError, match=r"^cumsum\(\) takes at most 3 arguments \(4 given\)$"):
        column.cumsum(0, True, None, None, None)
    with pytest.raises(TypeError, match=r"^cumsum\(\) got multiple values for keyword argument"):
        column.cumsum(0, True, None, dtype=None)


def test_idxmin_and_argmin_print_argmax(firepanda: ModuleType) -> None:
    """pandas checks all four of a series' with the validator it wrote for `argmax`."""
    column = firepanda.Series([1, 2])
    for method in ("idxmax", "idxmin", "argmax", "argmin"):
        with pytest.raises(ValueError, match=_NOT_TAKEN.format("out", "argmax")):
            getattr(column, method)(out=1)
        with pytest.raises(TypeError, match=_UNKNOWN.format("argmax", "zz")):
            getattr(column, method)(zz=1)
    assert column.argmax(0, True, None) == 1


def test_an_index_checks_numpy_keywords_too(firepanda: ModuleType) -> None:
    """An index's `all` takes a dtype, and its `argmin` prints its own name."""
    labels = firepanda.Index([1, 2])
    with pytest.raises(ValueError, match=_NOT_TAKEN.format("dtype", "all")):
        labels.all(dtype=1)
    with pytest.raises(TypeError, match=_UNKNOWN.format("any", "zz")):
        labels.any(zz=1)
    with pytest.raises(ValueError, match=_NOT_TAKEN.format("out", "argmin")):
        labels.argmin(out=1)
    with pytest.raises(ValueError, match=_NOT_TAKEN.format("keepdims", "max")):
        labels.max(keepdims=True)
    with pytest.raises(ValueError, match=_NOT_TAKEN.format("axis", "min")):
        labels.min(None, True, 1)
    with pytest.raises(ValueError, match=r"^`axis` must be fewer than the number of dimensions"):
        labels.max(axis=1)
    assert labels.min(out=None) == 1
