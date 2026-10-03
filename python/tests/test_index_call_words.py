"""Wrong calls on an index, and the numpy keywords of `take`, `transpose` and `to_numpy`.

Python words a wrong keyword with the function's qualified name, so each index
method carries the name the same pandas index prints: `Index.<name>` where pandas
defines it on the base class, the subclass or mixin where it is defined there,
and a bare name where one of pandas' validators does the refusing.
"""

from __future__ import annotations

from types import ModuleType

import pytest

_UNKNOWN = r"^{}\(\) got an unexpected keyword argument 'zz'$"
_NOT_TAKEN = r"^the '{}' parameter is not supported in the pandas implementation of {}\(\)$"
_TOO_MANY = r"^transpose\(\) takes at most 1 argument \(2 given\)$"


@pytest.mark.parametrize(
    ("build", "method", "shown"),
    [
        (lambda fp: fp.Index([1, 2]), "repeat", "Index.repeat"),
        (lambda fp: fp.Index([1, 2]), "searchsorted", "IndexOpsMixin.searchsorted"),
        (lambda fp: fp.Index([1, 2]), "tolist", "IndexOpsMixin.tolist"),
        (lambda fp: fp.Index([1, 2]), "isnull", "Index.isna"),
        (lambda fp: fp.RangeIndex(2), "searchsorted", "RangeIndex.searchsorted"),
        (lambda fp: fp.RangeIndex(2), "repeat", "Index.repeat"),
        (lambda fp: fp.RangeIndex(2), "copy", "RangeIndex.copy"),
        (lambda fp: fp.DatetimeIndex(["2024-01-01"]), "floor", "TimelikeOps.floor"),
        (lambda fp: fp.MultiIndex.from_tuples([(1, "a")]), "rename", "Index.set_names"),
    ],
)
def test_an_index_names_the_method_as_pandas_does(
    firepanda: ModuleType, build: object, method: str, shown: str
) -> None:
    """The name is the one the same kind of pandas index prints."""
    labels = build(firepanda)  # type: ignore[operator]
    with pytest.raises(TypeError, match=_UNKNOWN.format(shown)):
        getattr(labels, method)(zz=1)


def test_take_checks_numpy_keywords(firepanda: ModuleType) -> None:
    """`out` and `mode` must hold their defaults, and anything else is unknown."""
    frame = firepanda.DataFrame({"a": [1, 2], "b": [1.5, 2.5]})
    for owner in (frame, frame["a"], frame.index):
        with pytest.raises(TypeError, match=_UNKNOWN.format("take")):
            owner.take([0], zz=1)
        with pytest.raises(ValueError, match=_NOT_TAKEN.format("out", "take")):
            owner.take([0], out=1)
        assert len(owner.take([0], out=None, mode="raise")) == 1


def test_transpose_checks_its_axes(firepanda: ModuleType) -> None:
    """A value by position is `axes`, which must be None, and two are one too many."""
    frame = firepanda.DataFrame({"a": [1, 2], "b": [1.5, 2.5]})
    for owner in (frame, frame["a"], frame.index):
        with pytest.raises(ValueError, match=_NOT_TAKEN.format("axes", "transpose")):
            owner.transpose(1)
        with pytest.raises(TypeError, match=_TOO_MANY):
            owner.transpose(None, None)
        owner.transpose(None)
    with pytest.raises(TypeError, match=_UNKNOWN.format("transpose")):
        frame["a"].transpose(zz=1)
    assert frame.transpose(None).shape == (2, 2)


@pytest.mark.parametrize(
    ("dtype", "owner"),
    [
        ("int64", ""),
        ("Int64", "BaseMaskedArray."),
        ("boolean", "BaseMaskedArray."),
        ("str", "ArrowExtensionArray."),
        ("category", "ExtensionArray."),
    ],
)
def test_to_numpy_names_the_array_that_refuses(
    firepanda: ModuleType, dtype: str, owner: str
) -> None:
    """pandas hands an extension column's keywords to its array, which names itself."""
    values = [True] if dtype == "boolean" else ["a"] if dtype in ("str", "category") else [1]
    column = firepanda.Series(values, dtype=dtype)
    for holder in (column, firepanda.Index(column)):
        with pytest.raises(TypeError, match=_UNKNOWN.format(owner + "to_numpy")):
            holder.to_numpy(zz=1)


def test_pct_change_hands_the_rest_to_shift(firepanda: ModuleType) -> None:
    """`fill_value` fills the shifted rows, and an unknown keyword is shift's to refuse."""
    frame = firepanda.DataFrame({"a": [1, 2, 4], "b": [1.5, 3.0, 6.0]})
    assert frame.pct_change(fill_value=1)["a"].tolist() == [0.0, 1.0, 1.0]
    assert frame["b"].pct_change(fill_value=1).tolist() == [0.5, 1.0, 1.0]
    with pytest.raises(TypeError, match=r"shift\(\) got an unexpected keyword argument 'zz'"):
        frame.pct_change(zz=1)
    with pytest.raises(TypeError, match=r"shift\(\) got an unexpected keyword argument 'zz'"):
        frame["b"].pct_change(zz=1)


def test_a_multiindex_checks_numpy_keywords(firepanda: ModuleType) -> None:
    """The validators speak before the refusal of a reduction over tuples."""
    labels = firepanda.MultiIndex.from_tuples([(1, "a"), (2, "b")])
    with pytest.raises(ValueError, match=_NOT_TAKEN.format("out", "max")):
        labels.max(out=1)
    with pytest.raises(ValueError, match=_NOT_TAKEN.format("dtype", "all")):
        labels.all(dtype=1)
    with pytest.raises(TypeError, match=_UNKNOWN.format("to_numpy")):
        labels.to_numpy(zz=1)
    assert labels.max() == (2, "b")


def test_an_index_argsort_refuses_what_numpy_does_not_take(firepanda: ModuleType) -> None:
    """numpy's `argsort` is the one that refuses, under its own bare name."""
    labels = firepanda.Index([2, 1])
    with pytest.raises(TypeError, match=_UNKNOWN.format("argsort")):
        labels.argsort(zz=1)
    assert list(labels.argsort(kind="stable")) == [1, 0]
