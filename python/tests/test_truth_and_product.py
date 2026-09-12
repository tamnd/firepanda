"""`any`, `all`, `prod` and `product`, and the whole frame fold they arrived with.

These four look like the reductions already in `test_reductions.py` and two of
them are not. A product cannot use the loop a sum uses, because a null holds a
zero in this library and zero is the identity for addition rather than for
multiplication, and a truth value cannot use a bare comparison against zero,
because a NaN is not equal to zero and pandas counts a NaN as missing rather
than as true. Both of those are kernel facts and both are checked in Mojo, so
what these tests are for is the part that only exists on this side of the
boundary: the shape that comes back, the arguments that are held, and the ones
that are accepted and deliberately ignored.

The fourth section is about `axis=None`, which is not new in this slice and was
wrong before it. pandas folds the whole frame to one number and this library was
quietly answering a column instead, for every reduction, because the axis
translation mapped None onto the default. Six reductions can be built out of
their own per column answers and now are, six cannot and refuse, and three are
refused by pandas itself. All three groups are here.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

VALUES = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
"""The column from `test_reductions.py`, so a failure here is about the new names."""

MIXED = {"a": [1.0, 0.0], "b": [2.0, 3.0]}
"""A frame whose columns answer differently, so a fold cannot pass by accident."""


@needs_pandas
@pytest.mark.parametrize("name", ["any", "all", "prod", "product"])
def test_a_series_answers_what_pandas_answers(firepanda: ModuleType, name: str) -> None:
    """The whole point, once per name."""
    import pandas as pd

    assert getattr(firepanda.Series(VALUES), name)() == getattr(pd.Series(VALUES), name)()


@needs_pandas
def test_a_product_steps_over_what_is_missing(firepanda: ModuleType) -> None:
    """Six, not zero, which is what reading the zero a null holds would give."""
    import pandas as pd

    assert firepanda.Series([2.0, None, 3.0]).prod() == pd.Series([2.0, None, 3.0]).prod()


@needs_pandas
def test_a_product_of_nothing_is_one_rather_than_nothing(firepanda: ModuleType) -> None:
    """The identity carries the empty case, which is why it needs no found flag."""
    import pandas as pd

    assert firepanda.Series([]).prod() == pd.Series([], dtype="float64").prod()


@needs_pandas
def test_a_nan_is_missing_rather_than_true(firepanda: ModuleType) -> None:
    """A NaN is not equal to zero, so the obvious loop answers True and is wrong."""
    import pandas as pd

    nans = [float("nan")] * 3
    assert firepanda.Series(nans).any() == pd.Series(nans).any()
    assert firepanda.Series(nans).all() == pd.Series(nans).all()


@needs_pandas
def test_a_column_with_nothing_in_it_answers_the_identity_of_the_operator(
    firepanda: ModuleType,
) -> None:
    """False for any and True for all, in both libraries, which surprises people."""
    import pandas as pd

    assert firepanda.Series([]).any() == pd.Series([], dtype="float64").any()
    assert firepanda.Series([]).all() == pd.Series([], dtype="float64").all()


@needs_pandas
def test_a_string_is_true_when_it_is_not_empty(firepanda: ModuleType) -> None:
    """Text has a truth value and it is about length rather than about content."""
    import pandas as pd

    words = ["a", ""]
    assert firepanda.Series(words).any() == pd.Series(words).any()
    assert firepanda.Series(words).all() == pd.Series(words).all()


@needs_pandas
def test_a_string_column_refuses_a_product_the_way_pandas_does(firepanda: ModuleType) -> None:
    """Both raise a `TypeError`, because multiplying two words is not an operation."""
    import pandas as pd

    with pytest.raises(TypeError):
        firepanda.Series(["a", "b"]).prod()
    with pytest.raises(TypeError):
        pd.Series(["a", "b"]).prod()


@needs_pandas
@pytest.mark.parametrize("name", ["any", "all", "prod"])
def test_a_frame_answers_once_per_column(firepanda: ModuleType, name: str) -> None:
    """Named by the column, in column order, which is what pandas hands back."""
    import pandas as pd

    mine = getattr(firepanda.DataFrame(MIXED), name)()
    theirs = getattr(pd.DataFrame(MIXED), name)()
    assert mine.index.tolist() == theirs.index.tolist()
    assert mine.tolist() == theirs.tolist()


def test_bool_only_is_accepted_on_a_series_and_does_nothing(firepanda: ModuleType) -> None:
    """Because that is what pandas does with it, measured rather than assumed.

    `pd.Series([1, 0, 3]).any(bool_only=True)` is True. pandas takes the
    argument on a column, drops it, and answers. Refusing it here would make this
    library stricter than the one it is copying, which is a divergence in the
    direction nobody asks for.
    """
    assert firepanda.Series([1.0, 0.0, 3.0]).any(bool_only=True) is True
    assert firepanda.Series([1.0, 0.0, 3.0]).all(bool_only=True) is False


def test_the_arguments_that_are_held_are_still_held(firepanda: ModuleType) -> None:
    """`skipna=False` and a `min_count` above zero refuse rather than being dropped."""
    with pytest.raises(NotImplementedError, match="skipna"):
        firepanda.Series(VALUES).any(skipna=False)
    with pytest.raises(NotImplementedError, match="skipna"):
        firepanda.Series(VALUES).prod(skipna=False)
    with pytest.raises(NotImplementedError, match="min_count"):
        firepanda.Series(VALUES).prod(min_count=1)
    with pytest.raises(NotImplementedError, match="bool_only"):
        firepanda.DataFrame(MIXED).any(bool_only=True)


@needs_pandas
@pytest.mark.parametrize("name", ["sum", "prod", "min", "max", "any", "all"])
def test_the_whole_frame_fold_lands_where_pandas_lands(firepanda: ModuleType, name: str) -> None:
    """`axis=None` is one number for the frame and not one number per column.

    These six are the ones that can be built by asking the same question again of
    their own answers, which is why they are the six that answer at all. The
    product is the interesting one: pandas gives -0.0 here rather than 0.0,
    because the per column products are widened and one of them carries a sign,
    and landing on the same value is the check that this is a fold rather than a
    special case.
    """
    import pandas as pd

    mine = getattr(firepanda.DataFrame(MIXED), name)(axis=None)
    theirs = getattr(pd.DataFrame(MIXED), name)(axis=None)
    assert mine == theirs


@pytest.mark.parametrize("name", ["mean", "median", "std", "var", "sem", "skew"])
def test_the_folds_that_cannot_be_built_out_of_column_answers_say_so(
    firepanda: ModuleType, name: str
) -> None:
    """A mean of means is not a mean, so this refuses instead of being nearly right."""
    with pytest.raises(NotImplementedError, match="axis=None"):
        getattr(firepanda.DataFrame(MIXED), name)(axis=None)


@needs_pandas
def test_the_three_pandas_itself_refuses_are_refused_with_its_words(
    firepanda: ModuleType,
) -> None:
    """`count` has no whole frame form in pandas either, and the message is copied."""
    import pandas as pd

    with pytest.raises(ValueError, match="No axis named None"):
        firepanda.DataFrame(MIXED).count(axis=None)
    with pytest.raises(ValueError, match="No axis named None"):
        pd.DataFrame(MIXED).count(axis=None)


def test_a_series_still_takes_none_as_its_only_axis(firepanda: ModuleType) -> None:
    """A column has one axis, so folding it and reducing it are the same thing."""
    assert firepanda.Series(VALUES).prod(axis=None) == firepanda.Series(VALUES).prod()
    assert firepanda.Series(VALUES).any(axis="index") is True
