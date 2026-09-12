"""A column with no name, which is not the same thing as a column called nothing.

pandas tells the two apart. `pd.Series([1]).name` is `None` and
`pd.Series([1], name="").name` is `""`, and the difference is visible well past
the attribute: `to_frame` calls the first column `0` and the second one `""`,
`reset_index` does the same, and an operation between two differently named
columns lands on the absence rather than on the empty string.

This library held a name as a Mojo `String`, which has no absent value, so the
empty string was doing both jobs and every unnamed column reported `""`. That
was the difference blocking four cases on the conformance board, and it was one
difference in one place rather than four about the reductions that turned it up.
The name is an `Optional[String]` now, the same as an index level name has been
since it was written, and these are the tests that say so.

What is not fixed here is that pandas names the column of an unnamed series with
the integer `0` and this names it with the string `"0"`. Column labels are
strings in this library and that is a wider difference than one attribute.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


@needs_pandas
def test_a_column_built_without_a_name_has_none(firepanda: ModuleType) -> None:
    """The plain case, and the one every other line in this file rests on."""
    import pandas as pd

    assert firepanda.Series([1.0]).name is pd.Series([1.0]).name is None


@needs_pandas
def test_a_column_can_be_called_the_empty_string(firepanda: ModuleType) -> None:
    """Which is a name, and is why the absence needed somewhere else to live."""
    import pandas as pd

    assert firepanda.Series([1.0], name="").name == pd.Series([1.0], name="").name == ""


@needs_pandas
def test_renaming_to_none_clears_and_renaming_to_empty_does_not(
    firepanda: ModuleType,
) -> None:
    """Two calls that used to land on the same state and no longer do."""
    import pandas as pd

    for module in (firepanda, pd):
        held = module.Series([1.0], name="x")
        assert held.rename(None).name is None
        assert held.rename("").name == ""


@needs_pandas
@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.head(1),
        lambda s: s.sort_values(),
        lambda s: s.abs(),
        lambda s: s.astype("int64"),
        lambda s: s.duplicated(),
        lambda s: s.nlargest(1),
        lambda s: s.drop_duplicates(),
    ],
)
def test_an_operation_on_an_unnamed_column_answers_an_unnamed_one(
    firepanda: ModuleType, call: object
) -> None:
    """Several of these go out to a frame and back, which is where a name is lost.

    A frame column is a schema field and a field has a name, so a column with no
    name becomes one called `""` on the way in. The absence cannot survive the
    trip and is put back by hand on the way out, and this is the test that says
    every one of these members remembered to do it.
    """
    import pandas as pd

    assert call(firepanda.Series([3.0, 1.0])).name is None  # type: ignore[operator]
    assert call(pd.Series([3.0, 1.0])).name is None  # type: ignore[operator]


@needs_pandas
def test_an_operation_on_a_column_called_nothing_keeps_that_name(
    firepanda: ModuleType,
) -> None:
    """The other half of the round trip, which is the one that would go wrong if
    the absence were put back by asking whether the name was empty."""
    import pandas as pd

    assert firepanda.Series([3.0, 1.0], name="").head(1).name == ""
    assert pd.Series([3.0, 1.0], name="").head(1).name == ""


@needs_pandas
def test_two_columns_that_disagree_on_a_name_answer_one_with_none(
    firepanda: ModuleType,
) -> None:
    """A column called price plus a column called tax is neither of those."""
    import pandas as pd

    for module in (firepanda, pd):
        left = module.Series([1.0], name="price")
        right = module.Series([2.0], name="tax")
        assert (left + right).name is None


@needs_pandas
@pytest.mark.parametrize("name", ["sum", "count", "nunique", "any", "min"])
def test_a_frame_reduction_answers_a_column_with_no_name(firepanda: ModuleType, name: str) -> None:
    """One value per column is about none of the columns, so it carries no name.

    The per column reductions are what the truth and product slice could not put
    on the conformance board, because the comparison checks the name and the name
    was wrong whatever the numbers did.
    """
    import pandas as pd

    frame = {"a": [1.0, 0.0], "b": [2.0, 3.0]}
    assert getattr(firepanda.DataFrame(frame), name)().name is None
    assert getattr(pd.DataFrame(frame), name)().name is None


@needs_pandas
def test_a_frame_reduction_still_labels_its_answer_by_column(
    firepanda: ModuleType,
) -> None:
    """Because losing the name must not have cost the labels, which is the one
    way the change above could have gone wrong quietly."""
    import pandas as pd

    frame = {"a": [1.0, 0.0], "b": [2.0, 3.0]}
    mine = firepanda.DataFrame(frame).sum()
    theirs = pd.DataFrame(frame).sum()
    assert mine.index.tolist() == theirs.index.tolist() == ["a", "b"]
    assert mine.tolist() == theirs.tolist()


@needs_pandas
def test_a_grouped_size_answers_a_column_with_no_name(firepanda: ModuleType) -> None:
    """`size` is called `size` in the frame it comes out of and nothing in pandas."""
    import pandas as pd

    frame = {"k": ["a", "a", "b"], "v": [1.0, 2.0, 3.0]}
    assert firepanda.DataFrame(frame).groupby("k").size().name is None
    assert pd.DataFrame(frame).groupby("k").size().name is None


@needs_pandas
def test_an_unnamed_column_goes_into_a_frame_as_the_column_zero(
    firepanda: ModuleType,
) -> None:
    """And a column called nothing goes in under that name instead.

    The label is the string `"0"` here and the integer `0` in pandas, which is
    the wider difference the module docstring names, so this compares what the
    labels read as rather than what they are.
    """
    import pandas as pd

    for module in (firepanda, pd):
        assert [str(c) for c in module.Series([1.0]).to_frame().columns] == ["0"]
        assert [str(c) for c in module.Series([1.0], name="").to_frame().columns] == [""]


@needs_pandas
def test_an_unnamed_index_gives_an_unnamed_column_and_the_reverse(
    firepanda: ModuleType,
) -> None:
    """The two optional names meet here and neither is read off the other."""
    import pandas as pd

    for module in (firepanda, pd):
        assert module.Index([1, 2]).to_series().name is None
        assert module.Index([1, 2], name="").to_series().name == ""
        assert module.Index(module.Series([1, 2])).name is None
        assert module.Index(module.Series([1, 2], name="")).name == ""
