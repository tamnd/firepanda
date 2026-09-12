"""Walking a frame and a column, and asking whether something is in one.

Almost every rule in this area is a rule nobody would choose and everybody has to
match, so nearly every test here asserts the pandas answer and the firepanda one
in the same line. A frame iterates its column names and a column iterates its
values. `in` on a frame asks about names and `in` on a column asks about labels,
which means it never looks at a single value. And neither class can be used as a
truth value at all.

The one thing not checked against pandas is what a missing row looks like coming
out of an iteration, because pandas has four missing values and this library has
one. The three tests that touch a gap assert this library's answer and say so.

What is being tested is a set of Python protocols rather than a set of methods,
which is why several of these look like they are testing Python. They are testing
what Python does to a class that leaves a protocol out, which for `__iter__` is
not nothing: Python falls back to calling `__getitem__` with 0, 1, 2 and so on,
and that fallback reads labels here, so it used to answer, raise or silently stop
early depending on what the labels were.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

DATA: dict[str, list[Any]] = {"a": [1, 2], "b": [3.5, 4.5], "c": ["x", "y"]}
"""The three column frame, one column per kind that behaves differently."""

LABELS = ["p", "q", "r"]
VALUES = [1, 2, 3]
"""A column whose labels are not its positions, which is what makes `in` visible."""


def labelled(firepanda: ModuleType) -> Any:
    """The column above, built through `set_index` because that is the door there is."""
    return firepanda.DataFrame({"k": LABELS, "v": VALUES}).set_index("k")["v"]


def theirs() -> Any:
    """The same column in pandas."""
    import pandas as pd

    return pd.Series(VALUES, index=LABELS, name="v")


# ---------------------------------------------------------------------------
# A column walks its values
# ---------------------------------------------------------------------------


@needs_pandas
def test_iterating_a_column_gives_the_values(firepanda: ModuleType) -> None:
    assert list(labelled(firepanda)) == list(theirs()) == VALUES


def test_iterating_a_column_is_tolist(firepanda: ModuleType) -> None:
    made = labelled(firepanda)
    assert list(made) == made.tolist()


@needs_pandas
@pytest.mark.parametrize("call", [sum, min, max, sorted, list, tuple, set])
def test_the_builtins_that_iterate_all_work(firepanda: ModuleType, call: Any) -> None:
    """None of these works on a class that leaves `__iter__` out."""
    assert call(labelled(firepanda)) == call(theirs())


@needs_pandas
def test_a_comprehension_over_a_column(firepanda: ModuleType) -> None:
    assert [x * 2 for x in labelled(firepanda)] == [x * 2 for x in theirs()]


def test_iterating_does_not_stop_early_on_a_default_index(firepanda: ModuleType) -> None:
    """The fallback this replaced walked labels, so it ended where the labels ran out."""
    assert list(firepanda.Series([10, 20, 30, 40])) == [10, 20, 30, 40]


def test_iterating_an_empty_column(firepanda: ModuleType) -> None:
    assert list(firepanda.Series([])) == []


def test_iterating_a_column_of_text(firepanda: ModuleType) -> None:
    assert list(firepanda.Series(["a", "b"])) == ["a", "b"]


def test_a_missing_row_comes_out_as_none(firepanda: ModuleType) -> None:
    """This library's answer and not pandas', which gives nan for both of these."""
    assert list(firepanda.Series([1.5, None])) == [1.5, None]
    assert list(firepanda.Series(["a", None])) == ["a", None]


@needs_pandas
def test_iterating_a_category_gives_the_categories(firepanda: ModuleType) -> None:
    import pandas as pd

    words = ["a", "b", "a"]
    made = firepanda.Series(words).astype("category")
    assert list(made) == list(pd.Series(words).astype("category")) == words


@needs_pandas
def test_reversed_walks_a_column_backwards(firepanda: ModuleType) -> None:
    """Neither library defines `__reversed__`, so both fall back the same way."""
    import pandas as pd

    assert list(reversed(firepanda.Series(VALUES))) == list(reversed(pd.Series(VALUES)))


# ---------------------------------------------------------------------------
# A frame walks its column names
# ---------------------------------------------------------------------------


@needs_pandas
def test_iterating_a_frame_gives_the_column_names(firepanda: ModuleType) -> None:
    import pandas as pd

    assert list(firepanda.DataFrame(DATA)) == list(pd.DataFrame(DATA)) == ["a", "b", "c"]


@needs_pandas
def test_the_loop_the_rule_was_chosen_for(firepanda: ModuleType) -> None:
    import pandas as pd

    mine = firepanda.DataFrame(DATA)
    other = pd.DataFrame(DATA)
    assert [mine[name].tolist() for name in mine] == [list(other[name]) for name in other]


def test_iterating_an_empty_frame(firepanda: ModuleType) -> None:
    assert list(firepanda.DataFrame({})) == []


def test_a_frame_and_a_column_iterate_differently(firepanda: ModuleType) -> None:
    """Stated as its own test because it reads like a bug and is not."""
    made = firepanda.DataFrame({"a": [1, 2]})
    assert list(made) == ["a"]
    assert list(made["a"]) == [1, 2]


# ---------------------------------------------------------------------------
# What `in` asks about
# ---------------------------------------------------------------------------


@needs_pandas
def test_in_on_a_frame_asks_about_names(firepanda: ModuleType) -> None:
    import pandas as pd

    mine = firepanda.DataFrame(DATA)
    other = pd.DataFrame(DATA)
    assert ("a" in mine) is ("a" in other) is True
    assert ("zz" in mine) is ("zz" in other) is False


@needs_pandas
@pytest.mark.parametrize("key", [1, 1.5, None, True, ("a",)])
def test_in_on_a_frame_answers_no_rather_than_raising(firepanda: ModuleType, key: Any) -> None:
    """A key of a kind a name could never be is a question with the answer no."""
    import pandas as pd

    assert (key in firepanda.DataFrame(DATA)) is (key in pd.DataFrame(DATA)) is False


@needs_pandas
def test_in_on_a_column_asks_about_labels(firepanda: ModuleType) -> None:
    mine = labelled(firepanda)
    other = theirs()
    assert ("p" in mine) is ("p" in other) is True
    assert ("zz" in mine) is ("zz" in other) is False


@needs_pandas
def test_in_on_a_column_never_looks_at_values(firepanda: ModuleType) -> None:
    """The rule that surprises everybody: the values are 1, 2 and 3."""
    assert (1 in labelled(firepanda)) is (1 in theirs()) is False


@needs_pandas
def test_in_on_a_column_with_positions_for_labels(firepanda: ModuleType) -> None:
    import pandas as pd

    made = firepanda.Series([10, 20, 30])
    other = pd.Series([10, 20, 30])
    assert (0 in made) is (0 in other) is True
    assert (10 in made) is (10 in other) is False


@needs_pandas
@pytest.mark.parametrize("key", [1.5, None, "zz", True])
def test_in_on_a_column_answers_no_rather_than_raising(firepanda: ModuleType, key: Any) -> None:
    assert (key in labelled(firepanda)) is (key in theirs()) is False


def test_the_value_question_is_isin(firepanda: ModuleType) -> None:
    """Named here because it is what a caller who wanted the other answer needs."""
    assert bool(labelled(firepanda).isin([1]).any()) is True


# ---------------------------------------------------------------------------
# Neither class is a truth value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("values", [[1, 2], [1], [], [False]])
def test_a_column_has_no_truth_value(firepanda: ModuleType, values: list[Any]) -> None:
    """A column of one row included, which is the case people expect to work."""
    with pytest.raises(ValueError, match="truth value of a Series is ambiguous"):
        bool(firepanda.Series(values))


@pytest.mark.parametrize("data", [{"a": [1, 2]}, {"a": []}, {}])
def test_a_frame_has_no_truth_value(firepanda: ModuleType, data: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="truth value of a DataFrame is ambiguous"):
        bool(firepanda.DataFrame(data))


@needs_pandas
@pytest.mark.parametrize("kind", ["Series", "DataFrame"])
def test_the_refusal_is_word_for_word_pandas(firepanda: ModuleType, kind: str) -> None:
    import pandas as pd

    built = [1] if kind == "Series" else {"a": [1]}
    with pytest.raises(ValueError) as mine:
        bool(getattr(firepanda, kind)(built))
    with pytest.raises(ValueError) as wanted:
        bool(getattr(pd, kind)(built))
    assert str(mine.value) == str(wanted.value)


def test_if_on_a_column_raises_rather_than_guessing(firepanda: ModuleType) -> None:
    """The expression the refusal exists for, written the way somebody would write it."""
    made = firepanda.Series([1, 2])
    with pytest.raises(ValueError):
        if made:
            pass


# ---------------------------------------------------------------------------
# keys and items
# ---------------------------------------------------------------------------


@needs_pandas
def test_frame_keys_are_the_column_names(firepanda: ModuleType) -> None:
    import pandas as pd

    assert firepanda.DataFrame(DATA).keys() == list(pd.DataFrame(DATA).keys()) == ["a", "b", "c"]


def test_frame_keys_are_columns(firepanda: ModuleType) -> None:
    made = firepanda.DataFrame(DATA)
    assert made.keys() == made.columns


@needs_pandas
def test_column_keys_are_the_labels(firepanda: ModuleType) -> None:
    made = labelled(firepanda).keys()
    assert isinstance(made, firepanda.Index)
    assert made.tolist() == list(theirs().keys()) == LABELS


def test_column_keys_are_the_index(firepanda: ModuleType) -> None:
    made = labelled(firepanda)
    assert made.keys().tolist() == made.index.tolist()


@needs_pandas
def test_frame_items_give_name_and_column(firepanda: ModuleType) -> None:
    import pandas as pd

    mine = list(firepanda.DataFrame(DATA).items())
    other = list(pd.DataFrame(DATA).items())
    assert [name for name, _ in mine] == [name for name, _ in other]
    assert [held.tolist() for _, held in mine] == [list(held) for _, held in other]


def test_a_column_from_items_keeps_its_name_and_labels(firepanda: ModuleType) -> None:
    made = firepanda.DataFrame({"k": ["p", "q"], "v": [1, 2]}).set_index("k")
    name, held = next(iter(made.items()))
    assert name == "v"
    assert held.name == "v"
    assert held.index.tolist() == ["p", "q"]


@needs_pandas
def test_column_items_give_label_and_value(firepanda: ModuleType) -> None:
    assert (
        list(labelled(firepanda).items())
        == list(theirs().items())
        == list(zip(LABELS, VALUES, strict=True))
    )


def test_items_on_an_empty_column(firepanda: ModuleType) -> None:
    assert list(firepanda.Series([]).items()) == []


def test_items_on_an_empty_frame(firepanda: ModuleType) -> None:
    assert list(firepanda.DataFrame({}).items()) == []


def test_frame_items_is_lazy(firepanda: ModuleType) -> None:
    """A generator and not a list, which is what pandas gives and what a wide frame needs."""
    assert not isinstance(firepanda.DataFrame(DATA).items(), list)


# ---------------------------------------------------------------------------
# itertuples
# ---------------------------------------------------------------------------


@needs_pandas
def test_itertuples_matches_pandas(firepanda: ModuleType) -> None:
    import pandas as pd

    assert [tuple(row) for row in firepanda.DataFrame(DATA).itertuples()] == [
        tuple(row) for row in pd.DataFrame(DATA).itertuples()
    ]


@needs_pandas
def test_itertuples_field_names(firepanda: ModuleType) -> None:
    import pandas as pd

    mine = next(iter(firepanda.DataFrame(DATA).itertuples()))
    other = next(iter(pd.DataFrame(DATA).itertuples()))
    assert mine._fields == other._fields == ("Index", "a", "b", "c")


def test_itertuples_type_is_called_pandas_by_default(firepanda: ModuleType) -> None:
    """Copied rather than renamed, because code matches on the type name."""
    made = next(iter(firepanda.DataFrame(DATA).itertuples()))
    assert type(made).__name__ == "Pandas"


def test_itertuples_reads_fields_by_name(firepanda: ModuleType) -> None:
    rows = list(firepanda.DataFrame(DATA).itertuples())
    assert [row.a for row in rows] == [1, 2]
    assert [row.c for row in rows] == ["x", "y"]
    assert [row.Index for row in rows] == [0, 1]


@needs_pandas
def test_itertuples_without_the_index(firepanda: ModuleType) -> None:
    import pandas as pd

    mine = list(firepanda.DataFrame(DATA).itertuples(index=False))
    assert [tuple(row) for row in mine] == [
        tuple(row) for row in pd.DataFrame(DATA).itertuples(index=False)
    ]
    assert mine[0]._fields == ("a", "b", "c")


def test_itertuples_with_a_name(firepanda: ModuleType) -> None:
    made = next(iter(firepanda.DataFrame(DATA).itertuples(name="Row")))
    assert type(made).__name__ == "Row"


@needs_pandas
def test_itertuples_with_no_name_gives_plain_tuples(firepanda: ModuleType) -> None:
    import pandas as pd

    rows = list(firepanda.DataFrame(DATA).itertuples(name=None))
    assert rows == list(pd.DataFrame(DATA).itertuples(name=None))
    assert type(rows[0]) is tuple


def test_itertuples_keeps_the_labels(firepanda: ModuleType) -> None:
    made = firepanda.DataFrame({"k": ["p", "q"], "v": [1, 2]}).set_index("k")
    assert [row.Index for row in made.itertuples()] == ["p", "q"]


@needs_pandas
def test_itertuples_renames_a_field_that_could_not_be_a_name(firepanda: ModuleType) -> None:
    """Neither library writes this rule, both hand it to `collections.namedtuple`."""
    import pandas as pd

    data: dict[str, list[Any]] = {"a b": [1], "1x": [2], "class": [3]}
    mine = next(iter(firepanda.DataFrame(data).itertuples()))
    other = next(iter(pd.DataFrame(data).itertuples()))
    assert mine._fields == other._fields == ("Index", "_1", "_2", "_3")


def test_itertuples_on_an_empty_frame(firepanda: ModuleType) -> None:
    assert list(firepanda.DataFrame({"a": []}).itertuples()) == []


@needs_pandas
def test_itertuples_refuses_a_name_that_is_not_one(firepanda: ModuleType) -> None:
    import pandas as pd

    with pytest.raises(ValueError) as mine:
        list(firepanda.DataFrame(DATA).itertuples(name=5))
    with pytest.raises(ValueError) as wanted:
        list(pd.DataFrame(DATA).itertuples(name=5))
    assert str(mine.value) == str(wanted.value)


def test_itertuples_is_lazy(firepanda: ModuleType) -> None:
    assert not isinstance(firepanda.DataFrame(DATA).itertuples(), list)


def test_itertuples_carries_a_missing_row_as_none(firepanda: ModuleType) -> None:
    """This library's answer, where pandas gives nan."""
    made = firepanda.DataFrame({"a": [1.5, None]})
    assert [row.a for row in made.itertuples()] == [1.5, None]
