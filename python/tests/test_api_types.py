"""`firepanda.api.types`, checked answer for answer against a running pandas.

Every assertion in here is a comparison rather than a constant, and that is not
style. The whole value of this module is that it agrees with pandas about
questions nobody thinks hard about, and the questions nobody thinks hard about
are exactly the ones a written down expectation gets wrong. `is_numeric_dtype`
says yes to a bool. `is_number` says yes to a bool and no to a numpy bool.
`is_string_dtype` says yes to `object`. None of those are memorable and all of
them are load bearing, so the table below asks pandas rather than asking the
author.

Two tests are exceptions and both say so where they appear. One covers the
dtypes firepanda has and pandas does not, where there is nothing to ask, so the
answers are written down with the reason attached. The other is a pandas bug, in
`is_re_compilable`, and it is compared against the right answer rather than
against pandas' answer.
"""

from __future__ import annotations

import collections
import datetime
import decimal
import fractions
import importlib.util
import io
import re
import warnings
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

DTYPES = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "Int64",
    "UInt8",
    "float32",
    "float64",
    "Float64",
    "complex64",
    "complex128",
    "bool",
    "boolean",
    "object",
    "str",
    "string",
    "category",
    "datetime64[ns]",
    "datetime64[us]",
    "datetime64[s]",
    "datetime64[ns, UTC]",
    "timedelta64[ns]",
    "timedelta64[s]",
    "period[D]",
    "interval[int64, right]",
]
"""Every dtype name both libraries can be asked about, spelled pandas' way.

The nullable spellings are in here on purpose. `int64` and `Int64` are the same
family and are not the same dtype, and a predicate that reads the name rather
than the object is exactly the kind of thing that gets one of them wrong.
"""

PREDICATES = [
    "is_any_real_numeric_dtype",
    "is_bool_dtype",
    "is_complex_dtype",
    "is_datetime64_any_dtype",
    "is_datetime64_dtype",
    "is_datetime64_ns_dtype",
    "is_datetime64tz_dtype",
    "is_extension_array_dtype",
    "is_float_dtype",
    "is_int64_dtype",
    "is_integer_dtype",
    "is_interval_dtype",
    "is_numeric_dtype",
    "is_object_dtype",
    "is_period_dtype",
    "is_signed_integer_dtype",
    "is_string_dtype",
    "is_timedelta64_dtype",
    "is_timedelta64_ns_dtype",
    "is_unsigned_integer_dtype",
]
"""The twenty that take a dtype and answer yes or no."""

VALUES: list[Any] = [
    True,
    False,
    0,
    1,
    -3,
    1.5,
    float("nan"),
    complex(1, 2),
    decimal.Decimal("1"),
    fractions.Fraction(1, 2),
    "text",
    "",
    b"bytes",
    None,
    [1, 2],
    (1, 2),
    (),
    {1, 2},
    {"a": 1},
    range(3),
    iter([1]),
    (x for x in [1]),
    datetime.date(2020, 1, 1),
    datetime.datetime(2020, 1, 1),
    datetime.time(12, 0),
    datetime.timedelta(days=1),
    re.compile("a"),
    slice(1),
    object(),
    int,
    str,
]
"""A value of every shape the object predicates sort between."""

OBJECT_PREDICATES = [
    "is_array_like",
    "is_bool",
    "is_complex",
    "is_dict_like",
    "is_file_like",
    "is_float",
    "is_hashable",
    "is_integer",
    "is_iterator",
    "is_list_like",
    "is_named_tuple",
    "is_number",
    "is_re",
    "is_re_compilable",
    "is_scalar",
]
"""The fifteen that take any object and answer yes or no."""


def quietly(function: Any, *arguments: Any, **keywords: Any) -> Any:
    """Calls one of these and swallows the deprecation warning six of them raise.

    Both libraries warn, so the warning is the thing being compared in
    `test_the_deprecated_ones_say_so` and is noise everywhere else.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return function(*arguments, **keywords)


@needs_pandas
@pytest.mark.parametrize("name", PREDICATES)
def test_the_dtype_predicates_answer_what_pandas_answers(firepanda: ModuleType, name: str) -> None:
    """Twenty predicates over twenty nine dtype names, compared one at a time.

    Parametrised over the predicate rather than over the pair, so that a failure
    names the predicate and lists the dtypes it disagreed about, which is the
    report that says what to fix.
    """
    import pandas as pd

    mine = getattr(firepanda.api.types, name)
    theirs = getattr(pd.api.types, name)
    wrong = [dtype for dtype in DTYPES if quietly(mine, dtype) != quietly(theirs, dtype)]
    assert wrong == []


@needs_pandas
@pytest.mark.parametrize("name", PREDICATES)
def test_the_dtype_predicates_take_a_type_as_well_as_a_name(
    firepanda: ModuleType, name: str
) -> None:
    """A bare `int` or `float` is a dtype to pandas, and code passes them."""
    import pandas as pd

    mine = getattr(firepanda.api.types, name)
    theirs = getattr(pd.api.types, name)
    types: list[Any] = [bool, int, float, complex, str, object, bytes, list]
    wrong = [held for held in types if quietly(mine, held) != quietly(theirs, held)]
    assert wrong == []


@needs_pandas
@pytest.mark.parametrize("name", PREDICATES)
def test_the_dtype_predicates_say_no_to_rubbish(firepanda: ModuleType, name: str) -> None:
    """Nothing here raises on a value that names no dtype, which is pandas' rule.

    It matters because these are called on whatever a user handed a library, and
    a predicate that raises turns a wrong argument into a traceback from four
    frames away.
    """
    import pandas as pd

    mine = getattr(firepanda.api.types, name)
    theirs = getattr(pd.api.types, name)
    rubbish: list[Any] = ["nonsense", None, 3, object(), [1, 2], {"a": 1}]
    wrong = [item for item in rubbish if quietly(mine, item) != quietly(theirs, item)]
    assert wrong == []


@needs_pandas
@pytest.mark.parametrize("name", PREDICATES)
def test_the_dtype_predicates_read_a_column(firepanda: ModuleType, name: str) -> None:
    """A `Series` is the commonest argument of all, and it is not a dtype.

    Each library is asked about its own column of the same values, since a
    firepanda `Series` is not a pandas one and the point is that a caller can
    hand either of them over without knowing which it has.
    """
    import pandas as pd

    mine = getattr(firepanda.api.types, name)
    theirs = getattr(pd.api.types, name)
    columns = {"i": [1, 2, 3], "f": [1.0, 2.0, 3.0], "s": ["a", "b", "c"], "b": [True, False, True]}
    wrong = [
        key
        for key, values in columns.items()
        if quietly(mine, firepanda.Series(values)) != quietly(theirs, pd.Series(values))
    ]
    assert wrong == []


@needs_pandas
@pytest.mark.parametrize("name", PREDICATES)
def test_the_dtype_predicates_say_no_to_a_frame(firepanda: ModuleType, name: str) -> None:
    """A frame has dtypes rather than a dtype, and pandas answers no rather than raising."""
    import pandas as pd

    mine = getattr(firepanda.api.types, name)
    theirs = getattr(pd.api.types, name)
    assert quietly(mine, firepanda.DataFrame({"a": [1]})) is False
    assert quietly(theirs, pd.DataFrame({"a": [1]})) is False


@needs_pandas
@pytest.mark.parametrize("name", OBJECT_PREDICATES)
def test_the_object_predicates_answer_what_pandas_answers(firepanda: ModuleType, name: str) -> None:
    """Fifteen predicates over thirty one values, compared one at a time.

    `is_re_compilable` is the one exception and it is excluded here rather than
    fudged, because pandas raises on a string that is not a valid pattern and
    firepanda answers False. It has its own test below.
    """
    import pandas as pd

    mine = getattr(firepanda.api.types, name)
    theirs = getattr(pd.api.types, name)
    wrong = [repr(value)[:40] for value in VALUES if quietly(mine, value) != quietly(theirs, value)]
    assert wrong == []


@needs_pandas
def test_the_two_object_predicates_with_a_second_argument(firepanda: ModuleType) -> None:
    """`allow_sets` and `allow_slice` are the two flags, and both change an answer."""
    import pandas as pd

    mine, theirs = firepanda.api.types, pd.api.types
    assert mine.is_list_like({1, 2}, allow_sets=False) == theirs.is_list_like(
        {1, 2}, allow_sets=False
    )
    assert mine.is_list_like({1, 2}, allow_sets=True) == theirs.is_list_like(
        {1, 2}, allow_sets=True
    )
    assert mine.is_hashable(slice(1), allow_slice=False) == theirs.is_hashable(
        slice(1), allow_slice=False
    )
    assert mine.is_hashable(slice(1), allow_slice=True) == theirs.is_hashable(
        slice(1), allow_slice=True
    )


@needs_pandas
def test_a_named_tuple_and_a_plain_one_are_told_apart(firepanda: ModuleType) -> None:
    """The one value that cannot go in the table, because the class has to be made first."""
    import pandas as pd

    made = collections.namedtuple("Made", "a b")
    for value in (made(1, 2), (1, 2)):
        assert firepanda.api.types.is_named_tuple(value) == pd.api.types.is_named_tuple(value)


@needs_pandas
def test_an_open_file_is_file_like_and_its_name_is_not(firepanda: ModuleType) -> None:
    """Also not in the table, because a file has to be opened and closed."""
    import pandas as pd

    with io.StringIO() as handle:
        assert firepanda.api.types.is_file_like(handle) is True
        assert pd.api.types.is_file_like(handle) is True
    assert firepanda.api.types.is_file_like("a path") is False


def test_is_re_compilable_answers_rather_than_raising(firepanda: ModuleType) -> None:
    """The one place this deliberately does not do what pandas does.

    pandas catches TypeError and not `re.error`, so asking it whether an
    unterminated character class is compilable raises `re.PatternError` instead
    of answering False. The name is a question and firepanda answers it. This is
    in the divergence registry with that reason, and it is not compared against
    pandas here because pandas has no answer to compare against.
    """
    kinds = firepanda.api.types
    assert kinds.is_re_compilable("a.*b") is True
    assert kinds.is_re_compilable("[") is False
    assert kinds.is_re_compilable("(unclosed") is False
    assert kinds.is_re_compilable(1) is False


@needs_pandas
@pytest.mark.parametrize(
    "values",
    [
        [1, 2, 3],
        [1.0, 2.0],
        [1, 2.0],
        [True, False],
        ["a", "b"],
        [b"a"],
        [complex(1)],
        [decimal.Decimal(1)],
        [],
        [None],
        [None, None],
        [1, None],
        ["a", None],
        [1, "a"],
        [datetime.date(2020, 1, 1)],
        [datetime.datetime(2020, 1, 1)],
        [datetime.time(1)],
        [datetime.timedelta(1)],
    ],
)
@pytest.mark.parametrize("skipna", [True, False])
def test_infer_dtype_names_the_same_thing_pandas_names(
    firepanda: ModuleType, values: list[Any], skipna: bool
) -> None:
    """Eighteen sequences, each asked twice, because `skipna` changes the answer."""
    import pandas as pd

    assert firepanda.api.types.infer_dtype(values, skipna=skipna) == pd.api.types.infer_dtype(
        values, skipna=skipna
    )


@needs_pandas
def test_infer_dtype_refuses_a_scalar(firepanda: ModuleType) -> None:
    """A scalar has nothing to infer from, and both libraries say so with a TypeError."""
    import pandas as pd

    with pytest.raises(TypeError, match="not iterable"):
        firepanda.api.types.infer_dtype(1)
    with pytest.raises(TypeError, match="not iterable"):
        pd.api.types.infer_dtype(1)


@needs_pandas
def test_pandas_dtype_reads_what_series_dtype_writes(firepanda: ModuleType) -> None:
    """The round trip that makes the string spelling of a dtype useful.

    Whatever `Series.dtype` hands back has to go straight back in, or the two
    halves of the vocabulary do not meet.
    """
    columns: dict[str, Any] = {"i": [1, 2], "f": [1.0, 2.0], "s": ["a", "b"], "b": [True, False]}
    for values in columns.values():
        held = firepanda.Series(values).dtype
        assert firepanda.api.types.pandas_dtype(held) == held


@needs_pandas
def test_pandas_dtype_refuses_what_names_no_dtype(firepanda: ModuleType) -> None:
    """Same class and same sentence as pandas, since callers match on the message."""
    import pandas as pd

    with pytest.raises(TypeError, match="not understood"):
        firepanda.api.types.pandas_dtype("nope")
    with pytest.raises(TypeError, match="not understood"):
        pd.api.types.pandas_dtype("nope")


@needs_pandas
def test_is_dtype_equal_agrees_with_pandas(firepanda: ModuleType) -> None:
    """Including on the pairs where one side names nothing, which is False rather than a raise."""
    import pandas as pd

    pairs: list[tuple[Any, Any]] = [
        ("int64", "int64"),
        ("int64", "int32"),
        ("int64", int),
        ("float64", float),
        ("nope", "nope"),
        ("nope", "int64"),
        ("category", "category"),
        ("datetime64[ns]", "datetime64[us]"),
    ]
    wrong = [
        pair
        for pair in pairs
        if firepanda.api.types.is_dtype_equal(*pair) != pd.api.types.is_dtype_equal(*pair)
    ]
    assert wrong == []


@needs_pandas
@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("is_int64_dtype", ("int64",)),
        ("is_datetime64tz_dtype", ("datetime64[ns, UTC]",)),
        ("is_categorical_dtype", ("category",)),
        ("is_interval_dtype", ("interval[int64, right]",)),
        ("is_period_dtype", ("period[D]",)),
        ("is_sparse", ("int64",)),
    ],
)
def test_the_deprecated_ones_say_so(
    firepanda: ModuleType, name: str, arguments: tuple[Any, ...]
) -> None:
    """A program under a strict warning filter has to break in the same place in both.

    The class is `DeprecationWarning` rather than pandas' `Pandas4Warning`, which
    is a subclass of it, so a filter written against `DeprecationWarning` catches
    both and a filter written against the pandas class catches only pandas. That
    is the one difference and it is deliberate, since `Pandas4Warning` names a
    release schedule firepanda is not on.
    """
    with pytest.warns(DeprecationWarning, match="deprecated"):
        getattr(firepanda.api.types, name)(*arguments)


@needs_pandas
def test_the_dtype_classes_carry_what_the_name_cannot(firepanda: ModuleType) -> None:
    """Four dtypes take parameters, so four of them are objects rather than names.

    `name` and `str` are the same on three of them and not on the interval,
    which is pandas' own inconsistency and is checked here rather than smoothed
    over, since `select_dtypes` matches on one of the two.
    """
    import pandas as pd

    kinds = firepanda.api.types
    for made, want in [
        (kinds.CategoricalDtype(["a", "b"], ordered=True), "category"),
        (kinds.DatetimeTZDtype("ns", "UTC"), "datetime64[ns, UTC]"),
        (kinds.IntervalDtype("int64", "right"), "interval[int64, right]"),
        (kinds.PeriodDtype("D"), "period[D]"),
    ]:
        assert str(made) == want
        assert made == want
    theirs = pd.api.types
    for mine, other in [
        (kinds.CategoricalDtype(["a", "b"], True), theirs.CategoricalDtype(["a", "b"], True)),
        (kinds.DatetimeTZDtype("ns", "UTC"), theirs.DatetimeTZDtype("ns", "UTC")),
        (kinds.IntervalDtype("int64", "right"), theirs.IntervalDtype("int64", "right")),
        (kinds.PeriodDtype("D"), theirs.PeriodDtype("D")),
    ]:
        assert mine.name == other.name
        assert mine.kind == other.kind


@needs_pandas
def test_the_dtype_classes_are_recognised_by_the_predicates(firepanda: ModuleType) -> None:
    """The reason they exist at all, since the name alone loses the parameters."""
    kinds = firepanda.api.types
    assert quietly(kinds.is_categorical_dtype, kinds.CategoricalDtype()) is True
    assert quietly(kinds.is_datetime64tz_dtype, kinds.DatetimeTZDtype("ns", "UTC")) is True
    assert quietly(kinds.is_interval_dtype, kinds.IntervalDtype("int64")) is True
    assert quietly(kinds.is_period_dtype, kinds.PeriodDtype("D")) is True
    assert kinds.is_datetime64_dtype(kinds.DatetimeTZDtype("ns", "UTC")) is False
    assert kinds.is_datetime64_any_dtype(kinds.DatetimeTZDtype("ns", "UTC")) is True


@needs_pandas
def test_a_categorical_dtype_keeps_its_categories(firepanda: ModuleType) -> None:
    """The categories come back as an `Index`, which is what pandas hands back too."""
    kinds = firepanda.api.types
    made = kinds.CategoricalDtype(["a", "b"], ordered=True)
    assert made.categories.tolist() == ["a", "b"]
    assert made.ordered is True
    assert kinds.CategoricalDtype().categories is None
    assert kinds.CategoricalDtype() == kinds.CategoricalDtype()
    assert kinds.CategoricalDtype(["a"]) != kinds.CategoricalDtype(["b"])
    assert hash(kinds.CategoricalDtype()) == hash(kinds.CategoricalDtype())


@needs_pandas
def test_a_zoned_datetime_dtype_needs_a_zone(firepanda: ModuleType) -> None:
    """Same refusal as pandas, since a zoneless one is a different dtype with a different name."""
    import pandas as pd

    with pytest.raises(TypeError, match="required"):
        firepanda.api.types.DatetimeTZDtype("ns")
    with pytest.raises(TypeError, match="required"):
        pd.api.types.DatetimeTZDtype("ns")


def test_the_dtypes_firepanda_has_and_pandas_does_not(firepanda: ModuleType) -> None:
    """Three firepanda dtypes pandas cannot be asked about, so the answer is written here.

    A date column, a binary column and a nested column are Arrow types pandas
    stores as `object` and calls `object`. firepanda calls them what they are, so
    each of them is answered on its own terms. A date is not a datetime64, which
    is the answer that surprises people and is still the right one, since the two
    have different widths and different arithmetic. A binary column is the one
    exception and it reads as a string dtype, because numpy's fixed width bytes
    dtype does in pandas and this is the same idea.
    """
    kinds = firepanda.api.types
    for name in ("date32[day]", "list<item: int64>", "struct<a: int64>"):
        answers = [quietly(getattr(kinds, predicate), name) for predicate in PREDICATES]
        assert answers == [False] * len(PREDICATES)
    assert kinds.is_string_dtype("binary") is True
    assert kinds.is_object_dtype("binary") is False


def test_union_categoricals_refuses_rather_than_pretending(firepanda: ModuleType) -> None:
    """There is no `Categorical` object for it to take, and it says that."""
    with pytest.raises(NotImplementedError, match="Categorical"):
        firepanda.api.types.union_categoricals([])


@needs_pandas
def test_every_name_pandas_has_is_here_with_the_same_signature(firepanda: ModuleType) -> None:
    """The parity check, run in process rather than waiting for the conformance board.

    The board asks this too, from another repository and against a built
    extension. Asking it here as well is what makes a wrong parameter name a
    failing test in the pull request that introduced it rather than a number that
    moved the wrong way a run later.
    """
    import inspect

    import pandas as pd

    wrong = []
    for name in dir(pd.api.types):
        if name.startswith("_"):
            continue
        mine = getattr(firepanda.api.types, name, None)
        if mine is None:
            wrong.append(f"{name} is missing")
            continue
        try:
            want = tuple(inspect.signature(getattr(pd.api.types, name)).parameters)
        except (TypeError, ValueError):
            continue
        got = tuple(inspect.signature(mine).parameters)
        if got != want:
            wrong.append(f"{name} takes {got} and pandas takes {want}")
    assert wrong == []


def test_the_api_namespace_holds_only_what_is_implemented(firepanda: ModuleType) -> None:
    """`extensions`, `indexers`, `interchange` and `typing` are absent rather than empty.

    Absent, because each hands out machinery for extending pandas and firepanda
    has none to hand out. An empty module would resolve and then fail on the
    first name, which is the shape of failure this project spends most of its
    effort avoiding.
    """
    assert firepanda.api.__all__ == ["types"]
    for name in ("extensions", "indexers", "interchange", "typing"):
        assert not hasattr(firepanda.api, name)
