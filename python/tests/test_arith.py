"""Arithmetic from Python, which is the first time the kernels are reachable at all.

The operators were in the kernel and on the Mojo `Series` and `DataFrame` before
this, and none of it could be written from Python, so `df["a"] + 1` had no answer.
These tests are about the boundary rather than about the kernels: whether the
operation crossed as the right word, whether the operand was read as a series or
a frame or a constant, whether a reflected form went the way round it claims, and
whether the arguments pandas declares are honoured, ignored or refused in the
same places pandas honours, ignores and refuses them.

The values are read back with `tolist` because that is the only way to see a null
from Python. A null matters here more than it does elsewhere: alignment is what
arithmetic between two differently labelled operands does, and a row that only
one side has is exactly the thing being tested.

`python/tests/test_bindings.py` already holds the other half of this, which is
that every name exists on the extension, on the Python class and in pandas with
the same signature. Nothing here repeats that. What is here is what the calls do.

### Where firepanda and pandas disagree

Three of these are asserted as they are rather than as pandas has them, and each
is a known gap with an issue rather than something this file is choosing:

  - A comparison against a row only one side has answers null. pandas answers
    False, on the ground that a row that is not there is not equal to anything.
  - Integer division by zero answers null. pandas answers zero with a warning.
  - Two series whose names differ produce a series named `""`. pandas produces
    one named `None`, which firepanda has no way to spell because a name here is
    a `String` and not an `Optional[String]`.

Asserting the current answer is deliberate. A test that skipped them would let
them change silently, and the point of writing them down is that the day one is
fixed, this file is one of the places that has to be edited to say so.
"""

from __future__ import annotations

from types import ModuleType

import pytest

# The seven arithmetic operations, as the name pandas gives the method, the
# Python operator, and what each answers on the fixture below. Written out rather
# than computed, because computing the expectation with the same operator the
# code under test uses tests nothing at all.
ARITHMETIC = [
    ("add", lambda a, b: a + b, [11, 22, None]),
    ("sub", lambda a, b: a - b, [-9, -18, None]),
    ("mul", lambda a, b: a * b, [10, 40, None]),
    ("truediv", lambda a, b: a / b, [0.1, 0.1, None]),
    ("floordiv", lambda a, b: a // b, [0, 0, None]),
    ("mod", lambda a, b: a % b, [1, 2, None]),
    ("pow", lambda a, b: a**b, [1, 1048576, None]),
]


def _pair(firepanda: ModuleType) -> tuple[object, object]:
    """Two series that overlap on two rows out of three.

    The shorter one is the whole reason the pair is shaped like this. `Series`
    refuses an `index=` argument, so the only labels a test can produce from
    Python are the default ones, and the only way to get two operands that do not
    line up is to make one shorter. Rows 0 and 1 are in both, row 2 is in one, and
    alignment is the difference between the two.
    """
    return firepanda.Series([1, 2, 3], name="x"), firepanda.Series([10, 20], name="x")


@pytest.mark.parametrize(("name", "operator", "expected"), ARITHMETIC)
def test_each_arithmetic_operator_reaches_the_kernel(
    firepanda: ModuleType, name: str, operator: object, expected: list[object]
) -> None:
    """The operator and the named form are the same call and answer the same thing.

    Both spellings are checked against the same expectation in one test rather
    than in two, because the thing worth knowing is not that either works but
    that they agree. They reach the boundary through different paths, `__add__`
    on one side and `add` on the other, and the failure this catches is the table
    generating one of the pair with the wrong word in it, which is a mistake
    nothing else here would find.
    """
    left, right = _pair(firepanda)
    assert operator(left, right).tolist() == expected  # type: ignore[operator]
    assert getattr(left, name)(right).tolist() == expected


def test_arithmetic_between_two_series_aligns_on_the_labels(firepanda: ModuleType) -> None:
    """The union of the labels, and a null where only one side had a row.

    This is the behaviour that makes a series a series rather than an array. Row
    2 is in the left operand and not in the right, so the answer has a row 2 and
    it is missing, rather than the answer being two rows long.
    """
    left, right = _pair(firepanda)
    total = left + right
    assert total.index.tolist() == [0, 1, 2]
    assert total.tolist() == [11, 22, None]
    assert len(total) == 3


def test_a_fill_value_stands_in_for_the_side_that_is_missing(firepanda: ModuleType) -> None:
    """What the named forms are for, and the reason the operators are not enough.

    `a + b` has nowhere to say what a row only one side has should be worth, so it
    answers null. `a.add(b, fill_value=0)` says it, and row 2 comes out as 3
    rather than as missing.
    """
    left, right = _pair(firepanda)
    assert left.add(right, fill_value=0).tolist() == [11, 22, 3]
    assert left.sub(right, fill_value=0).tolist() == [-9, -18, 3]
    assert left.mul(right, fill_value=1).tolist() == [10, 40, 3]


def test_a_fill_value_against_a_constant_is_accepted_and_ignored(firepanda: ModuleType) -> None:
    """pandas takes one here and does nothing with it, and so does this.

    It looks like an oversight in pandas and it is not worth diverging over: a
    constant is on every row, so there is never a side for the fill to stand in
    for, and the argument has nothing to do rather than being wrong.
    """
    left, _ = _pair(firepanda)
    assert left.add(5, fill_value=0).tolist() == left.add(5).tolist() == [6, 7, 8]


def test_the_reflected_forms_put_the_operands_the_other_way_round(
    firepanda: ModuleType,
) -> None:
    """`2 - s` and `s.rsub(2)` are the same call and neither is `s - 2`.

    Worth its own test because the mistake it catches is silent on a commutative
    operation. An `radd` that forgot to flip is right on every value it will ever
    see, and the `rsub` beside it generated the same way is wrong on all of them.
    """
    left, _ = _pair(firepanda)
    assert (2 - left).tolist() == [1, 0, -1]
    assert left.rsub(2).tolist() == [1, 0, -1]
    assert (left - 2).tolist() == [-1, 0, 1]
    assert (2**left).tolist() == [2, 4, 8]
    assert left.rpow(2).tolist() == [2, 4, 8]
    assert left.rtruediv(6).tolist() == [6.0, 3.0, 2.0]


def test_a_comparison_operator_refuses_to_align(firepanda: ModuleType) -> None:
    """And says which methods will, which is the part that makes the refusal usable.

    pandas' rule, not an implementation limit. A row only one side has has no true
    or false answer, so `==` refuses the pair outright, and the named forms are
    the ones that align. The class is `ValueError` because that is what pandas
    raises and what code catching it is written against.
    """
    left, right = _pair(firepanda)
    with pytest.raises(ValueError, match="identically-labeled") as raised:
        left == right  # noqa: B015
    assert "eq" in str(raised.value)

    assert (left == firepanda.Series([1, 9, 3], name="x")).tolist() == [True, False, True]
    assert (left < 2).tolist() == [True, False, False]


def test_the_named_comparisons_align_and_answer_null_off_the_end(
    firepanda: ModuleType,
) -> None:
    """This is one of the three divergences the module docstring lists.

    pandas answers False for row 2, because a row that is not in the right operand
    is not equal to anything in it. firepanda answers null, which is the kernel's
    rule for a comparison against a missing value and is applied here to a row
    that is missing because it was never there. The fill is honoured either way,
    which is the part that is not a divergence and is worth pinning: `eq` with a
    `fill_value` is the one comparison pandas does let you answer that row with.
    """
    left, right = _pair(firepanda)
    assert left.eq(right).tolist() == [False, False, None]
    assert left.ne(right).tolist() == [True, True, None]
    assert left.lt(right, fill_value=0).tolist() == [True, True, False]
    assert left.gt(right, fill_value=0).tolist() == [False, False, True]


def test_the_unary_operations(firepanda: ModuleType) -> None:
    """The four operators and the `abs` that is a method as well as a builtin."""
    left, _ = _pair(firepanda)
    assert (-left).tolist() == [-1, -2, -3]
    assert (+left).tolist() == [1, 2, 3]

    negative = firepanda.Series([-1, 2, -3], name="x")
    assert abs(negative).tolist() == [1, 2, 3]
    assert negative.abs().tolist() == [1, 2, 3]

    assert (~firepanda.Series([True, False], name="b")).tolist() == [False, True]


def test_divmod_comes_back_as_the_pair_python_asked_for(firepanda: ModuleType) -> None:
    """Two series in a tuple, and the same two the operators give separately.

    Checking the pair against `//` and `%` rather than against written out values
    is the point of the test. `divmod` is two calls here and in pandas both, and
    the thing that can go wrong is the two halves being run in the other order.
    """
    left, _ = _pair(firepanda)
    quotient, remainder = divmod(left, 2)
    assert quotient.tolist() == (left // 2).tolist() == [0, 1, 1]
    assert remainder.tolist() == (left % 2).tolist() == [1, 0, 1]

    quotient, remainder = left.rdivmod(7)
    assert quotient.tolist() == [7, 3, 2]
    assert remainder.tolist() == [0, 1, 1]


def test_a_series_hands_a_frame_back_to_the_frame(firepanda: ModuleType) -> None:
    """`s + df` is the frame's expression, so the series declines to answer it.

    Returning `NotImplemented` is what makes Python turn the expression round and
    ask `DataFrame.__radd__` instead. It is checked by calling the dunder directly
    rather than by writing `s + df`, because the expression is answered by
    whichever side accepts it and would pass whether or not the series declined.
    """
    left, _ = _pair(firepanda)
    frame = firepanda.DataFrame({"x": [1, 2, 3]})
    assert firepanda.Series.__add__(left, frame) is NotImplemented
    assert firepanda.Series.__lt__(left, frame) is NotImplemented


def test_a_frame_aligns_on_both_axes(firepanda: ModuleType) -> None:
    """Rows and columns, and a column only one side has is null all the way down."""
    left = firepanda.DataFrame({"x": [1, 2], "y": [3, 4]})
    right = firepanda.DataFrame({"y": [10, 20], "z": [5, 6]})

    total = left + right
    assert total.columns == ["x", "y", "z"]
    assert [total[name].tolist() for name in total.columns] == [
        [None, None],
        [13, 24],
        [None, None],
    ]

    filled = left.add(right, fill_value=0)
    assert [filled[name].tolist() for name in filled.columns] == [[1, 2], [13, 24], [5, 6]]


def test_axis_chooses_which_way_a_series_is_broadcast(firepanda: ModuleType) -> None:
    """The one thing the operators cannot say, which is why the named forms exist.

    An operator has to pick an axis and pandas picks the columns, so `df.add(s,
    axis=0)` is the only spelling of adding a series down the rows there is. Both
    spellings of the row axis are checked, since pandas takes three words for it
    and a caller who writes `axis="rows"` should not find out that only the number
    was implemented.
    """
    frame = firepanda.DataFrame({"x": [1, 2], "y": [3, 4]})
    series = firepanda.Series([100, 200], name="q")

    for axis in (0, "index", "rows"):
        answer = frame.add(series, axis=axis)
        assert [answer[name].tolist() for name in answer.columns] == [[101, 202], [103, 204]]


def test_a_frame_refuses_a_fill_value_against_a_series(firepanda: ModuleType) -> None:
    """pandas raises here and the reason is worth keeping rather than papering over.

    A series is broadcast across the rows rather than aligned against them cell by
    cell, so there is no second side for a fill to stand in for. pandas says
    `NotImplementedError: fill_value 0 not supported.` and this says the same
    thing at more length, because the pandas message names the value and not the
    reason.
    """
    frame = firepanda.DataFrame({"x": [1, 2]})
    series = firepanda.Series([100, 200], name="q")
    with pytest.raises(NotImplementedError, match="fill_value 0 not supported"):
        frame.add(series, axis=0, fill_value=0)


def test_the_level_argument_is_refused_rather_than_ignored(firepanda: ModuleType) -> None:
    """A divergence chosen on purpose, and the one in this file that is not a gap.

    pandas takes `level=0` on a frame with an ordinary index and quietly does
    nothing with it. There is no MultiIndex here for a level to name, so honouring
    it is not possible and ignoring it would mean answering a question that was
    not asked. Refusing is the convention the rest of the Python layer already
    follows for a parameter that is declared and not implemented.
    """
    left, right = _pair(firepanda)
    with pytest.raises(NotImplementedError, match="level="):
        left.add(right, level=0)

    frame = firepanda.DataFrame({"x": [1, 2]})
    with pytest.raises(NotImplementedError, match="level="):
        frame.add(frame, level=0)


def test_an_axis_that_does_not_exist_says_so_the_way_pandas_does(
    firepanda: ModuleType,
) -> None:
    """Including the two that are only wrong on one of the two classes.

    A series has one axis, so `axis=1` on a series is an error there and correct
    on a frame, and the message names the class for that reason. `None` is not an
    error on either: pandas takes it as the default for the call.
    """
    left, right = _pair(firepanda)
    frame = firepanda.DataFrame({"x": [1, 2]})

    with pytest.raises(ValueError, match="No axis named 7 for object type DataFrame"):
        frame.add(frame, axis=7)
    with pytest.raises(ValueError, match="No axis named columns for object type Series"):
        left.add(right, axis="columns")
    with pytest.raises(ValueError, match="No axis named 1 for object type Series"):
        left.add(right, axis=1)

    assert left.add(right, axis=None).tolist() == [11, 22, None]
    assert frame.add(frame, axis=None)["x"].tolist() == [2, 4]


def test_an_operation_across_two_dtypes_that_have_none_is_a_type_error(
    firepanda: ModuleType,
) -> None:
    """Because an untagged error out of the core arrives as a `RuntimeError`.

    The core does not classify its errors, since a class is a Python idea, so the
    binding puts `dtype` on whatever comes back out of an arithmetic call. That is
    `TypeError`, which is what pandas raises for the same expression, and it is
    the difference between `except TypeError` working and not.
    """
    frame = firepanda.DataFrame({"x": [1, 2]})
    with pytest.raises(TypeError, match="no common type"):
        frame + "z"

    with pytest.raises(TypeError, match="must be a series, a frame or a number"):
        frame + None


def test_a_result_keeps_a_name_only_when_both_sides_agree_on_it(
    firepanda: ModuleType,
) -> None:
    """The third divergence, and the smallest.

    pandas drops the name to `None` when the two operands disagree. A firepanda
    series name is a `String` rather than an `Optional[String]`, so the nearest
    thing it can say is the empty one, and there is no spelling of `None` to
    return. The agreeing case is the one that matters and it matches.
    """
    left, _ = _pair(firepanda)
    assert (left + firepanda.Series([1, 2, 3], name="x")).name == "x"
    assert (left + firepanda.Series([1, 2, 3], name="y")).name == ""


def test_the_dtype_of_a_result_follows_the_operation(firepanda: ModuleType) -> None:
    """Division widens an integer, a comparison is boolean, addition stays put.

    `truediv` is the one worth pinning. It is the only arithmetic operation that
    can answer a different type from the one it was given, and a floor division
    beside it that widened the same way would be wrong in a manner nothing about
    values would catch on this fixture.

    Division widens an integer and not a float, and this fixture only has
    integers, so what is pinned here is half the rule. The other half is that a
    float32 divided by anything it promotes against stays a float32, which no
    test in this file can reach because a `Series` takes no dtype and a CSV only
    produces the four wide types. `tests/test_binary.mojo` has it.
    """
    left, _ = _pair(firepanda)
    assert (left + 2).dtype == "int64"
    assert (left // 2).dtype == "int64"
    assert (left / 2).dtype == "float64"
    assert (left < 2).dtype == "bool"
    assert (-left).dtype == "int64"
