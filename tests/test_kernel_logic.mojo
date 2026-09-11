"""Tests for the three valued connectives.

The truth tables are written out row by row, all nine combinations of each pair,
because that is what the rule is and there is no shorter honest way to say it.
The rest check the two things the table cannot: that the answer is the same once
the column is long enough to be split across morsels and run through a vector
loop, and that a row the rule decides holds the value it decided on rather than
a leftover.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_raises

from firepanda.array.any import AnyArray
from firepanda.array.array import Array, from_list
from firepanda.dtype.logical import LogicalType
from firepanda.kernel.logic import (
    LogicOp,
    is_logic_name,
    logic_any,
    logic_op,
    logic_type,
    logical_and,
    logical_not,
    logical_or,
)

comptime TRUE = 1
"""A present true, in the little integer encoding the tests are written in."""

comptime FALSE = 0
"""A present false."""

comptime NULL = -1
"""A null."""


def build(values: List[Int]) -> Array[DType.bool]:
    """Returns a boolean column written as trues, falses and nulls.

    Args:
        values: One of `TRUE`, `FALSE` or `NULL` per row.

    Returns:
        The column.
    """
    var out = Array[DType.bool](len(values))
    for i in range(len(values)):
        if values[i] == NULL:
            out.set_null(i)
        else:
            out[i] = values[i] == TRUE
    return out^


def assert_column(col: Array[DType.bool], expected: List[Int]) raises:
    """Checks a column against trues, falses and nulls.

    A null is checked twice over, for being null and for holding a false
    underneath, because a null that holds a true is the shape that reads
    correctly everywhere until something sums the column.

    Args:
        col: The column.
        expected: One of `TRUE`, `FALSE` or `NULL` per row.

    Raises:
        Error: If the column differs.
    """
    assert_equal(len(col), len(expected))
    for i in range(len(expected)):
        if expected[i] == NULL:
            assert_false(col.is_valid(i))
            assert_false(Bool(col[i]))
        else:
            assert_equal(col.is_valid(i), True)
            assert_equal(Bool(col[i]), expected[i] == TRUE)


def test_and_over_the_whole_truth_table() raises:
    var a = build([TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, NULL, NULL, NULL])
    var b = build([TRUE, FALSE, NULL, TRUE, FALSE, NULL, TRUE, FALSE, NULL])
    assert_column(
        logical_and(a, b),
        [TRUE, FALSE, NULL, FALSE, FALSE, FALSE, NULL, FALSE, NULL],
    )


def test_or_over_the_whole_truth_table() raises:
    var a = build([TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, NULL, NULL, NULL])
    var b = build([TRUE, FALSE, NULL, TRUE, FALSE, NULL, TRUE, FALSE, NULL])
    assert_column(
        logical_or(a, b),
        [TRUE, TRUE, TRUE, TRUE, FALSE, NULL, TRUE, NULL, NULL],
    )


def test_not_leaves_a_null_a_null() raises:
    assert_column(logical_not(build([TRUE, FALSE, NULL])), [FALSE, TRUE, NULL])


def test_a_false_decides_an_and_and_a_true_decides_an_or() raises:
    # The two rows the rule is about, on their own, so that a failure here says
    # which half of it broke.
    var nulls = build([NULL, NULL])
    var mixed = build([FALSE, TRUE])
    assert_column(logical_and(nulls, mixed), [FALSE, NULL])
    assert_column(logical_or(nulls, mixed), [NULL, TRUE])


def test_a_decided_row_holds_the_value_it_was_decided_to() raises:
    # A null row of the input carries a zero under the null, so an and that
    # decides false would read correctly even if nothing were written. An or
    # that decides true would not, and this is that row.
    var left = build([NULL])
    var right = build([TRUE])
    var answer = logical_or(left, right)
    assert_equal(answer.is_valid(0), True)
    assert_equal(Bool(answer[0]), True)


def test_neither_side_present_stays_null() raises:
    var a = build([NULL, NULL, NULL])
    var b = build([NULL, NULL, NULL])
    assert_column(logical_and(a, b), [NULL, NULL, NULL])
    assert_column(logical_or(a, b), [NULL, NULL, NULL])
    assert_equal(logical_and(a, b).null_count(), 3)


def test_a_column_with_no_nulls_answers_with_no_nulls() raises:
    var a = from_list[DType.bool]([True, False, True, False])
    var b = from_list[DType.bool]([True, True, False, False])
    assert_column(logical_and(a, b), [TRUE, FALSE, FALSE, FALSE])
    assert_column(logical_or(a, b), [TRUE, TRUE, TRUE, FALSE])
    assert_equal(logical_or(a, b).null_count(), 0)


def test_a_long_column_agrees_with_the_table_row_by_row() raises:
    # Long enough to be more than one vector and more than one validity word,
    # and a length that is not a multiple of either, so the tail is exercised.
    comptime n = 1000
    var pattern = List[Int]()
    var other = List[Int]()
    for i in range(n):
        pattern.append(i % 3 - 1)
        other.append((i // 3) % 3 - 1)

    var a = build(pattern)
    var b = build(other)
    var conjunction = logical_and(a, b)
    var disjunction = logical_or(a, b)
    var negation = logical_not(a)

    for i in range(n):
        var x = pattern[i]
        var y = other[i]

        var expect_and = NULL
        if x == FALSE or y == FALSE:
            expect_and = FALSE
        elif x == TRUE and y == TRUE:
            expect_and = TRUE
        assert_column(conjunction.slice(i, i + 1), [expect_and])

        var expect_or = NULL
        if x == TRUE or y == TRUE:
            expect_or = TRUE
        elif x == FALSE and y == FALSE:
            expect_or = FALSE
        assert_column(disjunction.slice(i, i + 1), [expect_or])

        var expect_not = NULL
        if x != NULL:
            expect_not = FALSE if x == TRUE else TRUE
        assert_column(negation.slice(i, i + 1), [expect_not])


def test_two_columns_of_different_lengths_have_no_rows_in_common() raises:
    with assert_raises(contains="the right has 2"):
        _ = logical_and(build([TRUE, TRUE, TRUE]), build([TRUE, TRUE]))


def test_the_erased_form_carries_the_same_answers() raises:
    var a = AnyArray(build([TRUE, FALSE, NULL]))
    var b = AnyArray(build([NULL, NULL, NULL]))
    assert_column(
        logic_any(a, b, LogicOp.AND).as_typed[DType.bool](),
        [NULL, FALSE, NULL],
    )
    assert_column(
        logic_any(a, b, LogicOp.OR).as_typed[DType.bool](),
        [TRUE, NULL, NULL],
    )
    assert_column(
        logic_any(a, LogicOp.NOT).as_typed[DType.bool](), [FALSE, TRUE, NULL]
    )


def test_a_connective_over_a_column_that_is_not_boolean_is_refused() raises:
    var numbers = AnyArray(from_list[DType.int64]([1, 2, 3]))
    var flags = AnyArray(build([TRUE, FALSE, TRUE]))
    with assert_raises(contains="two boolean columns"):
        _ = logic_any(numbers, flags, LogicOp.AND)
    with assert_raises(contains="a boolean column"):
        _ = logic_any(numbers, LogicOp.NOT)


def test_a_connective_given_the_wrong_number_of_columns_is_refused() raises:
    var flags = AnyArray(build([TRUE]))
    with assert_raises(contains="reads one column and was given two"):
        _ = logic_any(flags, AnyArray(build([TRUE])), LogicOp.NOT)
    with assert_raises(contains="reads two columns and was given one"):
        _ = logic_any(flags, LogicOp.AND)


def test_the_answer_type_is_bool_and_the_operand_has_to_be() raises:
    assert_equal(logic_type(LogicOp.OR, LogicalType.BOOL), LogicalType.BOOL)
    with assert_raises(contains="reads a boolean column"):
        _ = logic_type(LogicOp.AND, LogicalType.INT64)


def test_the_three_names_map_to_the_three_connectives() raises:
    assert_equal(logic_op("and"), LogicOp.AND)
    assert_equal(logic_op("or"), LogicOp.OR)
    assert_equal(logic_op("not"), LogicOp.NOT)
    assert_equal(String(LogicOp.AND), "and")
    assert_equal(String(LogicOp.OR), "or")
    assert_equal(String(LogicOp.NOT), "not")
    assert_equal(is_logic_name("and"), True)
    assert_false(is_logic_name("coalesce"))
    with assert_raises(contains="is not a connective"):
        _ = logic_op("nand")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
