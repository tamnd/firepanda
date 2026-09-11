"""What an arithmetic operator gives back, and why it is not what joins say.

Every case here was checked against DuckDB 1.5, and the module as a whole was
checked against it over all 5,046 combinations of six operators and twenty
nine types before this file was written. What is asserted here is the part a
reader needs to see stated: the cases where a reasonable rule gives a wrong
answer, and the pairs where two operators disagree about the same overflow.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.sql.arith import (
    OP_ADD,
    OP_DIVIDE,
    OP_INT_DIVIDE,
    OP_MODULO,
    OP_MULTIPLY,
    OP_SUBTRACT,
    PROMOTION_WIDTH,
    arithmetic_type,
    negation_type,
    no_such_operator,
    no_such_unary_operator,
    operator_name,
    promote,
    promotions,
)
from firepanda.sql.cast import common_type
from firepanda.sql.types import (
    TYPE_COUNT,
    TYPE_INVALID,
    BIGINT,
    DATE,
    DOUBLE,
    INTEGER,
    INTERVAL,
    NULL,
    SqlType,
    TIMESTAMP,
    TINYINT,
    VARCHAR,
    decimal,
    parse_type,
)


def _result(operator: UInt8, a: StringSlice, b: StringSlice) raises -> String:
    """What an operator gives back for two types written out.

    Args:
        operator: One of the `OP_` constants.
        a: The left operand's type as a query would write it.
        b: The right operand's type.

    Returns:
        The result type's name, or `INVALID` when nothing takes that pair.
    """
    return arithmetic_type(operator, parse_type(a), parse_type(b)).name()


def test_an_operator_is_not_the_lattice() raises:
    # The reason this module exists rather than reusing cast.mojo. Two types
    # that agree on a SMALLINT when a query puts them side by side widen to a
    # BIGINT when an operator runs on them, because DuckDB binds an operator
    # by picking an overload and there is no signed and unsigned overload.
    assert_equal(
        common_type(TINYINT, parse_type("UTINYINT")).name(), "SMALLINT"
    )
    assert_equal(_result(OP_ADD, "TINYINT", "UTINYINT"), "BIGINT")
    assert_equal(_result(OP_ADD, "INTEGER", "UINTEGER"), "BIGINT")
    assert_equal(_result(OP_ADD, "BIGINT", "UBIGINT"), "HUGEINT")
    assert_equal(_result(OP_ADD, "HUGEINT", "UHUGEINT"), "DOUBLE")


def test_two_integers_of_the_same_sign_widen_to_the_wider_one() raises:
    assert_equal(_result(OP_ADD, "TINYINT", "SMALLINT"), "SMALLINT")
    assert_equal(_result(OP_ADD, "INTEGER", "INTEGER"), "INTEGER")
    assert_equal(_result(OP_SUBTRACT, "INTEGER", "BIGINT"), "BIGINT")
    assert_equal(_result(OP_MULTIPLY, "UTINYINT", "UBIGINT"), "UBIGINT")


def test_the_same_table_serves_five_of_the_six_operators() raises:
    # Checked over every pair rather than assumed, which is worth a test
    # because a table that quietly stopped applying to one of them would be
    # invisible until a corpus query used it.
    for operator in [
        OP_ADD,
        OP_SUBTRACT,
        OP_MULTIPLY,
        OP_INT_DIVIDE,
        OP_MODULO,
    ]:
        assert_equal(_result(operator, "TINYINT", "UTINYINT"), "BIGINT")
        assert_equal(_result(operator, "SMALLINT", "INTEGER"), "INTEGER")


def test_dividing_two_integers_gives_a_double() raises:
    # The difference most likely to go unnoticed, because the query still runs
    # and the number is still a number. 1 / 2 is 0.5 here and 0 in Postgres.
    assert_equal(_result(OP_DIVIDE, "INTEGER", "INTEGER"), "DOUBLE")
    assert_equal(_result(OP_DIVIDE, "HUGEINT", "TINYINT"), "DOUBLE")
    assert_equal(_result(OP_DIVIDE, "FLOAT", "INTEGER"), "FLOAT")
    assert_equal(_result(OP_DIVIDE, "FLOAT", "DOUBLE"), "DOUBLE")
    assert_equal(_result(OP_DIVIDE, "DECIMAL(4,2)", "DECIMAL(6,3)"), "DOUBLE")
    # Which is what // is for, and it keeps the integer type.
    assert_equal(_result(OP_INT_DIVIDE, "INTEGER", "INTEGER"), "INTEGER")
    assert_equal(_result(OP_INT_DIVIDE, "TINYINT", "SMALLINT"), "SMALLINT")


def test_adding_two_decimals_leaves_room_for_the_carry() raises:
    assert_equal(
        _result(OP_ADD, "DECIMAL(4,2)", "DECIMAL(6,3)"), "DECIMAL(7,3)"
    )
    assert_equal(
        _result(OP_SUBTRACT, "DECIMAL(4,2)", "DECIMAL(6,3)"), "DECIMAL(7,3)"
    )
    assert_equal(
        _result(OP_ADD, "DECIMAL(30,2)", "DECIMAL(30,2)"), "DECIMAL(31,2)"
    )


def test_multiplying_two_decimals_adds_both_widths_and_both_scales() raises:
    assert_equal(
        _result(OP_MULTIPLY, "DECIMAL(4,2)", "DECIMAL(6,3)"), "DECIMAL(10,5)"
    )
    assert_equal(
        _result(OP_MULTIPLY, "DECIMAL(4,2)", "INTEGER"), "DECIMAL(14,2)"
    )


def test_an_integer_in_a_decimal_derivation_counts_its_digits() raises:
    assert_equal(_result(OP_ADD, "DECIMAL(4,2)", "INTEGER"), "DECIMAL(13,2)")
    assert_equal(_result(OP_ADD, "DECIMAL(4,2)", "TINYINT"), "DECIMAL(6,2)")
    # Twenty digits and not nineteen, which is the one place an unsigned type
    # needs a wider decimal than its signed twin.
    assert_equal(_result(OP_ADD, "DECIMAL(4,2)", "UBIGINT"), "DECIMAL(23,2)")


def test_a_decimal_too_wide_to_exist_quietly_loses_the_digit() raises:
    # Saturating rather than raising, so the carry that the derivation just
    # asked for is not there and nothing says so.
    assert_equal(
        _result(OP_ADD, "DECIMAL(38,2)", "DECIMAL(38,2)"), "DECIMAL(38,2)"
    )
    assert_equal(
        _result(OP_MULTIPLY, "DECIMAL(20,10)", "DECIMAL(20,10)"),
        "DECIMAL(38,20)",
    )


def test_modulo_and_addition_disagree_about_the_same_overflow() raises:
    # Modulo cannot grow, so it takes the join. What it does not do is
    # saturate: where + narrows the answer and carries on, % gives up on being
    # exact and hands back a DOUBLE. Two operators, two answers, both DuckDB's.
    assert_equal(
        _result(OP_MODULO, "DECIMAL(4,2)", "DECIMAL(6,3)"), "DECIMAL(6,3)"
    )
    assert_equal(
        _result(OP_MODULO, "DECIMAL(4,2)", "DECIMAL(38,2)"), "DECIMAL(38,2)"
    )
    assert_equal(_result(OP_MODULO, "DECIMAL(6,3)", "DECIMAL(38,2)"), "DOUBLE")
    assert_equal(_result(OP_MODULO, "HUGEINT", "DECIMAL(4,2)"), "DOUBLE")
    assert_equal(
        _result(OP_ADD, "DECIMAL(6,3)", "DECIMAL(38,2)"), "DECIMAL(38,3)"
    )


def test_a_decimal_meeting_a_binary_float_gives_up_being_exact() raises:
    assert_equal(_result(OP_ADD, "DECIMAL(4,2)", "DOUBLE"), "DOUBLE")
    assert_equal(_result(OP_MULTIPLY, "DECIMAL(4,2)", "FLOAT"), "FLOAT")
    assert_equal(_result(OP_INT_DIVIDE, "DECIMAL(4,2)", "INTEGER"), "DOUBLE")


def test_a_date_plus_days_is_a_date_until_the_count_gets_big() raises:
    # DATE + INTEGER is a date and DATE + BIGINT is an error, so the line is
    # what fits in an INTEGER rather than what is an integer.
    assert_equal(_result(OP_ADD, "DATE", "INTEGER"), "DATE")
    assert_equal(_result(OP_ADD, "DATE", "USMALLINT"), "DATE")
    assert_equal(_result(OP_SUBTRACT, "DATE", "INTEGER"), "DATE")
    assert_equal(_result(OP_ADD, "DATE", "BIGINT"), "INVALID")
    assert_equal(_result(OP_ADD, "DATE", "UINTEGER"), "INVALID")


def test_a_date_plus_an_interval_stops_being_a_date() raises:
    # An interval carries microseconds, so there is nowhere for them to go on
    # a date and the answer widens to a timestamp.
    assert_equal(_result(OP_ADD, "DATE", "INTERVAL"), "TIMESTAMP")
    assert_equal(_result(OP_ADD, "INTERVAL", "DATE"), "TIMESTAMP")
    assert_equal(_result(OP_SUBTRACT, "DATE", "INTERVAL"), "TIMESTAMP")
    assert_equal(_result(OP_ADD, "TIME", "INTERVAL"), "TIME")
    assert_equal(_result(OP_ADD, "TIMESTAMP", "INTERVAL"), "TIMESTAMP")
    assert_equal(
        _result(OP_ADD, "TIMESTAMPTZ", "INTERVAL"), "TIMESTAMP WITH TIME ZONE"
    )


def test_two_dates_subtract_to_a_number_and_not_an_interval() raises:
    # The one pair of points on the timeline whose difference is a count.
    assert_equal(_result(OP_SUBTRACT, "DATE", "DATE"), "BIGINT")
    assert_equal(_result(OP_SUBTRACT, "TIMESTAMP", "TIMESTAMP"), "INTERVAL")
    assert_equal(_result(OP_SUBTRACT, "TIMESTAMP", "DATE"), "INTERVAL")
    assert_equal(_result(OP_SUBTRACT, "DATE", "TIMESTAMP"), "INTERVAL")


def test_a_date_plus_a_time_is_a_timestamp_and_minus_is_nothing() raises:
    assert_equal(_result(OP_ADD, "DATE", "TIME"), "TIMESTAMP")
    assert_equal(_result(OP_ADD, "TIME", "DATE"), "TIMESTAMP")
    assert_equal(_result(OP_SUBTRACT, "DATE", "TIME"), "INVALID")


def test_an_interval_scales_by_a_number_one_way_round() raises:
    assert_equal(_result(OP_MULTIPLY, "INTERVAL", "INTEGER"), "INTERVAL")
    assert_equal(_result(OP_MULTIPLY, "DOUBLE", "INTERVAL"), "INTERVAL")
    assert_equal(_result(OP_DIVIDE, "INTERVAL", "INTEGER"), "INTERVAL")
    # A number over an interval is not defined, and neither is one interval
    # over another.
    assert_equal(_result(OP_DIVIDE, "INTEGER", "INTERVAL"), "INVALID")
    assert_equal(_result(OP_DIVIDE, "INTERVAL", "INTERVAL"), "INVALID")
    assert_equal(_result(OP_ADD, "INTERVAL", "INTERVAL"), "INTERVAL")


def test_a_pair_with_no_overload_gives_back_nothing() raises:
    assert_equal(_result(OP_ADD, "DATE", "DATE"), "INVALID")
    assert_equal(_result(OP_ADD, "TIMESTAMP", "TIMESTAMP"), "INVALID")
    assert_equal(_result(OP_ADD, "BOOLEAN", "BOOLEAN"), "INVALID")
    assert_equal(_result(OP_ADD, "VARCHAR", "INTEGER"), "INVALID")
    assert_equal(_result(OP_MULTIPLY, "DATE", "INTEGER"), "INVALID")


def test_a_null_literal_does_not_take_the_other_side() raises:
    # It looks like it does on the numbers, where every overload takes two of
    # the same type, and that is the whole of the resemblance.
    assert_equal(arithmetic_type(OP_ADD, NULL, INTEGER), INTEGER)
    assert_equal(arithmetic_type(OP_DIVIDE, NULL, INTEGER), DOUBLE)
    # Two untyped literals are not NULL, they are a BIGINT, because the
    # overload that gets picked is the widest signed integer one.
    assert_equal(arithmetic_type(OP_ADD, NULL, NULL), BIGINT)
    assert_equal(negation_type(NULL), BIGINT)


def test_a_null_literal_is_bound_by_overload_and_comes_out_lopsided() raises:
    # The same pair one way round and the other, with two different answers,
    # because a date minus a count of days and a date minus a date are both
    # candidates and only one of them fits on each side.
    assert_equal(arithmetic_type(OP_SUBTRACT, DATE, NULL), DATE)
    assert_equal(arithmetic_type(OP_SUBTRACT, NULL, DATE), BIGINT)
    assert_equal(arithmetic_type(OP_SUBTRACT, TIMESTAMP, NULL), TIMESTAMP)
    assert_equal(arithmetic_type(OP_SUBTRACT, NULL, TIMESTAMP), INTERVAL)


def test_a_null_literal_with_two_candidates_binds_to_neither() raises:
    # DuckDB refuses TIME + NULL rather than choosing between the interval
    # overload and the date one, and an interval has the same problem. The
    # refusal is reported as no overload rather than as an ambiguity until the
    # function registry can list what it could not choose between.
    assert_equal(_result(OP_ADD, "TIME", "NULL"), "INVALID")
    assert_equal(_result(OP_ADD, "INTERVAL", "NULL"), "INVALID")
    assert_equal(_result(OP_SUBTRACT, "NULL", "INTERVAL"), "INVALID")
    # Subtracting is not ambiguous for a time, because nothing subtracts a
    # date from one.
    assert_equal(_result(OP_SUBTRACT, "TIME", "NULL"), "TIME")


def test_a_decimal_meeting_a_null_literal_stops_being_a_decimal() raises:
    # A decimal derives its width from both operands, an untyped literal has
    # none to give, and DuckDB answers with the NULL type rather than with a
    # decimal or an error. Mirrored on purpose.
    assert_equal(_result(OP_ADD, "DECIMAL(4,2)", "NULL"), '"NULL"')
    assert_equal(_result(OP_MULTIPLY, "DECIMAL(4,2)", "NULL"), '"NULL"')
    assert_equal(_result(OP_DIVIDE, "DECIMAL(4,2)", "NULL"), "DOUBLE")
    assert_equal(_result(OP_INT_DIVIDE, "DECIMAL(4,2)", "NULL"), "DOUBLE")


def test_a_timestamp_in_another_unit_does_not_keep_it() raises:
    # There is one overload and it takes microseconds, so the second and the
    # nanosecond are both a plain TIMESTAMP once an interval has moved them.
    assert_equal(_result(OP_ADD, "TIMESTAMP_S", "INTERVAL"), "TIMESTAMP")
    assert_equal(_result(OP_SUBTRACT, "TIMESTAMP_NS", "INTERVAL"), "TIMESTAMP")
    assert_equal(_result(OP_SUBTRACT, "TIMESTAMP_MS", "DATE"), "INTERVAL")
    # Against a zoned one DuckDB can convert either side to reach that
    # overload, so it picks neither and the query is refused.
    assert_equal(_result(OP_SUBTRACT, "TIMESTAMPTZ", "TIMESTAMP_S"), "INVALID")
    # A nanosecond time has no arithmetic at all, not even with an interval.
    assert_equal(_result(OP_ADD, "TIME_NS", "INTERVAL"), "INVALID")


def test_a_zone_on_a_time_carries_onto_the_timestamp() raises:
    assert_equal(_result(OP_ADD, "DATE", "TIME"), "TIMESTAMP")
    assert_equal(_result(OP_ADD, "DATE", "TIMETZ"), "TIMESTAMP WITH TIME ZONE")
    assert_equal(_result(OP_ADD, "TIMETZ", "DATE"), "TIMESTAMP WITH TIME ZONE")
    assert_equal(_result(OP_ADD, "TIMETZ", "INTERVAL"), "TIME WITH TIME ZONE")


def test_unary_minus_leaves_the_type_alone() raises:
    assert_equal(negation_type(INTEGER), INTEGER)
    assert_equal(negation_type(decimal(4, 2)), decimal(4, 2))
    assert_equal(negation_type(INTERVAL), INTERVAL)
    # Including an unsigned one, which then overflows at runtime rather than
    # being widened first.
    assert_equal(negation_type(parse_type("UTINYINT")).name(), "UTINYINT")
    assert_equal(negation_type(VARCHAR).id, TYPE_INVALID)
    assert_equal(negation_type(DATE).id, TYPE_INVALID)
    assert_equal(negation_type(TIMESTAMP).id, TYPE_INVALID)


def test_the_message_for_a_pair_with_no_overload_is_duckdbs() raises:
    assert_equal(
        no_such_operator(OP_ADD, DATE, DATE),
        (
            "Binder Error: No function matches the given name and argument"
            " types '+(DATE, DATE)'. You might need to add explicit type"
            " casts."
        ),
    )
    assert_equal(
        no_such_unary_operator(OP_SUBTRACT, VARCHAR),
        (
            "Binder Error: No function matches the given name and argument"
            " types '-(VARCHAR)'. You might need to add explicit type casts."
        ),
    )


def test_an_operator_knows_how_it_is_written() raises:
    assert_equal(operator_name(OP_ADD), "+")
    assert_equal(operator_name(OP_DIVIDE), "/")
    assert_equal(operator_name(OP_INT_DIVIDE), "//")
    assert_equal(operator_name(OP_MODULO), "%")
    assert_equal(operator_name(200), "")


def test_the_promotion_table_is_the_shape_it_says() raises:
    var table = promotions()
    assert_equal(len(table), PROMOTION_WIDTH * PROMOTION_WIDTH)
    for entry in table:
        assert_true(entry < TYPE_COUNT)
    # Symmetric, because the order a query wrote its operands in should not
    # change the type of the column.
    for row in range(PROMOTION_WIDTH):
        for column in range(PROMOTION_WIDTH):
            assert_equal(
                table[row * PROMOTION_WIDTH + column],
                table[column * PROMOTION_WIDTH + row],
            )


def test_promotion_covers_the_numbers_and_nothing_else() raises:
    assert_equal(promote(TINYINT, BIGINT), BIGINT)
    assert_equal(promote(VARCHAR, INTEGER).id, TYPE_INVALID)
    assert_equal(promote(decimal(4, 2), INTEGER).id, TYPE_INVALID)
    assert_false(promote(DATE, INTEGER).id == INTEGER.id)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
