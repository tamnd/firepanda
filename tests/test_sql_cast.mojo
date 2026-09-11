"""The type lattice, and what two types agree on when a query mixes them.

The values here are DuckDB 1.5's answers to
`typeof(coalesce(NULL::a, NULL::b))`. Two of them are wrong on their own terms
and are asserted anyway, because a lattice that disagreed with DuckDB would
bind queries DuckDB refuses and refuse queries DuckDB binds, and either of
those is worse than reproducing a defect.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_true,
)

from firepanda.sql.cast import (
    NUMERIC_WIDTH,
    TEMPORAL_WIDTH,
    as_decimal,
    cannot_mix,
    common_type,
    decimal_width,
    is_numeric_id,
    is_temporal_id,
    numeric_joins,
    temporal_joins,
)
from firepanda.sql.types import (
    TYPE_COUNT,
    TYPE_INVALID,
    BIGINT,
    BOOLEAN,
    DATE,
    DOUBLE,
    FLOAT,
    HUGEINT,
    INTEGER,
    INTERVAL,
    NULL,
    SqlType,
    TIMESTAMP,
    TIMESTAMP_TZ,
    TINYINT,
    VARCHAR,
    decimal,
    parse_type,
)


def _joined(a: StringSlice, b: StringSlice) raises -> String:
    """What two types written out agree on.

    Args:
        a: One type as a query would write it.
        b: The other.

    Returns:
        The common type's name, or `INVALID` when they do not mix.
    """
    return common_type(parse_type(a), parse_type(b)).name()


def test_a_type_agrees_with_itself() raises:
    assert_equal(common_type(INTEGER, INTEGER), INTEGER)
    assert_equal(common_type(VARCHAR, VARCHAR), VARCHAR)
    assert_equal(common_type(decimal(4, 2), decimal(4, 2)), decimal(4, 2))
    assert_equal(_joined("BLOB", "BLOB"), "BLOB")


def test_the_null_literal_takes_whatever_it_is_next_to() raises:
    assert_equal(common_type(NULL, INTEGER), INTEGER)
    assert_equal(common_type(INTEGER, NULL), INTEGER)
    assert_equal(common_type(NULL, VARCHAR), VARCHAR)
    assert_equal(common_type(NULL, NULL), NULL)


def test_two_signed_integers_agree_on_the_wider_one() raises:
    assert_equal(_joined("TINYINT", "SMALLINT"), "SMALLINT")
    assert_equal(_joined("INTEGER", "BIGINT"), "BIGINT")
    assert_equal(_joined("BIGINT", "HUGEINT"), "HUGEINT")
    assert_equal(_joined("HUGEINT", "TINYINT"), "HUGEINT")


def test_two_unsigned_integers_agree_on_the_wider_one() raises:
    assert_equal(_joined("UTINYINT", "USMALLINT"), "USMALLINT")
    assert_equal(_joined("UINTEGER", "UBIGINT"), "UBIGINT")
    assert_equal(_joined("UTINYINT", "UHUGEINT"), "UHUGEINT")


def test_a_signed_and_an_unsigned_integer_agree_on_something_wider() raises:
    assert_equal(_joined("TINYINT", "UTINYINT"), "SMALLINT")
    assert_equal(_joined("TINYINT", "USMALLINT"), "INTEGER")
    assert_equal(_joined("TINYINT", "UINTEGER"), "BIGINT")
    assert_equal(_joined("TINYINT", "UBIGINT"), "HUGEINT")
    assert_equal(_joined("INTEGER", "UTINYINT"), "INTEGER")


def test_a_tinyint_and_a_uhugeint_agree_on_a_type_that_cannot_hold_one() raises:
    # DuckDB really does say SMALLINT here, and a query with a value in it
    # large enough to matter then fails at runtime rather than at bind time.
    # It is a defect and it is reproduced deliberately, because the point of
    # this table is to agree with DuckDB and not to be right about integers.
    assert_equal(_joined("TINYINT", "UHUGEINT"), "SMALLINT")
    assert_equal(_joined("SMALLINT", "UHUGEINT"), "INTEGER")
    assert_equal(_joined("BIGINT", "UHUGEINT"), "HUGEINT")
    assert_equal(_joined("HUGEINT", "UHUGEINT"), "DOUBLE")


def test_a_boolean_agrees_with_an_integer_and_not_with_a_float() raises:
    assert_equal(common_type(BOOLEAN, INTEGER), INTEGER)
    assert_equal(_joined("BOOLEAN", "UTINYINT"), "UTINYINT")
    assert_equal(common_type(BOOLEAN, FLOAT).id, TYPE_INVALID)
    assert_equal(common_type(BOOLEAN, DOUBLE).id, TYPE_INVALID)


def test_an_integer_and_a_float_agree_on_the_float() raises:
    assert_equal(common_type(INTEGER, FLOAT), FLOAT)
    assert_equal(common_type(HUGEINT, FLOAT), FLOAT)
    assert_equal(common_type(FLOAT, DOUBLE), DOUBLE)
    assert_equal(common_type(BIGINT, DOUBLE), DOUBLE)


def test_two_decimals_agree_on_one_wide_enough_for_both() raises:
    # The scale is the larger of the two and the width is the larger integer
    # part plus that scale, so neither side loses a digit.
    assert_equal(_joined("DECIMAL(4,2)", "DECIMAL(6,3)"), "DECIMAL(6,3)")
    assert_equal(_joined("DECIMAL(4,2)", "DECIMAL(4,0)"), "DECIMAL(6,2)")
    assert_equal(_joined("DECIMAL(10,5)", "DECIMAL(3,1)"), "DECIMAL(10,5)")


def test_an_integer_meeting_a_decimal_counts_as_one() raises:
    assert_equal(_joined("DECIMAL(4,2)", "TINYINT"), "DECIMAL(5,2)")
    assert_equal(_joined("DECIMAL(4,2)", "INTEGER"), "DECIMAL(12,2)")
    assert_equal(_joined("DECIMAL(4,2)", "BIGINT"), "DECIMAL(21,2)")
    assert_equal(_joined("INTEGER", "DECIMAL(4,2)"), "DECIMAL(12,2)")


def test_a_decimal_too_wide_to_exist_stops_at_thirty_eight() raises:
    # Saturating rather than raising. DuckDB gives DECIMAL(38,2) here, which
    # is narrower than a HUGEINT needs, and does not complain about it.
    assert_equal(_joined("DECIMAL(4,2)", "HUGEINT"), "DECIMAL(38,2)")
    assert_equal(_joined("DECIMAL(38,10)", "BIGINT"), "DECIMAL(38,10)")


def test_two_decimals_that_do_not_fit_give_up_scale_and_not_digits() raises:
    # Which is the opposite of what saturating the width alone would do. The
    # digits in front of the point are what a value needs to exist at all, so
    # they are kept and the scale takes the loss.
    assert_equal(_joined("DECIMAL(30,25)", "DECIMAL(30,4)"), "DECIMAL(38,12)")
    assert_equal(_joined("DECIMAL(18,18)", "DECIMAL(38,10)"), "DECIMAL(38,10)")
    assert_equal(_joined("DECIMAL(38,37)", "DECIMAL(20,19)"), "DECIMAL(38,37)")
    # An integer against a decimal does not do that. It keeps the decimal's
    # scale and saturates the width, which is the test above, and the two rules
    # give different answers for the same pair of widths.
    assert_equal(_joined("DECIMAL(30,25)", "HUGEINT"), "DECIMAL(38,25)")
    assert_equal(_joined("DECIMAL(18,18)", "HUGEINT"), "DECIMAL(38,18)")
    assert_equal(_joined("DECIMAL(38,0)", "DECIMAL(4,2)"), "DECIMAL(38,0)")


def test_a_decimal_meeting_a_binary_float_gives_up_being_exact() raises:
    # The one place the reason a decimal exists is thrown away, and DuckDB
    # throws it away silently rather than asking.
    assert_equal(_joined("DECIMAL(4,2)", "DOUBLE"), "DOUBLE")
    assert_equal(_joined("DECIMAL(4,2)", "FLOAT"), "FLOAT")
    assert_equal(_joined("DOUBLE", "DECIMAL(38,10)"), "DOUBLE")


def test_an_integer_widens_to_a_decimal_that_holds_it() raises:
    assert_equal(decimal_width(TINYINT.id), 3)
    assert_equal(decimal_width(INTEGER.id), 10)
    assert_equal(decimal_width(BIGINT.id), 19)
    # Twenty and not nineteen, because a UBIGINT has one more digit than a
    # BIGINT rather than one fewer.
    assert_equal(decimal_width(parse_type("UBIGINT").id), 20)
    assert_equal(decimal_width(HUGEINT.id), 38)
    assert_equal(decimal_width(VARCHAR.id), 0)
    assert_equal(as_decimal(INTEGER), decimal(10, 0))
    assert_equal(as_decimal(decimal(4, 2)), decimal(4, 2))
    assert_equal(as_decimal(VARCHAR).id, TYPE_INVALID)


def test_a_date_and_a_timestamp_agree_on_the_timestamp() raises:
    assert_equal(common_type(DATE, TIMESTAMP), TIMESTAMP)
    assert_equal(common_type(DATE, TIMESTAMP_TZ), TIMESTAMP_TZ)
    assert_equal(common_type(TIMESTAMP, TIMESTAMP_TZ), TIMESTAMP_TZ)
    assert_equal(_joined("TIMESTAMP_S", "TIMESTAMP_NS"), "TIMESTAMP_NS")
    assert_equal(_joined("TIMESTAMP_S", "TIMESTAMP"), "TIMESTAMP")


def test_a_date_and_a_time_do_not_agree_on_anything() raises:
    # Which looks like an oversight, and is what DuckDB does.
    assert_equal(_joined("DATE", "TIME"), "INVALID")
    assert_equal(_joined("TIME", "TIME_NS"), "INVALID")
    assert_equal(_joined("TIME", "TIMETZ"), "INVALID")


def test_an_interval_agrees_with_nothing_but_an_interval() raises:
    assert_equal(common_type(INTERVAL, INTERVAL), INTERVAL)
    assert_equal(common_type(INTERVAL, TIMESTAMP).id, TYPE_INVALID)
    assert_equal(common_type(INTERVAL, BIGINT).id, TYPE_INVALID)


def test_a_string_agrees_with_nothing_but_a_string() raises:
    # coalesce('a', 1) is an INTEGER in DuckDB, but that is a literal with an
    # open type and not a VARCHAR, and the difference belongs to the binder.
    assert_equal(common_type(VARCHAR, INTEGER).id, TYPE_INVALID)
    assert_equal(common_type(VARCHAR, DATE).id, TYPE_INVALID)
    assert_equal(_joined("BLOB", "VARCHAR"), "INVALID")


def test_the_message_for_two_types_that_do_not_mix_is_duckdbs() raises:
    assert_equal(
        cannot_mix(VARCHAR, INTEGER, "COALESCE"),
        (
            "Binder Error: Cannot mix values of type VARCHAR and INTEGER in"
            " COALESCE operator - an explicit cast is required"
        ),
    )
    assert_equal(
        cannot_mix(decimal(4, 2), DATE, "CASE"),
        (
            "Binder Error: Cannot mix values of type DECIMAL(4,2) and DATE in"
            " CASE operator - an explicit cast is required"
        ),
    )


def test_the_join_is_the_same_either_way_round() raises:
    # A lattice that was not symmetric would make the order a query wrote its
    # branches in change the type of the column, which nothing should.
    var numeric = numeric_joins()
    assert_equal(len(numeric), NUMERIC_WIDTH * NUMERIC_WIDTH)
    for row in range(NUMERIC_WIDTH):
        for column in range(NUMERIC_WIDTH):
            assert_equal(
                numeric[row * NUMERIC_WIDTH + column],
                numeric[column * NUMERIC_WIDTH + row],
            )
    var temporal = temporal_joins()
    assert_equal(len(temporal), TEMPORAL_WIDTH * TEMPORAL_WIDTH)
    for row in range(TEMPORAL_WIDTH):
        for column in range(TEMPORAL_WIDTH):
            assert_equal(
                temporal[row * TEMPORAL_WIDTH + column],
                temporal[column * TEMPORAL_WIDTH + row],
            )


def test_every_entry_in_the_tables_is_a_type() raises:
    for entry in numeric_joins():
        assert_true(entry < TYPE_COUNT)
    for entry in temporal_joins():
        assert_true(entry < TYPE_COUNT)


def test_a_type_on_the_diagonal_joins_to_itself() raises:
    var numeric = numeric_joins()
    for row in range(NUMERIC_WIDTH):
        assert_not_equal(numeric[row * NUMERIC_WIDTH + row], TYPE_INVALID)
    var temporal = temporal_joins()
    for row in range(TEMPORAL_WIDTH):
        assert_not_equal(temporal[row * TEMPORAL_WIDTH + row], TYPE_INVALID)


def test_the_tables_cover_the_types_they_claim_to() raises:
    assert_true(is_numeric_id(BOOLEAN.id))
    assert_true(is_numeric_id(DOUBLE.id))
    assert_false(is_numeric_id(decimal(4, 2).id))
    assert_false(is_numeric_id(VARCHAR.id))
    assert_true(is_temporal_id(DATE.id))
    assert_true(is_temporal_id(INTERVAL.id))
    assert_false(is_temporal_id(VARCHAR.id))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
