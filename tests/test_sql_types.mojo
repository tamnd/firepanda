"""The SQL type set, and the spellings a query is allowed to use for it.

Every value asserted here was read out of DuckDB 1.5 rather than out of the
documentation, because the interesting cases are the ones where a reasonable
guess is wrong: `int8` is `BIGINT` and not `TINYINT`, a bare `DECIMAL` is
`DECIMAL(18,3)` and not `DECIMAL(38,0)`, and `VARCHAR(3)` is `VARCHAR` with the
length thrown away. Each of those is a silently different answer rather than an
error if we get it backwards.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_raises,
    assert_true,
)

from firepanda.sql.types import (
    DECIMAL_DEFAULT_SCALE,
    DECIMAL_DEFAULT_WIDTH,
    DECIMAL_MAX_WIDTH,
    TYPE_ARRAY,
    TYPE_BIGINT,
    TYPE_COUNT,
    TYPE_DOUBLE,
    TYPE_FLOAT,
    TYPE_HUGEINT,
    TYPE_INTEGER,
    TYPE_INVALID,
    TYPE_LIST,
    TYPE_NULL,
    TYPE_SMALLINT,
    TYPE_TINYINT,
    TYPE_VARCHAR,
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
    nearest_type,
    parse_type,
    spellings,
    type_for,
    type_name,
    type_names,
)


def test_a_spelling_is_not_a_type() raises:
    # The whole reason this table exists. A C programmer reads int8 as a byte
    # and DuckDB reads it as eight bytes, and taking the guess would silently
    # narrow a column by a factor of eight.
    assert_equal(type_for("int8"), TYPE_BIGINT)
    assert_equal(type_for("int1"), TYPE_TINYINT)
    assert_equal(type_for("int2"), TYPE_SMALLINT)
    assert_equal(type_for("int4"), TYPE_INTEGER)
    assert_equal(type_for("int64"), TYPE_BIGINT)
    assert_equal(type_for("int128"), TYPE_HUGEINT)
    assert_equal(type_for("float8"), TYPE_DOUBLE)
    assert_equal(type_for("float4"), TYPE_FLOAT)


def test_a_type_name_reads_in_any_case() raises:
    assert_equal(type_for("VARCHAR"), TYPE_VARCHAR)
    assert_equal(type_for("VarChar"), TYPE_VARCHAR)
    assert_equal(type_for("varchar"), TYPE_VARCHAR)


def test_a_name_that_is_not_a_type_is_invalid() raises:
    assert_equal(type_for("nosuchtype"), TYPE_INVALID)
    assert_equal(type_for(""), TYPE_INVALID)


def test_every_spelling_in_the_table_means_a_type() raises:
    # 82 spellings over 39 types, counted out of duckdb_types() on 1.5.
    var table = spellings()
    assert_equal(len(table), 82)
    assert_equal(len(type_names()), Int(TYPE_COUNT))
    for entry in table:
        assert_not_equal(entry.id, TYPE_INVALID)
        assert_true(entry.id < TYPE_COUNT)
        assert_equal(type_for(entry.text), entry.id)


def test_the_null_literal_prints_with_quotes_in_it() raises:
    # typeof(NULL) really is "NULL" with the quote marks, which looks like a
    # typo and is not.
    assert_equal(type_name(TYPE_NULL), '"NULL"')
    assert_equal(NULL.name(), '"NULL"')


def test_a_type_with_no_name_is_invalid() raises:
    assert_equal(type_name(TYPE_COUNT), "INVALID")
    assert_equal(type_name(200), "INVALID")


def test_a_timestamp_with_a_zone_prints_the_long_way() raises:
    assert_equal(TIMESTAMP_TZ.name(), "TIMESTAMP WITH TIME ZONE")
    assert_equal(type_for("timestamptz"), TIMESTAMP_TZ.id)
    assert_equal(type_for("datetime"), TIMESTAMP.id)


def test_a_list_and_an_array_are_different_types() raises:
    # typeof([1, 2, 3]::INT[3]) is INTEGER[3] and typeof([1, 2, 3]) is
    # INTEGER[], so one identifier for both would lose the length.
    assert_not_equal(TYPE_LIST, TYPE_ARRAY)
    assert_equal(type_for("list"), TYPE_LIST)
    assert_equal(type_for("array"), TYPE_ARRAY)


def test_a_bare_decimal_is_eighteen_wide_and_three_deep() raises:
    var parsed = parse_type("DECIMAL")
    assert_equal(parsed.width, DECIMAL_DEFAULT_WIDTH)
    assert_equal(parsed.scale, DECIMAL_DEFAULT_SCALE)
    assert_equal(parsed.name(), "DECIMAL(18,3)")


def test_a_decimal_with_one_argument_has_no_scale() raises:
    assert_equal(parse_type("DECIMAL(4)").name(), "DECIMAL(4,0)")


def test_a_decimal_keeps_both_of_its_arguments() raises:
    assert_equal(parse_type("DECIMAL(9,2)").name(), "DECIMAL(9,2)")
    assert_equal(parse_type("numeric( 9 , 2 )").name(), "DECIMAL(9,2)")


def test_two_decimals_of_different_shapes_are_different_types() raises:
    assert_not_equal(decimal(4, 2), decimal(5, 2))
    assert_not_equal(decimal(4, 2), decimal(4, 1))
    assert_equal(decimal(4, 2), decimal(4, 2))
    assert_not_equal(decimal(4, 2), BIGINT)


def test_a_decimal_wider_than_it_can_hold_is_refused() raises:
    with assert_raises(contains="DECIMAL type width must be between 1 and 38"):
        _ = decimal(DECIMAL_MAX_WIDTH + 1, 0)
    with assert_raises(contains="DECIMAL type width must be between 1 and 38"):
        _ = decimal(0, 0)
    with assert_raises(contains="DECIMAL type width must be between 1 and 38"):
        _ = parse_type("DECIMAL(300,0)")


def test_a_decimal_deeper_than_it_is_wide_is_refused() raises:
    with assert_raises(
        contains="DECIMAL type scale cannot be greater than width"
    ):
        _ = decimal(4, 5)
    with assert_raises(
        contains="DECIMAL type scale cannot be greater than width"
    ):
        _ = parse_type("DECIMAL(4,5)")


def test_a_length_on_a_varchar_parses_and_goes_nowhere() raises:
    # 'abcd'::VARCHAR(3) is 'abcd'. A truncation here would be helpful and
    # wrong, so there is nowhere on the type to put the three.
    var parsed = parse_type("VARCHAR(3)")
    assert_equal(parsed, VARCHAR)
    assert_equal(parsed.name(), "VARCHAR")
    assert_equal(parsed.width, 0)


def test_a_type_argument_that_is_not_a_number_is_refused() raises:
    with assert_raises(contains="is not a number a type can be given"):
        _ = parse_type("DECIMAL(x)")
    with assert_raises(contains="a type argument is empty"):
        _ = parse_type("DECIMAL(,2)")


def test_a_type_nobody_spells_that_way_is_refused() raises:
    with assert_raises(
        contains="Catalog Error: Type with name nosuchtype does not exist!"
    ):
        _ = parse_type("nosuchtype")


def test_a_near_miss_on_a_type_name_is_offered_back() raises:
    assert_equal(nearest_type("varcha"), "varchar")
    assert_equal(nearest_type("intger"), "integer")
    # DuckDB answers this one with Did you mean "struct"?, which is no use to
    # anybody, so a name with nothing near it gets no suggestion here.
    assert_equal(nearest_type("nosuchtype"), "")
    with assert_raises(contains='Did you mean "varchar"?'):
        _ = parse_type("varcha")


def test_the_integer_types_know_how_wide_they_are() raises:
    assert_equal(TINYINT.bits(), 8)
    assert_equal(INTEGER.bits(), 32)
    assert_equal(BIGINT.bits(), 64)
    assert_equal(HUGEINT.bits(), 128)
    assert_equal(SqlType(type_for("uhugeint")).bits(), 128)
    assert_equal(VARCHAR.bits(), 0)
    assert_equal(DOUBLE.bits(), 0)


def test_the_unsigned_integers_are_integers_and_are_not_signed() raises:
    var unsigned = SqlType(type_for("ubigint"))
    assert_true(unsigned.is_integer())
    assert_false(unsigned.is_signed())
    assert_true(BIGINT.is_integer())
    assert_true(BIGINT.is_signed())
    assert_false(BOOLEAN.is_integer())
    assert_false(DOUBLE.is_integer())


def test_arithmetic_applies_to_the_numbers_and_nothing_else() raises:
    assert_true(INTEGER.is_numeric())
    assert_true(FLOAT.is_numeric())
    assert_true(decimal(4, 2).is_numeric())
    assert_true(decimal(4, 2).is_decimal())
    assert_false(FLOAT.is_decimal())
    assert_false(VARCHAR.is_numeric())
    assert_false(DATE.is_numeric())
    assert_false(BOOLEAN.is_numeric())


def test_an_interval_is_not_a_point_in_time() raises:
    # It is what the difference of two of them is, and it does not sit on the
    # line with them: months and days are not fixed length.
    assert_true(DATE.is_temporal())
    assert_true(TIMESTAMP.is_temporal())
    assert_true(TIMESTAMP_TZ.is_temporal())
    assert_true(SqlType(type_for("time_ns")).is_temporal())
    assert_false(INTERVAL.is_temporal())
    assert_false(BIGINT.is_temporal())


def test_a_type_writes_itself_the_way_typeof_prints_it() raises:
    assert_equal(String(BIGINT), "BIGINT")
    assert_equal(String(decimal(9, 2)), "DECIMAL(9,2)")
    assert_equal(String(TIMESTAMP_TZ), "TIMESTAMP WITH TIME ZONE")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
