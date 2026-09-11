"""The implicit cast lattice: which casts exist, and what they cost.

Every number asserted here was measured off DuckDB 1.5 by tools/gen_casts.py
rather than reasoned about, so what is worth asserting is the part nobody would
guess: `TINYINT` does not reach `VARCHAR`, `UTINYINT` reaches `SMALLINT` and
not `TINYINT`, and a `NULL` is priced by a rule of its own.

The costs themselves are checked as an ordering and not as numbers. They are a
linear program's answer, so a DuckDB that changed one of its own costs without
changing which overload wins would move every number here and break nothing,
and a test that pinned them down would fail for no reason at all.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.sql.casts import Casts
from firepanda.sql.generated.casts import (
    ANY_COST,
    CAST_COUNT,
    DUCKDB_VERSION,
    NO_CAST,
    SCALAR_COUNT,
)
from firepanda.sql.types import (
    TYPE_BIGINT,
    TYPE_BOOLEAN,
    TYPE_COUNT,
    TYPE_DOUBLE,
    TYPE_INTEGER,
    TYPE_INVALID,
    TYPE_LIST,
    TYPE_NULL,
    TYPE_SMALLINT,
    TYPE_STRUCT,
    TYPE_TINYINT,
    TYPE_UTINYINT,
    TYPE_VARCHAR,
    TYPE_VARIANT,
)


def test_the_table_reads_back_as_wide_as_it_says() raises:
    var casts = Casts()
    assert_equal(len(casts), SCALAR_COUNT)


def test_the_table_records_which_duckdb_it_came_off() raises:
    # DuckDB has changed which casts are implicit between releases, and a query
    # that binds one way on 1.5 and another way on 1.6 is a compatibility break
    # worth seeing rather than inheriting.
    assert_true(DUCKDB_VERSION.startswith("1."))


def test_a_type_costs_nothing_to_itself() raises:
    var casts = Casts()
    for id in range(Int(TYPE_COUNT)):
        if not casts.covers(UInt8(id)):
            continue
        assert_equal(casts.cost(UInt8(id), UInt8(id)), 0)


def test_the_table_speaks_for_scalars_and_says_so() raises:
    var casts = Casts()
    assert_true(casts.covers(TYPE_BOOLEAN))
    assert_true(casts.covers(TYPE_VARIANT))
    # A list, an array, a struct, a map and a union cast by their elements, and
    # a `SqlType` carries no element type, so there is nothing here to say
    # about them.
    assert_false(casts.covers(TYPE_LIST))
    assert_false(casts.covers(TYPE_STRUCT))
    assert_false(casts.covers(TYPE_INVALID))


def test_a_type_the_table_does_not_cover_reaches_nothing() raises:
    var casts = Casts()
    assert_equal(casts.cost(TYPE_LIST, TYPE_VARCHAR), NO_CAST)
    assert_equal(casts.cost(TYPE_VARCHAR, TYPE_LIST), NO_CAST)
    assert_equal(casts.cost(TYPE_COUNT, TYPE_VARCHAR), NO_CAST)
    assert_equal(casts.cost(TYPE_VARCHAR, TYPE_COUNT), NO_CAST)


def test_an_unsigned_type_reaches_the_wider_signed_one_and_not_the_same_one() raises:
    var casts = Casts()
    # The one nobody guesses. A `UTINYINT` holds 255 and a `TINYINT` holds 127,
    # so the cast that looks like the obvious one is the one DuckDB refuses.
    assert_true(casts.reaches(TYPE_UTINYINT, TYPE_SMALLINT))
    assert_false(casts.reaches(TYPE_UTINYINT, TYPE_TINYINT))
    assert_false(casts.reaches(TYPE_TINYINT, TYPE_UTINYINT))


def test_a_number_does_not_reach_a_string() raises:
    var casts = Casts()
    # Which is why `length` over an integer column is a binder error and not a
    # count of its digits.
    assert_false(casts.reaches(TYPE_TINYINT, TYPE_VARCHAR))
    assert_false(casts.reaches(TYPE_INTEGER, TYPE_VARCHAR))


def test_widening_an_integer_beats_making_it_a_double() raises:
    var casts = Casts()
    assert_true(
        casts.cost(TYPE_INTEGER, TYPE_BIGINT)
        < casts.cost(TYPE_INTEGER, TYPE_DOUBLE)
    )
    assert_true(
        casts.cost(TYPE_SMALLINT, TYPE_INTEGER)
        < casts.cost(TYPE_SMALLINT, TYPE_DOUBLE)
    )


def test_a_variant_reaches_everything_and_almost_nothing_reaches_it() raises:
    var casts = Casts()
    for id in range(Int(TYPE_COUNT)):
        if not casts.covers(UInt8(id)):
            continue
        assert_true(casts.reaches(TYPE_VARIANT, UInt8(id)))
        if id == Int(TYPE_NULL) or id == Int(TYPE_VARIANT):
            continue
        assert_false(casts.reaches(UInt8(id), TYPE_VARIANT))


def test_a_null_is_priced_by_a_rule_of_its_own() raises:
    var casts = Casts()
    # A `NULL` is not a value being converted, and DuckDB does not charge for
    # it as though it were: `abs(NULL)` binds to the `BIGINT` overload, which
    # only happens because `BIGINT` is the cheapest thing a `NULL` becomes.
    assert_true(
        casts.cost(TYPE_NULL, TYPE_BIGINT) < casts.cost(TYPE_NULL, TYPE_VARCHAR)
    )
    assert_true(
        casts.cost(TYPE_NULL, TYPE_BIGINT)
        < casts.cost(TYPE_INTEGER, TYPE_BIGINT)
    )


def test_an_any_parameter_is_not_free() raises:
    # `first` has an `ANY` overload and a `DECIMAL` one and takes the `ANY` one
    # for an integer column, which only says something because both cost
    # something.
    assert_true(ANY_COST > 0)


def test_the_table_holds_as_many_casts_as_it_says() raises:
    var casts = Casts()
    var edges = 0
    for target in range(Int(TYPE_COUNT)):
        for source in range(Int(TYPE_COUNT)):
            if casts.cost(UInt8(source), UInt8(target)) > 0:
                edges += 1
    assert_equal(edges, CAST_COUNT)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
