"""Picking one overload out of several, and refusing when there is no one.

Every call asserted here was put to DuckDB 1.5 first and the answer copied
down, including the two it refuses. `abs(BOOLEAN)` and `length(TINYINT)` are
binder errors because the casts those calls would need do not exist, and
`century(NULL)` is a binder error because both of the casts it could take cost
the same.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.sql.casts import Casts
from firepanda.sql.generated.casts import ANY_COST
from firepanda.sql.registry import Registry
from firepanda.sql.resolve import (
    NO_MATCH,
    TEMPLATE_COST,
    ambiguity,
    resolve,
    score,
)
from firepanda.sql.types import (
    SqlType,
    TYPE_BOOLEAN,
    TYPE_COUNT,
    TYPE_DECIMAL,
    TYPE_DOUBLE,
    TYPE_FLOAT,
    TYPE_INTEGER,
    TYPE_INTERVAL,
    TYPE_LIST,
    TYPE_MAP,
    TYPE_NULL,
    TYPE_TINYINT,
    TYPE_UTINYINT,
    TYPE_VARCHAR,
)


def _bind(
    registry: Registry, casts: Casts, name: String, arguments: List[SqlType]
) raises -> String:
    """The overload a call resolves to, written the way an error message
    writes it.

    Args:
        registry: The catalog.
        casts: The cast lattice.
        name: The name the call wrote.
        arguments: The argument types.

    Returns:
        The candidate line, or an empty string when nothing fits.
    """
    var at = registry.find(name)
    var resolved = resolve(registry, casts, at, arguments)
    if not resolved.matched():
        return String()
    return registry.signature_text(name, registry.signatures(at)[resolved.at])


def _cost(
    registry: Registry, casts: Casts, name: String, arguments: List[SqlType]
) raises -> Int:
    """What a call costs to resolve.

    Args:
        registry: The catalog.
        casts: The cast lattice.
        name: The name the call wrote.
        arguments: The argument types.

    Returns:
        The total, or `NO_MATCH` when nothing fits.
    """
    return resolve(registry, casts, registry.find(name), arguments).cost


def test_an_argument_of_the_right_type_costs_nothing() raises:
    var registry = Registry()
    var casts = Casts()
    assert_equal(
        _bind(registry, casts, "abs", [SqlType(TYPE_TINYINT)]),
        "abs(TINYINT) -> TINYINT",
    )
    assert_equal(_cost(registry, casts, "abs", [SqlType(TYPE_TINYINT)]), 0)
    # The one that looks wrong and is not: a `sum` over booleans counts them,
    # and DuckDB counts them into a `HUGEINT`.
    assert_equal(
        _bind(registry, casts, "sum", [SqlType(TYPE_BOOLEAN)]),
        "sum(BOOLEAN) -> HUGEINT",
    )


def test_an_unsigned_argument_keeps_its_own_overload() raises:
    var registry = Registry()
    var casts = Casts()
    assert_equal(
        _bind(registry, casts, "abs", [SqlType(TYPE_UTINYINT)]),
        "abs(UTINYINT) -> UTINYINT",
    )
    assert_equal(_cost(registry, casts, "abs", [SqlType(TYPE_UTINYINT)]), 0)


def test_a_call_that_needs_a_cast_takes_the_cheapest_one() raises:
    var registry = Registry()
    var casts = Casts()
    # `sqrt` has no integer overload, so an integer goes to a double.
    assert_equal(
        _bind(registry, casts, "sqrt", [SqlType(TYPE_INTEGER)]),
        "sqrt(DOUBLE) -> DOUBLE",
    )
    assert_true(_cost(registry, casts, "sqrt", [SqlType(TYPE_INTEGER)]) > 0)
    # `epoch_ms` has a `BIGINT` overload and a `TIMESTAMP` one, and an integer
    # is a cheaper `BIGINT` than it is a `TIMESTAMP`.
    assert_equal(
        _bind(registry, casts, "epoch_ms", [SqlType(TYPE_INTEGER)]),
        "epoch_ms(BIGINT) -> TIMESTAMP",
    )


def test_a_call_that_fits_nothing_resolves_to_nothing() raises:
    var registry = Registry()
    var casts = Casts()
    # No cast from a boolean to any number there is.
    assert_equal(_bind(registry, casts, "abs", [SqlType(TYPE_BOOLEAN)]), "")
    assert_equal(
        _cost(registry, casts, "abs", [SqlType(TYPE_BOOLEAN)]), NO_MATCH
    )
    # And none from a number to a string, which is why this is an error and
    # not a count of the digits.
    assert_equal(_bind(registry, casts, "length", [SqlType(TYPE_TINYINT)]), "")
    assert_equal(_bind(registry, casts, "upper", [SqlType(TYPE_BOOLEAN)]), "")


def test_an_any_parameter_takes_anything_and_is_not_free() raises:
    var registry = Registry()
    var casts = Casts()
    assert_equal(
        _bind(registry, casts, "min", [SqlType(TYPE_UTINYINT)]),
        "min(ANY) -> ANY",
    )
    assert_equal(
        _cost(registry, casts, "min", [SqlType(TYPE_UTINYINT)]), Int(ANY_COST)
    )


def test_an_exact_match_beats_an_any_parameter() raises:
    var registry = Registry()
    var casts = Casts()
    # `first` has an `ANY` overload and a `DECIMAL` one, and a decimal column
    # takes the second one because an exact match is free and `ANY` is not.
    assert_equal(
        _bind(registry, casts, "first", [SqlType(TYPE_DECIMAL)]),
        "first(DECIMAL) -> DECIMAL",
    )
    assert_equal(_cost(registry, casts, "first", [SqlType(TYPE_DECIMAL)]), 0)
    assert_equal(
        _bind(registry, casts, "first", [SqlType(TYPE_INTEGER)]),
        "first(ANY) -> ANY",
    )


def test_a_template_letter_costs_more_than_any_so_first_always_binds() raises:
    var registry = Registry()
    var casts = Casts()
    assert_true(TEMPLATE_COST > Int(ANY_COST))
    # `first` carries an `ANY` overload and a `T` one, and DuckDB binds it over
    # any column there is rather than refusing it as ambiguous, which is the
    # whole reason the two cannot cost the same.
    var at = registry.find("first")
    for id in range(Int(TYPE_COUNT)):
        var resolved = resolve(registry, casts, at, [SqlType(UInt8(id))])
        assert_true(resolved.matched())
        assert_false(resolved.ambiguous())


def test_a_macro_resolves_on_its_argument_count_alone() raises:
    var registry = Registry()
    var casts = Casts()
    # `date_add` is a macro, so it declares parameter names and no types, and
    # nothing about the arguments can make it cost anything.
    assert_equal(
        _bind(
            registry,
            casts,
            "date_add",
            [SqlType(TYPE_DECIMAL), SqlType(TYPE_INTERVAL)],
        ),
        "date_add(date, interval)",
    )
    assert_equal(
        _cost(
            registry,
            casts,
            "date_add",
            [SqlType(TYPE_DECIMAL), SqlType(TYPE_INTERVAL)],
        ),
        0,
    )
    assert_equal(
        _bind(registry, casts, "date_add", [SqlType(TYPE_DECIMAL)]), ""
    )


def test_a_varargs_overload_takes_as_many_as_it_is_given() raises:
    var registry = Registry()
    var casts = Casts()
    assert_equal(
        _bind(
            registry,
            casts,
            "greatest",
            [SqlType(TYPE_INTEGER), SqlType(TYPE_FLOAT)],
        ),
        "greatest(ANY, [ANY...]) -> ANY",
    )
    assert_equal(
        _bind(
            registry,
            casts,
            "concat",
            [
                SqlType(TYPE_VARCHAR),
                SqlType(TYPE_INTEGER),
                SqlType(TYPE_DOUBLE),
            ],
        ),
        "concat(ANY, [ANY...]) -> ANY",
    )


def test_a_container_parameter_matches_on_shape() raises:
    var registry = Registry()
    var casts = Casts()
    # `contains` has a string overload, a list one, a map one and a struct one,
    # and the argument's shape is what tells them apart.
    assert_equal(
        _bind(
            registry,
            casts,
            "contains",
            [SqlType(TYPE_VARCHAR), SqlType(TYPE_VARCHAR)],
        ),
        "contains(VARCHAR, VARCHAR) -> BOOLEAN",
    )
    assert_equal(
        _bind(
            registry,
            casts,
            "contains",
            [SqlType(TYPE_LIST), SqlType(TYPE_VARCHAR)],
        ),
        "contains(T[], T) -> BOOLEAN",
    )
    assert_equal(
        _bind(
            registry,
            casts,
            "contains",
            [SqlType(TYPE_MAP), SqlType(TYPE_VARCHAR)],
        ),
        "contains(MAP(K, V), K) -> BOOLEAN",
    )


def test_a_call_with_no_cheapest_overload_is_refused() raises:
    var registry = Registry()
    var casts = Casts()
    var at = registry.find("century")
    var resolved = resolve(registry, casts, at, [SqlType(TYPE_NULL)])
    assert_true(resolved.ambiguous())
    assert_equal(len(resolved.tied), 2)
    assert_equal(
        ambiguity(registry, "century", at, [String('"NULL"')], resolved),
        (
            "Binder Error: Could not choose a best candidate function for the"
            ' function call "century("NULL")". In order to select one, please'
            " add explicit type casts.\n\tCandidate functions:\n"
            "\tcentury(INTERVAL) -> BIGINT\n\tcentury(DATE) -> BIGINT\n"
        ),
    )


def test_the_candidates_of_a_refusal_are_not_in_catalog_order() raises:
    var registry = Registry()
    var casts = Casts()
    # DuckDB keeps the first overload to reach the cheapest price apart from
    # the ones that later match it, and writes it out last. `epoch` lists
    # `DATE` at the bottom for that reason and no other, since `DATE` is the
    # first of its overloads in the catalog.
    var at = registry.find("epoch")
    var resolved = resolve(registry, casts, at, [SqlType(TYPE_NULL)])
    assert_true(resolved.ambiguous())
    assert_equal(
        ambiguity(registry, "epoch", at, [String('"NULL"')], resolved),
        (
            "Binder Error: Could not choose a best candidate function for the"
            ' function call "epoch("NULL")". In order to select one, please add'
            " explicit type casts.\n\tCandidate functions:\n"
            "\tepoch(INTERVAL) -> DOUBLE\n\tepoch(TIME) -> DOUBLE\n"
            "\tepoch(TIME WITH TIME ZONE) -> DOUBLE\n\tepoch(TIME_NS) ->"
            " DOUBLE\n\tepoch(DATE) -> DOUBLE\n"
        ),
    )


def test_a_call_that_binds_outright_carries_no_candidates() raises:
    var registry = Registry()
    var casts = Casts()
    var resolved = resolve(
        registry, casts, registry.find("abs"), [SqlType(TYPE_INTEGER)]
    )
    assert_true(resolved.matched())
    assert_false(resolved.ambiguous())
    assert_equal(len(resolved.tied), 0)


def test_one_signature_can_be_priced_on_its_own() raises:
    var registry = Registry()
    var casts = Casts()
    var at = registry.find("abs")
    var overloads = registry.signatures(at)
    var cheapest = NO_MATCH
    for overload in overloads:
        var total = score(registry, casts, overload, [SqlType(TYPE_TINYINT)])
        if total == NO_MATCH:
            continue
        if cheapest == NO_MATCH or total < cheapest:
            cheapest = total
    assert_equal(cheapest, 0)
    # The wrong argument count is not a price, it is a refusal.
    assert_equal(
        score(
            registry,
            casts,
            overloads[0],
            [SqlType(TYPE_TINYINT), SqlType(TYPE_TINYINT)],
        ),
        NO_MATCH,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
