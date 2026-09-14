"""What a call comes out as, once an overload has won.

Every type asserted here was put to DuckDB 1.5 first over a table of columns
and the answer copied down. A column rather than a literal for the same reason
the differential uses one: DuckDB folds an expression whose arguments are all
constants before `typeof` can be read off it.
"""

from std.testing import TestSuite, assert_equal, assert_true

from firepanda.sql.casts import Casts
from firepanda.sql.registry import Registry
from firepanda.sql.resolve import resolve
from firepanda.sql.result import result_type
from firepanda.sql.types import (
    INVALID,
    SqlType,
    TYPE_BOOLEAN,
    TYPE_DATE,
    TYPE_INTEGER,
    TYPE_INTERVAL,
    TYPE_TINYINT,
    TYPE_UUID,
    TYPE_VARCHAR,
    decimal,
)


def _typed(
    registry: Registry, casts: Casts, name: String, arguments: List[SqlType]
) raises -> String:
    """What a call comes out as, written the way `typeof` writes it.

    Args:
        registry: The catalog.
        casts: The cast lattice.
        name: The name the call wrote.
        arguments: The argument types.

    Returns:
        The type's name, or an empty string where the call does not resolve or
        the type cannot be said.
    """
    var at = registry.find(name)
    var resolved = resolve(registry, casts, at, arguments)
    if not resolved.matched() or resolved.ambiguous():
        return String()
    var type = result_type(
        registry, name, registry.signatures(at)[resolved.at], arguments
    )
    if type == INVALID:
        return String()
    return type.name()


def test_a_concrete_return_is_what_the_catalog_says() raises:
    """The ordinary case, where nothing is derived at all."""
    var registry = Registry()
    var casts = Casts()
    assert_equal(
        _typed(registry, casts, "length", [SqlType(TYPE_VARCHAR)]),
        String("BIGINT"),
    )
    assert_equal(
        _typed(registry, casts, "abs", [SqlType(TYPE_INTEGER)]),
        String("INTEGER"),
    )


def test_a_template_return_is_the_argument_it_was_given() raises:
    """`first(ANY) -> ANY` over a column is that column's type."""
    var registry = Registry()
    var casts = Casts()
    assert_equal(
        _typed(registry, casts, "first", [SqlType(TYPE_UUID)]), String("UUID")
    )
    assert_equal(
        _typed(registry, casts, "any_value", [SqlType(TYPE_INTERVAL)]),
        String("INTERVAL"),
    )
    assert_equal(
        _typed(registry, casts, "first", [decimal(5, 2)]),
        String("DECIMAL(5,2)"),
    )


def test_the_first_template_parameter_is_the_one_that_answers() raises:
    """`arg_max` orders by its second argument and returns its first.

    Both parameters are `ANY` and the two arguments the other way round give
    the other type, which is what says the position matters rather than the
    pair.
    """
    var registry = Registry()
    var casts = Casts()
    assert_equal(
        _typed(
            registry,
            casts,
            "arg_max",
            [SqlType(TYPE_BOOLEAN), SqlType(TYPE_INTERVAL)],
        ),
        String("BOOLEAN"),
    )
    assert_equal(
        _typed(
            registry,
            casts,
            "arg_max",
            [SqlType(TYPE_INTERVAL), SqlType(TYPE_BOOLEAN)],
        ),
        String("INTERVAL"),
    )


def test_a_variadic_template_is_what_every_argument_agrees_on() raises:
    """`greatest` casts its arguments together, so the answer is none of them.

    `DECIMAL(5,2)` and `INTEGER` agree on `DECIMAL(12,2)`, which is wider than
    either, and adding a `TINYINT` changes nothing because it already fits.
    """
    var registry = Registry()
    var casts = Casts()
    assert_equal(
        _typed(
            registry, casts, "greatest", [decimal(5, 2), SqlType(TYPE_INTEGER)]
        ),
        String("DECIMAL(12,2)"),
    )
    assert_equal(
        _typed(
            registry,
            casts,
            "greatest",
            [decimal(5, 2), SqlType(TYPE_INTEGER), SqlType(TYPE_TINYINT)],
        ),
        String("DECIMAL(12,2)"),
    )


def test_a_bare_decimal_return_is_the_width_the_arguments_agree_on() raises:
    """A bare `DECIMAL` parameter is a cast target, so the result is the cast.

    `mod(DECIMAL(5,2), INTEGER)` casts both to `DECIMAL(12,2)` and comes out as
    that, which neither argument was.
    """
    var registry = Registry()
    var casts = Casts()
    assert_equal(
        _typed(registry, casts, "abs", [decimal(5, 2)]), String("DECIMAL(5,2)")
    )
    assert_equal(
        _typed(registry, casts, "mod", [decimal(5, 2), SqlType(TYPE_INTEGER)]),
        String("DECIMAL(12,2)"),
    )
    assert_equal(
        _typed(registry, casts, "mod", [decimal(5, 2), decimal(18, 6)]),
        String("DECIMAL(18,6)"),
    )


def test_rounding_a_decimal_keeps_its_width_and_drops_its_scale() raises:
    """`ceil`, `floor` and `round` all leave room for every digit in front."""
    var registry = Registry()
    var casts = Casts()
    assert_equal(
        _typed(registry, casts, "ceil", [decimal(5, 2)]), String("DECIMAL(5,0)")
    )
    assert_equal(
        _typed(registry, casts, "floor", [decimal(18, 6)]),
        String("DECIMAL(18,0)"),
    )
    assert_equal(
        _typed(registry, casts, "round", [decimal(18, 18)]),
        String("DECIMAL(18,0)"),
    )


def test_summing_a_decimal_widens_it_and_averaging_one_leaves_it() raises:
    """The two names that declare the same return and produce different types.

    `sum` and `avg` both say `DECIMAL` in the catalog. `sum` widens to 38
    digits and keeps the scale, so an overflow takes the longest it can to
    arrive, and `avg` gives a `DOUBLE` and does not keep anything.
    """
    var registry = Registry()
    var casts = Casts()
    assert_equal(
        _typed(registry, casts, "sum", [decimal(5, 2)]), String("DECIMAL(38,2)")
    )
    assert_equal(
        _typed(registry, casts, "sum", [decimal(38, 10)]),
        String("DECIMAL(38,10)"),
    )
    assert_equal(
        _typed(registry, casts, "avg", [decimal(5, 2)]), String("DOUBLE")
    )


def test_a_median_widens_where_the_midpoint_is_not_one_of_the_rows() raises:
    """A median over an even count is the midpoint of the middle two.

    An integer has no midpoint, so ten of them give a `DOUBLE`, and a `DATE`
    gives a `TIMESTAMP`. A `FLOAT` keeps itself because the midpoint of two
    floats is a float, and so does everything the midpoint means nothing for.
    """
    var registry = Registry()
    var casts = Casts()
    assert_equal(
        _typed(registry, casts, "median", [SqlType(TYPE_INTEGER)]),
        String("DOUBLE"),
    )
    assert_equal(
        _typed(registry, casts, "median", [SqlType(TYPE_DATE)]),
        String("TIMESTAMP"),
    )
    assert_equal(
        _typed(registry, casts, "median", [decimal(5, 2)]),
        String("DECIMAL(5,2)"),
    )
    assert_equal(
        _typed(registry, casts, "median", [SqlType(TYPE_VARCHAR)]),
        String("VARCHAR"),
    )


def test_concat_is_a_varchar_whatever_its_signature_says() raises:
    """It is declared over `ANY` and returns `ANY` and is not either."""
    var registry = Registry()
    var casts = Casts()
    assert_equal(
        _typed(registry, casts, "concat", [SqlType(TYPE_INTEGER)]),
        String("VARCHAR"),
    )
    assert_equal(
        _typed(
            registry,
            casts,
            "concat",
            [SqlType(TYPE_VARCHAR), SqlType(TYPE_DATE)],
        ),
        String("VARCHAR"),
    )


def test_a_container_return_is_left_unsaid() raises:
    """`list(T) -> T[]` needs an element type `SqlType` does not carry.

    The call resolves, so the refusal here is about the type and not about the
    binding, which is the whole of what issue #780 is left with.
    """
    var registry = Registry()
    var casts = Casts()
    var at = registry.find("list")
    var arguments = List[SqlType]()
    arguments.append(SqlType(TYPE_INTEGER))
    assert_true(resolve(registry, casts, at, arguments).matched())
    assert_equal(_typed(registry, casts, "list", arguments), String())
    assert_equal(
        _typed(registry, casts, "histogram", [SqlType(TYPE_INTEGER)]),
        String(),
    )
    assert_equal(
        _typed(
            registry,
            casts,
            "str_split",
            [SqlType(TYPE_VARCHAR), SqlType(TYPE_VARCHAR)],
        ),
        String(),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
