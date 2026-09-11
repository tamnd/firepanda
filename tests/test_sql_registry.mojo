"""The tier 1 function catalog: what a name is, and what its overloads are.

Every signature asserted here was read off DuckDB 1.5 rather than written down
from what a function ought to do, so the ones worth naming are the ones that
look wrong: `sum(BOOLEAN)` gives back a `HUGEINT`, `min` has a second overload
that takes a count and gives back a list, `length` accepts a `BIT`, and
`round(DECIMAL)` stays a decimal instead of going to a double.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_true,
)

from firepanda.sql.catalog import NOT_FOUND
from firepanda.sql.classify import aggregate_names, window_only_names
from firepanda.sql.generated.functions import (
    DUCKDB_VERSION,
    KIND_AGGREGATE,
    KIND_MACRO,
    KIND_SCALAR,
    NAME_COUNT,
    OVERLOAD_COUNT,
)
from firepanda.sql.registry import (
    NO_SLOT,
    ROLE_ANY,
    ROLE_EXACT,
    ROLE_LIST,
    ROLE_TEMPLATE,
    Registry,
)
from firepanda.sql.types import TYPE_INVALID


def _signature(registry: Registry, name: String, at: Int) -> String:
    """The nth overload of a name, printed the way an error message prints it.

    Args:
        registry: The catalog.
        name: The name to look up.
        at: Which overload, in catalog order.

    Returns:
        The candidate line.
    """
    var found = registry.find(name)
    return registry.signature_text(name, registry.signatures(found)[at])


def test_the_table_reads_back_as_long_as_it_says() raises:
    var registry = Registry()
    assert_equal(len(registry), NAME_COUNT)
    assert_equal(len(registry.overloads), OVERLOAD_COUNT)
    assert_equal(len(registry.roles), len(registry.spellings))


def test_the_table_records_which_duckdb_it_came_off() raises:
    # A signature that changes between releases changes what firepanda binds,
    # so the version is part of the table and not a note in a commit message.
    assert_true(DUCKDB_VERSION.startswith("1."))


def test_a_name_is_found_without_regard_to_case() raises:
    var registry = Registry()
    assert_not_equal(registry.find("upper"), NOT_FOUND)
    assert_equal(registry.find("UPPER"), registry.find("upper"))
    assert_equal(registry.find("UpPeR"), registry.find("upper"))


def test_a_name_that_is_in_no_catalog_is_not_found() raises:
    var registry = Registry()
    assert_equal(registry.find("upperr"), NOT_FOUND)
    assert_false(registry.contains("nosuchfunction"))


def test_the_three_names_that_are_grammar_rules_are_not_here() raises:
    # `coalesce` and `ifnull` are grammar rules, which is why the wrong number
    # of arguments to either is a parser error, and `current_timestamp` is a
    # keyword. All three belong to the transformer.
    var registry = Registry()
    assert_false(registry.contains("coalesce"))
    assert_false(registry.contains("ifnull"))
    assert_false(registry.contains("current_timestamp"))


def test_the_names_are_sorted_so_a_lookup_is_a_binary_search() raises:
    var registry = Registry()
    for at in range(1, len(registry)):
        assert_true(registry.names[at - 1] < registry.names[at])


def test_every_name_is_found_at_its_own_position() raises:
    var registry = Registry()
    for at in range(len(registry)):
        assert_equal(registry.find(registry.names[at]), at)


def test_a_name_is_one_kind_and_only_one() raises:
    var registry = Registry()
    assert_equal(registry.kind_of(registry.find("upper")), KIND_SCALAR)
    assert_equal(registry.kind_of(registry.find("sum")), KIND_AGGREGATE)
    assert_equal(registry.kind_of(registry.find("date_add")), KIND_MACRO)


def test_a_name_a_table_function_shares_is_still_the_scalar_one() raises:
    # `repeat` and `histogram` are each a table function as well, and a call
    # written in an expression never means the table one.
    var registry = Registry()
    assert_equal(registry.kind_of(registry.find("repeat")), KIND_SCALAR)
    assert_equal(registry.kind_of(registry.find("histogram")), KIND_AGGREGATE)


def test_an_aggregate_is_an_aggregate() raises:
    var registry = Registry()
    assert_true(registry.is_aggregate("sum"))
    assert_true(registry.is_aggregate("SUM"))
    assert_true(registry.is_aggregate("string_agg"))
    assert_false(registry.is_aggregate("upper"))
    assert_false(registry.is_aggregate("nosuchfunction"))


def test_the_classifiers_aggregates_are_aggregates_here_too() raises:
    # `classify.mojo` carries its own name list because it runs before anything
    # has read a catalog, and the two were copied off the same DuckDB. This is
    # what catches them drifting apart.
    var registry = Registry()
    for name in aggregate_names():
        if registry.contains(name):
            assert_true(registry.is_aggregate(name))


def test_a_window_only_name_is_an_aggregate_in_the_catalog() raises:
    # DuckDB files `rank` and `lag` under `aggregate` the same as `sum`, so the
    # kind column does not say which of them needs an `OVER`. That question is
    # the classifier's and it is not answered here.
    var registry = Registry()
    for name in window_only_names():
        if registry.contains(name):
            assert_true(registry.is_aggregate(name))


def test_an_alias_carries_the_whole_overload_list() raises:
    var registry = Registry()
    var substr = registry.find("substr")
    var substring = registry.find("substring")
    assert_equal(registry.counts[substr], registry.counts[substring])
    assert_equal(registry.aliases[substr], "substring")
    assert_equal(registry.aliases[substring], String())


def test_an_alias_prints_the_name_the_query_wrote() raises:
    # The candidate list uses the spelling in the query, so asking about
    # `substr` never answers with `substring`.
    var registry = Registry()
    assert_equal(
        _signature(registry, "substr", 1), "substr(VARCHAR, BIGINT) -> VARCHAR"
    )
    assert_equal(
        _signature(registry, "substring", 1),
        "substring(VARCHAR, BIGINT) -> VARCHAR",
    )


def test_sum_of_a_boolean_gives_back_a_hugeint() raises:
    # Nobody would write this down, which is the whole argument for generating
    # the table instead of typing it.
    var registry = Registry()
    assert_equal(_signature(registry, "sum", 1), "sum(BOOLEAN) -> HUGEINT")


def test_min_has_a_second_overload_that_gives_back_a_list() raises:
    var registry = Registry()
    assert_equal(_signature(registry, "min", 0), "min(ANY) -> ANY")
    assert_equal(_signature(registry, "min", 1), "min(ANY, BIGINT) -> ANY[]")


def test_length_accepts_a_bit() raises:
    var registry = Registry()
    assert_equal(_signature(registry, "length", 1), "length(BIT) -> BIGINT")
    assert_equal(_signature(registry, "length", 2), "length(ANY[]) -> BIGINT")


def test_rounding_a_decimal_keeps_the_decimal() raises:
    var registry = Registry()
    var at = registry.find("round")
    var seen = False
    for overload in registry.signatures(at):
        if (
            registry.signature_text("round", overload)
            == "round(DECIMAL, INTEGER) -> DECIMAL"
        ):
            seen = True
    assert_true(seen)


def test_an_overload_with_no_parameters_at_all() raises:
    var registry = Registry()
    assert_equal(_signature(registry, "pi", 0), "pi() -> DOUBLE")
    assert_equal(_signature(registry, "count", 1), "count() -> BIGINT")


def test_a_variadic_writes_its_trailing_type_in_brackets() raises:
    var registry = Registry()
    assert_equal(
        _signature(registry, "concat", 0), "concat(ANY, [ANY...]) -> ANY"
    )


def test_a_variadic_takes_anything_from_its_arity_up() raises:
    var registry = Registry()
    var concat = registry.signatures(registry.find("concat"))[0]
    assert_true(concat.variadic())
    assert_false(concat.accepts(0))
    assert_true(concat.accepts(1))
    assert_true(concat.accepts(9))


def test_a_fixed_arity_takes_one_count_and_nothing_else() raises:
    var registry = Registry()
    var upper = registry.signatures(registry.find("upper"))[0]
    assert_false(upper.variadic())
    assert_equal(upper.varargs, NO_SLOT)
    assert_false(upper.accepts(0))
    assert_true(upper.accepts(1))
    assert_false(upper.accepts(2))


def test_the_parameter_past_a_variadics_own_slots_is_the_trailing_one() raises:
    var registry = Registry()
    var concat = registry.signatures(registry.find("concat"))[0]
    assert_equal(registry.spelling(registry.parameter(concat, 0)), "ANY")
    assert_equal(registry.spelling(registry.parameter(concat, 5)), "ANY")


def test_the_parameter_past_a_fixed_arity_is_no_slot() raises:
    var registry = Registry()
    var upper = registry.signatures(registry.find("upper"))[0]
    assert_equal(registry.parameter(upper, 1), NO_SLOT)
    assert_equal(registry.spelling(NO_SLOT), String())


def test_a_macro_writes_its_parameter_names_and_no_return() raises:
    # A macro has no parameter types at all, which is why DuckDB shows names
    # where it would otherwise show a signature.
    var registry = Registry()
    assert_equal(
        _signature(registry, "date_add", 0), "date_add(date, interval)"
    )
    assert_equal(
        _signature(registry, "split_part", 0),
        "split_part(string, delimiter, position)",
    )


def test_a_macro_still_knows_how_many_arguments_it_takes() raises:
    var registry = Registry()
    var date_add = registry.signatures(registry.find("date_add"))[0]
    assert_equal(date_add.returns, NO_SLOT)
    assert_true(date_add.accepts(2))
    assert_false(date_add.accepts(1))


def test_a_concrete_spelling_parses_to_a_type() raises:
    var registry = Registry()
    var upper = registry.signatures(registry.find("upper"))[0]
    var slot = Int(registry.parameter(upper, 0))
    assert_equal(registry.roles[slot], ROLE_EXACT)
    assert_not_equal(registry.types[slot].id, TYPE_INVALID)


def test_any_is_its_own_role_rather_than_a_type() raises:
    var registry = Registry()
    var typeof = registry.signatures(registry.find("typeof"))[0]
    var slot = Int(registry.parameter(typeof, 0))
    assert_equal(registry.spelling(registry.parameter(typeof, 0)), "ANY")
    assert_equal(registry.roles[slot], ROLE_ANY)


def test_a_one_letter_name_is_a_template() raises:
    # `lag(T, BIGINT, T)` has to give back whatever the call passed, so the
    # letter is a slot to be bound rather than a type to be matched.
    var registry = Registry()
    var lag = registry.signatures(registry.find("lag"))[0]
    var slot = Int(registry.parameter(lag, 0))
    assert_equal(registry.spelling(registry.parameter(lag, 0)), "T")
    assert_equal(registry.roles[slot], ROLE_TEMPLATE)


def test_a_spelling_ending_in_brackets_is_a_list_of_the_rest() raises:
    var registry = Registry()
    var length = registry.signatures(registry.find("length"))[2]
    var slot = Int(registry.parameter(length, 0))
    assert_equal(registry.spelling(registry.parameter(length, 0)), "ANY[]")
    assert_equal(registry.roles[slot], ROLE_LIST)
    assert_equal(registry.elements[slot], ROLE_ANY)


def test_a_list_of_a_template_keeps_the_letter() raises:
    var registry = Registry()
    var list = registry.signatures(registry.find("list"))[0]
    var slot = Int(list.returns)
    assert_equal(registry.spelling(list.returns), "T[]")
    assert_equal(registry.roles[slot], ROLE_LIST)
    assert_equal(registry.elements[slot], ROLE_TEMPLATE)


def test_every_spelling_a_signature_names_is_in_the_text_section() raises:
    var registry = Registry()
    for overload in registry.overloads:
        assert_true(Int(overload.returns) < len(registry.spellings))
        assert_true(Int(overload.varargs) < len(registry.spellings))
        for at in range(Int(overload.arity)):
            var slot = registry.parameters[Int(overload.first) + at]
            assert_true(slot != NO_SLOT)
            assert_true(Int(slot) < len(registry.spellings))


def test_every_name_has_at_least_one_overload() raises:
    var registry = Registry()
    for at in range(len(registry)):
        assert_true(registry.counts[at] > 0)
        assert_equal(len(registry.signatures(at)), Int(registry.counts[at]))


def test_an_unknown_name_is_a_catalog_error_with_a_suggestion() raises:
    var registry = Registry()
    assert_equal(
        registry.unknown("lenght"),
        (
            "Catalog Error: Scalar Function with name lenght does not exist!\n"
            'Did you mean "length"?'
        ),
    )


def test_an_unknown_aggregate_is_still_called_a_scalar_function() raises:
    # DuckDB says `Scalar Function` whatever the name resembles, and copying
    # that is the point.
    var registry = Registry()
    assert_true(registry.unknown("summ").startswith("Catalog Error: Scalar"))
    assert_true(registry.unknown("summ").endswith('Did you mean "sum"?'))


def test_a_name_that_resembles_nothing_gets_no_suggestion() raises:
    # DuckDB has no threshold here and will answer `zzzzzzqq` with whichever
    # name sorts nearest, which is not worth copying.
    var registry = Registry()
    assert_equal(
        registry.unknown("zzzzzzqq"),
        "Catalog Error: Scalar Function with name zzzzzzqq does not exist!",
    )
    assert_equal(registry.nearest("zzzzzzqq"), NOT_FOUND)


def test_a_call_that_fits_no_overload_lists_every_candidate() raises:
    var registry = Registry()
    assert_equal(
        registry.no_match("upper", registry.find("upper"), [String("BOOLEAN")]),
        (
            "Binder Error: No function matches the given name and argument"
            " types 'upper(BOOLEAN)'. You might need to add explicit type"
            " casts.\n\tCandidate functions:\n\tupper(VARCHAR) -> VARCHAR\n"
        ),
    )


def test_a_call_with_no_arguments_writes_empty_brackets() raises:
    var registry = Registry()
    assert_equal(
        registry.no_match("concat", registry.find("concat"), List[String]()),
        (
            "Binder Error: No function matches the given name and argument"
            " types 'concat()'. You might need to add explicit type casts.\n"
            "\tCandidate functions:\n\tconcat(ANY, [ANY...]) -> ANY\n"
        ),
    )


def test_the_candidates_are_listed_in_catalog_order() raises:
    var registry = Registry()
    var message = registry.no_match(
        "md5", registry.find("md5"), [String("INTEGER")]
    )
    assert_true(
        message.endswith("\tmd5(VARCHAR) -> VARCHAR\n\tmd5(BLOB) -> VARCHAR\n")
    )


def test_a_call_to_a_macro_is_a_different_sentence() raises:
    var registry = Registry()
    assert_equal(
        registry.no_match(
            "date_add", registry.find("date_add"), [String("INTEGER")]
        ),
        (
            "Binder Error: Macro date_add() does not support the supplied"
            " arguments. You might need to add explicit type casts.\n"
            "Candidate macros:\n\tdate_add(date, interval)"
        ),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
