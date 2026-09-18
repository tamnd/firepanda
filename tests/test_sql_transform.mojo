"""The transformer, from a parse tree to the expression AST.

Almost every test in here goes through the printer, which is why the printer
was written first. An expression is parsed, transformed and printed, and the
test names the text it expects. That reads as a claim about SQL rather than as
a claim about field numbering in an arena, and it fails in a way somebody can
act on.

The precedence tests are the ones with teeth. The grammar spells sixteen levels
as a chain of pass through rules, so a wrong fold gives a tree that still parses
and still prints and still answers the wrong number. `1 + 2 * 3` is the whole
argument for the printer parenthesizing everything: the text it gives back says
what shape the tree has.

The round trip tests print an expression, transform the printed text again, and
check the two agree. That is the property the corpus test will lean on when it
runs the same loop over 71,438 statements, so it is checked here on the shapes
that are easy to reason about first.

The refusal tests check that what is not supported yet says so by name. A
transformer that quietly dropped a clause would be worse than one that stopped,
because the query would come back with an answer that is wrong rather than with
an error somebody can read. See docs/specs/sql/05-ast-and-binder.md sections 3
and 4.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.sql import Grammar, Transform
from firepanda.sql.ast import Ast
from firepanda.sql.printer import print_expr


def _printed(
    sql: StringSlice, grammar: Grammar, rules: Transform
) raises -> String:
    """Parses one expression, transforms it and prints it back.

    Args:
        sql: The expression on its own.
        grammar: A loaded grammar.
        rules: The jump table built from it.

    Returns:
        The printed text.

    Raises:
        Error: If it does not parse, or holds something with no case.
    """
    var ast = Ast()
    var node = rules.parse_expression(sql, grammar, ast)
    return print_expr(ast, node, grammar)


def _round_trips(
    sql: StringSlice, grammar: Grammar, rules: Transform
) raises -> String:
    """Prints an expression, then prints what that text transforms to.

    Args:
        sql: The expression on its own.
        grammar: A loaded grammar.
        rules: The jump table built from it.

    Returns:
        The printed text, once the two passes have been shown to agree.

    Raises:
        Error: If either pass fails, or the two disagree.
    """
    var once = _printed(sql, grammar, rules)
    var twice: String
    try:
        twice = _printed(once, grammar, rules)
    except e:
        raise Error(
            String(
                (
                    "the printer wrote something the transformer cannot read"
                    " back: "
                ),
                once,
                "\n",
                e,
            )
        )
    if once != twice:
        raise Error(
            String(
                "a second pass changed the expression:\n  first  ",
                once,
                "\n  second ",
                twice,
            )
        )
    return once^


def test_multiplication_binds_tighter_than_addition() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("1 + 2 * 3", g, rules), "(1 + (2 * 3))")


def test_addition_binds_tighter_than_comparison() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("a + 1 < b", g, rules), "((a + 1) < b)")


def test_subtraction_folds_left() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("1 - 2 - 3", g, rules), "((1 - 2) - 3)")


def test_and_binds_tighter_than_or() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("a OR b AND c", g, rules), "(a OR (b AND c))")


def test_parentheses_change_the_shape_and_the_printed_text() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("(1 + 2) * 3", g, rules), "((1 + 2) * 3)")


def test_not_takes_the_whole_comparison() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("NOT a = b", g, rules), "(NOT (a = b))")


def test_two_nots_are_not_no_nots() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("NOT NOT a", g, rules), "(NOT (NOT a))")


def test_exponentiation_binds_tighter_than_multiplication() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("2 * 3 ^ 4", g, rules), "(2 * (3 ^ 4))")


def test_a_keyword_operator_is_written_back_in_upper_case() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("a and b", g, rules), "(a AND b)")


def test_a_bare_name_is_one_part() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("a", g, rules), "a")


def test_a_name_is_folded_down_and_a_quoted_one_is_not() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("MixedCase", g, rules), "mixedcase")
    assert_equal(_printed('"MixedCase"', g, rules), '"MixedCase"')


def test_a_dotted_name_keeps_every_part() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("s.t.c", g, rules), "s.t.c")


def test_a_string_keeps_its_value_and_not_its_quotes() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("'it''s'", g, rules), "'it''s'")


def test_a_number_loses_its_digit_separators() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("1_000_000", g, rules), "1000000")


def test_the_three_constants() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("null", g, rules), "NULL")
    assert_equal(_printed("true", g, rules), "TRUE")
    assert_equal(_printed("false", g, rules), "FALSE")


def test_a_call_keeps_its_arguments_in_order() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("f(a, 1, 'x')", g, rules), "f(a, 1, 'x')")


def test_count_star_is_a_flag_and_not_an_argument() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("count(*)", g, rules), "count(*)")


def test_a_distinct_call_says_so() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("count(DISTINCT a)", g, rules), "count(DISTINCT a)")


def test_all_is_the_default_and_is_not_written_back() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("count(ALL a)", g, rules), "count(a)")


def test_a_qualified_call_keeps_its_qualification() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("main.f(a)", g, rules), "main.f(a)")


def test_coalesce_becomes_an_ordinary_call() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("COALESCE(a, b, 1)", g, rules), "coalesce(a, b, 1)")


def test_nullif_becomes_an_ordinary_call() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("NULLIF(a, b)", g, rules), "nullif(a, b)")


def test_an_extract_prints_as_the_call_it_is() raises:
    # The keyword form is not a call in the grammar and `date_part` is the name
    # DuckDB itself shows in a plan, so the one turns into the other here and
    # nothing after this point has two shapes to read.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("EXTRACT(YEAR FROM x)", g, rules), "date_part('year', x)"
    )


def test_the_field_comes_out_in_lower_case_however_it_was_written() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("EXTRACT(Year FROM x)", g, rules), "date_part('year', x)"
    )
    assert_equal(
        _printed("extract('YEAR' FROM x)", g, rules), "date_part('year', x)"
    )


def test_a_field_the_grammar_has_no_keyword_for_is_still_read() raises:
    # Thirteen of them are keywords in the grammar and the rest arrive as
    # ordinary identifiers. Both reach the call as the same text.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("EXTRACT(isodow FROM x)", g, rules), "date_part('isodow', x)"
    )


def test_the_three_spellings_of_a_substring_print_as_the_one_call() raises:
    # The keyword form is not a call in the grammar, it is a rule with a `FROM`
    # and a `FOR` under it, and the comma form goes through the same rule. Both
    # come out the other side as the call the rest of the code knows.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("SUBSTRING(a, 2, 3)", g, rules), "substring(a, 2, 3)")
    assert_equal(
        _printed("SUBSTRING(a FROM 2 FOR 3)", g, rules), "substring(a, 2, 3)"
    )
    assert_equal(_printed("substr(a, 2, 3)", g, rules), "substr(a, 2, 3)")


def test_a_substring_written_with_only_a_for_gets_a_start_of_one() raises:
    # The standard lets the `FROM` go, and a window that does not say where it
    # begins begins at the first character. The one is written in rather than
    # left out because everything downstream counts arguments.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("SUBSTRING(a FOR 3)", g, rules), "substring(a, 1, 3)")


def test_a_substring_with_no_length_keeps_the_one_number() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("SUBSTRING(a FROM 2)", g, rules), "substring(a, 2)")
    assert_equal(_printed("SUBSTRING(a, 2)", g, rules), "substring(a, 2)")


def test_both_cast_spellings_print_as_the_same_one() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("CAST(a AS INTEGER)", g, rules), "CAST(a AS INTEGER)")
    assert_equal(_printed("a::INTEGER", g, rules), "CAST(a AS INTEGER)")


def test_try_cast_stays_a_try_cast() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("TRY_CAST(a AS INTEGER)", g, rules), "TRY_CAST(a AS INTEGER)"
    )


def test_a_searched_case() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("CASE WHEN a THEN 1 ELSE 2 END", g, rules),
        "CASE WHEN a THEN 1 ELSE 2 END",
    )


def test_a_simple_case_keeps_its_operand() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("CASE a WHEN 1 THEN 'x' WHEN 2 THEN 'y' END", g, rules),
        "CASE a WHEN 1 THEN 'x' WHEN 2 THEN 'y' END",
    )


def test_between_and_not_between() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("a BETWEEN 1 AND 2", g, rules), "(a BETWEEN 1 AND 2)")
    assert_equal(
        _printed("a NOT BETWEEN 1 AND 2", g, rules), "(a NOT BETWEEN 1 AND 2)"
    )


def test_in_and_not_in_over_a_list() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("a IN (1, 2)", g, rules), "(a IN (1, 2))")
    assert_equal(_printed("a NOT IN (1, 2)", g, rules), "(a NOT IN (1, 2))")


def test_like_and_not_like() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("a LIKE 'x%'", g, rules), "(a LIKE 'x%')")
    assert_equal(_printed("a NOT LIKE 'x%'", g, rules), "(NOT (a LIKE 'x%'))")


def test_the_three_null_tests_become_one_node() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("a IS NULL", g, rules), "(a IS NULL)")
    assert_equal(_printed("a ISNULL", g, rules), "(a IS NULL)")
    assert_equal(_printed("a NOTNULL", g, rules), "(a IS NOT NULL)")
    assert_equal(_printed("a IS NOT NULL", g, rules), "(a IS NOT NULL)")


def test_is_distinct_from_stays_one_operator() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("a IS DISTINCT FROM b", g, rules), "(a IS DISTINCT FROM b)"
    )
    assert_equal(
        _printed("a IS NOT DISTINCT FROM b", g, rules),
        "(a IS NOT DISTINCT FROM b)",
    )


def test_collate_keeps_a_name_and_not_a_column() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("a COLLATE nocase", g, rules), "(a COLLATE nocase)")


def test_an_interval_keeps_its_amount_and_its_unit() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("INTERVAL '1' DAY", g, rules), "INTERVAL '1' DAY")
    assert_equal(_printed("INTERVAL 5 MONTH", g, rules), "INTERVAL 5 MONTH")


def test_a_plural_unit_and_a_singular_one_are_the_same_unit() raises:
    # The grammar gives the two spellings one rule, and the unit comes off the
    # rule and not off the text, so there is no table of plurals anywhere.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("INTERVAL 5 MONTHS", g, rules), "INTERVAL 5 MONTH")
    assert_equal(
        _printed("INTERVAL '2' CENTURIES", g, rules), "INTERVAL '2' CENTURY"
    )
    assert_equal(_printed("INTERVAL 1 days", g, rules), "INTERVAL 1 DAY")


def test_a_compound_unit_stays_one_unit() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("INTERVAL '1' YEAR TO MONTH", g, rules),
        "INTERVAL '1' YEAR TO MONTH",
    )
    assert_equal(
        _printed("INTERVAL '1' HOUR TO SECOND", g, rules),
        "INTERVAL '1' HOUR TO SECOND",
    )


def test_an_interval_with_the_unit_inside_the_string_keeps_the_string() raises:
    # Nothing here reads the string, because the text is the value and taking a
    # duration out of it is the work a duration type does.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("INTERVAL '1 day'", g, rules), "INTERVAL '1 day'")
    assert_equal(_printed("INTERVAL '1'", g, rules), "INTERVAL '1'")


def test_an_amount_that_is_not_a_literal_keeps_its_parentheses() raises:
    # The grammar takes a string, a number or a parenthesized expression there
    # and nothing else, so an amount that is neither of the first two has to
    # keep them or it stops being a query. A string or a number inside them
    # loses them, which is the shorter way of writing the same interval.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("INTERVAL (x) DAY", g, rules), "INTERVAL (x) DAY")
    assert_equal(
        _printed("INTERVAL ('7') WEEKS", g, rules), "INTERVAL '7' WEEK"
    )


def test_a_list_constructor_and_the_array_spelling_agree() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("[1, 2]", g, rules), "[1, 2]")
    assert_equal(_printed("ARRAY[1, 2]", g, rules), "[1, 2]")


def test_a_struct_constructor() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("{'a': 1, 'b': x}", g, rules), "{'a': 1, 'b': x}")


def test_the_parameter_spellings() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("?", g, rules), "?")
    assert_equal(_printed("$1", g, rules), "$1")
    assert_equal(_printed("$name", g, rules), "$name")


def test_a_bare_star() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("*", g, rules), "*")


def test_a_qualified_star() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("t.*", g, rules), "t.*")


def test_a_star_with_every_modifier() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed(
            "* EXCLUDE (a) REPLACE (x + 1 AS b) RENAME (c AS d)", g, rules
        ),
        "* EXCLUDE (a) REPLACE ((x + 1) AS b) RENAME (c AS d)",
    )


def test_except_is_the_other_spelling_of_exclude() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("* EXCEPT (a)", g, rules), "* EXCLUDE (a)")


def test_a_prefix_minus_is_not_part_of_the_number() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("-a", g, rules), "(-a)")


def test_a_nested_expression_round_trips() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _round_trips("a * (b + c) - d / e", g, rules),
        "((a * (b + c)) - (d / e))",
    )


def test_every_shape_here_round_trips() raises:
    var g = Grammar()
    var rules = Transform(g)
    var cases: List[StaticString] = [
        "1 + 2 * 3 - 4 / 5",
        "a AND b OR NOT c",
        "f(a, g(b, 'x'), 1)",
        "count(DISTINCT a)",
        "CASE WHEN a > 1 THEN 'big' WHEN a > 0 THEN 'small' ELSE NULL END",
        "a BETWEEN 1 AND 2",
        "a NOT IN (1, 2, 3)",
        "a IS NOT NULL",
        "a IS NOT DISTINCT FROM b",
        "CAST(a AS DECIMAL(10, 2))",
        "a::VARCHAR",
        "[1, [2, 3]]",
        "{'k': f(a)}",
        '"Odd Name" + 1',
        "t.* EXCLUDE (a, b)",
        "'a''b' || 'c'",
        "a COLLATE nocase",
        "-a + +b",
        "a AT TIME ZONE 'UTC'",
        "a SIMILAR TO 'x'",
        "a ILIKE 'x'",
        "$1 + $two",
        "d + INTERVAL 3 MONTH",
        "a[1] + b[2:3]",
        "(a, b) = ROW(1, 2)",
        "INTERVAL (a + 1) DAY",
        "f(1, b := 2, c => 3)",
        "list_apply(l, lambda x: x + 1)",
        "list_reduce(l, lambda acc, e: acc + e)",
        "[x + 1 FOR x IN l]",
        "[x FOR x IN l IF x > 2]",
        "[x + y FOR x, y IN l]",
        "f(x).b.c",
        "x.b.a[1].c",
        "string_agg(a, ',' ORDER BY b DESC NULLS LAST)",
        "mode() WITHIN GROUP (ORDER BY a)",
        "lag(a IGNORE NULLS) OVER ()",
        "sum(a) EXPORT_STATE",
    ]
    for sample in cases:
        _ = _round_trips(sample, g, rules)


def test_case_does_not_change_the_tree() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("A Is Not Distinct From B", g, rules),
        _printed("a IS NOT DISTINCT FROM b", g, rules),
    )


def test_a_subquery_in_an_expression_reaches_the_statement_arena() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("(SELECT 1)", g, rules), "(SELECT 1)")
    assert_equal(_printed("a + (SELECT 1)", g, rules), "(a + (SELECT 1))")


def test_a_call_carries_the_window_that_over_names() raises:
    # The window itself is tested in `test_sql_window.mojo`. This is here for
    # the call side of it, `OVER` being one of the six things a call may carry
    # after its name.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("sum(a) OVER ()", g, rules), "sum(a) OVER ()")


def test_a_filter_becomes_a_case_around_the_argument() raises:
    # The clause does not survive the transform. A fold that passes over a null
    # cannot tell a row that was taken away from one handed to it as a null, so
    # the two spell the same question and only one of them needs a node.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("sum(a) FILTER (WHERE b > 1)", g, rules),
        "sum(CASE WHEN (b > 1) THEN a END)",
    )
    assert_equal(
        _printed("sum(a) FILTER (b > 1)", g, rules),
        "sum(CASE WHEN (b > 1) THEN a END)",
    )


def test_a_filter_on_a_star_count_counts_a_one_instead() raises:
    # There is no argument to put under the `CASE`, and counting a constant on
    # the rows the predicate keeps is counting those rows.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("count(*) FILTER (WHERE b > 1)", g, rules),
        "count(CASE WHEN (b > 1) THEN 1 END)",
    )


def test_a_filter_keeps_the_distinct_and_the_window_beside_it() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("count(DISTINCT a) FILTER (WHERE b > 1)", g, rules),
        "count(DISTINCT CASE WHEN (b > 1) THEN a END)",
    )
    assert_equal(
        _printed("sum(a) FILTER (WHERE b > 1) OVER ()", g, rules),
        "sum(CASE WHEN (b > 1) THEN a END) OVER ()",
    )


def test_a_filter_on_something_it_cannot_be_rewritten_into_refuses() raises:
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="FILTER on upper"):
        _ = _printed("upper(a) FILTER (WHERE b > 1)", g, rules)
    with assert_raises(contains="FILTER on first"):
        _ = _printed("first(a) FILTER (WHERE b > 1)", g, rules)
    with assert_raises(contains="FILTER on last"):
        _ = _printed("last(a) FILTER (WHERE b > 1)", g, rules)


def test_a_filter_on_any_value_is_rewritten_like_the_rest() raises:
    # It used to be refused beside `first` and `last` and it is not, because it
    # passes over a null, so a row the predicate turned into a null and a row it
    # took away are the same row to it. The other two read the null as a value
    # and cannot say that. See #888.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("any_value(a) FILTER (WHERE b > 1)", g, rules),
        "any_value(CASE WHEN (b > 1) THEN a END)",
    )


def test_within_group_is_read_and_written_back_where_it_was() raises:
    # The entries are the same run the in call `ORDER BY` uses, and the flag is
    # what says which side of the closing parenthesis they go back on.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("quantile(a, 0.5) WITHIN GROUP (ORDER BY a)", g, rules),
        "quantile(a, 0.5) WITHIN GROUP (ORDER BY a)",
    )
    assert_equal(
        _printed("mode() WITHIN GROUP (ORDER BY a DESC, b)", g, rules),
        "mode() WITHIN GROUP (ORDER BY a DESC, b)",
    )


def test_an_ordered_aggregate_keeps_the_order_it_was_given() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("string_agg(a ORDER BY b)", g, rules),
        "string_agg(a ORDER BY b)",
    )
    assert_equal(
        _printed("string_agg(a, ',' ORDER BY b NULLS FIRST)", g, rules),
        "string_agg(a, ',' ORDER BY b NULLS FIRST)",
    )


def test_a_call_with_only_an_order_by_in_it_still_reads() raises:
    # The argument list is optional in the grammar, so the run is the order
    # entries and nothing else and the printer has no comma to write.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("f(ORDER BY a)", g, rules), "f(ORDER BY a)")


def test_a_null_treatment_is_kept_in_the_spelling_it_was_written() raises:
    # `RESPECT NULLS` is the default said out loud, and it is kept apart from
    # not saying it so that the printer writes back what the query wrote.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("lag(a IGNORE NULLS) OVER ()", g, rules),
        "lag(a IGNORE NULLS) OVER ()",
    )
    assert_equal(
        _printed("lag(a RESPECT NULLS) OVER ()", g, rules),
        "lag(a RESPECT NULLS) OVER ()",
    )
    assert_equal(_printed("lag(a) OVER ()", g, rules), "lag(a) OVER ()")


def test_export_state_goes_after_the_parenthesis() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("sum(a) EXPORT_STATE", g, rules), "sum(a) EXPORT_STATE"
    )


def test_a_filter_and_an_order_by_are_both_kept() raises:
    # The filter rewrites the arguments and the order entries sit behind them
    # in the same run, so this is the test that the rewrite stops where the
    # arguments do.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("sum(a ORDER BY b) FILTER (WHERE b > 1)", g, rules),
        "sum(CASE WHEN (b > 1) THEN a END ORDER BY b)",
    )


def test_two_order_by_clauses_on_one_call_refuse() raises:
    # DuckDB says "cannot use multiple ORDER BY clauses with WITHIN GROUP" and
    # turns the query down. The grammar takes it, so the transformer is where
    # it is turned down here.
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="ORDER inside a call"):
        _ = _printed(
            "string_agg(a ORDER BY b) WITHIN GROUP (ORDER BY a)", g, rules
        )


def test_is_unknown_refuses_rather_than_becoming_is_null() raises:
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="IS UNKNOWN"):
        _ = _printed("a IS UNKNOWN", g, rules)


def test_an_escaped_string_refuses_rather_than_decoding_half_of_it() raises:
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="E'...' string"):
        _ = _printed("E'a\\nb'", g, rules)


def test_a_subscript_and_the_slices_around_it() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("a[1]", g, rules), "a[1]")
    assert_equal(_printed("a[1:2]", g, rules), "a[1:2]")
    assert_equal(_printed("a[1:4:2]", g, rules), "a[1:4:2]")


def test_a_slice_that_left_a_bound_out_leaves_it_out_again() raises:
    # `a[1]` and `a[1:]` both have a start and neither has an end, so the colon
    # is the only thing that tells them apart and the node carries a flag for
    # it. Printing one as the other would be a different query.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("a[:2]", g, rules), "a[:2]")
    assert_equal(_printed("a[1:]", g, rules), "a[1:]")
    assert_equal(_printed("a[:]", g, rules), "a[:]")


def test_a_subscript_takes_what_is_in_front_of_it_and_not_more() raises:
    # A subscript binds tighter than any operator, and every operand that binds
    # looser already prints inside its own parentheses, so nothing here has to
    # add a pair to keep the shape.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("a[1][2]", g, rules), "a[1][2]")
    assert_equal(_printed("(a + b)[1]", g, rules), "(a + b)[1]")
    assert_equal(_printed("f(x)[1]", g, rules), "f(x)[1]")
    assert_equal(_printed("[1, 2, 3][2]", g, rules), "[1, 2, 3][2]")


def test_an_argument_passed_by_name_keeps_the_name_and_the_value() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("f(a := 1)", g, rules), "f(a := 1)")
    assert_equal(_printed("f(1, b := 2)", g, rules), "f(1, b := 2)")


def test_the_two_spellings_of_the_assignment_are_both_written_back() raises:
    # DuckDB prints `:=` for both and firepanda prints what was written. The
    # two mean the same thing, so a query that wrote `=>` gets it back rather
    # than being told it should have written the other one.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("f(a => 1)", g, rules), "f(a => 1)")
    assert_equal(_printed("f(a := 1)", g, rules), "f(a := 1)")


def test_a_parameter_name_is_quoted_by_the_rule_of_its_own_position() raises:
    # `TypeFuncName` takes an unreserved keyword and a type or function name
    # keyword bare, and a column name keyword not at all, so `header` comes
    # back as it was written and `coalesce` comes back in quotes. Printing
    # `coalesce` bare here gives text that does not parse.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("read_csv('x', header := TRUE)", g, rules),
        "read_csv('x', header := TRUE)",
    )
    assert_equal(_printed("f(left := 1)", g, rules), "f(left := 1)")
    assert_equal(_printed("f(coalesce := 1)", g, rules), 'f("coalesce" := 1)')


def test_a_name_passed_by_name_is_folded_like_any_other_name() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("f(Header := 1)", g, rules), "f(header := 1)")
    assert_equal(_printed('f("Header" := 1)', g, rules), 'f("Header" := 1)')


def test_a_lambda_keeps_its_parameters_and_its_body() raises:
    # The body is printed the way every other operand is, fully parenthesized,
    # because the printer parenthesizes for a reparse rather than for a reader.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("list_apply(l, lambda x: x + 1)", g, rules),
        "list_apply(l, lambda x: (x + 1))",
    )
    assert_equal(
        _printed("list_reduce(l, lambda acc, e: acc + e)", g, rules),
        "list_reduce(l, lambda acc, e: (acc + e))",
    )


def test_a_lambda_parameter_is_quoted_like_a_column_name() raises:
    # `ColIdOrString` is the rule, which is the ordinary column name position
    # and also takes a string literal standing in for a name. Both come back in
    # the one spelling a name has here.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("lambda year: year", g, rules), "lambda year: year")
    assert_equal(
        _printed('lambda "odd name": 1', g, rules), 'lambda "odd name": 1'
    )
    assert_equal(_printed("lambda 'q': 1", g, rules), "lambda q: 1")


def test_the_arrow_spelling_is_an_operator_and_not_a_lambda() raises:
    # `->` reaches into a JSON value as well as writing a lambda, so the two
    # are the same text and the grammar reads both as one operator. Telling
    # them apart is a question about what is on either side of it, which is a
    # question for whoever has the types.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("x -> x + 1", g, rules), "(x -> (x + 1))")


def test_a_dot_on_something_that_is_not_a_name_reaches_a_field() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("f(x).b", g, rules), "f(x).b")
    assert_equal(_printed("columns[1].type", g, rules), "columns[1].type")
    assert_equal(_printed("{'title': x}.title", g, rules), "{'title': x}.title")
    assert_equal(_printed("(a + 1).b", g, rules), "(a + 1).b")


def test_a_dot_on_a_name_is_still_one_longer_name() raises:
    # Which part of `a.b` is the table and which is the column is a question
    # for the binder, so a name keeps its parts and does not become a field
    # access. Parentheses around a name do not change that, since the rule for
    # them is a pass through and what comes out of it is the name.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("a.b", g, rules), "a.b")
    assert_equal(_printed("(a).b", g, rules), "a.b")


def test_a_field_name_takes_every_keyword_bare() raises:
    # `ColLabel` is the widest class in the language, so a word that would be
    # quoted as a column stands bare after a dot. A name that was quoted for
    # any other reason still comes back quoted.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("f(x).select", g, rules), "f(x).select")
    assert_equal(_printed('f(x)."B"', g, rules), 'f(x)."B"')
    assert_equal(_printed('f(x)."odd name"', g, rules), 'f(x)."odd name"')


def test_a_comprehension_keeps_its_three_parts() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("[x + 1 FOR x IN l]", g, rules), "[(x + 1) FOR x IN l]"
    )
    assert_equal(_printed("[x FOR x, y IN l]", g, rules), "[x FOR x, y IN l]")


def test_a_comprehension_keeps_the_condition_on_the_end() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("[x FOR x IN l IF x > 2]", g, rules),
        "[x FOR x IN l IF (x > 2)]",
    )


def test_a_comprehension_name_is_quoted_like_a_lambda_parameter() raises:
    # The same `ColIdOrString` rule stands in both places, so a keyword comes
    # back in quotes and a string literal comes back as the name it stood for.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed('[1 FOR "select" IN l]', g, rules), '[1 FOR "select" IN l]'
    )
    assert_equal(_printed("[1 FOR 'q' IN l]", g, rules), "[1 FOR q IN l]")


def test_a_comprehension_is_not_a_list_of_one() raises:
    # `[x]` and `[x FOR x IN l]` open the same way and the grammar tries the
    # comprehension first, so the two have to come back as what they were.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("[x]", g, rules), "[x]")
    assert_equal(_printed("[x FOR x IN [l]]", g, rules), "[x FOR x IN [l]]")


def test_a_refusal_says_where_it_was() raises:
    # The example is whichever refusal the transformer still makes. What is
    # under test is the shape of the message and not the feature, so any of
    # them would do and this one changes as the list gets shorter.
    var g = Grammar()
    var rules = Transform(g)
    var sql = "MAP {'a': 1}"
    with assert_raises(contains="LINE 1: MAP {'a': 1}"):
        _ = _printed(sql, g, rules)
    with assert_raises(contains="issues/"):
        _ = _printed(sql, g, rules)


def test_a_long_chain_of_tails_is_built_once() raises:
    # A fold asks for all of its operands before it builds anything, so a level
    # with a long run of tails costs one pass and not one per tail. If that
    # ever regresses this test still passes and gets slow, so it also counts
    # the arena, which a fold that ran twice would leave garbage in.
    var g = Grammar()
    var rules = Transform(g)
    var sql = String("1")
    for _ in range(500):
        sql += " + 1"
    var ast = Ast()
    var node = rules.parse_expression(sql, g, ast)
    # 501 literals and 500 binary nodes, after the one slot the arena keeps at
    # index 0 so that 0 can mean no node.
    assert_equal(len(ast.exprs), 1002)
    assert_equal(Int(node), 1001)


def test_a_chain_two_thousand_deep_prints() raises:
    # The corpus has this shape in overflow/expression_tree_depth.test, at
    # eight kilobytes of `x + x`. A left fold makes it a tree two thousand
    # deep, and a printer that calls itself once per child does not come back
    # from that: it overruns the stack and takes the process with it, so there
    # is nothing to catch and nothing to report. See #368.
    var g = Grammar()
    var rules = Transform(g)
    var sql = String("1")
    for _ in range(2000):
        sql += " + 1"
    var ast = Ast()
    var node = rules.parse_expression(sql, g, ast)
    var text = print_expr(ast, node, g)
    # One pair of parentheses per operator, both of them outside everything the
    # operator already wrote, so the text opens with two thousand of them.
    assert_equal(text.byte_length(), 2000 * 2 + 2001 + 2000 * 3)
    assert_true(text.startswith("((("))
    assert_true(text.endswith("+ 1)"))


def test_a_call_with_many_arguments_round_trips() raises:
    var g = Grammar()
    var rules = Transform(g)
    var sql = String("f(0")
    for i in range(1, 200):
        sql += String(", ", i)
    sql += ")"
    assert_equal(_printed(sql, g, rules), sql)


def test_the_table_has_an_entry_for_every_rule_index() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(len(rules.actions), len(g.names))
    assert_true(rules.expression_rule >= 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
