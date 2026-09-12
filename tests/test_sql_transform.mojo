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
    # the call side of it, since `OVER` used to be one of the four call
    # modifiers that refused and the other three still do.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("sum(a) OVER ()", g, rules), "sum(a) OVER ()")


def test_an_ordered_aggregate_refuses() raises:
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="ORDER inside a call"):
        _ = _printed("string_agg(a ORDER BY b)", g, rules)


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


def test_a_subscript_refuses() raises:
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="a slice or a subscript"):
        _ = _printed("a[1]", g, rules)


def test_a_refusal_says_where_it_was() raises:
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="LINE 1: INTERVAL 1 DAY"):
        _ = _printed("INTERVAL 1 DAY", g, rules)
    with assert_raises(contains="issues/"):
        _ = _printed("INTERVAL 1 DAY", g, rules)


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
