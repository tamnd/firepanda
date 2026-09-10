"""Statements, table references, and the printer that turns them back into SQL.

The expression tests live next door in `test_sql_ast.mojo`. These are about the
other two arenas and the way they refer to each other, which is the part that
has to be got right before anything binds against it.

The printer tests name the SQL they are producing, so a failure reads as a
disagreement about SQL rather than about which field held what.

The round trip tests are the ones with teeth. They print a statement and parse
the text with the real parser, which is the property the printer exists to have
and the property the transformer will lean on for its own tests. Two of them go
further and check that the printed text puts parentheses back exactly where the
query had them, since that is the one place the printer is allowed to leave
them out and the one place a mistake would silently change what a query means.

See docs/specs/sql/05-ast-and-binder.md sections 2 and 6.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.sql import Grammar, parse
from firepanda.sql.ast import (
    Ast,
    Expr,
    Ref,
    Stmt,
    CLAUSE_SLOTS,
    GROUP_ALL,
    GROUP_CUBE,
    GROUP_EMPTY,
    GROUP_ROLLUP,
    GROUP_SETS,
    LIMIT_ALL,
    LIMIT_PERCENT,
    LITERAL_NUMBER,
    LITERAL_STRING,
    MATERIALIZE_NO,
    MATERIALIZE_YES,
    NO_NODE,
    NULLS_FIRST,
    NULLS_LAST,
    SELECT_ALL,
    SELECT_DISTINCT,
    SORT_ASCENDING,
    SORT_DESCENDING,
)
from firepanda.sql.printer import print_ref, print_stmt


def _parts(*names: StaticString) -> List[String]:
    """Builds a name part list, because a list literal of strings is a mouthful.

    Args:
        names: The parts, outermost first.

    Returns:
        The parts as owned strings.
    """
    var out = List[String]()
    for name in names:
        out.append(String(name))
    return out^


def _one(mut ast: Ast) -> UInt32:
    """Builds the literal `1`, which most of these tests want somewhere.

    Args:
        ast: The AST.

    Returns:
        The node index.
    """
    return ast.literal(LITERAL_NUMBER, "1")


def _select_one(mut ast: Ast) -> UInt32:
    """Builds `SELECT 1` as a whole statement.

    Args:
        ast: The AST.

    Returns:
        The statement index.
    """
    var item = ast.item(_one(ast))
    return ast.select(ast.query(projection=[item]))


def _round_trips(ast: Ast, node: UInt32, g: Grammar) raises -> String:
    """Prints a statement and checks the text parses.

    Args:
        ast: The AST.
        node: The statement index.
        g: A loaded grammar.

    Returns:
        The printed statement, so a caller can check the text as well.

    Raises:
        Error: If the printed text did not parse.
    """
    var text = print_stmt(ast, node, g)
    try:
        _ = parse(text, g)
    except e:
        raise Error(
            String(
                "the printer wrote something that will not parse: ",
                text,
                "\n",
                e,
            )
        )
    return text^


# ---------------------------------------------------------------------------
# The arenas
# ---------------------------------------------------------------------------


def test_every_arena_starts_with_a_null_node() raises:
    # The convention the whole AST rests on. A field holding 0 means the part is
    # absent, in all three arenas, so all three have to reserve index 0.
    var ast = Ast()
    assert_equal(len(ast.exprs), 1)
    assert_equal(len(ast.refs), 1)
    assert_equal(len(ast.stmts), 1)
    assert_equal(ast.refs[0].kind, 0)
    assert_equal(ast.stmts[0].kind, 0)


def test_a_built_node_is_never_index_zero() raises:
    var ast = Ast()
    assert_true(ast.table(_parts("t")) != NO_NODE)
    assert_true(_select_one(ast) != NO_NODE)


def test_an_alias_run_is_the_name_and_then_the_columns() raises:
    # One run holds both, which works because column aliases without a table
    # alias are not something SQL can write.
    var ast = Ast()
    var run = ast.alias("t", _parts("a", "b"))
    assert_equal(ast.length(run), 3)
    assert_equal(ast.text(ast.at(run, 0)), "t")
    assert_equal(ast.text(ast.at(run, 2)), "b")


def test_no_alias_is_an_empty_run() raises:
    var ast = Ast()
    assert_equal(ast.alias(""), NO_NODE)
    assert_equal(ast.length(ast.alias("")), 0)


def test_a_query_always_has_a_full_clause_run() raises:
    # The clauses are read by index, so the run has to be there in full even
    # when every clause in it is empty.
    var ast = Ast()
    var node = ast.query()
    assert_equal(ast.length(ast.stmts[Int(node)].children), CLAUSE_SLOTS)


def test_reading_a_slot_of_a_missing_run_gives_the_null_node() raises:
    var ast = Ast()
    assert_equal(ast.slot(NO_NODE, 3), NO_NODE)


# ---------------------------------------------------------------------------
# Table references
# ---------------------------------------------------------------------------


def test_a_table_prints_every_name_part_it_has() raises:
    var g = Grammar()
    var ast = Ast()
    assert_equal(print_ref(ast, ast.table(_parts("t")), g), "t")
    assert_equal(print_ref(ast, ast.table(_parts("db", "s", "t")), g), "db.s.t")


def test_a_table_prints_its_alias_and_its_column_aliases() raises:
    var g = Grammar()
    var ast = Ast()
    assert_equal(print_ref(ast, ast.table(_parts("t"), "a"), g), "t AS a")
    assert_equal(
        print_ref(ast, ast.table(_parts("t"), "a", _parts("x", "y")), g),
        "t AS a (x, y)",
    )


def test_a_table_with_no_name_parts_is_an_error() raises:
    var g = Grammar()
    var ast = Ast()
    var node = ast.add_ref(Ref(kind=1, token=0))
    with assert_raises(contains="no name parts"):
        _ = print_ref(ast, node, g)


def test_a_subquery_in_a_from_prints_in_parentheses() raises:
    var g = Grammar()
    var ast = Ast()
    var node = ast.subquery_ref(_select_one(ast), "s")
    assert_equal(print_ref(ast, node, g), "(SELECT 1) AS s")


def test_lateral_prints_before_what_it_applies_to() raises:
    var g = Grammar()
    var ast = Ast()
    var node = ast.subquery_ref(_select_one(ast), "s", lateral=True)
    assert_equal(print_ref(ast, node, g), "LATERAL (SELECT 1) AS s")


def test_a_table_function_prints_its_arguments() raises:
    var g = Grammar()
    var ast = Ast()
    var ten = ast.literal(LITERAL_NUMBER, "10")
    var node = ast.function_ref(_parts("range"), [ten], "r")
    assert_equal(print_ref(ast, node, g), "range(10) AS r")


def test_a_table_function_with_no_arguments_still_prints_them() raises:
    var g = Grammar()
    var ast = Ast()
    var node = ast.function_ref(_parts("main", "gen"))
    assert_equal(print_ref(ast, node, g), "main.gen()")


def test_a_table_function_with_no_name_is_an_error() raises:
    var g = Grammar()
    var ast = Ast()
    var node = ast.add_ref(Ref(kind=3, token=0))
    with assert_raises(contains="no name"):
        _ = print_ref(ast, node, g)


def test_a_join_prints_its_sides_and_its_condition() raises:
    var g = Grammar()
    var ast = Ast()
    var on = ast.binary(
        "=", ast.column(_parts("t", "k")), ast.column(_parts("u", "k"))
    )
    var node = ast.join(
        "LEFT OUTER JOIN", ast.table(_parts("t")), ast.table(_parts("u")), on
    )
    assert_equal(print_ref(ast, node, g), "t LEFT OUTER JOIN u ON (t.k = u.k)")


def test_a_join_that_takes_no_condition_prints_none() raises:
    var g = Grammar()
    var ast = Ast()
    var node = ast.join(
        "CROSS JOIN", ast.table(_parts("t")), ast.table(_parts("u"))
    )
    assert_equal(print_ref(ast, node, g), "t CROSS JOIN u")


def test_a_using_join_prints_its_column_names() raises:
    var g = Grammar()
    var ast = Ast()
    var node = ast.join_using(
        "JOIN", ast.table(_parts("t")), ast.table(_parts("u")), _parts("a", "b")
    )
    assert_equal(print_ref(ast, node, g), "t JOIN u USING (a, b)")


def test_a_using_join_with_no_names_is_an_error() raises:
    var g = Grammar()
    var ast = Ast()
    var node = ast.join_using(
        "JOIN", ast.table(_parts("t")), ast.table(_parts("u")), List[String]()
    )
    with assert_raises(contains="no column names"):
        _ = print_ref(ast, node, g)


def test_a_join_with_more_than_one_condition_is_an_error() raises:
    # The condition run holds at most one entry. A longer one means whoever
    # built the node put something else in there.
    var g = Grammar()
    var ast = Ast()
    var one = _one(ast)
    var node = ast.add_ref(
        Ref(
            kind=4,
            token=0,
            a=ast.table(_parts("t")),
            b=ast.table(_parts("u")),
            children=ast.run([one, one]),
            payload=ast.intern("JOIN"),
        )
    )
    with assert_raises(contains="2 conditions"):
        _ = print_ref(ast, node, g)


def test_parentheses_around_a_reference_are_kept() raises:
    # The only thing that can put a join on the right of another join, so the
    # node has to survive and the printer has to put the text back.
    var g = Grammar()
    var ast = Ast()
    var inner = ast.join(
        "CROSS JOIN", ast.table(_parts("u")), ast.table(_parts("v"))
    )
    var node = ast.join(
        "CROSS JOIN", ast.table(_parts("t")), ast.parens_ref(inner)
    )
    assert_equal(print_ref(ast, node, g), "t CROSS JOIN (u CROSS JOIN v)")


def test_the_null_reference_is_not_printable() raises:
    var g = Grammar()
    var ast = Ast()
    with assert_raises(contains="null table reference"):
        _ = print_ref(ast, NO_NODE, g)


def test_a_reference_kind_nobody_knows_is_an_error() raises:
    var g = Grammar()
    var ast = Ast()
    var node = ast.add_ref(Ref(kind=200, token=0))
    with assert_raises(contains="no case for reference kind 200"):
        _ = print_ref(ast, node, g)


# ---------------------------------------------------------------------------
# The SELECT block
# ---------------------------------------------------------------------------


def test_a_select_prints_its_list_in_order() raises:
    var g = Grammar()
    var ast = Ast()
    var items = List[UInt32]()
    items.append(ast.item(ast.column(_parts("a"))))
    items.append(ast.item(ast.column(_parts("b")), "second"))
    assert_equal(
        print_stmt(ast, ast.select(ast.query(projection=items)), g),
        "SELECT a, b AS second",
    )


def test_a_from_with_nothing_selected_prints_the_from_first_form() raises:
    # DuckDB lets a query start with `FROM`, and inventing a star to fill the
    # gap would print back something the query did not write.
    var g = Grammar()
    var ast = Ast()
    var node = ast.select(ast.query(tables=[ast.table(_parts("t"))]))
    assert_equal(print_stmt(ast, node, g), "FROM t")


def test_distinct_and_all_print_where_sql_puts_them() raises:
    var g = Grammar()
    var ast = Ast()
    var item = ast.item(ast.column(_parts("a")))
    assert_equal(
        print_stmt(
            ast,
            ast.select(ast.query(projection=[item], flags=SELECT_DISTINCT)),
            g,
        ),
        "SELECT DISTINCT a",
    )
    assert_equal(
        print_stmt(
            ast,
            ast.select(ast.query(projection=[item], flags=SELECT_ALL)),
            g,
        ),
        "SELECT ALL a",
    )


def test_distinct_on_prints_its_expressions() raises:
    var g = Grammar()
    var ast = Ast()
    var item = ast.item(ast.column(_parts("a")))
    var node = ast.select(
        ast.query(
            projection=[item],
            flags=SELECT_DISTINCT,
            distinct_on=[ast.column(_parts("b"))],
        )
    )
    assert_equal(print_stmt(ast, node, g), "SELECT DISTINCT ON (b) a")


def test_where_having_and_qualify_print_their_expressions() raises:
    var g = Grammar()
    var ast = Ast()
    var item = ast.item(ast.column(_parts("a")))
    var truth = ast.literal(LITERAL_NUMBER, "1")
    var node = ast.select(
        ast.query(
            projection=[item],
            tables=[ast.table(_parts("t"))],
            filter=truth,
            having=truth,
            qualify=truth,
        )
    )
    assert_equal(
        print_stmt(ast, node, g),
        "SELECT a FROM t WHERE 1 HAVING 1 QUALIFY 1",
    )


def test_a_comma_separated_from_stays_comma_separated() raises:
    # A comma is a cross join, but rewriting it into one would print back
    # something the query did not write.
    var g = Grammar()
    var ast = Ast()
    var tables = List[UInt32]()
    tables.append(ast.table(_parts("t")))
    tables.append(ast.table(_parts("u"), "b"))
    var node = ast.select(ast.query(tables=tables))
    assert_equal(print_stmt(ast, node, g), "FROM t, u AS b")


def test_a_select_list_entry_has_to_be_a_list_entry() raises:
    var g = Grammar()
    var ast = Ast()
    var node = ast.select(ast.query(projection=[_select_one(ast)]))
    with assert_raises(contains="SELECT list holding statement kind"):
        _ = print_stmt(ast, node, g)


# ---------------------------------------------------------------------------
# GROUP BY
# ---------------------------------------------------------------------------


def test_group_by_prints_its_expressions() raises:
    var g = Grammar()
    var ast = Ast()
    var grouping = List[UInt32]()
    grouping.append(ast.group(expression=ast.column(_parts("a"))))
    grouping.append(ast.group(expression=ast.column(_parts("b"))))
    var node = ast.select(
        ast.query(tables=[ast.table(_parts("t"))], grouping=grouping)
    )
    assert_equal(print_stmt(ast, node, g), "FROM t GROUP BY a, b")


def test_group_by_all_is_one_entry_with_a_tag() raises:
    var g = Grammar()
    var ast = Ast()
    var node = ast.select(
        ast.query(
            tables=[ast.table(_parts("t"))], grouping=[ast.group(GROUP_ALL)]
        )
    )
    assert_equal(print_stmt(ast, node, g), "FROM t GROUP BY ALL")


def test_cube_and_rollup_print_their_entries() raises:
    var g = Grammar()
    var ast = Ast()
    var a = ast.group(expression=ast.column(_parts("a")))
    var b = ast.group(expression=ast.column(_parts("b")))
    var cube = ast.group(GROUP_CUBE, entries=[a, b])
    var rollup = ast.group(GROUP_ROLLUP, entries=[a])
    var grouping = List[UInt32]()
    grouping.append(cube)
    grouping.append(rollup)
    var node = ast.select(
        ast.query(tables=[ast.table(_parts("t"))], grouping=grouping)
    )
    assert_equal(
        print_stmt(ast, node, g), "FROM t GROUP BY CUBE (a, b), ROLLUP (a)"
    )


def test_grouping_sets_nest_and_the_empty_set_prints_as_parentheses() raises:
    var g = Grammar()
    var ast = Ast()
    var a = ast.group(expression=ast.column(_parts("a")))
    var empty = ast.group(GROUP_EMPTY)
    var sets = ast.group(GROUP_SETS, entries=[a, empty])
    var node = ast.select(
        ast.query(tables=[ast.table(_parts("t"))], grouping=[sets])
    )
    assert_equal(
        print_stmt(ast, node, g), "FROM t GROUP BY GROUPING SETS (a, ())"
    )


def test_a_grouping_tag_nobody_knows_is_an_error() raises:
    var g = Grammar()
    var ast = Ast()
    var node = ast.select(
        ast.query(tables=[ast.table(_parts("t"))], grouping=[ast.group(200)])
    )
    with assert_raises(contains="no case for grouping 200"):
        _ = print_stmt(ast, node, g)


# ---------------------------------------------------------------------------
# ORDER BY, LIMIT and OFFSET
# ---------------------------------------------------------------------------


def test_order_by_prints_its_direction_and_its_nulls() raises:
    var g = Grammar()
    var ast = Ast()
    var order = List[UInt32]()
    order.append(
        ast.order(ast.column(_parts("a")), SORT_DESCENDING, NULLS_FIRST)
    )
    order.append(ast.order(ast.column(_parts("b")), SORT_ASCENDING, NULLS_LAST))
    order.append(ast.order(ast.column(_parts("c"))))
    var node = ast.select(
        ast.query(tables=[ast.table(_parts("t"))]), ast.modifiers(order=order)
    )
    assert_equal(
        print_stmt(ast, node, g),
        "FROM t ORDER BY a DESC NULLS FIRST, b ASC NULLS LAST, c",
    )


def test_order_by_all_is_an_entry_with_no_expression() raises:
    var g = Grammar()
    var ast = Ast()
    var node = ast.select(
        ast.query(tables=[ast.table(_parts("t"))]),
        ast.modifiers(order=[ast.order(direction=SORT_DESCENDING)]),
    )
    assert_equal(print_stmt(ast, node, g), "FROM t ORDER BY ALL DESC")


def test_limit_and_offset_print_their_expressions() raises:
    var g = Grammar()
    var ast = Ast()
    var node = ast.select(
        ast.query(tables=[ast.table(_parts("t"))]),
        ast.modifiers(
            limit=ast.literal(LITERAL_NUMBER, "10"),
            offset=ast.literal(LITERAL_NUMBER, "5"),
        ),
    )
    assert_equal(print_stmt(ast, node, g), "FROM t LIMIT 10 OFFSET 5")


def test_limit_all_and_limit_percent_print_their_flags() raises:
    var g = Grammar()
    var ast = Ast()
    var t = ast.table(_parts("t"))
    var all = ast.select(ast.query(tables=[t]), ast.modifiers(flags=LIMIT_ALL))
    assert_equal(print_stmt(ast, all, g), "FROM t LIMIT ALL")
    var percent = ast.select(
        ast.query(tables=[t]),
        ast.modifiers(
            limit=ast.literal(LITERAL_NUMBER, "10"), flags=LIMIT_PERCENT
        ),
    )
    assert_equal(print_stmt(ast, percent, g), "FROM t LIMIT 10%")


def test_a_trailing_node_that_is_not_modifiers_is_an_error() raises:
    var g = Grammar()
    var ast = Ast()
    var node = ast.select(ast.query(), _select_one(ast))
    with assert_raises(contains="trailed by statement kind"):
        _ = print_stmt(ast, node, g)


# ---------------------------------------------------------------------------
# WITH
# ---------------------------------------------------------------------------


def test_a_with_prints_its_entries_before_the_query() raises:
    var g = Grammar()
    var ast = Ast()
    var body = _select_one(ast)
    var node = ast.select(
        ast.query(tables=[ast.table(_parts("a"))]),
        ctes=[ast.cte("a", body)],
    )
    assert_equal(print_stmt(ast, node, g), "WITH a AS (SELECT 1) FROM a")


def test_a_with_prints_recursive_and_its_column_aliases() raises:
    var g = Grammar()
    var ast = Ast()
    var body = _select_one(ast)
    var node = ast.select(
        ast.query(tables=[ast.table(_parts("a"))]),
        ctes=[ast.cte("a", body, _parts("n"))],
        recursive=True,
    )
    assert_equal(
        print_stmt(ast, node, g), "WITH RECURSIVE a (n) AS (SELECT 1) FROM a"
    )


def test_materialized_prints_both_ways_round() raises:
    var g = Grammar()
    var ast = Ast()
    var yes = ast.cte("a", _select_one(ast), materialize=MATERIALIZE_YES)
    var no = ast.cte("b", _select_one(ast), materialize=MATERIALIZE_NO)
    var ctes = List[UInt32]()
    ctes.append(yes)
    ctes.append(no)
    var node = ast.select(ast.query(tables=[ast.table(_parts("a"))]), ctes=ctes)
    assert_equal(
        print_stmt(ast, node, g),
        (
            "WITH a AS MATERIALIZED (SELECT 1), b AS NOT MATERIALIZED"
            " (SELECT 1) FROM a"
        ),
    )


def test_a_with_entry_has_to_be_a_with_entry() raises:
    var g = Grammar()
    var ast = Ast()
    var node = ast.select(ast.query(), ctes=[_select_one(ast)])
    with assert_raises(contains="WITH holding statement kind"):
        _ = print_stmt(ast, node, g)


# ---------------------------------------------------------------------------
# Set operations, VALUES and TABLE
# ---------------------------------------------------------------------------


def test_a_set_operation_prints_its_operator_as_it_was_written() raises:
    var g = Grammar()
    var ast = Ast()
    var left = ast.query(projection=[ast.item(_one(ast))])
    var right = ast.query(projection=[ast.item(_one(ast))])
    var node = ast.select(ast.set_operation("UNION ALL BY NAME", left, right))
    assert_equal(
        print_stmt(ast, node, g), "SELECT 1 UNION ALL BY NAME SELECT 1"
    )


def test_a_set_operand_that_was_parenthesized_gets_its_parentheses_back() raises:
    # The one place the printer puts parentheses in around a statement, and the
    # reason it can leave them out everywhere else.
    var g = Grammar()
    var ast = Ast()
    var left = ast.select(
        ast.query(projection=[ast.item(_one(ast))]),
        ast.modifiers(limit=_one(ast)),
    )
    var right = ast.query(projection=[ast.item(_one(ast))])
    var node = ast.select(ast.set_operation("UNION", left, right))
    assert_equal(print_stmt(ast, node, g), "(SELECT 1 LIMIT 1) UNION SELECT 1")


def test_values_prints_a_row_at_a_time() raises:
    var g = Grammar()
    var ast = Ast()
    var one = _one(ast)
    var two = ast.literal(LITERAL_NUMBER, "2")
    var rows = List[List[UInt32]]()
    rows.append([one, two])
    rows.append([two, one])
    var node = ast.select(ast.values(rows))
    assert_equal(print_stmt(ast, node, g), "VALUES (1, 2), (2, 1)")


def test_a_values_with_no_rows_is_an_error() raises:
    var g = Grammar()
    var ast = Ast()
    var node = ast.select(ast.values(List[List[UInt32]]()))
    with assert_raises(contains="no rows"):
        _ = print_stmt(ast, node, g)


def test_a_table_statement_prints_its_name() raises:
    var g = Grammar()
    var ast = Ast()
    var node = ast.select(ast.table_statement(_parts("s", "t")))
    assert_equal(print_stmt(ast, node, g), "TABLE s.t")


def test_a_table_statement_with_no_name_is_an_error() raises:
    var g = Grammar()
    var ast = Ast()
    var node = ast.select(ast.table_statement(List[String]()))
    with assert_raises(contains="no name"):
        _ = print_stmt(ast, node, g)


def test_the_null_statement_is_not_printable() raises:
    var g = Grammar()
    var ast = Ast()
    with assert_raises(contains="null statement"):
        _ = print_stmt(ast, NO_NODE, g)


def test_a_statement_kind_nobody_knows_is_an_error() raises:
    var g = Grammar()
    var ast = Ast()
    var node = ast.add_stmt(Stmt(kind=200, token=0))
    with assert_raises(contains="no case for statement kind 200"):
        _ = print_stmt(ast, node, g)


# ---------------------------------------------------------------------------
# The expressions that hold a statement
# ---------------------------------------------------------------------------


def test_a_scalar_subquery_prints_in_parentheses() raises:
    var g = Grammar()
    var ast = Ast()
    var node = ast.select(
        ast.query(projection=[ast.item(ast.subquery(_select_one(ast)))])
    )
    assert_equal(print_stmt(ast, node, g), "SELECT (SELECT 1)")


def test_exists_prints_both_ways_round() raises:
    var g = Grammar()
    var ast = Ast()
    var yes = ast.exists(_select_one(ast))
    var no = ast.exists(_select_one(ast), negated=True)
    var node = ast.select(
        ast.query(
            tables=[ast.table(_parts("t"))], filter=ast.binary("AND", yes, no)
        )
    )
    assert_equal(
        print_stmt(ast, node, g),
        "FROM t WHERE ((EXISTS (SELECT 1)) AND (NOT EXISTS (SELECT 1)))",
    )


def test_in_with_a_subquery_on_the_right_prints_the_subquery() raises:
    var g = Grammar()
    var ast = Ast()
    var test = ast.in_subquery(ast.column(_parts("a")), _select_one(ast))
    var node = ast.select(
        ast.query(tables=[ast.table(_parts("t"))], filter=test)
    )
    assert_equal(print_stmt(ast, node, g), "FROM t WHERE (a IN (SELECT 1))")


def test_not_in_with_a_subquery_keeps_its_not() raises:
    var g = Grammar()
    var ast = Ast()
    var test = ast.in_subquery(
        ast.column(_parts("a")), _select_one(ast), negated=True
    )
    var node = ast.select(
        ast.query(tables=[ast.table(_parts("t"))], filter=test)
    )
    assert_equal(print_stmt(ast, node, g), "FROM t WHERE (a NOT IN (SELECT 1))")


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_printed_statements_parse() raises:
    # The property the printer exists to have, over one statement of every shape
    # this arena can hold.
    var g = Grammar()
    var ast = Ast()
    var a = ast.column(_parts("a"))
    var one = _one(ast)
    var t = ast.table(_parts("t"))
    var u = ast.table(_parts("u"), "b")

    _ = _round_trips(ast, _select_one(ast), g)
    _ = _round_trips(ast, ast.select(ast.query(tables=[t])), g)
    _ = _round_trips(
        ast,
        ast.select(
            ast.query(projection=[ast.item(a, "x")], tables=[t], filter=one)
        ),
        g,
    )
    _ = _round_trips(
        ast,
        ast.select(
            ast.query(
                projection=[ast.item(a)],
                tables=[t],
                flags=SELECT_DISTINCT,
                distinct_on=[a],
            )
        ),
        g,
    )
    _ = _round_trips(
        ast,
        ast.select(
            ast.query(
                tables=[t],
                grouping=[ast.group(expression=a)],
                having=one,
                qualify=one,
            )
        ),
        g,
    )
    _ = _round_trips(
        ast,
        ast.select(ast.query(tables=[t], grouping=[ast.group(GROUP_ALL)])),
        g,
    )
    _ = _round_trips(
        ast,
        ast.select(
            ast.query(
                tables=[t],
                grouping=[
                    ast.group(
                        GROUP_SETS,
                        entries=[
                            ast.group(expression=a),
                            ast.group(GROUP_EMPTY),
                        ],
                    )
                ],
            )
        ),
        g,
    )
    _ = _round_trips(
        ast,
        ast.select(
            ast.query(tables=[t]),
            ast.modifiers(
                order=[ast.order(a, SORT_DESCENDING, NULLS_LAST)],
                limit=one,
                offset=one,
            ),
        ),
        g,
    )
    _ = _round_trips(
        ast,
        ast.select(ast.query(tables=[t]), ast.modifiers(order=[ast.order()])),
        g,
    )
    _ = _round_trips(
        ast,
        ast.select(ast.query(tables=[t]), ast.modifiers(flags=LIMIT_ALL)),
        g,
    )
    _ = _round_trips(
        ast,
        ast.select(ast.query(tables=[ast.join("CROSS JOIN", t, u)])),
        g,
    )
    _ = _round_trips(
        ast,
        ast.select(
            ast.query(
                tables=[
                    ast.join(
                        "LEFT OUTER JOIN",
                        t,
                        u,
                        ast.binary(
                            "=",
                            ast.column(_parts("t", "k")),
                            ast.column(_parts("b", "k")),
                        ),
                    )
                ]
            )
        ),
        g,
    )
    _ = _round_trips(
        ast,
        ast.select(
            ast.query(tables=[ast.join_using("JOIN", t, u, _parts("k"))])
        ),
        g,
    )
    _ = _round_trips(
        ast,
        ast.select(
            ast.query(
                tables=[
                    ast.join(
                        "CROSS JOIN",
                        t,
                        ast.parens_ref(ast.join("CROSS JOIN", u, t)),
                    )
                ]
            )
        ),
        g,
    )
    _ = _round_trips(
        ast,
        ast.select(
            ast.query(
                tables=[ast.subquery_ref(_select_one(ast), "s", _parts("n"))]
            )
        ),
        g,
    )
    _ = _round_trips(
        ast,
        ast.select(
            ast.query(
                tables=[
                    ast.function_ref(
                        _parts("range"),
                        [ast.literal(LITERAL_NUMBER, "10")],
                        "r",
                    )
                ]
            )
        ),
        g,
    )
    _ = _round_trips(
        ast,
        ast.select(
            ast.query(tables=[t]),
            ctes=[ast.cte("c", _select_one(ast), _parts("n"))],
            recursive=True,
        ),
        g,
    )
    _ = _round_trips(
        ast,
        ast.select(
            ast.set_operation(
                "UNION ALL",
                ast.query(projection=[ast.item(one)]),
                ast.query(projection=[ast.item(one)]),
            )
        ),
        g,
    )
    _ = _round_trips(ast, ast.select(ast.table_statement(_parts("t"))), g)
    _ = _round_trips(
        ast,
        ast.select(ast.query(tables=[t], filter=ast.exists(_select_one(ast)))),
        g,
    )
    _ = _round_trips(
        ast,
        ast.select(
            ast.query(
                tables=[t],
                filter=ast.in_subquery(a, _select_one(ast), negated=True),
            )
        ),
        g,
    )
    _ = _round_trips(
        ast,
        ast.select(
            ast.query(projection=[ast.item(ast.subquery(_select_one(ast)))])
        ),
        g,
    )


def test_a_values_statement_round_trips() raises:
    var g = Grammar()
    var ast = Ast()
    var rows = List[List[UInt32]]()
    rows.append([_one(ast), ast.literal(LITERAL_STRING, "a")])
    rows.append([_one(ast), ast.literal(LITERAL_STRING, "b")])
    assert_equal(
        _round_trips(ast, ast.select(ast.values(rows)), g),
        "VALUES (1, 'a'), (1, 'b')",
    )


def test_an_awkward_name_survives_the_round_trip() raises:
    # A table alias that is a reserved word has to come back quoted, or the text
    # parses as something else entirely.
    var g = Grammar()
    var ast = Ast()
    var node = ast.select(
        ast.query(tables=[ast.table(_parts("Mixed Case"), "select")])
    )
    assert_equal(_round_trips(ast, node, g), 'FROM "Mixed Case" AS "select"')


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
