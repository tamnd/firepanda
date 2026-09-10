"""Window specifications, from `OVER ()` to a named frame with an exclusion.

The shape of these tests is the shape of test_sql_select.mojo: parse, transform,
print, and name the text that should come back. Reading the text back is what
catches a clause that was quietly dropped, and a window has six parts that could
each be dropped on its own.

Two of them come back written differently from how they went in, and both are
here with a test saying so. `OVER w` prints as `OVER (w)`, because the AST
records that the window is a name and not which of DuckDB's two spellings for
that was used. `RANGE CURRENT ROW` with no `EXCLUDE` prints without one, because
no exclusion and `EXCLUDE NO OTHERS` are the same thing and only the second was
written down.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.sql import Grammar, Transform
from firepanda.sql.ast import (
    Ast,
    BOUND_CURRENT_ROW,
    BOUND_NONE,
    BOUND_PRECEDING,
    BOUND_UNBOUNDED_FOLLOWING,
    BOUND_UNBOUNDED_PRECEDING,
    CLAUSE_WINDOW,
    EXCLUDE_NONE,
    EXCLUDE_TIES,
    EXPR_FRAME,
    EXPR_FUNCTION,
    EXPR_WINDOW,
    FRAME_GROUPS,
    FRAME_RANGE,
    FRAME_ROWS,
    NO_NODE,
    STMT_WINDOW,
    frame_end,
    frame_exclude,
    frame_mode,
    frame_start,
    frame_tags,
)
from firepanda.sql.printer import print_stmt


def _printed(
    sql: StringSlice, grammar: Grammar, rules: Transform
) raises -> String:
    """Parses one statement, transforms it, prints it and prints it again.

    Args:
        sql: The whole statement.
        grammar: A loaded grammar.
        rules: The jump table built from it.

    Returns:
        The printed text.

    Raises:
        Error: If it does not parse, or holds something with no case, or a
            second pass changes it.
    """
    var ast = Ast()
    var once = print_stmt(
        ast, rules.parse_statement(sql, grammar, ast), grammar
    )
    var again = Ast()
    var twice = print_stmt(
        again, rules.parse_statement(once, grammar, again), grammar
    )
    if once != twice:
        raise Error(
            String(
                "a second pass changed the statement:\n  first  ",
                once,
                "\n  second ",
                twice,
            )
        )
    return once^


def test_a_window_with_nothing_in_it() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT rank() OVER () FROM t", g, rules),
        "SELECT rank() OVER () FROM t",
    )


def test_the_three_clauses_a_window_is_made_of() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT rank() OVER (PARTITION BY a, b) FROM t", g, rules),
        "SELECT rank() OVER (PARTITION BY a, b) FROM t",
    )
    assert_equal(
        _printed("SELECT rank() OVER (ORDER BY a DESC) FROM t", g, rules),
        "SELECT rank() OVER (ORDER BY a DESC) FROM t",
    )
    assert_equal(
        _printed(
            "SELECT sum(a) OVER (ROWS UNBOUNDED PRECEDING) FROM t", g, rules
        ),
        "SELECT sum(a) OVER (ROWS UNBOUNDED PRECEDING) FROM t",
    )


def test_all_three_at_once_keep_their_order() raises:
    # The order is the grammar's order. Printing them any other way produces
    # text that does not parse, which the second pass in `_printed` would catch,
    # but the text is named here so the intent is on the page.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed(
            (
                "SELECT sum(a) OVER (PARTITION BY b, c ORDER BY d ASC NULLS"
                " LAST ROWS BETWEEN 1 PRECEDING AND CURRENT ROW EXCLUDE TIES)"
                " FROM t"
            ),
            g,
            rules,
        ),
        (
            "SELECT sum(a) OVER (PARTITION BY b, c ORDER BY d ASC NULLS LAST"
            " ROWS BETWEEN 1 PRECEDING AND CURRENT ROW EXCLUDE TIES) FROM t"
        ),
    )


def test_the_three_framings() raises:
    var g = Grammar()
    var rules = Transform(g)
    for framing in ["ROWS", "RANGE", "GROUPS"]:
        var sql = String(
            "SELECT sum(a) OVER (ORDER BY b ",
            framing,
            " BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) FROM t",
        )
        assert_equal(_printed(sql, g, rules), sql)


def test_every_way_a_bound_can_be_written() raises:
    var g = Grammar()
    var rules = Transform(g)
    var bounds: List[StaticString] = [
        "UNBOUNDED PRECEDING",
        "1 PRECEDING",
        "CURRENT ROW",
        "1 FOLLOWING",
        "UNBOUNDED FOLLOWING",
    ]
    for start in bounds:
        for end in bounds:
            var sql = String(
                "SELECT sum(a) OVER (ROWS BETWEEN ",
                start,
                " AND ",
                end,
                ") FROM t",
            )
            assert_equal(_printed(sql, g, rules), sql)


def test_a_frame_with_one_bound_stays_a_frame_with_one_bound() raises:
    # `ROWS 1 PRECEDING` is not `ROWS BETWEEN 1 PRECEDING AND CURRENT ROW`, even
    # though the two run the same. Filling in the end would be this stage
    # deciding something, and this stage decides nothing.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT sum(a) OVER (ROWS 1 PRECEDING) FROM t", g, rules),
        "SELECT sum(a) OVER (ROWS 1 PRECEDING) FROM t",
    )


def test_the_four_exclusions() raises:
    var g = Grammar()
    var rules = Transform(g)
    var exclusions: List[StaticString] = [
        "CURRENT ROW",
        "GROUP",
        "TIES",
        "NO OTHERS",
    ]
    for what in exclusions:
        var sql = String(
            "SELECT sum(a) OVER (ROWS UNBOUNDED PRECEDING EXCLUDE ",
            what,
            ") FROM t",
        )
        assert_equal(_printed(sql, g, rules), sql)


def test_no_exclusion_is_not_written_as_one() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT sum(a) OVER (RANGE CURRENT ROW) FROM t", g, rules),
        "SELECT sum(a) OVER (RANGE CURRENT ROW) FROM t",
    )


def test_a_named_window_is_printed_in_parentheses() raises:
    # `OVER w` and `OVER (w)` are the same window and the AST keeps one of them.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed(
            "SELECT rank() OVER w FROM t WINDOW w AS (PARTITION BY a)",
            g,
            rules,
        ),
        "SELECT rank() OVER (w) FROM t WINDOW w AS (PARTITION BY a)",
    )
    assert_equal(
        _printed(
            "SELECT rank() OVER (w) FROM t WINDOW w AS (PARTITION BY a)",
            g,
            rules,
        ),
        "SELECT rank() OVER (w) FROM t WINDOW w AS (PARTITION BY a)",
    )


def test_a_window_written_on_top_of_a_named_one() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed(
            (
                "SELECT rank() OVER (w ORDER BY b DESC) FROM t WINDOW w AS"
                " (PARTITION BY a)"
            ),
            g,
            rules,
        ),
        (
            "SELECT rank() OVER (w ORDER BY b DESC) FROM t WINDOW w AS"
            " (PARTITION BY a)"
        ),
    )


def test_a_window_clause_with_several_definitions() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed(
            (
                "SELECT a FROM t WINDOW w AS (ORDER BY b), v AS (w RANGE 5"
                " FOLLOWING)"
            ),
            g,
            rules,
        ),
        "SELECT a FROM t WINDOW w AS (ORDER BY b), v AS (w RANGE 5 FOLLOWING)",
    )


def test_the_window_clause_comes_before_the_qualify() raises:
    # `SimpleSelect` puts `WINDOW` before `QUALIFY`, so the printer has to as
    # well or the text it wrote is a syntax error.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed(
            (
                "SELECT a FROM t WINDOW w AS (ORDER BY b) QUALIFY rank() OVER w"
                " = 1"
            ),
            g,
            rules,
        ),
        (
            "SELECT a FROM t WINDOW w AS (ORDER BY b) QUALIFY (rank() OVER (w)"
            " = 1)"
        ),
    )


def test_a_window_on_a_star_call() raises:
    # `count(*)` takes a different path out of the printer from `count(a)`,
    # because the star is a flag rather than an argument, and the `OVER` has to
    # be written on both.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT count(*) OVER (ORDER BY a) FROM t", g, rules),
        "SELECT count(*) OVER (ORDER BY a) FROM t",
    )


def test_a_window_name_that_needs_quoting() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed(
            'SELECT rank() OVER "my w" FROM t WINDOW "my w" AS (ORDER BY a)',
            g,
            rules,
        ),
        'SELECT rank() OVER ("my w") FROM t WINDOW "my w" AS (ORDER BY a)',
    )


def test_an_expression_bound_is_an_expression() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed(
            (
                "SELECT sum(a) OVER (ORDER BY b RANGE BETWEEN x + 1 PRECEDING"
                " AND y * 2 FOLLOWING) FROM t"
            ),
            g,
            rules,
        ),
        (
            "SELECT sum(a) OVER (ORDER BY b RANGE BETWEEN (x + 1) PRECEDING AND"
            " (y * 2) FOLLOWING) FROM t"
        ),
    )


def test_the_other_call_modifiers_still_refuse_by_name() raises:
    # `OVER` is the only one of the four that is implemented, and the other
    # three have to keep saying so rather than being silently ignored now that
    # the loop over them no longer stops at the first.
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="FILTER"):
        _ = _printed("SELECT sum(a) FILTER (WHERE b) FROM t", g, rules)
    with assert_raises(contains="WITHIN"):
        _ = _printed(
            "SELECT quantile(a) WITHIN GROUP (ORDER BY a) FROM t", g, rules
        )


def test_a_filter_next_to_an_over_still_refuses() raises:
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="FILTER"):
        _ = _printed(
            "SELECT sum(a) FILTER (WHERE b) OVER (ORDER BY c) FROM t", g, rules
        )


def test_the_nodes_a_window_query_builds() raises:
    # The text tests above would still pass if the window hung off the wrong
    # field, so one of them looks at the arenas.
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement(
        (
            "SELECT rank() OVER (PARTITION BY a ORDER BY b ROWS BETWEEN 1"
            " PRECEDING AND CURRENT ROW EXCLUDE TIES) FROM t WINDOW w AS"
            " (ORDER BY c)"
        ),
        g,
        ast,
    )

    # `parse_statement` hands back the `STMT_SELECT` that holds the `WITH` and
    # the `ORDER BY`, and the query with the clauses on it is under that.
    var query = ast.stmts[Int(node)].a
    var clauses = ast.stmts[Int(query)].children
    var windows = ast.slot(clauses, CLAUSE_WINDOW)
    assert_equal(ast.length(windows), 1)
    var definition = ast.at(windows, 0)
    assert_equal(ast.stmts[Int(definition)].kind, STMT_WINDOW)
    assert_equal(ast.text(ast.stmts[Int(definition)].payload), "w")

    var projection = ast.items(ast.slot(clauses, 0))
    var call = ast.stmts[Int(projection[0])].a
    assert_equal(ast.exprs[Int(call)].kind, EXPR_FUNCTION)
    var window = ast.exprs[Int(call)].b
    assert_equal(ast.exprs[Int(window)].kind, EXPR_WINDOW)
    assert_equal(ast.length(ast.exprs[Int(window)].children), 1)
    assert_equal(ast.length(ast.exprs[Int(window)].a), 1)
    assert_equal(ast.exprs[Int(window)].payload, NO_NODE)

    var frame = ast.exprs[Int(window)].b
    assert_equal(ast.exprs[Int(frame)].kind, EXPR_FRAME)
    var tags = ast.exprs[Int(frame)].payload
    assert_equal(frame_mode(tags), FRAME_ROWS)
    assert_equal(frame_start(tags), BOUND_PRECEDING)
    assert_equal(frame_end(tags), BOUND_CURRENT_ROW)
    assert_equal(frame_exclude(tags), EXCLUDE_TIES)
    assert_true(ast.exprs[Int(frame)].a != NO_NODE)
    assert_equal(ast.exprs[Int(frame)].b, NO_NODE)


def test_the_four_frame_tags_pack_and_come_back() raises:
    # Four tags share one field, so the packing is worth a test of its own
    # rather than only being checked through a query that happens to use it.
    var tags = frame_tags(
        FRAME_GROUPS,
        BOUND_UNBOUNDED_PRECEDING,
        BOUND_UNBOUNDED_FOLLOWING,
        EXCLUDE_TIES,
    )
    assert_equal(frame_mode(tags), FRAME_GROUPS)
    assert_equal(frame_start(tags), BOUND_UNBOUNDED_PRECEDING)
    assert_equal(frame_end(tags), BOUND_UNBOUNDED_FOLLOWING)
    assert_equal(frame_exclude(tags), EXCLUDE_TIES)

    var plain = frame_tags(
        FRAME_RANGE, BOUND_CURRENT_ROW, BOUND_NONE, EXCLUDE_NONE
    )
    assert_equal(frame_mode(plain), FRAME_RANGE)
    assert_equal(frame_start(plain), BOUND_CURRENT_ROW)
    assert_equal(frame_end(plain), BOUND_NONE)
    assert_equal(frame_exclude(plain), EXCLUDE_NONE)


def test_the_builders_and_the_printer_agree() raises:
    # A window built by hand rather than parsed, because the binder will build
    # them that way and nothing else in here would catch a builder that put a
    # part in the wrong field.
    var g = Grammar()
    var ast = Ast()
    var partition = List[UInt32]()
    partition.append(ast.column(["a"]))
    var ordering = List[UInt32]()
    ordering.append(ast.order(ast.column(["b"])))
    var frame = ast.frame(
        FRAME_ROWS,
        BOUND_UNBOUNDED_PRECEDING,
        BOUND_CURRENT_ROW,
        EXCLUDE_TIES,
    )
    var window = ast.window(partition, ordering, frame)
    var definition = ast.window_definition("w", window)

    var items = List[UInt32]()
    items.append(ast.item(ast.call("rank", List[UInt32]())))
    var tables = List[UInt32]()
    tables.append(ast.table(["t"]))
    var windows = List[UInt32]()
    windows.append(definition)
    var built = ast.query(
        items,
        tables,
        NO_NODE,
        List[UInt32](),
        NO_NODE,
        NO_NODE,
        0,
        List[UInt32](),
        windows,
    )
    assert_equal(
        print_stmt(ast, built, g),
        (
            "SELECT rank() FROM t WINDOW w AS (PARTITION BY a ORDER BY b ROWS"
            " BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE TIES)"
        ),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
