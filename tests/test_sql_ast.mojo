"""The expression AST and the printer that turns it back into SQL.

Three kinds of test are in here and they check different things.

The arena tests are about the layout the rest of the engine is going to rely on:
that index 0 is the null node, that an empty run is run 0, that interning the
same text twice gives back the same index, and that text stays readable after
the pool has grown. None of those are interesting until something gets one of
them wrong, at which point every one of them is a silent corruption rather than
a crash.

The printer tests are about the text. Each one names the SQL it is producing, so
a failure reads as a disagreement about SQL rather than about field numbering.

The round trip tests are the ones with teeth. They print an expression, put it
in a `SELECT`, and parse it with the real parser. That is the property the
printer exists to have, and it is the property the transformer will lean on for
its own tests, so it gets checked here before anything depends on it.

See docs/specs/sql/05-ast-and-binder.md sections 2 and 6.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.sql import Grammar, parse
from firepanda.sql.ast import (
    Ast,
    Expr,
    CALL_DISTINCT,
    CALL_STAR,
    EXPR_STAR,
    LITERAL_BOOLEAN,
    LITERAL_NULL,
    LITERAL_NUMBER,
    LITERAL_STRING,
    NO_NODE,
)
from firepanda.sql.printer import (
    needs_quoting,
    print_expr,
    quote_name,
    quote_string,
)


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


def _round_trips(ast: Ast, node: UInt32, g: Grammar) raises -> String:
    """Prints a node and checks the text parses inside a statement.

    `SELECT <expr> FROM t` rather than `SELECT <expr>`, because a star wants a
    table to come from and every other expression is happy either way.

    Args:
        ast: The AST.
        node: The node.
        g: A loaded grammar.

    Returns:
        The printed expression, so a caller can check the text as well.

    Raises:
        Error: If the printed text did not parse.
    """
    var text = print_expr(ast, node, g)
    var statement = String("SELECT ", text, " FROM t")
    try:
        _ = parse(statement, g)
    except e:
        raise Error(
            String(
                "the printer wrote something that will not parse: ",
                statement,
                "\n",
                e,
            )
        )
    return text^


# ---------------------------------------------------------------------------
# The arenas
# ---------------------------------------------------------------------------


def test_index_zero_is_the_null_node() raises:
    # Every consumer is going to read a missing operand as 0, so there has to be
    # a node at 0 for the arena to be indexable at all, and nothing may ever be
    # handed 0 as a real node.
    var ast = Ast()
    assert_equal(len(ast.exprs), 1)
    assert_equal(Int(ast.exprs[0].kind), 0)
    assert_equal(Int(NO_NODE), 0)
    assert_true(ast.literal(LITERAL_NUMBER, "1") != NO_NODE)


def test_an_empty_run_is_run_zero() raises:
    var ast = Ast()
    assert_equal(ast.run(List[UInt32]()), NO_NODE)
    assert_equal(ast.length(NO_NODE), 0)
    assert_equal(len(ast.items(NO_NODE)), 0)


def test_a_run_is_a_count_and_then_its_entries() raises:
    var ast = Ast()
    var run = ast.run([7, 8, 9])
    assert_true(run != NO_NODE)
    assert_equal(ast.length(run), 3)
    assert_equal(Int(ast.at(run, 0)), 7)
    assert_equal(Int(ast.at(run, 2)), 9)

    # Two runs in the same side list do not read each other's entries.
    var second = ast.run([1, 2])
    assert_equal(ast.length(run), 3)
    assert_equal(ast.length(second), 2)
    assert_equal(Int(ast.at(second, 1)), 2)


def test_the_same_text_is_interned_once() raises:
    # A real query says the same column name in the select list, the group by
    # and the order by, so this is the common case rather than a nicety.
    var ast = Ast()
    var first = ast.intern("total_amount")
    var again = ast.intern("total_amount")
    var other = ast.intern("order_id")
    assert_equal(first, again)
    assert_true(first != other)
    assert_equal(len(ast.strings), 3)


def test_the_empty_string_is_pool_index_zero() raises:
    var ast = Ast()
    assert_equal(Int(ast.intern("")), 0)
    assert_equal(ast.text(0).byte_length(), 0)


def test_text_stays_readable_after_the_pool_grows() raises:
    # `text` gives back a reference rather than a slice for exactly this reason.
    # A short string keeps its bytes inside itself, so a slice into one would be
    # pointing at freed memory the moment the pool reallocated.
    var ast = Ast()
    var first = ast.intern("a")
    for i in range(1000):
        _ = ast.intern(String("column_", i))
    assert_equal(ast.text(first), "a")


# ---------------------------------------------------------------------------
# Printing, one kind at a time
# ---------------------------------------------------------------------------


def test_a_literal_prints_as_sql_spells_it() raises:
    var g = Grammar()
    var ast = Ast()
    assert_equal(print_expr(ast, ast.literal(LITERAL_NULL, ""), g), "NULL")
    assert_equal(
        print_expr(ast, ast.literal(LITERAL_BOOLEAN, "TRUE"), g), "TRUE"
    )
    assert_equal(print_expr(ast, ast.literal(LITERAL_NUMBER, "42"), g), "42")
    assert_equal(
        print_expr(ast, ast.literal(LITERAL_STRING, "hello"), g), "'hello'"
    )


def test_a_string_literal_gets_its_quotes_doubled() raises:
    # The AST holds the decoded value, so the printer is the thing that has to
    # put an escape back, and a doubled quote is the form that needs no prefix.
    var g = Grammar()
    var ast = Ast()
    assert_equal(
        print_expr(ast, ast.literal(LITERAL_STRING, "it's"), g), "'it''s'"
    )
    assert_equal(quote_string(""), "''")
    assert_equal(quote_string("'"), "''''")


def test_a_column_prints_every_part_it_has() raises:
    var g = Grammar()
    var ast = Ast()
    assert_equal(print_expr(ast, ast.column(_parts("a")), g), "a")
    assert_equal(print_expr(ast, ast.column(_parts("s", "t", "c")), g), "s.t.c")


def test_a_column_with_no_parts_is_an_error() raises:
    var g = Grammar()
    var ast = Ast()
    var empty = ast.column(List[String]())
    with assert_raises(contains="no name parts"):
        _ = print_expr(ast, empty, g)


def test_a_star_prints_its_qualifier_and_its_exclusions() raises:
    var g = Grammar()
    var ast = Ast()
    assert_equal(print_expr(ast, ast.star(), g), "*")
    assert_equal(print_expr(ast, ast.star(qualifier=_parts("t")), g), "t.*")
    assert_equal(
        print_expr(ast, ast.star(exclude=_parts("a", "b")), g),
        "* EXCLUDE (a, b)",
    )


def test_a_star_prints_replace_and_rename() raises:
    # `star` does not build these two, because one takes expressions and the
    # other takes name pairs, so they are assembled the way the transformer will
    # assemble them.
    var g = Grammar()
    var ast = Ast()
    var sum = ast.binary(
        "+", ast.column(_parts("x")), ast.literal(LITERAL_NUMBER, "1")
    )
    var replace = ast.run([ast.intern("b"), sum])
    var rename = ast.run([ast.intern("c"), ast.intern("d")])
    var node = ast.add(Expr(kind=EXPR_STAR, token=0, b=replace, payload=rename))
    assert_equal(
        print_expr(ast, node, g), "* REPLACE ((x + 1) AS b) RENAME (c AS d)"
    )


def test_a_call_prints_its_name_before_its_arguments() raises:
    var g = Grammar()
    var ast = Ast()
    var args: List[UInt32] = [
        ast.column(_parts("a")),
        ast.literal(LITERAL_NUMBER, "2"),
    ]
    assert_equal(print_expr(ast, ast.call("round", args), g), "round(a, 2)")
    assert_equal(print_expr(ast, ast.call("now", List[UInt32]()), g), "now()")


def test_a_call_prints_its_flags() raises:
    var g = Grammar()
    var ast = Ast()
    assert_equal(
        print_expr(ast, ast.call("count", List[UInt32](), CALL_STAR), g),
        "count(*)",
    )
    var one: List[UInt32] = [ast.column(_parts("a"))]
    assert_equal(
        print_expr(ast, ast.call("count", one, CALL_DISTINCT), g),
        "count(DISTINCT a)",
    )


def test_an_operator_prints_where_sql_puts_it() raises:
    # This is the whole reason an operator is not a call. The printer has to
    # know that `+` goes between its operands and `round` goes in front.
    var g = Grammar()
    var ast = Ast()
    var a = ast.column(_parts("a"))
    var b = ast.column(_parts("b"))
    assert_equal(print_expr(ast, ast.binary("+", a, b), g), "(a + b)")
    assert_equal(print_expr(ast, ast.binary("AND", a, b), g), "(a AND b)")


def test_a_word_operator_gets_a_space_and_a_symbol_does_not() raises:
    var g = Grammar()
    var ast = Ast()
    var a = ast.column(_parts("a"))
    assert_equal(print_expr(ast, ast.unary("NOT", a), g), "(NOT a)")
    assert_equal(print_expr(ast, ast.unary("-", a), g), "(-a)")


def test_every_operand_is_parenthesized() raises:
    # The printer does not work out which parentheses it could leave out, and
    # this is why: the two trees below are different, and the parentheses are
    # the only thing in the text that says so.
    var g = Grammar()
    var ast = Ast()
    var a = ast.column(_parts("a"))
    var b = ast.column(_parts("b"))
    var c = ast.column(_parts("c"))
    assert_equal(
        print_expr(ast, ast.binary("*", ast.binary("+", a, b), c), g),
        "((a + b) * c)",
    )
    assert_equal(
        print_expr(ast, ast.binary("+", a, ast.binary("*", b, c)), g),
        "(a + (b * c))",
    )


def test_a_cast_prints_the_type_as_it_was_written() raises:
    var g = Grammar()
    var ast = Ast()
    var a = ast.column(_parts("a"))
    assert_equal(
        print_expr(ast, ast.cast(a, "INTEGER"), g), "CAST(a AS INTEGER)"
    )
    assert_equal(
        print_expr(ast, ast.cast(a, "DECIMAL(10, 2)", tries=True), g),
        "TRY_CAST(a AS DECIMAL(10, 2))",
    )


def test_a_searched_case_prints_its_arms_in_order() raises:
    var g = Grammar()
    var ast = Ast()
    var zero = ast.literal(LITERAL_NUMBER, "0")
    var a = ast.column(_parts("a"))
    var arms: List[UInt32] = [
        ast.binary(">", a, zero),
        ast.literal(LITERAL_STRING, "up"),
        ast.binary("<", a, zero),
        ast.literal(LITERAL_STRING, "down"),
    ]
    var node = ast.case(arms, ast.literal(LITERAL_STRING, "flat"))
    assert_equal(
        print_expr(ast, node, g),
        "CASE WHEN (a > 0) THEN 'up' WHEN (a < 0) THEN 'down' ELSE 'flat' END",
    )


def test_a_simple_case_prints_its_operand_after_the_keyword() raises:
    var g = Grammar()
    var ast = Ast()
    var arms: List[UInt32] = [
        ast.literal(LITERAL_NUMBER, "1"),
        ast.literal(LITERAL_STRING, "one"),
    ]
    var node = ast.case(arms, operand=ast.column(_parts("a")))
    assert_equal(print_expr(ast, node, g), "CASE a WHEN 1 THEN 'one' END")


def test_a_case_wants_an_even_number_of_arm_entries() raises:
    var ast = Ast()
    with assert_raises(contains="even number"):
        _ = ast.case([ast.literal(LITERAL_NUMBER, "1")])
    with assert_raises(contains="at least two"):
        _ = ast.case(List[UInt32]())


def test_between_stays_one_node_and_prints_as_one() raises:
    # Rewriting it into two comparisons here would evaluate the operand twice
    # and print back something nobody wrote.
    var g = Grammar()
    var ast = Ast()
    var a = ast.column(_parts("a"))
    var low = ast.literal(LITERAL_NUMBER, "1")
    var high = ast.literal(LITERAL_NUMBER, "9")
    assert_equal(
        print_expr(ast, ast.between(a, low, high), g), "(a BETWEEN 1 AND 9)"
    )
    assert_equal(
        print_expr(ast, ast.between(a, low, high, negated=True), g),
        "(a NOT BETWEEN 1 AND 9)",
    )


def test_in_prints_its_candidates() raises:
    var g = Grammar()
    var ast = Ast()
    var a = ast.column(_parts("a"))
    var candidates: List[UInt32] = [
        ast.literal(LITERAL_NUMBER, "1"),
        ast.literal(LITERAL_NUMBER, "2"),
    ]
    assert_equal(
        print_expr(ast, ast.in_list(a, candidates), g), "(a IN (1, 2))"
    )
    assert_equal(
        print_expr(ast, ast.in_list(a, candidates, negated=True), g),
        "(a NOT IN (1, 2))",
    )


def test_a_list_prints_in_brackets() raises:
    var g = Grammar()
    var ast = Ast()
    var elements: List[UInt32] = [
        ast.literal(LITERAL_NUMBER, "1"),
        ast.literal(LITERAL_NUMBER, "2"),
    ]
    assert_equal(print_expr(ast, ast.list_of(elements), g), "[1, 2]")
    assert_equal(print_expr(ast, ast.list_of(List[UInt32]()), g), "[]")


def test_a_struct_prints_its_field_names_as_strings() raises:
    # A field name is not an expression, so it is stored as text and printed as
    # a string literal rather than going through the literal case.
    var g = Grammar()
    var ast = Ast()
    var values: List[UInt32] = [
        ast.literal(LITERAL_NUMBER, "1"),
        ast.literal(LITERAL_STRING, "x"),
    ]
    assert_equal(
        print_expr(ast, ast.struct_of(_parts("a", "b"), values), g),
        "{'a': 1, 'b': 'x'}",
    )


def test_a_struct_wants_one_value_per_name() raises:
    var ast = Ast()
    var one: List[UInt32] = [ast.literal(LITERAL_NUMBER, "1")]
    with assert_raises(contains="one value per field name"):
        _ = ast.struct_of(_parts("a", "b"), one)


def test_collate_prints_after_its_operand() raises:
    var g = Grammar()
    var ast = Ast()
    assert_equal(
        print_expr(ast, ast.collate(ast.column(_parts("a")), "nocase"), g),
        "(a COLLATE nocase)",
    )


def test_a_parameter_prints_the_form_it_was_written_in() raises:
    # The two forms are not interchangeable in DuckDB, so the sigil is stored
    # rather than reconstructed from whether there is a name.
    var g = Grammar()
    var ast = Ast()
    assert_equal(print_expr(ast, ast.parameter("?"), g), "?")
    assert_equal(print_expr(ast, ast.parameter("$", "1"), g), "$1")
    assert_equal(print_expr(ast, ast.parameter("$", "name"), g), "$name")


def test_the_null_node_is_not_printable() raises:
    var g = Grammar()
    var ast = Ast()
    with assert_raises(contains="null node"):
        _ = print_expr(ast, NO_NODE, g)


# ---------------------------------------------------------------------------
# Quoting
# ---------------------------------------------------------------------------


def test_a_plain_name_is_left_alone() raises:
    var g = Grammar()
    assert_equal(quote_name("total", g), "total")
    assert_equal(quote_name("_x9", g), "_x9")


def test_a_name_that_would_not_read_back_is_quoted() raises:
    # The tokenizer folds an unquoted word to lower case, so a capital in a name
    # can only have arrived quoted, and printing it bare would rename the column.
    var g = Grammar()
    assert_equal(quote_name("Total", g), '"Total"')
    assert_equal(quote_name("two words", g), '"two words"')
    assert_equal(quote_name("9lives", g), '"9lives"')
    assert_equal(quote_name("", g), '""')


def test_a_quote_inside_a_name_is_doubled() raises:
    var g = Grammar()
    assert_equal(quote_name('sa"id', g), '"sa""id"')


def test_a_reserved_keyword_is_quoted_and_an_unreserved_one_is_not() raises:
    # `select` cannot stand where a name is wanted and `name` can, which is a
    # question only the vendored keyword table can answer.
    var g = Grammar()
    assert_true(needs_quoting("select", g))
    assert_true(not needs_quoting("name", g))
    assert_equal(quote_name("select", g), '"select"')
    assert_equal(quote_name("name", g), "name")


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_printed_expressions_parse() raises:
    # The property the printer exists to have. Everything the transformer builds
    # is going to be checked this way, so the check itself is checked here.
    var g = Grammar()
    var ast = Ast()
    var a = ast.column(_parts("a"))
    var b = ast.column(_parts("b"))
    var one = ast.literal(LITERAL_NUMBER, "1")

    _ = _round_trips(ast, one, g)
    _ = _round_trips(ast, ast.literal(LITERAL_NULL, ""), g)
    _ = _round_trips(ast, ast.literal(LITERAL_BOOLEAN, "TRUE"), g)
    _ = _round_trips(ast, ast.literal(LITERAL_STRING, "it's"), g)
    _ = _round_trips(ast, ast.column(_parts("s", "t", "c")), g)
    _ = _round_trips(ast, ast.star(), g)
    _ = _round_trips(ast, ast.star(qualifier=_parts("t")), g)
    _ = _round_trips(ast, ast.star(exclude=_parts("a", "b")), g)
    _ = _round_trips(ast, ast.binary("*", ast.binary("+", a, b), one), g)
    _ = _round_trips(ast, ast.unary("NOT", a), g)
    _ = _round_trips(ast, ast.unary("-", one), g)
    _ = _round_trips(ast, ast.call("round", [a, one]), g)
    _ = _round_trips(ast, ast.call("now", List[UInt32]()), g)
    _ = _round_trips(ast, ast.call("count", List[UInt32](), CALL_STAR), g)
    _ = _round_trips(ast, ast.call("count", [a], CALL_DISTINCT), g)
    _ = _round_trips(ast, ast.cast(a, "INTEGER"), g)
    _ = _round_trips(ast, ast.cast(a, "DECIMAL(10, 2)", tries=True), g)
    _ = _round_trips(ast, ast.between(a, one, one), g)
    _ = _round_trips(ast, ast.between(a, one, one, negated=True), g)
    _ = _round_trips(ast, ast.in_list(a, [one, one]), g)
    _ = _round_trips(ast, ast.in_list(a, [one], negated=True), g)
    _ = _round_trips(ast, ast.list_of([one, one]), g)
    _ = _round_trips(ast, ast.list_of(List[UInt32]()), g)
    _ = _round_trips(ast, ast.collate(a, "nocase"), g)
    _ = _round_trips(ast, ast.parameter("?"), g)
    _ = _round_trips(ast, ast.parameter("$", "1"), g)
    _ = _round_trips(ast, ast.parameter("$", "name"), g)
    _ = _round_trips(ast, ast.struct_of(_parts("a"), [one]), g)
    _ = _round_trips(
        ast, ast.case([a, one], ast.literal(LITERAL_NUMBER, "0")), g
    )
    _ = _round_trips(ast, ast.case([one, one], operand=a), g)


def test_a_star_with_every_modifier_parses() raises:
    var g = Grammar()
    var ast = Ast()
    var sum = ast.binary(
        "+", ast.column(_parts("x")), ast.literal(LITERAL_NUMBER, "1")
    )
    var node = ast.add(
        Expr(
            kind=EXPR_STAR,
            token=0,
            a=ast.run([ast.intern("a")]),
            b=ast.run([ast.intern("b"), sum]),
            payload=ast.run([ast.intern("c"), ast.intern("d")]),
        )
    )
    assert_equal(
        _round_trips(ast, node, g),
        "* EXCLUDE (a) REPLACE ((x + 1) AS b) RENAME (c AS d)",
    )


def test_an_awkward_name_survives_the_round_trip() raises:
    # A name that needs quoting is the case where a printer that guessed would
    # produce text that parses and means something else, so it is checked
    # against the parser rather than only against a string.
    var g = Grammar()
    var ast = Ast()
    assert_equal(_round_trips(ast, ast.column(_parts("Total")), g), '"Total"')
    assert_equal(_round_trips(ast, ast.column(_parts("select")), g), '"select"')
    assert_equal(
        _round_trips(ast, ast.column(_parts("two words")), g), '"two words"'
    )
    assert_equal(_round_trips(ast, ast.column(_parts('sa"id')), g), '"sa""id"')
    assert_equal(
        _round_trips(ast, ast.column(_parts("select", "from")), g),
        '"select"."from"',
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
