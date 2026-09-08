"""The PEG matcher.

The first half of this file is a differential corpus. Every statement in it was
run against DuckDB 1.5.5 on an M4 through `json_serialize_sql`, and a statement
counts as rejected only when DuckDB said `syntax error`, because a binder error
means the parser was happy and the parser is all this file is about. The lists
are the answer DuckDB gave, not the answer we would like, so a disagreement here
is a compatibility bug and not a test to be edited.

The second half is about the shapes the matcher produces and the errors it
writes, which the corpus cannot see because it only records accept and reject.

See docs/specs/sql/04-the-parser.md sections 3 to 6.
"""

from std.bit import pop_count
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)
from firepanda.sql import Grammar
from firepanda.sql.matcher import (
    MAX_DEPTH,
    NO_NODE,
    Parse,
    parse,
    parse_rule,
    parse_unfiltered,
)
from firepanda.sql.token import TOKEN_END

comptime _OPEN = ~UInt64(0)
"""The filter word of a node that lets every token through."""


def accepted() -> List[StaticString]:
    """The statements DuckDB 1.5.5 parses.

    A function rather than a constant, because a list of string literals is a
    comptime value that will not materialize into a runtime one.

    Returns:
        The statements.
    """
    return [
        "SELECT 1",
        "SELECT 1;",
        "SELECT 1; SELECT 2;",
        "",
        "SELECT * FROM t",
        "SELECT a.b.c FROM x",
        "SELECT a AS b FROM t",
        "SELECT a b FROM t",
        "SELECT t.* FROM t",
        "SELECT COUNT(*) FROM t GROUP BY a HAVING COUNT(*) > 1",
        "SELECT a FROM t WHERE a IN (1,2,3)",
        "SELECT a FROM t WHERE a BETWEEN 1 AND 2",
        "SELECT CASE WHEN a THEN 1 ELSE 2 END FROM t",
        "SELECT a FROM t ORDER BY a DESC NULLS LAST LIMIT 10 OFFSET 5",
        "SELECT a FROM t1 JOIN t2 ON t1.a = t2.a",
        "SELECT a FROM t1 LEFT OUTER JOIN t2 USING (a)",
        "WITH x AS (SELECT 1) SELECT * FROM x",
        (
            "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x WHERE"
            " n<5) SELECT * FROM x"
        ),
        "SELECT a FROM t UNION SELECT b FROM u",
        "SELECT a, SUM(b) OVER (PARTITION BY c ORDER BY d) FROM t",
        "CREATE TABLE t (a INTEGER, b VARCHAR)",
        "CREATE TABLE t AS SELECT 1",
        "CREATE OR REPLACE VIEW v AS SELECT 1",
        "INSERT INTO t VALUES (1, 'a')",
        "INSERT INTO t (a) SELECT 1",
        "UPDATE t SET a = 1 WHERE b = 2",
        "DELETE FROM t WHERE a = 1",
        "DROP TABLE IF EXISTS t",
        "ALTER TABLE t ADD COLUMN c INTEGER",
        "COPY t TO 'x.csv' (FORMAT CSV, HEADER)",
        "COPY t FROM 'x.csv'",
        "PRAGMA table_info('t')",
        "SET memory_limit = '1GB'",
        "EXPLAIN SELECT 1",
        "DESCRIBE SELECT 1",
        "SELECT * EXCLUDE (a) FROM t",
        "SELECT * REPLACE (a+1 AS a) FROM t",
        "SELECT COLUMNS(*) FROM t",
        "FROM t SELECT a",
        "FROM t",
        "SELECT [1,2,3]",
        "SELECT {'a': 1}",
        "SELECT a[1]",
        "SELECT a[1:2]",
        "SELECT list_transform([1,2], x -> x + 1)",
        "SELECT CAST(1 AS VARCHAR)",
        "SELECT 1::INT",
        "SELECT INTERVAL 1 DAY",
        "SELECT DATE '2020-01-01'",
        "SELECT TIMESTAMP '2020-01-01 00:00:00'",
        "SELECT a FROM t TABLESAMPLE 10%",
        "SELECT a, b, FROM t",
        "SELECT a FROM t WHERE a LIKE 'x%' ESCAPE '\\'",
        "SELECT a FROM t QUALIFY row_number() OVER () = 1",
        "PIVOT t ON a USING sum(b)",
        "SELECT 1 @@ 2",
        "SELECT 1 <-1",
        "SELECT 3 %% 2",
        "SELECT $1",
        "SELECT ?",
        "SELECT ?1",
        "SELECT #1 FROM (SELECT 42) t",
        "SELECT * FROM 'x.parquet'",
        "SELECT db.select FROM t",
        "SELECT",
        "SELECT (((1)))",
        "attach 'x.db'",
        "CREATE MACRO m(a) AS a + 1",
        "CREATE INDEX i ON t (a)",
        "BEGIN TRANSACTION",
        "COMMIT",
        "CHECKPOINT",
        "SELECT unnest([1,2])",
        "SELECT a FROM t ANTI JOIN u ON t.a=u.a",
        "SELECT a FROM t POSITIONAL JOIN u",
        "SELECT a FROM t ASOF JOIN u ON t.a >= u.a",
        "SELECT * FROM range(10)",
        "SELECT try_cast(1 AS INT)",
        "SELECT a NOT SIMILAR TO 'x'",
        "SELECT NOT a",
        "SELECT a IS DISTINCT FROM b",
        "SELECT a COLLATE NOCASE",
        "SELECT grouping_id(a) FROM t GROUP BY GROUPING SETS ((a))",
        "SELECT a FROM t GROUP BY ALL",
        "SELECT a FROM t ORDER BY ALL",
        "SELECT sum(a) FILTER (WHERE b > 1) FROM t",
        "SELECT string_agg(a, ',' ORDER BY b) FROM t",
        "SELECT * FROM t AS x(a, b)",
        "SELECT * FROM (VALUES (1),(2)) v(a)",
        "SELECT * FROM read_csv('x.csv', header=true)",
        "SELECT 1 WHERE false",
        "SELECT * FROM t SEMI JOIN u ON t.a=u.a",
        "SELECT a::INT[3]",
        "SELECT a::STRUCT(x INT)",
        "SELECT a::MAP(INT, INT)",
        "SELECT a.b[1].c",
        "SELECT 1 IS NOT NULL",
        "SELECT E'\\n'",
        "SELECT U&'\\0041'",
        "SELECT $tag$body$tag$",
        "SELECT 1 /* c */ + 2",
        "-- only a comment",
        "SELECT 1 -- trailing",
        "SELECT 1_000",
        "SELECT .5",
        "SELECT 1.",
        "SELECT 1e5",
        "SELECT 0x1F",
        "SELECT FROM t",
        "SELECT a FRM",
        "SELECT a,",
        "SELECT name FROM t",
        "SELECT type FROM t",
        "SELECT key FROM t",
        "CREATE TABLE t (name INTEGER)",
    ]


def rejected() -> List[StaticString]:
    """The statements DuckDB 1.5.5 answers with a syntax error.

    Returns:
        The statements.
    """
    return [
        "SELECT 'a' 'b'",
        "SELECT CAST(1 AS 'int')",
        "SELECT select FROM t",
        "CREATE TABLE t (a select)",
        "SELECT * FRM t",
        "SELECT 1 FROM",
        "SELECT ((1)",
        "SELECT :name",
        "SELECT offset",
        "SELECT offset FROM t",
        "CREATE TABLE t (offset INTEGER)",
        "SELECT a FROM t WHERE",
        "SELECT 1 +",
        "SELECT * FROM",
        "SELECT a FROM t GROUP",
        "SELECT 1 AS",
        "SELECT (1",
        "SELECT 1)",
        "SELECT a ORDER",
    ]


def _nest(depth: Int) -> String:
    """Builds `SELECT` wrapped in some number of parentheses.

    Args:
        depth: How many parentheses on each side.

    Returns:
        The query.
    """
    var out = String("SELECT ")
    for _ in range(depth):
        out += "("
    out += "1"
    for _ in range(depth):
        out += ")"
    return out^


def _tree_shapes() -> List[String]:
    """Statements whose trees are worth walking node by node.

    The last three are here for the memo table rather than for the grammar. A
    nested call parses its argument once as a type and once as an expression, so
    the second walk is a memo hit and the tree it hands back was built by an
    attempt that failed, which is the shape where a wrong answer would hide.

    Returns:
        The statements.
    """
    return [
        String("SELECT a, b FROM t WHERE a = 1"),
        String("SELECT a FROM t WHERE a = 1"),
        String("SELECT f(f(f(1)))"),
        String("SELECT f(f(1), f(1))"),
        String("SELECT [[[1]]], f(1) + f(1)"),
    ]


def _accepts(sql: StringSlice, g: Grammar) -> Bool:
    """Says whether a statement parses, throwing the tree away.

    Args:
        sql: The query.
        g: A loaded grammar.

    Returns:
        Whether it parsed.
    """
    try:
        _ = parse(sql, g)
        return True
    except:
        return False


def _outcome(p: Parse, g: Grammar) -> String:
    """Writes a whole parse down, so that two of them can be compared.

    Every field of every node, because the point of comparing is to catch a
    difference nobody thought to look for.

    Args:
        p: The parse.
        g: The grammar it was made against.

    Returns:
        One line per node, and the root last.
    """
    var out = String()
    for i in range(len(p.nodes)):
        var n = p.nodes[i]
        out += String(
            g.names[Int(n.rule)],
            " ",
            Int(n.token_start),
            " ",
            Int(n.token_end),
            " ",
            Int(n.first_child),
            " ",
            Int(n.next_sibling),
            "\n",
        )
    return out + String("root ", Int(p.root))


def _either_way(sql: StringSlice, g: Grammar) -> String:
    """Parses once with the filter and once without, and says what differed.

    Args:
        sql: The query.
        g: A loaded grammar.

    Returns:
        The empty string when the two runs agreed, and what they disagreed
        about otherwise.
    """
    var fast: String
    var slow: String
    try:
        fast = _outcome(parse(sql, g), g)
    except e:
        fast = String("error: ", e)
    try:
        slow = _outcome(parse_unfiltered(sql, g), g)
    except e:
        slow = String("error: ", e)
    if fast == slow:
        return String()
    return String(sql, "\n  filtered:   ", fast, "\n  unfiltered: ", slow)


def _rule_of(p: Parse, node: UInt32, g: Grammar) -> String:
    """Names the rule a node matched.

    Args:
        p: The parse.
        node: The node index.
        g: The grammar the parse was made against.

    Returns:
        The rule name.
    """
    return g.names[Int(p.nodes[Int(node)].rule)]


# ---------------------------------------------------------------------------
# The differential corpus
# ---------------------------------------------------------------------------


def test_every_statement_duckdb_parses_parses_here() raises:
    var g = Grammar()
    for sql in accepted():
        var ok = True
        var why = String()
        try:
            _ = parse(sql, g)
        except e:
            ok = False
            why = String(e)
        assert_true(
            ok, String("rejected a statement DuckDB takes: ", sql, " | ", why)
        )


def test_every_statement_duckdb_refuses_is_refused_here() raises:
    var g = Grammar()
    for sql in rejected():
        assert_false(
            _accepts(sql, g),
            String("accepted a statement DuckDB calls a syntax error: ", sql),
        )


# ---------------------------------------------------------------------------
# The first token filter
# ---------------------------------------------------------------------------


def test_the_filter_changes_nothing_on_the_whole_corpus() raises:
    # The filter is the only part of the matcher that is allowed to be wrong in
    # a way the corpus cannot see, because a filter that is one bit too tight
    # rejects a statement that used to parse and nothing else changes. So every
    # statement runs twice, and the two runs have to agree node for node and
    # word for word.
    var g = Grammar()
    for sql in accepted():
        assert_equal(_either_way(sql, g), "")
    for sql in rejected():
        assert_equal(_either_way(sql, g), "")


def test_the_filter_says_no_to_most_nodes() raises:
    # A filter with every bit set everywhere would pass the test above and buy
    # nothing at all, so this is the one that says the table has content in it.
    var g = Grammar()
    var open = 0
    var narrow = 0
    var bits = 0
    for i in range(1, len(g.first)):
        if g.first[i] == _OPEN:
            open += 1
        else:
            narrow += 1
            bits += Int(pop_count(g.first[i]))
    assert_true(
        narrow > open,
        String(
            "only ",
            narrow,
            " of ",
            narrow + open,
            " nodes can be filtered out, which is too few to pay for the table",
        ),
    )
    assert_true(
        bits < narrow * 4,
        String(
            "a node that filters has ",
            bits // narrow,
            " of 64 bits set on average, which is too many to reject much",
        ),
    )


def test_a_rule_that_wants_one_keyword_says_so_in_one_bit() raises:
    # `Program <- TopLevelStatement*` matches the empty string, so it has to be
    # open or a parse that consumes nothing would be filtered away.
    # `CallStatement <- 'CALL' ...` can start with nothing but CALL, so it has
    # to be down to one bit, and that is what makes the choice over the eighty
    # kinds of statement cost eighty ands rather than eighty recursions.
    var g = Grammar()
    assert_equal(g.first[Int(g.roots[g.rule("Program")])], _OPEN)
    assert_equal(
        Int(pop_count(g.first[Int(g.roots[g.rule("CallStatement")])])), 1
    )


# ---------------------------------------------------------------------------
# The tree
# ---------------------------------------------------------------------------


def test_a_parse_is_rooted_at_program() raises:
    var g = Grammar()
    var p = parse("SELECT 1", g)
    assert_equal(_rule_of(p, p.root, g), "Program")


def test_the_root_covers_every_token_but_the_end() raises:
    # The end of input token is never consumed, because EndOfInput matches
    # without moving, so a whole statement covers one token less than the vector
    # holds.
    var g = Grammar()
    var p = parse("SELECT a FROM t", g)
    var root = p.nodes[Int(p.root)]
    assert_equal(Int(root.token_start), 0)
    assert_equal(Int(root.token_end), len(p.tokens) - 1)
    assert_equal(Int(p.tokens[len(p.tokens) - 1].kind), Int(TOKEN_END))


def test_two_statements_are_three_children_of_the_program() raises:
    # `TopLevelStatement <- Statement? (';'+ / EndOfInput)`, so the semicolon
    # after the last statement ends that statement and then end of input is
    # itself a third TopLevelStatement with no Statement in it. That is the tree
    # the grammar describes, and the transformer skips the empty one rather than
    # the matcher pretending it is not there.
    var g = Grammar()
    var p = parse("SELECT 1; SELECT 2;", g)
    var kids = p.children(p.root)
    assert_equal(len(kids), 3)
    for kid in kids:
        assert_equal(_rule_of(p, kid, g), "TopLevelStatement")
    var last = p.nodes[Int(kids[2])]
    assert_equal(last.token_start, last.token_end)


def test_an_empty_query_is_a_program_with_one_empty_statement() raises:
    # The empty string is a legal program, and for the same reason it is one
    # empty TopLevelStatement rather than none.
    var g = Grammar()
    var p = parse("   -- nothing here\n", g)
    var kids = p.children(p.root)
    assert_equal(len(kids), 1)
    assert_equal(p.nodes[Int(kids[0])].first_child, NO_NODE)


def test_a_child_is_always_built_before_its_parent() raises:
    # The arena invariant the whole matcher rests on. It holds over every node
    # and not just the reachable ones, because a failed attempt leaves its nodes
    # where they are for the memo table to point at, and a memo hit copies the
    # root of what it found rather than moving it.
    var g = Grammar()
    for sql in _tree_shapes():
        var p = parse(sql, g)
        for i in range(1, len(p.nodes)):
            var node = p.nodes[i]
            assert_true(
                Int(node.first_child) < i, "a node was built before its child"
            )
            assert_true(
                Int(node.next_sibling) > i or node.next_sibling == NO_NODE,
                "a node points backwards at a sibling",
            )


def test_a_nodes_span_covers_its_childrens() raises:
    var g = Grammar()
    for sql in _tree_shapes():
        var p = parse(sql, g)
        var stack = List[UInt32]()
        stack.append(p.root)
        while len(stack) > 0:
            var index = stack.pop()
            var node = p.nodes[Int(index)]
            for kid in p.children(index):
                var child = p.nodes[Int(kid)]
                assert_true(
                    child.token_start >= node.token_start
                    and child.token_end <= node.token_end,
                    "a child ran outside its parent",
                )
                stack.append(kid)


def test_a_replayed_subtree_is_the_one_a_fresh_parse_builds() raises:
    # A memo hit hands back a subtree somebody else built, so the shapes that
    # hit it a lot are the ones where a wrong answer would hide. Every node
    # under the root has to say the same thing about its own span as the tokens
    # under it do, and the root has to cover the statement.
    var g = Grammar()
    for sql in _tree_shapes():
        var p = parse(sql, g)
        var root = p.nodes[Int(p.root)]
        assert_equal(Int(root.token_start), 0, String("root of ", sql))
        assert_equal(
            Int(root.token_end), len(p.tokens) - 1, String("root of ", sql)
        )
        var stack = List[UInt32]()
        stack.append(p.root)
        while len(stack) > 0:
            var index = stack.pop()
            var node = p.nodes[Int(index)]
            var at = node.token_start
            for kid in p.children(index):
                var child = p.nodes[Int(kid)]
                assert_true(
                    child.token_start >= at,
                    String("children overlapped in ", sql),
                )
                at = child.token_end
                stack.append(kid)
            assert_true(
                at <= node.token_end,
                String("a child ran past its parent in ", sql),
            )


def test_parse_rule_asks_about_one_corner_of_the_grammar() raises:
    # This is how a test reaches a rule without wrapping it in a statement, and
    # it is what a fuzzer aims at one rule.
    var g = Grammar()
    var expression = -1
    for i in range(len(g.names)):
        if g.names[i] == "Expression":
            expression = i
    assert_true(expression >= 0, "the grammar lost its Expression rule")
    var p = parse_rule("1 + 2 * 3", g, expression)
    assert_equal(_rule_of(p, p.root, g), "Expression")
    with assert_raises():
        _ = parse_rule("1 +", g, expression)


# ---------------------------------------------------------------------------
# The rules that are matched from code
# ---------------------------------------------------------------------------


def test_a_quoted_string_is_a_name_only_where_a_table_goes() raises:
    # `FROM 'data.parquet'` has no grammar rule. It works because the table name
    # position is the one suggestion whose identifier matcher takes a single
    # quoted string, and `CAST(1 AS 'int')` is a syntax error for the same
    # reason.
    var g = Grammar()
    assert_true(_accepts("SELECT * FROM 'x.parquet'", g))
    assert_false(_accepts("SELECT CAST(1 AS 'int')", g))


def test_a_reserved_word_is_a_name_after_a_dot_and_not_before_one() raises:
    # Nine of the twenty four overridden rules run the reserved identifier
    # matcher, which drops the keyword check entirely. That is the whole of the
    # difference between these two.
    var g = Grammar()
    assert_true(_accepts("SELECT db.select FROM t", g))
    assert_false(_accepts("SELECT select FROM t", g))


def test_a_word_can_be_a_name_only_if_its_class_says_so() raises:
    # `name` is in the column name class and `offset` is not, so one of these is
    # a column and the other is a syntax error, and the only thing that decides
    # it is the vendored keyword list. Getting a word into the wrong class is
    # exactly the failure we promised never to have.
    var g = Grammar()
    assert_true(_accepts("SELECT name FROM t", g))
    assert_true(_accepts("CREATE TABLE t (name INTEGER)", g))
    assert_false(_accepts("SELECT offset FROM t", g))
    assert_false(_accepts("CREATE TABLE t (offset INTEGER)", g))


def test_an_operator_the_grammar_writes_out_is_not_a_catalog_lookup() raises:
    # OperatorLiteral has to refuse `<=`, because the grammar has its own node
    # for it and reading it as a catalog operator would take the wrong branch of
    # an ordered choice. `@@` is nobody's node, so it goes through.
    var g = Grammar()
    assert_true(_accepts("SELECT 1 @@ 2", g))
    assert_true(_accepts("SELECT 1 <= 2", g))
    assert_true(_accepts("SELECT 1 <-1", g))
    assert_true(_accepts("SELECT 3 %% 2", g))


def test_a_parameter_is_a_marker_and_then_a_name_or_a_number() raises:
    # All four parameter rules are two nodes, so the tokenizer hands the marker
    # back on its own and the matcher has a node to spend each token on.
    var g = Grammar()
    assert_true(_accepts("SELECT ?", g))
    assert_true(_accepts("SELECT ?1", g))
    assert_true(_accepts("SELECT $1", g))
    assert_true(_accepts("SELECT $name", g))
    # There is no colon form, whatever an earlier draft of the notes said.
    assert_false(_accepts("SELECT :name", g))


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_an_error_points_at_the_furthest_token_reached() raises:
    var g = Grammar()
    with assert_raises(contains='Parser Error: syntax error at or near "t"'):
        _ = parse("SELECT * FRM t", g)


def test_an_error_draws_a_caret_under_the_token() raises:
    var g = Grammar()
    var message = String()
    try:
        _ = parse("SELECT * FRM t", g)
    except e:
        message = String(e)
    var lines = message.split("\n")
    assert_equal(len(lines), 4)
    assert_equal(lines[1], "")
    assert_true(lines[2].startswith("LINE 1: "))
    # The caret sits under the first byte of the token it names, which is where
    # the `t` is once the LINE prefix is counted in.
    assert_equal(lines[3].find("^"), lines[2].find(" t") + 1)


def test_running_out_of_input_names_no_token() raises:
    # There is nothing to name and nothing to point at, so DuckDB drops the LINE
    # and the caret rather than pointing past the end, and so do we.
    var g = Grammar()
    var message = String()
    try:
        _ = parse("SELECT 1 FROM", g)
    except e:
        message = String(e)
    assert_equal(message, "Parser Error: syntax error at end of input")


def test_an_error_names_the_token_duckdb_names() raises:
    # The first line of each of these is what DuckDB 1.5.5 wrote for the same
    # query. It is the furthest position test with teeth: a terminal that fails
    # inside a negative lookahead failed on purpose and must not move the
    # furthest position, and when it does the message ends up naming whatever
    # the grammar was checking was absent instead of what a reader would point
    # at.
    var g = Grammar()
    var queries: List[StaticString] = [
        "SELECT * FRM t",
        "SELECT 1)",
        "SELECT offset FROM t",
        "SELECT :name",
        "CREATE TABLE t (offset INTEGER)",
        "SELECT a FROM t WHERE",
        "SELECT (1",
    ]
    var expected: List[StaticString] = [
        'Parser Error: syntax error at or near "t"',
        'Parser Error: syntax error at or near ")"',
        'Parser Error: syntax error at or near "FROM"',
        'Parser Error: syntax error at or near ":"',
        'Parser Error: syntax error at or near "offset"',
        "Parser Error: syntax error at end of input",
        "Parser Error: syntax error at end of input",
    ]
    for i in range(len(queries)):
        var message = String()
        try:
            _ = parse(queries[i], g)
        except e:
            message = String(e)
        assert_equal(message.split("\n")[0], String(expected[i]))


def test_nesting_deeper_than_the_guard_is_an_error_and_not_a_crash() raises:
    var g = Grammar()
    # Forty frames before any nesting and twenty one per parenthesis, so this is
    # comfortably over the guard whatever MAX_DEPTH is set to next.
    var deep = _nest(MAX_DEPTH // 21 + 8)
    with assert_raises(contains="memory exhausted at or near"):
        _ = parse(deep, g)


def test_nesting_the_guard_allows_still_parses() raises:
    var g = Grammar()
    assert_true(_accepts(_nest(MAX_DEPTH // 21 - 8), g))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
