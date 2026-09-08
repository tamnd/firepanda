"""Tests for the generated grammar table and the reader that loads it.

The generator has its own round trip check, in Python, which renders the table
back to PEG and compares it to the vendored files. This file is the other half:
it checks that the table survives the trip through a `StaticString` into Mojo
with every index still pointing where it did, because a table that loads without
complaint and has one node index off by one is a parser that quietly accepts a
different language.

So most of what is here walks the whole table rather than spot checking. There
are 4,422 nodes and checking all of them costs a millisecond, and the specific
rules asserted at the bottom are the ones a person would notice were wrong.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.sql import Grammar, memoized_rules, overridden_rules
from firepanda.sql.generated.keywords import (
    KEYWORD_COLUMN_NAME,
    KEYWORD_COUNT,
    KEYWORD_FUNC_NAME,
    KEYWORD_MAX_LENGTH,
    KEYWORD_RESERVED,
    KEYWORD_TYPE_NAME,
    KEYWORD_UNRESERVED,
)
from firepanda.sql.generated.rules import (
    FLAG_WORD,
    MATCHER_COUNT,
    MATCHER_IDENTIFIER,
    MATCHER_NONE,
    MATCHER_NUMBER_LITERAL,
    MATCHER_OPERATOR,
    MATCHER_RESERVED_IDENTIFIER,
    MATCHER_STRING_LITERAL,
    MEMOIZED_COUNT,
    NODE_CAPTURE,
    NODE_CHOICE,
    NODE_CLASS,
    NODE_COUNT,
    NODE_KEYWORDS,
    NODE_LIT,
    NODE_NOT,
    NODE_OPT,
    NODE_PLUS,
    NODE_REF,
    NODE_SEQ,
    NODE_STAR,
    OVERRIDDEN_COUNT,
    RULE_COUNT,
    RULE_END_OF_INPUT,
    RULE_WHITESPACE,
    STRING_COUNT,
    SUGGEST_NONE,
    SUGGEST_SCALAR_FUNCTION_NAME,
    SUGGEST_TABLE_NAME,
    SUGGEST_TYPE_NAME,
    SUGGEST_VARIABLE,
    SUGGESTION_COUNT,
)
from firepanda.sql.table import _Reader


def test_the_table_loads_and_the_counts_agree() raises:
    var g = Grammar()
    assert_equal(len(g.nodes), NODE_COUNT)
    assert_equal(len(g.strings), STRING_COUNT)
    assert_equal(len(g.names), RULE_COUNT)
    assert_equal(len(g.roots), RULE_COUNT)
    assert_equal(len(g.memoized), RULE_COUNT)
    assert_equal(len(g.matchers), RULE_COUNT)
    assert_equal(len(g.suggestions), RULE_COUNT)
    assert_equal(len(g.keywords), KEYWORD_COUNT)
    assert_equal(len(g.keyword_classes), KEYWORD_COUNT)


def test_node_zero_is_the_null_node() raises:
    # Every absent child and every absent sibling points here, so it has to stay
    # at index 0 and stay empty. If the generator ever emits a real node first,
    # half the tree silently gains a child.
    var g = Grammar()
    assert_equal(Int(g.nodes[0].kind), 0)
    assert_equal(Int(g.nodes[0].child), 0)
    assert_equal(Int(g.nodes[0].sibling), 0)


def test_every_index_in_the_table_is_in_range() raises:
    var g = Grammar()
    for i in range(1, len(g.nodes)):
        var node = g.nodes[i]
        assert_true(
            Int(node.child) < len(g.nodes), String("child out of range at ", i)
        )
        assert_true(
            Int(node.sibling) < len(g.nodes),
            String("sibling out of range at ", i),
        )
        if (
            node.kind == NODE_LIT
            or node.kind == NODE_CLASS
            or node.kind == NODE_CAPTURE
        ):
            assert_true(
                Int(node.payload) < len(g.strings),
                String("string out of range at ", i),
            )
        elif node.kind == NODE_REF:
            # A reference resolves to a rule, or to EndOfInput, which the
            # matcher supplies and which gets the index one past the last rule.
            assert_true(
                Int(node.payload) <= RULE_END_OF_INPUT,
                String("reference out of range at ", i),
            )

    for i in range(len(g.roots)):
        assert_true(Int(g.roots[i]) > 0, String("rule ", i, " has no root"))
        assert_true(
            Int(g.roots[i]) < len(g.nodes), String("root out of range at ", i)
        )


def test_every_node_kind_is_one_the_reader_knows() raises:
    var g = Grammar()
    for i in range(1, len(g.nodes)):
        var kind = g.nodes[i].kind
        assert_true(
            kind == NODE_SEQ
            or kind == NODE_CHOICE
            or kind == NODE_OPT
            or kind == NODE_STAR
            or kind == NODE_PLUS
            or kind == NODE_NOT
            or kind == NODE_REF
            or kind == NODE_LIT
            or kind == NODE_CLASS
            or kind == NODE_CAPTURE
            or kind == NODE_KEYWORDS,
            String("unknown kind ", Int(kind), " at node ", i),
        )


def test_the_shape_of_a_node_matches_its_kind() raises:
    # A leaf with children, or a repeat with two, is a generator bug that would
    # otherwise show up as a mysterious parse failure much later.
    var g = Grammar()
    for i in range(1, len(g.nodes)):
        var kind = g.nodes[i].kind
        var children = len(g.children(i))
        if (
            kind == NODE_REF
            or kind == NODE_LIT
            or kind == NODE_CLASS
            or kind == NODE_CAPTURE
            or kind == NODE_KEYWORDS
        ):
            assert_equal(children, 0, String("leaf with children at ", i))
        elif (
            kind == NODE_OPT
            or kind == NODE_STAR
            or kind == NODE_PLUS
            or kind == NODE_NOT
        ):
            assert_equal(
                children, 1, String("suffix with ", children, " at ", i)
            )
        else:
            assert_true(
                children >= 2, String("group with ", children, " at ", i)
            )


def test_every_node_is_reachable_from_exactly_one_rule() raises:
    # The nodes are one array shared by 1,187 trees. An unreferenced node means
    # the generator wrote something it then dropped, and a node reached twice
    # means two rules share a subtree, which the flattener is not supposed to do.
    var g = Grammar()
    var seen = List[Int](length=len(g.nodes), fill=0)
    for r in range(len(g.roots)):
        var stack = List[Int]()
        stack.append(Int(g.roots[r]))
        while len(stack) > 0:
            var node = stack.pop()
            seen[node] += 1
            for child in g.children(node):
                stack.append(child)

    for i in range(1, len(g.nodes)):
        assert_equal(
            seen[i], 1, String("node ", i, " reached ", seen[i], " times")
        )


def test_rule_names_are_unique_and_sorted_after_the_first() raises:
    var g = Grammar()
    # %whitespace comes first because the generator loads common.gram first, and
    # everything after it is sorted, which is what keeps the diff of a grammar
    # bump readable.
    assert_equal(g.names[RULE_WHITESPACE], "%whitespace")
    for i in range(2, len(g.names)):
        assert_true(
            g.names[i - 1] < g.names[i],
            String("names out of order at ", i, ": ", g.names[i]),
        )


def test_a_rule_can_be_found_by_name() raises:
    var g = Grammar()
    assert_equal(g.names[g.rule("SelectStatement")], "SelectStatement")
    assert_equal(g.rule("NoSuchRuleExists"), -1)


def test_rules_render_back_as_the_grammar_wrote_them() raises:
    # Read these against firepanda/sql/grammar/statements/. They are here so that
    # a bump that changes them has to be looked at rather than rubber stamped,
    # and they cover a reference, a sequence, a suffix, a lookahead, a capture
    # and an expanded parameterized rule.
    var g = Grammar()
    assert_equal(
        g.write_rule(g.rule("SelectStatement")), "SelectStatementInternal"
    )
    assert_equal(
        g.write_rule(g.rule("SelectStatementInternal")),
        "(WithClause? SelectSetOpChain ResultModifiers?)",
    )
    assert_equal(
        g.write_rule(g.rule("PlainIdentifier")),
        "(!ReservedKeyword <[a-z_]i[a-z0-9_]i*>)",
    )
    assert_equal(g.write_rule(RULE_WHITESPACE), "[ \\t\\n\\r]*")


def test_parameterized_rules_are_expanded_at_generation_time() raises:
    # List(D) <- D (',' D)* ','? in the grammar. The matcher has no environment
    # to thread a parameter through, so each call site gets its own rule.
    var g = Grammar()
    var index = g.rule("List_Expression")
    assert_true(index >= 0, "List(Expression) was not expanded")
    assert_equal(g.write_rule(index), "(Expression (',' Expression)* ','?)")


def test_the_keyword_classes_are_reachable_as_rules() raises:
    var g = Grammar()
    assert_equal(g.write_rule(g.rule("ReservedKeyword")), "<keywords 1>")
    assert_equal(g.write_rule(g.rule("UnreservedKeyword")), "<keywords 2>")
    assert_equal(g.write_rule(g.rule("ColumnNameKeyword")), "<keywords 4>")
    assert_equal(g.write_rule(g.rule("FuncNameKeyword")), "<keywords 8>")
    assert_equal(g.write_rule(g.rule("TypeNameKeyword")), "<keywords 16>")


def test_word_literals_are_flagged_and_upper_cased() raises:
    # The flag is what stops SELECT matching the front of SELECTED. A literal
    # that is punctuation must not carry it, or every operator becomes a word.
    var g = Grammar()
    var words = 0
    for i in range(1, len(g.nodes)):
        var node = g.nodes[i]
        if node.kind != NODE_LIT:
            assert_equal(
                Int(node.flags), 0, String("flag on a non literal at ", i)
            )
            continue
        var text = g.strings[Int(node.payload)]
        if node.flags & FLAG_WORD:
            words += 1
            assert_true(
                text.byte_length() > 0, String("empty word literal at ", i)
            )
            assert_equal(
                text, text.upper(), String("word not upper case at ", i)
            )
    # Most keywords reach the matcher through the five keyword lists rather than
    # as literals, so this is a floor and not a count.
    assert_true(words > 500, String("only ", words, " word literals"))


def test_the_memoized_rules_are_duckdbs_list() raises:
    var g = Grammar()
    var rules = memoized_rules(g)
    assert_equal(len(rules), MEMOIZED_COUNT)
    assert_equal(len(rules), 22)
    # The whole list is the expression chain, which is where a PEG parser
    # without memoization goes quadratic. See docs/specs/sql/04-the-parser.md.
    for index in rules:
        assert_true(
            g.names[index].endswith("Expression")
            or g.names[index] == "Identifier"
            or g.names[index] == "ColId"
            or g.names[index] == "ColumnReference",
            String("unexpected memoized rule ", g.names[index]),
        )
    assert_true(g.memoized[g.rule("Expression")], "Expression is not memoized")
    assert_true(
        g.memoized[g.rule("FunctionExpression")],
        "FunctionExpression is not memoized",
    )
    assert_false(g.memoized[g.rule("SelectStatement")])


def test_the_overridden_rules_are_duckdbs_list() raises:
    var g = Grammar()
    var rules = overridden_rules(g)
    assert_equal(len(rules), OVERRIDDEN_COUNT)
    assert_equal(len(rules), 24)
    # Every one of them is a name or a literal, which is the whole point: these
    # are the places where the grammar stops describing characters and hands the
    # job to the tokenizer. See docs/specs/sql/04-the-parser.md section 2.
    assert_equal(g.matchers[g.rule("OperatorLiteral")], MATCHER_OPERATOR)
    assert_equal(g.matchers[g.rule("NumberLiteral")], MATCHER_NUMBER_LITERAL)
    assert_equal(g.matchers[g.rule("StringLiteral")], MATCHER_STRING_LITERAL)
    assert_equal(g.matchers[g.rule("Identifier")], MATCHER_IDENTIFIER)
    assert_equal(g.matchers[g.rule("TableName")], MATCHER_IDENTIFIER)
    assert_equal(
        g.matchers[g.rule("ReservedIdentifier")], MATCHER_RESERVED_IDENTIFIER
    )
    # ReservedKeyword is the one row where grammar_types.yml and the parser
    # disagree. The yml says identifier_string, which is a transformer result
    # type, and compiled_grammar.cpp builds a ReservedIdentifierMatcher. The
    # C++ is what runs, so the C++ is what the vendor script reads.
    assert_equal(
        g.matchers[g.rule("ReservedKeyword")], MATCHER_RESERVED_IDENTIFIER
    )
    # The rules that carry no override are the overwhelming majority, and a
    # generator bug that set the column on everything would still pass the count
    # check above if the count moved with it.
    assert_equal(g.matchers[g.rule("SelectStatement")], MATCHER_NONE)
    assert_equal(g.matchers[g.rule("Expression")], MATCHER_NONE)


def test_the_suggestions_say_which_names_each_position_takes() raises:
    # The suggestion is not decoration. IdentifierMatcher asks it which keyword
    # class the position tolerates and whether a single quoted string is a name
    # there, so TableName and TypeName take different words even though both
    # run the same matcher.
    var g = Grammar()
    assert_equal(g.suggestions[g.rule("TableName")], SUGGEST_TABLE_NAME)
    assert_equal(g.suggestions[g.rule("TypeName")], SUGGEST_TYPE_NAME)
    assert_equal(
        g.suggestions[g.rule("FunctionName")], SUGGEST_SCALAR_FUNCTION_NAME
    )
    assert_equal(g.suggestions[g.rule("Identifier")], SUGGEST_VARIABLE)
    # The three matchers that take no suggestion, and every rule with no
    # override at all, read zero.
    assert_equal(g.suggestions[g.rule("NumberLiteral")], SUGGEST_NONE)
    assert_equal(g.suggestions[g.rule("StringLiteral")], SUGGEST_NONE)
    assert_equal(g.suggestions[g.rule("OperatorLiteral")], SUGGEST_NONE)
    assert_equal(g.suggestions[g.rule("SelectStatement")], SUGGEST_NONE)


def test_only_an_overridden_rule_carries_a_suggestion() raises:
    var g = Grammar()
    for i in range(len(g.matchers)):
        assert_true(
            Int(g.suggestions[i]) < SUGGESTION_COUNT,
            String("rule ", g.names[i], " names an unknown suggestion"),
        )
        if g.matchers[i] == MATCHER_NONE:
            assert_equal(
                g.suggestions[i],
                SUGGEST_NONE,
                String(g.names[i], " has a suggestion but no matcher"),
            )


def test_an_overridden_rule_keeps_the_body_the_grammar_gave_it() raises:
    # The bodies of these rules are placeholders that upstream's matcher never
    # reads, and the temptation is to drop them. They stay, because the
    # generator's round trip check is what proves we read the grammar text
    # correctly, and it can only check text that is still there. This is the
    # rule that would break first: `+` is not an identifier.
    var g = Grammar()
    assert_equal(g.write_rule(g.rule("OperatorLiteral")), "Identifier")


def test_no_rule_names_a_matcher_this_build_does_not_have() raises:
    var g = Grammar()
    for i in range(len(g.matchers)):
        assert_true(
            Int(g.matchers[i]) < MATCHER_COUNT,
            String("rule ", g.names[i], " names an unknown matcher"),
        )


def test_the_keyword_table_is_sorted_and_bisects() raises:
    var g = Grammar()
    for i in range(1, len(g.keywords)):
        assert_true(
            g.keywords[i - 1] < g.keywords[i],
            String("keywords out of order at ", i, ": ", g.keywords[i]),
        )
    # Every word has to be findable, because a keyword the bisection misses
    # becomes an identifier and the query parses as something else.
    for i in range(len(g.keywords)):
        assert_equal(
            g.keyword_class(g.keywords[i]),
            g.keyword_classes[i],
            String("bisection missed ", g.keywords[i]),
        )


def test_keywords_are_lower_case_and_within_the_declared_length() raises:
    var g = Grammar()
    for word in g.keywords:
        assert_equal(word, word.lower(), String(word, " is not lower case"))
        assert_true(word.byte_length() > 0, "empty keyword")
        assert_true(
            word.byte_length() <= KEYWORD_MAX_LENGTH,
            String(word, " is longer than KEYWORD_MAX_LENGTH"),
        )


def test_a_word_that_is_not_a_keyword_is_in_no_class() raises:
    var g = Grammar()
    assert_equal(g.keyword_class("frobnicate"), 0)
    assert_equal(g.keyword_class(""), 0)
    # Case folding is the caller's job, so an unfolded word is a miss rather
    # than a silent hit. The tokenizer lower cases before it asks.
    assert_equal(g.keyword_class("SELECT"), 0)


def test_the_classes_a_word_is_in() raises:
    var g = Grammar()
    assert_equal(g.keyword_class("select"), KEYWORD_RESERVED)
    assert_equal(g.keyword_class("abort"), KEYWORD_UNRESERVED)
    assert_equal(g.keyword_class("between"), KEYWORD_COLUMN_NAME)
    # 26 words are both, which is the reason for one table with a mask instead
    # of five tables.
    assert_equal(g.keyword_class("left"), KEYWORD_FUNC_NAME | KEYWORD_TYPE_NAME)
    assert_equal(
        g.keyword_class("binary"), KEYWORD_FUNC_NAME | KEYWORD_TYPE_NAME
    )


def test_a_truncated_table_is_an_error_rather_than_a_wrong_grammar() raises:
    # The reader is fed by a generator in this repository, so the only way it
    # sees bad input is a build that went wrong. It has to say so rather than
    # load half a grammar.
    var reader = _Reader(StaticString("N 2\n1 0 0 0 0\n").as_bytes())
    assert_equal(reader.section(UInt8(ord("N"))), 2)
    _ = reader.number()
    _ = reader.number()
    _ = reader.number()
    _ = reader.number()
    _ = reader.number()
    reader.end_of_line()
    with assert_raises(contains="expected a number"):
        _ = reader.number()


def test_a_section_header_that_is_not_there_is_an_error() raises:
    var reader = _Reader(StaticString("\nX 3\n").as_bytes())
    with assert_raises(contains="expected a section header"):
        _ = reader.section(UInt8(ord("N")))


def test_trailing_text_on_a_record_is_an_error() raises:
    var reader = _Reader(StaticString("N 3 and then some\n").as_bytes())
    with assert_raises(contains="unread text"):
        _ = reader.section(UInt8(ord("N")))


def test_a_length_prefixed_string_keeps_its_spaces() raises:
    # A literal in the grammar can be a single space, which is why the string
    # section is length prefixed rather than split on the separator.
    var reader = _Reader(StaticString("3  a \n").as_bytes())
    var length = reader.number()
    assert_equal(length, 3)
    assert_equal(reader.text(length), " a ")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
