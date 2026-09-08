"""The tokenizer.

Every awkward case here names the query that established it. They were run
against DuckDB 1.5.5 on an M4, because the vendored grammar does not describe
numbers, strings or operators at all: those four rules are in
firepanda/sql/grammar/matcher_overrides.list, which is DuckDB saying out loud
that its own matcher ignores the bodies. So the specification for this file is
behaviour, and a test that does not say which query it came from is a guess.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.sql import Grammar
from firepanda.sql.generated.keywords import (
    KEYWORD_RESERVED,
    KEYWORD_UNRESERVED,
)
from firepanda.sql.token import (
    FLAG_CONTINUED,
    FLAG_DECIMAL,
    FLAG_DOLLAR,
    FLAG_ESCAPE,
    FLAG_EXPONENT,
    FLAG_UNICODE,
    NO_KEYWORD,
    TOKEN_END,
    TOKEN_IDENTIFIER,
    TOKEN_KEYWORD,
    TOKEN_NUMBER,
    TOKEN_OPERATOR,
    TOKEN_PARAMETER,
    TOKEN_PUNCTUATION,
    TOKEN_QUOTED_IDENTIFIER,
    TOKEN_STRING,
    Token,
    token_text,
    tokenize,
)


def _kind_name(kind: UInt8) -> String:
    if kind == TOKEN_IDENTIFIER:
        return "id"
    if kind == TOKEN_QUOTED_IDENTIFIER:
        return "quoted"
    if kind == TOKEN_KEYWORD:
        return "kw"
    if kind == TOKEN_NUMBER:
        return "num"
    if kind == TOKEN_STRING:
        return "str"
    if kind == TOKEN_OPERATOR:
        return "op"
    if kind == TOKEN_PUNCTUATION:
        return "punct"
    if kind == TOKEN_PARAMETER:
        return "param"
    return "end"


def _render(sql: StringSlice, grammar: Grammar) raises -> String:
    """Tokenizes and writes the result as `kind:text kind:text`.

    Comparing one string rather than a dozen fields is what makes a failure here
    readable, and the text is the token's own bytes, so a token that swallowed a
    byte it should not have shows up as the wrong text rather than as a length.
    """
    var tokens = tokenize(sql, grammar)
    var out = String()
    for token in tokens:
        if token.kind == TOKEN_END:
            break
        if out.byte_length() > 0:
            out += " "
        out += String(_kind_name(token.kind), ":", token_text(sql, token))
    return out^


def _only(sql: StringSlice, grammar: Grammar) raises -> Token:
    """Tokenizes something that must come out as exactly one token."""
    var tokens = tokenize(sql, grammar)
    if len(tokens) != 2:
        raise Error(String("expected one token from ", sql))
    return tokens[0]


# ---------------------------------------------------------------------------
# The shape of the output
# ---------------------------------------------------------------------------


def test_a_statement_comes_apart_the_way_it_reads() raises:
    var g = Grammar()
    assert_equal(
        _render("SELECT a, b FROM t WHERE x = 1", g),
        "kw:SELECT id:a punct:, id:b kw:FROM id:t kw:WHERE id:x op:= num:1",
    )


def test_an_empty_query_is_just_the_end_token() raises:
    var g = Grammar()
    var tokens = tokenize("", g)
    assert_equal(len(tokens), 1)
    assert_equal(tokens[0].kind, TOKEN_END)


def test_a_query_of_nothing_but_blanks_is_just_the_end_token() raises:
    var g = Grammar()
    var tokens = tokenize("  \n\t -- a comment\n /* and */ ", g)
    assert_equal(len(tokens), 1)
    assert_equal(tokens[0].kind, TOKEN_END)


def test_the_end_token_sits_at_the_end_of_the_text() raises:
    # The matcher walks off the end of the vector on every failed alternative,
    # so the end token has to have a position a caret can point at.
    var g = Grammar()
    var tokens = tokenize("SELECT 1", g)
    assert_equal(tokens[len(tokens) - 1].kind, TOKEN_END)
    assert_equal(tokens[len(tokens) - 1].start, 8)


# ---------------------------------------------------------------------------
# Words and keywords
# ---------------------------------------------------------------------------


def test_case_does_not_matter_to_a_keyword() raises:
    var g = Grammar()
    var upper = _only("SELECT", g)
    var lower = _only("select", g)
    var mixed = _only("SeLeCt", g)
    assert_equal(upper.kind, TOKEN_KEYWORD)
    assert_equal(upper.keyword, lower.keyword)
    assert_equal(upper.keyword, mixed.keyword)
    assert_equal(
        g.keyword_classes[Int(upper.keyword)] & KEYWORD_RESERVED,
        KEYWORD_RESERVED,
    )


def test_a_keyword_token_keeps_the_bytes_the_query_wrote() raises:
    # Folding happens in a scratch buffer for the lookup and nowhere else, so
    # an error message can quote the query back the way it was typed.
    var g = Grammar()
    assert_equal(_render("SeLeCt", g), "kw:SeLeCt")


def test_a_word_that_is_not_a_keyword_is_an_identifier() raises:
    var g = Grammar()
    var token = _only("customer_id", g)
    assert_equal(token.kind, TOKEN_IDENTIFIER)
    assert_equal(token.keyword, NO_KEYWORD)


def test_an_unreserved_keyword_is_still_marked_as_a_keyword() raises:
    # Whether it can be a name here is the matcher's decision, from the class
    # mask and the rule it is standing in. The tokenizer only reports the class.
    var g = Grammar()
    var token = _only("year", g)
    assert_equal(token.kind, TOKEN_KEYWORD)
    assert_true(
        (g.keyword_classes[Int(token.keyword)] & KEYWORD_UNRESERVED) != 0
    )


def test_a_word_longer_than_any_keyword_skips_the_lookup() raises:
    # The fold buffer is sized to the longest keyword, so this is the path that
    # would read off the end of it if the length guard were wrong.
    var g = Grammar()
    var token = _only("a_column_name_far_longer_than_any_keyword", g)
    assert_equal(token.kind, TOKEN_IDENTIFIER)
    assert_equal(token.keyword, NO_KEYWORD)


def test_a_word_can_hold_a_dollar_after_the_first_byte() raises:
    # SELECT a$b FROM (SELECT 1 AS a$b) is one column in DuckDB.
    var g = Grammar()
    assert_equal(_render("a$b", g), "id:a$b")


def test_a_word_can_be_written_in_utf8() raises:
    # SELECT café FROM (SELECT 1 AS café) works, even though the grammar writes
    # PlainIdentifier as [a-z_]i[a-z0-9_]i*, which is ASCII only.
    var g = Grammar()
    assert_equal(_render("café", g), "id:café")


def test_a_quoted_identifier_keeps_its_case_and_is_never_a_keyword() raises:
    var g = Grammar()
    var token = _only('"Select"', g)
    assert_equal(token.kind, TOKEN_QUOTED_IDENTIFIER)
    assert_equal(token.keyword, NO_KEYWORD)
    assert_equal(token_text('"Select"', token), '"Select"')


def test_a_quoted_identifier_can_hold_a_doubled_quote() raises:
    # SELECT "a""b" FROM (SELECT 1 AS "a""b") is the column a"b.
    var g = Grammar()
    assert_equal(_render('"a""b"', g), 'quoted:"a""b"')


def test_an_empty_quoted_identifier_is_refused() raises:
    # SELECT 1 AS "" is `zero-length delimited identifier` in DuckDB, not an
    # empty name.
    var g = Grammar()
    with assert_raises(contains="zero-length delimited identifier"):
        _ = tokenize('SELECT 1 AS ""', g)


def test_an_unterminated_quoted_identifier_is_refused() raises:
    var g = Grammar()
    with assert_raises(contains="unterminated quoted identifier"):
        _ = tokenize('SELECT "x', g)


def test_a_unicode_quoted_identifier_is_marked() raises:
    var g = Grammar()
    var token = _only('U&"a"', g)
    assert_equal(token.kind, TOKEN_QUOTED_IDENTIFIER)
    assert_equal(token.flags, FLAG_UNICODE)


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


def test_a_plain_integer() raises:
    var g = Grammar()
    var token = _only("42", g)
    assert_equal(token.kind, TOKEN_NUMBER)
    assert_equal(token.flags, 0)


def test_a_decimal_point_means_decimal_and_not_double() raises:
    # This is why SELECT 1.1 + 2.2 is exactly 3.3 and its type is DECIMAL(3,1).
    var g = Grammar()
    assert_equal(_only("1.1", g).flags, FLAG_DECIMAL)


def test_a_number_can_end_in_its_point() raises:
    # SELECT 1. is DECIMAL(1,0).
    var g = Grammar()
    assert_equal(_render("1.", g), "num:1.")
    assert_equal(_only("1.", g).flags, FLAG_DECIMAL)


def test_a_number_can_start_with_its_point() raises:
    # SELECT .5 is DECIMAL(1,1).
    var g = Grammar()
    assert_equal(_render(".5", g), "num:.5")
    assert_equal(_only(".5", g).flags, FLAG_DECIMAL)


def test_a_second_point_is_not_part_of_the_number() raises:
    # SELECT 1.2.3 fails in DuckDB at or near ".3", so .3 is its own token.
    var g = Grammar()
    assert_equal(_render("1.2.3", g), "num:1.2 num:.3")


def test_an_exponent_makes_it_double() raises:
    var g = Grammar()
    assert_equal(_only("1e5", g).flags, FLAG_EXPONENT)
    assert_equal(_only("1E5", g).flags, FLAG_EXPONENT)
    assert_equal(_only("1e+5", g).flags, FLAG_EXPONENT)
    assert_equal(_only("1e-5", g).flags, FLAG_EXPONENT)


def test_an_exponent_beats_the_point() raises:
    # SELECT 1.5e-2 is DOUBLE, not DECIMAL, so the two flags are not both set.
    var g = Grammar()
    assert_equal(_only("1.5e-2", g).flags, FLAG_EXPONENT)
    assert_equal(_render("1.e5", g), "num:1.e5")


def test_an_e_with_no_digits_after_it_is_given_back() raises:
    # SELECT 1e is 1 aliased e, and SELECT 1e_5 is 1 aliased e_5. Both are
    # valid queries, so the tokenizer has to hand the e back rather than fail.
    var g = Grammar()
    assert_equal(_render("1e", g), "num:1 id:e")
    assert_equal(_render("1e_5", g), "num:1 id:e_5")
    assert_equal(_render("1e5e5", g), "num:1e5 id:e5")


def test_an_underscore_separates_digits_and_nothing_else() raises:
    # SELECT 1_000 is 1000. SELECT 1_ is 1 aliased _. SELECT 1__0 is 1 aliased
    # __0. So an underscore is a separator only with a digit on both sides.
    var g = Grammar()
    assert_equal(_render("1_000", g), "num:1_000")
    assert_equal(_render("1_000_000", g), "num:1_000_000")
    assert_equal(_render("1_000.000_1", g), "num:1_000.000_1")
    assert_equal(_render("1_", g), "num:1 id:_")
    assert_equal(_render("1__0", g), "num:1 id:__0")


def test_a_letter_after_a_number_starts_a_new_token() raises:
    # SELECT 1a is 1 aliased a, and SELECT 1.5abc is 1.5 aliased abc.
    var g = Grammar()
    assert_equal(_render("1a", g), "num:1 id:a")
    assert_equal(_render("1.5abc", g), "num:1.5 id:abc")


def test_hex_and_binary_are_not_number_literals() raises:
    # SELECT 0x1F returns 0 with the column named x1F, so DuckDB reads it as
    # the number 0 and the identifier x1F. There is no hex literal, and the
    # same goes for 0b101 and 0o17. This is the one place the research notes
    # for this milestone were wrong.
    var g = Grammar()
    assert_equal(_render("0x1F", g), "num:0 id:x1F")
    assert_equal(_render("0b101", g), "num:0 id:b101")
    assert_equal(_render("0o17", g), "num:0 id:o17")


def test_a_sign_is_an_operator_and_not_part_of_the_number() raises:
    # The grammar writes NumberLiteral as [+-]?[0-9]*..., but that rule is a
    # placeholder the matcher never reads, and a tokenizer that took the sign
    # would turn `a-1` into `a` and `-1` with no operator between them.
    var g = Grammar()
    assert_equal(_render("a-1", g), "id:a op:- num:1")
    assert_equal(_render("-1", g), "op:- num:1")


# ---------------------------------------------------------------------------
# Strings
# ---------------------------------------------------------------------------


def test_a_plain_string_keeps_its_quotes() raises:
    var g = Grammar()
    var token = _only("'abc'", g)
    assert_equal(token.kind, TOKEN_STRING)
    assert_equal(token.flags, 0)
    assert_equal(token_text("'abc'", token), "'abc'")


def test_a_doubled_quote_stays_inside_the_string() raises:
    var g = Grammar()
    assert_equal(_render("'a''b'", g), "str:'a''b'")
    # SELECT '''' is one quote character.
    assert_equal(_render("''''", g), "str:''''")


def test_an_empty_string_is_fine() raises:
    # SELECT '' = '' is true, so unlike "" the empty string is a value.
    var g = Grammar()
    assert_equal(_render("'' = ''", g), "str:'' op:= str:''")


def test_an_escape_string_is_marked() raises:
    # SELECT E'\n' is a newline and SELECT e'\x41' is A, so either case works.
    var g = Grammar()
    assert_equal(_only("E'a'", g).flags, FLAG_ESCAPE)
    assert_equal(_only("e'a'", g).flags, FLAG_ESCAPE)


def test_a_prefix_only_counts_when_the_quote_is_next_to_it() raises:
    # SELECT e 'a' is the identifier e and then a string, not an escape string.
    var g = Grammar()
    assert_equal(_render("e 'a'", g), "id:e str:'a'")


def test_a_unicode_string_is_marked() raises:
    var g = Grammar()
    assert_equal(_only("U&'a'", g).flags, FLAG_UNICODE)


def test_strings_join_across_a_newline_and_not_across_a_space() raises:
    # SELECT 'a'\n'b' is 'ab'. SELECT 'a' 'b' is a syntax error. This is
    # Postgres's rule and DuckDB kept it.
    var g = Grammar()
    assert_equal(_render("'a'\n'b'", g), "str:'a'\n'b'")
    assert_equal(_only("'a'\n'b'", g).flags, FLAG_CONTINUED)
    assert_equal(_render("'a' 'b'", g), "str:'a' str:'b'")


def test_a_line_comment_keeps_the_join_and_a_block_comment_breaks_it() raises:
    # SELECT 'a'--c\n'b' is 'ab'. SELECT 'a'\n/*c*/\n'b' is a syntax error.
    # Both were checked, because guessing either way is a silent difference.
    var g = Grammar()
    assert_equal(_only("'a'--c\n'b'", g).flags, FLAG_CONTINUED)
    var split = tokenize("'a'\n/*c*/\n'b'", g)
    assert_equal(len(split), 3)
    assert_equal(split[0].flags, 0)


def test_an_unterminated_string_is_refused() raises:
    var g = Grammar()
    with assert_raises(contains="unterminated quoted string"):
        _ = tokenize("SELECT 'x", g)


def test_the_error_points_at_the_quote_that_was_never_closed() raises:
    var g = Grammar()
    with assert_raises(contains="LINE 1: SELECT 'x"):
        _ = tokenize("SELECT 'x", g)
    with assert_raises(contains="\n               ^"):
        _ = tokenize("SELECT 'x", g)


def test_the_error_counts_lines_and_shows_only_the_broken_one() raises:
    var g = Grammar()
    with assert_raises(contains="LINE 3: WHERE x = 'oops"):
        _ = tokenize("SELECT a\nFROM t\nWHERE x = 'oops", g)


# ---------------------------------------------------------------------------
# Dollar quoting, which is also where parameters live
# ---------------------------------------------------------------------------


def test_a_dollar_quoted_string_has_no_escapes() raises:
    var g = Grammar()
    var token = _only("$tag$a'b\\c$tag$", g)
    assert_equal(token.kind, TOKEN_STRING)
    assert_equal(token.flags, FLAG_DOLLAR)


def test_a_dollar_quote_can_have_an_empty_tag() raises:
    var g = Grammar()
    assert_equal(_render("$$abc$$", g), "str:$$abc$$")
    assert_equal(_render("$$$$", g), "str:$$$$")


def test_a_dollar_quote_tag_can_start_with_an_underscore() raises:
    var g = Grammar()
    assert_equal(_render("$_x$a$_x$", g), "str:$_x$a$_x$")


def test_a_partial_tag_inside_a_dollar_quote_does_not_close_it() raises:
    # SELECT $tag$a$tagx$tag$ is the string a$tagx.
    var g = Grammar()
    assert_equal(_render("$tag$a$tagx$tag$", g), "str:$tag$a$tagx$tag$")


def test_a_tag_cannot_start_with_a_digit() raises:
    # SELECT $1$a$1$ fails in DuckDB with `unterminated dollar-quoted string`,
    # which only happens if $1 was read as a parameter and the dollar quote
    # started at $a$. So a digit after the dollar means a parameter.
    var g = Grammar()
    with assert_raises(contains="unterminated dollar-quoted string"):
        _ = tokenize("SELECT $1$a$1$", g)


def test_an_unterminated_dollar_quote_is_refused() raises:
    var g = Grammar()
    with assert_raises(contains="unterminated dollar-quoted string"):
        _ = tokenize("SELECT $tag$abc", g)


def test_a_parameter_marker_comes_out_on_its_own() raises:
    # The grammar has four parameter rules in expression.gram and every one of
    # them is two nodes: `'?' NumberLiteral`, `'?'`, `'$' NumberLiteral` and
    # `'$' ColLabel`. So the marker is one token and what follows it is another,
    # and the matcher gets a stream the grammar has nodes for. There is no
    # `:name` form, whatever the research notes said: SELECT :name is a syntax
    # error in DuckDB.
    var g = Grammar()
    assert_equal(_only("?", g).kind, TOKEN_PARAMETER)
    assert_equal(_render("?1", g), "param:? num:1")
    assert_equal(_render("$1", g), "param:$ num:1")
    assert_equal(_render("$total", g), "param:$ id:total")
    # ColLabel takes an unreserved keyword too, so `$offset` is a named
    # parameter and not a syntax error.
    assert_equal(_render("$offset", g), "param:$ kw:offset")


def test_a_colon_is_punctuation_and_not_a_parameter() raises:
    # SELECT :name is a syntax error, and list[1:2] needs the colon on its own.
    var g = Grammar()
    assert_equal(
        _render("a[1:2]", g), "id:a punct:[ num:1 punct:: num:2 punct:]"
    )


# ---------------------------------------------------------------------------
# Operators and punctuation
# ---------------------------------------------------------------------------


def test_the_multi_character_operators_come_out_whole() raises:
    var g = Grammar()
    assert_equal(_render("a || b", g), "id:a op:|| id:b")
    assert_equal(_render("a ->> b", g), "id:a op:->> id:b")
    assert_equal(_render("a !~~* b", g), "id:a op:!~~* id:b")
    assert_equal(_render("a >>= b", g), "id:a op:>>= id:b")


def test_a_cast_and_a_named_argument_are_operators() raises:
    var g = Grammar()
    assert_equal(_render("1 :: VARCHAR", g), "num:1 op::: kw:VARCHAR")
    assert_equal(_render("a := 1", g), "id:a op::= num:1")


def test_a_run_gives_back_a_trailing_sign_unless_it_earned_it() raises:
    # SELECT 1 =- 1 is 1 = -1, so `=-` splits. SELECT 1 !=- 1 asks the catalog
    # for an operator named `!=-`, so `!=-` does not. The difference is whether
    # the run contains one of ~ ! @ # ^ & | `, which is Postgres's rule.
    var g = Grammar()
    assert_equal(_render("1 =- 1", g), "num:1 op:= op:- num:1")
    assert_equal(_render("1 !=- 1", g), "num:1 op:!=- num:1")
    assert_equal(_render("1 - -1", g), "num:1 op:- op:- num:1")


def test_punctuation_is_one_byte_at_a_time() raises:
    var g = Grammar()
    assert_equal(
        _render("f(a, b);", g), "id:f punct:( id:a punct:, id:b punct:) punct:;"
    )


def test_a_dot_between_names_is_punctuation() raises:
    var g = Grammar()
    assert_equal(_render("db.main.t", g), "id:db punct:. id:main punct:. id:t")


# ---------------------------------------------------------------------------
# Whitespace and comments
# ---------------------------------------------------------------------------


def test_a_line_comment_runs_to_the_newline() raises:
    var g = Grammar()
    assert_equal(_render("1 -- two\n3", g), "num:1 num:3")


def test_a_line_comment_can_end_the_query() raises:
    # SELECT 1 --x is 1, so a comment with no newline after it is not an error.
    var g = Grammar()
    assert_equal(_render("1 --x", g), "num:1")


def test_block_comments_nest() raises:
    # SELECT 1 /* /* nested */ still comment */ + 1 is 2. A scanner that stops
    # at the first */ would go on to parse `still comment */ + 1` as SQL.
    var g = Grammar()
    assert_equal(_render("1 /* /* x */ y */ + 1", g), "num:1 op:+ num:1")


def test_an_unterminated_block_comment_is_refused() raises:
    var g = Grammar()
    with assert_raises(contains="unterminated /* comment"):
        _ = tokenize("SELECT 1 /*x", g)


def test_an_unclosed_nested_block_comment_is_refused() raises:
    var g = Grammar()
    with assert_raises(contains="unterminated /* comment"):
        _ = tokenize("SELECT 1 /* /* x */", g)


def test_a_form_feed_separates_tokens_and_a_vertical_tab_does_not() raises:
    # SELECT 1\f+\f1 is 2 and SELECT 1\v+\v1 is a syntax error, so the list is
    # the binary's and not the grammar's [ \t\n\r].
    var g = Grammar()
    assert_equal(_render("1\x0c+\x0c1", g), "num:1 op:+ num:1")
    assert_equal(_render("1\x0b+\x0b1", g), "num:1 op:\x0b op:+ op:\x0b num:1")


def test_a_division_is_not_a_comment() raises:
    var g = Grammar()
    assert_equal(_render("a / b", g), "id:a op:/ id:b")
    assert_equal(_render("a // b", g), "id:a op:// id:b")


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------


def test_every_token_covers_bytes_that_are_really_there() raises:
    # The tokens are the only thing the matcher sees and every error message is
    # rendered from a span, so a token that runs past the end of the query is a
    # crash in the error path of a failed parse.
    var g = Grammar()
    var sql = "SELECT $tag$x$tag$, 'a'\n'b', e'q', ?, $2, a.b[1:2] -- done"
    var tokens = tokenize(sql, g)
    var previous = 0
    for token in tokens:
        assert_true(Int(token.start) >= previous, "tokens went backwards")
        assert_true(
            Int(token.start) + Int(token.length) <= sql.byte_length(),
            "a token ran off the end",
        )
        previous = Int(token.start)


def test_a_token_never_covers_nothing_except_the_end() raises:
    var g = Grammar()
    var sql = "SELECT a, 1, 'x' FROM t"
    for token in tokenize(sql, g):
        if token.kind != TOKEN_END:
            assert_true(token.length > 0, "a token covered no bytes")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
