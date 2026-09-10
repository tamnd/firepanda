"""The refusal table.

Two things are worth testing here and neither of them is the wording. The
constants have to line up with the table, because a constant one out of step
names the wrong feature in an error that reads as if it were right. And a
refusal has to carry all four of its parts, because the part that gets dropped
is the explanation and the whole point of the table is that it does not get
dropped.
"""

from std.testing import TestSuite, assert_equal, assert_true

from firepanda.sql import Grammar, Transform
from firepanda.sql.ast import Ast
from firepanda.sql.unsupported import (
    ALIAS_COLON,
    CALL_MODIFIER,
    ESCAPE_STRING,
    IS_UNKNOWN,
    NO_CASE,
    OPERATOR,
    SUBSCRIPT,
    Refusal,
    filled,
    issue_link,
    not_implemented,
    refusal,
    sql_support,
    support_table,
)


def test_the_constants_line_up_with_the_table() raises:
    assert_equal(refusal(OPERATOR).feature, "operator")
    assert_equal(refusal(IS_UNKNOWN).feature, "is-unknown")
    assert_equal(refusal(SUBSCRIPT).feature, "subscript")
    assert_equal(refusal(CALL_MODIFIER).feature, "call-modifier")
    assert_equal(refusal(ALIAS_COLON).feature, "alias-colon")
    assert_equal(refusal(ESCAPE_STRING).feature, "escape-string")
    assert_equal(refusal(NO_CASE).feature, "no-case")


def test_the_last_constant_is_the_last_entry() raises:
    assert_equal(Int(NO_CASE) + 1, len(sql_support()))


def test_no_two_entries_share_a_name() raises:
    var seen = sql_support()
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            assert_true(
                seen[i].feature != seen[j].feature,
                String("two entries are named ", seen[i].feature),
            )


def test_every_entry_has_all_four_parts() raises:
    for entry in sql_support():
        assert_true(entry.feature.byte_length() > 0, "an entry has no name")
        assert_true(
            entry.message.byte_length() > 0,
            String(entry.feature, " has no message"),
        )
        assert_true(
            entry.explanation.byte_length() > 0,
            String(entry.feature, " has no explanation"),
        )
        assert_true(
            entry.issue > 0, String(entry.feature, " points at no issue")
        )


def test_a_name_is_a_slug_rather_than_a_sentence() raises:
    for entry in sql_support():
        for byte in entry.feature.as_bytes():
            var ordinary = (
                byte >= Byte(ord("a")) and byte <= Byte(ord("z"))
            ) or byte == Byte(ord("-"))
            assert_true(ordinary, String(entry.feature, " is not a slug"))


def test_an_explanation_ends_in_a_full_stop() raises:
    for entry in sql_support():
        var text = String(entry.explanation)
        assert_equal(
            text[byte = text.byte_length() - 1 : text.byte_length()],
            ".",
            String(entry.feature, " does not end its explanation"),
        )


def test_a_message_holds_at_most_one_hole() raises:
    for entry in sql_support():
        var text = String(entry.message)
        var at = text.find("{}")
        if at < 0:
            continue
        assert_equal(
            text.find("{}", at + 2),
            -1,
            String(entry.feature, " has two holes in one message"),
        )


def test_filling_a_hole() raises:
    assert_equal(filled("{} on a call", "ORDER BY"), "ORDER BY on a call")
    assert_equal(filled("IS UNKNOWN", "ignored"), "IS UNKNOWN")
    assert_equal(filled("grammar rule {}", "9"), "grammar rule 9")


def test_a_refusal_names_the_feature_the_position_and_the_way_out() raises:
    var text = String(
        not_implemented(SUBSCRIPT, "", "LINE 1: SELECT a[1]\n               ^")
    )
    assert_true(
        text.startswith(
            "Not Implemented Error: firepanda does not support a slice or a"
            " subscript."
        ),
        text,
    )
    assert_true("LINE 1: SELECT a[1]" in text, text)
    assert_true("^" in text, text)
    assert_true("brackets" in text, text)
    assert_true("https://github.com/tamnd/firepanda/issues/" in text, text)


def test_a_refusal_with_no_position_still_carries_the_rest() raises:
    var text = String(not_implemented(IS_UNKNOWN, "", ""))
    assert_equal(
        text,
        String(
            "Not Implemented Error: firepanda does not support IS UNKNOWN.\n"
            "It means IS NULL over a boolean, which firepanda does have, so"
            " write that instead. See"
            " https://github.com/tamnd/firepanda/issues/13"
        ),
    )


def test_a_link_is_a_link() raises:
    assert_equal(
        issue_link(304), "See https://github.com/tamnd/firepanda/issues/304"
    )


def test_the_transformer_refuses_in_the_table_shape() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var sql = "SELECT a[1] FROM t"
    var text = String()
    try:
        _ = rules.parse_statement(sql, g, ast)
    except error:
        text = String(error)
    assert_true(
        text.startswith("Not Implemented Error: firepanda does not support"),
        text,
    )
    assert_true("LINE 1: SELECT a[1] FROM t" in text, text)
    assert_true("^" in text, text)
    assert_true("issues/" in text, text)


def test_the_caret_points_at_the_thing_that_was_refused() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var text = String()
    try:
        _ = rules.parse_statement("SELECT x IS UNKNOWN", g, ast)
    except error:
        text = String(error)
    assert_true("LINE 1: SELECT x IS UNKNOWN\n                 ^" in text, text)


def test_a_refusal_on_the_second_line_counts_lines() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var text = String()
    try:
        _ = rules.parse_statement("SELECT 1\nFROM t AT (VERSION => 2)", g, ast)
    except error:
        text = String(error)
    assert_true("LINE 2: FROM t AT (VERSION => 2)" in text, text)


def test_the_readme_table_is_the_refusal_table() raises:
    # The README's list of what the SQL front end will not do is generated by
    # `tools/sql_support.mojo` out of `sql_support()`. This is what makes
    # forgetting to run it a test failure rather than a stale README nobody
    # notices for a month. The test suite runs from the repository root.
    var handle = open("README.md", "r")
    var readme = handle.read()
    handle.close()
    var table = support_table()
    assert_true(
        table in readme,
        (
            "README.md does not hold the current support table, run 'pixi run"
            " sql-support' and commit the result"
        ),
    )


def test_every_entry_reaches_the_readme_by_name() raises:
    # `in` on the whole block would still pass if the renderer dropped a column,
    # so the names are checked one at a time as well.
    var handle = open("README.md", "r")
    var readme = handle.read()
    handle.close()
    for entry in sql_support():
        assert_true(
            String("| `", entry.feature, "` |") in readme,
            String(entry.feature, " is not in the README table"),
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
