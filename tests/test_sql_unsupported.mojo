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
    AGGREGATE_FILTER,
    CALL_MODIFIER,
    LIST_VALUE,
    MODIFIER_SCHEMA,
    NO_CASE,
    NO_REFUSAL,
    OPERATOR,
    QUOTED_NAME,
    SUBSCRIPT,
    Refusal,
    feature_of,
    filled,
    issue_link,
    not_implemented,
    refusal,
    sql_support,
    support_table,
)


def test_the_constants_line_up_with_the_table() raises:
    assert_equal(refusal(OPERATOR).feature, "operator")
    assert_equal(refusal(SUBSCRIPT).feature, "subscript")
    assert_equal(refusal(CALL_MODIFIER).feature, "call-modifier")
    assert_equal(refusal(NO_CASE).feature, "no-case")
    assert_equal(refusal(AGGREGATE_FILTER).feature, "aggregate-filter")
    assert_equal(refusal(LIST_VALUE).feature, "list-value")
    assert_equal(refusal(MODIFIER_SCHEMA).feature, "modifier-schema")


def test_the_last_constant_is_the_last_entry() raises:
    assert_equal(Int(MODIFIER_SCHEMA) + 1, len(sql_support()))


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
    var text = String(not_implemented(QUOTED_NAME, "", ""))
    assert_equal(
        text,
        String(
            "Not Implemented Error: firepanda does not support anything but a"
            " name here.\n"
            "Only a name fits in this position. Dots in it are fine, since"
            " that is how two collations are composed. See"
            " https://github.com/tamnd/firepanda/issues/13"
        ),
    )


def test_every_entry_can_be_read_back_off_its_own_message() raises:
    # This is the one that matters, and it is a loop rather than a handful of
    # examples because the way back has to work for every entry or the counts
    # it feeds are wrong in a way nobody would notice. The detail is a word no
    # message contains, so an entry with a hole cannot match by accident.
    var table = sql_support()
    for i in range(len(table)):
        var text = String(not_implemented(UInt16(i), "quack", ""))
        assert_equal(Int(feature_of(text)), i, table[i].feature)


def test_reading_back_a_refusal_that_carries_a_position() raises:
    # The caret block sits between the first line and the explanation and can
    # hold any text at all, including text that looks like another message, so
    # only the first line is read.
    var text = String(
        not_implemented(SUBSCRIPT, "", "LINE 1: SELECT a[1]\n               ^")
    )
    assert_equal(feature_of(text), SUBSCRIPT)


def test_text_that_is_not_a_refusal_reads_back_as_none() raises:
    assert_equal(
        feature_of("Parser Error: syntax error at or near 'x'"), NO_REFUSAL
    )
    assert_equal(feature_of(""), NO_REFUSAL)
    assert_equal(
        feature_of("Not Implemented Error: firepanda does not support quack."),
        NO_REFUSAL,
    )


def test_a_link_is_a_link() raises:
    assert_equal(
        issue_link(304), "See https://github.com/tamnd/firepanda/issues/304"
    )


def test_the_transformer_refuses_in_the_table_shape() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var sql = "SELECT MAP {'a': 1} FROM t"
    var text = String()
    try:
        _ = rules.parse_statement(sql, g, ast)
    except error:
        text = String(error)
    assert_true(
        text.startswith("Not Implemented Error: firepanda does not support"),
        text,
    )
    assert_true("LINE 1: " + sql in text, text)
    assert_true("^" in text, text)
    assert_true("issues/" in text, text)


def test_the_caret_points_at_the_thing_that_was_refused() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var text = String()
    try:
        _ = rules.parse_statement("SELECT MAP {'a': 1}", g, ast)
    except error:
        text = String(error)
    assert_true("LINE 1: SELECT MAP {'a': 1}\n               ^" in text, text)


def test_a_refusal_on_the_second_line_counts_lines() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var text = String()
    try:
        _ = rules.parse_statement("SELECT 1\nFROM t WHERE MAP {'a': 1}", g, ast)
    except error:
        text = String(error)
    assert_true("LINE 2: FROM t WHERE MAP {'a': 1}" in text, text)


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
