"""Tests that each of the three methods runs the pattern pandas would run.

The differential next door compares answers, which is the stronger test and is
blind in one direction: a rewrite this file gets wrong in the same way for every
pattern would still agree with pandas if the engine agreed with RE2, and a
rewrite that quietly refused a family of patterns would agree with pandas on
every pattern it did not refuse. So the pattern text itself is asserted here,
character for character, against what upstream builds.

The cases are the branches rather than a sample. `fullmatch` has four of them,
one per combination of the pattern already carrying an anchor at either end, and
three of the four are one line of upstream each. The flag group hoist is the
fifth and is this library's own, so it gets the most attention.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.kernel.regex.method import (
    METHOD_CONTAINS,
    METHOD_FULLMATCH,
    METHOD_MATCH,
    anchored,
    leading_flags,
    preprocessed,
    program_for,
)


def test_a_pattern_with_no_flag_group_measures_nothing() raises:
    """The common case, where the hoist has nothing to do."""
    assert_equal(leading_flags("a"), 0)
    assert_equal(leading_flags(""), 0)
    assert_equal(leading_flags("a(?i)"), 0)


def test_a_flag_group_is_measured_and_its_lookalikes_are_not() raises:
    """Every other group opening `(?` has a second character that is not a flag
    letter, which is what stops the scan at once."""
    assert_equal(leading_flags("(?i)a"), 4)
    assert_equal(leading_flags("(?ims)a"), 6)
    assert_equal(leading_flags("(?i-sx)a"), 7)
    assert_equal(leading_flags("(?:a)"), 0)
    assert_equal(leading_flags("(?=a)"), 0)
    assert_equal(leading_flags("(?P<n>a)"), 0)
    assert_equal(leading_flags("(?#c)a"), 0)


def test_an_empty_or_unfinished_flag_group_is_left_alone() raises:
    """Neither is a flag group to anybody, and moving one would move a syntax
    error rather than a flag."""
    assert_equal(leading_flags("(?)"), 0)
    assert_equal(leading_flags("(?i"), 0)
    assert_equal(leading_flags("(?"), 0)


def test_a_trailing_end_of_text_is_spelled_the_way_re2_spells_it() raises:
    """RE2 has no `\\Z` and pandas rewrites the one place it can."""
    assert_equal(preprocessed("a\\Z"), "a\\z")
    assert_equal(preprocessed("\\Z"), "\\z")
    assert_equal(preprocessed("a\\\\\\Z"), "a\\\\\\z")


def test_an_end_of_text_that_is_not_one_is_left_alone() raises:
    """A `\\Z` in the middle is a pattern Arrow refuses, and `\\\\Z` is a
    backslash and then a letter, which is not an escape at all."""
    assert_equal(preprocessed("a\\Zb"), "a\\Zb")
    assert_equal(preprocessed("a\\\\Z"), "a\\\\Z")
    assert_equal(preprocessed("aZ"), "aZ")


def test_contains_runs_the_pattern_as_written() raises:
    """It is the question the engine answers, so there is nothing to rewrite."""
    assert_equal(anchored(METHOD_CONTAINS, "a|b"), "a|b")
    assert_equal(anchored(METHOD_CONTAINS, "^a$"), "^a$")
    assert_equal(anchored(METHOD_CONTAINS, "(?i)a"), "(?i)a")


def test_match_anchors_the_whole_pattern_rather_than_its_first_arm() raises:
    """The group is what makes `str.match("a|b")` ask whether a row starts with
    either, and is the reason the rewrite is not a `^` and a join."""
    assert_equal(anchored(METHOD_MATCH, "a|b"), "^(a|b)")
    assert_equal(anchored(METHOD_MATCH, ""), "^()")
    assert_equal(anchored(METHOD_MATCH, "a$"), "^(a$)")


def test_match_moves_one_caret_and_leaves_the_rest() raises:
    """Upstream strips exactly one, so `^^a` keeps the second, which asserts the
    same position twice and says the same thing."""
    assert_equal(anchored(METHOD_MATCH, "^a"), "^(a)")
    assert_equal(anchored(METHOD_MATCH, "^^a"), "^(^a)")
    assert_equal(anchored(METHOD_MATCH, "a^b"), "^(a^b)")


def test_fullmatch_adds_the_anchor_that_is_missing_at_each_end() raises:
    """Four branches, one per pattern already carrying an anchor at either end,
    and the last of them adds nothing but the group `match` adds."""
    assert_equal(anchored(METHOD_FULLMATCH, "a"), "^((a)$)")
    assert_equal(anchored(METHOD_FULLMATCH, "^a"), "^((a)$)")
    assert_equal(anchored(METHOD_FULLMATCH, "a$"), "^((a)$)")
    assert_equal(anchored(METHOD_FULLMATCH, "^a$"), "^(a$)")


def test_fullmatch_reads_an_escaped_dollar_as_a_dollar_sign() raises:
    """A pattern ending in a dollar sign the caller wanted printed has no anchor
    on it, so one is added."""
    assert_equal(anchored(METHOD_FULLMATCH, "a\\$"), "^((a\\$)$)")
    assert_equal(anchored(METHOD_FULLMATCH, "a$$"), "^((a$)$)")


def test_a_leading_flag_group_stays_in_front_for_the_grammar() raises:
    """Python's grammar refuses a global flag group anywhere but the front, and
    this library parses its own rewrite where pandas hands it to Arrow."""
    assert_equal(anchored(METHOD_MATCH, "(?s)a.b"), "(?s)\\A(a.b)")
    assert_equal(anchored(METHOD_FULLMATCH, "(?s)a.b"), "(?s)\\A((a.b)\\z)")


def test_a_hoisted_group_cannot_reach_the_anchors_that_were_added() raises:
    """The whole of the difference between the hoist and what upstream writes.
    `(?m)` in front of the rewrite would make the added `^` and `$` match at
    every line rather than at the ends of the row, which is a different column
    and not a slower one. `\\A` and `\\z` are the same two positions with no
    flag able to touch them."""
    assert_equal(anchored(METHOD_MATCH, "(?m)b"), "(?m)\\A(b)")
    assert_equal(anchored(METHOD_FULLMATCH, "(?m)b"), "(?m)\\A((b)\\z)")
    assert_equal(anchored(METHOD_FULLMATCH, "(?m)"), "(?m)\\A(()\\z)")


def test_a_hoisted_group_keeps_the_caret_the_caller_wrote() raises:
    """Upstream strips one `^` from the front of the pattern and a pattern
    opening with a flag group has none there, so there is nothing to strip and
    the caller's own anchor stays inside the group where the flag can reach
    it."""
    assert_equal(anchored(METHOD_MATCH, "(?m)^b"), "(?m)\\A(^b)")
    assert_equal(anchored(METHOD_FULLMATCH, "(?m)^b$"), "(?m)\\A((^b)\\z)")


def test_a_pattern_the_engine_can_run_compiles_for_all_three() raises:
    """The ordinary case, and the one the accessor is wired to."""
    for method in [METHOD_CONTAINS, METHOD_MATCH, METHOD_FULLMATCH]:
        var program = program_for(method, "a.b")
        assert_true(program.ok)
        assert_equal(program.problem, "")


def test_a_pattern_the_other_engine_would_run_is_a_gap_here() raises:
    """A lookaround goes to Python's `re` upstream and this library's Python
    engine has no lookaround yet, so the refusal is its own and names the
    construct rather than naming the engine. It used to say the engine was not
    written at all, which was true until document 81 wrote it and which threw
    away the one thing the caller could act on."""
    var program = program_for(METHOD_MATCH, "a(?=b)")
    assert_false(program.ok)
    assert_true(program.gap)
    assert_equal(program.problem, "this engine has no lookaround yet")


def test_the_engine_is_picked_before_the_pattern_is_rewritten() raises:
    """A flag group and a lookahead together, which reads and answers upstream
    and stops being a pattern at all once it is wrapped. Deciding on the
    rewrite would send it to an engine that has never heard of a lookahead and
    report the wrong reason for refusing it.

    The flag used to be a refusal of its own and used to be the one the
    compiler met first, so this case used to answer with the folding sentence.
    Document 83 spent the flag while the pattern is being compiled, so the
    lookahead is now the only shortfall left in this pattern and the refusal is
    the one the caller can act on."""
    var program = program_for(METHOD_FULLMATCH, "(?i)(?=a)")
    assert_false(program.ok)
    assert_true(program.gap)
    assert_equal(program.problem, "this engine has no lookaround yet")


def test_a_pattern_re2_refuses_is_refused_rather_than_held_out() raises:
    """A comment group is Python syntax and not RE2 syntax, and pandas hands it
    to RE2 anyway, so the refusal belongs to the pattern rather than to this
    library and the flag says so."""
    var program = program_for(METHOD_MATCH, "a(?#note)b")
    assert_false(program.ok)
    assert_false(program.gap)


def test_a_pattern_the_grammar_cannot_read_is_refused_as_written() raises:
    """The rewrite closes a bracket nobody opened, and a pattern that is good
    only after being wrapped is a pattern upstream refused before it wrapped
    anything."""
    for method in [METHOD_CONTAINS, METHOD_MATCH, METHOD_FULLMATCH]:
        var program = program_for(method, ")a")
        assert_false(program.ok)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
