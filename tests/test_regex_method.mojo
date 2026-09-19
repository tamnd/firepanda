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
    METHOD_COUNT,
    METHOD_EXTRACT,
    METHOD_FULLMATCH,
    METHOD_MATCH,
    anchored,
    leading_flags,
    preprocessed,
    program_for,
    python_anchored,
)
from firepanda.kernel.regex.parse import parse_pattern
from firepanda.kernel.regex.pike import matches_text
from firepanda.kernel.regex.tokens import (
    FLAG_ASCII,
    FLAG_IGNORECASE,
    FLAG_LOCALE,
    FLAG_MULTILINE,
    FLAG_VERBOSE,
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
    """A lookahead beside a conditional group goes to Python's `re` upstream and
    this library has no engine that can run both at once, so the refusal is its
    own and names what is in the way rather than naming the engine. It used to
    say the engine was not written at all, which was true until document 81
    wrote it and which threw away the one thing the caller could act on.

    This row used to ask about a lookaround, then about a backreference, then
    about an atomic group and then about a conditional group, all of which are
    answered now, so what is left of it is the pairing. The lookahead in front
    is still what routes the call, since a conditional group is not one of the
    constructs the router walks for. Documents 93, 94, 95, 99 and 100."""
    var program = program_for(METHOD_MATCH, "(?=a)(a)(?(1)b|c)")
    assert_false(program.ok)
    assert_true(program.gap)
    assert_equal(
        program.problem,
        "this engine has no lookaround beside a conditional group yet",
    )


def test_the_engine_is_picked_before_the_pattern_is_rewritten() raises:
    """A flag group and a backreference together, which reads and answers
    upstream and stops being a pattern at all once it is wrapped. Deciding on
    the rewrite would send it to an engine that has never heard of a
    backreference and report the wrong reason for refusing it.

    The flag used to be a refusal of its own and used to be the one the
    compiler met first, so this case used to answer with the folding sentence.
    Document 83 spent the flag while the pattern is being compiled, so the
    construct is now the only shortfall left in this pattern and the refusal is
    the one the caller can act on.

    The construct here has been three of them. It was a lookaround until
    documents 93 and 94 answered both halves, then a backreference under the
    wide reading of this very flag until document 97 answered that, and it is
    now the one pair that is still refused on this engine, a lookaround
    standing beside a backreference. Each time the row kept its point, which is
    that the refusal names the construct rather than the rewrite."""
    var program = program_for(METHOD_FULLMATCH, "(?i)(?=a)(b)\\1")
    assert_false(program.ok)
    assert_true(program.gap)
    assert_equal(
        program.problem,
        "this engine has no lookaround beside a backreference yet",
    )


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


def test_a_flag_passed_beside_the_pattern_folds_what_a_written_one_folds() raises:
    """The `case` argument and `(?i)` are one fact spelled two ways.

    Upstream makes them one fact by compiling the pattern with the argument
    before anything routes or rewrites it, and this makes them one fact by
    seeding the parser. Either way what reaches the compiler is a tree with a
    flag on it and no memory of how the flag got there, which is why this can be
    asserted as an equality over rows rather than as a rule about arguments.
    """
    var rows = [
        String("abc"),
        String("ABC"),
        String("k"),
        String("\u212a"),
        String("s"),
        String("\u017f"),
        String("\u03c3"),
        String("\u03a3"),
        String("0"),
        String(""),
    ]
    for method in [METHOD_CONTAINS, METHOD_MATCH, METHOD_FULLMATCH]:
        for pattern in [
            String("a"),
            String("[a-z]"),
            String("k"),
            String("\u03c3"),
            String("a.c"),
        ]:
            var argued = program_for(method, pattern, FLAG_IGNORECASE)
            var written = program_for(method, String("(?i)", pattern))
            assert_true(argued.ok)
            assert_true(written.ok)
            for row in rows:
                assert_equal(
                    matches_text(argued, row), matches_text(written, row)
                )


def test_a_flag_passed_beside_the_pattern_survives_the_rewrite() raises:
    """`match` and `fullmatch` compile a second time, on a pattern this file
    built rather than on the one the caller wrote, and the flag has to be handed
    to that parse as well. A seeded flag dropped on the way would leave these
    two folding and `contains` not, which is the shape of bug that shows up as
    one method out of three disagreeing with pandas."""
    assert_true(
        matches_text(
            program_for(METHOD_FULLMATCH, "[a-z]+", FLAG_IGNORECASE), "ABC"
        )
    )
    assert_true(
        matches_text(program_for(METHOD_MATCH, "ab", FLAG_IGNORECASE), "ABc")
    )
    assert_false(
        matches_text(
            program_for(METHOD_FULLMATCH, "[a-z]+", FLAG_IGNORECASE), "AB1"
        )
    )


def test_a_flag_passed_beside_the_pattern_reaches_whichever_engine_runs_it() raises:
    """The flag is spent by the compiler and the compiler is told which engine
    it is compiling for, so an argued flag picks up the engine difference the
    same way a written one does. The dotted capital I is the whole of that
    difference: Python folds it onto a plain `i` and RE2 leaves it alone."""
    assert_false(
        matches_text(
            program_for(METHOD_CONTAINS, "i", FLAG_IGNORECASE), "\u0130"
        )
    )
    assert_true(
        matches_text(
            program_for(METHOD_EXTRACT, "i", FLAG_IGNORECASE), "\u0130"
        )
    )


def test_an_alphabet_passed_beside_the_pattern_meets_the_one_it_wrote() raises:
    """Seeding rather than merging afterwards means the checks that read the
    flags read the argument too, so a pattern written `(?u)` and handed the
    other alphabet is refused for the reason a pattern writing both is. Nothing
    upstream sends this combination, and the point is that the check cannot be
    walked around rather than that the combination matters."""
    var tree = parse_pattern("(?u)a", FLAG_ASCII)
    assert_false(tree.ok)
    assert_equal(tree.problem, "ASCII and UNICODE flags are incompatible")
    assert_false(program_for(METHOD_CONTAINS, "(?u)a", FLAG_ASCII).ok)


def test_an_argued_call_is_anchored_from_outside_the_pattern() raises:
    """The other rewrite, which is the one upstream does not do.

    A call that landed on Python's engine is answered there by `regex.match` and
    `regex.fullmatch`, and those anchor from outside the pattern where the Arrow
    rewrite glues anchors inside it. So the two positions written here are the
    ones no flag can move, nothing is cropped off either end, and the three
    methods that ask about a whole row rather than its front get the pattern
    back unchanged.

    The closing anchor is spelled `\\Z` and not `\\z`. They are one position and
    the choice would be free if the two spellings had always been legal in the
    same places, and they have not: Python got `\\z` in 3.14 and every older one
    reads it as a bad escape. A pattern written here is written for Python's
    engine, so it is spelled the way Python has always spelled it. Document 91.
    """
    assert_equal(python_anchored(METHOD_CONTAINS, "a"), "a")
    assert_equal(python_anchored(METHOD_COUNT, "^a$"), "^a$")
    assert_equal(python_anchored(METHOD_MATCH, "a"), "\\A(?:a)")
    assert_equal(python_anchored(METHOD_FULLMATCH, "a"), "\\A(?:a)\\Z")
    assert_equal(python_anchored(METHOD_FULLMATCH, "^a$"), "\\A(?:^a$)\\Z")
    assert_equal(python_anchored(METHOD_FULLMATCH, "(?i)a"), "(?i)\\A(?:a)\\Z")
    assert_equal(python_anchored(METHOD_MATCH, "(?ims)a"), "(?ims)\\A(?:a)")


def test_a_verbose_pattern_is_closed_on_a_line_of_its_own() raises:
    """The bracket that would otherwise be written into a comment.

    A verbose pattern can end in the middle of a comment, because a comment ends
    at a newline or at the end of the pattern and both are allowed. Glue a `)`
    onto the end of one and the bracket is inside the comment, the group is
    never closed and the grammar refuses a pattern the caller wrote correctly.
    The newline is the only thing that ends a comment, and verbose mode throws
    a newline away, so it changes the answer nowhere and saves it here.
    """
    assert_equal(
        python_anchored(METHOD_FULLMATCH, "a # c", True), "\\A(?:a # c\n)\\Z"
    )
    assert_equal(python_anchored(METHOD_MATCH, "a # c", True), "\\A(?:a # c\n)")
    assert_equal(
        python_anchored(METHOD_FULLMATCH, "(?x)a # c", True),
        "(?x)\\A(?:a # c\n)\\Z",
    )
    assert_equal(python_anchored(METHOD_FULLMATCH, "a # c"), "\\A(?:a # c)\\Z")
    assert_true(
        program_for(METHOD_FULLMATCH, "a # c", FLAG_VERBOSE, argued=True).ok
    )
    assert_true(program_for(METHOD_FULLMATCH, "(?x)a # c", 0, argued=True).ok)


def test_an_argued_flag_moves_the_call_and_a_written_one_does_not() raises:
    """The same bit, spelled two ways, landing on two engines.

    This is the whole reason the flags and the fact that they were argued cross
    the door as two things rather than as one number. The dotted capital I is
    where the two fold tables part company: Python folds it onto a plain `i` and
    RE2 leaves it alone, so `contains("i", case=False)` and
    `contains("i", flags=re.IGNORECASE)` answer differently upstream and have to
    answer differently here.
    """
    assert_false(
        matches_text(
            program_for(METHOD_CONTAINS, "i", FLAG_IGNORECASE), "\u0130"
        )
    )
    assert_true(
        matches_text(
            program_for(METHOD_CONTAINS, "i", FLAG_IGNORECASE, argued=True),
            "\u0130",
        )
    )


def test_an_argued_call_reads_its_classes_the_way_python_reads_them() raises:
    """The second of the four differences between the engines, and the one that
    covers the most patterns. A word character is 63 code points to RE2 and
    138558 to Python, so a flag that moved the call moved what `\\w` means with
    it, which is true upstream and is what makes the routing worth getting
    right rather than merely tidy."""
    assert_false(matches_text(program_for(METHOD_CONTAINS, "\\w"), "\u00e9"))
    assert_true(
        matches_text(
            program_for(METHOD_CONTAINS, "\\w", FLAG_MULTILINE, argued=True),
            "\u00e9",
        )
    )


def test_an_argued_fullmatch_will_not_stop_short_of_a_trailing_newline() raises:
    """What the anchors above are for. `re.fullmatch("a", "a\\n")` finds
    nothing and `^(a)$` with the multiline flag on matches the first line of it,
    so a rewrite that copied Arrow's anchors would answer True where pandas
    answers False. The measurement is pandas 3.0.5 on a column of `a\\n` and
    `a`, which answers False and then True."""
    var program = program_for(
        METHOD_FULLMATCH, "a", FLAG_MULTILINE, argued=True
    )
    assert_true(program.ok)
    assert_false(matches_text(program, "a\n"))
    assert_true(matches_text(program, "a"))


def test_an_argued_call_is_refused_by_pythons_rules_and_not_by_re2s() raises:
    """The letters the two engines will not take are not the same letters.

    `(?a)` is a syntax error to RE2 and a flag Python reads perfectly well, and
    an argued call carrying it used to be this library falling short rather
    than agreeing with anybody. It is answered now, which leaves one letter of
    the seven: `(?L)` is refused by Python itself on a pattern made of text,
    which is what every pattern here is made of, so that one is an error on both
    sides and says so.
    """
    var narrow = program_for(METHOD_CONTAINS, "a", FLAG_ASCII, argued=True)
    assert_true(narrow.ok)
    var locale = program_for(METHOD_CONTAINS, "a", FLAG_LOCALE, argued=True)
    assert_false(locale.ok)
    assert_false(locale.gap)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
