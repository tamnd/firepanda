"""Tests that a pattern reaches the engine pandas would have sent it to.

This is a small file about a large consequence. The decision it tests is one
walk over a parsed pattern looking for three op codes, and what hangs on it is
which of two engines computes a column of answers, where the two engines
disagree about what `\\d` means, about what `$` matches and about several other
things a caller would never think to check.

The walk is incomplete upstream and the incompleteness is reproduced here on
purpose, so most of this file is pairs. A pattern that routes to Python and the
same pattern with a quantifier on it, which routes to RE2 and raises there. A
negative lookaround with a body and the empty one, which is not a lookaround by
the time the parser is done with it. Written as pairs because either half on its
own reads as an arbitrary fact, and together they are the rule.

Every expectation here was measured against pandas before it was written down,
and the same expectations are checked again over thirty thousand generated
patterns by `tests/differential/regex.mojo`. What this file adds is the reason:
a differential says the two agree, and these say what they agree about.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.kernel.regex.parse import parse_pattern
from firepanda.kernel.regex.route import (
    ENGINE_PYTHON,
    ENGINE_RE2,
    engine_for,
    holds_unsupported,
    reads_as_python,
)


def python_answers(pattern: StringSlice) -> Bool:
    """Whether a pattern routes to Python's `re`.

    Args:
        pattern: The pattern.

    Returns:
        True when Python answers it.
    """
    return engine_for(pattern) == ENGINE_PYTHON


def test_a_plain_pattern_goes_to_arrow() raises:
    """The common case, and the reason the router exists at all: almost
    everything a caller writes is answered by RE2."""
    assert_equal(engine_for("abc"), ENGINE_RE2)
    assert_equal(engine_for("a.*b"), ENGINE_RE2)
    assert_equal(engine_for("^[a-z]+$"), ENGINE_RE2)


def test_a_lookaround_sends_the_pattern_to_python() raises:
    """All four of them, since the two directions and the two polarities are
    four op codes and the walk looks for two."""
    assert_true(python_answers("(?=a)"))
    assert_true(python_answers("(?!a)"))
    assert_true(python_answers("(?<=a)"))
    assert_true(python_answers("(?<!a)"))


def test_a_backreference_sends_the_pattern_to_python() raises:
    """The third op code the walk looks for, and the one that cannot be
    answered in linear time, which is why RE2 does not have it."""
    assert_true(python_answers("(a)\\1"))
    assert_true(python_answers("(?P<n>a)(?P=n)"))


def test_a_quantifier_hides_a_lookaround_from_the_walk() raises:
    """The upstream bug this reproduces, in one pair.

    pandas recurses into a subpattern and a branch and nothing else, so a repeat
    node is opaque to it. `(?=a)` is answered by Python and gives a column of
    booleans, and `(?=a)?` is handed to RE2, which has never heard of `(?=` and
    raises an error naming a library the caller did not call. Making an
    assertion optional should not change which engine runs.
    """
    assert_true(python_answers("(?=a)"))
    assert_false(python_answers("(?=a)*"))
    assert_false(python_answers("(?=a)?"))
    assert_false(python_answers("(?=a){2}"))


def test_the_same_hiding_happens_through_a_group() raises:
    """Wrapping the assertion first does not help, because the repeat is still
    the node between the walk and the assertion."""
    assert_true(python_answers("(?:(?=a))"))
    assert_false(python_answers("(?:(?=a))*"))
    assert_true(python_answers("((?=a))"))
    assert_false(python_answers("((?=a))*"))


def test_an_atomic_group_hides_a_lookaround_too() raises:
    """The fourth of the five node kinds the walk does not enter. RE2 refuses
    the atomic group as well, so this pattern raises for two reasons."""
    assert_false(python_answers("(?>(?=a))"))


def test_a_conditional_hides_one_as_well() raises:
    """The fifth. A conditional is itself a construct RE2 does not have, so a
    pattern reaching this line was going to raise whatever the assertion inside
    it did."""
    assert_false(python_answers("(a)(?(1)(?=b)|c)"))


def test_a_branch_is_walked_into() raises:
    """One of the two kinds the walk does enter, so an assertion in any
    alternative is found."""
    assert_true(python_answers("a|(?=b)"))
    assert_true(python_answers("(?=b)|a"))
    assert_true(python_answers("a|b|c|(?<!d)"))


def test_a_subpattern_is_walked_into_to_any_depth() raises:
    """The other kind, and the recursion has no ceiling, so nesting alone never
    hides anything."""
    assert_true(python_answers("(((((?=a)))))"))
    assert_true(python_answers("(a(b(c(?=d))))"))


def test_a_sequence_is_walked_through_without_counting_as_a_level() raises:
    """Where this parser and Python's differ in shape and not in answer.

    Python inlines a non capturing group, so its assertion sits at the level the
    walk reads. This keeps a sequence node instead and steps through it. The
    trees are different and the routing is the same, which is the only thing
    that has to hold.
    """
    assert_true(python_answers("(?:(?:(?:(?=a))))"))
    assert_true(python_answers("ab(?:c(?=d))"))


def test_an_empty_negative_lookaround_is_not_a_lookaround_any_more() raises:
    """The pair that shows the collapse is a routing decision.

    `(?!)` becomes a node that never matches, so there is no assertion left to
    find and the pattern goes to RE2, which raises. `(?=)` keeps its node and
    goes to Python, which answers it. One character apart, two engines.
    """
    assert_false(python_answers("(?!)"))
    assert_false(python_answers("(?<!)"))
    assert_true(python_answers("(?=)"))
    assert_true(python_answers("(?<=)"))


def test_a_pattern_that_does_not_parse_goes_to_arrow() raises:
    """The parse error is caught upstream and the pattern goes to Arrow.

    So an unreadable pattern and a plain one route the same way, and they are
    told apart only by asking the second question.
    """
    assert_false(python_answers("(a"))
    assert_false(reads_as_python("(a"))
    assert_false(python_answers("a**"))
    assert_false(reads_as_python("a**"))


def test_the_two_questions_are_separate() raises:
    """A pattern reaches RE2 either by holding nothing interesting or by being
    unreadable, and a caller explaining a decision needs to know which."""
    assert_true(reads_as_python("abc"))
    assert_false(python_answers("abc"))
    assert_true(reads_as_python("(?=a)"))
    assert_true(python_answers("(?=a)"))


def test_a_refused_pattern_that_holds_an_assertion_still_goes_to_arrow() raises:
    """The parse failure wins, because there is no tree to walk.

    This is worth an assertion of its own because it is the case where the two
    questions pull in opposite directions, and a router that answered the
    second one from a half built tree would send a broken pattern to Python.
    """
    assert_false(python_answers("(?=a)("))
    assert_false(reads_as_python("(?=a)("))


def test_a_property_escape_is_unreadable_and_therefore_an_arrow_pattern() raises:
    """`\\p{L}` is RE2's syntax and Python has no idea what it is.

    So a caller writing one has written a pattern that only works because it
    failed to parse, which is a load bearing accident: add a lookahead to it and
    it still fails to parse, still goes to RE2, and now raises there.
    """
    assert_false(reads_as_python("\\p{L}"))
    assert_false(python_answers("\\p{L}"))
    assert_false(reads_as_python("\\p{L}(?=x)"))
    assert_false(python_answers("\\p{L}(?=x)"))


def test_a_global_flag_group_in_the_wrong_place_changes_the_engine() raises:
    """Python 3.11 made the position a parse error, and a parse error is a
    routing decision, so `a(?i)b` is an RE2 pattern and `(?i)ab` is not."""
    assert_true(reads_as_python("(?i)ab"))
    assert_false(reads_as_python("a(?i)b"))
    assert_false(python_answers("a(?i)(?=b)"))
    assert_true(python_answers("(?i)a(?=b)"))


def test_both_alphabets_at_once_is_read_as_a_pattern_that_did_not_parse() raises:
    """The one place where copying pandas exactly was not available.

    Python reports this by raising `ValueError`, pandas catches only parse
    errors, and the exception comes out of the caller's `str.contains` naming a
    module they never imported. Reading it as a failed parse routes it to Arrow,
    which is the same place every other failed parse goes.
    """
    assert_false(reads_as_python("(?a)(?u)"))
    assert_false(python_answers("(?a)(?u)"))
    assert_false(python_answers("(?a)(?u)(?=a)"))


def test_the_walk_can_be_asked_about_a_tree_that_is_already_parsed() raises:
    """The form the layer above uses, since it will have the tree in hand for
    the engine and should not parse the pattern twice."""
    assert_true(holds_unsupported(parse_pattern("(?=a)")))
    assert_false(holds_unsupported(parse_pattern("a")))
    assert_false(holds_unsupported(parse_pattern("(a")))


def test_a_pattern_python_refuses_goes_to_arrow_however_it_was_read() raises:
    """The constructs this library reads out of RE2's grammar all route the way
    a pattern Python cannot read routes, which is to Arrow.

    A tree that exists is not a tree pandas had. Documents 102 through 108 each
    added a construct that parses here and does not parse there, and each one of
    them can be written beside a lookaround, which is the pairing that makes
    this a routing question rather than a bookkeeping one.
    `(?P<n>\\p{Lu})(?P=n)` holds a backreference the walk can see and is a
    pattern pandas hands to Arrow without ever reaching the walk.
    """
    for one in [
        String("a(?i)b"),
        String("\\p{Lu}"),
        String("^*"),
        String("\\Qa+b\\E"),
        String("\\12"),
        String("(?P<1n>b)"),
        String("(?P<n>a)(?P<n>b)"),
        String("[\\d-a]"),
    ]:
        assert_false(reads_as_python(one))
        assert_false(python_answers(one))
        assert_false(python_answers(one + "(?=x)"))
        assert_false(holds_unsupported(parse_pattern(one + "(?=x)")))
    assert_false(python_answers("(?P<n>\\p{Lu})(?P=n)"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
