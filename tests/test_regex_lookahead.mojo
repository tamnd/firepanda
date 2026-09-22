"""A lookahead on the engine that copies Python.

RE2 has never had a lookaround and pandas knows it, so a pattern holding one is
routed away from Arrow and answered by `re`. This library routed it the same way
and then refused it, which made every one of those patterns a column a caller
did not get. 700 of the 30052 held out patterns are this construct, which is the
largest single thing the Python engine was turning down after the flags landed.

The half here is the lookahead. The lookbehind is a different question and has a
file of its own beside this one, because reading one means knowing the width of
the body and refusing a body that has not got one. Documents 93 and 94.

The rows below ask the compiler and the machine directly rather than through the
accessor, for the reason the `\\z` slice found: the accessor cannot reach every
method on Python's engine, and a test that can only ask the questions the door
allows is a test that stops where the door does.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
)

from firepanda.kernel.regex.method import (
    METHOD_CONTAINS,
    METHOD_COUNT,
    METHOD_EXTRACT,
    METHOD_MATCH,
    program_for,
)
from firepanda.kernel.regex.parse import parse_pattern
from firepanda.kernel.regex.count import counted_python_text
from firepanda.kernel.regex.pike import matches_text
from firepanda.kernel.regex.program import compile_program
from firepanda.kernel.regex.route import ENGINE_PYTHON, ENGINE_RE2


def said(pattern: StringSlice, engine: UInt8, captures: Bool = False) -> String:
    """What the compiler says about a pattern on an engine.

    Args:
        pattern: The pattern.
        engine: Which engine.
        captures: Whether the caller asked for the groups.

    Returns:
        The word ok, or an exclamation mark and the problem.
    """
    var program = compile_program(parse_pattern(pattern), engine, captures, 14)
    if program.ok:
        return String("ok")
    return String("!", program.problem)


def hits(pattern: StringSlice, text: StringSlice) -> Bool:
    """Whether a pattern matches somewhere in a text, on Python's engine.

    Args:
        pattern: The pattern.
        text: The text.

    Returns:
        True when it matches.
    """
    var program = compile_program(parse_pattern(pattern), ENGINE_PYTHON)
    return matches_text(program, text)


def found(pattern: StringSlice, text: StringSlice) raises -> Int:
    """How many times a pattern is found in a text, Python's way of counting.

    The captures are on because Python's counting rule reads where a match
    started as well as where it ended, which is what tells it whether the match
    had any width in it.

    Args:
        pattern: The pattern.
        text: The text.

    Returns:
        The count.

    Raises:
        Error: If the row ran out of steps, which no row in this file does.
    """
    var program = compile_program(
        parse_pattern(pattern), ENGINE_PYTHON, captures=True
    )
    return counted_python_text(program, text)


def test_the_positive_form_asks_about_the_text_in_front() raises:
    """`(?=b)` at a position is true when a `b` starts there, and the position
    does not move, so `a(?=b)` matches `ab` and reports a width of one."""
    assert_equal(said("a(?=b)", ENGINE_PYTHON), "ok")
    assert_true(hits("a(?=b)", "ab"))
    assert_false(hits("a(?=b)", "ac"))
    assert_true(hits("(?=b)", "ab"))
    assert_false(hits("(?=b)", "aa"))


def test_the_negative_form_is_the_same_question_answered_the_other_way() raises:
    """One instruction with a bit on it rather than two instructions, which is
    the whole of the difference between the two forms here."""
    assert_equal(said("a(?!b)", ENGINE_PYTHON), "ok")
    assert_false(hits("a(?!b)", "ab"))
    assert_true(hits("a(?!b)", "ac"))
    assert_true(hits("a(?!b)", "a"))


def test_the_body_has_to_start_where_the_pattern_has_got_to() raises:
    """The row that separates a lookahead from a search for the same body. `ab`
    holds a `b`, so a pattern that only looked for one anywhere ahead would say
    yes to `(?=b)a` and it says no."""
    assert_false(hits("(?=b)a", "ab"))
    assert_true(hits("(?=a)a", "ab"))


def test_the_body_reads_the_whole_row_rather_than_the_rest_of_it() raises:
    """A lookahead body is run against the real text from the position, not
    against a piece of it cut off there, so an anchor inside one asks about the
    row it is really in."""
    assert_true(hits("a(?=b$)", "ab"))
    assert_false(hits("a(?=b$)", "abc"))
    assert_true(hits("(?=^a)a", "ab"))


def test_a_lookahead_can_hold_a_lookahead() raises:
    """The nested machine runs the same instruction as the one outside it, so
    the nesting works by recursion rather than by a case. `(?=a(?=b))` is a
    pattern upstream answers and there is no reason for it to be special
    here."""
    assert_equal(said("(?=a(?=b))ab", ENGINE_PYTHON), "ok")
    assert_true(hits("(?=a(?=b))ab", "ab"))
    assert_false(hits("(?=a(?=b))ab", "ac"))


def test_the_width_of_a_lookahead_is_nothing_so_the_count_says_so() raises:
    """A pattern that is only a lookahead matches with no width at every
    position it holds at, which is what `str.count` sees and why counting one
    is a different number from counting the body."""
    assert_equal(found("(?=a)", "aaa"), 3)
    assert_equal(found("(?=a)", "bbb"), 0)
    assert_equal(found("a(?=b)", "abab"), 2)


def test_re2_still_refuses_the_construct_because_upstream_does() raises:
    """The refusal on the other engine is agreement with pandas rather than a
    shortfall, since a pattern holding a lookaround never reaches Arrow at all
    and one that somehow did would raise there."""
    assert_equal(said("a(?=b)", ENGINE_RE2), "!RE2 has no lookaround")
    assert_equal(said("a(?!b)", ENGINE_RE2), "!RE2 has no lookaround")
    assert_equal(said("(?<=a)b", ENGINE_RE2), "!RE2 has no lookaround")


def test_a_lookbehind_is_the_other_half_and_answers_too() raises:
    """One row here rather than a file, since the other half has a file of its
    own next door. What it is doing in this one is saying that landing the
    second direction did not cost the first: both forms of both directions
    compile on the engine that copies Python and none of them compile on the
    other."""
    assert_equal(said("(?<=a)b", ENGINE_PYTHON), "ok")
    assert_equal(said("(?<!a)b", ENGINE_PYTHON), "ok")
    assert_true(hits("(?<=a)b", "ab"))
    assert_false(hits("(?<=a)b", "cb"))


def test_a_group_inside_one_keeps_what_it_matched() raises:
    """A group inside a lookahead keeps what it matched upstream, and it keeps
    it here, because the second machine carries slots and the thread outside
    takes back the ones the winning arm wrote. It used to be refused for a
    caller who asked for the groups and answered for everybody else, which is
    the shape the assertions below had until document 119."""
    assert_equal(said("(?=(a))a", ENGINE_PYTHON, False), "ok")
    assert_equal(said("(?=(a))a", ENGINE_PYTHON, True), "ok")
    assert_equal(said("(?=(?:a))a", ENGINE_PYTHON, True), "ok")
    assert_equal(said("(a)(?=b)", ENGINE_PYTHON, True), "ok")
    assert_true(hits("(?=(a))a", "ab"))
    assert_false(hits("(?=(a))a", "bb"))


def test_a_negative_lookaround_with_nothing_in_it_never_matches() raises:
    """The parser collapses `(?!)` into a node of its own, because upstream
    does, and the node compiles to a set with no members. RE2 never sees the
    pattern, since the router looks for assertions by name and this is not one,
    so the refusal there is untouched."""
    assert_equal(said("(?!)", ENGINE_PYTHON), "ok")
    assert_false(hits("(?!)", "a"))
    assert_false(hits("(?!)", ""))
    assert_true(hits("a|(?!)", "a"))
    assert_equal(
        said("(?!)", ENGINE_RE2), "!RE2 has no empty negative lookaround"
    )


def test_a_lookahead_still_travels_through_the_method_that_never_routes() raises:
    """`extract` is compiled for Python whatever the pattern holds, so it is the
    one method that reaches this without a flag in sight, and it is also the one
    that asks for captures. Both shapes compile since document 119, the group
    outside the lookahead and the group inside it, and the second of them is
    the one this method used to turn down."""
    var program = program_for(METHOD_EXTRACT, "(?=a)(b)", 0, False, 14)
    assert_true(program.ok)
    var inside = program_for(METHOD_EXTRACT, "(?=(a))b", 0, False, 14)
    assert_true(inside.ok)


def test_an_unflagged_contains_is_still_a_question_for_the_router() raises:
    """Nothing here changes which engine answers a pattern. The router sends a
    lookaround to Python because pandas does, and that decision is made before
    the compiler is asked anything at all."""
    var program = program_for(METHOD_CONTAINS, "a(?=b)", 0, False, 14)
    assert_true(program.ok)
    assert_true(program.python)


def test_a_routed_call_is_rewritten_the_way_an_argued_one_is() raises:
    """The branch that was correct only because it always refused.

    There are two ways a call reaches Python's engine, a `flags` argument and a
    construct, and they used to be two branches. The first anchored the pattern
    and asked for whatever slots the method needs. The second handed the
    caller's own pattern over unanchored and with no slots, which cost nothing
    while every construct was refused and costs two wrong answers now that one
    of them is not. `match` without the anchor is a search, and a counting scan
    without slot zero cannot tell a match of no width from any other.
    """
    var m = program_for(METHOD_MATCH, "a(?=b)", 0, False, 14)
    assert_true(m.ok)
    assert_true(m.python)
    assert_true(m.anchored)
    var c = program_for(METHOD_COUNT, "a(?=b)", 0, False, 14)
    assert_true(c.ok)
    assert_true(c.slots > 0)


def test_the_constructs_that_are_still_refused_are_refused_the_same_way() raises:
    """The other half of the branch above, which is that folding the two did not
    change what a caller hears about a construct that has not landed. The
    message names the construct rather than the rewrite, because the walk that
    finds it reads the whole tree and the anchors are not part of it."""
    var back = program_for(METHOD_MATCH, "(?=a)(b)\\1", 0, False, 14)
    assert_false(back.ok)
    assert_equal(
        back.problem, "this engine has no lookaround beside a backreference yet"
    )
    var atomic = program_for(METHOD_COUNT, "(?=a)(?>a)b", 0, False, 14)
    assert_false(atomic.ok)
    assert_equal(
        atomic.problem,
        "this engine has no lookaround beside an atomic group yet",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
