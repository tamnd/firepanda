"""A lookbehind on the engine that copies Python.

The other half of the construct next door, and the half that needed something
the first one did not. A lookahead starts its body where the thread is standing
and so can be answered without knowing anything about the body at all. A
lookbehind has to start the body far enough back that it ends where the thread
is standing, which means the compiler has to work out how many characters the
body always reads, and has to refuse the bodies that do not always read the same
number. Upstream refuses exactly those, with `look-behind requires fixed-width
pattern`, so the refusal here is agreement rather than a shortfall and is not
flagged as a gap.

336 of the 30052 held out patterns are this construct. The rows below were
measured against a running Python 3.13 and the refusal set was diffed against it
shape by shape, because which bodies Python calls fixed width is a rule with
several edges in it and every one of them is a place to be wrong: an alternation
is fixed only when its arms agree, a repeat is fixed only when its two bounds
are the same number, and an assertion of any kind is fixed at nothing.

Document 94.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
)

from firepanda.kernel.regex.backtrack import held_text
from firepanda.kernel.regex.count import counted_python_text
from firepanda.kernel.regex.method import METHOD_COUNT, program_for
from firepanda.kernel.regex.parse import parse_pattern
from firepanda.kernel.regex.program import compile_program
from firepanda.kernel.regex.replace import (
    parse_rewrite_python,
    replaced_python_text,
)
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


def hits(pattern: StringSlice, text: StringSlice) raises -> Bool:
    """Whether a pattern matches somewhere in a text, on Python's engine.

    Through the door that picks between the two machines rather than through
    the machine, because a lookbehind standing beside a backreference, an
    atomic group or a conditional group is read by the backtracker and one
    standing alone is read by the machine next door. Document 120.

    Args:
        pattern: The pattern.
        text: The text.

    Returns:
        True when it matches.

    Raises:
        Error: If the row ran out of steps, which no row in this file does.
    """
    var program = compile_program(parse_pattern(pattern), ENGINE_PYTHON)
    return held_text(program, text)


def found(pattern: StringSlice, text: StringSlice) raises -> Int:
    """How many times a pattern is found in a text, Python's way of counting.

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


def subbed(pattern: StringSlice, text: StringSlice) raises -> String:
    """A row with every match marked, Python's way of replacing.

    Args:
        pattern: The pattern.
        text: The row.

    Returns:
        The row with a sharp where each match was.

    Raises:
        Error: If the pattern did not compile, which in this file is a mistake
            in the test rather than an answer.
    """
    var program = compile_program(
        parse_pattern(pattern), ENGINE_PYTHON, captures=True
    )
    if not program.ok:
        raise Error(
            String("pattern ", pattern, " did not compile: ", program.problem)
        )
    var rewrite = parse_rewrite_python("#", program.groups, program.labels)
    return replaced_python_text(program, rewrite, text, -1)


def test_the_positive_form_asks_about_the_text_behind() raises:
    """`(?<=a)` at a position is true when an `a` ends there, and the position
    does not move, so `(?<=a)b` matches `ab` and reports a width of one."""
    assert_equal(said("(?<=a)b", ENGINE_PYTHON), "ok")
    assert_true(hits("(?<=a)b", "ab"))
    assert_false(hits("(?<=a)b", "cb"))
    assert_true(hits("(?<=ab)c", "abc"))
    assert_false(hits("(?<=ab)c", "zbc"))


def test_the_negative_form_is_the_same_question_answered_the_other_way() raises:
    """One bit on one instruction rather than a second instruction, which is the
    whole of the difference between the two forms, and the same arrangement the
    lookahead uses."""
    assert_equal(said("(?<!a)b", ENGINE_PYTHON), "ok")
    assert_false(hits("(?<!a)b", "ab"))
    assert_true(hits("(?<!a)b", "cb"))
    assert_true(hits("(?<!a)b", "b"))


def test_a_body_wider_than_the_text_behind_is_a_no_without_a_machine() raises:
    """There is no position to start the body from, so the positive form is
    False and the negative form is True, and neither of them runs anything. The
    row that shows it is the negative one: `(?<!ab)` at the front of a row holds
    because nothing can be behind the front."""
    assert_false(hits("(?<=ab)c", "bc"))
    assert_true(hits("(?<!ab)c", "bc"))
    assert_equal(found("(?<!ab)", "b"), 2)


def test_the_body_has_to_end_where_the_pattern_has_got_to() raises:
    """The row that separates a lookbehind from a search for the same body
    anywhere earlier. `abc` holds an `a`, so a pattern that looked for one
    anywhere behind would say yes to `(?<=a)c` and it says no."""
    assert_false(hits("(?<=a)c", "abc"))
    assert_true(hits("(?<=b)c", "abc"))


def test_the_body_reads_the_whole_row_rather_than_a_piece_of_it() raises:
    """A lookbehind body is run against the real text from a position in it,
    not against a piece of it cut off anywhere, so an anchor inside one asks
    about the row it is really in. `(?<=^a)` holds at position one of `ab` and
    at no position of `ba`."""
    assert_true(hits("(?<=^a)b", "ab"))
    assert_false(hits("(?<=^b)a", "cba"))
    assert_false(hits("(?<=a$)b", "ab"))


def test_a_body_that_reads_nothing_is_a_width_of_nothing() raises:
    """Which is what keeps `(?<=^)` and `(?<=\\b)` from being special cases.
    They are bodies of width zero, so the second machine is seeded where the
    thread already is, and an assertion inside a wider body adds nothing to the
    width rather than refusing it."""
    assert_equal(said("(?<=)a", ENGINE_PYTHON), "ok")
    assert_true(hits("(?<=^)a", "ab"))
    assert_false(hits("(?<=^)b", "ab"))
    assert_equal(subbed("(?<=\\b)\\w", "ab cd"), "#b #d")
    assert_equal(said("(?<=(?=a)a)b", ENGINE_PYTHON), "ok")
    assert_true(hits("(?<=(?=a)a)b", "ab"))


def test_a_lookbehind_can_hold_a_lookbehind() raises:
    """The second machine runs the same instruction as the one outside it, so
    the nesting works by recursion rather than by a case, and the inner one is
    seeded from a position the outer one worked out."""
    assert_equal(said("(?<=(?<=a)b)c", ENGINE_PYTHON), "ok")
    assert_true(hits("(?<=(?<=a)b)c", "abc"))
    assert_false(hits("(?<=(?<=a)b)c", "zbc"))


def test_the_bodies_python_calls_fixed_width_are_the_ones_that_compile() raises:
    """The refusal set, diffed shape by shape against Python 3.13.

    An alternation is fixed when every arm is fixed at the same number, so
    `a|b` reads and `a|bc` does not. A repeat is fixed when its two bounds are
    the same number, so `a{2}` and `a{2,2}` read and `a{2,3}`, `a?`, `a*`, `a+`
    and `a*?` do not. A star anywhere inside a sequence takes the whole body
    with it, which is `ab*c`.
    """
    assert_equal(said("(?<=a|b)c", ENGINE_PYTHON), "ok")
    assert_equal(said("(?<=ab|cd)e", ENGINE_PYTHON), "ok")
    assert_equal(said("(?<=a{2})b", ENGINE_PYTHON), "ok")
    assert_equal(said("(?<=a{2,2})b", ENGINE_PYTHON), "ok")
    assert_equal(said("(?<=a{0})b", ENGINE_PYTHON), "ok")
    assert_equal(said("(?<=a{2}b{3})c", ENGINE_PYTHON), "ok")
    assert_equal(said("(?<=(?:a){2})b", ENGINE_PYTHON), "ok")
    var refused = String(
        "!a lookbehind wants a body that always reads the same number of"
        " characters"
    )
    assert_equal(said("(?<=a|bc)d", ENGINE_PYTHON), refused)
    assert_equal(said("(?<=a?)b", ENGINE_PYTHON), refused)
    assert_equal(said("(?<=a*)b", ENGINE_PYTHON), refused)
    assert_equal(said("(?<=a+)b", ENGINE_PYTHON), refused)
    assert_equal(said("(?<=a*?)b", ENGINE_PYTHON), refused)
    assert_equal(said("(?<=a{2,3})b", ENGINE_PYTHON), refused)
    assert_equal(said("(?<=ab*c)d", ENGINE_PYTHON), refused)
    assert_equal(said("(?<=a(b|cd))e", ENGINE_PYTHON), refused)


def test_the_width_refusal_is_agreement_and_not_a_shortfall() raises:
    """Which is the flag rather than the words. Python raises for these too, so
    a caller who hears this has written a pattern nothing answers, and telling
    them it is not implemented yet would be telling them to wait for something
    that is never coming. Every other refusal on this engine is a gap and this
    one is not."""
    var program = compile_program(parse_pattern("(?<=a*)b"), ENGINE_PYTHON)
    assert_false(program.ok)
    assert_false(program.gap)
    var routed = program_for(METHOD_COUNT, "(?<=a*)b", 0, False, 14)
    assert_false(routed.ok)
    assert_false(routed.gap)


def test_the_counting_and_replacing_scans_read_one_the_same_way() raises:
    """A lookbehind has no width of its own, so a pattern made only of one
    matches with no width at every position it holds at, and the scan rule
    document 93 wrote down is what decides where the next one is looked for."""
    assert_equal(found("(?<=a)", "aaa"), 3)
    assert_equal(subbed("(?<=a)", "aaa"), "a#a#a#")
    assert_equal(found("(?<!a)", "aaa"), 1)
    assert_equal(subbed("(?<!a)", "aaa"), "#aaa")
    assert_equal(found("(?<=ab)c", "abcabc"), 2)
    assert_equal(subbed("(?<=ab)c", "abcabc"), "ab#ab#")


def test_the_scan_counts_characters_rather_than_bytes_behind_it() raises:
    """The width is a number of characters and the position it is taken from is
    a position in characters, so a row with a two byte character in it is
    counted back over the way Python counts it. A width in bytes would land in
    the middle of the sharp s and read nothing at all."""
    assert_true(hits("(?<=ß)x", "ßx"))
    assert_equal(subbed("(?<=ß)x", "ßx"), "ß#")
    assert_equal(subbed("(?<=.)x", "ßx"), "ß#")


def test_a_group_inside_one_keeps_what_it_matched() raises:
    """The same rule the lookahead has and by the same route, which is that the
    second machine carries slots now and hands them back. Both directions are
    one function and neither of them knows which one it is answering, so the
    only way this could have gone differently is if the width arithmetic had
    put the body somewhere else, and the row below is what says it has not."""
    assert_equal(said("(?<=(a))b", ENGINE_PYTHON, False), "ok")
    assert_equal(said("(?<=(a))b", ENGINE_PYTHON, True), "ok")
    assert_equal(said("(?<=(?:a))b", ENGINE_PYTHON, True), "ok")
    assert_equal(said("(a)(?<=a)", ENGINE_PYTHON, True), "ok")
    assert_true(hits("(?<=(a))b", "ab"))
    assert_false(hits("(?<=(a))b", "cb"))


def test_re2_still_refuses_the_construct_because_upstream_does() raises:
    """Nothing here reaches RE2. A pattern holding a lookaround is routed to
    Python by the router, for the same reason pandas routes it, and one that
    somehow arrived would be refused in RE2's voice rather than in this
    engine's."""
    assert_equal(said("(?<=a)b", ENGINE_RE2), "!RE2 has no lookaround")
    assert_equal(said("(?<!a)b", ENGINE_RE2), "!RE2 has no lookaround")
    assert_equal(said("(?<=a*)b", ENGINE_RE2), "!RE2 has no lookaround")


def test_an_unflagged_call_is_still_routed_to_python_by_the_construct() raises:
    """The router looks for an assertion by name and finds one here, which is
    what sends the pattern to the engine that can answer it. That decision is
    made before the compiler is asked anything, which is why the width refusal
    above arrives in this library's voice rather than RE2's."""
    var program = program_for(METHOD_COUNT, "(?<=a)b", 0, False, 14)
    assert_true(program.ok)
    assert_true(program.python)
    assert_true(program.slots > 0)


def test_the_pairings_that_used_to_be_refused_run_here_too() raises:
    """A lookbehind is the same walk as a lookahead started earlier, so a
    backreference, an atomic group or a conditional beside one meets it the
    same way. The rows are separate from the lookahead's because the body of a
    lookbehind is entered at a position the outside walk has already gone past,
    and a slot written there is one the outside walk has to be able to undo.
    Document 120."""
    assert_true(hits("(?<=(a))b\\1", "aba"))
    assert_false(hits("(?<=(a))b\\1", "abc"))
    assert_true(hits("(?<=(?>a))b", "ab"))
    assert_true(hits("(?<=a(?>b))c", "abc"))
    assert_true(hits("(?<=(a))(?(1)b|c)", "ab"))
    assert_false(hits("(?<=(a))(?(1)b|c)", "zc"))
    assert_true(hits("(?<!(a))b(?(1)x|y)", "cby"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
