"""Throwing away a choice, on the engine that copies Python.

The second construct that decides which engine answers rather than what the
answer is, and it decides it for the opposite reason to the first. A
backreference is a question the state cache and the Pike machine cannot answer,
because a state there is a set of instructions and a reference asks what the
path took. A cut is an answer they cannot give: it says the ways the group could
have matched instead are gone, and there are no ways sitting anywhere to throw
away in a walk that is following all of them at once. The backtracker holds its
choices on a stack, so throwing some of them away is a thing you can do to it.

The possessive quantifier is the same construct spelled shorter. `a*+` is
`(?>a*)` and is compiled as exactly that, a mark, a greedy repeat and a cut, so
the rows below test one mechanism twice rather than two mechanisms once, which
is also how upstream reads it.

196 of the 30052 held out patterns are the atomic group and 393 are the
possessive quantifier, which together are the largest pair left after the named
character. They are reachable from pandas only through the flags path, since
pandas routes a pattern with neither a lookaround nor a backreference in it to
Arrow, and Arrow refuses both. The rows below were measured against a running
Python 3.13.

Document 99.
"""

from std.collections.span import Span
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
)

from firepanda.kernel.regex.backtrack import Bounded, held_text
from firepanda.kernel.regex.count import counted_python_text
from firepanda.kernel.regex.dfa import Cache
from firepanda.kernel.regex.method import (
    METHOD_CONTAINS,
    METHOD_COUNT,
    METHOD_EXTRACT,
    program_for,
)
from firepanda.kernel.regex.parse import decoded, parse_pattern
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


def caught(
    pattern: StringSlice, text: StringSlice, group: Int
) raises -> String:
    """What one group of the leftmost match holds, or a question mark.

    Args:
        pattern: The pattern.
        text: The row.
        group: Which group, counting from one.

    Returns:
        The group's text, a question mark when the group took no part, or an
        exclamation mark when the pattern did not match at all.

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
    var points = decoded(text)
    var bounded = Bounded(program)
    var slots = List[Int32]()
    if bounded.search(program, Span(points), 0, 0, slots) < 0:
        return String("!")
    var opened = Int(slots[2 * group])
    var closed = Int(slots[2 * group + 1])
    if opened < 0 or closed < opened:
        return String("?")
    var out = String()
    for i in range(opened, closed):
        out += chr(Int(points[i]))
    return out^


def test_a_group_keeps_the_first_way_it_matched() raises:
    """Which is the whole of the construct. `(?>a|ab)c` takes the first arm,
    reads `a`, and the cut throws the second arm away, so the `c` has nothing
    to come back to and the pattern fails where `(?:a|ab)c` would have
    succeeded. Written the other way round, `(?>ab|a)c`, the first arm is the
    one that leads to a match and there is nothing to come back for."""
    assert_equal(said("(?>a|ab)c", ENGINE_PYTHON), "ok")
    assert_false(hits("(?>a|ab)c", "abc"))
    assert_true(hits("(?>ab|a)c", "abc"))
    assert_true(hits("(?>a|ab)", "ab"))
    assert_false(hits("x(?>a|ab)c", "xabc"))


def test_a_repeat_inside_one_keeps_everything_it_took() raises:
    """A greedy repeat under a cut is the shape the construct is usually
    written for. `(?>a*)a` cannot match anything at all, because the repeat
    takes every `a` there is and the cut means it never gives one back, and
    `(?>a*)b` matches because there was nothing to give back in the first
    place."""
    assert_false(hits("(?>a*)a", "aaa"))
    assert_false(hits("(?>a*)ab", "aaab"))
    assert_true(hits("(?>a*)b", "aaab"))
    assert_true(hits("(?>a+)b", "aaab"))
    assert_true(hits("(?>a?)b", "ab"))
    assert_false(hits("(?>a?)ab", "ab"))
    assert_false(hits("^(?>a*)a$", "aaa"))
    assert_true(hits("(?>[^/]*)/", "aa/bb"))
    assert_false(hits("(?>\\w+)o", "hello"))
    assert_true(hits("(?>\\w+)\\s", "hello world"))


def test_the_possessive_quantifier_is_the_same_thing_written_shorter() raises:
    """`a*+` is `(?>a*)` and is compiled as that, a mark, a greedy repeat and a
    cut. So every row above has a twin here and the twin gives the same answer,
    which is the point of there being one mechanism rather than two. The
    counted form goes the same way, since a count under a cut is still a
    repeat."""
    assert_equal(said("a*+b", ENGINE_PYTHON), "ok")
    assert_false(hits("a*+a", "aaa"))
    assert_true(hits("a*+b", "aaab"))
    assert_true(hits("a++b", "aaab"))
    assert_true(hits("a?+b", "ab"))
    assert_false(hits("a?+ab", "ab"))
    assert_true(hits("a{2,}+b", "aaab"))
    assert_false(hits("a{2,}+ab", "aaab"))
    assert_false(hits("^a*+a$", "aaa"))
    assert_true(hits("\\d++\\.", "12.5"))


def test_one_inside_another_cuts_only_its_own_choices() raises:
    """The stack is marked where the group opened and the cut throws away
    everything above the nearest mark. That is the right mark without anything
    having to be numbered, because a group nested inside this one has either
    reached its own cut, which took its mark off, or failed, which popped its
    mark off, and either way it is gone before the outer one runs.

    The third row is the one that shows the cuts are separate. `(?>(?>ab|a)b)c`
    fails because the inner group takes `ab` and the inner cut means it never
    tries `a`, so the outer `b` has nothing to read."""
    assert_true(hits("(?>a(?>b)c)", "abc"))
    assert_true(hits("(?>a(?>b)c)d", "abcd"))
    assert_true(hits("(?>(?>a)|b)c", "bc"))
    assert_false(hits("(?>(?>ab|a)b)c", "abc"))


def test_a_choice_outside_the_group_survives_it() raises:
    """The mark is where the group opened, so a choice made before that is
    below the mark and the cut leaves it alone. `(?>a)?ab` matches `ab` because
    the question mark belongs to the pattern rather than to the group: the
    group is entered, matches `a`, and the arm that skips the group entirely
    was put on the stack before the mark was and is still there to be taken."""
    assert_true(hits("(?>a)?b", "ab"))
    assert_true(hits("(?>a)?ab", "ab"))
    assert_true(hits("(?>a|b)+c", "abac"))
    assert_true(hits("(?>)a", "a"))


def test_a_repeat_over_a_group_that_matches_nothing_still_answers() raises:
    """`(?>a*)*b` is a repeat with no bound on it over a body that can match
    nothing, which is a loop with no way out, and what stops it is the bitmap
    dropping the second arrival at an instruction and a position. An atomic
    group does not take that away the way a backreference does: what the group
    matches from a position is the same whichever path arrived there, so the
    second arrival really is the same question. Document 99 section 5."""
    assert_true(hits("(?>a*)*b", "aaab"))
    assert_true(hits("(?>a*)+b", "aaab"))
    assert_true(hits("(a*+)*b", "aaab"))


def test_the_groups_inside_one_are_still_read_out() raises:
    """A cut throws away the choices and keeps the saves, because a save is not
    a choice: it is what a slot held before the group wrote to it, and the
    group as a whole can still fail on what comes after it. So the slots are
    right at the match and are put back when there is no match."""
    assert_equal(caught("(?>(a+))b", "aab", 1), "aa")
    assert_equal(caught("(?>(a)|(b))c", "bc", 1), "?")
    assert_equal(caught("(?>(a)|(b))c", "bc", 2), "b")
    assert_equal(caught("(?>(a+)(b+))c", "aabbc", 1), "aa")
    assert_equal(caught("(?>(a+)(b+))c", "aabbc", 2), "bb")
    assert_equal(caught("(?>(a)b|(a))c", "ac", 1), "?")
    assert_equal(caught("(?>(a)b|(a))c", "ac", 2), "a")
    assert_equal(caught("(a)*+b", "aab", 1), "a")
    assert_equal(caught("((a)|b)*+c", "abc", 1), "b")
    assert_equal(caught("((a)|b)*+c", "abc", 2), "a")


def test_the_counting_and_replacing_scans_read_one() raises:
    """Both scans go through the same door the search does, so the answers are
    the ones upstream gives, including the empty matches a cut leaves behind
    when the repeat under it has taken everything it can."""
    assert_equal(found("(?>a*)", "aab"), 3)
    assert_equal(subbed("(?>a*)", "aab"), "##b#")
    assert_equal(found("a*+", "aab"), 3)
    assert_equal(subbed("a*+", "aab"), "##b#")
    assert_equal(found("(?>a)", "aaa"), 3)
    assert_equal(subbed("(?>a)", "aaa"), "###")
    assert_equal(found("(?>a|b)", "abab"), 4)
    assert_equal(found("(?>\\w)+", "ab cd"), 2)
    assert_equal(subbed("(?>\\w)+", "ab cd"), "# #")


def test_a_reference_and_a_cut_run_on_the_same_engine() raises:
    """They arrive at the backtracker for two different reasons and neither of
    them is in the other's way, so a pattern holding both compiles and runs.
    The bitmap runs under the backreference's narrower rule, which is the
    stricter of the two, and the cut is obeyed on the same stack."""
    assert_equal(said("(?>(a))\\1", ENGINE_PYTHON), "ok")
    assert_true(hits("(?>(a))\\1", "aa"))
    assert_true(hits("(?>(a)|(b))\\1", "aa"))
    assert_true(hits("(a)(?>\\1)b", "aab"))


def test_the_other_two_engines_turn_one_down() raises:
    """The state cache and the machine both refuse a program holding a cut, in
    their own words, which is the pair of refusals that makes the backtracker
    the only engine such a program can reach. The compiler says so too, on the
    flag it sets rather than in a sentence."""
    var program = compile_program(
        parse_pattern("(?>a*)b"), ENGINE_PYTHON, alphabet=True
    )
    assert_true(program.ok)
    assert_true(program.cuts)
    assert_false(program.refs)
    var cache = Cache(program)
    assert_false(cache.ok)
    assert_equal(
        cache.problem, "the pattern throws away a choice it could have made"
    )
    var bounded = Bounded(program)
    assert_true(bounded.ok)
    var possessive = compile_program(parse_pattern("a*+b"), ENGINE_PYTHON)
    assert_true(possessive.ok)
    assert_true(possessive.cuts)


def test_re2_still_has_neither_of_them() raises:
    """Which is what pandas depends on, since a pattern with no lookaround and
    no backreference in it is one pandas hands to Arrow and Arrow is RE2. The
    two refusals are the ones that were there before this and they are still
    word for word the ones a caller sees."""
    assert_equal(said("(?>a)b", ENGINE_RE2), "!RE2 has no atomic group")
    assert_equal(said("a*+b", ENGINE_RE2), "!RE2 has no possessive quantifier")


def test_a_lookaround_beside_one_runs_here_now() raises:
    """It used to be refused, because a lookaround is a search inside a search
    and the backtracker had one stack to run it on, so the program went to the
    machine and the machine cannot obey a cut. The body is walked on the same
    stack from the height it stood at now, so the two constructs sit beside
    each other and a cut inside a body reaches back to its own mark and no
    further. Document 120."""
    assert_equal(said("(?=a)(?>a)b", ENGINE_PYTHON), "ok")
    assert_equal(said("(?<=a)b*+", ENGINE_PYTHON), "ok")
    assert_equal(said("(?=a)(a)\\1", ENGINE_PYTHON), "ok")
    assert_true(hits("(?=ab)(?>a+)b", "ab"))
    assert_false(hits("(?=aa)(?>a+)a", "aa"))
    assert_true(hits("(?>a+)(?=b)", "aab"))
    assert_true(hits("(?<=(?>a))b", "ab"))
    assert_true(hits("(?<=a(?>b))c", "abc"))


def test_the_router_sends_one_to_pythons_engine_only_on_the_flags_path() raises:
    """Which is what pandas does and this library follows it. The walk over the
    tree that picks the engine looks for a lookaround and a backreference and
    nothing else, so a pattern whose only unusual thing is a cut is routed to
    RE2 and refused there, exactly as pandas hands it to Arrow. Naming a flag
    is what moves the call, because a flag is what moves pandas onto `re`."""
    var routed = program_for(METHOD_CONTAINS, "(?>a)b", 0, False, 14)
    assert_false(routed.ok)
    assert_equal(routed.problem, "RE2 has no atomic group")
    var argued = program_for(METHOD_CONTAINS, "(?>a)b", 0, True, 14)
    assert_true(argued.ok)
    assert_true(argued.cuts)
    var counted = program_for(METHOD_COUNT, "a*+b", 0, True, 14)
    assert_true(counted.ok)
    assert_true(counted.cuts)
    var pulled = program_for(METHOD_EXTRACT, "(?>(a+))b", 0, False, 14)
    assert_true(pulled.ok)
    assert_true(pulled.cuts)


def test_the_scan_reads_characters_rather_than_bytes() raises:
    """A cut is about the stack rather than about the text, so there is nothing
    here that could read a byte where a character was meant. The row is here so
    that the next person to change the stack has a two byte character in front
    of them."""
    assert_true(hits("(?>ß*)x", "ßßx"))
    assert_false(hits("(?>ß*)ß", "ßßß"))
    assert_equal(subbed("(?>ß+)", "ßßx"), "#x")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
