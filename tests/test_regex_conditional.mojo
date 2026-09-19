"""Asking whether a group took part, on the engine that copies Python.

The third construct that decides which engine answers rather than what the
answer is, and it decides it for the first reason rather than the second. An
atomic group is an answer the state cache and the Pike machine cannot give,
because it throws away choices and there are no choices sitting anywhere in a
walk that is following all of them at once. A conditional is a question they
cannot answer, which is what a backreference is: whether a group took part is a
fact about the path that arrived, and two threads merged at the same
instruction and the same position arrived by different paths.

What it costs and what it does not are worth keeping apart. It costs the bitmap,
exactly as a backreference does, since a position is no longer enough to say
what happens next. It does not cost a second reading of the text, since the
question is settled by looking at two slots and going one way or the other, and
it costs no backtracking at all, since both arms are written out and exactly one
of them is entered.

Reaching it from pandas is the flags path and nothing else, since pandas routes
a pattern with neither a lookaround nor a backreference in it to Arrow and Arrow
refuses this too. The rows below were measured against a running Python 3.13.

Document 100.
"""

from std.collections.span import Span
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
)

from firepanda.kernel.regex.backtrack import Bounded
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


def first(pattern: StringSlice, text: StringSlice) raises -> String:
    """The leftmost match of a pattern in a text, and where it began.

    Written this way rather than as a yes or no because every interesting row
    here is about which arm was taken, and the two arms are usually different
    text starting at different places. A conditional that matched nothing at all
    and one that did not match are two answers a boolean cannot tell apart.

    Args:
        pattern: The pattern.
        text: The row.

    Returns:
        The matched text, an at sign and the position it began at, or a dash
        when nothing matched.

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
        return String("-")
    var opened = Int(slots[0])
    var out = String()
    for i in range(opened, Int(slots[1])):
        out += chr(Int(points[i]))
    return String(out, "@", opened)


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


def test_the_two_arms_are_chosen_by_a_group_rather_than_by_the_text() raises:
    """Which is the whole of the construct. `(a)?(?(1)b|c)` reads `ab` where
    the group took part and `c` where it did not, and the two rows that show it
    is really the group being asked about are `ac` and `c`. Against `ac` the
    match does not begin at nought at all: the group takes the `a`, the first
    arm wants a `b` and there is none, the group is given back so that it never
    took part, the second arm wants a `c` and the `c` is at one. So the search
    moves on and matches the second arm alone, one along."""
    assert_equal(said("(a)?(?(1)b|c)", ENGINE_PYTHON), "ok")
    assert_equal(first("(a)?(?(1)b|c)", "ab"), "ab@0")
    assert_equal(first("(a)?(?(1)b|c)", "ac"), "c@1")
    assert_equal(first("(a)?(?(1)b|c)", "b"), "-")
    assert_equal(first("(a)?(?(1)b|c)", "c"), "c@0")
    assert_equal(first("(a)?(?(1)b|c)", "abc"), "ab@0")
    assert_equal(first("(a)?(?(1)b|c)", ""), "-")
    assert_equal(first("(a)?(?(1)b|c)", "aab"), "ab@1")
    assert_equal(first("(a)?(?(1)b|c)", "xa"), "-")
    assert_equal(first("(a)?(?(1)b|c)", "bx"), "-")


def test_a_group_that_must_take_part_leaves_only_the_first_arm() raises:
    """`(a)(?(1)b|c)` has no question mark on the group, so the group takes part
    at every position the pattern can begin at and the second arm is written for
    nothing. That is not a thing to optimise away, because upstream compiles it
    and a caller who wrote it wants the same answer here."""
    assert_equal(first("(a)(?(1)b|c)", "ab"), "ab@0")
    assert_equal(first("(a)(?(1)b|c)", "ac"), "-")
    assert_equal(first("(a)(?(1)b|c)", "abc"), "ab@0")
    assert_equal(first("(a)(?(1)b|c)", "aab"), "ab@1")
    assert_equal(first("(a)(?(1)b|c)", "c"), "-")


def test_one_arm_is_a_conditional_that_can_always_go_on() raises:
    """A conditional written with no second arm has the test jump to the end of
    the construct, so a group that did not take part costs nothing and the
    pattern carries on. `(a)?(?(1)b)` therefore matches the empty string
    anywhere, which is what upstream does with it, and the only place it matches
    anything at all is in front of an `ab`."""
    assert_equal(first("(a)?(?(1)b)", "ab"), "ab@0")
    assert_equal(first("(a)?(?(1)b)", "ac"), "@0")
    assert_equal(first("(a)?(?(1)b)", "b"), "@0")
    assert_equal(first("(a)?(?(1)b)", ""), "@0")
    assert_equal(first("(a)?(?(1)b)", "xa"), "@0")


def test_both_arms_can_be_empty_and_one_of_them_can_be() raises:
    """`(?(1))` is a conditional with an empty first arm and no second one,
    which the parser reads as a construct rather than as a mistake, so it
    compiles to a test that goes to the same place either way. `(a)?(?(1)|c)` is
    the interesting half of that: the group taking part is the way out and the
    group not taking part still has to read a `c`."""
    assert_equal(first("(a)?(?(1))", "ab"), "a@0")
    assert_equal(first("(a)?(?(1))", "b"), "@0")
    assert_equal(first("(a)?(?(1))", ""), "@0")
    assert_equal(first("(a)?(?(1)|c)", "ab"), "a@0")
    assert_equal(first("(a)?(?(1)|c)", "ac"), "a@0")
    assert_equal(first("(a)?(?(1)|c)", "b"), "-")
    assert_equal(first("(a)?(?(1)|c)", "c"), "c@0")
    assert_equal(first("(a)?(?(1)|c)", "xa"), "a@1")


def test_a_group_can_be_named_or_nested_and_the_number_is_the_same() raises:
    """The test carries the first slot of the group, which is the group's number
    doubled, and a name is turned into a number by the parser long before the
    compiler sees it. So `(?P<n>a)?(?(n)b|c)` is the numbered pattern and
    `((a))?(?(2)b|c)` asks about the inner group of a nested pair, which for
    these texts takes part exactly when the outer one does."""
    assert_equal(first("(?P<n>a)?(?(n)b|c)", "ab"), "ab@0")
    assert_equal(first("(?P<n>a)?(?(n)b|c)", "ac"), "c@1")
    assert_equal(first("(?P<n>a)?(?(n)b|c)", "c"), "c@0")
    assert_equal(first("((a))?(?(2)b|c)", "ab"), "ab@0")
    assert_equal(first("((a))?(?(2)b|c)", "ac"), "c@1")
    assert_equal(first("((a))?(?(2)b|c)", "c"), "c@0")


def test_a_group_the_pattern_has_not_reached_yet_has_not_taken_part() raises:
    """`(?(1)a|b)(x)` asks about a group that is opened after it, and the answer
    is the second arm every time. Nothing in the compiler says so: the slots
    start at minus one and the test reads them, so a group that has not been
    reached reads exactly like one that was skipped. Upstream agrees."""
    assert_equal(first("(?(1)a|b)(x)", "bx"), "bx@0")
    assert_equal(first("(?(1)a|b)(x)", "ax"), "-")
    assert_equal(first("(?(1)a|b)(x)", "x"), "-")
    assert_equal(first("(?(1)a|b)(x)", "abx"), "bx@1")


def test_a_group_that_matched_nothing_has_still_taken_part() raises:
    """Which is the difference between a slot pair that was written and a slot
    pair that holds something, and the test asks the first question. `(a*)`
    matches the empty string at the front of `b`, so the group took part, so
    `(a*)(?(1)b|c)` takes the first arm and matches the `b`. Against `ac` there
    is no way to make the group not take part, so the `c` is never reachable."""
    assert_equal(first("(a*)(?(1)b|c)", "ab"), "ab@0")
    assert_equal(first("(a*)(?(1)b|c)", "ac"), "-")
    assert_equal(first("(a*)(?(1)b|c)", "b"), "b@0")
    assert_equal(first("(a*)(?(1)b|c)", "c"), "-")
    assert_equal(first("(a*)(?(1)b|c)", "aab"), "aab@0")
    assert_equal(first("(a*)(?(1)b|c)", "bx"), "b@0")
    assert_equal(caught("(a*)(?(1)b|c)", "b", 1), "")


def test_two_paths_to_the_same_place_can_answer_it_differently() raises:
    """This is the row the bitmap had to be given up for. `(?:(a)|a)(?(1)b|c)`
    against `ac` has two ways to read the `a`: the first arm sets group one and
    the second arm does not. Both arrive at the test at position one. The first
    one gets there first, takes the first arm, wants a `b`, and fails. If the
    arrival were remembered the second one would be dropped as a repeat of a
    question already answered, and the `c` would never be read.

    So `memo` is off for a program holding a test, exactly as it is for one
    holding a backreference, and the answer here is `ac` rather than nothing.
    Document 100 section 5."""
    assert_equal(first("(?:(a)|a)(?(1)b|c)", "ab"), "ab@0")
    assert_equal(first("(?:(a)|a)(?(1)b|c)", "ac"), "ac@0")
    assert_equal(first("(?:(a)|a)(?(1)b|c)", "aab"), "ab@1")
    assert_equal(first("(?:(a)|a)(?(1)b|c)", "c"), "-")
    assert_equal(caught("(?:(a)|a)(?(1)b|c)", "ac", 1), "?")
    assert_equal(caught("(?:(a)|a)(?(1)b|c)", "ab", 1), "a")


def test_a_repeat_over_one_still_answers() raises:
    """A repeat with no bound over a body that can match nothing is a loop with
    no way out, and the bitmap is what usually stops it. With the bitmap off the
    step count underneath is what stops it, which is the arrangement a
    backreference already runs under, so `(a)?(?(1)b|c)*` answers rather than
    running away."""
    assert_equal(first("(a)?(?(1)b|c)*", "ab"), "ab@0")
    assert_equal(first("(a)?(?(1)b|c)*", "ac"), "a@0")
    assert_equal(first("(a)?(?(1)b|c)*", "b"), "@0")
    assert_equal(first("(a)?(?(1)b|c)*", "c"), "c@0")
    assert_equal(first("(a)?(?(1)b|c)*", "aab"), "a@0")


def test_the_groups_are_read_out_beside_the_arms() raises:
    """A test reads the slots and writes none, so the groups at the end of a
    match are the ones the arms left there. The two rows that matter are the two
    ways a group can fail to hold text: never opened, which reads as a question
    mark, and opened over nothing, which reads as the empty string."""
    assert_equal(caught("(a)?(?(1)b|c)", "ab", 1), "a")
    assert_equal(caught("(a)?(?(1)b|c)", "ac", 1), "?")
    assert_equal(caught("(?P<n>a)?(?(n)b|c)", "ac", 1), "?")
    assert_equal(caught("((a))?(?(2)b|c)", "ab", 2), "a")
    assert_equal(caught("((a))?(?(2)b|c)", "c", 1), "?")


def test_the_counting_and_replacing_scans_read_one() raises:
    """Both scans go through the same door the search does, so the answers are
    the ones upstream gives, including the empty match a one armed conditional
    leaves at the end of a row."""
    assert_equal(found("(a)?(?(1)b|c)", "abac"), 2)
    assert_equal(subbed("(a)?(?(1)b|c)", "abac"), "#a#")
    assert_equal(found("(a)?(?(1)b|c)", "acab"), 2)
    assert_equal(subbed("(a)?(?(1)b|c)", "acab"), "a##")
    assert_equal(found("(a)?(?(1)b)", "ab"), 2)
    assert_equal(subbed("(a)?(?(1)b)", "ab"), "##")
    assert_equal(found("(a*)(?(1)b|c)", "aabc"), 1)
    assert_equal(subbed("(a*)(?(1)b|c)", "aabc"), "#c")
    assert_equal(found("(?(1)a|b)(x)", "bxbx"), 2)
    assert_equal(found("(?:(a)|a)(?(1)b|c)", "acab"), 2)
    assert_equal(subbed("(?:(a)|a)(?(1)b|c)", "acab"), "##")
    assert_equal(found("(a)?(?(1)|c)", "ac"), 2)


def test_a_group_number_nothing_opens_is_not_a_pattern() raises:
    """`(?(2)a|b)` names a group the pattern does not have, and upstream refuses
    it at compile time with an invalid group reference. Here it does not reach
    the compiler at all: the parser turns it down, which is the same answer in
    this library's words."""
    assert_equal(
        said("(?(2)a|b)", ENGINE_PYTHON),
        "!Python's grammar cannot read this pattern",
    )
    assert_equal(
        said("(?(1)a|b)", ENGINE_PYTHON),
        "!Python's grammar cannot read this pattern",
    )


def test_a_reference_and_a_test_run_on_the_same_engine() raises:
    """They arrive at the backtracker for the same reason, so a pattern holding
    both compiles and runs, and neither of them makes the other any worse: the
    bitmap was already off for either one alone."""
    assert_equal(said("(a)?(?(1)b|c)(d)\\2", ENGINE_PYTHON), "ok")
    assert_equal(first("(a)?(?(1)b|c)(d)\\2", "abdd"), "abdd@0")
    assert_equal(first("(a)?(?(1)b|c)(d)\\2", "acdd"), "cdd@1")
    assert_equal(first("(a)?(?(1)b|c)(d)\\2", "add"), "-")
    assert_equal(said("(?>(a))?(?(1)b|c)", ENGINE_PYTHON), "ok")
    assert_equal(first("(?>(a))?(?(1)b|c)", "ab"), "ab@0")
    assert_equal(first("(?>(a))?(?(1)b|c)", "ac"), "c@1")


def test_the_other_two_engines_turn_one_down() raises:
    """The state cache refuses a program holding a test in its own words, which
    with the machine dropping the thread is what makes the backtracker the only
    engine such a program can reach. The compiler says so too, on the flag it
    sets rather than in a sentence, and the flag it sets is the new one rather
    than either of the two beside it."""
    var program = compile_program(
        parse_pattern("(a)?(?(1)b|c)"),
        ENGINE_PYTHON,
        captures=True,
        alphabet=True,
    )
    assert_true(program.ok)
    assert_true(program.asks)
    assert_false(program.refs)
    assert_false(program.cuts)
    var cache = Cache(program)
    assert_false(cache.ok)
    assert_equal(cache.problem, "the pattern asks whether a group took part")
    var bounded = Bounded(program)
    assert_true(bounded.ok)


def test_the_slots_are_kept_whatever_the_caller_asked_for() raises:
    """A test reads a slot, so a slot that was never written is not a thing to
    guess about, and the compiler turns the groups on for any pattern holding
    one however the call was made. That is the same rule a backreference gets
    and it is the same line of code that gives it."""
    var plain = compile_program(parse_pattern("(a)?(?(1)b|c)"), ENGINE_PYTHON)
    assert_true(plain.ok)
    assert_true(plain.asks)
    assert_equal(first("(a)?(?(1)b|c)", "ac"), "c@1")


def test_re2_still_has_no_conditional_group() raises:
    """Which is what pandas depends on, since a pattern with no lookaround and
    no backreference in it is one pandas hands to Arrow and Arrow is RE2. The
    refusal is the one that was there before this and it is still word for word
    the one a caller sees."""
    assert_equal(
        said("(a)(?(1)b|c)", ENGINE_RE2), "!RE2 has no conditional group"
    )
    assert_equal(
        said("(?(1)a|b)(x)", ENGINE_RE2), "!RE2 has no conditional group"
    )


def test_a_lookaround_beside_one_is_still_refused() raises:
    """For the reason a lookaround beside a backreference is. A lookaround is a
    search inside a search and the backtracker has one stack to run it on, so
    the program goes to the machine, and the machine cannot answer a test. The
    sentence names the conditional, and a pattern holding a backreference as
    well is named by the older construct, since a person told about either one
    has been told what to take out."""
    assert_equal(
        said("(?=a)(a)?(?(1)b|c)", ENGINE_PYTHON),
        "!this engine has no lookaround beside a conditional group yet",
    )
    assert_equal(
        said("(?<=a)(b)?(?(1)c|d)", ENGINE_PYTHON),
        "!this engine has no lookaround beside a conditional group yet",
    )
    assert_equal(
        said("(?=a)(a)?(?(1)b|c)(d)\\2", ENGINE_PYTHON),
        "!this engine has no lookaround beside a backreference yet",
    )


def test_a_conditional_inside_a_lookbehind_is_measured_for_width() raises:
    """A lookbehind has to know how wide it is, and a conditional is as wide as
    its arms when they agree and refused when they do not. That refusal is
    upstream's and it is a `ValueError`, which is a different answer from the
    gap the two of them together get, so the width is asked even though a
    program that passes the width goes on to be refused for the other reason.
    Document 100 section 7."""
    assert_equal(
        said("(?P<n>a)(?<=(?(1)b|c))", ENGINE_PYTHON),
        "!this engine has no lookaround beside a conditional group yet",
    )
    assert_equal(
        said("(?P<n>a)(?<=(?(1)b|cc))", ENGINE_PYTHON),
        (
            "!a lookbehind wants a body that always reads the same number of"
            " characters"
        ),
    )
    assert_equal(
        said("(?P<n>a)(?<=(?(1)b))", ENGINE_PYTHON),
        (
            "!a lookbehind wants a body that always reads the same number of"
            " characters"
        ),
    )


def test_the_router_sends_one_to_pythons_engine_only_on_the_flags_path() raises:
    """Which is what pandas does and this library follows it. The walk over the
    tree that picks the engine looks for a lookaround and a backreference and
    nothing else, so a pattern whose only unusual thing is a conditional is
    routed to RE2 and refused there, exactly as pandas hands it to Arrow. Naming
    a flag is what moves the call, because a flag is what moves pandas onto
    `re`."""
    var routed = program_for(METHOD_CONTAINS, "(a)(?(1)b|c)", 0, False, 14)
    assert_false(routed.ok)
    assert_equal(routed.problem, "RE2 has no conditional group")
    var argued = program_for(METHOD_CONTAINS, "(a)(?(1)b|c)", 0, True, 14)
    assert_true(argued.ok)
    assert_true(argued.asks)
    var counted = program_for(METHOD_COUNT, "(a)?(?(1)b|c)", 0, True, 14)
    assert_true(counted.ok)
    assert_true(counted.asks)
    var pulled = program_for(METHOD_EXTRACT, "(a)?(?(1)b|c)", 0, False, 14)
    assert_true(pulled.ok)
    assert_true(pulled.asks)


def test_the_scan_reads_characters_rather_than_bytes() raises:
    """A test is about the slots rather than about the text, so there is nothing
    here that could read a byte where a character was meant. The row is here so
    that the next person to change the arms has a two byte character in front of
    them."""
    assert_equal(first("(ß)?(?(1)x|y)", "ßx"), "ßx@0")
    assert_equal(first("(ß)?(?(1)x|y)", "ßy"), "y@1")
    assert_equal(subbed("(ß)?(?(1)x|y)", "ßxßy"), "#ß#")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
