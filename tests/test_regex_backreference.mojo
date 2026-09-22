"""Reading back what a group matched, on the engine that copies Python.

The construct that decides which engine answers rather than what the answer is.
Every other one is a question about the program and the position, and both the
state cache and the Pike machine are built on that being the whole question: the
cache's state is a set of instructions and the machine merges two threads the
moment they stand at the same instruction and the same position. A backreference
asks what the path that arrived matched, so those two threads are no longer the
same question and neither of those engines can be asked. The backtracker keeps a
path, so it can, and it is the only one of the three that can.

That costs the bitmap most of what it is worth, for the same reason and in the
same sentence: a pair of an instruction and a position may be dropped because it
says everything about what is left to do, and here it does not. What does say
everything is the pair and the slots, so a program holding one of these keeps
its bitmap and forgets everything in it the moment a slot changes value, and is
bounded by a count of steps rather than by one visit per cell. It is the one
shape the backtracker never hands back.

839 of the 30052 held out patterns are this construct, which makes it the
largest of the five Python has and RE2 has not. The rows below were measured
against a running Python 3.13, including the folding ones, because a
backreference under the ignore case flag is compared by simple lowercase where a
literal is compared by the whole fold orbit and the two disagree on real text.

Documents 95 and 97.
"""

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
    METHOD_FULLMATCH,
    METHOD_MATCH,
    program_for,
    python_anchored,
)
from firepanda.kernel.regex.parse import parse_pattern
from firepanda.kernel.regex.pike import Machine
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


def test_a_reference_reads_the_text_the_group_matched() raises:
    """The plain shape. `(a)\\1` wants the group's own text again straight
    after it, so it matches `aa` and not `ab`, and the numbered spelling and
    the named one are the same instruction."""
    assert_equal(said("(a)\\1", ENGINE_PYTHON), "ok")
    assert_true(hits("(a)\\1", "aa"))
    assert_false(hits("(a)\\1", "ab"))
    assert_true(hits("(\\w)\\1", "abb"))
    assert_false(hits("(\\w)\\1", "abc"))
    assert_true(hits("(ab)\\1", "abab"))
    assert_false(hits("(ab)\\1", "abac"))
    assert_true(hits("(?P<x>a)(?P=x)", "aa"))
    assert_true(hits("(a)(b)\\2\\1", "abba"))
    assert_false(hits("(a)(b)\\2\\1", "abab"))


def test_a_reference_is_read_at_the_width_the_group_took() raises:
    """Which is not the width the pattern looks like it has. `(a*)\\1` reads
    back however many the group settled on, so the whole match is twice that,
    and the group gives characters back until the two halves agree."""
    assert_true(hits("^(a*)\\1$", "aaaa"))
    assert_false(hits("^(a*)\\1$", "aaa"))
    assert_true(hits("^(.)(.)\\2\\1$", "abba"))
    assert_false(hits("^(.)(.)\\2\\1$", "abab"))


def test_a_group_that_never_took_part_fails_the_reference() raises:
    """And a group that took part and matched nothing does not, which is the
    whole of the difference between `(a)?\\1b` and `(a?)\\1b`. In the first the
    group is skipped and its two ends are still unset, and upstream fails a
    reference to an unset group rather than treating it as empty. In the second
    the group ran and matched nothing, so the reference reads nothing and finds
    it."""
    assert_false(hits("(a)?\\1b", "b"))
    assert_true(hits("(a?)\\1b", "b"))
    assert_true(hits("(a)?\\1b", "aab"))
    assert_false(hits("(a)?\\1b", "ab"))


def test_a_loop_over_a_reference_that_matches_nothing_still_answers() raises:
    """Three patterns the corpus found, all of them the same shape: a group
    that matches nothing, read back under a repeat with no bound on it.

    A repeat over a body that matches nothing is a loop with no way out, and
    what stops it everywhere else in this library is the bitmap, which drops the
    second arrival at an instruction and a position. Turning the bitmap off for
    a backreference would turn that off with it, and these three would walk to
    the step bound and raise rather than answer.

    They answer because the bitmap is kept and forgotten on a slot changing
    rather than turned off, and a turn of a loop that matches nothing changes no
    slot. Python answers all three and these are its answers. Document 95
    section 3.
    """
    assert_true(hits("(?P<n>(|))(?P=n)*", "ab"))
    assert_equal(found("(?P<n>(|))(?P=n)*", "ab"), 3)
    assert_true(hits("(?P<n>\\b)(?P=n)*", "ab"))
    assert_equal(found("(?P<n>\\b)(?P=n)*", "ab"), 2)
    assert_false(hits("(?P<n>\\b)(?P=n)*", ""))
    assert_true(hits("(?P<n>\\b)(?P=n)+?[a-z]", "ab"))
    assert_false(hits("(?P<n>\\b)(?P=n)+?[a-z]", "12"))


def test_the_ascii_reading_of_the_ignore_case_flag_folds_the_letters() raises:
    """Under `(?ai)` the comparison is the twenty six letters and nothing else,
    which is a subtraction rather than a table. It is a rule of its own rather
    than the wide rule narrowed down, so the last two rows are the ones the
    lowercase table would have answered the other way."""
    assert_equal(said("(?ai)(a)\\1", ENGINE_PYTHON), "ok")
    assert_true(hits("(?ai)(a)\\1", "aA"))
    assert_true(hits("(?ai)(a)\\1", "Aa"))
    assert_true(hits("(?ai)(a)\\1", "aa"))
    assert_false(hits("(?ai)(a)\\1", "ab"))
    assert_true(hits("(?ai)(ab)\\1", "abAB"))
    assert_false(hits("(?ai)(k)\\1", "kK"))
    assert_false(hits("(?ai)(é)\\1", "éÉ"))


def test_the_wide_ignore_case_flag_lowers_both_characters() raises:
    """Under `(?i)` upstream compares the two by simple lowercase, which is the
    table in lowerdata.mojo and is read while the row is walked rather than
    spent while the pattern is compiled. The rows here are the ones where that
    table has something to say: the Kelvin sign lowers onto a `k`, the dotted
    capital I onto an `i`, and the capital sharp s onto the small one."""
    assert_equal(said("(?i)(a)\\1", ENGINE_PYTHON), "ok")
    assert_true(hits("(?i)(a)\\1", "aA"))
    assert_true(hits("(?i)(a)\\1", "Aa"))
    assert_false(hits("(?i)(a)\\1", "ab"))
    assert_true(hits("(?i)(ab)\\1", "abAB"))
    assert_true(hits("(?i)(k)\\1", "kK"))
    assert_true(hits("(?i)(İ)\\1", "İi"))
    assert_true(hits("(?i)(é)\\1", "éÉ"))
    assert_true(hits("(?i)(ß)\\1", "ßẞ"))
    assert_true(hits("(?i)(А)\\1", "Аа"))


def test_a_reference_is_lowered_where_a_literal_beside_it_is_folded() raises:
    """The whole of why the lowercase table exists beside the fold table. A
    literal under this flag is widened while the pattern is compiled into
    everything that folds onto it, and the long s is in the orbit of `s`, so
    `(?i)ss` reads one. A reference is lowered while the row is walked instead,
    and the long s lowers to itself, so `(?i)(s)\\1` reads none. Upstream
    answers those two exactly that way and the sigma pair goes the same way.
    Document 97."""
    assert_true(hits("(?i)ss", "sſ"))
    assert_false(hits("(?i)(s)\\1", "sſ"))
    assert_true(hits("(?i)σσ", "σς"))
    assert_false(hits("(?i)(σ)\\1", "σς"))
    # The dotless i is the same disagreement the other way round. It lowers to
    # itself, so a plain `i` after it is not a reading back of it, while
    # `(?i)i` matches it because Python folds all four of that family together.
    assert_true(hits("(?i)i", "ı"))
    assert_false(hits("(?i)(ı)\\1", "ıi"))
    assert_true(hits("(?i)(ı)\\1", "ıı"))


def test_the_ignore_case_flag_is_read_where_the_reference_stands() raises:
    """Rather than over the pattern, because the flag is scoped. `(?i:(a)\\1)`
    has the reference inside the scope and folds, and `(?i:(a))\\1` has it
    outside and does not, so the second one wants the two characters to be the
    same character even though the group that caught the first was folding."""
    assert_equal(said("(?i:(a)\\1)", ENGINE_PYTHON), "ok")
    assert_true(hits("(?i:(a)\\1)", "AA"))
    assert_true(hits("(?i:(a)\\1)", "Aa"))
    assert_equal(said("(?i:(a))\\1", ENGINE_PYTHON), "ok")
    assert_true(hits("(?i:(a))\\1", "AA"))
    assert_true(hits("(?i:(a))\\1", "aa"))
    assert_false(hits("(?i:(a))\\1", "Aa"))
    assert_false(hits("(?i:(a))\\1", "aA"))


def test_the_other_engine_still_refuses_one() raises:
    """RE2 has no such syntax at all, so the refusal there is agreement with
    RE2 rather than a shortfall here, and it keeps the words it always had."""
    assert_equal(said("(a)\\1", ENGINE_RE2), "!RE2 has no backreference")
    var program = compile_program(parse_pattern("(a)\\1"), ENGINE_RE2)
    assert_false(program.ok)


def test_a_lookaround_beside_one_is_answered_here() raises:
    """The two constructs used to live on different engines. A lookaround is a
    search inside a search and the machine was what ran the inner one, and a
    backreference is the one shape the machine cannot be handed, so a pattern
    holding both had nowhere to go. This engine runs the inner search itself
    now, on the stack it already has, so both of them are here. Document 120."""
    assert_equal(said("(?=a)(a)\\1", ENGINE_PYTHON), "ok")
    assert_equal(said("(a)\\1(?=b)", ENGINE_PYTHON), "ok")
    assert_equal(said("(?<=a)(b)\\1", ENGINE_PYTHON), "ok")
    assert_true(hits("(?=(a))a\\1", "aa"))
    assert_false(hits("(?=(a))a\\1", "ab"))
    assert_true(hits("(a)\\1(?=b)", "aab"))
    assert_false(hits("(a)\\1(?=b)", "aac"))
    assert_true(hits("(?<=(a))b\\1", "aba"))
    assert_false(hits("(?<=(a))b\\1", "abc"))


def test_a_construct_under_a_repeat_of_zero_is_not_in_the_program() raises:
    """Which is why the flags that pick the engine are set from the
    instructions rather than from the tree. `(?=a){0}` is parsed and never
    written, so a pattern with that and a backreference in it is a program
    holding one construct rather than two, and the engine it goes to is the
    engine that construct needs."""
    assert_equal(said("(?=a){0}(b)\\1", ENGINE_PYTHON), "ok")
    assert_true(hits("(?=a){0}(b)\\1", "bb"))


def test_a_pattern_holding_one_carries_slots_nobody_asked_for() raises:
    """Because the instruction reads one and there is nowhere else for it to
    read. A caller asking whether a row holds `(\\w)\\1` asked about a row and
    not about the groups in it, and gets the slots anyway, which is a cost the
    pattern brings rather than one the question does."""
    var plain = compile_program(parse_pattern("(\\w)x"), ENGINE_PYTHON)
    assert_equal(plain.slots, 0)
    assert_false(plain.refs)
    var reads = compile_program(parse_pattern("(\\w)\\1"), ENGINE_PYTHON)
    assert_equal(reads.slots, 4)
    assert_true(reads.refs)


def test_the_flag_is_on_the_program_and_the_other_engines_read_it() raises:
    """The backtracker takes such a program and never hands it back, and both
    the machine and the state cache turn one down. So the flag is what decides
    which of the three answers, rather than the row."""
    var reads = compile_program(
        parse_pattern("(\\w)\\1"), ENGINE_PYTHON, alphabet=True
    )
    assert_true(reads.refs)
    var bounded = Bounded(reads)
    assert_true(bounded.ok)
    var machine = Machine(reads)
    var cache = Cache(reads)
    assert_false(cache.ok)
    assert_equal(cache.problem, "the pattern asks about what it matched before")
    _ = machine^
    _ = bounded^


def test_the_counting_and_replacing_scans_read_one_the_same_way() raises:
    """The scans are the engine's own and nothing in them knows what a
    backreference is, so the only thing worth checking is that the rule for
    where to look next is the one document 93 wrote down."""
    assert_equal(found("(\\w)\\1", "aabbcd"), 2)
    assert_equal(subbed("(\\w)\\1", "aabbcd"), "##cd")
    assert_equal(found("(a*)\\1", "aa"), 2)
    assert_equal(subbed("(a*)\\1", "aa"), "##")
    assert_equal(found("(a)\\1", "aaaa"), 2)
    assert_equal(subbed("(a)\\1", "aaaa"), "##")


def test_the_scan_reads_characters_rather_than_bytes() raises:
    """The two ends of a group are positions in characters, so a row with a two
    byte character in it is read back the way Python reads it. A width in bytes
    would take half of the sharp s and compare it against nothing."""
    assert_true(hits("(ß)\\1", "ßß"))
    assert_false(hits("(ß)\\1", "ßs"))
    assert_equal(subbed("(ß)\\1", "ßß"), "#")
    assert_true(hits("(.)\\1", "ßß"))


def test_the_router_sends_one_to_pythons_engine() raises:
    """A backreference is a construct RE2 has not got, so a call naming one is
    routed by the walk over the tree rather than by a flag the caller passed,
    and it comes back with a program in it now rather than with a refusal."""
    var routed = program_for(METHOD_CONTAINS, "(a)\\1", 0, False, 14)
    assert_true(routed.ok)
    assert_true(routed.refs)
    var argued = program_for(METHOD_COUNT, "(a)\\1", 0, True, 14)
    assert_true(argued.ok)
    assert_true(argued.refs)


def test_the_anchoring_bracket_does_not_renumber_the_groups() raises:
    """`match` and `fullmatch` on this engine are answered by a pattern with an
    anchor glued to each end and a bracket around the middle, and the bracket
    has to be a non capturing one. A capturing one numbers every group the
    caller wrote one higher, and a reference follows the numbering, so `(a)\\1`
    would come out naming the wrapper. The wrapper is still open where the
    reference stands, so it reads as a group that never took part and the
    pattern matches nothing at all."""
    assert_equal(
        python_anchored(METHOD_FULLMATCH, "(a)\\1"), "\\A(?:(a)\\1)\\Z"
    )
    var whole = program_for(METHOD_FULLMATCH, "(a)\\1", 0, False, 14)
    assert_true(whole.ok)
    assert_true(held_text(whole, "aa"))
    assert_false(held_text(whole, "aab"))
    var front = program_for(METHOD_MATCH, "(a)\\1", 0, False, 14)
    assert_true(front.ok)
    assert_true(held_text(front, "aab"))
    assert_false(held_text(front, "baa"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
