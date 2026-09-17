"""Tests for the counting and replacing scans Python's engine runs.

The engine next door answers whether a pattern matches, and that question has
one answer whoever asks it. Counting and replacing are not one question. They
are a loop around the engine, and the loop pandas runs depends on which engine
the call landed on, because Arrow's two kernels and Python's `re` walk a row by
different rules. `firepanda/kernel/regex/pike.mojo` has Arrow's counting loop
beside Python's and `firepanda/kernel/regex/replace.mojo` has the other pair.

Every number and every string asserted here was measured against a running
Python 3.13 rather than reasoned out of either implementation, because the
rules are small enough to guess at and wrong enough when guessed. The
replacement grammar in particular was measured escape by escape, since two of
its four branches exist only to tell an octal escape from a group reference and
neither of them is written down anywhere as a rule.

The pairs that matter most are the ones where the two scans disagree on the
same input, so those are asserted side by side: `count("^")` on three letters,
an empty pattern against a row with a sharp s in it, and a boundary against a
row of spaces. A test that shows one scan's answer says the scan runs. A test
that shows both says which one a call gets.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.kernel.regex.parse import parse_pattern
from firepanda.kernel.regex.pike import counts_python_text, counts_text
from firepanda.kernel.regex.program import Program, compile_program
from firepanda.kernel.regex.replace import (
    parse_rewrite_python,
    replaced_python_text,
)
from firepanda.kernel.regex.route import ENGINE_PYTHON, ENGINE_RE2
from firepanda.kernel.regex.tokens import FLAG_IGNORECASE, FLAG_MULTILINE


def built(pattern: StringSlice, flags: Int32 = 0) raises -> Program:
    """Compiles a pattern for Python's engine with captures on.

    Args:
        pattern: The pattern.
        flags: The letters it is read with.

    Returns:
        The program.

    Raises:
        Error: If it did not compile, which in this file is a mistake in the
            test rather than an answer.
    """
    var program = compile_program(
        parse_pattern(pattern, flags), ENGINE_PYTHON, captures=True
    )
    if not program.ok:
        raise Error(
            String("pattern ", pattern, " did not compile: ", program.problem)
        )
    return program^


def subbed(
    pattern: StringSlice,
    text: StringSlice,
    replacement: String,
    limit: Int = -1,
    flags: Int32 = 0,
) raises -> String:
    """Runs one pattern and one replacement over one row, Python's way.

    Args:
        pattern: The pattern.
        text: The row.
        replacement: The replacement, read by Python's grammar.
        limit: How many to replace, negative for all of them.
        flags: The letters the pattern is read with.

    Returns:
        The row with the matches replaced.

    Raises:
        Error: If the pattern or the replacement could not be read.
    """
    var program = built(pattern, flags)
    var rewrite = parse_rewrite_python(
        replacement, program.groups, program.labels
    )
    if not rewrite.ok:
        raise Error(String("replacement not read: ", rewrite.problem))
    return replaced_python_text(program, rewrite, text, limit)


def test_the_counting_scan_steps_one_character_past_an_empty_match() raises:
    """`re.finditer` moves on by one character after a match of no width, which
    is what makes an empty pattern count the characters of a row and not its
    bytes. Arrow's loop moves on by one byte and counts a sharp s twice, which
    is measured next door and is the reason there are two loops rather than a
    flag."""
    assert_equal(counts_python_text(built("a*"), "abc"), 4)
    assert_equal(counts_python_text(built(""), "ßx"), 3)
    assert_equal(counts_text(built(""), "ßx"), 4)


def test_the_counting_scan_never_cuts_the_row() raises:
    """Arrow's loop makes the rest of the row into a new text after every match,
    so a front anchor matches again at every position and `count("^")` on three
    letters is four, one of them being the front of an empty piece of text left
    over at the end. Python searches from an offset into the whole row, so the
    front is the front once."""
    assert_equal(counts_python_text(built("^"), "abc"), 1)
    assert_equal(counts_text(built("^"), "abc"), 4)
    assert_equal(counts_python_text(built("^", FLAG_MULTILINE), "a\nb\nc"), 3)


def test_the_counting_scan_reads_a_boundary_against_the_whole_row() raises:
    """Four boundaries in five characters, because the two spaces on each side
    of the letter are one word character away from a run of spaces. An empty row
    has no boundary at all, which is Python's answer and not RE2's."""
    assert_equal(counts_python_text(built("\\B"), "  a  "), 4)
    assert_equal(counts_python_text(built("\\B"), ""), 0)
    assert_equal(counts_python_text(built("k", FLAG_IGNORECASE), "KaKb"), 2)
    assert_equal(counts_python_text(built("(a)"), "aaa"), 3)


def test_the_replacing_scan_follows_the_same_rule_as_the_counting_one() raises:
    """One rule for both, which is `re.finditer`, so the marks land where the
    count said they would. The row of spaces is the one worth reading twice: a
    star matches nothing at every position and the whole of the letter run in
    the middle, so the middle mark is doubled and the ends are not."""
    assert_equal(subbed("a*", "abc", "#"), "##b#c#")
    assert_equal(subbed("a*", "", "#"), "#")
    assert_equal(subbed("a*", "  a  ", "#"), "# # ## # #")
    assert_equal(subbed("\\B", "  a  ", "#"), "# # a # #")
    assert_equal(subbed("", "abc", "#"), "#a#b#c#")


def test_a_limit_stops_the_scan_where_the_last_match_ended() raises:
    """The two cursors are what this asserts. A scan stopped by its count writes
    the rest of the row out from the end of the last match rather than from
    where it was about to look, so the letter the empty match sat in front of is
    still there. `##bc` and not `##c`, measured."""
    assert_equal(subbed("a*", "abc", "#", limit=2), "##bc")
    assert_equal(subbed("a", "AaA", "#", limit=2, flags=FLAG_IGNORECASE), "##A")
    assert_equal(subbed("a*", "abc", "#", limit=0), "abc")


def test_a_flag_is_read_by_both_halves_of_the_scan() raises:
    """The flag is compiled into the pattern rather than handed to the loop, so
    there is nothing for the loop to do about it, and that is worth one test all
    the same because the loop is where the flag has to survive to."""
    assert_equal(subbed("k", "KaKb", "#", flags=FLAG_IGNORECASE), "#a#b")
    assert_equal(subbed("^", "a\nb", "#", flags=FLAG_MULTILINE), "#a\n#b")


def test_the_replacement_grammar_names_a_group_three_ways() raises:
    """A number, a number in angle brackets and a name in angle brackets, plus
    the whole match as group zero, which RE2's grammar spells with an ampersand
    and cannot spell by name at all."""
    assert_equal(subbed("(a)(b)", "ab", "\\g<2>\\g<1>"), "ba")
    assert_equal(subbed("(?P<x>a)(b)", "ab", "[\\g<x>|\\2|\\g<0>]"), "[a|b|ab]")


def test_the_replacement_grammar_tells_an_octal_escape_from_a_group() raises:
    """Three digits after the backslash are an octal escape when all three are
    octal and a two digit group reference otherwise, which is the one rule in
    this grammar that reads ahead. A backslash and a zero is always octal, and a
    backslash in front of anything that is neither a letter nor a digit keeps
    the backslash as well as the character."""
    assert_equal(subbed("(a)(b)", "ab", "\\123"), "S")
    assert_equal(subbed("(a)(b)", "ab", "\\377"), "ÿ")
    assert_equal(subbed("(a)(b)", "ab", "\\n"), "\n")
    assert_equal(subbed("(a)(b)", "ab", "\\\\"), "\\")
    assert_equal(subbed("(a)(b)", "ab", "\\-"), "\\-")


def test_the_replacement_grammar_refuses_what_python_refuses() raises:
    """Four refusals, in this library's words rather than Python's, because the
    two differ about the exception class as well as the wording: `re` raises a
    `PatternError`, which is not a `ValueError`, and this side raises a
    `ValueError` for both grammars. Document 86 has that one written down."""
    var labels = List[String]()
    var one = parse_rewrite_python("\\8", 2, labels)
    assert_false(one.ok)
    assert_equal(
        one.problem, "the replacement asks for group 8 and the pattern has 2"
    )
    var two = parse_rewrite_python("\\s", 2, labels)
    assert_false(two.ok)
    assert_true(two.problem.startswith("a backslash in a replacement"))
    var three = parse_rewrite_python("a\\", 2, labels)
    assert_false(three.ok)
    assert_equal(three.problem, "a replacement cannot end in a backslash")
    var four = parse_rewrite_python("\\400", 2, labels)
    assert_false(four.ok)
    assert_equal(four.problem, "an octal escape in a replacement runs past 377")
    var five = parse_rewrite_python("\\g<>", 2, labels)
    assert_false(five.ok)
    assert_equal(five.problem, "a replacement names a group with no name")
    var six = parse_rewrite_python("\\gx", 2, labels)
    assert_false(six.ok)
    assert_true(six.problem.startswith("a backslash g in a replacement"))


def test_a_program_carries_which_scan_it_wants() raises:
    """The engine is a fact about the compiled pattern rather than an argument
    threaded through four layers, which is what lets the column walk pick the
    loop without being told. Every program compiled for the other engine says
    False and is walked Arrow's way."""
    assert_true(built("a").python)
    assert_true(compile_program(parse_pattern("a"), ENGINE_PYTHON).python)
    assert_false(compile_program(parse_pattern("a"), ENGINE_RE2).python)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
