"""Tests that the same pattern compiled for the other engine answers Python.

Everything else in this directory asks the RE2 question, because until now that
was the only question there was. pandas answers `contains`, `match`, `fullmatch`,
`count`, `replace` and `split` out of Arrow and it answers `extract`,
`extractall` and `findall` out of Python's own `re`, and the two disagree about
three things a pattern can say. Document 81 has where each of them was measured.

The three are asserted here as pairs, the same pattern against the same text
compiled twice, because that is the only shape that says anything. A test that
shows `\\w` matching a letter in Greek is a test about Greek. A test that shows
the same `\\w` matching it for one engine and not for the other is a test about
the thing that will be got wrong.

The differential next door is the stronger test of whether the classes are
right, since it asks a running Python about the generated corpus rather than
about the dozen characters somebody thought of. What is here is the handful a
reader should be able to look at without running anything.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.kernel.regex.parse import parse_pattern
from firepanda.kernel.regex.pike import matches_text
from firepanda.kernel.regex.program import compile_program
from firepanda.kernel.regex.route import ENGINE_PYTHON, ENGINE_RE2


def ours(pattern: StringSlice, text: StringSlice) raises -> Bool:
    """Whether a pattern read the way Python reads it matches the text.

    Args:
        pattern: The pattern.
        text: The text.

    Returns:
        True when some part of it matches.

    Raises:
        Error: If the pattern did not compile, which in this file is a mistake
            in the test rather than an answer.
    """
    var program = compile_program(parse_pattern(pattern), ENGINE_PYTHON)
    if not program.ok:
        raise Error(
            String("pattern ", pattern, " did not compile: ", program.problem)
        )
    return matches_text(program, text)


def theirs(pattern: StringSlice, text: StringSlice) raises -> Bool:
    """The same question asked of the RE2 reading.

    Args:
        pattern: The pattern.
        text: The text.

    Returns:
        True when some part of it matches.

    Raises:
        Error: If the pattern did not compile.
    """
    var program = compile_program(parse_pattern(pattern), ENGINE_RE2)
    if not program.ok:
        raise Error(
            String("pattern ", pattern, " did not compile: ", program.problem)
        )
    return matches_text(program, text)


def test_a_word_character_is_every_letter_there_is() raises:
    """`str.count(r"\\w")` on a row of `café` is 3 and `str.findall` of the same
    pattern on the same row is 4 long, and this is that difference in one
    line."""
    assert_true(ours("\\w", "é"))
    assert_false(theirs("\\w", "é"))
    assert_true(ours("\\w", "Ж"))
    assert_false(theirs("\\w", "Ж"))
    assert_true(ours("\\w", "一"))
    assert_false(theirs("\\w", "一"))


def test_the_ascii_word_characters_are_in_both_readings() raises:
    """The wider class holds the narrower one, so nothing that used to match
    stopped matching."""
    assert_true(ours("\\w", "a"))
    assert_true(theirs("\\w", "a"))
    assert_true(ours("\\w", "_"))
    assert_true(theirs("\\w", "_"))
    assert_true(ours("\\w", "7"))
    assert_true(theirs("\\w", "7"))


def test_a_digit_is_every_decimal_digit_and_not_every_number() raises:
    """Python's `\\d` is `str.isdecimal` and not `str.isdigit`, so an Arabic
    Indic digit is one and a superscript two is not, which is the distinction a
    reader gets wrong from memory."""
    assert_true(ours("\\d", "٣"))
    assert_false(theirs("\\d", "٣"))
    assert_true(ours("\\d", "１"))
    assert_false(theirs("\\d", "１"))
    assert_false(ours("\\d", "²"))
    assert_false(theirs("\\d", "²"))


def test_a_space_is_wider_than_the_five_re2_has() raises:
    """RE2 has five and Python has 29, and the five are the ones a reader would
    list. A vertical tab and a no break space are both in the wider class and
    neither is in the narrower one, and a zero width space is in neither, which
    is the character that catches somebody reading the name rather than the
    table."""
    assert_true(ours("\\s", "\x0b"))
    assert_false(theirs("\\s", "\x0b"))
    assert_true(ours("\\s", "\u00a0"))
    assert_false(theirs("\\s", "\u00a0"))
    assert_false(ours("\\s", "\u200b"))
    assert_false(theirs("\\s", "\u200b"))


def test_a_capital_class_is_the_complement_of_the_wider_one() raises:
    """`\\W` has to be read against whichever `\\w` is running, or a letter in
    Greek is in both of them."""
    assert_false(ours("\\W", "é"))
    assert_true(theirs("\\W", "é"))
    assert_false(ours("\\D", "٣"))
    assert_true(theirs("\\D", "٣"))


def test_a_class_inside_brackets_is_read_the_same_way() raises:
    """The category is merged with whatever else is in the brackets, and the
    merge has to happen after the reading rather than before it."""
    assert_true(ours("[\\d,]", "٣"))
    assert_false(theirs("[\\d,]", "٣"))
    assert_true(ours("[\\d,]", ","))
    assert_true(theirs("[\\d,]", ","))


def test_a_dollar_sign_matches_before_a_newline_that_ends_the_text() raises:
    """`re.search("a$", "a\\n")` finds something and the same pattern through
    Arrow finds nothing, and pandas papers over one instance of this by
    rewriting a trailing `\\Z` and not the other."""
    assert_true(ours("a$", "a\n"))
    assert_false(theirs("a$", "a\n"))
    assert_true(ours("a$", "a"))
    assert_true(theirs("a$", "a"))
    assert_false(ours("a$", "a\n\n"))
    assert_false(theirs("a$", "a\n\n"))


def test_a_word_boundary_is_asked_against_the_wider_class() raises:
    """A boundary is written against whichever `\\w` is running, so the two ends
    of a word in Greek are boundaries to one engine and are not to the other.

    The third pair is the one worth reading twice, because it goes the other
    way. Between a letter in ASCII and a letter outside it, RE2 sees a word
    character beside something that is not one and puts a boundary there, and
    Python sees two word characters and does not."""
    assert_true(ours("\\bé", "é"))
    assert_false(theirs("\\bé", "é"))
    assert_true(ours("é\\b", "é"))
    assert_false(theirs("é\\b", "é"))
    assert_true(theirs("a\\bé", "aé"))
    assert_false(ours("a\\bé", "aé"))


def test_a_non_boundary_is_answered_rather_than_refused() raises:
    """RE2 asks this one between bytes and is refused here for it. Python asks
    it between characters, which is what this machine already walks, so there is
    nothing to refuse."""
    assert_true(ours("\\B", "abc"))
    assert_true(ours("\\B", "ééé"))
    assert_false(ours("\\B", "a b"))
    var program = compile_program(parse_pattern("\\B"), ENGINE_RE2)
    assert_false(program.ok)
    assert_true(program.gap)


def test_the_boundary_is_at_the_edge_of_a_word_and_not_inside_it() raises:
    """The pair that says the underscore is a word character and the hyphen is
    not, which is true of both readings and is what the wider one had to keep
    true."""
    assert_true(ours("a\\b-", "a-b"))
    assert_false(ours("a\\b_", "a_b"))
    assert_true(ours("é\\b-", "é-é"))
    assert_false(ours("é\\b_", "é_é"))


def test_everything_the_two_readings_share_is_still_shared() raises:
    """The three differences are the whole of the difference, so a pattern
    holding none of them compiles to the same answer twice."""
    assert_true(ours("^ab+c", "abbc"))
    assert_true(theirs("^ab+c", "abbc"))
    assert_true(ours("a|b", "zb"))
    assert_true(theirs("a|b", "zb"))
    assert_false(ours("(?:xy){2}", "xyx"))
    assert_false(theirs("(?:xy){2}", "xyx"))


def test_a_pattern_python_answers_and_re2_refuses_is_still_a_gap() raises:
    """The constructs are the part of the router that is not closed by this. A
    named character is a pattern pandas answers and this library cannot resolve
    the name in yet, and saying so is the whole of what the flag is for.

    The lookahead was this row's example until document 93 answered it, the
    lookbehind was until document 94 did, the backreference was until document
    95 did, the atomic group was until document 99 did, the conditional group
    was until document 100 did and the three pairings of a lookaround with one
    of the other three were until document 120 did, which is every construct
    moving out of this list and leaving the table of names behind."""
    var program = compile_program(parse_pattern("\\N{BULLET}"), ENGINE_PYTHON)
    assert_false(program.ok)
    assert_true(program.gap)
    assert_equal(program.problem, "a named character is not resolved yet")


def test_syntax_only_re2_refuses_is_not_refused_for_python() raises:
    """The parser reads a comment group and a `\\u` escape and a `\\Z` in the
    middle of a pattern into the nodes Python reads them into, and then records
    that RE2 has no such thing. That record is a fact about the other engine, so
    this side compiles the tree it already has rather than throwing it away."""
    assert_true(ours("(?#a comment)b", "b"))
    assert_false(
        compile_program(parse_pattern("(?#a comment)b"), ENGINE_RE2).ok
    )
    assert_true(ours("\\u0041", "A"))
    assert_false(compile_program(parse_pattern("\\u0041"), ENGINE_RE2).ok)
    assert_false(ours("a\\Zb", "ab"))
    assert_true(ours("a\\Z", "a"))
    assert_false(compile_program(parse_pattern("a\\Zb"), ENGINE_RE2).ok)


def test_syntax_the_two_engines_read_differently_is_read_python_s_way() raises:
    """`a{,2}` is `a{0,2}` to Python and five characters to RE2, and
    `[[:alpha:]]` is a POSIX class to one and a bracket with some letters in it
    to the other. Both trees here are Python's reading, which is the only one
    this engine is being asked for."""
    assert_true(ours("^a{,2}$", "aa"))
    assert_false(ours("^a{,2}$", "aaa"))
    assert_false(compile_program(parse_pattern("^a{,2}$"), ENGINE_RE2).ok)
    assert_true(ours("[[:alpha:]]", ":]"))
    assert_false(ours("[[:alpha:]]", "q]"))


def test_the_three_letters_re2_never_had_all_mean_something_here() raises:
    """The four letters RE2 has never heard of are three that this engine reads
    and one that nothing does. `(?u)` asks for the classes this engine reads
    anyway, `(?x)` is spent in the parser, `(?a)` narrows the classes back to
    the ones RE2 uses, and `(?L)` never arrives, because Python will not take it
    on a pattern made of text and this parser is Python's grammar."""
    assert_true(ours("(?u)\\w", "é"))
    assert_false(compile_program(parse_pattern("(?u)\\w"), ENGINE_RE2).ok)
    var locale = compile_program(parse_pattern("(?L)a"), ENGINE_PYTHON)
    assert_false(locale.ok)
    assert_false(locale.gap)
    assert_equal(locale.problem, "Python's grammar cannot read this pattern")
    assert_true(ours("(?x)a b", "ab"))
    assert_false(ours("(?a)\\w", "é"))
    assert_false(compile_program(parse_pattern("(?x)a b"), ENGINE_RE2).ok)
    assert_false(compile_program(parse_pattern("(?a)\\w"), ENGINE_RE2).ok)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
