"""Tests that a compiled pattern matches what it should and terminates.

The differential next door is the stronger test of what matches, since it asks
pandas about thirty thousand generated patterns over sixteen pieces of text and
this file has a few dozen of each. What is here is the part a generated corpus
is bad at.

Half of it is the corners the corpus reaches rarely and nobody would notice: an
empty pattern, an empty text, an anchor with nothing on one side of it, a match
that starts at the last position. A generator that writes a pattern out of atoms
and quantifiers gets to these eventually and does not tell you which ones it
reached.

The other half is the two patterns that would hang. `(a*)*` against a few
letters and `(a+)+b` against a line of them are the reason this machine holds
every position at once instead of backtracking, and neither of them is a wrong
answer when it goes wrong. It is a test that never finishes, which is not the
kind of failure a differential reports.
"""

from std.testing import TestSuite, assert_false, assert_true

from firepanda.kernel.regex.parse import parse_pattern
from firepanda.kernel.regex.pike import matches_text
from firepanda.kernel.regex.program import compile_program
from firepanda.kernel.regex.route import ENGINE_RE2


def hits(pattern: StringSlice, text: StringSlice) raises -> Bool:
    """Whether a pattern matches somewhere in a piece of text.

    Args:
        pattern: The pattern.
        text: The text.

    Returns:
        True when some part of it matches.

    Raises:
        Error: If the pattern did not compile, which in this file is a mistake
            in the test rather than an answer.
    """
    var program = compile_program(parse_pattern(pattern), ENGINE_RE2)
    if not program.ok:
        raise Error(
            String("pattern ", pattern, " did not compile: ", program.problem)
        )
    return matches_text(program, text)


def test_a_literal_is_found_anywhere_in_the_text() raises:
    """The search is unanchored, because that is the only question pandas asks
    of the engine."""
    assert_true(hits("b", "abc"))
    assert_true(hits("a", "a"))
    assert_true(hits("c", "abc"))
    assert_false(hits("d", "abc"))


def test_an_empty_pattern_matches_anything_including_nothing() raises:
    """Both engines say so, and it is the corner a generator produces least
    often."""
    assert_true(hits("", ""))
    assert_true(hits("", "a"))
    assert_true(hits("a*", ""))


def test_a_star_matches_nothing_and_a_plus_does_not() raises:
    """The one difference between them, stated where it can be seen."""
    assert_true(hits("ab*c", "ac"))
    assert_false(hits("ab+c", "ac"))
    assert_true(hits("ab+c", "abbc"))


def test_a_count_takes_exactly_what_it_says() raises:
    """Exactly, in both directions, which is why the third of these is False
    even though there are more than two letters between the other two."""
    assert_true(hits("ab{2}c", "abbc"))
    assert_true(hits("ab{2}c", "xabbcy"))
    assert_false(hits("ab{2}c", "abbbc"))
    assert_false(hits("ab{2}c", "abc"))
    assert_true(hits("^a{2,3}$", "aaa"))
    assert_false(hits("^a{2,3}$", "aaaa"))


def test_an_alternation_tries_every_arm() raises:
    """Including the one that would have been passed over by a machine that
    took the first arm that could start."""
    assert_true(hits("^(?:ab|a)$", "a"))
    assert_true(hits("^(?:a|ab)$", "ab"))
    assert_false(hits("^(?:a|ab)$", "abc"))


def test_a_class_answers_for_what_is_in_it() raises:
    """Three letters and a range, from either side of the brackets."""
    assert_true(hits("[abc]", "b"))
    assert_false(hits("[abc]", "d"))
    assert_true(hits("[^abc]", "d"))
    assert_false(hits("[^abc]", "a"))
    assert_true(hits("[a-c]", "c"))


def test_the_perl_classes_are_ascii_here() raises:
    """An Arabic Indic digit is a digit to Python and not to RE2, which is the
    single largest difference between the two engines and is the one thing in
    this file most likely to look like a bug."""
    assert_true(hits("\\d", "007"))
    assert_false(hits("\\d", "٣٤"))
    assert_true(hits("\\w", "a1_"))
    assert_false(hits("\\w", " "))
    assert_true(hits("\\s", "a b"))
    assert_false(hits("\\s", "a\x0bb"))


def test_a_full_stop_stops_at_a_newline_unless_the_pattern_says_not_to() raises:
    """Which both engines agree about, and so does the flag that changes it."""
    assert_false(hits("^a.b$", "a\nb"))
    assert_true(hits("(?s)^a.b$", "a\nb"))
    assert_true(hits("^a.b$", "axb"))


def test_the_end_of_the_text_is_the_end_of_the_text() raises:
    """The RE2 reading. Python's `$` also matches just before a newline that
    ends the text, so `re.search("a$", "a\\n")` finds something and this does
    not, and that difference is deliberate."""
    assert_true(hits("a$", "a"))
    assert_false(hits("a$", "a\n"))
    assert_true(hits("^a", "a\n"))
    assert_false(hits("^a", "\na"))


def test_a_word_boundary_is_between_a_word_character_and_something_else() raises:
    """And a boundary at the end of the text needs nothing on the far side of
    it."""
    assert_true(hits("\\bab\\b", "ab"))
    assert_true(hits("\\bab\\b", "x ab y"))
    assert_false(hits("\\bab\\b", "xaby"))
    assert_true(hits("\\ba", "a"))


def test_a_match_can_start_at_the_last_position() raises:
    """A thread is seeded at every position including the one past the end, and
    forgetting the last one is a bug that passes almost every test."""
    assert_true(hits("b$", "ab"))
    assert_true(hits("$", "ab"))
    assert_true(hits("^$", ""))


def test_an_anchored_pattern_answers_what_it_did_before_the_shortcut() raises:
    """The scan starts no attempt above position zero for a pattern that opens
    with `^` or `\\A`, and stops reading the row once the attempt at zero has
    died. Everything here is a case where that could be got wrong: a match that
    is not at the start, a match that ends at the last position, a row long
    enough that the walk goes on well past the anchor, and the empty row."""
    assert_true(hits("^abc", "abcdef"))
    assert_false(hits("^abc", "xabcdef"))
    assert_true(hits("\\Aabc", "abc"))
    assert_false(hits("\\Aabc", " abc"))
    assert_true(hits("^a.*z", "abcdefghijklmnopqrstuvwxyz"))
    assert_false(hits("^b.*z", "abcdefghijklmnopqrstuvwxyz"))
    assert_true(hits("^a*$", "aaaaaaaa"))
    assert_false(hits("^a*$", "aaaaaaab"))
    assert_true(hits("^", ""))
    assert_false(hits("^a", ""))
    assert_true(hits("^https?://([^/]+)/", "http://example.com/page"))
    assert_false(hits("^https?://([^/]+)/", "ftp://example.com/page"))
    assert_true(hits("(?m)^b", "a\nb"))
    assert_true(hits("(?m)^b$", "a\nb\nc"))


def test_a_repeat_whose_body_can_match_nothing_terminates() raises:
    """The whole reason an instruction is added to a list at most once per
    position. Without that, this is an infinite loop rather than a slow one, and
    a test that does not finish is not a test that fails."""
    assert_true(hits("(a*)*", "aaa"))
    assert_true(hits("(a*)*", ""))
    assert_true(hits("^(a*)*$", "aaa"))
    assert_true(hits("(|a)*b", "aab"))


def test_the_pattern_that_hangs_a_backtracking_engine() raises:
    """Sixty letters and no `b`, which is the textbook way to make an engine
    that backtracks take longer than anyone will wait. Here it is sixty steps,
    and the assertion is that this test finishes at all."""
    var line = String("")
    for _ in range(60):
        line += "a"
    assert_false(hits("^(a+)+b$", line))
    assert_true(hits("^(a+)+b$", line + "b"))


def test_a_pattern_longer_than_the_text_answers_rather_than_reads_past() raises:
    """The other side of the position loop, which is cheap to get wrong in a
    machine that steps one character at a time."""
    assert_false(hits("abcdef", "abc"))
    assert_false(hits("a", ""))
    assert_false(hits("[abc]", ""))


def test_text_outside_ascii_is_read_a_character_at_a_time() raises:
    """The machine runs over code points rather than bytes, so a character that
    takes three bytes to write is one step."""
    assert_true(hits("^h.llo$", "héllo"))
    assert_true(hits("^...$", "ΑΒΓ"))
    assert_false(hits("^....$", "ΑΒΓ"))
    assert_true(hits("é", "héllo"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
