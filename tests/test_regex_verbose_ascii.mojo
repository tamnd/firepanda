"""The last two of the seven letters, which are a grammar and an alphabet.

`(?x)` and `(?a)` were the only flags any name on the `str` accessor still
turned down, and they are here together because that is the only thing they
have in common. Verbose mode changes what the characters of a pattern mean and
is spent in the parser. The ascii flag changes what six classes, a fold and a
word boundary cover and is spent in the compiler. Neither of them is visible to
the machine that runs the program.

Every row below was produced by asking a running CPython and a running pandas
before it was written down, which is the procedure document 87 adopted. The
ones worth knowing about by eye are the three that separate this from a guess:
whitespace inside a character class is not skipped, a brace run that fails to
parse is a literal and its spaces are then skipped, and Python's ASCII `\\s`
holds a vertical tab where RE2's never did.

Document 88 is the whole of it.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.kernel.regex.parse import parse_pattern
from firepanda.kernel.regex.pike import matches_text
from firepanda.kernel.regex.program import compile_program
from firepanda.kernel.regex.route import ENGINE_PYTHON, ENGINE_RE2
from firepanda.kernel.regex.tokens import (
    FLAG_ASCII,
    FLAG_IGNORECASE,
    FLAG_VERBOSE,
)


def ours(
    pattern: StringSlice, text: StringSlice, flags: Int32 = 0
) raises -> Bool:
    """Whether a pattern read Python's way under some letters matches the text.

    Args:
        pattern: The pattern.
        text: The text.
        flags: The letters, as `FLAG_` bits, standing for what a caller passed
            beside the pattern rather than inside it.

    Returns:
        True when some part of it matches.

    Raises:
        Error: If the pattern did not compile, which in this file is a mistake
            in the test rather than an answer.
    """
    var program = compile_program(parse_pattern(pattern, flags), ENGINE_PYTHON)
    if not program.ok:
        raise Error(
            String("pattern ", pattern, " did not compile: ", program.problem)
        )
    return matches_text(program, text)


def reads(pattern: StringSlice, flags: Int32 = 0) -> Bool:
    """Whether the grammar could read the pattern at all.

    Args:
        pattern: The pattern.
        flags: The letters, as `FLAG_` bits.

    Returns:
        True when it parsed.
    """
    return parse_pattern(pattern, flags).ok


def why(pattern: StringSlice, flags: Int32 = 0) -> String:
    """The reason the grammar could not read the pattern.

    Args:
        pattern: The pattern.
        flags: The letters, as `FLAG_` bits.

    Returns:
        The message, which is Python's own wording.
    """
    return parse_pattern(pattern, flags).problem.copy()


def test_verbose_throws_away_the_six_characters_python_names() raises:
    """The set is Python's `WHITESPACE` and not Unicode's, which is a difference
    a caller can write down. The file separator is whitespace to `str.isspace`
    and is kept here, so a pattern holding one still means something."""
    assert_true(ours("a b", "ab", FLAG_VERBOSE))
    assert_false(ours("a b", "a b", FLAG_VERBOSE))
    assert_true(ours("a\t\n\x0b\x0c\rb", "ab", FLAG_VERBOSE))
    assert_false(ours("a\x1cb", "ab", FLAG_VERBOSE))
    assert_true(ours("a\x1cb", "a\x1cb", FLAG_VERBOSE))


def test_a_backslash_in_front_of_a_space_is_a_space() raises:
    """The one way to write a literal space in a verbose pattern outside a
    class, and the reason the skip cannot be a pass over the pattern before it
    is parsed."""
    assert_true(ours("a\\ b", "a b", FLAG_VERBOSE))
    assert_false(ours("a\\ b", "ab", FLAG_VERBOSE))


def test_whitespace_inside_a_class_is_still_a_character() raises:
    """The skip is at the top of the item loop and the class loop is somewhere
    else, so `[a b]` holds three characters and one of them is a space. This is
    the row that says the rule is about where the skip happens rather than about
    which characters it covers."""
    assert_true(ours("[a b]", " ", FLAG_VERBOSE))
    assert_true(ours("[a b]", "b", FLAG_VERBOSE))
    assert_true(ours("[ ]", " ", FLAG_VERBOSE))


def test_a_hash_runs_to_the_end_of_the_line() raises:
    """A comment, which ends at a newline or at the end of the pattern. The
    escape and the class put the character back, the same two ways they put a
    space back."""
    assert_true(ours("a # comment\nb", "ab", FLAG_VERBOSE))
    assert_true(ours("a#c\nb", "ab", FLAG_VERBOSE))
    assert_true(ours("a #c", "a", FLAG_VERBOSE))
    assert_true(ours("a\\#b", "a#b", FLAG_VERBOSE))
    assert_true(ours("[a#b]", "#", FLAG_VERBOSE))


def test_a_brace_run_that_fails_is_a_literal_and_then_the_spaces_go() raises:
    """The counted scan does not skip anything either, so `a{1, 2}` is not a
    repeat. What happens next is the part worth pinning: the `{` falls back to a
    literal, the run that follows it is read as literals, and the spaces inside
    that run are skipped by the item loop after all. So the pattern matches
    `a{1,2}` written without a space."""
    assert_false(ours("a{1, 2}", "aa", FLAG_VERBOSE))
    assert_true(ours("a{1, 2}", "a{1,2}", FLAG_VERBOSE))
    assert_true(ours("a{1,2}", "aa", FLAG_VERBOSE))


def test_a_quantifier_is_reached_past_whitespace_and_read_without_it() raises:
    """Two rules from one place. The item loop skips, so `a *` is a repeat. The
    peek for the lazy marker does not, so the `?` in `a * ?` is a second
    quantifier and the error is Python's own multiple repeat."""
    assert_true(ours("a *", "aaa", FLAG_VERBOSE))
    assert_false(reads("a * ?", FLAG_VERBOSE))
    assert_equal(why("a * ?", FLAG_VERBOSE), "multiple repeat")
    assert_false(reads("( ?: a)", FLAG_VERBOSE))
    assert_equal(why("( ?: a)", FLAG_VERBOSE), "nothing to repeat")


def test_the_letter_can_be_written_as_well_as_passed() raises:
    """`(?x)` is an item that produces no node, so it turns the flag on partway
    through the pass and everything after it is read the new way. It still has
    to be at the start, and whitespace skipped before another global flag group
    does not count as a start being used up."""
    assert_true(ours("(?x)a b", "ab"))
    assert_true(ours("(?x) (?i)a", "A"))
    assert_true(ours("(?x)#c\n(?i)a", "A"))
    assert_false(reads(" (?x)a b"))


def test_the_ascii_flag_narrows_the_three_classes() raises:
    """Six of them, since the capital letters are the same sets complemented.
    The vertical tab is the row that matters: Python's ASCII `\\s` holds one and
    RE2's does not, so the narrow sets are two thirds of RE2's and not all of
    it."""
    assert_false(ours("\\w", "\u00e9", FLAG_ASCII))
    assert_true(ours("\\w", "\u00e9"))
    assert_false(ours("\\d", "\u0663", FLAG_ASCII))
    assert_true(ours("\\d", "\u0663"))
    assert_false(ours("\\s", "\u00a0", FLAG_ASCII))
    assert_true(ours("\\s", "\u00a0"))
    assert_true(ours("\\s", "\x0b", FLAG_ASCII))
    assert_false(ours("\\s", "\x1c", FLAG_ASCII))
    assert_true(ours("\\s", "\x1c"))
    assert_true(ours("\\W", "\u00e9", FLAG_ASCII))
    assert_true(ours("\\S", "\u00a0", FLAG_ASCII))
    assert_false(ours("[\\w]", "\u00e9", FLAG_ASCII))


def test_the_ascii_flag_folds_only_the_twenty_six_letters() raises:
    """The Kelvin sign, the long s and the two Turkish letters all stop folding,
    which makes `(?ia)` a third answer rather than either of the two this
    library already had. The negated class is the check that the narrowing is on
    the fold and not on the class: `(?i)[^k]` matches the Kelvin sign under the
    letter and does not without it."""
    assert_true(ours("k", "K", FLAG_IGNORECASE | FLAG_ASCII))
    assert_false(ours("k", "\u212a", FLAG_IGNORECASE | FLAG_ASCII))
    assert_true(ours("k", "\u212a", FLAG_IGNORECASE))
    assert_false(ours("s", "\u017f", FLAG_IGNORECASE | FLAG_ASCII))
    assert_true(ours("s", "\u017f", FLAG_IGNORECASE))
    assert_false(ours("i", "\u0130", FLAG_IGNORECASE | FLAG_ASCII))
    assert_false(ours("\u00e9", "\u00c9", FLAG_IGNORECASE | FLAG_ASCII))
    assert_true(ours("\u00e9", "\u00c9", FLAG_IGNORECASE))
    assert_true(ours("\u00e9", "\u00e9", FLAG_IGNORECASE | FLAG_ASCII))
    assert_true(ours("[\u00e9]", "\u00e9", FLAG_IGNORECASE | FLAG_ASCII))
    assert_false(ours("[j-l]", "\u212a", FLAG_IGNORECASE | FLAG_ASCII))
    assert_true(ours("[^k]", "\u212a", FLAG_IGNORECASE | FLAG_ASCII))
    assert_false(ours("[^k]", "\u212a", FLAG_IGNORECASE))
    assert_true(ours("\\W", "\u212a", FLAG_IGNORECASE | FLAG_ASCII))
    assert_false(ours("\\W", "\u212a", FLAG_IGNORECASE))


def test_the_ascii_flag_narrows_the_word_boundary_too() raises:
    """`\\b` and `\\B` ask about a word character, so the letter that says which
    characters those are moves them. It moves them onto the code RE2 already
    uses, because the ASCII word class is one set and both engines have it."""
    assert_true(ours("\\bx", "\u00e9x", FLAG_ASCII))
    assert_false(ours("\\bx", "\u00e9x"))
    assert_false(ours("\\Bx", "\u00e9x", FLAG_ASCII))
    assert_true(ours("\\Bx", "\u00e9x"))
    assert_true(ours("\\B", "\u00e9", FLAG_ASCII))
    assert_false(ours("\\B", "\u00e9"))
    assert_false(ours("\\B", "a\u00e9b", FLAG_ASCII))
    assert_true(ours("\\B", "a\u00e9b"))


def test_the_empty_row_is_not_an_alphabet_question() raises:
    """The row this nearly went out wrong on. Python fails a `\\B` on an empty
    row and RE2 matches one, which is a special case about the row rather than
    about which characters are word characters, so it survives the narrowing.
    Sending `\\B` to RE2's value under the letter would have taken RE2's answer
    here along with RE2's alphabet, and the sweep next door caught it on this
    exact cell."""
    assert_false(ours("\\B", "", FLAG_ASCII))
    assert_false(ours("\\B", ""))
    assert_false(ours("\\b", "", FLAG_ASCII))
    assert_false(ours("\\b", ""))
    var re2 = compile_program(parse_pattern("\\B"), ENGINE_RE2)
    assert_false(re2.ok)
    assert_true(re2.gap)


def test_the_two_letters_work_together() raises:
    """Nothing about either of them touches the other, which is worth one row
    rather than a section."""
    assert_true(ours("\\w b", "xb", FLAG_VERBOSE | FLAG_ASCII))
    assert_false(ours("\\w b", "\u00e9b", FLAG_VERBOSE | FLAG_ASCII))
    assert_true(ours("(?ax)\\w b", "xb"))


def test_both_alphabets_at_once_is_still_refused() raises:
    """Python raises a `ValueError` for this rather than a parse error and
    pandas does not catch it, which document 76 section 8 wrote down. This reads
    it as a pattern that did not parse, and the letter arriving as an argument
    rather than inside the pattern makes no difference to that."""
    assert_false(reads("(?a)(?u)a"))
    assert_equal(why("(?a)(?u)a"), "ASCII and UNICODE flags are incompatible")
    assert_false(reads("(?u)a", FLAG_ASCII))


def test_re2_still_refuses_both_letters_in_its_own_words() raises:
    """The engine underneath has not gained anything. A pattern carrying either
    letter and no route away from Arrow is refused here with the sentence RE2
    raises, which is what the differential compares against."""
    var verbose = compile_program(parse_pattern("(?x)a b"), ENGINE_RE2)
    assert_false(verbose.ok)
    assert_false(verbose.gap)
    assert_equal(verbose.problem, "invalid perl operator: (?x")
    var narrow = compile_program(parse_pattern("(?a)\\w"), ENGINE_RE2)
    assert_false(narrow.ok)
    assert_false(narrow.gap)
    assert_equal(narrow.problem, "invalid perl operator: (?a")


def test_a_scoped_group_is_still_the_thing_with_nowhere_to_go() raises:
    """The parser reads `(?x:...)` and throws the letters away, so the refusal
    moved from one sentence to another rather than going. What is missing is a
    place to hang a flag on a node, which is the same thing missing for every
    other letter in a scoped group."""
    var scoped = compile_program(parse_pattern("(?x:a b)"), ENGINE_PYTHON)
    assert_false(scoped.ok)
    assert_true(scoped.gap)
    assert_equal(scoped.problem, "a scoped flag group is not carried yet")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
