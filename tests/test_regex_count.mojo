"""Tests the scan that counts matches, against numbers taken from pandas.

The engine underneath is checked by the differential and by
`test_regex_pike.mojo`. What is checked here is the loop around it, which is
three rules Arrow follows and Python's `re` does not, and every number in this
file was read off pandas 3.0.5 rather than worked out from the rules. That order
matters: the rules were written down after the numbers were measured, the first
two guesses at them fitted some of the numbers and not all of them, and a test
written from the rules would have agreed with each wrong guess in turn.

The three rules, with the case that pins each one.

The rest of the row becomes the text after a match, so an anchor is judged
against the rest and not against the row. `count("^a")` on `aaa` is three.

The cursor moves in bytes, so a match of no width steps a third of the way
through a three byte character. `count("x*")` on a two character Arabic word
written in four bytes is five.

The cursor moves to where the match ended unless the match ended where the
cursor already was. A match of no width found further along the row is counted
where it was found and again from there, which is why `count("\\b")` on a word
with spaces on both sides of it is two and not three.

The last of the three is the one that took the longest to see, because two
simpler rules that are easy to reach for both give the right answer for a great
many patterns and the wrong one for `\\b`.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.exec.morsel import MORSEL_ROWS
from firepanda.kernel.agg import sum_of
from firepanda.kernel.compare import not_equal
from firepanda.kernel.concat import concat_strings
from firepanda.kernel.regex.column import text_count_regex
from firepanda.kernel.regex.method import METHOD_COUNT, program_for
from firepanda.kernel.regex.pike import counts_text
from firepanda.kernel.regex.program import Program
from firepanda.kernel.scalar import text_count_regex_scalar


def compiled(pattern: String) raises -> Program:
    """Compiles a pattern the way `str.count` would, or fails the test.

    Args:
        pattern: The pattern.

    Returns:
        The program.

    Raises:
        Error: If it did not compile, which in this file is a mistake in the
            test rather than an answer.
    """
    var program = program_for(METHOD_COUNT, pattern)
    if not program.ok:
        raise Error(
            String("pattern ", pattern, " did not compile: ", program.problem)
        )
    return program^


def seen(pattern: StringSlice, text: StringSlice) raises -> Int:
    """How many times a pattern matches in a piece of text.

    Args:
        pattern: The pattern.
        text: The text.

    Returns:
        The count.

    Raises:
        Error: If the pattern did not compile.
    """
    return counts_text(compiled(String(pattern)), text)


def test_counts_a_literal() raises:
    """A pattern with no metacharacter counts what a byte search counts.

    The cursor moves past the whole match, so the matches do not overlap and
    `aa` is in `aaaa` twice rather than three times. This is the one rule the
    literal path already had and the scan has to agree with it, since a caller
    who adds a full stop to their pattern should not find the rest of the
    answer changing shape.
    """
    assert_equal(seen("a", "aaaa"), 4, "a in aaaa")
    assert_equal(seen("aa", "aaaa"), 2, "aa in aaaa")
    assert_equal(seen("aa", "aaa"), 1, "aa in aaa")
    assert_equal(seen("b", "aaaa"), 0, "b in aaaa")
    assert_equal(seen("abc", ""), 0, "abc in nothing")


def test_prefers_the_first_arm() raises:
    """The match ends where the pattern says, not where the text allows.

    `a|aa` ends after one character and `aa|a` ends after two, which is the
    difference between leftmost first and leftmost longest and is visible in
    the count rather than in any single match: four letters hold four of the
    first and two of the second.
    """
    assert_equal(seen("a|aa", "aaaa"), 4, "a|aa in aaaa")
    assert_equal(seen("aa|a", "aaaa"), 2, "aa|a in aaaa")
    assert_equal(seen("a|aa", "aa"), 2, "a|aa in aa")
    assert_equal(seen("aa|a", "aa"), 1, "aa|a in aa")
    assert_equal(seen("(?:ab)+", "abab"), 1, "(?:ab)+ in abab")


def test_the_rest_of_the_row_becomes_the_text() raises:
    """An anchor is judged against what is left rather than against the row.

    This is the rule that surprises, and it is not a corner: `str.count("^a")`
    on a row of three letters is three in pandas and one in Python. Every
    assertion that reads the start of the text moves with the cursor, so `\\A`
    and a multiline `^` do the same thing.
    """
    assert_equal(seen("^a", "aaa"), 3, "^a in aaa")
    assert_equal(seen("\\Aa", "aaa"), 3, "\\Aa in aaa")
    assert_equal(seen("(?m)^a", "aaa"), 3, "(?m)^a in aaa")
    assert_equal(seen("^a", "aba"), 1, "^a in aba")
    assert_equal(seen("^", "abc"), 4, "^ in abc")


def test_the_end_of_the_row_does_not_move() raises:
    """The end is the end, so a pattern anchored there matches once.

    Twice, in fact, and the second one is the scan finding the same position
    again from an empty rest of the row. That is the third rule showing through
    on the simplest pattern it touches, and the number is pandas' number.
    """
    assert_equal(seen("$", "abc"), 2, "$ in abc")
    assert_equal(seen("\\z", "abc"), 2, "\\z in abc")
    assert_equal(seen("a$", "aaa"), 1, "a$ in aaa")
    assert_equal(seen("a$", "a\n"), 0, "a$ in a and a newline")
    assert_equal(seen("(?m)a$", "a\nb"), 1, "(?m)a$ in a, a newline and b")


def test_a_match_of_no_width_moves_one_byte() raises:
    """An empty pattern counts the bytes of a row and one more.

    Not the characters, which is the difference Python's `re` shows: the same
    five character word with one accented letter in it is seven to pandas and
    six to `re`. The literal path counts an empty pattern the same way and
    document 66 measured it there first, so the two paths agree on the one
    pattern they can both be asked.
    """
    assert_equal(seen("x*", "abc"), 4, "x* in abc")
    assert_equal(seen("", "abc"), 4, "nothing in abc")
    assert_equal(seen("a{0}", "abc"), 4, "a{0} in abc")
    assert_equal(seen("x*", "héllo"), 7, "x* in héllo")
    assert_equal(seen("x*", "٣٤"), 5, "x* in two Arabic digits")
    assert_equal(seen("x*", ""), 1, "x* in nothing")


def test_a_character_can_be_matched_around_a_byte() raises:
    """A scan that stopped in the middle of a character finds the next one.

    The rest of the row after such a stop begins with a byte that is not a
    character, and nothing matches it, so a pattern that can match nothing
    matches nothing there and moves on. What must not happen is the character
    after it being missed, which is what `é*` is here to show: it counts one
    real match and five of no width, and a scan that lost the accented letter
    would count six of no width and read the same.
    """
    assert_equal(seen("é*", "héllo"), 6, "é* in héllo")
    assert_equal(seen("(?:é)?", "héllo"), 6, "(?:é)? in héllo")
    assert_equal(seen("l*", "aé"), 4, "l* in aé")
    assert_equal(seen("é", "ééé"), 3, "é in ééé")
    assert_equal(seen(".", "héllo"), 5, ". in héllo")


def test_a_boundary_is_counted_where_it_is_found() raises:
    """The rule that two simpler ones get wrong, and the pattern that shows it.

    A boundary in the middle of a row is found from the start of the row, which
    moves the cursor to it, and then found again from there. A scan that moved
    the cursor past it would count one, and a scan that moved the cursor one
    byte at a time would count three. pandas counts two.

    The word boundary also reads `\\w` as ASCII, so a row of Arabic digits has
    no boundary anywhere in it, which is RE2's reading and not Python's.

    The other half of the pair, `\\B`, is not here because this library refuses
    it. `tests/test_regex_program.mojo` holds that refusal and says why, which
    is that RE2 reads a non boundary between the bytes of a character as well as
    between characters.
    """
    assert_equal(seen("\\b", "  a  "), 2, "\\b in a with spaces both sides")
    assert_equal(seen("\\b", "a  "), 1, "\\b in a with spaces after")
    assert_equal(seen("\\b", "  a"), 2, "\\b in a with spaces before")
    assert_equal(seen("\\b", "ab ba"), 5, "\\b in ab ba")
    assert_equal(seen("\\b", "aaaa"), 4, "\\b in aaaa")
    assert_equal(seen("\\b", "٣٤"), 0, "\\b in two Arabic digits")


def test_a_row_that_cannot_match_is_cheap_and_zero() raises:
    """A pattern that matches nothing stops after one pass over the row.

    There is nothing to assert about the cost from here, so what is asserted is
    the answer, on a row long enough that a scan which failed to stop would be
    noticed by the suite taking longer rather than by this line.
    """
    var long = String("")
    for _ in range(4000):
        long += "a"
    assert_equal(seen("qqq", long), 0, "qqq in four thousand letters")
    assert_equal(seen("b+", long), 0, "b+ in four thousand letters")
    assert_equal(seen("a+", long), 1, "a+ in four thousand letters")


def sample() -> StringArray:
    """A column with a null in it, for the kernel and its twin.

    The rows are the ones the scan is fussiest about: one that matches many
    times, an empty one, one holding a character wider than a byte, and two
    where a boundary of no width is found ahead of the cursor.

    Returns:
        The column.
    """
    var given = List[String]()
    given.append(String("aaaa"))
    given.append(String(""))
    given.append(String("ignored"))
    given.append(String("héllo"))
    given.append(String("  a  "))
    given.append(String("ab ba"))
    var builder = StringBuilder(capacity=len(given))
    for i in range(len(given)):
        if i == 2:
            builder.append_null()
        else:
            builder.append(given[i].as_bytes())
    return builder^.finish()


def test_the_kernel_agrees_with_its_twin() raises:
    """The counting kernel against the one row at a time version of itself.

    The same narrow check `test_regex_column.mojo` explains: both sides run the
    same engine, so what is being checked is the morsel split, the null repair
    and the machine that is built once and handed every row of a morsel.
    """
    var col = sample()
    while len(col) < MORSEL_ROWS:
        var pair = List[StringArray]()
        pair.append(col.copy())
        pair.append(col.copy())
        col = concat_strings(pair)
    var tail = List[StringArray]()
    tail.append(col^)
    tail.append(sample())
    col = concat_strings(tail)

    var program = compiled("a|\\b")
    var counted = text_count_regex(col, program)
    var want = text_count_regex_scalar(col, program)
    assert_equal(
        Int(sum_of(not_equal(counted, want)).value), 0, "rows disagreeing"
    )
    # And that the twin was not vacuously right about a column that came out
    # all zero or all null. The three numbers are pandas' numbers for these
    # three rows and this pattern.
    assert_equal(Int(counted[0]), 4, "the first row")
    assert_equal(Int(counted[1]), 0, "the empty row")
    assert_equal(Int(counted[4]), 1, "the row with a boundary in the middle")
    assert_true(counted.is_valid(0), "the first row is not null")
    assert_false(counted.is_valid(2), "the null row")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
