"""Tests the scan that replaces matches, against text taken from pandas.

Every answer in this file was read off pandas 3.0.5 rather than worked out from
the rules, and the order matters for the same reason it did for counting: the
rules were written down after the measurements, and a test written from the
rules would have agreed with a wrong reading of them.

The three rules this scan follows, with the case that pins each one.

The text is not cut after a match, so an anchor keeps meaning what it meant.
`replace("(?m)^", "#")` on a row holding a newline puts a marker at the start
and after the newline, and nothing anywhere else. The scan that counts cuts, so
the two scans in the same Arrow library disagree about this pattern.

The cursor moves a character at a time. `replace("x*", "#")` on a five character
word with an accented letter in it gives six markers and not seven, where the
scan that counts answers seven because it moves a byte at a time.

A match of no width that lands exactly where the last match ended is thrown
away, and one character is copied across instead. `replace("a*", "#")` on `abc`
is `#b#c#`, where Python's `re.sub` gives `##b#c#`.

The replacement is RE2's rewrite string rather than Python's, which is a
narrower grammar: `\\0` and `\\1` through `\\9` and `\\\\` and nothing else.
"""

from std.collections.span import Span
from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.kernel.regex.method import METHOD_REPLACE, program_for
from firepanda.kernel.regex.parse import decode_into
from firepanda.kernel.regex.pike import Machine
from firepanda.kernel.regex.program import Program
from firepanda.kernel.regex.replace import (
    Rewrite,
    parse_rewrite,
    replaced,
    replaced_text,
)


def compiled(pattern: String) raises -> Program:
    """Compiles a pattern the way `str.replace` would, or fails the test.

    Args:
        pattern: The pattern.

    Returns:
        The program.

    Raises:
        Error: If it did not compile, which in this file is a mistake in the
            test rather than an answer.
    """
    var program = program_for(METHOD_REPLACE, pattern)
    if not program.ok:
        raise Error(
            String("pattern ", pattern, " did not compile: ", program.problem)
        )
    return program^


def swapped(
    pattern: StringSlice, repl: StringSlice, text: StringSlice
) raises -> String:
    """The text with every match of a pattern replaced.

    Args:
        pattern: The pattern.
        repl: The replacement.
        text: The text.

    Returns:
        The answer.

    Raises:
        Error: If the pattern or the replacement could not be read.
    """
    var program = compiled(String(pattern))
    var rewrite = parse_rewrite(String(repl), program.groups)
    if not rewrite.ok:
        raise Error(
            String("replacement ", repl, " was refused: ", rewrite.problem)
        )
    return replaced_text(program, rewrite, text)


def test_a_literal_is_replaced_everywhere() raises:
    """The plain case, which the literal path already answers the same way.

    It is here because the two paths have to agree: a caller who adds a full
    stop to a pattern should not find the rest of the answer changing shape.
    """
    assert_equal(swapped("a", "Z", "  a  "), "  Z  ", "a in a with spaces")
    assert_equal(swapped("a", "Z", "aaaa"), "ZZZZ", "a in aaaa")
    assert_equal(swapped("a", "Z", "héllo"), "héllo", "a in a row with none")
    assert_equal(swapped("a", "Z", ""), "", "a in nothing")
    assert_equal(swapped(".", "-", "٣٤"), "--", "a full stop reads characters")


def test_the_text_is_not_cut_so_an_anchor_stays_where_it_was() raises:
    """The first rule, and the one where the two Arrow scans part company.

    `str.count("^a")` on `aaa` is three because the counting scan makes the rest
    of the row into a text of its own. Replacing does not, so `^` is the start
    of the row, once, and a multiline `^` is the start of the row and the
    position after each newline.
    """
    assert_equal(swapped("^", "#", "  a  "), "#  a  ", "^ at the start")
    assert_equal(swapped("\\A", "#", "abc"), "#abc", "\\A at the start")
    assert_equal(swapped("^a", "#", "aaa"), "#aa", "^a is not every a")
    assert_equal(swapped("(?m)^", "#", "a\nb"), "#a\n#b", "(?m)^ on two lines")
    assert_equal(swapped("(?m)$", "#", "a\nb"), "a#\nb#", "(?m)$ on two lines")
    assert_equal(swapped("$", "#", "a\nb"), "a\nb#", "$ is only the end")
    assert_equal(swapped("\\z", "#", "abc"), "abc#", "\\z is only the end")


def test_the_cursor_moves_a_character_at_a_time() raises:
    """The second rule, which is the counting scan's rule the other way round.

    A row of five characters written in six bytes takes six markers from a
    pattern that matches nothing, where `str.count` on the same row and the same
    pattern answers seven. Both numbers are pandas'.
    """
    assert_equal(swapped("x*", "#", "abc"), "#a#b#c#", "x* in abc")
    assert_equal(swapped("x*", "#", "héllo"), "#h#é#l#l#o#", "x* in héllo")
    assert_equal(swapped("x*", "#", "٣٤"), "#٣#٤#", "x* in two Arabic digits")
    assert_equal(swapped("x*", "#", ""), "#", "x* in nothing")
    assert_equal(swapped("", "#", "abc"), "#a#b#c#", "an empty pattern")


def test_an_empty_match_where_the_last_one_ended_is_thrown_away() raises:
    """The third rule, which is the one Python's `re` does not have.

    `re.sub("a*", "#", "abc")` is `##b#c#`, because the empty match just after
    the `a` is a match like any other. Arrow refuses it and copies the `b` out
    instead, so the answer is `#b#c#` and is one marker shorter.
    """
    assert_equal(swapped("a*", "#", "abc"), "#b#c#", "a* in abc")
    assert_equal(swapped("a*", "#", "aaaa"), "#", "a* in aaaa")
    assert_equal(
        swapped("a*", "#", "  a  "), "# # # # #", "a* in a with spaces"
    )
    assert_equal(swapped("b?", "#", "abc"), "#a#c#", "b? in abc")
    assert_equal(swapped("b?", "#", "aaaa"), "#a#a#a#a#", "b? in aaaa")
    assert_equal(swapped("a*", "#", ""), "#", "a* in nothing")


def test_a_boundary_is_ascii_and_is_found_ahead_of_the_cursor() raises:
    """A word boundary, which reads `\\w` as ASCII the way RE2 does.

    An accented letter is not a word character to RE2, so a five letter word
    with one in the middle has four boundaries rather than two, and a row of
    Arabic digits has none at all. Python reads both the other way.
    """
    assert_equal(swapped("\\b", "#", "  a  "), "  #a#  ", "\\b around one a")
    assert_equal(swapped("\\b", "#", "abc"), "#abc#", "\\b around a word")
    assert_equal(swapped("\\b", "#", "héllo"), "#h#é#llo#", "\\b in héllo")
    assert_equal(swapped("\\b", "#", "٣٤"), "٣٤", "\\b in two Arabic digits")
    assert_equal(swapped("\\b", "#", "ab ba"), "#ab# #ba#", "\\b in two words")


def test_the_first_arm_wins() raises:
    """Leftmost first, which is visible in what is left rather than in a count.

    `a|ab` takes the one letter and leaves the `b`, and `ab|a` takes both. Both
    are RE2's reading and Python's, and a leftmost longest engine would answer
    the same thing twice.
    """
    assert_equal(swapped("a|ab", "#", "abc"), "#bc", "a|ab in abc")
    assert_equal(swapped("ab|a", "#", "abc"), "#c", "ab|a in abc")
    assert_equal(swapped("a|ab", "#", "ab ba"), "#b b#", "a|ab in ab ba")
    assert_equal(swapped("ab|a", "#", "ab ba"), "# b#", "ab|a in ab ba")


def test_a_group_is_written_where_the_replacement_asks_for_it() raises:
    """What a group held, put back in the order the replacement wants.

    `\\0` is the whole match, which RE2 has and Python's `re` spells `\\g<0>`,
    and the numbered groups count from one in the order their brackets opened.
    """
    assert_equal(
        swapped("[0-9]+", "<\\0>", "x1y22z333"),
        "x<1>y<22>z<333>",
        "the whole match",
    )
    assert_equal(
        swapped("([a-z])([0-9]+)", "\\2\\1", "x1y22z333"),
        "1x22y333z",
        "two groups the other way round",
    )
    assert_equal(
        swapped("(a+)", "\\1\\1", "aabc"), "aaaabc", "one group written twice"
    )
    assert_equal(
        swapped("(a)|(b)", "[\\1\\2]", "ab ba"),
        "[a][b] [b][a]",
        "an arm that did not take part writes nothing",
    )
    assert_equal(
        swapped("(a)(b)?", "<\\1|\\2>", "abc"),
        "<a|b>c",
        "a group that did take part",
    )
    assert_equal(
        swapped("(a)(b)?", "<\\1|\\2>", "aaaa"),
        "<a|><a|><a|><a|>",
        "a group that did not",
    )


def test_a_backslash_in_a_replacement_is_one_byte() raises:
    """`\\\\` is a backslash, and it is the only escape that is not a digit.

    The row here is one where the answer is longer than the pattern makes it
    look, since four letters each become one backslash.
    """
    assert_equal(swapped("a", "\\\\", "abc"), "\\bc", "one backslash")
    assert_equal(swapped("a", "\\\\", "aaaa"), "\\\\\\\\", "four of them")
    assert_equal(swapped("a", "x\\\\y", "abc"), "x\\ybc", "one in the middle")


def test_a_replacement_that_re2_cannot_read_is_refused() raises:
    """The three ways a rewrite string is wrong, and none of them is a raise.

    A refusal is a value here for the reason a refused pattern is: the layer
    holding the call decides what it becomes in Python, and every one of these
    is a refusal RE2 makes too, so all three are a `ValueError` upstream.
    """
    var one = parse_rewrite(String("a\\"), 0)
    assert_false(one.ok, "a replacement ending in a backslash")
    var two = parse_rewrite(String("a\\n"), 0)
    assert_false(two.ok, "a backslash and a letter")
    var three = parse_rewrite(String("\\1"), 0)
    assert_false(three.ok, "a group the pattern does not have")
    var four = parse_rewrite(String("\\1"), 1)
    assert_true(four.ok, "a group the pattern does have")
    var five = parse_rewrite(String("\\0"), 0)
    assert_true(five.ok, "the whole match, which every pattern has")


def test_a_two_digit_group_is_a_group_and_a_digit() raises:
    """`\\10` is group one and then the character zero, which is RE2's reading.

    Python's `re` reads it as group ten when there are ten, so this is a real
    difference and not a corner nobody reaches. There is no way to name a tenth
    group in a rewrite string at all.
    """
    assert_equal(
        swapped("(a)", "\\10", "abc"), "a0bc", "one group and a literal zero"
    )
    var refused = parse_rewrite(String("\\10"), 0)
    assert_false(refused.ok, "a pattern with no group at all")


def test_a_row_that_matches_nothing_comes_back_as_it_was() raises:
    """The cheap case, which has to copy the row across rather than drop it."""
    assert_equal(
        swapped("qqq", "#", "abc"), "abc", "a pattern that is not there"
    )
    assert_equal(swapped("q+", "#", ""), "", "an empty row")
    assert_equal(swapped("[0-9]", "#", "héllo"), "héllo", "a row of letters")


def bounded(
    pattern: StringSlice, repl: StringSlice, text: StringSlice, limit: Int
) raises -> String:
    """The text with the first `limit` matches replaced.

    The same as `swapped` with a count on it, written out here because
    `replaced_text` does not take one and the scan is what is being asked
    about.

    Args:
        pattern: The pattern.
        repl: The replacement.
        text: The text.
        limit: How many matches to replace.

    Returns:
        The answer.

    Raises:
        Error: If the pattern or the replacement could not be read.
    """
    var program = compiled(String(pattern))
    var rewrite = parse_rewrite(String(repl), program.groups)
    if not rewrite.ok:
        raise Error(
            String("replacement ", repl, " was refused: ", rewrite.problem)
        )
    var points = List[UInt32]()
    var bytes = text.as_bytes()
    decode_into(bytes, points)
    var machine = Machine(program)
    var offsets = List[Int]()
    var found = List[Int32]()
    var out = List[UInt8]()
    replaced(
        program,
        rewrite,
        bytes,
        Span(points),
        machine,
        offsets,
        found,
        out,
        limit,
    )
    return String(StringSlice(unsafe_from_utf8=Span(out)))


def test_a_limit_stops_the_scan_and_the_rest_of_the_row_is_copied() raises:
    """What SQL's `REGEXP_REPLACE` asks for, which no `str` caller ever does.

    The tail is the part worth asserting. A scan that stopped and returned what
    it had would drop everything after the last match it made, which is a wrong
    answer on every row that matched fewer times than it holds.
    """
    assert_equal(bounded("a", "Z", "aaaa", 1), "Zaaa", "one of four")
    assert_equal(bounded("a", "Z", "aaaa", 2), "ZZaa", "two of four")
    assert_equal(bounded("a", "Z", "aaaa", 9), "ZZZZ", "more than there are")
    assert_equal(bounded("a", "Z", "aaaa", -1), "ZZZZ", "and all of them")
    assert_equal(bounded("a", "Z", "  a  ", 1), "  Z  ", "the ends are kept")
    assert_equal(bounded("q", "Z", "abc", 1), "abc", "a row with no match")


def test_a_limit_of_zero_writes_the_row_out_as_it_arrived() raises:
    """The edge nobody writes on purpose and every off by one lands on."""
    assert_equal(bounded("a", "Z", "aaaa", 0), "aaaa", "nothing was replaced")
    assert_equal(bounded("a", "Z", "", 0), "", "and an empty row is empty")


def test_a_limit_counts_matches_and_not_the_characters_stepped_over() raises:
    """An empty match that is thrown away is not one of the replacements.

    `a*` on `abc` gives `#b#c#` with no limit. The `a` is one match, the empty
    match after it is refused and the `b` is copied across without counting, so
    a limit of two reaches the marker before the `c` and stops there.
    """
    assert_equal(bounded("a*", "#", "abc", 1), "#bc", "the run of a is one")
    assert_equal(bounded("a*", "#", "abc", 2), "#b#c", "and the next is two")
    assert_equal(bounded("a*", "#", "abc", 3), "#b#c#", "and three is all")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
