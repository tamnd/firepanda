"""Tests the kernel that pulls capture groups out of a column.

This is the first regular expression kernel here whose answer is more than one
column, and the first that runs Python's engine rather than RE2's, so the two
things worth testing are the shape of the answer and which alphabet the pattern
was read in.

The shape is three facts that look like one until a row disagrees with itself.
A row with no match is null in every column. A row whose match left a group out
is null in that column and holds text in the others. And a null row is null
everywhere, which is the same answer as a row with no match and arrives by a
different route, so a kernel that muddled the two would pass every test that
only looked at the answer.

The alphabet is one assertion and would be one whichever way it went, which is
why it is here: `\\w` reads 138558 code points for this method and 63 for
`str.count`, on the same accessor, and a kernel that quietly compiled for RE2
would answer this column correctly for every pattern made of ASCII.
"""

from std.testing import TestSuite, assert_equal, assert_true

from firepanda.array.strings import StringArray, StringBuilder
from firepanda.exec import MORSEL_ROWS
from firepanda.kernel.regex.column import text_extract_regex
from firepanda.kernel.regex.method import METHOD_EXTRACT, program_for
from firepanda.kernel.regex.program import Program


def compiled(pattern: String) raises -> Program:
    """Compiles a pattern the way `str.extract` would, or fails the test.

    Args:
        pattern: The pattern.

    Returns:
        The program, which carries captures and was built for Python's engine.

    Raises:
        Error: If it did not compile, which in this file is a mistake in the
            test rather than an answer.
    """
    var program = program_for(METHOD_EXTRACT, pattern)
    if not program.ok:
        raise Error(
            String("pattern ", pattern, " did not compile: ", program.problem)
        )
    return program^


def column(rows: List[String], nulls: List[Int]) raises -> StringArray:
    """Builds a text column, with the rows at the given positions missing.

    Args:
        rows: The text of every row, including a placeholder for the missing
            ones so that the two lists line up by position.
        nulls: Which rows are missing.

    Returns:
        The column.

    Raises:
        Error: If the builder cannot allocate.
    """
    var built = StringBuilder(capacity=len(rows))
    for i in range(len(rows)):
        var missing = False
        for at in nulls:
            if at == i:
                missing = True
        if missing:
            built.append_null()
        else:
            built.append(rows[i].as_bytes())
    return built^.finish()


def read(a: StringArray, at: Int) -> String:
    """One row of a column as text, or the word null.

    Writing the word rather than asserting validity separately keeps a failure
    readable: a test that expected `b` and got nothing says so in one line.

    Args:
        a: The column.
        at: The row.

    Returns:
        The row, or `null`.
    """
    if not a.is_valid(at):
        return String("null")
    return String(StringSlice(unsafe_from_utf8=a.unsafe_bytes(at)))


def tiled(
    rows: List[String], nulls: List[Int], times: Int
) raises -> StringArray:
    """The same rows over and over, for a column taller than one morsel.

    `column` walks the null positions once per row, which is fine for the eight
    row columns above and is a second pass over a hundred thousand rows here, so
    this one takes the pattern of rows and repeats it instead.

    Args:
        rows: The text of every row of one tile.
        nulls: Which rows of the tile are missing.
        times: How many tiles.

    Returns:
        The column, `times` times as tall as the tile.

    Raises:
        Error: If the builder cannot allocate.
    """
    var built = StringBuilder(capacity=len(rows) * times)
    for _ in range(times):
        for i in range(len(rows)):
            var missing = False
            for at in nulls:
                if at == i:
                    missing = True
            if missing:
                built.append_null()
            else:
                built.append(rows[i].as_bytes())
    return built^.finish()


def same(a: StringArray, i: Int, b: StringArray, j: Int) -> Bool:
    """Whether two rows of two columns say the same thing.

    `read` builds a string per call, which is what makes a failure readable and
    is more than the tall column below wants to pay a hundred thousand times, so
    the comparison there goes byte by byte instead.

    Args:
        a: One column.
        i: The row of it.
        b: The other column.
        j: The row of that one.

    Returns:
        True if both rows are missing, or both hold the same bytes.
    """
    if a.is_valid(i) != b.is_valid(j):
        return False
    if not a.is_valid(i):
        return True
    var x = a.unsafe_bytes(i)
    var y = b.unsafe_bytes(j)
    if len(x) != len(y):
        return False
    for k in range(len(x)):
        if x[k] != y[k]:
            return False
    return True


def test_a_group_comes_back_as_a_column_of_what_it_held() raises:
    """The ordinary case, which is one group over rows that all match."""
    var a = column(["ab1", "cd2", "ef3"], [])
    var out = text_extract_regex(a, compiled("([a-z])\\d"))
    assert_equal(len(out), 1)
    assert_equal(read(out[0], 0), "b")
    assert_equal(read(out[0], 1), "d")
    assert_equal(read(out[0], 2), "f")


def test_the_match_is_looked_for_anywhere_rather_than_at_the_front() raises:
    """`str.extract` runs `search` and not `match`, so a pattern with no anchor
    finds its match wherever it falls. This is the assertion that would fail if
    the anchoring the three mask methods do had been copied across."""
    var a = column(["xxa1", "a1"], [])
    var out = text_extract_regex(a, compiled("([a-z])(\\d)"))
    assert_equal(len(out), 2)
    assert_equal(read(out[0], 0), "a")
    assert_equal(read(out[1], 0), "1")
    assert_equal(read(out[0], 1), "a")


def test_a_row_with_no_match_is_missing_in_every_column() raises:
    """The columns of one row agree about whether there was a match at all,
    which is what makes a frame of them readable across rather than only down.
    """
    var a = column(["a1", "zz", "b2"], [])
    var out = text_extract_regex(a, compiled("([a-z])(\\d)"))
    assert_equal(read(out[0], 1), "null")
    assert_equal(read(out[1], 1), "null")
    assert_equal(read(out[0], 0), "a")
    assert_equal(read(out[1], 2), "2")


def test_a_group_that_took_no_part_is_missing_on_its_own() raises:
    """The one case where the columns of a row disagree, and the reason the
    answer above is not simply a match flag repeated across the width. The row
    matched, so the first group holds text, and the optional group was never
    entered, so the second is nothing."""
    var a = column(["a", "ax"], [])
    var out = text_extract_regex(a, compiled("(a)(x)?"))
    assert_equal(read(out[0], 0), "a")
    assert_equal(read(out[1], 0), "null")
    assert_equal(read(out[0], 1), "a")
    assert_equal(read(out[1], 1), "x")


def test_a_missing_row_is_missing_in_every_column() raises:
    """The same answer a row with no match gets, arrived at without running the
    engine, which is worth a test of its own because the two routes write the
    same thing and only one of them looks at the text."""
    var a = column(["a1", "", "b2"], [1])
    var out = text_extract_regex(a, compiled("([a-z])(\\d)"))
    assert_equal(read(out[0], 1), "null")
    assert_equal(read(out[1], 1), "null")
    assert_equal(read(out[0], 2), "b")


def test_a_group_can_hold_nothing_without_being_missing() raises:
    """A group that matched the empty string is not a group that was left out,
    and a column has to be able to say which. Both rows below matched and the
    first one's group is empty text rather than a null."""
    var a = column(["b", "ab"], [])
    var out = text_extract_regex(a, compiled("(a*)b"))
    assert_true(out[0].is_valid(0))
    assert_equal(read(out[0], 0), "")
    assert_equal(read(out[0], 1), "a")


def test_the_class_is_read_in_pythons_alphabet_and_not_re2s() raises:
    """The one assertion here about which engine ran. `str.count` reads `\\w` as
    63 characters of ASCII because it goes to Arrow, and this method reads it as
    138558 code points because it does not go to Arrow at all. A row of Greek
    matches for one reading and not the other."""
    var a = column(["ωx", "-x"], [])
    var out = text_extract_regex(a, compiled("(\\w)x"))
    assert_equal(read(out[0], 0), "ω")
    assert_equal(read(out[0], 1), "null")


def test_a_group_can_hold_more_than_one_byte_a_character() raises:
    """The slicing is by byte and the engine walks characters, so a row holding
    anything outside ASCII is where an offset that was not converted shows up.
    Every character here is two bytes, so a group cut at character positions
    would come back as the wrong half of the row rather than as nothing."""
    var a = column(["ααβγ"], [])
    var out = text_extract_regex(a, compiled("(α+)(β)"))
    assert_equal(read(out[0], 0), "αα")
    assert_equal(read(out[1], 0), "β")


def test_the_leftmost_match_wins_and_not_the_longest() raises:
    """Python takes the first match the pattern can make from the leftmost
    position it can start at, which is not the same as the longest one, and the
    two differ in what the groups hold rather than in whether there was a
    match."""
    var a = column(["abcd"], [])
    var out = text_extract_regex(a, compiled("(ab|abc)"))
    assert_equal(read(out[0], 0), "ab")


def test_several_rows_run_on_one_machine_without_leaking() raises:
    """Everything the scan needs is built once and handed to every row, so a row
    that left a thread or an offset behind would change the answer of the row
    after it. The rows below are chosen to disagree in length, in whether they
    match, and in where the match falls."""
    var a = column(["a1", "no", "zzzzzzzzb2", "", "c3", "d", "e5"], [3])
    var out = text_extract_regex(a, compiled("([a-z])(\\d)"))
    assert_equal(read(out[0], 0), "a")
    assert_equal(read(out[0], 1), "null")
    assert_equal(read(out[0], 2), "b")
    assert_equal(read(out[1], 2), "2")
    assert_equal(read(out[0], 3), "null")
    assert_equal(read(out[0], 4), "c")
    assert_equal(read(out[0], 5), "null")
    assert_equal(read(out[0], 6), "e")


def test_a_named_group_is_a_group_that_also_has_a_label() raises:
    """Naming a group changes what the answer is called and nothing about what
    it holds, and the labels ride on the program rather than being worked out
    again from the pattern."""
    var program = compiled("(?P<letter>[a-z])(\\d)")
    assert_equal(len(program.labels), 2)
    assert_equal(program.labels[0], "letter")
    assert_equal(program.labels[1], "")
    var a = column(["a1"], [])
    var out = text_extract_regex(a, program)
    assert_equal(read(out[0], 0), "a")
    assert_equal(read(out[1], 0), "1")


def test_a_group_inside_a_lookahead_is_pulled_out_like_any_other() raises:
    """The construct this method could not avoid asking about, since a capture
    inside a lookahead was refused only for a caller who wants the groups and
    this method is the one that wants them. It was the last one left: a
    backreference was refused until document 95, an atomic group until document
    99, a conditional group until document 100 and this until document 119.

    The body is wider than the match here on purpose. `(?=(ab))a` reads two
    characters inside the lookahead, gives the position back and then matches
    one, so the group holds `ab` where the match holds `a`, and a kernel that
    quietly reported the text the match covered rather than the text the body
    covered would answer the first row with the wrong letter count. The second
    row is a row the body turns down, which is null everywhere."""
    var program = compiled("(?=(ab))a")
    var a = column(["ab", "ac"], [])
    var out = text_extract_regex(a, program)
    assert_equal(len(out), 1)
    assert_equal(read(out[0], 0), "ab")
    assert_equal(read(out[0], 1), "null")


def test_the_groups_a_lookaround_leaves_are_the_ones_upstream_leaves() raises:
    """Seven shapes, every answer read off a running CPython rather than
    reasoned about, because the interesting half of this is which group is left
    unset rather than which text lands in the ones that are set.

    The first says the body is matched the way the pattern prefers rather than
    the way that reads most, since `a|ab` takes its first arm and the match
    that follows reads both characters anyway. The second is the same question
    asked of two groups: the arm nobody took is null and stays null. The third
    is a group each side of the assertion. The fourth is the rule for a
    negative one, which is that a body that matched is a body the pattern threw
    away, so its group is not set. The fifth is a lookahead inside a lookahead,
    which is the nesting working by recursion. The sixth is the one that would
    have caught a machine that wrote the body's slots into the path before
    knowing the body matched: the arm holding the assertion fails at position
    zero and the arm beside it wins, so group two is null rather than holding
    what the failed body read. And the seventh is the other direction."""
    var prefers = text_extract_regex(
        column(["ab"], []), compiled("(?=(a|ab))ab")
    )
    assert_equal(read(prefers[0], 0), "a")

    var arms = text_extract_regex(
        column(["ab"], []), compiled("(?=(ab)|(a))ab")
    )
    assert_equal(read(arms[0], 0), "ab")
    assert_equal(read(arms[1], 0), "null")

    var both = text_extract_regex(column(["ab"], []), compiled("(a)(?=(b))"))
    assert_equal(read(both[0], 0), "a")
    assert_equal(read(both[1], 0), "b")

    var negative = text_extract_regex(
        column(["cb"], []), compiled("(?!(a))(b)")
    )
    assert_equal(read(negative[0], 0), "null")
    assert_equal(read(negative[1], 0), "b")

    var nested = text_extract_regex(
        column(["ab"], []), compiled("(?=(a(?=(b))))ab")
    )
    assert_equal(read(nested[0], 0), "a")
    assert_equal(read(nested[1], 0), "b")

    var dead = text_extract_regex(column(["ba"], []), compiled("((?=(a))|b)a"))
    assert_equal(read(dead[0], 0), "b")
    assert_equal(read(dead[1], 0), "null")

    var behind = text_extract_regex(column(["ab"], []), compiled("(?<=(a))b"))
    assert_equal(read(behind[0], 0), "a")


def test_syntax_only_re2_refuses_is_compiled_rather_than_refused() raises:
    """A comment group is Python syntax RE2 has never had, and the five methods
    that go to Arrow refuse it because Arrow does. This one does not go to
    Arrow, so refusing it would be refusing on behalf of an engine nobody
    asked."""
    var a = column(["ab"], [])
    var out = text_extract_regex(a, compiled("(?#note)(a)"))
    assert_equal(read(out[0], 0), "a")


def test_extracting_past_one_morsel_says_what_one_morsel_said() raises:
    """A column that crosses the morsel split answers what the same rows answer
    inside one morsel. The kernel builds a payload per morsel per group and puts
    them end to end afterwards, so a row past the split whose group is too long
    to live inside its view is reading an offset that was moved, and the tile
    below holds one of those in each of the two groups. The other rows are the
    three ways a group comes back missing, which is what would go wrong if the
    validity of one morsel were written over the validity of another."""
    var rows: List[String] = [
        "abcdefghijklmnopq1234567890123",
        "ab2",
        "cd",
        "999",
        "",
    ]
    var nulls: List[Int] = [4]
    var program = compiled("([a-z]+)(\\d+)?")
    var short = text_extract_regex(tiled(rows, nulls, 1), program)
    assert_equal(read(short[0], 0), "abcdefghijklmnopq")
    assert_equal(read(short[1], 0), "1234567890123")
    assert_equal(read(short[1], 2), "null")
    assert_equal(read(short[0], 3), "null")
    assert_equal(read(short[0], 4), "null")

    var times = MORSEL_ROWS // len(rows) + 2
    var a = tiled(rows, nulls, times)
    assert_true(len(a) > MORSEL_ROWS)
    var tall = text_extract_regex(a, program)
    assert_equal(len(tall), len(short))
    for g in range(len(tall)):
        assert_equal(len(tall[g]), len(a))
        var wrong = 0
        for i in range(len(tall[g])):
            if not same(tall[g], i, short[g], i % len(rows)):
                wrong += 1
        assert_equal(wrong, 0, String("rows disagreeing in group ", g))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
