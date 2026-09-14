"""Tests for substring search over a text column, and for the matcher behind it.

Every test here is the kernel against the scalar twin in `scalar.mojo`, which
tries every position and compares every byte, because the whole content of the
kernel is the positions it manages not to try. A twin that skipped the same way
would agree with the kernel about anything the skipping got wrong.

The matcher at the end is held to the same rule and the twin is chosen the same
way. The kernel walks both strings once and remembers one wildcard to go back
to, so its twin fills a table and reconsiders nothing, and the thing the kernel
could get wrong is the thing the twin has no way to get wrong. Every expected
answer in those tests was read off DuckDB 1.5.1, so the twin holds the two
implementations together and the numbers hold both of them to the dialect.

The lengths are chosen to walk both halves of the search. `SCAN_WIDTH` is
sixteen and a block reads through where the needle's last byte would fall, so a
row under twenty bytes never enters the block loop and a row of a hundred goes
round it several times. Both are here, and so is a match that starts at byte
thirty one, which is the one an off by one in the block loop's limit would lose.

One thing about this file is worth knowing before adding to it. The per test
times the harness prints are not wall clock and should not be used to decide
anything. The last test has been reported as anything from twenty one to a
hundred and ninety seconds across runs whose real time never moved off five. If
the question is how long something takes, time the whole file, and throw away
the first run after a sync: that one pays for a cold compile cache and costs
about twenty five seconds against five for a warm one.

The last test still hands whole columns to the library and asserts once rather
than reading rows one at a time, which is worth doing on its own merits. The
comparison and the reduction are both kernels and the only thing crossing back
into the test is a count, so a failure anywhere in a hundred thousand rows is
one assertion rather than a hundred thousand.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.array import Array
from firepanda.array.strings import (
    StringArray,
    StringBuilder,
    strings_from_list,
)
from firepanda.exec.morsel import MORSEL_ROWS
from firepanda.kernel.agg import sum_of
from firepanda.kernel.compare import not_equal
from firepanda.kernel.concat import concat_strings
from firepanda.kernel.pattern import (
    MatchKind,
    find_bytes,
    read_pattern,
    text_contains,
    text_contains_in_order,
    text_count,
    text_ends_with,
    text_equals,
    text_contains_folded,
    text_equals_folded,
    text_like,
    text_replace,
    text_replace_folded,
    text_starts_with,
    text_starts_with_folded,
)
from firepanda.kernel.scalar import (
    text_contains_in_order_scalar,
    text_contains_scalar,
    text_count_scalar,
    text_ends_with_scalar,
    text_contains_folded_scalar,
    text_equals_folded_scalar,
    text_equals_scalar,
    text_like_scalar,
    text_replace_folded_scalar,
    text_replace_scalar,
    text_starts_with_folded_scalar,
    text_starts_with_scalar,
)


def padded(prefix: String, count: Int, tail: String) -> String:
    """Builds a string of a chosen length with a known front and back.

    Args:
        prefix: What the string starts with.
        count: How many filler bytes to put between the two.
        tail: What the string ends with.

    Returns:
        The string.
    """
    var out = String(prefix)
    for i in range(count):
        out += "abcdefghij"[byte=i % 10]
    return out + tail


def sample() -> StringArray:
    """Builds the column every test below searches.

    The rows cover the cases the two loops split on: shorter than a block,
    longer than several, a match at the front, a match at the back, one that
    starts at byte thirty one so that it crosses the block boundary, and two
    nulls.

    Returns:
        The column.
    """
    var rows = List[String]()
    rows.append("green")
    rows.append("forest green thread")
    rows.append("ignored")
    rows.append("gree")
    rows.append("ignored")
    rows.append(padded("", 31, "green tail"))
    rows.append(padded("green", 40, "green"))
    rows.append(padded("", 100, "green"))
    rows.append("greengreen")
    rows.append("ggggggggggggggggggggggggggggggggreen")

    var builder = StringBuilder(capacity=len(rows))
    for i in range(len(rows)):
        if i == 2 or i == 4:
            builder.append_null()
        else:
            builder.append(rows[i].as_bytes())
    return builder^.finish()


def agrees(
    got: Array[DType.bool], want: Array[DType.bool], label: String
) raises:
    """Asserts that a kernel answer matches the twin's, row by row.

    Args:
        got: The kernel's answer.
        want: The twin's answer.
        label: What to name in the failure.

    Raises:
        AssertionError: On the first row that differs.
    """
    assert_equal(len(got), len(want), label + ": lengths differ")
    for i in range(len(got)):
        assert_equal(got.is_valid(i), want.is_valid(i), label + ": validity")
        if got.is_valid(i):
            assert_equal(got[i], want[i], label + ": row " + String(i))


def check_contains(col: StringArray, needle: String) raises:
    """Runs the contains kernel and its twin and asserts they agree.

    Args:
        col: The column.
        needle: The substring.

    Raises:
        AssertionError: If they disagree.
    """
    agrees(
        text_contains(col, needle.as_bytes()),
        text_contains_scalar(col, needle),
        "contains " + needle,
    )


def check_starts(col: StringArray, needle: String) raises:
    """Runs the starts with kernel and its twin and asserts they agree.

    Args:
        col: The column.
        needle: The substring.

    Raises:
        AssertionError: If they disagree.
    """
    agrees(
        text_starts_with(col, needle.as_bytes()),
        text_starts_with_scalar(col, needle),
        "starts_with " + needle,
    )


def check_ends(col: StringArray, needle: String) raises:
    """Runs the ends with kernel and its twin and asserts they agree.

    Args:
        col: The column.
        needle: The substring.

    Raises:
        AssertionError: If they disagree.
    """
    agrees(
        text_ends_with(col, needle.as_bytes()),
        text_ends_with_scalar(col, needle),
        "ends_with " + needle,
    )


def check_pair(col: StringArray, first: String, second: String) raises:
    """Runs the ordered pair kernel and its twin and asserts they agree.

    Args:
        col: The column.
        first: The substring that must come first.
        second: The substring that must follow it.

    Raises:
        AssertionError: If they disagree.
    """
    agrees(
        text_contains_in_order(col, first.as_bytes(), second.as_bytes()),
        text_contains_in_order_scalar(col, first, second),
        "in_order " + first + " " + second,
    )


def check_equals(col: StringArray, other: String) raises:
    """Runs the equality kernel and its twin and asserts they agree.

    Args:
        col: The column.
        other: The string every row is compared against.

    Raises:
        AssertionError: If they disagree.
    """
    agrees(
        text_equals(col, other.as_bytes()),
        text_equals_scalar(col, other),
        "equals " + other,
    )


def check_count(col: StringArray, needle: String) raises:
    """Runs the count kernel and its twin and asserts they agree.

    Args:
        col: The column.
        needle: The substring to count.

    Raises:
        AssertionError: If they disagree.
    """
    var got = text_count(col, needle.as_bytes())
    var want = text_count_scalar(col, needle)
    assert_equal(len(got), len(want), "count " + needle + ": lengths differ")
    for i in range(len(got)):
        assert_equal(
            got.is_valid(i), want.is_valid(i), "count " + needle + ": validity"
        )
        if got.is_valid(i):
            assert_equal(
                got[i], want[i], "count " + needle + " row " + String(i)
            )


def test_contains_matches_the_twin() raises:
    var col = sample()
    check_contains(col, "green")
    check_contains(col, "g")
    check_contains(col, "greenx")
    check_contains(col, "")
    check_contains(col, "reen")
    check_contains(col, "ggggg")


def test_starts_with_matches_the_twin() raises:
    var col = sample()
    check_starts(col, "green")
    check_starts(col, "g")
    check_starts(col, "greenx")
    check_starts(col, "")
    check_starts(col, "forest")


def test_ends_with_matches_the_twin() raises:
    var col = sample()
    check_ends(col, "green")
    check_ends(col, "n")
    check_ends(col, "xgreen")
    check_ends(col, "")
    check_ends(col, "thread")


def test_two_substrings_in_order_match_the_twin() raises:
    var col = sample()
    check_pair(col, "green", "green")
    check_pair(col, "forest", "green")
    check_pair(col, "green", "forest")
    check_pair(col, "g", "n")
    check_pair(col, "", "green")


def test_two_substrings_in_order_is_not_two_contains() raises:
    # The reason the pair kernel exists. Both runs are present and they are in
    # the wrong order, which `LIKE '%bc%a%'` rejects and two independent contains
    # calls would accept.
    var col = strings_from_list(["abc"])
    assert_false(
        text_contains_in_order(col, "bc".as_bytes(), "a".as_bytes())[0]
    )
    assert_true(text_contains(col, "bc".as_bytes())[0])
    assert_true(text_contains(col, "a".as_bytes())[0])


def test_two_substrings_in_order_do_not_overlap() raises:
    # `LIKE '%aa%aa%'` needs four a's and not three, because the second run
    # starts after the first one ends rather than one byte into it.
    var col = strings_from_list(["aaa", "aaaa"])
    var mask = text_contains_in_order(col, "aa".as_bytes(), "aa".as_bytes())
    assert_false(mask[0])
    assert_true(mask[1])


def test_a_null_row_stays_null() raises:
    var col = sample()
    var mask = text_contains(col, "green".as_bytes())
    assert_false(mask.is_valid(2))
    assert_false(mask.is_valid(4))
    assert_true(mask.is_valid(0))
    assert_true(mask[0])


def test_find_reports_the_first_match_and_not_any_match() raises:
    var hay = String("greengreen")
    assert_equal(find_bytes(hay.as_bytes(), "green".as_bytes(), 0), 0)
    assert_equal(find_bytes(hay.as_bytes(), "green".as_bytes(), 1), 5)
    assert_equal(find_bytes(hay.as_bytes(), "green".as_bytes(), 6), -1)


def test_find_does_not_read_past_the_end() raises:
    # A candidate starting inside the last few bytes cannot fit, and a search
    # that checked it anyway would read whatever follows the string, which in a
    # column is the next element and is very likely to match.
    var hay = String("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab")
    assert_equal(hay.byte_length(), 32)
    assert_equal(find_bytes(hay.as_bytes(), "bc".as_bytes(), 0), -1)
    assert_equal(find_bytes(hay.as_bytes(), "b".as_bytes(), 0), 31)


def test_a_needle_longer_than_the_haystack_is_absent() raises:
    var col = strings_from_list(["ab"])
    assert_false(text_contains(col, "abcdef".as_bytes())[0])
    assert_false(text_starts_with(col, "abcdef".as_bytes())[0])
    assert_false(text_ends_with(col, "abcdef".as_bytes())[0])


def test_an_empty_needle_is_everywhere() raises:
    var col = strings_from_list(["ab", ""])
    for i in range(2):
        assert_true(text_contains(col, "".as_bytes())[i])
        assert_true(text_starts_with(col, "".as_bytes())[i])
        assert_true(text_ends_with(col, "".as_bytes())[i])


def test_a_column_either_side_of_the_morsel_split_matches_the_twin() raises:
    # The search runs on every core above a row count no short column reaches,
    # so the split between morsels has to be walked as well as the loop inside
    # one. The column is four rows doubled until it fills a morsel and then four
    # more on top, which is the smallest column that has a second morsel at all,
    # and the second morsel is the one that matters: it is where a kernel that
    # wrote its answer at `i - start` rather than at `i`, or repaired the
    # validity of the wrong range, comes apart.
    #
    # Every row is checked, and not one of them is read from this file. The twin
    # is asked for the whole column in one call, the two answers are compared by
    # a kernel and reduced by another, and the only thing crossing back into the
    # test is a count. Warm runs of the file with and without this test land in
    # the same five second band, so the whole thing is free next to compiling it.
    def unit() -> StringArray:
        return strings_from_list(
            ["forest green thread", padded("", 40, "green"), "nothing", "gree"]
        )

    var col = unit()
    while len(col) < MORSEL_ROWS:
        var pair = List[StringArray]()
        pair.append(col.copy())
        pair.append(col.copy())
        col = concat_strings(pair)
    assert_equal(len(col), MORSEL_ROWS)
    var tail = List[StringArray]()
    tail.append(col^)
    tail.append(unit())
    col = concat_strings(tail)

    var mask = text_contains(col, "green".as_bytes())
    var want = text_contains_scalar(col, "green")
    assert_equal(len(mask), MORSEL_ROWS + 4)
    assert_equal(
        Int(sum_of(not_equal(mask, want)).value), 0, "rows disagreeing"
    )
    # And that the twin was not vacuously right about every row, which is what a
    # column that came out empty or all null would let it be. Rows nought and
    # one hold the needle and rows two and three do not, and doubling keeps that
    # true, so exactly half the column matches.
    assert_equal(Int(sum_of(want).value), len(col) // 2)


def _read(pattern: String) raises -> String:
    """Reads a pattern and writes back which search it is and what it holds.

    Args:
        pattern: The pattern as a query would write it.

    Returns:
        The search, then the runs it reads, in brackets.
    """
    var got = read_pattern(pattern)
    return String(got.kind, " [", got.first, "] [", got.second, "]")


def test_a_pattern_with_no_wildcard_is_an_equality() raises:
    assert_equal(_read("abc"), "equals [abc] []")
    # The empty pattern too, which matches the empty element and nothing else.
    assert_equal(_read(""), "equals [] []")


def test_a_percent_at_one_end_is_a_prefix_or_a_suffix() raises:
    assert_equal(_read("abc%"), "starts with [abc] []")
    assert_equal(_read("%abc"), "ends with [abc] []")


def test_a_percent_at_both_ends_is_a_substring() raises:
    assert_equal(_read("%abc%"), "contains [abc] []")


def test_two_runs_each_wrapped_in_a_percent_are_read_in_order() raises:
    assert_equal(_read("%ab%cd%"), "contains in order [ab] [cd]")


def test_a_lone_percent_is_a_suffix_of_nothing() raises:
    # Which every element ends with and no null does, and that is what
    # `LIKE '%'` means. Two of them is the same thing said twice.
    assert_equal(_read("%"), "ends with [] []")
    assert_equal(_read("%%"), "contains [] []")


def test_an_underscore_sends_the_whole_pattern_to_the_matcher() raises:
    # And the pattern arrives whole, wildcards and all, because the matcher
    # reads them itself rather than being handed runs.
    assert_equal(_read("a_c"), "matches [a_c] []")
    assert_equal(_read("_"), "matches [_] []")


def test_an_underscore_wins_over_a_shape_that_would_otherwise_fit() raises:
    # This is the part that is correctness and not speed. `%a_b%` has a run
    # wrapped in percent signs on each side, so counting the runs would call it
    # a substring search and the underscore inside would be compared as an
    # ordinary byte. The check for one comes first for exactly that reason.
    assert_equal(_read("%a_b%"), "matches [%a_b%] []")
    assert_equal(_read("a_c%"), "matches [a_c%] []")
    assert_equal(_read("%a_c"), "matches [%a_c] []")


def test_a_run_in_the_middle_is_the_matcher_rather_than_a_wider_search() raises:
    # Reading `a%c` as the prefix alone would keep every row starting with an a,
    # which is more rows than the query asked for and nothing would say so.
    # None of these is one of the four, so all of them go to the sixth search.
    assert_equal(_read("a%c"), "matches [a%c] []")
    assert_equal(_read("%a%b"), "matches [%a%b] []")
    assert_equal(_read("%a%%b%"), "matches [%a%%b%] []")


def test_equality_matches_the_twin() raises:
    var col = sample()
    check_equals(col, "green")
    check_equals(col, "gree")
    check_equals(col, "")
    check_equals(col, "greengreen")
    check_equals(col, "forest green thread")
    check_equals(col, padded("", 100, "green"))


def test_count_matches_the_twin() raises:
    var col = sample()
    check_count(col, "green")
    check_count(col, "g")
    check_count(col, "gg")
    check_count(col, "greenx")
    check_count(col, "")
    check_count(col, "reen")


def test_a_count_does_not_let_matches_overlap() raises:
    # Four a's hold two runs of two and not three, because the cursor moves past
    # the whole needle after a hit. A regular expression engine says the same,
    # and this is the only rule in the kernel a caller is likely to have an
    # opinion about, so it is asserted on its own and not only through the twin.
    var col = strings_from_list(["aaaa", "aaaaa", "abab", "aa"])
    var got = text_count(col, "aa".as_bytes())
    assert_equal(got[0], 2)
    assert_equal(got[1], 2)
    assert_equal(got[2], 0)
    assert_equal(got[3], 1)


def test_an_empty_needle_is_counted_in_bytes_and_not_characters() raises:
    # This is Arrow's rule and it is the answer pandas gives, and it is not the
    # one Python's re module gives. A five character word holding one accented
    # letter is six bytes, so it holds seven empty matches here and six there.
    var col = strings_from_list(["hello", "h\u00e9llo", "", "\u65e5\u672c"])
    var got = text_count(col, "".as_bytes())
    assert_equal(got[0], 6)
    assert_equal(got[1], 7)
    assert_equal(got[2], 1)
    assert_equal(got[3], 7)


def test_equality_reads_no_bytes_when_the_lengths_disagree() raises:
    # Not a timing assertion, which a test cannot make. It is the boundary the
    # length check creates: a row one byte longer than the pattern and sharing
    # every byte of it is not equal, and an off by one in the comparison would
    # call it equal without ever reading past the pattern.
    var col = strings_from_list(["green", "greens", "gree", "GREEN"])
    var got = text_equals(col, "green".as_bytes())
    assert_true(got[0])
    assert_false(got[1])
    assert_false(got[2])
    assert_false(got[3])


def test_an_empty_pattern_is_equal_to_the_empty_row_alone() raises:
    var col = strings_from_list(["", "a", ""])
    var got = text_equals(col, "".as_bytes())
    assert_true(got[0])
    assert_false(got[1])
    assert_true(got[2])


def check_replace(
    col: StringArray, needle: String, repl: String, limit: Int
) raises:
    """Runs the replace kernel and its twin and asserts they agree.

    Args:
        col: The column.
        needle: The substring to look for.
        repl: What to put in its place.
        limit: How many matches per row.

    Raises:
        AssertionError: If they disagree.
    """
    var got = text_replace(col, needle.as_bytes(), repl.as_bytes(), limit)
    var want = text_replace_scalar(col, needle, repl, limit)
    var what = "replace " + needle + " with " + repl
    assert_equal(len(got), len(want), what + ": lengths differ")
    for i in range(len(got)):
        assert_equal(got.is_valid(i), want.is_valid(i), what + ": validity")
        if got.is_valid(i):
            assert_equal(got[i], want[i], what + " row " + String(i))


def test_replace_matches_the_twin() raises:
    var col = sample()
    check_replace(col, "green", "GREEN", -1)
    check_replace(col, "green", "", -1)
    check_replace(col, "g", "..", -1)
    check_replace(col, "green", "GREEN", 1)
    check_replace(col, "green", "GREEN", 2)
    check_replace(col, "green", "GREEN", 0)
    check_replace(col, "missing", "x", -1)
    check_replace(col, "", "-", -1)
    check_replace(col, "", "-", 3)


def test_replace_does_not_let_matches_overlap() raises:
    var col = strings_from_list(["aaaa", "aaaaa", "abab", "aa", "a"])
    var got = text_replace(col, "aa".as_bytes(), "X".as_bytes(), -1)
    assert_equal(got[0], "XX", "four a's hold two runs of two")
    assert_equal(got[1], "XXa", "five a's hold two and a leftover")
    assert_equal(got[2], "abab", "no run of two here")
    assert_equal(got[3], "X", "exactly one run")
    assert_equal(got[4], "a", "not long enough to hold one")


def test_an_empty_pattern_is_replaced_between_characters() raises:
    """Which is characters, where the same argument to count is bytes."""
    var col = strings_from_list(["hello", "h\u00e9llo", "", "\u65e5\u672c"])
    var got = text_replace(col, "".as_bytes(), "-".as_bytes(), -1)
    assert_equal(got[0], "-h-e-l-l-o-", "five characters take six dashes")
    assert_equal(
        got[1], "-h-\u00e9-l-l-o-", "six bytes but still five characters"
    )
    assert_equal(got[2], "-", "an empty row takes one")
    assert_equal(got[3], "-\u65e5-\u672c-", "two characters take three")


def test_an_empty_pattern_stops_when_the_limit_is_reached() raises:
    var col = strings_from_list(["abcabc", "ab", ""])
    var got = text_replace(col, "".as_bytes(), "-".as_bytes(), 2)
    assert_equal(got[0], "-a-bcabc", "two dashes and then the rest")
    assert_equal(got[1], "-a-b", "the row ends before the limit does")
    assert_equal(got[2], "-", "one place to put a dash and the limit is two")


def test_a_limit_of_zero_hands_the_row_back() raises:
    var col = sample()
    var got = text_replace(col, "green".as_bytes(), "X".as_bytes(), 0)
    for i in range(len(col)):
        assert_equal(got.is_valid(i), col.is_valid(i), "validity is kept")
        if col.is_valid(i):
            assert_equal(got[i], col[i], "row " + String(i) + " is unchanged")


def test_replace_keeps_a_missing_row_missing() raises:
    var col = sample()
    var got = text_replace(col, "green".as_bytes(), "X".as_bytes(), -1)
    assert_false(got.is_valid(2), "a missing row has nothing to replace")
    assert_false(got.is_valid(4), "and neither has the other one")


def probe() -> StringArray:
    """The eight rows every matcher answer below was read off DuckDB for.

    Chosen for what a wildcard can get wrong rather than for what a search can.
    Two of them hold characters wider than a byte, one is empty, one is a single
    character, and one alternates so that a pattern with two wildcards in it has
    somewhere to go wrong.

    Returns:
        The column, with no null in it. The nulls have their own test, because
        every number here is a DuckDB answer and DuckDB says null rather than
        true or false for a missing row.
    """
    return strings_from_list(
        [
            "green",
            "forest green thread",
            "abc",
            "héllo",
            "",
            "a",
            "日本語",
            "aXbXc",
        ]
    )


def matched(col: StringArray, pattern: String, want: List[Int]) raises:
    """Asserts the matcher answers a pattern the way DuckDB did, row by row.

    The twin runs on the same call and has to agree as well, which is what
    makes the general matcher held to the same standard as the four searches
    above it rather than to a list of numbers alone.

    Args:
        col: The column.
        pattern: The pattern, wildcards and all.
        want: One per row, one for a match and nought for none.

    Raises:
        AssertionError: On the first row that differs.
    """
    var got = text_like(col, pattern.as_bytes())
    agrees(got, text_like_scalar(col, pattern), "like " + pattern)
    assert_equal(len(got), len(want), "like " + pattern + ": lengths differ")
    for i in range(len(got)):
        assert_equal(
            got[i],
            want[i] == 1,
            "like " + pattern + " row " + String(i),
        )


def test_an_underscore_stands_for_one_character() raises:
    var col = probe()
    matched(col, "a_c", [0, 0, 1, 0, 0, 0, 0, 0])
    matched(col, "_", [0, 0, 0, 0, 0, 1, 0, 0])
    matched(col, "__", [0, 0, 0, 0, 0, 0, 0, 0])


def test_an_underscore_counts_characters_and_not_bytes() raises:
    # The one thing about `_` that is easy to get wrong. The accented letter is
    # two bytes and one character, so the pattern with one underscore matches
    # and the one with two does not.
    var col = probe()
    matched(col, "h_llo", [0, 0, 0, 1, 0, 0, 0, 0])
    matched(col, "h__llo", [0, 0, 0, 0, 0, 0, 0, 0])
    # And the same said with three byte characters, where a byte counter would
    # need three underscores rather than one.
    matched(col, "日_語", [0, 0, 0, 0, 0, 0, 1, 0])
    matched(col, "%_語", [0, 0, 0, 0, 0, 0, 1, 0])


def test_a_run_at_each_end_is_answered_rather_than_refused() raises:
    # The shape the four searches could not read. A prefix and a suffix at once,
    # which is more than either kernel can say on its own.
    var col = probe()
    matched(col, "a%c", [0, 0, 1, 0, 0, 0, 0, 1])
    matched(col, "%e%n", [1, 0, 0, 0, 0, 0, 0, 0])
    matched(col, "%a%b", [0, 0, 0, 0, 0, 0, 0, 0])


def test_two_wildcards_make_the_walk_go_back() raises:
    # `%X%X%` is the pattern that fails if the walk gives the first wildcard
    # everything it can take and has no way back, and `a%b%c` is the same
    # question with the ends anchored.
    var col = probe()
    matched(col, "%X%X%", [0, 0, 0, 0, 0, 0, 0, 1])
    matched(col, "a%b%c", [0, 0, 1, 0, 0, 0, 0, 1])
    matched(col, "a_b_c", [0, 0, 0, 0, 0, 0, 0, 1])
    matched(col, "%gree_%", [1, 1, 0, 0, 0, 0, 0, 0])


def test_a_percent_takes_nothing_as_readily_as_something() raises:
    var col = probe()
    matched(col, "%%c", [0, 0, 1, 0, 0, 0, 0, 1])
    matched(col, "%thread", [0, 1, 0, 0, 0, 0, 0, 0])
    # A lone `%` matches every row including the empty one, and `_%` matches
    # every row that has a character in it, which is the empty one's difference.
    matched(col, "%", [1, 1, 1, 1, 1, 1, 1, 1])
    matched(col, "_%", [1, 1, 1, 1, 0, 1, 1, 1])


def test_the_matcher_keeps_a_missing_row_missing() raises:
    var col = sample()
    var got = text_like(col, "%g_een%".as_bytes())
    agrees(got, text_like_scalar(col, "%g_een%"), "like %g_een%")
    assert_false(got.is_valid(2), "a missing row matches nothing")
    assert_false(got.is_valid(4), "and neither does the other one")
    assert_true(got[0], "green has an underscore's worth in the middle")


def test_the_matcher_agrees_with_the_twin_over_a_column_of_rows() raises:
    # The rows the searches use, which are long enough to walk a pattern round
    # a row several times, against patterns that use both wildcards together.
    var col = sample()
    var patterns = [
        String("%g%n%"),
        String("g_een%"),
        String("%g_een"),
        String("%green%green%"),
        String("_%_%_"),
        String("%%%"),
        String("gree_"),
        String("%a%b%c%d%"),
    ]
    for pattern in patterns:
        agrees(
            text_like(col, pattern.as_bytes()),
            text_like_scalar(col, pattern),
            "like " + pattern,
        )


# ---------------------------------------------------------------------------
# The case insensitive half
#
# Every test below runs the kernel and the twin against each other first, and
# the two do not share an idea. The kernel folds the pattern once and folds the
# row one character at a time as the search walks it, so it never holds a copy
# of anything the size of a column. The twin builds the folded copy of the whole
# row and then runs a plain substring search over it that knows nothing about
# case. A defect in the walk shows up as a disagreement; a defect in the table
# would have to be in both, which is why the table is checked against pyarrow by
# its generator and against pandas by the Python tests.
# ---------------------------------------------------------------------------


def folded_sample() -> StringArray:
    """Builds the column the case insensitive tests read.

    The rows are the ones a search fold gets wrong if it is the wrong fold. The
    sharp s and the ligature are the pair that separates the search fold from
    `str.casefold`, because folding for a reader sends them to two characters
    and a search sends them to themselves. The long s and the Kelvin sign are
    the pair that separates it from lower case, because Arrow lowers neither and
    folds both. The final sigma is the third such pair and the Turkish dotless i
    is the one that folds to nothing at all.

    Returns:
        The column, with two nulls in it.
    """
    var rows = List[String]()
    rows.append("Green")
    rows.append("GREEN")
    rows.append("ignored")
    rows.append("green")
    rows.append("ignored")
    rows.append("STRASSE")
    rows.append("straße")
    rows.append("Straße")
    rows.append("ſtraße")
    rows.append("ﬁance")
    rows.append("FIANCE")
    rows.append("KELVIN")
    rows.append("Kelvin")
    rows.append("ΣΟΦΟΣ")
    rows.append("σοφος")
    rows.append("İstanbul")
    rows.append("ıstanbul")
    rows.append("")

    var builder = StringBuilder(capacity=len(rows))
    for i in range(len(rows)):
        if i == 2 or i == 4:
            builder.append_null()
        else:
            builder.append(rows[i].as_bytes())
    return builder^.finish()


def check_folded(col: StringArray, needle: String) raises:
    """Runs the three folded flag kernels and their twins and compares.

    Args:
        col: The column.
        needle: The pattern.

    Raises:
        AssertionError: On the first row any of the three disagrees on.
    """
    agrees(
        text_contains_folded(col, needle.as_bytes()),
        text_contains_folded_scalar(col, needle),
        "contains folded " + needle,
    )
    agrees(
        text_starts_with_folded(col, needle.as_bytes()),
        text_starts_with_folded_scalar(col, needle),
        "starts with folded " + needle,
    )
    agrees(
        text_equals_folded(col, needle.as_bytes()),
        text_equals_folded_scalar(col, needle),
        "equals folded " + needle,
    )


def read(col: StringArray, i: Int) -> String:
    """One row of a text column as a string.

    Args:
        col: The column.
        i: The row.

    Returns:
        The row.
    """
    return col[i]


def test_the_folded_kernels_agree_with_the_twin() raises:
    """On every pattern that matters, which is the whole point of the twin."""
    var col = folded_sample()
    var patterns: List[String] = [
        "green",
        "GREEN",
        "Green",
        "straße",
        "strasse",
        "ss",
        "SS",
        "ß",
        "ſ",
        "s",
        "S",
        "k",
        "K",
        "σ",
        "ς",
        "Σ",
        "fi",
        "ﬁ",
        "i",
        "I",
        "ı",
        "İ",
        "e",
    ]
    for j in range(len(patterns)):
        check_folded(col, patterns[j])


def test_a_search_folds_one_character_to_one_character() raises:
    """Which is the whole difference between this fold and `str.casefold`.

    Folding a row for a reader sends the sharp s to two letters, so `Straße`
    and `STRASSE` fold to the same thing and a reader would call them the same
    word. A search does not get to do that, and pandas does not do it either:
    Arrow's `ignore_case` folds one code point to one code point, so these two
    rows are different rows to a case insensitive search.
    """
    var rows: List[String] = ["STRASSE", "straße", "Straße"]
    var col = strings_from_list(rows)
    var got = text_contains_folded(col, "straße".as_bytes())
    assert_false(got[0], "STRASSE does not hold the sharp s spelling")
    assert_true(got[1], "and the sharp s spelling holds itself")
    assert_true(got[2], "in either case")
    var other = text_contains_folded(col, "strasse".as_bytes())
    assert_true(other[0], "the double s spelling holds itself")
    assert_false(other[1], "and is not found in the sharp s spelling")


def test_the_search_fold_is_not_the_lower_case_either() raises:
    """Three pairs Arrow folds together and lowering leaves apart.

    The long s, the Kelvin sign and the micro sign all lower to themselves and
    all fold to an ordinary letter, so a case insensitive search finds them and
    a search over two lowered copies does not.
    """
    var rows: List[String] = ["ſ", "K", "µ"]
    var col = strings_from_list(rows)
    assert_true(text_equals_folded(col, "s".as_bytes())[0], "long s is an s")
    assert_true(text_equals_folded(col, "k".as_bytes())[1], "kelvin is a k")
    assert_true(text_equals_folded(col, "μ".as_bytes())[2], "micro is a mu")


def test_a_folded_match_can_cover_a_different_number_of_bytes() raises:
    """Which is why the search reports where a match ends and not how long it is.

    The long s is two bytes and the letter it folds to is one, so a pattern of
    one byte matches two bytes of the row, and a replace that assumed otherwise
    would cut the row in the middle of a character.
    """
    var rows: List[String] = ["ſtraße", "Straße"]
    var col = strings_from_list(rows)
    var got = text_replace_folded(col, "s".as_bytes(), "X".as_bytes(), -1)
    assert_equal(read(got, 0), "Xtraße", "the two byte s is replaced whole")
    assert_equal(read(got, 1), "Xtraße", "and so is the one byte one")


def test_folded_replace_keeps_the_case_of_what_it_did_not_touch() raises:
    """Because replacing a pattern is not folding a column."""
    var rows: List[String] = ["ABCdefABC", "abcDEFabc"]
    var col = strings_from_list(rows)
    var got = text_replace_folded(col, "abc".as_bytes(), "-".as_bytes(), -1)
    assert_equal(read(got, 0), "-def-", "the rest of the row is untouched")
    assert_equal(read(got, 1), "-DEF-", "in whatever case it was written in")


def test_folded_replace_obeys_the_count() raises:
    """At every sign it can have, as the exact one does."""
    var rows: List[String] = ["aAaA"]
    var col = strings_from_list(rows)
    assert_equal(
        read(text_replace_folded(col, "a".as_bytes(), "X".as_bytes(), 1), 0),
        "XAaA",
        "one from the left",
    )
    assert_equal(
        read(text_replace_folded(col, "a".as_bytes(), "X".as_bytes(), 3), 0),
        "XXXA",
        "three from the left",
    )
    assert_equal(
        read(text_replace_folded(col, "a".as_bytes(), "X".as_bytes(), -1), 0),
        "XXXX",
        "all of them",
    )
    assert_equal(
        read(text_replace_folded(col, "a".as_bytes(), "X".as_bytes(), 0), 0),
        "aAaA",
        "and none at all",
    )


def test_folded_replace_agrees_with_the_twin() raises:
    """On the rows where a match is a different width from the pattern."""
    var col = folded_sample()
    var patterns: List[String] = ["s", "ss", "ß", "e", "i", "σ", "green"]
    for j in range(len(patterns)):
        var got = text_replace_folded(
            col, patterns[j].as_bytes(), "-".as_bytes(), -1
        )
        var want = text_replace_folded_scalar(col, patterns[j], "-", -1)
        assert_equal(len(got), len(want), "lengths differ")
        for i in range(len(got)):
            assert_equal(got.is_valid(i), want.is_valid(i), "validity")
            if got.is_valid(i):
                assert_equal(
                    read(got, i),
                    read(want, i),
                    "row " + String(i) + " of " + patterns[j],
                )


def test_an_empty_pattern_has_nothing_to_do_with_case() raises:
    """So it goes to the exact kernel, which already has the rule for it."""
    var rows: List[String] = ["héllo", ""]
    var col = strings_from_list(rows)
    var got = text_replace_folded(col, "".as_bytes(), "-".as_bytes(), -1)
    assert_equal(read(got, 0), "-h-é-l-l-o-", "counted in characters")
    assert_equal(read(got, 1), "-", "and once in an empty row")
    assert_true(
        text_contains_folded(col, "".as_bytes())[0],
        "and every row holds nothing",
    )


def test_a_folded_search_keeps_a_missing_row_missing() raises:
    """As every other kernel in this file does."""
    var col = folded_sample()
    var got = text_contains_folded(col, "green".as_bytes())
    assert_false(got.is_valid(2), "a missing row is missing")
    assert_false(got.is_valid(4), "and so is the other one")
    var written = text_replace_folded(col, "e".as_bytes(), "-".as_bytes(), -1)
    assert_false(written.is_valid(2), "a missing row has nothing to replace")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
