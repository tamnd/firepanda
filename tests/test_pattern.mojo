"""Tests for substring search over a text column.

Every test here is the kernel against the scalar twin in `scalar.mojo`, which
tries every position and compares every byte, because the whole content of the
kernel is the positions it manages not to try. A twin that skipped the same way
would agree with the kernel about anything the skipping got wrong.

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
    find_bytes,
    text_contains,
    text_contains_in_order,
    text_ends_with,
    text_starts_with,
)
from firepanda.kernel.scalar import (
    text_contains_in_order_scalar,
    text_contains_scalar,
    text_ends_with_scalar,
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
