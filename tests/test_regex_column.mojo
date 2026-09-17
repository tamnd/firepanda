"""Tests the regular expression column kernel against its scalar twin.

The twin here is a narrower check than the rule this package usually holds to,
because it runs the same engine the kernel does rather than an obviously right
implementation of the same question. `firepanda/kernel/regex/column.mojo` says
why at length and says what checks the engine, which is the differential against
pandas and not this file.

What is left for the twin is everything the kernel does around the engine, and
it is not nothing: the morsel split, the null repair, and the buffers that are
built once and handed to every row of a morsel. A machine that failed to forget
the row before it would pass every test in `test_regex_pike.mojo`, which runs
one row per machine, and would come apart here on the second row of a column.

So the patterns below are chosen for what a row leaves behind rather than for
what they match. A pattern that matches early and returns, a pattern that
matches at the last position, a pattern that runs out of threads part way, and
an empty row between two long ones, all in one column in an order that makes a
leftover thread from one row change the answer of the next.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.array import Array
from firepanda.array.strings import (
    StringArray,
    StringBuilder,
    strings_from_list,
)
from firepanda.array.strview import INLINE_CAPACITY
from firepanda.exec.morsel import MORSEL_ROWS
from firepanda.kernel.agg import sum_of
from firepanda.kernel.compare import not_equal
from firepanda.kernel.concat import concat_strings
from firepanda.kernel.regex.column import (
    text_matches_regex,
    text_replace_regex,
)
from firepanda.kernel.regex.method import (
    METHOD_CONTAINS,
    METHOD_REPLACE,
    program_for,
)
from firepanda.kernel.regex.program import Program
from firepanda.kernel.regex.replace import parse_rewrite
from firepanda.kernel.scalar import text_matches_regex_scalar


def compiled(pattern: String) raises -> Program:
    """Compiles a pattern for the engine, or fails the test.

    Args:
        pattern: The pattern.

    Returns:
        The program.

    Raises:
        Error: If it did not compile, which in this file is a mistake in the
            test rather than an answer.
    """
    var program = program_for(METHOD_CONTAINS, pattern)
    if not program.ok:
        raise Error(
            String("pattern ", pattern, " did not compile: ", program.problem)
        )
    return program^


def rows() -> List[String]:
    """The text every test below runs its patterns over.

    Ordered so that a row which leaves state behind is followed by one the state
    would change the answer of: the long row of letters comes before the empty
    one, and the row that matches at its last character comes before a row that
    matches at its first.

    Returns:
        The rows, before the nulls are punched into them.
    """
    var out = List[String]()
    out.append(String("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"))
    out.append(String(""))
    out.append(String("ignored"))
    out.append(String("xyzzy"))
    out.append(String("ignored"))
    out.append(String("b"))
    out.append(String("ab"))
    out.append(String("ba"))
    out.append(String("héllo ΑΒΓ"))
    out.append(String("a\nb"))
    return out^


def sample() -> StringArray:
    """The column, with a null in it either side of a row that matches.

    Returns:
        The column.
    """
    var given = rows()
    var builder = StringBuilder(capacity=len(given))
    for i in range(len(given)):
        if i == 2 or i == 4:
            builder.append_null()
        else:
            builder.append(given[i].as_bytes())
    return builder^.finish()


def grown() raises -> StringArray:
    """The sample repeated until it is taller than one morsel.

    The repeat is a doubling and then one more copy on the end, so the answer
    is the sample tiled a whole number of times and row `i` of it is row
    `i % 10` of the sample. Both morsel tests lean on that: it is what lets an
    answer for a tall column be checked against an answer for a short one
    without running anything twice.

    Returns:
        The column.

    Raises:
        Error: If a concatenation cannot allocate.
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
    return concat_strings(tail)


def agrees(
    got: Array[DType.bool], want: Array[DType.bool], label: String
) raises:
    """Asserts that the kernel's answer matches the twin's, row by row.

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


def check(col: StringArray, pattern: String) raises:
    """Runs the kernel and its twin over a column and asserts they agree.

    Args:
        col: The column.
        pattern: The pattern.

    Raises:
        AssertionError: If they disagree.
        Error: If the pattern did not compile.
    """
    var program = compiled(pattern)
    agrees(
        text_matches_regex(col, program),
        text_matches_regex_scalar(col, program),
        "matches " + pattern,
    )


def test_a_column_matches_the_twin() raises:
    """The patterns are the shapes that leave a machine in different states:
    one that finds its answer at the first position, one that finds it at the
    last, one that finds nothing after walking every position, and one that
    matches the empty row and so matches everything."""
    var col = sample()
    check(col, "a")
    check(col, "b$")
    check(col, "^a+b$")
    check(col, "q")
    check(col, "")
    check(col, "a|b")
    check(col, "[^ab]")
    check(col, "a.b")


def test_a_row_is_not_told_what_the_row_before_it_matched() raises:
    """The one thing the reused machine could get wrong, asserted on the
    answers rather than on the buffers. `^a` is true of the first row and false
    of the one after it, and a machine that kept a thread from the row before
    would say so."""
    var col = strings_from_list(["aaa", "bbb", "aaa", "", "aaa"])
    var program = compiled("^a")
    var mask = text_matches_regex(col, program)
    assert_true(mask[0])
    assert_false(mask[1])
    assert_true(mask[2])
    assert_false(mask[3])
    assert_true(mask[4])


def test_an_answer_that_was_found_early_does_not_leak_into_the_next_row() raises:
    """The other order. A row that matches at its first position returns while
    the machine still holds threads for everything after it, and the next row
    matches nothing at all."""
    var col = strings_from_list(["abbbbbbbbbb", "zzz", "abbbbbbbbbb", "zzz"])
    var program = compiled("ab*")
    var mask = text_matches_regex(col, program)
    assert_true(mask[0])
    assert_false(mask[1])
    assert_true(mask[2])
    assert_false(mask[3])


def test_a_null_row_stays_null_and_is_not_read() raises:
    """Null in, null out, which is the rule the repair holds up. The row under
    a null is the empty string, which matches most of the patterns above, so a
    kernel that repaired nothing would answer true rather than nothing."""
    var col = sample()
    var mask = text_matches_regex(col, compiled(""))
    assert_false(mask.is_valid(2))
    assert_false(mask.is_valid(4))
    assert_true(mask.is_valid(0))
    assert_true(mask.is_valid(len(mask) - 1))


def test_a_column_either_side_of_the_morsel_split_matches_the_twin() raises:
    """The kernel runs on every core above a row count no short column reaches,
    so the split between morsels has to be walked as well as the rows inside
    one. This is where a kernel that wrote its answer at `i - start`, or
    repaired the validity of the wrong range, or built one machine for the whole
    column and shared it between threads, comes apart.

    Every row is checked and none of them is read from this file. The twin is
    asked for the whole column in one call, the two answers are compared by a
    kernel and reduced by another, and the only thing crossing back here is a
    count."""
    var col = grown()
    assert_true(len(col) > MORSEL_ROWS)

    var program = compiled("^a+b$")
    var mask = text_matches_regex(col, program)
    var want = text_matches_regex_scalar(col, program)
    assert_equal(
        Int(sum_of(not_equal(mask, want)).value), 0, "rows disagreeing"
    )
    # And that the twin was not vacuously right about a column that came out
    # all false or all null, which is what would make the comparison above pass
    # without either side having matched anything.
    assert_true(mask[0])
    assert_false(mask[1])
    assert_false(mask.is_valid(2))


def test_replacing_past_one_morsel_says_what_one_morsel_said() raises:
    """The replaced column is built a payload per morsel and joined at the end,
    so a column short enough to fit one morsel never reaches the join at all.
    This runs the same rows twice, once short and once tiled past the point
    where every core takes a share, and asks whether row `i` of the tall answer
    is still what row `i` of the short one was.

    The sample is what makes it a check rather than a shape. It holds a row
    whose answer is too long to sit inside a view and so goes into a payload, a
    row short enough to sit inside one, an empty row and two nulls, and the two
    assertions under the call say so rather than trusting that they do. A join
    that moved the first morsel's offsets and left the rest where they were
    would answer the first 131072 rows correctly and read whatever happened to
    be in front of it after that."""
    var program = program_for(METHOD_REPLACE, String("a"))
    if not program.ok:
        raise Error(String("the pattern did not compile: ", program.problem))
    var rewrite = parse_rewrite(String("Z"), program.groups)
    assert_true(rewrite.ok, "the replacement was refused")

    var short = text_replace_regex(sample(), program, rewrite)
    assert_true(
        short[0].byte_length() > INLINE_CAPACITY, "a row with a payload"
    )
    assert_true(
        short[3].byte_length() <= INLINE_CAPACITY, "and one without one"
    )

    var col = grown()
    assert_true(len(col) > MORSEL_ROWS)
    var tall = text_replace_regex(col, program, rewrite)
    assert_equal(len(tall), len(col), "the answer is as tall as the column")

    var wrong = 0
    for i in range(len(tall)):
        var want = i % len(short)
        if tall.is_valid(i) != short.is_valid(want):
            wrong += 1
        elif tall.is_valid(i) and tall[i] != short[want]:
            wrong += 1
    assert_equal(wrong, 0, "rows disagreeing with the one morsel answer")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
