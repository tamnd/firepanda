"""Tests for comparing text columns.

The numeric comparison kernels are checked one dtype at a time against loops that
do the same thing more slowly. Text has no dtypes to enumerate and the interesting
cases are not about width, they are about where in the bytes the answer is
decided. There are three places, and a comparison that only got two of them right
would still pass anything written casually.

The first is inside the view. An element of twelve bytes or fewer is held whole
in its sixteen byte view, zero padded, and equality on two of those is meant to be
a handful of register compares with nothing loaded. The pairs here that are both
short exercise that path, and the ones that are short and unequal in the last byte
are what catches a comparison that stopped at the four byte prefix.

The second is the prefix of a long element. Two long strings that differ in their
first four bytes are settled without either payload being touched, so a pair that
agrees for eleven bytes and differs at the twelfth is the case that has to leave
the view, and it appears in both the equality and the ordering tests.

The third is length. "am" against "amsterdam" is equal on every byte they share
and the shorter one still sorts first, which is the case a comparison written as a
loop over the shorter length gets wrong in the direction that looks right.

Nulls are the same three valued logic as everywhere else. A comparison touching a
null is null, not false, and both sides of the pair are checked because the
kernel reaches the two through different accessors.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import (
    StringArray,
    StringBuilder,
    strings_from_list,
)
from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.kernel.binary import BinaryOp, binary_any, binary_value_any
from firepanda.kernel.compare import (
    CMP_EQ,
    CMP_GE,
    CMP_GT,
    CMP_LE,
    CMP_LT,
    CMP_NE,
)
from firepanda.kernel.scalar import (
    compare_text_const_scalar,
    compare_text_scalar,
)
from firepanda.kernel.text import compare_text, compare_text_const
from firepanda.testing.rng import Rng


def text(values: List[String]) -> StringArray:
    """Builds a string column from a list."""
    return strings_from_list(values)


def with_nulls(values: List[String], present: List[Bool]) -> StringArray:
    """Builds a string column with nulls where asked."""
    var builder = StringBuilder(capacity=len(values))
    for i in range(len(values)):
        if present[i]:
            builder.append(values[i].as_bytes())
        else:
            builder.append_null()
    return builder^.finish()


def _word(mut rng: Rng) -> String:
    """Builds a random word over a four letter alphabet.

    The alphabet is small so that two draws collide often and the equality path
    is exercised on both answers. The length runs from nothing to twenty, which
    puts elements on both sides of the twelve byte boundary and includes the
    empty string.

    Args:
        rng: The source of randomness.

    Returns:
        The word.
    """
    var letters: List[String] = ["a", "b", "c", "d"]
    var count = rng.next_below(21)
    var out = String("")
    for _ in range(count):
        out += letters[rng.next_below(4)]
    return out^


def same_answer(
    got: Array[DType.bool], want: Array[DType.bool], what: String
) raises:
    """Asserts that a kernel and its twin agree, values and validity both."""
    assert_equal(len(got), len(want), what + ": length")
    for i in range(len(got)):
        assert_equal(
            got.is_valid(i),
            want.is_valid(i),
            what + ": validity at " + String(i),
        )
        if got.is_valid(i):
            assert_equal(
                Bool(got[i]), Bool(want[i]), what + ": value at " + String(i)
            )


def test_two_short_columns_compare_for_equality() raises:
    var a = text(["ab", "cd", "ef"])
    var b = text(["ab", "ce", "ef"])
    var got = compare_text[CMP_EQ](a, b)
    assert_true(got[0], "ab equals ab")
    assert_true(not got[1], "cd does not equal ce")
    assert_true(got[2], "ef equals ef")


def test_equality_reads_past_the_prefix() raises:
    """Two long strings agreeing for eleven bytes and differing at the twelfth.
    A comparison that stopped at the four byte prefix would call these equal."""
    var a = text(["abcdefghijkl mnop", "abcdefghijkl qrst"])
    var b = text(["abcdefghijkl mnop", "abcdefghijkl qrsu"])
    var got = compare_text[CMP_EQ](a, b)
    assert_true(got[0], "identical past the prefix")
    assert_true(not got[1], "differ in the last byte of a long element")


def test_equality_is_not_fooled_by_a_prefix() raises:
    var a = text(["am", "amsterdam"])
    var b = text(["amsterdam", "am"])
    var got = compare_text[CMP_EQ](a, b)
    assert_true(not got[0], "am is not amsterdam")
    assert_true(not got[1], "amsterdam is not am")


def test_inequality_is_the_negation() raises:
    var a = text(["ab", "cd"])
    var b = text(["ab", "ce"])
    var got = compare_text[CMP_NE](a, b)
    assert_true(not got[0], "equal means not unequal")
    assert_true(got[1], "unequal")


def test_ordering_puts_the_shorter_prefix_first() raises:
    var a = text(["am", "amsterdam"])
    var b = text(["amsterdam", "am"])
    var less = compare_text[CMP_LT](a, b)
    assert_true(less[0], "am sorts before amsterdam")
    assert_true(not less[1], "amsterdam does not sort before am")


def test_ordering_compares_bytes_unsigned() raises:
    """Uppercase sorts before lowercase, which is what a byte comparison does and
    what `ORDER BY` with no collation named on it does."""
    var a = text(["Zurich"])
    var b = text(["ambridge"])
    var got = compare_text[CMP_LT](a, b)
    assert_true(got[0], "Z is 90 and a is 97")


def test_ordering_reads_past_the_prefix() raises:
    var a = text(["abcdefghijkl mnop"])
    var b = text(["abcdefghijkl mnoq"])
    assert_true(compare_text[CMP_LT](a, b)[0], "p before q")
    assert_true(not compare_text[CMP_GT](a, b)[0], "p is not after q")


def test_the_four_orderings_agree_on_a_tie() raises:
    var a = text(["same"])
    var b = text(["same"])
    assert_true(not compare_text[CMP_LT](a, b)[0], "not less")
    assert_true(compare_text[CMP_LE](a, b)[0], "less or equal")
    assert_true(not compare_text[CMP_GT](a, b)[0], "not greater")
    assert_true(compare_text[CMP_GE](a, b)[0], "greater or equal")


def test_a_null_on_either_side_makes_the_answer_null() raises:
    var a = with_nulls(["ab", "cd", "ef"], [True, False, True])
    var b = with_nulls(["ab", "cd", "ef"], [True, True, False])
    var got = compare_text[CMP_EQ](a, b)
    assert_true(got.is_valid(0), "neither side missing")
    assert_true(not got.is_valid(1), "left missing")
    assert_true(not got.is_valid(2), "right missing")


def test_a_null_is_null_under_inequality_too() raises:
    """The equality loop answers false for a null and the negation would turn
    that into true, so this is the one that catches a missing repair pass."""
    var a = with_nulls(["ab", "cd"], [True, False])
    var b = text(["ab", "cd"])
    var got = compare_text[CMP_NE](a, b)
    assert_true(got.is_valid(0), "present")
    assert_true(not got.is_valid(1), "missing stays missing, not true")


def test_a_short_constant_compares_against_every_row() raises:
    var a = text(["ab", "cd", "ab"])
    var word = String("ab")
    var got = compare_text_const[CMP_EQ](a, word.as_bytes())
    assert_true(got[0], "first")
    assert_true(not got[1], "second")
    assert_true(got[2], "third")


def test_a_long_constant_compares_against_every_row() raises:
    """A constant of more than twelve bytes cannot be turned into a view, so this
    takes the other branch of the constant kernel."""
    var a = text(["abcdefghijkl mnop", "abcdefghijkl mnoq", "short"])
    var word = String("abcdefghijkl mnop")
    var got = compare_text_const[CMP_EQ](a, word.as_bytes())
    assert_true(got[0], "identical")
    assert_true(not got[1], "differs in the last byte")
    assert_true(not got[2], "different length")


def test_a_constant_orders_against_a_column() raises:
    var a = text(["apple", "banana", "cherry"])
    var word = String("banana")
    var got = compare_text_const[CMP_LT](a, word.as_bytes())
    assert_true(got[0], "apple before banana")
    assert_true(not got[1], "banana is not before itself")
    assert_true(not got[2], "cherry is after banana")


def test_a_constant_against_a_null_row_is_null() raises:
    var a = with_nulls(["ab", "cd"], [True, False])
    var word = String("ab")
    var got = compare_text_const[CMP_EQ](a, word.as_bytes())
    assert_true(got.is_valid(0), "present")
    assert_true(not got.is_valid(1), "missing")


def test_an_empty_constant_is_not_a_null() raises:
    """An empty string and a null are both zero bytes and both hold a view of
    length zero, so the only thing telling them apart is the validity bitmap."""
    var a = with_nulls(["", "ab"], [True, True])
    var b = with_nulls(["", "ab"], [False, True])
    var word = String("")
    var got = compare_text_const[CMP_EQ](a, word.as_bytes())
    assert_true(got.is_valid(0), "the empty string is a value")
    assert_true(got[0], "and it equals itself")
    var pair = compare_text[CMP_EQ](a, b)
    assert_true(not pair.is_valid(0), "a null on the right is still a null")


def test_the_erased_path_answers_bool() raises:
    var got = binary_any(
        AnyArray(text(["ab", "cd"])), AnyArray(text(["ab", "ce"])), BinaryOp.LT
    )
    assert_true(got.type == LogicalType.BOOL, "result type")
    ref values = got.as_typed_view[DType.bool]()
    assert_true(not values[0], "ab is not before ab")
    assert_true(values[1], "cd is before ce")


def test_a_text_constant_goes_through_the_erased_path() raises:
    var got = binary_value_any(
        AnyArray(text(["ab", "cd"])), Value(String("ab")), BinaryOp.EQ
    )
    assert_true(got.type == LogicalType.BOOL, "result type")
    ref values = got.as_typed_view[DType.bool]()
    assert_true(values[0], "first")
    assert_true(not values[1], "second")


def test_a_text_constant_on_the_left_is_turned_round() raises:
    """`"b" < x` is `x > "b"`, and the mirroring happens above the kernel."""
    var got = binary_value_any(
        AnyArray(text(["a", "c"])),
        Value(String("b")),
        BinaryOp.LT,
        value_on_left=True,
    )
    ref values = got.as_typed_view[DType.bool]()
    assert_true(not values[0], "b is not less than a")
    assert_true(values[1], "b is less than c")


def test_a_null_text_constant_makes_every_row_null() raises:
    var got = binary_value_any(
        AnyArray(text(["ab", "cd"])),
        Value(null=LogicalType.STRING),
        BinaryOp.EQ,
    )
    assert_true(got.type == LogicalType.BOOL, "result type")
    assert_equal(got.null_count(), 2, "every row")


def test_the_kernels_agree_with_their_twins_on_random_data() raises:
    """Random pairs drawn from an alphabet small enough that ties happen often,
    at lengths that straddle the twelve byte boundary in both directions."""
    var rng = Rng(20260902)
    var rows = 400

    var left = StringBuilder(capacity=rows)
    var right = StringBuilder(capacity=rows)
    for _ in range(rows):
        if rng.next_below(10) == 0:
            left.append_null()
        else:
            var word = _word(rng)
            left.append(word.as_bytes())
        if rng.next_below(10) == 0:
            right.append_null()
        else:
            var word = _word(rng)
            right.append(word.as_bytes())
    var a = left^.finish()
    var b = right^.finish()

    same_answer(
        compare_text[CMP_EQ](a, b), compare_text_scalar[CMP_EQ](a, b), "eq"
    )
    same_answer(
        compare_text[CMP_NE](a, b), compare_text_scalar[CMP_NE](a, b), "ne"
    )
    same_answer(
        compare_text[CMP_LT](a, b), compare_text_scalar[CMP_LT](a, b), "lt"
    )
    same_answer(
        compare_text[CMP_LE](a, b), compare_text_scalar[CMP_LE](a, b), "le"
    )
    same_answer(
        compare_text[CMP_GT](a, b), compare_text_scalar[CMP_GT](a, b), "gt"
    )
    same_answer(
        compare_text[CMP_GE](a, b), compare_text_scalar[CMP_GE](a, b), "ge"
    )

    var short = String("ab")
    same_answer(
        compare_text_const[CMP_EQ](a, short.as_bytes()),
        compare_text_const_scalar[CMP_EQ](a, short),
        "const eq short",
    )
    same_answer(
        compare_text_const[CMP_LT](a, short.as_bytes()),
        compare_text_const_scalar[CMP_LT](a, short),
        "const lt short",
    )
    var long = String("abababababababab")
    same_answer(
        compare_text_const[CMP_EQ](a, long.as_bytes()),
        compare_text_const_scalar[CMP_EQ](a, long),
        "const eq long",
    )
    same_answer(
        compare_text_const[CMP_GE](a, long.as_bytes()),
        compare_text_const_scalar[CMP_GE](a, long),
        "const ge long",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
