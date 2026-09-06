"""Tests for arithmetic and comparison on a series.

Every expected value in here was read off a running pandas 3.0 rather than
worked out by hand, because the interesting part of this feature is not the
arithmetic. It is which rows the arithmetic runs on, and that is decided by the
row labels in a way that has three separate rules in it.

The first rule is that `+` matches rows by label and `==` refuses to. pandas
aligns an arithmetic operation and raises on a comparison between two series
that are not labelled the same, and the flexible `eq` aligns after all. All
three of those are tested here, because a library that aligned everything and a
library that aligned nothing would each pass half of them.

The second is `fill_value`, which fills a row that one side is missing and
leaves a row that both sides are missing alone. That is a rule about the two
reindexed masks rather than about the arithmetic, and it is the one that is
easiest to write as "fill every null" by mistake.

The third is what happens to a label that appears twice, which is a join and not
a union in pandas and is refused here for now. The test says the refusal
happens; the message says why.

The four unary operations are here too, and they need none of the above because
one operand brings its own labels and there is nothing to line it up with. What
they do need is the two places pandas and numpy disagree: `-` on a boolean
column answers the logical not where numpy refuses the expression, and `~` picks
the logical not or the bitwise one by the type.

Two divergences show up in these tests and both are the same argument the
project has already made. An integer column that meets a zero divisor answers a
null and stays an integer, where pandas widens the whole column to float64 to
make room for an infinity. And an aligned result stays an integer with nulls in
it, where pandas widens to float64 to hold a NaN. An Arrow column has somewhere
to record absence and a numpy array does not, so the widening is a workaround
firepanda does not need.
"""

from std.collections import Optional
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array, from_list
from firepanda.array.strings import strings_from_list
from firepanda.array.value import Value
from firepanda.frame.index import Index
from firepanda.frame.series import Series
from firepanda.kernel.binary import BinaryOp


def unnamed() -> Optional[String]:
    """A level name of `None`, spelled once because it is written a lot."""
    return Optional[String]()


def series(name: String, values: List[Int64], at: List[Int64]) raises -> Series:
    """An int64 series under the given row labels."""
    var out = Series(name, from_list[DType.int64](values))
    out.index = Index(AnyArray(from_list[DType.int64](at)), unnamed())
    return out^


def text_labelled(
    name: String, values: List[Int64], at: List[String]
) raises -> Series:
    """An int64 series under text row labels."""
    var out = Series(name, from_list[DType.int64](values))
    out.index = Index(AnyArray(strings_from_list(at)), unnamed())
    return out^


def flag_series(name: String, values: List[Bool]) raises -> Series:
    """A boolean series under a default range index."""
    var out = Array[DType.bool](len(values))
    for i in range(len(values)):
        out.set_valid(i, values[i])
    return Series(name, out^)


def ints(s: Series) raises -> List[Int64]:
    """The values, with a null read as zero."""
    ref view = s.values.as_typed_view[DType.int64]()
    var out = List[Int64](capacity=len(view))
    for i in range(len(view)):
        out.append(view[i])
    return out^


def floats(s: Series) raises -> List[Float64]:
    """The values of a float64 series."""
    ref view = s.values.as_typed_view[DType.float64]()
    var out = List[Float64](capacity=len(view))
    for i in range(len(view)):
        out.append(view[i])
    return out^


def flags(s: Series) raises -> List[Bool]:
    """The values of a boolean series."""
    ref view = s.values.as_typed_view[DType.bool]()
    var out = List[Bool](capacity=len(view))
    for i in range(len(view)):
        out.append(view[i])
    return out^


def label_at(s: Series, i: Int) raises -> Int:
    """The row label in position `i`, as an integer."""
    return Int(s.index.materialize().as_typed[DType.int64]()[i])


def test_two_series_on_the_same_labels_add_row_by_row() raises:
    """The short circuit. Two indexes that hold the same labels are not unioned
    and not looked up in, so this is the path that runs on two columns of one
    frame and it has to be the cheap one."""
    var a = series("v", [1, 2, 3], [10, 11, 12])
    var b = series("v", [10, 20, 30], [10, 11, 12])
    var got = a + b
    assert_equal(len(got), 3, "three rows")
    assert_equal(ints(got)[0], Int64(11), "row 0")
    assert_equal(ints(got)[2], Int64(33), "row 2")
    assert_equal(label_at(got, 0), 10, "label 0")
    assert_equal(label_at(got, 2), 12, "label 2")


def test_two_series_on_different_labels_add_on_the_union() raises:
    """`Series([1, 2, 3], index=[1, 2, 3]) + Series([10, 20, 30],
    index=[2, 3, 4])` is `{1: nan, 2: 12, 3: 23, 4: nan}` in pandas. The two
    labels only one side has come back missing rather than being dropped, which
    is what makes this an outer join and not an inner one."""
    var a = series("v", [1, 2, 3], [1, 2, 3])
    var b = series("v", [10, 20, 30], [2, 3, 4])
    var got = a + b
    assert_equal(len(got), 4, "four rows")
    assert_true(not got.is_valid(0), "label 1 is on the left only")
    assert_equal(ints(got)[1], Int64(12), "label 2")
    assert_equal(ints(got)[2], Int64(23), "label 3")
    assert_true(not got.is_valid(3), "label 4 is on the right only")
    assert_equal(label_at(got, 3), 4, "the union is in label order")


def test_an_aligned_integer_result_stays_an_integer() raises:
    """In pandas this comes back float64, because a numpy int64 array has
    nowhere to put the NaN the two absent rows need. An Arrow column has a
    validity bitmap and does not need to widen, so this is a registered
    divergence and not an oversight."""
    var a = series("v", [1, 2, 3], [1, 2, 3])
    var b = series("v", [10, 20, 30], [2, 3, 4])
    assert_equal((a + b).dtype(), DType.int64, "still int64")


def test_a_fill_value_fills_the_side_that_is_missing() raises:
    """`a.add(b, fill_value=0)` on the same two series is
    `{1: 1, 2: 12, 3: 23, 4: 30}`, so the fill happens on the side that has no
    row rather than on the result."""
    var a = series("v", [1, 2, 3], [1, 2, 3])
    var b = series("v", [10, 20, 30], [2, 3, 4])
    var got = a.add(b, Value(Int64(0)))
    assert_equal(ints(got)[0], Int64(1), "label 1, right side filled")
    assert_equal(ints(got)[1], Int64(12), "label 2")
    assert_equal(ints(got)[3], Int64(30), "label 4, left side filled")
    assert_equal(got.null_count(), 0, "nothing missing")


def test_a_fill_value_leaves_a_row_both_sides_are_missing_from() raises:
    """The rule that is easy to get wrong by filling every null. Two columns
    that are both null at a label answer null there whatever the fill value is,
    which pandas does and which a straight coalesce on both sides would not."""
    var a = series("v", [1, 0], [1, 2])
    a.values.as_typed_view[DType.int64]().set_null(1)
    var b = series("v", [0, 0], [1, 2])
    b.values.as_typed_view[DType.int64]().set_null(0)
    b.values.as_typed_view[DType.int64]().set_null(1)

    var got = a.add(b, Value(Int64(0)))
    assert_equal(ints(got)[0], Int64(1), "only the right side was missing")
    assert_true(not got.is_valid(1), "both sides were missing")


def test_the_result_keeps_a_name_both_operands_agree_on() raises:
    var a = series("v", [1], [1])
    var b = series("v", [2], [1])
    assert_equal((a + b).name, "v", "the shared name")


def test_the_result_drops_a_name_the_operands_disagree_on() raises:
    """The name goes away, on pandas' reasoning that a column called price plus
    a column called tax is neither of those things."""
    var a = series("price", [1], [1])
    var b = series("tax", [2], [1])
    assert_equal((a + b).name, "", "no name")


def test_the_reflected_forms_turn_the_operands_round() raises:
    """`a.rsub(b)` is `b - a` and not `a - b`, on the labels `a` brings. pandas
    gives `{1: nan, 2: 8, 3: 17, 4: nan}` for the two series below."""
    var a = series("v", [1, 2, 3], [1, 2, 3])
    var b = series("v", [10, 20, 30], [2, 3, 4])
    var got = a.rsub(b)
    assert_equal(ints(got)[1], Int64(8), "label 2")
    assert_equal(ints(got)[2], Int64(17), "label 3")
    assert_true(not got.is_valid(0), "label 1")


def test_a_reordered_index_still_matches_by_label() raises:
    """The test that would pass on a positional implementation only by
    accident. pandas gives `{a: 21, b: 12}` for these two, so the labels drive
    the pairing and the order they were written in does not."""
    var a = text_labelled("v", [1, 2], ["a", "b"])
    var b = text_labelled("v", [10, 20], ["b", "a"])
    var got = a + b
    assert_equal(len(got), 2, "two rows")
    assert_equal(ints(got)[0], Int64(21), "label a")
    assert_equal(ints(got)[1], Int64(12), "label b")


def test_the_seven_operators_against_a_constant() raises:
    """Every value measured on `Series([7, -7, 10])`. The two worth looking at
    are the floor division and the remainder, which follow Python's sign rule
    and not C's, so minus seven floor divided by three is minus three."""
    var s = series("v", [7, -7, 10], [1, 2, 3])
    var three = Value(Int64(3))

    assert_equal(ints(s + three)[1], Int64(-4), "add")
    assert_equal(ints(s - three)[1], Int64(-10), "subtract")
    assert_equal(ints(s * three)[1], Int64(-21), "multiply")
    assert_equal(floats(s / Value(Int64(4)))[1], Float64(-1.75), "divide")
    assert_equal(ints(s // three)[1], Int64(-3), "floor divide")
    assert_equal(ints(s % three)[1], Int64(2), "remainder")
    assert_equal(ints(s ** Value(Int64(2)))[1], Int64(49), "power")


def test_a_constant_on_the_left_is_not_the_same_as_one_on_the_right() raises:
    """`3 - s` is `[-4, 10, -7]` where `s - 3` is `[4, -10, 7]`, and
    `100 // s` is `[14, -15, 10]`. The flipped forms have to reach the loop as
    flipped rather than being turned round by the caller, since none of the
    three can be."""
    var s = series("v", [7, -7, 10], [1, 2, 3])

    assert_equal(ints(s.__rsub__(Value(Int64(3))))[0], Int64(-4), "3 - s")
    assert_equal(ints(s.__rsub__(Value(Int64(3))))[1], Int64(10), "row 1")
    assert_equal(
        ints(s.__rfloordiv__(Value(Int64(100))))[1], Int64(-15), "100 // s"
    )
    assert_equal(ints(s.__rmod__(Value(Int64(100))))[1], Int64(-5), "100 % s")
    assert_equal(
        floats(s.__rtruediv__(Value(Int64(100))))[2], Float64(10.0), "100 / s"
    )


def test_a_constant_base_raised_to_every_row() raises:
    """`2 ** Series([1, 2, 3])` is `[2, 4, 8]`."""
    var s = series("v", [1, 2, 3], [1, 2, 3])
    var got = s.__rpow__(Value(Int64(2)))
    assert_equal(ints(got)[0], Int64(2), "row 0")
    assert_equal(ints(got)[2], Int64(8), "row 2")


def test_the_six_comparisons_against_a_constant() raises:
    """Measured on `Series([1, 2, 3]) <op> 2`."""
    var s = series("v", [1, 2, 3], [1, 2, 3])
    var two = Value(Int64(2))

    assert_equal(flags(s == two)[1], True, "equal")
    assert_equal(flags(s != two)[1], False, "not equal")
    assert_equal(flags(s < two)[0], True, "less")
    assert_equal(flags(s <= two)[1], True, "less or equal")
    assert_equal(flags(s > two)[2], True, "greater")
    assert_equal(flags(s >= two)[1], True, "greater or equal")
    assert_equal((s == two).dtype(), DType.bool, "a comparison answers bool")


def test_arithmetic_against_a_constant_keeps_the_labels() raises:
    """Nothing aligns against a constant, so the labels come through as they
    were rather than being replaced by a range."""
    var s = series("v", [1, 2], [40, 50])
    var got = s + Value(Int64(1))
    assert_equal(label_at(got, 0), 40, "label 0")
    assert_equal(label_at(got, 1), 50, "label 1")
    assert_equal(got.name, "v", "and the name")


def test_comparing_two_series_needs_the_same_labels() raises:
    """`ValueError: Can only compare identically-labeled Series objects` is what
    pandas raises here, and it is the one place where the comparison operators
    and the arithmetic operators disagree about what to do."""
    var a = series("v", [1, 2, 3], [1, 2, 3])
    var b = series("v", [1, 2, 3], [2, 3, 4])
    with assert_raises(contains="identically-labeled"):
        _ = a == b


def test_comparing_two_series_of_different_lengths_is_refused() raises:
    """Two different lengths cannot be identically labelled, so this is the
    same refusal reaching the same caller by a shorter route."""
    var a = series("v", [1, 2, 3], [1, 2, 3])
    var b = series("v", [1, 2], [1, 2])
    with assert_raises(contains="identically-labeled"):
        _ = a < b


def test_comparing_two_series_on_the_same_labels_is_row_by_row() raises:
    var a = series("v", [1, 5, 3], [1, 2, 3])
    var b = series("v", [1, 2, 9], [1, 2, 3])
    var got = a == b
    assert_equal(flags(got)[0], True, "row 0")
    assert_equal(flags(got)[1], False, "row 1")
    assert_equal(flags(got)[2], False, "row 2")


def test_the_flexible_comparison_aligns_where_the_operator_refuses() raises:
    """`a.eq(b)` on two indexes that differ answers a row per label in the
    union instead of raising, which is the whole reason both spellings exist."""
    var a = series("v", [1, 2, 3], [1, 2, 3])
    var b = series("v", [1, 2, 3], [2, 3, 4])
    var got = a.eq(b)
    assert_equal(len(got), 4, "four rows")
    assert_equal(got.dtype(), DType.bool, "a comparison answers bool")
    assert_equal(flags(got)[1], False, "label 2 is 2 against 1")


def test_a_duplicated_label_is_refused_when_the_labels_differ() raises:
    """A repeated label is paired with every copy of it on the other side in
    pandas, so a series labelled `a a b` plus one labelled `a b` is three rows
    and two `a` labels on both sides would be four. That is a join and not the
    union this does, so it is turned away rather than answered wrongly."""
    var a = text_labelled("v", [1, 2, 3], ["a", "a", "b"])
    var b = text_labelled("v", [10, 20], ["a", "b"])
    with assert_raises(contains="duplicated row label"):
        _ = a + b


def test_two_equal_indexes_with_duplicates_still_align() raises:
    """The short circuit does not care whether a label repeats, because two
    equal indexes pair row zero with row zero and there is nothing to look up.
    pandas answers the same thing for the same reason."""
    var a = text_labelled("v", [1, 2], ["a", "a"])
    var b = text_labelled("v", [5, 6], ["a", "a"])
    var got = a + b
    assert_equal(len(got), 2, "two rows")
    assert_equal(ints(got)[0], Int64(6), "row 0")
    assert_equal(ints(got)[1], Int64(8), "row 1")


def test_an_integer_row_divided_by_zero_is_missing() raises:
    """The registered divergence. pandas widens the whole column to float64 and
    writes an infinity, and firepanda answers a null and stays int64, which is
    the only one of pandas' four answers to this that is a function of the
    operand types alone."""
    var s = series("v", [7, 8], [1, 2])
    var got = s // Value(Int64(0))
    assert_equal(got.dtype(), DType.int64, "still int64")
    assert_equal(got.null_count(), 2, "both rows missing")


def test_dividing_answers_float64_whatever_went_in() raises:
    var s = series("v", [7, 8], [1, 2])
    var got = s / Value(Int64(2))
    assert_equal(got.dtype(), DType.float64, "float64")
    assert_equal(floats(got)[0], Float64(3.5), "row 0")


def test_a_text_column_has_no_arithmetic() raises:
    """In pandas this is a `TypeError` naming both sides, and the refusal here
    names both sides too. It comes out of the promotion rather than out of the
    loop, so nothing is attempted before it is turned down."""
    var s = Series("v", strings_from_list(["a", "b"]))
    with assert_raises(contains="no common type for string and int64"):
        _ = s + Value(Int64(1))


def words(name: String, values: List[String], at: List[Int64]) raises -> Series:
    """A text series under the given row labels."""
    var out = Series(name, strings_from_list(values))
    out.index = Index(AnyArray(from_list[DType.int64](at)), unnamed())
    return out^


def test_comparing_a_text_column_with_a_string() raises:
    """A text column has no arithmetic and it does have an ordering, so `<`
    against a string is a real question with a real answer. pandas compares
    lexicographically and answers a boolean column, and this is four of the
    conformance failures #238 counted."""
    var s = words("t", ["a", "b", "c"], [1, 2, 3])
    var below = s.binary(Value(String("b")), BinaryOp.LT)
    assert_equal(flags(below)[0], True, "a is below b")
    assert_equal(flags(below)[1], False, "b is not below itself")
    assert_equal(flags(below)[2], False, "c is above b")


def test_the_six_comparisons_against_a_string() raises:
    """All six, because the ordering is the thing being tested rather than the
    dispatch, and an ordering that got one of them backwards would still pass a
    test of the other five."""
    var s = words("t", ["a", "b", "c"], [1, 2, 3])
    var b = Value(String("b"))
    assert_equal(flags(s.binary(b, BinaryOp.EQ))[1], True, "eq")
    assert_equal(flags(s.binary(b, BinaryOp.NE))[1], False, "ne")
    assert_equal(flags(s.binary(b, BinaryOp.LE))[1], True, "le")
    assert_equal(flags(s.binary(b, BinaryOp.GT))[2], True, "gt")
    assert_equal(flags(s.binary(b, BinaryOp.GE))[1], True, "ge")


def test_a_string_on_the_left_turns_the_comparison_round() raises:
    """`"b" > s` is `s < "b"` and not `s > "b"`, which is the whole reason the
    constant carries a side rather than being assumed to be on the right."""
    var s = words("t", ["a", "b", "c"], [1, 2, 3])
    var b = Value(String("b"))
    var got = s.binary(b, BinaryOp.GT, value_on_left=True)
    assert_equal(flags(got)[0], True, "b is above a")
    assert_equal(flags(got)[2], False, "b is not above c")


def test_comparing_two_text_columns_is_row_by_row() raises:
    """The column form of the same thing, which goes down a different loop."""
    var s = words("t", ["a", "b", "c"], [1, 2, 3])
    var t = words("t", ["a", "x", "b"], [1, 2, 3])
    assert_equal(flags(s.compare(t, BinaryOp.EQ))[0], True, "row 0")
    assert_equal(flags(s.compare(t, BinaryOp.LT))[1], True, "row 1")
    assert_equal(flags(s.compare(t, BinaryOp.LT))[2], False, "row 2")


def test_adding_two_text_columns_is_refused_for_now() raises:
    """Concatenation is the one arithmetic operation a text column does have in
    pandas, where `["a", "b"] + ["x", "y"]` is `["ax", "by"]`, and the kernel
    does not have it. The alignment above it is ready, so this is a loop in
    `firepanda/kernel/text.mojo` and not a design question, and the test says
    which of the two it is rather than leaving the gap unrecorded."""
    var s = words("t", ["a", "b"], [1, 2])
    var t = words("t", ["x", "y"], [1, 2])
    with assert_raises(contains="+ is not defined on string"):
        _ = s + t


def test_dividing_two_series_answers_a_pair() raises:
    """`divmod` is on a series in pandas and not on a frame, and it is two
    passes there as well, so this is two calls with one name on them rather
    than a kernel that writes two columns."""
    var a = series("v", [7, -7], [1, 2])
    var pair = a.__divmod__(Value(Int64(3)))
    assert_equal(ints(pair[0])[0], Int64(2), "7 // 3")
    assert_equal(ints(pair[0])[1], Int64(-3), "-7 // 3, Python's sign rule")
    assert_equal(ints(pair[1])[0], Int64(1), "7 % 3")
    assert_equal(ints(pair[1])[1], Int64(2), "-7 % 3")


def test_dividing_a_constant_by_a_series_answers_a_pair() raises:
    """The reflected form, measured the same way."""
    var a = series("v", [7, -7], [1, 2])
    var pair = a.__rdivmod__(Value(Int64(3)))
    assert_equal(ints(pair[0])[0], Int64(0), "3 // 7")
    assert_equal(ints(pair[0])[1], Int64(-1), "3 // -7")
    assert_equal(ints(pair[1])[0], Int64(3), "3 % 7")
    assert_equal(ints(pair[1])[1], Int64(-4), "3 % -7")


def test_dividing_two_series_by_each_other_answers_a_pair() raises:
    """The column form, which aligns on the labels the way every other pair of
    columns does."""
    var a = series("v", [7, -7], [1, 2])
    var b = series("v", [3, 3], [1, 2])
    var pair = a.__divmod__(b)
    assert_equal(ints(pair[0])[1], Int64(-3), "-7 // 3")
    assert_equal(ints(pair[1])[1], Int64(2), "-7 % 3")


def test_a_cast_keeps_the_row_labels() raises:
    """Every operation that answers one row per input row keeps the labels, and
    the cast is the one that used to drop them. It matters here rather than in
    its own file because a cast that reset the labels to a range would silently
    change what the next arithmetic aligned against."""
    var s = series("v", [1, 2], [40, 50])
    var got = s.cast(DType.float64)
    assert_equal(label_at(got, 0), 40, "label 0")
    assert_equal(label_at(got, 1), 50, "label 1")


def test_a_fill_keeps_the_row_labels() raises:
    var s = series("v", [1, 0], [40, 50])
    s.values.as_typed_view[DType.int64]().set_null(1)
    var got = s.fill_forward()
    assert_equal(label_at(got, 1), 50, "label 1")
    assert_equal(ints(got)[1], Int64(1), "and the value came forward")


def test_the_four_unary_operations_on_an_integer_column() raises:
    """Measured on `Series([7, -7, 10], index=[40, 50, 60], name='v')`. Unary
    plus is the column back unchanged, which is the whole of what pandas does
    with it."""
    var s = series("v", [7, -7, 10], [40, 50, 60])

    assert_equal(ints(-s)[0], Int64(-7), "negate row 0")
    assert_equal(ints(-s)[1], Int64(7), "negate row 1")
    assert_equal(ints(+s)[1], Int64(-7), "unary plus changes nothing")
    assert_equal(ints(s.abs())[1], Int64(7), "absolute")
    assert_equal(ints(~s)[0], Int64(-8), "the bitwise not of seven")


def test_a_unary_operation_keeps_the_labels_and_the_name() raises:
    """One operand, so nothing aligns and nothing is renamed. A negation that
    reset the labels to a range would break the next addition silently."""
    var s = series("v", [7, -7, 10], [40, 50, 60])
    var got = -s
    assert_equal(got.name, "v", "the name")
    assert_equal(label_at(got, 0), 40, "label 0")
    assert_equal(label_at(got, 2), 60, "label 2")


def test_negating_a_bool_column_is_the_logical_not() raises:
    """This is pandas' answer and not numpy's. numpy refuses `-` on a boolean
    array outright and says the boolean negative is not supported, and pandas
    catches that and sends the column to the inversion instead, so
    `-Series([True, False])` is `[False, True]`."""
    var s = flag_series("v", [True, False])
    assert_equal(flags(-s)[0], False, "row 0")
    assert_equal(flags(-s)[1], True, "row 1")
    assert_equal((-s).dtype(), DType.bool, "still bool")


def test_inverting_a_bool_column_negates_a_mask() raises:
    """`~mask` is the reason the boolean branch of the kernel exists. Without
    it the commonest line in any pandas program, `df[~mask]`, has no spelling
    here at all."""
    var s = flag_series("v", [True, False])
    assert_equal(flags(~s)[0], False, "row 0")
    assert_equal(flags(~s)[1], True, "row 1")


def test_taking_the_absolute_value_of_a_bool_column_changes_nothing() raises:
    """`abs(Series([True, False]))` is `[True, False]` in pandas, which
    follows from the values already being zero and one."""
    var s = flag_series("v", [True, False])
    assert_equal(flags(s.abs())[0], True, "row 0")
    assert_equal(flags(s.abs())[1], False, "row 1")


def test_inverting_a_float_column_is_refused() raises:
    """`ufunc 'invert' not supported` is what numpy says, and pandas passes
    that on rather than inventing a meaning for it."""
    var s = series("v", [1, 2], [1, 2]).cast(DType.float64)
    with assert_raises():
        _ = ~s


def test_negating_a_float_zero_sets_the_sign_bit() raises:
    """The test that catches a negation written as a subtraction from zero.
    `-Series([0.0])` has the sign bit set in pandas and `0.0 - 0.0` does not,
    which is a difference between two values that compare equal, so a test that
    only compared numbers would never see it."""
    var s = series("v", [0], [1]).cast(DType.float64)
    var negated = floats(-s)[0]
    assert_equal(negated, Float64(0.0), "it is still zero")
    assert_true(
        Float64(1.0) / negated < Float64(0.0), "and it is the negative zero"
    )


def test_an_integer_raised_to_a_negative_power_is_refused() raises:
    """`Series([1, 2]) ** -1` raises `ValueError: Integers to negative integer
    powers are not allowed.` in pandas, and the refusal comes through the series
    layer rather than being swallowed on the way."""
    var s = series("v", [1, 2], [1, 2])
    with assert_raises(contains="negative integer powers"):
        _ = s ** Value(Int64(-1))


def test_the_named_forms_reach_the_same_loops_as_the_operators() raises:
    """One assertion per named form, so that a wiring mistake in any of the
    twenty is a failure here rather than a surprise in Python.

    `rpow` gets operands of its own because the negative row that makes every
    other reflected form interesting is the one thing `rpow` cannot take: three
    raised to minus seven is refused rather than answered, on integers, by
    numpy and by firepanda alike.
    """
    var a = series("v", [7, -7, 10], [1, 2, 3])
    var b = series("v", [3, 3, 3], [1, 2, 3])
    var up = series("v", [7, 2, 10], [1, 2, 3])

    assert_equal(ints(a.add(b))[0], Int64(10), "add")
    assert_equal(ints(a.radd(b))[0], Int64(10), "radd")
    assert_equal(ints(a.sub(b))[0], Int64(4), "sub")
    assert_equal(ints(a.rsub(b))[0], Int64(-4), "rsub")
    assert_equal(ints(a.mul(b))[0], Int64(21), "mul")
    assert_equal(ints(a.rmul(b))[0], Int64(21), "rmul")
    assert_equal(floats(a.truediv(b))[2], Float64(10.0 / 3.0), "truediv")
    assert_equal(floats(a.rtruediv(b))[2], Float64(0.3), "rtruediv")
    assert_equal(ints(a.floordiv(b))[1], Int64(-3), "floordiv")
    assert_equal(ints(a.rfloordiv(b))[1], Int64(-1), "rfloordiv")
    assert_equal(ints(a.mod(b))[1], Int64(2), "mod")
    assert_equal(ints(a.rmod(b))[1], Int64(-4), "rmod")
    assert_equal(ints(a.pow(b))[0], Int64(343), "pow")
    assert_equal(ints(up.rpow(b))[1], Int64(9), "rpow")
    assert_equal(ints(up.rpow(b))[2], Int64(59049), "rpow, the wide row")

    assert_equal(flags(a.eq(b))[0], False, "eq")
    assert_equal(flags(a.ne(b))[0], True, "ne")
    assert_equal(flags(a.lt(b))[1], True, "lt")
    assert_equal(flags(a.le(b))[1], True, "le")
    assert_equal(flags(a.gt(b))[0], True, "gt")
    assert_equal(flags(a.ge(b))[0], True, "ge")


def test_the_erased_entry_point_takes_the_operation_as_a_value() raises:
    """The form a plan calls, where the operation is a runtime value rather than
    seven methods. Everything above is two lines on top of this."""
    var a = series("v", [1, 2], [1, 2])
    var b = series("v", [10, 20], [1, 2])
    assert_equal(ints(a.binary(b, BinaryOp.ADD))[0], Int64(11), "add")
    assert_equal(
        ints(a.binary(b, BinaryOp.SUB, flip=True))[0], Int64(9), "flipped"
    )
    assert_equal(
        ints(a.binary(Value(Int64(4)), BinaryOp.SUB, value_on_left=True))[0],
        Int64(3),
        "constant on the left",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
