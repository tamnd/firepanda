"""A key tuple packed into one byte string per row.

The claim the packing makes is injectivity: two rows pack to the same bytes
exactly when they hold the same tuple. So these tests are mostly pairs of rows
built to be as close to each other as a tuple can get without being equal, and
the question asked of each pair is whether the bytes came out different.

The pair worth naming is the one a concatenation without lengths gets wrong.
`("ab", "c")` and `("a", "bc")` are different tuples whose letters in order are
the same letters, so a packing that writes the bytes and nothing else answers
that they are equal. The four byte length in front of each string is what stops
that, and the test below is the only reason it is there.

Usage:
    pixi run mojo run -I . tests/test_join_packed.mojo
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.join.packed import pack_keys


def _ints(values: List[Int64]) raises -> AnyArray:
    """Builds an int64 column with no nulls in it.

    Args:
        values: The values.

    Returns:
        The column.
    """
    var out = Array[DType.int64](overwritten=len(values))
    for i in range(len(values)):
        out[i] = values[i]
    return AnyArray(out^)


def _text(values: List[String]) raises -> AnyArray:
    """Builds a text column with no nulls in it.

    Args:
        values: The values.

    Returns:
        The column.
    """
    var built = StringBuilder(capacity=len(values))
    for i in range(len(values)):
        built.append(values[i].as_bytes())
    return AnyArray(built^.finish())


def _packed(columns: List[AnyArray], rows: Int) raises -> StringArray:
    """Packs every column of a frame, in order.

    Args:
        columns: The columns, all of them keys.
        rows: How many rows.

    Returns:
        The packed column, as text.
    """
    var keys = List[Int]()
    for k in range(len(columns)):
        keys.append(k)
    return pack_keys(columns, keys, rows).into_strings()


def test_two_rows_with_the_same_tuple_pack_to_the_same_bytes() raises:
    var columns: List[AnyArray] = [_ints([7, 7]), _ints([3, 3])]
    var packed = _packed(columns, 2)
    assert_true(packed.element_equals(0, 1), "the same tuple packs the same")


def test_a_tuple_that_differs_in_its_second_key_packs_differently() raises:
    var columns: List[AnyArray] = [_ints([7, 7]), _ints([3, 4])]
    var packed = _packed(columns, 2)
    assert_false(packed.element_equals(0, 1), "the tuples are not the same")


def test_two_keys_swapped_between_the_columns_pack_differently() raises:
    # (7, 3) against (3, 7). A packing that added the keys, or that sorted them,
    # or that hashed them into one number without an order would answer that
    # these agree.
    var columns: List[AnyArray] = [_ints([7, 3]), _ints([3, 7])]
    var packed = _packed(columns, 2)
    assert_false(packed.element_equals(0, 1), "a tuple is ordered")


def test_a_string_carries_its_length_so_the_split_between_keys_is_readable() raises:
    # ("ab", "c") and ("a", "bc"). The letters in order are the same letters.
    var columns: List[AnyArray] = [_text(["ab", "a"]), _text(["c", "bc"])]
    var packed = _packed(columns, 2)
    assert_false(
        packed.element_equals(0, 1),
        "the length in front of each string is what tells the two apart",
    )


def test_a_string_beside_a_number_packs_and_still_separates() raises:
    var columns: List[AnyArray] = [_ints([1, 1]), _text(["forest", "forestry"])]
    var packed = _packed(columns, 2)
    assert_false(packed.element_equals(0, 1), "a prefix is not the string")


def test_a_null_in_any_key_packs_the_whole_row_to_null() raises:
    var left = _ints([1, 2, 3])
    var right = _ints([4, 5, 6])
    right.data.validity.set(1, False)
    var pair: List[AnyArray] = [left^, right^]
    var packed = _packed(pair, 3)
    assert_true(packed.is_valid(0), "a row with no null keys is present")
    assert_false(
        packed.is_valid(1),
        "and a row whose second key is null cannot pair, so it packs to null",
    )
    assert_true(packed.is_valid(2), "the rows after it are untouched")


def test_one_key_packs_to_the_bytes_of_that_key_alone() raises:
    var columns: List[AnyArray] = [_ints([5, 5, 6])]
    var packed = _packed(columns, 3)
    assert_equal(packed.byte_length(0), 8, "eight bytes for an int64")
    assert_true(packed.element_equals(0, 1), "equal values, equal bytes")
    assert_false(packed.element_equals(0, 2), "different values, different")


def test_packing_no_rows_gives_a_column_of_no_rows() raises:
    var columns: List[AnyArray] = [_ints(List[Int64]()), _ints(List[Int64]())]
    var packed = _packed(columns, 0)
    assert_equal(len(packed), 0, "nothing in, nothing out")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
