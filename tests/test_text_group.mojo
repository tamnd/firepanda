"""Tests for a text column used as a key, in a group by and in a join.

Grouping on text is the hash table with the key comparison put back, and the
comparison is the thing worth testing. Everywhere else in `firepanda/hash` a
hash match is a proof, because `mix` is a bijection and the key fits in the
hash. Text does not fit, so a match is a candidate, and the tests here are built
out of keys that agree for as long as possible before they differ: on their first
four bytes, which is the prefix stored in the view, and on their first twelve,
which is everything an inline element has.

What cannot be tested from out here is the case the comparison exists for, two
different strings landing on the same 64 bits. Finding a pair costs about four
billion hashes and a test that spent that would be a test nobody runs. The fuzz
harness does not find one either. So that branch is justified by the argument on
`HashTable.build_strings` rather than by a case, and what these tests can and do
check is that the comparison never rejects a pair it should accept, which is the
failure that would actually happen.

The other half of this file is the join, which reaches text through the same
`group_ordinals` the group by does, plus one path of its own: an outer join has
to take each key value from whichever side the row came from, and getting that
wrong puts a null in the column the rows were matched on.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.array.any import AnyArray
from firepanda.array.array import Array, from_list
from firepanda.array.strings import (
    StringArray,
    StringBuilder,
    strings_from_list,
)
from firepanda.frame.frame import DataFrame
from firepanda.frame.groupby import AggSpec
from firepanda.frame.series import Series
from firepanda.hash.factorize import (
    _factorize_strings_parallel,
    _factorize_strings_serial,
    factorize_strings,
)
from firepanda.hash.function import DEFAULT_SEED, hash_bytes
from firepanda.hash.scalar import factorize_strings_linear
from firepanda.join.pairs import JoinKind
from firepanda.kernel.group import AggKind
from firepanda.testing.rng import Rng


def text(values: List[String]) -> StringArray:
    """Builds a string column from a list."""
    return strings_from_list(values)


def with_nulls(values: List[String], present: List[Bool]) -> StringArray:
    """Builds a string column with nulls where asked.

    Args:
        values: The elements. The entry under a false flag is ignored.
        present: One flag per element.

    Returns:
        The column.
    """
    var builder = StringBuilder(capacity=len(values))
    for i in range(len(values)):
        if present[i]:
            builder.append(values[i].as_bytes())
        else:
            builder.append_null()
    return builder^.finish()


def ints(values: List[Scalar[DType.int64]]) -> Array[DType.int64]:
    """Builds an int64 column."""
    return from_list(values)


def keyed(
    var keys: StringArray, values: List[Scalar[DType.int64]]
) raises -> DataFrame:
    """Builds a two column frame of a text key and an int64 value."""
    var series = List[Series]()
    series.append(Series("k", keys^))
    series.append(Series("v", ints(values)))
    return DataFrame.from_series(series^)


def codes_of(col: StringArray) raises -> List[Int]:
    """Runs the fast factorize and returns its ordinals as a list."""
    var codes = factorize_strings(col).into_codes()
    var out = List[Int](capacity=len(codes))
    for i in range(len(codes)):
        out.append(Int(codes[i]))
    return out^


def test_the_ordinals_are_first_seen_order() raises:
    var codes = codes_of(text(["b", "a", "b", "c", "a"]))
    assert_equal(codes[0], 0)
    assert_equal(codes[1], 1)
    assert_equal(codes[2], 0)
    assert_equal(codes[3], 2)
    assert_equal(codes[4], 1)


def test_keys_that_share_a_prefix_are_different_groups() raises:
    """Four bytes is what a view holds, so this is where a prefix-only compare
    would merge two groups."""
    var codes = codes_of(text(["amsterdam", "amersfoort", "amsterdam"]))
    assert_equal(codes[0], 0)
    assert_equal(codes[1], 1)
    assert_equal(codes[2], 0)


def test_keys_that_agree_past_the_inline_limit_are_different_groups() raises:
    """Twelve bytes is everything an inline element has, so these two are only
    distinguishable by their payloads."""
    var codes = codes_of(
        text(
            [
                "shared_prefix_left",
                "shared_prefix_right",
                "shared_prefix_left",
                "shared_prefix_right",
            ]
        )
    )
    assert_equal(codes[0], 0)
    assert_equal(codes[1], 1)
    assert_equal(codes[2], 0)
    assert_equal(codes[3], 1)


def test_an_empty_key_is_a_group_and_is_not_a_null() raises:
    var codes = codes_of(
        with_nulls(["", "x", "", "y"], [True, True, True, False])
    )
    assert_equal(codes[0], 1)
    assert_equal(codes[2], 1)
    assert_equal(codes[3], 0)


def test_the_null_group_is_ordinal_zero() raises:
    var col = with_nulls(["a", "b", "c"], [True, False, True])
    var found = factorize_strings(col)
    assert_equal(found.null_group, 0)
    assert_equal(found.count(), 3)
    var codes = found^.into_codes()
    assert_equal(Int(codes[0]), 1)
    assert_equal(Int(codes[1]), 0)
    assert_equal(Int(codes[2]), 2)


def test_a_column_with_no_nulls_has_no_null_group() raises:
    var found = factorize_strings(text(["a", "b"]))
    assert_equal(found.null_group, -1)
    assert_equal(found.count(), 2)


def test_an_empty_column_factorizes_to_nothing() raises:
    var found = factorize_strings(text([]))
    assert_equal(found.count(), 0)
    assert_equal(len(found^.into_codes()), 0)


def test_the_representative_rows_are_the_first_of_each_group() raises:
    var found = factorize_strings(text(["p", "q", "p", "r"]))
    assert_equal(len(found.firsts), 3)
    assert_equal(found.firsts[0], 0)
    assert_equal(found.firsts[1], 1)
    assert_equal(found.firsts[2], 3)


def test_the_fast_factorize_agrees_with_its_twin_on_random_text() raises:
    """A quarter of the elements share a prefix, because random letters over a
    four letter alphabet still collide too rarely to reach the comparison."""
    var rng = Rng(0x5EED1234)
    var builder = StringBuilder(capacity=600)
    for _ in range(600):
        if rng.next_below(8) == 0:
            builder.append_null()
            continue
        var value = String()
        if rng.next_below(4) == 0:
            value += "sharedprefix"
        for _ in range(Int(rng.next_below(6)) + 1):
            value += chr(97 + Int(rng.next_below(4)))
        builder.append(value.as_bytes())
    var col = builder^.finish()

    var fast = factorize_strings(StringArray(copy=col)).into_codes()
    var want = factorize_strings_linear(col)
    assert_equal(len(fast), len(want))
    for i in range(len(want)):
        assert_equal(Int(fast[i]), Int(want[i]), String("row ", i))


def test_the_table_grows_without_losing_a_group() raises:
    """Past the first resizing checkpoint, so the build rehashes at least once.
    """
    var builder = StringBuilder(capacity=9000)
    for i in range(9000):
        builder.append(String("key_", i).as_bytes())
    var col = builder^.finish()
    var found = factorize_strings(col)
    assert_equal(found.count(), 9000)
    var codes = found^.into_codes()
    for i in range(9000):
        assert_equal(Int(codes[i]), i)


def test_hash_bytes_separates_a_trailing_zero_from_nothing_at_all() raises:
    """The tail packs into a word that starts at zero, so without the length
    folded in these two are the same hash."""
    var short = String("ab")
    var padded = List[UInt8](capacity=3)
    padded.append(UInt8(97))
    padded.append(UInt8(98))
    padded.append(UInt8(0))
    assert_true(hash_bytes(short.as_bytes()) != hash_bytes(Span(padded)))


def test_hash_bytes_is_the_same_for_the_same_bytes() raises:
    var one = String("a longer field than eight bytes")
    var two = String("a longer field than eight bytes")
    assert_equal(hash_bytes(one.as_bytes()), hash_bytes(two.as_bytes()))
    assert_true(
        hash_bytes(one.as_bytes(), DEFAULT_SEED)
        != hash_bytes(one.as_bytes(), DEFAULT_SEED + 1)
    )


def test_a_frame_groups_by_a_text_key() raises:
    var frame = keyed(text(["oslo", "lima", "oslo", "cairo"]), [1, 2, 3, 4])
    var specs = List[AggSpec]()
    specs.append(AggSpec("v", AggKind.SUM))
    var out = frame.group_by(["k"], specs)

    assert_equal(len(out), 3)
    var keys = out.column("k")
    var sums = out.column("v_sum").as_typed[DType.int64]()
    assert_equal(keys.text(0), "cairo")
    assert_equal(sums[0], 4)
    assert_equal(keys.text(1), "lima")
    assert_equal(sums[1], 2)
    assert_equal(keys.text(2), "oslo")
    assert_equal(sums[2], 4)


def test_a_group_by_on_text_keeps_first_seen_order_when_asked() raises:
    var frame = keyed(text(["oslo", "lima", "oslo"]), [1, 2, 3])
    var specs = List[AggSpec]()
    specs.append(AggSpec("v", AggKind.SUM))
    var out = frame.group_by(["k"], specs, sort=False)
    assert_equal(out.column("k").text(0), "oslo")
    assert_equal(out.column("k").text(1), "lima")


def test_a_null_key_is_dropped_or_kept_as_pandas_does() raises:
    var frame = keyed(
        with_nulls(["a", "b", "a", "x"], [True, True, True, False]),
        [1, 2, 3, 4],
    )
    var specs = List[AggSpec]()
    specs.append(AggSpec("v", AggKind.SUM))

    var dropped = frame.group_by(["k"], specs)
    assert_equal(len(dropped), 2)

    var kept = frame.group_by(["k"], specs, dropna=False)
    assert_equal(len(kept), 3)
    var keys = kept.column("k")
    assert_false(keys.is_valid(2))
    assert_equal(kept.column("v_sum").as_typed[DType.int64]()[2], 4)


def test_a_text_key_combines_with_a_number_key() raises:
    var series = List[Series]()
    series.append(Series("k", text(["a", "a", "b", "a"])))
    series.append(Series("n", ints([1, 2, 1, 1])))
    series.append(Series("v", ints([10, 20, 30, 40])))
    var frame = DataFrame.from_series(series^)

    var specs = List[AggSpec]()
    specs.append(AggSpec("v", AggKind.SUM))
    var out = frame.group_by(["k", "n"], specs)

    assert_equal(len(out), 3)
    assert_equal(out.column("k").text(0), "a")
    assert_equal(out.column("n").as_typed[DType.int64]()[0], 1)
    assert_equal(out.column("v_sum").as_typed[DType.int64]()[0], 50)


def test_grouping_with_no_reductions_gives_the_distinct_keys() raises:
    var frame = keyed(text(["b", "a", "b", "a"]), [1, 2, 3, 4])
    var out = frame.group_by(["k"], List[AggSpec]())
    assert_equal(len(out), 2)
    assert_equal(out.column("k").text(0), "a")
    assert_equal(out.column("k").text(1), "b")


def test_an_inner_join_matches_on_text() raises:
    var left = keyed(text(["oslo", "lima", "cairo"]), [1, 2, 3])
    var right_series = List[Series]()
    right_series.append(Series("k", text(["lima", "oslo", "quito"])))
    right_series.append(Series("w", ints([70, 80, 90])))
    var right = DataFrame.from_series(right_series^)

    var out = left.join(right, ["k"]).sort_by("k")
    assert_equal(len(out), 2)
    assert_equal(out.column("k").text(0), "lima")
    assert_equal(out.column("w").as_typed[DType.int64]()[0], 70)
    assert_equal(out.column("k").text(1), "oslo")
    assert_equal(out.column("w").as_typed[DType.int64]()[1], 80)


def test_an_outer_join_takes_the_key_from_whichever_side_had_the_row() raises:
    """The path that would otherwise put a null in the column the rows matched
    on."""
    var left = keyed(text(["oslo", "lima"]), [1, 2])
    var right_series = List[Series]()
    right_series.append(Series("k", text(["lima", "quito"])))
    right_series.append(Series("w", ints([70, 90])))
    var right = DataFrame.from_series(right_series^)

    var out = left.join(right, ["k"], JoinKind.OUTER).sort_by("k")
    assert_equal(len(out), 3)
    assert_equal(out.column("k").text(0), "lima")
    assert_equal(out.column("k").text(1), "oslo")
    assert_equal(out.column("k").text(2), "quito")
    assert_false(out.column("w").is_valid(1))
    assert_false(out.column("v").is_valid(2))


def test_a_semi_join_on_text_keeps_the_left_rows_that_matched() raises:
    var left = keyed(text(["oslo", "lima", "cairo"]), [1, 2, 3])
    var right_series = List[Series]()
    right_series.append(Series("k", text(["lima", "lima", "quito"])))
    var right = DataFrame.from_series(right_series^)

    var out = left.join(right, ["k"], JoinKind.SEMI)
    assert_equal(len(out), 1)
    assert_equal(out.column("k").text(0), "lima")


def test_joining_text_against_bytes_is_refused() raises:
    """Both are uint8 physically, so the dtype check alone lets this through."""
    var left = keyed(text(["a"]), [1])
    var right_series = List[Series]()
    right_series.append(Series("k", Array[DType.uint8](1)))
    right_series.append(Series("w", ints([9])))
    var right = DataFrame.from_series(right_series^)
    with assert_raises():
        _ = left.join(right, ["k"])


def repeating_text(n: Int, groups: Int) raises -> StringArray:
    """Builds `n` rows of text cycling through `groups` distinct keys.

    Every key starts with the same four bytes, which is the prefix a view
    stores, so a probe that matches on the hash still has to read the payload to
    settle. A generator that varied the first byte would leave the comparison
    doing nothing on the route these tests are here to check.

    Args:
        n: How many rows.
        groups: How many distinct keys to cycle through.

    Returns:
        The column.
    """
    var builder = StringBuilder(capacity=n)
    for i in range(n):
        builder.append(String("key_", (i * 37) % groups).as_bytes())
    return builder^.finish()


def same_string_routes(col: StringArray, workers: Int, what: String) raises:
    """Runs both string routes over one column and asserts they agree.

    The comparison is exact: the group count, the null group, every ordinal and
    every representative row. It scans for the first disagreement and asserts
    once rather than asserting per row, because building a failure message for
    a row that passes costs more than the factorization it is checking.

    Args:
        col: The column.
        workers: How many slices to give the parallel route.
        what: A label for the failure message.
    """
    var serial = _factorize_strings_serial(col, DEFAULT_SEED)
    var parallel = _factorize_strings_parallel(col, DEFAULT_SEED, workers)

    assert_equal(parallel.count(), serial.count(), what)
    assert_equal(parallel.null_group, serial.null_group, what)
    assert_equal(len(parallel.firsts), len(serial.firsts), what)

    var row = -1
    for i in range(len(serial.codes)):
        if serial.codes[i] != parallel.codes[i]:
            row = i
            break
    assert_equal(row, -1, String(what, ": ordinals differ"))

    var group = -1
    for g in range(len(serial.firsts)):
        if serial.firsts[g] != parallel.firsts[g]:
            group = g
            break
    assert_equal(group, -1, String(what, ": representative rows differ"))


def test_the_parallel_string_route_agrees_on_a_few_groups() raises:
    same_string_routes(repeating_text(4096, 100), 4, "few")


def test_the_parallel_string_route_agrees_on_many_groups() raises:
    same_string_routes(repeating_text(4096, 3000), 8, "many")


def test_the_parallel_string_route_agrees_when_every_row_is_its_own_key() raises:
    same_string_routes(repeating_text(4096, 4096), 6, "distinct")


def test_the_parallel_string_route_agrees_on_an_uneven_slice_count() raises:
    same_string_routes(repeating_text(4096, 1500), 7, "uneven")


def test_the_parallel_string_route_agrees_with_one_worker() raises:
    same_string_routes(repeating_text(4096, 500), 1, "one worker")


def test_the_parallel_string_route_agrees_with_nulls_scattered() raises:
    var builder = StringBuilder(capacity=4096)
    for i in range(4096):
        if i % 11 == 0:
            builder.append_null()
        else:
            builder.append(String("key_", i % 400).as_bytes())
    same_string_routes(builder^.finish(), 5, "nulls")


def test_the_parallel_string_route_agrees_when_every_row_is_null() raises:
    var builder = StringBuilder(capacity=4096)
    for _ in range(4096):
        builder.append_null()
    same_string_routes(builder^.finish(), 4, "all null")


def test_the_parallel_string_route_keeps_first_appearance_order_across_slices() raises:
    # Four quarters, each introducing one key nothing before it had, so the
    # ordinals can only come out right if the merge walks the workers in order.
    var quarter = 1024
    var builder = StringBuilder(capacity=quarter * 4)
    for q in range(4):
        for i in range(quarter):
            if i == 0:
                builder.append(String("new_", q).as_bytes())
            else:
                builder.append(String("key_", i % 7).as_bytes())
    var col = builder^.finish()

    var found = _factorize_strings_parallel(col, DEFAULT_SEED, 4)
    assert_equal(found.count(), 4 + 7)
    assert_equal(found.null_group, -1)

    # The first quarter introduces `new_0` and then seven shared keys, so the
    # first eight ordinals are settled before any other slice contributes one.
    assert_equal(Int(found.codes[0]), 0)
    for i in range(1, 8):
        assert_equal(Int(found.codes[i]), i)
    assert_equal(Int(found.codes[quarter]), 8)
    assert_equal(Int(found.codes[quarter * 2]), 9)
    assert_equal(Int(found.codes[quarter * 3]), 10)
    assert_equal(found.firsts[8], quarter)
    assert_equal(found.firsts[9], quarter * 2)
    assert_equal(found.firsts[10], quarter * 3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
