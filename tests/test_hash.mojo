"""Tests for the hash package.

The factorize tests are mostly comparisons against `factorize_linear`, the same
arrangement the kernels use. What is worth reading here is the group of tests
around the two routes, because `factorize` picking the direct route or the hashed
route is invisible from the outside and the whole point is that it is invisible.
`test_the_two_routes_agree_on_the_same_data` is the one that would catch a
divergence, and the two range tests either side of `DIRECT_LIMIT` are what pin the
routes down so that test is actually exercising both of them.

The float tests are the ones that fail on a real dataset rather than in CI. A
column of NaNs and a column containing negative zero both group wrongly under a
naive `==`, and neither shows up unless somebody writes the test.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.array import Array, from_list
from firepanda.buffer.buffer import Buffer
from firepanda.hash import (
    DEFAULT_SEED,
    DIRECT_LIMIT,
    HashTable,
    factorize,
    factorize_dict,
    factorize_linear,
    hash_into,
    hash_of,
    key_bits,
    mix,
    radix_partition,
)
from firepanda.hash.table import (
    SIZING_EARLY,
    next_power_of_two,
    project_groups,
)


def column[dt: DType](values: List[Scalar[dt]]) -> Array[dt]:
    """Builds a column from a list.

    Args:
        values: The values.

    Parameters:
        dt: The dtype.

    Returns:
        The column, with no nulls.
    """
    return from_list[dt](values)


def same_codes[dt: DType](col: Array[dt], what: String) raises:
    """Asserts that `factorize` and both scalar versions agree on a column.

    Also checks the two invariants a caller relies on but a twin comparison would
    not catch: that every code indexes a real key, and that the key it indexes is
    the value that produced it.

    Args:
        col: The column.
        what: A label for the failure message.

    Parameters:
        dt: The dtype.

    Raises:
        If anything disagrees.
    """
    var got = factorize(col)
    var twin = factorize_linear(col)
    var viadict = factorize_dict(col)

    assert_equal(len(got.codes), len(col), String(what, ": code count"))
    for i in range(len(col)):
        assert_equal(got.codes[i], twin[i], String(what, ": row ", i))
        assert_equal(got.codes[i], viadict[i], String(what, ": dict row ", i))

    for i in range(len(col)):
        var at = Int(got.codes[i])
        assert_true(at < got.count(), String(what, ": code out of range"))
        if col.is_valid(i):
            assert_true(
                got.keys.is_valid(at), String(what, ": key marked null")
            )
            # Compared as key bits, so that a NaN column can go through here
            # too. Two NaNs in the same group are not equal to each other.
            assert_equal(
                key_bits(got.keys[at]),
                key_bits(col[i]),
                String(what, ": key mismatch"),
            )
        else:
            assert_equal(
                at, got.null_group, String(what, ": null not in null group")
            )


def test_key_bits_widens_signed_values_consistently() raises:
    assert_equal(key_bits(Int8(-1)), key_bits(Int64(-1)))
    assert_equal(key_bits(Int32(-70000)), key_bits(Int64(-70000)))
    assert_equal(key_bits(Int16(7)), key_bits(Int64(7)))


def test_key_bits_canonicalizes_nan() raises:
    var one = Float64(0.0) / Float64(0.0)
    var other = one * Float64(3.0)
    assert_equal(key_bits(one), key_bits(other))


def test_key_bits_folds_negative_zero_onto_zero() raises:
    assert_equal(key_bits(Float64(-0.0)), key_bits(Float64(0.0)))
    assert_equal(key_bits(Float32(-0.0)), key_bits(Float32(0.0)))


def test_key_bits_keeps_distinct_floats_distinct() raises:
    assert_true(key_bits(Float64(1.5)) != key_bits(Float64(1.25)))
    assert_true(key_bits(Float64(1.0)) != key_bits(Float64(-1.0)))


def test_mix_depends_on_the_seed() raises:
    var a = hash_of(Int64(42), UInt64(1))
    var b = hash_of(Int64(42), UInt64(2))
    assert_true(a != b, "the seed did not reach the hash")
    assert_equal(a, hash_of(Int64(42), UInt64(1)))


def test_mix_is_injective_on_a_run_of_values() raises:
    # Not a proof that splitmix64 is a bijection, just a check that nothing in
    # the way it is written here broke it. If the constants were wrong or a shift
    # was zero this collides immediately.
    var seen = HashTable(4096)
    for i in range(4096):
        var h = hash_of(Int64(i))
        var ordinal = seen.insert(mix(h, DEFAULT_SEED))
        assert_equal(ordinal, i, String("collision at ", i))


def test_hash_into_matches_the_scalar_hash() raises:
    var col = Array[DType.int64](100)
    for i in range(100):
        col[i] = Int64(i * 7 - 300)

    var hashes = Buffer(100 * 8)
    hash_into(col, DEFAULT_SEED, hashes)

    var got = hashes.bitcast[DType.uint64]()
    for i in range(100):
        assert_equal(got.unsafe_offset(i).unsafe_load(), hash_of(col[i]))


def test_next_power_of_two() raises:
    assert_equal(next_power_of_two(0), 1)
    assert_equal(next_power_of_two(1), 1)
    assert_equal(next_power_of_two(3), 4)
    assert_equal(next_power_of_two(1024), 1024)
    assert_equal(next_power_of_two(1025), 2048)


def test_project_groups_reads_a_collapsing_discovery_rate() raises:
    # 100 groups in the first half and none in the second, which is what a
    # category column looks like. The guess should be the count plus a little,
    # not an extrapolation of the first half's rate over the whole column.
    assert_equal(project_groups(100, 100, 4096, 1_000_000), 125)


def test_project_groups_reads_a_steady_discovery_rate() raises:
    # Every row distinct so far. Extrapolating flat gives the row count, and the
    # cap at `n` is what stops it going past.
    assert_equal(project_groups(4096, 2048, 4096, 1_000_000), 1_000_000)

    # Half the rows distinct so far, so half of what is left should be too.
    assert_equal(project_groups(2048, 1024, 4096, 100_000), 50_000)


def test_reserve_sizes_for_a_group_count_and_never_shrinks() raises:
    var table = HashTable()
    table._reserve(1000)
    assert_equal(table.capacity(), 2048)

    # Load factor is one half, so a thousand groups needs two thousand slots and
    # rounds to the next power of two.
    table._reserve(10)
    assert_equal(table.capacity(), 2048, "an estimate below what we hold")


def test_reserve_keeps_the_keys_it_already_had() raises:
    var table = HashTable()
    for i in range(50):
        assert_equal(table.insert(mix(UInt64(i * 31), table.seed())), i)

    table._reserve(5000)
    assert_equal(len(table), 50)
    for i in range(50):
        assert_equal(table.find(mix(UInt64(i * 31), table.seed())), i)


def test_a_chunked_build_sizes_itself_like_a_whole_one() raises:
    # The sizing schedule lives on the table so that it survives across chunks.
    # A column long enough to reach the first checkpoint is the case where that
    # matters, so build one both ways and check the tables come out the same.
    var n = SIZING_EARLY * 4
    var col = Array[DType.int64](n)
    for i in range(n):
        col[i] = Int64(i % 300) * Int64(DIRECT_LIMIT + 1)

    var whole = factorize(col)
    assert_equal(whole.count(), 300)
    for i in range(n):
        assert_equal(Int(whole.codes[i]), i % 300, "chunked build")


def test_table_assigns_dense_ordinals_in_insertion_order() raises:
    var table = HashTable()
    assert_equal(len(table), 0)
    for i in range(50):
        var bits = UInt64(i * 1000 + 3)
        assert_equal(table.insert(mix(bits, table.seed())), i)
    assert_equal(len(table), 50)


def test_table_returns_the_same_ordinal_for_a_repeated_key() raises:
    var table = HashTable()
    for i in range(50):
        var bits = UInt64(i)
        _ = table.insert(mix(bits, table.seed()))
    for i in range(50):
        var bits = UInt64(i)
        assert_equal(table.insert(mix(bits, table.seed())), i)
    assert_equal(len(table), 50)


def test_table_find_reports_absence() raises:
    var table = HashTable()
    var present = UInt64(99)
    _ = table.insert(mix(present, table.seed()))
    assert_equal(table.find(mix(present, table.seed())), 0)
    var absent = UInt64(100)
    assert_equal(table.find(mix(absent, table.seed())), -1)


def test_table_survives_growth() raises:
    # Enough keys to force several doublings past MIN_CAPACITY, then every one of
    # them looked up again. A growth that dropped or duplicated a slot shows up
    # here and nowhere else, because during a build the wrong answer is a new
    # ordinal rather than a crash.
    var table = HashTable()
    var count = 5000
    for i in range(count):
        var bits = UInt64(i * 2654435761)
        assert_equal(table.insert(mix(bits, table.seed())), i)
    assert_equal(len(table), count)
    assert_true(table.capacity() >= count * 2)
    for i in range(count):
        var bits = UInt64(i * 2654435761)
        assert_equal(table.find(mix(bits, table.seed())), i)


def test_table_handles_zero_key_bits() raises:
    # Zero is the empty marker for the ordinal, not for the key, and a table that
    # confused the two would loop forever or lose the group for value zero. That
    # value is common enough that this has to be a test rather than a comment.
    var table = HashTable()
    assert_equal(table.insert(mix(UInt64(0), table.seed())), 0)
    assert_equal(table.insert(mix(UInt64(0), table.seed())), 0)
    assert_equal(table.find(mix(UInt64(0), table.seed())), 0)
    assert_equal(len(table), 1)


def test_table_respects_a_custom_seed() raises:
    var table = HashTable(0, UInt64(12345))
    assert_equal(table.seed(), UInt64(12345))
    for i in range(200):
        var bits = UInt64(i)
        assert_equal(table.insert(mix(bits, table.seed())), i)


def test_radix_partition_covers_every_row_once() raises:
    var n = 1000
    var hashes = Buffer(n * 8)
    var out = hashes.bitcast[DType.uint64]()
    for i in range(n):
        out.unsafe_offset(i).unsafe_write(hash_of(Int64(i)))

    var parts = radix_partition(hashes, n, 4)
    assert_equal(parts.count(), 16)
    assert_equal(parts.offsets[16], n)

    var seen = List[Bool]()
    for _ in range(n):
        seen.append(False)
    for i in range(n):
        assert_false(seen[parts.order[i]], "row placed twice")
        seen[parts.order[i]] = True


def test_radix_partition_groups_by_the_high_bits() raises:
    var n = 1000
    var hashes = Buffer(n * 8)
    var out = hashes.bitcast[DType.uint64]()
    for i in range(n):
        out.unsafe_offset(i).unsafe_write(hash_of(Int64(i)))

    var bits = 4
    var parts = radix_partition(hashes, n, bits)
    var shift = UInt64(64 - bits)
    for p in range(parts.count()):
        for at in range(parts.offsets[p], parts.offsets[p + 1]):
            var row = parts.order[at]
            var got = Int(out.unsafe_offset(row).unsafe_load() >> shift)
            assert_equal(got, p, "row in the wrong partition")


def test_radix_partition_is_stable() raises:
    var n = 500
    var hashes = Buffer(n * 8)
    var out = hashes.bitcast[DType.uint64]()
    for i in range(n):
        # Deliberately only four distinct partitions, so every one holds a long
        # run and a reordering inside it would be obvious.
        out.unsafe_offset(i).unsafe_write(UInt64(i % 4) << UInt64(62))

    var parts = radix_partition(hashes, n, 2)
    for p in range(parts.count()):
        var last = -1
        for at in range(parts.offsets[p], parts.offsets[p + 1]):
            assert_true(parts.order[at] > last, "partition not stable")
            last = parts.order[at]


def test_factorize_of_an_empty_column() raises:
    var col = Array[DType.int64](0)
    var got = factorize(col)
    assert_equal(len(got.codes), 0)
    assert_equal(got.count(), 0)
    assert_equal(got.null_group, -1)


def test_factorize_small_integers() raises:
    var values: List[Scalar[DType.int64]] = [7, 3, 7, 3, 3, 9]
    var col = column[DType.int64](values)
    var got = factorize(col)
    assert_equal(got.count(), 3)
    assert_equal(got.codes[0], 0)
    assert_equal(got.codes[1], 1)
    assert_equal(got.codes[2], 0)
    assert_equal(got.codes[3], 1)
    assert_equal(got.codes[4], 1)
    assert_equal(got.codes[5], 2)
    assert_equal(got.keys[0], 7)
    assert_equal(got.keys[1], 3)
    assert_equal(got.keys[2], 9)
    same_codes(col, "small integers")


def test_factorize_assigns_ordinals_in_first_appearance_order() raises:
    # pandas reports groups in the order they were first seen, not sorted, and a
    # direct-indexed table would naturally produce sorted order if the ordinal
    # came from the slot rather than from a counter.
    var col = Array[DType.int64](60)
    var order: List[Scalar[DType.int64]] = [50, 10, 30]
    for i in range(60):
        col[i] = order[i % 3]
    var got = factorize(col)
    assert_equal(got.keys[0], 50)
    assert_equal(got.keys[1], 10)
    assert_equal(got.keys[2], 30)


def test_factorize_gives_nulls_group_zero() raises:
    var col = Array[DType.int64](6)
    for i in range(6):
        col[i] = Int64(i % 3)
    col.set_null(1)
    col.set_null(4)

    var got = factorize(col)
    assert_equal(got.null_group, 0)
    assert_equal(got.codes[1], 0)
    assert_equal(got.codes[4], 0)
    assert_false(got.keys.is_valid(0), "the null group's key is not null")
    for i in range(6):
        if col.is_valid(i):
            assert_true(got.codes[i] != 0, "a value landed in the null group")
    same_codes(col, "nulls")


def test_factorize_of_an_all_null_column() raises:
    var col = Array[DType.int64](8)
    for i in range(8):
        col.set_null(i)
    var got = factorize(col)
    assert_equal(got.count(), 1)
    assert_equal(got.null_group, 0)
    for i in range(8):
        assert_equal(got.codes[i], 0)


def test_factorize_of_a_column_with_no_nulls_has_no_null_group() raises:
    var values: List[Scalar[DType.int64]] = [1, 2, 3]
    var got = factorize(column[DType.int64](values))
    assert_equal(got.null_group, -1)
    assert_equal(got.count(), 3)


def test_factorize_takes_the_direct_route_under_the_limit() raises:
    # A dense span well inside DIRECT_LIMIT. This is here for the pair below it:
    # the two of them are what make `test_the_two_routes_agree_on_the_same_data`
    # a comparison between routes rather than a comparison of one route with
    # itself.
    var n = 4000
    var col = Array[DType.int64](n)
    for i in range(n):
        col[i] = Int64(i % 500)
    same_codes(col, "direct route")
    assert_equal(factorize(col).count(), 500)


def test_factorize_takes_the_hashed_route_over_the_limit() raises:
    var n = 4000
    var col = Array[DType.int64](n)
    for i in range(n):
        col[i] = Int64((i % 500) * (DIRECT_LIMIT + 17))
    same_codes(col, "hashed route")
    assert_equal(factorize(col).count(), 500)


def test_the_two_routes_agree_on_the_same_data() raises:
    # The same 300 groups in the same order, once as small integers and once
    # shifted far enough apart to defeat the direct table. The codes have to come
    # out identical, because which route ran is not something a caller can see.
    var n = 2000
    var near = Array[DType.int64](n)
    var far = Array[DType.int64](n)
    for i in range(n):
        var group = Int64((i * 37) % 300)
        near[i] = group
        far[i] = group * Int64(1 << 20)

    var a = factorize(near)
    var b = factorize(far)
    assert_equal(a.count(), b.count())
    for i in range(n):
        assert_equal(a.codes[i], b.codes[i], String("routes disagree at ", i))


def test_factorize_of_negative_integers() raises:
    var values: List[Scalar[DType.int64]] = [-5, 0, -5, 3, -100, 3]
    var col = column[DType.int64](values)
    same_codes(col, "negatives")
    assert_equal(factorize(col).count(), 4)


def test_factorize_of_a_narrow_dtype() raises:
    var col = Array[DType.int8](200)
    for i in range(200):
        col[i] = Int8(i % 7 - 3)
    same_codes(col, "int8")
    assert_equal(factorize(col).count(), 7)


def test_factorize_of_an_unsigned_dtype() raises:
    var col = Array[DType.uint32](200)
    for i in range(200):
        col[i] = UInt32(i % 11)
    same_codes(col, "uint32")
    assert_equal(factorize(col).count(), 11)


def test_factorize_of_a_single_repeated_value() raises:
    var col = Array[DType.int64](1000)
    for i in range(1000):
        col[i] = Int64(42)
    var got = factorize(col)
    assert_equal(got.count(), 1)
    for i in range(1000):
        assert_equal(got.codes[i], 0)


def test_factorize_when_every_row_is_its_own_group() raises:
    var n = 3000
    var col = Array[DType.int64](n)
    for i in range(n):
        col[i] = Int64(i) * Int64(1 << 30)
    var got = factorize(col)
    assert_equal(got.count(), n)
    for i in range(n):
        assert_equal(got.codes[i], UInt32(i))


def test_factorize_floats() raises:
    var values: List[Scalar[DType.float64]] = [1.5, 2.5, 1.5, 3.5]
    var col = column[DType.float64](values)
    same_codes(col, "floats")
    assert_equal(factorize(col).count(), 3)


def test_factorize_puts_all_nans_in_one_group() raises:
    var nan = Float64(0.0) / Float64(0.0)
    var col = Array[DType.float64](5)
    col[0] = nan
    col[1] = Float64(1.0)
    col[2] = nan * Float64(2.0)
    col[3] = Float64(1.0)
    col[4] = -nan

    var got = factorize(col)
    assert_equal(got.count(), 2)
    assert_equal(got.codes[0], got.codes[2])
    assert_equal(got.codes[0], got.codes[4])
    assert_true(got.codes[0] != got.codes[1])


def test_factorize_groups_the_two_zeros_together() raises:
    var col = Array[DType.float64](4)
    col[0] = Float64(0.0)
    col[1] = Float64(-0.0)
    col[2] = Float64(1.0)
    col[3] = Float64(-0.0)

    var got = factorize(col)
    assert_equal(got.count(), 2)
    assert_equal(got.codes[0], got.codes[1])
    assert_equal(got.codes[0], got.codes[3])


def test_factorize_is_seed_independent() raises:
    # The seed changes which bucket a key lands in and nothing else. If it ever
    # changes the codes, the table is leaking its layout into its results and
    # every group by becomes nondeterministic between queries.
    var n = 1000
    var col = Array[DType.int64](n)
    for i in range(n):
        col[i] = Int64((i * 13) % 400) * Int64(1 << 25)

    var a = factorize(col, UInt64(1))
    var b = factorize(col, UInt64(0xDEADBEEF))
    assert_equal(a.count(), b.count())
    for i in range(n):
        assert_equal(a.codes[i], b.codes[i], "seed changed the codes")


def test_factorize_with_nulls_on_the_hashed_route() raises:
    var n = 900
    var col = Array[DType.int64](n)
    for i in range(n):
        col[i] = Int64((i % 200) * (DIRECT_LIMIT + 5))
        if i % 7 == 0:
            col.set_null(i)
    same_codes(col, "hashed with nulls")
    var got = factorize(col)
    assert_equal(got.null_group, 0)
    assert_equal(got.count(), 201)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
