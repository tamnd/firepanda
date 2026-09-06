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

The last three tests are not about answers at all, they are about where the two
split thresholds sit relative to the chunk size, to the minimum slice and to each
other. Nothing they check can make a wrong group, so nothing else in this file
would catch it, and what it costs when it is wrong is several times the runtime
of every group by a pipeline does.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.array import Array, from_list
from firepanda.buffer.buffer import Buffer
from firepanda.hash import (
    DEFAULT_SEED,
    DIRECT_LIMIT,
    HashTable,
    direct_plan,
    factorize,
    factorize_dense,
    factorize_dict,
    factorize_linear,
    hash_into,
    hash_of,
    key_bits,
    mix,
    radix_partition,
)
from firepanda.exec import MORSEL_ROWS, worker_count
from firepanda.hash.factorize import (
    DIRECT_SHARE,
    PARALLEL_MIN_SLICE,
    PARALLEL_ROWS,
    PARALLEL_STRING_ROWS,
    PLAN_PREFIX_ROWS,
    RANK_BLOCK,
    Factorized,
    _estimate_groups,
    _factorize_direct_parallel,
    _factorize_direct_serial,
    _factorize_hashed_parallel,
    _factorize_hashed_partitioned,
    _factorize_hashed_serial,
    _parallel_workers,
)
from firepanda.hash.table import (
    SIZING_EARLY,
    SIZING_LATE,
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
    var keys = got.keys(col)
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
            assert_true(keys.is_valid(at), String(what, ": key marked null"))
            # Compared as key bits, so that a NaN column can go through here
            # too. Two NaNs in the same group are not equal to each other.
            assert_equal(
                key_bits(keys[at]),
                key_bits(col[i]),
                String(what, ": key mismatch"),
            )
        else:
            assert_equal(
                at, got.null_group, String(what, ": null not in null group")
            )


def same_routes[dt: DType](col: Array[dt], workers: Int, what: String) raises:
    """Asserts the parallel hashed route lands exactly on the serial one.

    Every ordinal, every key and the null group, not merely the same partition
    of the rows. The parallel route is a substitution for the serial one rather
    than an equivalent-up-to-renumbering version of it, and a test that only
    checked which rows ended up together would pass on a route that had quietly
    lost first-appearance order.

    The worker count is passed in rather than taken from the machine so that the
    test means the same thing on a runner with one core as on one with thirty
    two, and so that a slice count that does not divide the column is covered.

    The comparisons find the first disagreement and then assert once, rather
    than asserting per row. Building a failure message for every one of a
    hundred thousand rows that were about to pass costs more than the
    factorization being tested.

    Args:
        col: The column.
        workers: How many slices to cut it into.
        what: A label for the failure message.

    Parameters:
        dt: The dtype.

    Raises:
        If the two routes disagree anywhere.
    """
    var many = _factorize_hashed_parallel[dt](col, DEFAULT_SEED, workers)
    lands_on_the_serial_route[dt](col, many^, what)


def same_partitioned_routes[
    dt: DType
](col: Array[dt], workers: Int, what: String) raises:
    """Asserts the partitioned hashed route lands exactly on the serial one.

    Held to the standard `same_routes` holds the slice route to, and it is a
    harder standard to meet here. The slice route gets first-appearance order
    out of the shape of its own work, because its workers hold contiguous slices
    in order. The partitioned route cuts by hash, so the order is recovered
    afterwards by `_rank_by_first_row` and a test that only checked which rows
    ended up together would pass on a route that had lost it.

    Args:
        col: The column.
        workers: How many slices to hash and scatter in parallel.
        what: A label for the failure message.

    Parameters:
        dt: The dtype.

    Raises:
        If the two routes disagree anywhere.
    """
    var many = _factorize_hashed_partitioned[dt](col, DEFAULT_SEED, workers)
    lands_on_the_serial_route[dt](col, many^, what)


def lands_on_the_serial_route[
    dt: DType
](col: Array[dt], var many: Factorized[dt], what: String) raises:
    """Asserts a factorization is the one the serial route would have produced.

    Args:
        col: The column it came from.
        many: What some parallel route made of it.
        what: A label for the failure message.

    Parameters:
        dt: The dtype.

    Raises:
        If the two disagree anywhere.
    """
    var one = _factorize_hashed_serial[dt](col, DEFAULT_SEED)

    assert_equal(many.count(), one.count(), String(what, ": group count"))
    assert_equal(many.null_group, one.null_group, String(what, ": null group"))

    var row = -1
    for i in range(len(col)):
        if many.codes[i] != one.codes[i]:
            row = i
            break
    assert_equal(row, -1, String(what, ": ordinals differ at row ", row))

    var many_keys = many.keys(col)
    var one_keys = one.keys(col)
    var key = -1
    for g in range(one.count()):
        if many_keys.is_valid(g) != one_keys.is_valid(g):
            key = g
            break
        if one_keys.is_valid(g) and key_bits(many_keys[g]) != key_bits(
            one_keys[g]
        ):
            key = g
            break
    assert_equal(key, -1, String(what, ": keys differ at group ", key))


def hashed_column(n: Int, groups: Int) -> Array[DType.int64]:
    """Builds a column the direct route would refuse, with a known group count.

    The values are spread past `DIRECT_LIMIT` so that `factorize` would send this
    to the hash table, and they are strided so that every group appears in every
    slice a parallel build could cut.

    Args:
        n: The number of rows.
        groups: The number of distinct values.

    Returns:
        The column, with no nulls.
    """
    var col = Array[DType.int64](n)
    for i in range(n):
        col[i] = Int64((i * 37) % groups) * Int64(DIRECT_LIMIT + 17)
    return col^


def same_direct_routes(
    col: Array[DType.int32], span: Int, workers: Int, what: String
) raises:
    """Asserts the parallel direct route lands exactly on the serial one.

    Same standard `same_routes` holds the hashed route to, and for the same
    reason. The direct route is what an integer key column with a narrow range
    takes, so the ordinals it hands back are what a group by over such a column
    numbers its groups with, and a merge that got first-appearance order wrong
    would renumber every one of them.

    Args:
        col: The column, with every value in `[0, span)`.
        span: The table width.
        workers: How many slices to cut the column into.
        what: A label for the failure message.

    Raises:
        If the two routes disagree anywhere.
    """
    var one = _factorize_direct_serial[DType.int32](col, span, Int32(0))
    var many = _factorize_direct_parallel[DType.int32](
        col, span, Int32(0), workers
    )

    assert_equal(many.count(), one.count(), String(what, ": group count"))
    assert_equal(many.null_group, one.null_group, String(what, ": null group"))

    var row = -1
    for i in range(len(col)):
        if many.codes[i] != one.codes[i]:
            row = i
            break
    assert_equal(row, -1, String(what, ": ordinals differ at row ", row))

    var many_keys = many.keys(col)
    var one_keys = one.keys(col)
    var key = -1
    for g in range(one.count()):
        if many_keys.is_valid(g) != one_keys.is_valid(g):
            key = g
            break
        if one_keys.is_valid(g) and key_bits(many_keys[g]) != key_bits(
            one_keys[g]
        ):
            key = g
            break
    assert_equal(key, -1, String(what, ": keys differ at group ", key))


def direct_column(n: Int, span: Int) -> Array[DType.int32]:
    """Builds a column the direct route would take, covering the whole range.

    The values are strided rather than sequential so that a group's first row
    and the slice it lands in are not the same thing, which is what a merge that
    took the last worker to see a value rather than the first would get wrong.

    Args:
        n: The number of rows.
        span: The value range, every value of which appears.

    Returns:
        The column, with no nulls.
    """
    var col = Array[DType.int32](n)
    for i in range(n):
        col[i] = Int32((i * 37) % span)
    return col^


def test_the_parallel_direct_route_agrees_on_a_narrow_range() raises:
    same_direct_routes(direct_column(8192, 100), 100, 4, "direct 100 values")


def test_the_parallel_direct_route_agrees_past_one_claim_block() raises:
    same_direct_routes(direct_column(70000, 9000), 9000, 8, "direct 9k values")


def test_the_parallel_direct_route_agrees_when_every_row_is_a_group() raises:
    var n = 8192
    var col = Array[DType.int32](n)
    for i in range(n):
        col[i] = Int32(i)
    same_direct_routes(col, n, 6, "direct all distinct")


def test_the_parallel_direct_route_agrees_with_nulls_present() raises:
    var col = direct_column(8192, 300)
    for i in range(len(col)):
        if i % 11 == 0:
            col.set_null(i)
    same_direct_routes(col, 300, 8, "direct with nulls")


def test_the_parallel_direct_route_leaves_most_of_its_range_unused() raises:
    var n = 8192
    var col = Array[DType.int32](n)
    for i in range(n):
        col[i] = Int32((i % 3) * 20000)
    same_direct_routes(col, 40001, 16, "direct sparse range")


def test_a_direct_slice_is_read_to_its_last_row_for_one_group() raises:
    """A worker on the direct route stops scanning once its table is full, and
    the last row of a slice is where that has to still be true.

    Four of the five values are everywhere and the fifth is on the last row of
    every slice, so a worker that stopped a group early, or that counted the
    group before recording the row it was on, would lose that value or record
    the wrong row for it. Both show up against the serial route, which reads
    every row and never stops.
    """
    var n = 8192
    var workers = 4
    var col = Array[DType.int32](n)
    for i in range(n):
        col[i] = Int32(i % 4)
    for w in range(workers):
        col[n * (w + 1) // workers - 1] = Int32(4)
    same_direct_routes(col, 5, workers, "direct last row is a group")


def test_a_direct_slice_stops_once_it_has_seen_everything() raises:
    """The other side of it: a slice whose table fills in its first few rows has
    nothing left to learn from the rest of it and stops reading.

    Every value is in the first twenty rows and the remaining eight thousand are
    repeats, so a worker that kept going would record nothing further, which is
    what makes the two routes still have to agree. The nulls are here because
    they are the rows a worker steps over without touching its table, and a stop
    condition that counted rows rather than groups would trip on them.
    """
    var n = 8192
    var col = Array[DType.int32](n)
    for i in range(n):
        col[i] = Int32(i % 20)
    for i in range(len(col)):
        if i % 13 == 0:
            col.set_null(i)
    same_direct_routes(col, 20, 8, "direct table fills early")


def test_the_parallel_route_agrees_on_a_low_cardinality_column() raises:
    same_routes(hashed_column(8192, 100), 4, "parallel 100 groups")


def test_the_parallel_route_agrees_on_a_high_cardinality_column() raises:
    same_routes(hashed_column(8192, 6000), 8, "parallel 10k groups")


def test_the_parallel_route_agrees_when_every_row_is_its_own_group() raises:
    var n = 8192
    var col = Array[DType.int64](n)
    for i in range(n):
        col[i] = Int64(i) * Int64(DIRECT_LIMIT + 17)
    same_routes(col, 6, "parallel all distinct")


def test_the_parallel_route_agrees_on_an_uneven_slice_count() raises:
    # Twenty four thousand rows into seven slices, so no slice boundary lands
    # where it would have if the column had been cut evenly.
    same_routes(hashed_column(8192, 3000), 7, "parallel 7 workers")


def test_the_parallel_route_on_one_worker_is_the_serial_route() raises:
    same_routes(hashed_column(8192, 500), 1, "parallel 1 worker")


def test_the_parallel_route_agrees_with_nulls_scattered_through_it() raises:
    var n = 8192
    var col = hashed_column(n, 2000)
    for i in range(n):
        if i % 11 == 0:
            col.set_null(i)
    same_routes(col, 5, "parallel with nulls")

    var got = _factorize_hashed_parallel(col, DEFAULT_SEED, 5)
    assert_equal(got.null_group, 0)
    assert_false(got.keys(col).is_valid(0))


def test_the_parallel_route_agrees_when_every_row_is_null() raises:
    var n = 8192
    var col = hashed_column(n, 64)
    for i in range(n):
        col.set_null(i)
    same_routes(col, 4, "parallel all null")

    var got = _factorize_hashed_parallel(col, DEFAULT_SEED, 4)
    assert_equal(got.count(), 1)
    var bad = -1
    for i in range(n):
        if got.codes[i] != 0:
            bad = i
            break
    assert_equal(bad, -1, String("null row not in group zero at ", bad))


def test_the_parallel_route_keeps_first_appearance_order_across_slices() raises:
    # Each quarter of the column introduces one group of its own, in order, so
    # the ordinals only come out right if the merge walks the slices in order.
    var n = 8192
    var col = Array[DType.int64](n)
    var step = Int64(DIRECT_LIMIT + 17)
    for i in range(n):
        col[i] = Int64(i * 4 // n) * step

    var got = _factorize_hashed_parallel(col, DEFAULT_SEED, 4)
    assert_equal(got.count(), 4)
    var bad = -1
    for i in range(n):
        if got.codes[i] != UInt32(i * 4 // n):
            bad = i
            break
    assert_equal(bad, -1, String("out of first appearance order at ", bad))
    var keys = got.keys(col)
    for g in range(4):
        assert_equal(keys[g], Int64(g) * step)


def test_the_parallel_merge_agrees_across_many_buckets() raises:
    # Sixteen workers puts the merge into sixty four buckets and three thousand
    # groups spreads entries across all of them, so no bucket sees the column in
    # the order the workers offered it and the numbering has to be recovered
    # from the ranking pass rather than from the order a bucket happened to run.
    same_routes(hashed_column(16384, 3000), 16, "parallel 16 workers")


def test_the_parallel_merge_agrees_past_one_ranking_block() raises:
    # An entry per row is the cheapest way to get more entries than one
    # `MERGE_BLOCK` holds, since a column can never produce more of them than it
    # has rows. The ranking pass then has to carry a running count from the
    # first block into the second, and one block off by one would renumber every
    # group after it.
    same_routes(hashed_column(70000, 70000), 16, "parallel past one block")


def test_the_parallel_merge_leaves_most_of_its_buckets_empty() raises:
    # Three groups across sixteen workers is three of sixty four buckets holding
    # anything at all. A bucket with nothing in it contributes nothing to the
    # numbering and must not shift what the others get.
    var n = 8192
    var col = Array[DType.int64](n)
    var step = Int64(DIRECT_LIMIT + 17)
    for i in range(n):
        col[i] = Int64(i % 3) * step

    var got = _factorize_hashed_parallel(col, DEFAULT_SEED, 16)
    assert_equal(got.count(), 3)
    var bad = -1
    for i in range(n):
        if got.codes[i] != UInt32(i % 3):
            bad = i
            break
    assert_equal(bad, -1, String("an empty bucket moved an ordinal at ", bad))


def test_the_partitioned_route_agrees_with_the_serial_one() raises:
    # Seven workers over a column that does not divide into seven, and three
    # thousand groups spread across all sixty four partitions, so no partition
    # sees the groups in the order the column offered them.
    same_partitioned_routes(hashed_column(24000, 3000), 7, "partitioned 7")


def test_the_partitioned_route_agrees_when_every_row_is_its_own_group() raises:
    # More rows than one `RANK_BLOCK` holds, and a group per row, so the
    # numbering has to carry a running count from the first block of rows into
    # the second. A block off by one would renumber every group after it.
    same_partitioned_routes(
        hashed_column(RANK_BLOCK + 4096, RANK_BLOCK + 4096),
        8,
        "partitioned distinct",
    )


def test_the_partitioned_route_agrees_with_nulls_scattered_through_it() raises:
    var n = 8192
    var col = hashed_column(n, 2000)
    for i in range(n):
        if i % 11 == 0:
            col.set_null(i)
    same_partitioned_routes(col, 5, "partitioned with nulls")

    var got = _factorize_hashed_partitioned(col, DEFAULT_SEED, 5)
    assert_equal(got.null_group, 0)
    assert_false(got.keys(col).is_valid(0))


def test_the_partitioned_route_agrees_when_every_row_is_null() raises:
    # Nothing reaches a partition at all, so every array the route sizes from
    # the placed rows is empty and the ranking has no groups to number.
    var n = 8192
    var col = hashed_column(n, 64)
    for i in range(n):
        col.set_null(i)
    same_partitioned_routes(col, 4, "partitioned all null")

    var got = _factorize_hashed_partitioned(col, DEFAULT_SEED, 4)
    assert_equal(got.count(), 1)
    var bad = -1
    for i in range(n):
        if got.codes[i] != 0:
            bad = i
            break
    assert_equal(bad, -1, String("null row not in group zero at ", bad))


def test_the_partitioned_route_keeps_first_appearance_order() raises:
    # Four groups, each introduced by its own quarter of the column, in order.
    # Which partition each lands in is the hash's business and has nothing to do
    # with that order, so the ordinals only come out right if the numbering is
    # recovered from the representative rows.
    var n = 8192
    var col = Array[DType.int64](n)
    var step = Int64(DIRECT_LIMIT + 17)
    for i in range(n):
        col[i] = Int64(i * 4 // n) * step

    var got = _factorize_hashed_partitioned(col, DEFAULT_SEED, 4)
    assert_equal(got.count(), 4)
    var bad = -1
    for i in range(n):
        if got.codes[i] != UInt32(i * 4 // n):
            bad = i
            break
    assert_equal(bad, -1, String("out of first appearance order at ", bad))
    var keys = got.keys(col)
    for g in range(4):
        assert_equal(keys[g], Int64(g) * step)


def test_the_partitioned_route_leaves_most_of_its_partitions_empty() raises:
    # Three groups across sixteen workers is three of sixty four partitions
    # holding anything at all, and one block of the ranking holding all three.
    # An empty partition contributes nothing and must not shift what the others
    # get.
    var n = 8192
    var col = Array[DType.int64](n)
    var step = Int64(DIRECT_LIMIT + 17)
    for i in range(n):
        col[i] = Int64(i % 3) * step

    var got = _factorize_hashed_partitioned(col, DEFAULT_SEED, 16)
    assert_equal(got.count(), 3)
    var bad = -1
    for i in range(n):
        if got.codes[i] != UInt32(i % 3):
            bad = i
            break
    assert_equal(
        bad, -1, String("an empty partition moved an ordinal at ", bad)
    )


def test_the_parallel_route_sizes_a_slice_past_the_late_checkpoint() raises:
    # Two slices, each longer than `SIZING_LATE`, so each worker's own row count
    # crosses both sizing checkpoints. A worker that counted rows from the start
    # of the column rather than from the start of its own slice would size the
    # second slice off the first row it ever saw, or not size it at all.
    #
    # The expected answer is written out rather than taken from the serial
    # route, because the column is long enough that running it twice is most of
    # what this file costs. Three hundred groups keeps the tables small so the
    # row count is what is being paid for.
    var n = 2 * (SIZING_LATE + 2048)
    var col = Array[DType.int64](n)
    for i in range(n):
        col[i] = Int64(i % 300) * Int64(DIRECT_LIMIT + 17)

    var got = _factorize_hashed_parallel(col, DEFAULT_SEED, 2)
    assert_equal(got.count(), 300)
    var bad = -1
    for i in range(n):
        if got.codes[i] != UInt32(i % 300):
            bad = i
            break
    assert_equal(bad, -1, String("slice sizing changed the answer at ", bad))


def test_the_parallel_route_agrees_on_floats() raises:
    var n = 8192
    var nan = Float64(0.0) / Float64(0.0)
    var col = Array[DType.float64](n)
    for i in range(n):
        col[i] = Float64((i * 37) % 700) * Float64(0.5)
    for i in range(0, n, 13):
        col[i] = nan
    same_routes(col, 4, "parallel floats")


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
    var keys = got.keys(col)
    assert_equal(keys[0], 7)
    assert_equal(keys[1], 3)
    assert_equal(keys[2], 9)
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
    var keys = got.keys(col)
    assert_equal(keys[0], 50)
    assert_equal(keys[1], 10)
    assert_equal(keys[2], 30)


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
    assert_false(got.keys(col).is_valid(0), "the null group's key is not null")
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


def test_factorize_takes_a_table_wider_than_the_limit_on_a_long_column() raises:
    """A span past `DIRECT_LIMIT` and inside the share, which is now direct.

    The bound `factorize` offers the direct table on a long column is a slot for
    every `DIRECT_SHARE` rows, so this column takes a table of ninety thousand
    slots where it used to be hashed. The same groups in the same order are built
    again far enough apart that no row count could accept them, which pins that
    second column to the hash, and the two have to come out with identical codes.
    """
    var n = DIRECT_LIMIT * DIRECT_SHARE * 2
    var near = Array[DType.int64](n)
    var far = Array[DType.int64](n)
    for i in range(n):
        var value = Int64((i * 7919) % 90000)
        near[i] = value
        far[i] = value * Int64(1 << 20)

    var direct = factorize(near)
    var hashed = factorize(far)
    assert_equal(direct.count(), 90000)
    assert_equal(direct.count(), hashed.count())
    var bad = -1
    for i in range(n):
        if direct.codes[i] != hashed.codes[i]:
            bad = i
            break
    assert_equal(bad, -1, String("routes disagree at ", bad))


def test_a_wide_direct_table_still_puts_the_nulls_at_ordinal_zero() raises:
    """The null group on a span the direct route only reaches on a long column.

    Nulls at ordinal zero and the keys following in first-appearance order is a
    promise both routes make, and the direct one had never been asked to make it
    at a span this wide because nothing sent it one.
    """
    var n = DIRECT_LIMIT * DIRECT_SHARE * 2
    var near = Array[DType.int64](n)
    var far = Array[DType.int64](n)
    for i in range(n):
        var value = Int64((i * 7919) % 90000)
        near[i] = value
        far[i] = value * Int64(1 << 20)
        if i % 7 == 0:
            near.set_null(i)
            far.set_null(i)

    var direct = factorize(near)
    var hashed = factorize(far)
    assert_equal(direct.null_group, 0)
    assert_equal(direct.count(), hashed.count())
    var bad = -1
    for i in range(n):
        if direct.codes[i] != hashed.codes[i]:
            bad = i
            break
    assert_equal(bad, -1, String("routes disagree at ", bad))


def test_a_wide_direct_table_numbers_its_rows_on_every_core() raises:
    """The route a range too wide to merge takes on a column long enough to split.

    A direct factorize gives each worker a table and merges them afterwards, and a
    range this wide cannot afford that, so the discovery stays on one thread. The
    pass that numbers the rows does not, and this is the column shape that reaches
    it: past `PARALLEL_ROWS`, and a span past what `DIRECT_MERGE_BYTES` will let
    four workers hold. The two tests above are the same shape at an eighth of the
    height, which is under the threshold, so nothing was running this before.

    The same groups in the same order built far enough apart that no row count
    could accept them pins the second column to the hash, and the two have to come
    out with identical codes, because which route ran is not something a caller
    can see.
    """
    var n = PARALLEL_ROWS + 1021
    var near = Array[DType.int64](n)
    var far = Array[DType.int64](n)
    for i in range(n):
        var value = Int64((i * 7919) % 90000)
        near[i] = value
        far[i] = value * Int64(1 << 20)

    var direct = factorize(near)
    var hashed = factorize(far)
    assert_equal(direct.count(), 90000)
    assert_equal(direct.count(), hashed.count())
    var bad = -1
    for i in range(n):
        if direct.codes[i] != hashed.codes[i]:
            bad = i
            break
    assert_equal(bad, -1, String("routes disagree at ", bad))


def test_a_wide_direct_table_split_across_cores_keeps_nulls_at_zero() raises:
    """Nulls on the route that discovers on one thread and numbers on all of them.

    The two passes read the validity bitmap separately, the first to skip a null
    and the second to write it a zero, so a null is the thing most likely to come
    out differently on this route than on the one that does both at once.
    """
    var n = PARALLEL_ROWS + 1021
    var near = Array[DType.int64](n)
    var far = Array[DType.int64](n)
    for i in range(n):
        var value = Int64((i * 7919) % 90000)
        near[i] = value
        far[i] = value * Int64(1 << 20)
        if i % 7 == 0:
            near.set_null(i)
            far.set_null(i)

    var direct = factorize(near)
    var hashed = factorize(far)
    assert_equal(direct.null_group, 0)
    assert_equal(direct.count(), hashed.count())
    var bad = -1
    for i in range(n):
        if direct.codes[i] != hashed.codes[i]:
            bad = i
            break
    assert_equal(bad, -1, String("routes disagree at ", bad))


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


def test_factorize_dense_takes_a_table_wider_than_the_scan_would() raises:
    """A span past `DIRECT_LIMIT` that one route indexes and the other hashes.

    Eighty thousand slots is over the limit a scan will accept and no wider than
    the column, so this is the case the function exists for. Only five hundred
    values in the span are used, which is the point: the route is chosen from the
    range the caller promised, not from how much of it turns out to be occupied,
    and the answer has to be the one the hash gives either way.
    """
    var span = DIRECT_LIMIT + 15000
    var n = span
    var col = Array[DType.int64](n)
    for i in range(n):
        col[i] = Int64((i * 7) % 500)

    var dense = factorize_dense(col, span)
    var hashed = factorize(col)
    assert_equal(dense.count(), 500)
    assert_equal(dense.count(), hashed.count())
    ref got = dense.codes
    ref want = hashed.codes
    var bad = -1
    for i in range(n):
        if got[i] != want[i]:
            bad = i
            break
    assert_equal(bad, -1, String("routes disagree at ", bad))


def test_factorize_dense_declines_a_table_wider_than_the_column() raises:
    # Values spread over a span far larger than the column, which is the shape a
    # table would be mostly empty for. The values are still inside the span they
    # were promised, so the only thing being checked is that declining the table
    # changes the cost and not the answer.
    var n = 2000
    var col = Array[DType.int64](n)
    for i in range(n):
        col[i] = Int64((i % 300) * 900_000)

    var declined = factorize_dense(col, 300 * 900_000)
    var reference = factorize(col)
    assert_equal(declined.count(), 300)
    ref got = declined.codes
    ref want = reference.codes
    var bad = -1
    for i in range(n):
        if got[i] != want[i]:
            bad = i
            break
    assert_equal(bad, -1, String("declining the table moved a code at ", bad))


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


def test_estimate_groups_on_a_sample_that_stopped_discovering() raises:
    """Both counts equal means the keys ran out well inside the sample, and the
    answer is what was found plus the quarter of headroom for a tail no sample
    sees."""
    assert_equal(_estimate_groups(100, 100, 65536, 10_000_000), 125)


def test_estimate_groups_when_every_row_is_a_new_key() raises:
    """A ratio of two is the model saying the column is larger than the sample
    can bound, and the only honest answer left is the column."""
    assert_equal(_estimate_groups(65536, 32768, 65536, 10_000_000), 10_000_000)


def test_estimate_groups_recovers_a_middling_cardinality() raises:
    """The shape this function exists for, and the one the flat extrapolation it
    replaced was worst at. A hundred thousand keys evenly spread over ten
    million rows put 48074 of themselves in a 65536 row sample and 27941 in the
    first half of it."""
    var got = _estimate_groups(48074, 27941, 65536, 10_000_000)
    assert_true(got > 90_000, String("estimated ", got))
    assert_true(got < 110_000, String("estimated ", got))


def test_estimate_groups_recovers_a_high_cardinality() raises:
    """A million keys over the same ten million rows. Worth checking separately
    because the ratio here is 1.97, close enough to the two that means give up
    that a bisection with the wrong bracket would."""
    var got = _estimate_groups(63435, 32237, 65536, 10_000_000)
    assert_true(got > 850_000, String("estimated ", got))
    assert_true(got < 1_200_000, String("estimated ", got))


def test_estimate_groups_stays_between_what_was_seen_and_the_column() raises:
    """Neither bound is reachable by the model on its own, so both are clamps
    rather than consequences, and a caller sizing anything off this needs them
    to hold."""
    for seen in range(2, 400, 37):
        for half in range(1, seen + 1, 5):
            var got = _estimate_groups(seen, half, 800, 100_000)
            assert_true(got >= seen, String("seen ", seen, " half ", half))
            assert_true(got <= 100_000, String("seen ", seen, " half ", half))


def test_parallel_workers_takes_every_core_on_few_groups() raises:
    """A hundred groups over ten million rows costs a hundred groups per worker
    to merge, which is nothing against the slice they came from, so there is
    nothing to trade and the answer is whatever the machine has."""
    var most = min(worker_count(), 10_000_000 // PARALLEL_MIN_SLICE)
    assert_equal(_parallel_workers(100, 10_000_000), most)


def test_parallel_workers_splits_a_column_of_distinct_keys() raises:
    """This used to be refused, because the merge was the whole column over
    again when every worker's slice was its own group count. The merge runs on
    every core now and the build still divides, so the split wins here too, at
    a hundred and twenty nine milliseconds against a hundred and seventy four
    for ten million distinct integers."""
    var most = min(worker_count(), 10_000_000 // PARALLEL_MIN_SLICE)
    if most < 8:
        return
    assert_true(_parallel_workers(10_000_000, 10_000_000) > 1)
    assert_true(_parallel_workers(5_000_000, 10_000_000) > 1)


def test_parallel_workers_splits_a_middling_column() raises:
    """The shape the whole model is for. A hundred thousand groups over ten
    million rows is worth splitting, and before this it was not split at all,
    because every core was the only alternative to none of them and every core
    on this column merges more than it builds."""
    var most = min(worker_count(), 10_000_000 // PARALLEL_MIN_SLICE)
    if most < 8:
        return
    assert_true(_parallel_workers(100_000, 10_000_000) > 1)


def test_parallel_workers_stops_short_of_the_cores_it_has() raises:
    """Half a million groups is sixteen megabytes of table on its own, so two
    workers already have more of it than the cache holds and the rest would be
    probing memory. The answer has to be below the core count without being
    one, which is the part a count that could only be all of them or none could
    not express."""
    var most = min(worker_count(), 10_000_000 // PARALLEL_MIN_SLICE)
    if most < 8:
        return
    var got = _parallel_workers(500_000, 10_000_000)
    assert_true(got > 1, String("chose ", got, " of ", most))
    assert_true(got < most, String("chose ", got, " of ", most))


def test_parallel_workers_never_slices_below_the_minimum() raises:
    """The worker count is what divides the column, so it has to leave every
    worker a slice worth waking a thread for.

    The range starts at the lower of the two thresholds and runs past the
    higher one, so that it covers every column height either route can offer
    this function. Written against `PARALLEL_ROWS` alone it would have gone
    empty and passed without testing anything the moment that constant moved
    above the end of the range.
    """
    for rows in range(PARALLEL_STRING_ROWS, PARALLEL_ROWS * 2, 1 << 16):
        var got = _parallel_workers(4, rows)
        assert_true(got >= 1, String("rows ", rows, " chose ", got))
        assert_true(
            rows // got >= PARALLEL_MIN_SLICE,
            String("rows ", rows, " chose ", got),
        )


def test_parallel_workers_is_monotone_in_cardinality() raises:
    """Adding groups to a column only ever makes the merge longer, so it can
    only ever want the same worker count or fewer. A model that wobbled here
    would be routing on noise."""
    var last = _parallel_workers(1, 10_000_000)
    var groups = 2
    while groups <= 8_000_000:
        var got = _parallel_workers(groups, 10_000_000)
        assert_true(got <= last, String("groups ", groups, " chose ", got))
        last = got
        groups *= 2


def test_a_chunk_sized_factorize_stays_on_one_thread() raises:
    """A column exactly one chunk tall is below both thresholds, not on one.

    The streaming engine hands an operator one chunk at a time, so a chunk is
    the size a factorize is asked for over and over in a pipeline. When the
    threshold was `MORSEL_ROWS` those two numbers were equal and the gate is
    `>=`, so every one of those factorizes went parallel, and it went parallel
    on `MORSEL_ROWS // PARALLEL_MIN_SLICE` workers, which is the fewest the
    split ever runs on and the worst point on the curve. On the i9-13900K that
    cost 5.0x against not splitting at all.

    This is written as an inequality between the constants rather than as a
    number, because the bug was the relationship and not any one value. Moving
    the chunk size keeps it just as much as moving a threshold does, and both
    thresholds have to hold it, which is the part that would be easy to lose
    the next time one of them is retuned on its own.
    """
    assert_true(
        PARALLEL_ROWS > MORSEL_ROWS,
        String(
            "a factorize of one chunk must stay serial, but the threshold ",
            PARALLEL_ROWS,
            " is not above the chunk size ",
            MORSEL_ROWS,
        ),
    )
    assert_true(
        PARALLEL_STRING_ROWS > MORSEL_ROWS,
        String(
            (
                "a text factorize of one chunk must stay serial, but the"
                " threshold "
            ),
            PARALLEL_STRING_ROWS,
            " is not above the chunk size ",
            MORSEL_ROWS,
        ),
    )


def test_the_split_thresholds_leave_room_for_more_than_a_few_slices() raises:
    """A column that splits at all splits enough ways to be worth it.

    A threshold and the minimum slice are two halves of one rule, and the
    quotient between them is the worker count the very first column over the
    line gets. Four of them measurably lost on every route tested, so the floor
    is eight. This is a floor and not the measured value: the numeric threshold
    is far above it, because where that one belongs was settled by measuring the
    crossover rather than by counting slices.
    """
    assert_true(
        PARALLEL_ROWS // PARALLEL_MIN_SLICE >= 8,
        String(
            "the shortest column that splits gets ",
            PARALLEL_ROWS // PARALLEL_MIN_SLICE,
            " workers, which is too few to pay the merge back",
        ),
    )
    assert_true(
        PARALLEL_STRING_ROWS // PARALLEL_MIN_SLICE >= 8,
        String(
            "the shortest text column that splits gets ",
            PARALLEL_STRING_ROWS // PARALLEL_MIN_SLICE,
            " workers, which is too few to pay the merge back",
        ),
    )


def test_text_splits_earlier_than_numbers() raises:
    """The two thresholds are ordered, and the order is not arbitrary.

    What decides where a split starts to pay is the ratio between what a row
    costs on one thread and what the merge costs, and a string key costs around
    thirty times what an int64 key costs. So text has to cross first. If these
    two are ever made equal again, one of them is in the wrong place, which is
    exactly the state this pair of constants was in before they were split.
    """
    assert_true(
        PARALLEL_STRING_ROWS < PARALLEL_ROWS,
        String(
            "text at ",
            PARALLEL_STRING_ROWS,
            " must split earlier than numbers at ",
            PARALLEL_ROWS,
        ),
    )


def _split_scan_column(n: Int, at: Int, far: Int64) -> Array[DType.int64]:
    """A column of two values with a third planted at one row.

    Args:
        n: The number of rows.
        at: The row the planted value goes on, or -1 for none.
        far: The planted value.

    Returns:
        The column, with no nulls.
    """
    var col = Array[DType.int64](n)
    for i in range(n):
        col[i] = Int64(i % 7)
    if at >= 0:
        col[at] = far
    return col^


def test_the_range_scan_agrees_across_the_row_it_splits_at() raises:
    """`direct_plan` reads `PLAN_PREFIX_ROWS` where it stands and hands the rest
    to the cores, and the point of the split is that where a value sits does not
    change what comes out.

    So the same wide value is planted on the last row the prefix reads, on the
    first row it does not, on a morsel boundary past that and on the last row of
    the column, and every one of those has to come back with the eight groups
    the column has. All of them sit past the seventh row, so the seven narrow
    values are still numbered by where they first appear and the planted one is
    still the group nobody had seen. The narrow column with nothing planted is
    the other half of it: the scan has to get all the way to the end and still
    say the range is seven.
    """
    var rows = PLAN_PREFIX_ROWS * 5
    var far = Int64(DIRECT_LIMIT) * 4

    var narrow = factorize(_split_scan_column(rows, -1, far))
    assert_equal(narrow.count(), 7)

    var places = [
        7,
        PLAN_PREFIX_ROWS - 1,
        PLAN_PREFIX_ROWS,
        PLAN_PREFIX_ROWS + MORSEL_ROWS,
        rows - 1,
    ]
    for k in range(len(places)):
        var at = places[k]
        var planted = factorize(_split_scan_column(rows, at, far))
        assert_equal(planted.count(), 8)
        assert_equal(Int(planted.codes[at]), 7)
        for i in range(rows):
            if i != at:
                assert_equal(Int(planted.codes[i]), i % 7)


def test_the_range_scan_skips_nulls_on_both_sides_of_the_split() raises:
    """A null holds no value, so it is not in the range whichever side of the
    boundary the scan splits at it sits on.

    The values here sit far from zero and a null's slot holds zero, so a scan
    that read one would come back with a range four times what a table is worth
    instead of a range of seven. Both nulls are placed where a scan that got the
    split wrong would miss them: one on the last row the prefix reads and one on
    the last row of the column.
    """
    var rows = PLAN_PREFIX_ROWS * 5
    var base = Int64(DIRECT_LIMIT) * 4
    var col = Array[DType.int64](rows)
    for i in range(rows):
        col[i] = base + Int64(i % 7)
    col.set_null(PLAN_PREFIX_ROWS - 1)
    col.set_null(rows - 1)

    var plan = direct_plan(col, rows // DIRECT_SHARE)
    assert_equal(plan.span, 7)
    assert_equal(Int(plan.base), Int(base))

    var found = factorize(col)
    assert_equal(found.count(), 8)
    assert_equal(found.null_group, 0)
    assert_equal(Int(found.codes[PLAN_PREFIX_ROWS - 1]), 0)
    assert_equal(Int(found.codes[rows - 1]), 0)
    assert_equal(Int(found.codes[0]), 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
