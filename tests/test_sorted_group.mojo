"""Tests for the group by that walks a sorted key instead of hashing it.

There is only one thing worth testing about a second route to an answer the
library already has a route to, and it is that the two agree. So most of what is
here is a pair: the same rows grouped once with the sortedness flag set and once
without it, asserted equal. A route that gave a different answer would pass any
test written only against itself.

The other half is the refusals. `sorted_ordinals` hands back nothing rather than
a wrong answer for a column it cannot walk, and each of those cases is here with
the ordinary route's answer beside it, because a refusal that silently changed
the result would be worse than no route at all.

The parallel split gets its own tests because the walk is the one kernel in this
file whose correctness depends on where the cuts land. A worker handed the middle
of the column reads the row before its own first row to decide whether a group
opens there, and its ordinals start at a base the counting pass worked out, so a
column long enough to be split several ways with a run crossing every cut is the
shape that catches an off by one in either.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import from_list
from firepanda.array.chunked import Sortedness
from firepanda.array.strings import StringBuilder
from firepanda.frame.frame import PROVE_SORTED_ROWS, DataFrame
from firepanda.frame.groupby import AggSpec
from firepanda.frame.series import Series
from firepanda.hash.grouping import group_ordinals
from firepanda.hash.sorted import PARALLEL_RUN_ROWS, sorted_ordinals
from firepanda.kernel.group import AggKind


def ints(name: String, values: List[Scalar[DType.int64]]) raises -> Series:
    """Builds an int64 series with no nulls.

    Args:
        name: The column name.
        values: The values.

    Returns:
        The series.

    Raises:
        Error: If the series cannot be built.
    """
    return Series(name, AnyArray(from_list(values)))


def assert_routes_agree(var frame: DataFrame, key: String, what: String) raises:
    """Groups one frame both ways and asserts the two answers are the same one.

    The key column carries the sortedness flag, which is what sends it down the
    walk; `group_ordinals` is called directly for the answer the flag would
    otherwise not have changed. Both are read in place rather than copied into
    lists, because these tests run past the parallel split and every pass over
    half a million rows here is an interpreted one.

    Failures are reported as the first row or group that disagreed rather than
    as an assertion each. A quarter of a million assertions, each building a
    String to say which row it was about, costs more than everything else in
    this file together and says nothing the first one does not.

    Args:
        frame: The frame. Its key column must be sorted and free of nulls.
        key: Which column is the key.
        what: What is being compared, for the failure message.

    Raises:
        Error: If the column is missing, cannot be grouped, or the two routes
            give different answers.
    """
    var at = frame.index_of(key)
    var by_hash = group_ordinals(frame.column_refs(), [at], frame.rows)

    var by_walk = sorted_ordinals(frame.columns[at].only())
    assert_true(Bool(by_walk), what + ": the walk took the column")
    ref got = by_walk.value()
    assert_equal(got.groups, by_hash.groups, what + ": the same group count")

    var row = -1
    for i in range(frame.rows):
        if got.codes[i] != by_hash.codes[i]:
            row = i
            break
    assert_equal(row, -1, what + ": the first row that disagreed")

    var group = -1
    for g in range(got.groups):
        if got.rows_at[g] != by_hash.rows_at[g]:
            group = g
            break
    assert_equal(group, -1, what + ": the first group starting elsewhere")


def test_a_sorted_key_walks_to_the_same_ordinals() raises:
    var frame = DataFrame.from_series(
        [
            ints("k", [Int64(1), 1, 1, 4, 4, 9]),
            ints("v", [Int64(1), 2, 3, 4, 5, 6]),
        ]
    )
    frame.columns[0].mark_sorted(Sortedness.ASCENDING)
    assert_routes_agree(frame^, "k", "three runs")


def test_a_descending_key_walks_too() raises:
    # Equal values are adjacent either way round, which is all the walk needs,
    # and the runs come out in the order they appear either way round too.
    var frame = DataFrame.from_series(
        [
            ints("k", [Int64(9), 4, 4, 1, 1, 1]),
            ints("v", [Int64(1), 2, 3, 4, 5, 6]),
        ]
    )
    frame.columns[0].mark_sorted(Sortedness.DESCENDING)
    assert_routes_agree(frame^, "k", "three runs, largest first")


def test_every_row_its_own_group_and_every_row_the_same_group() raises:
    var distinct = DataFrame.from_series([ints("k", [Int64(1), 2, 3, 4])])
    distinct.columns[0].mark_sorted(Sortedness.ASCENDING)
    assert_routes_agree(distinct^, "k", "four groups of one")

    var flat = DataFrame.from_series([ints("k", [Int64(7), 7, 7, 7])])
    flat.columns[0].mark_sorted(Sortedness.CONSTANT)
    assert_routes_agree(flat^, "k", "one group of four")


def test_one_row_and_no_rows() raises:
    var single = DataFrame.from_series([ints("k", [Int64(3)])])
    single.columns[0].mark_sorted(Sortedness.CONSTANT)
    var got = sorted_ordinals(single.columns[0].only())
    assert_true(Bool(got), "one row is a column like any other")
    assert_equal(got.value().groups, 1, "and it is one group")
    assert_equal(got.value().rows_at[0], 0, "starting on row zero")

    var empty = sorted_ordinals(
        AnyArray(from_list(List[Scalar[DType.int64]]()))
    )
    assert_true(Bool(empty), "no rows is not a refusal")
    assert_equal(empty.value().groups, 0, "and it is no groups")


def test_a_run_across_every_worker_boundary() raises:
    """The shape an off by one in the split shows up on.

    Every group is longer than the stretch a worker gets on any machine this
    runs on, so a run crosses every cut and no worker's first row opens a group
    except worker zero's. That is what makes the base ordinal the counting pass
    computed the only thing keeping the two halves of the column in step.
    """
    comptime rows = 4 * PARALLEL_RUN_ROWS
    comptime per = rows // 3
    var values = List[Scalar[DType.int64]](capacity=rows)
    for i in range(rows):
        values.append(Int64(i // per))
    var frame = DataFrame.from_series([ints("k", values)])
    frame.columns[0].mark_sorted(Sortedness.ASCENDING)
    assert_routes_agree(frame^, "k", "three very long runs")


def test_a_group_boundary_on_every_row_past_the_split() raises:
    """The other end of the same question, with every row opening a group.

    Here every worker's first row does open a group, so the base the counting
    pass hands each worker has to be exactly the number of rows before it, and a
    prefix sum that was off by one anywhere would show up as a shifted block
    rather than as a single wrong row.
    """
    comptime rows = 4 * PARALLEL_RUN_ROWS + 7
    var values = List[Scalar[DType.int64]](capacity=rows)
    for i in range(rows):
        values.append(Int64(i))
    var frame = DataFrame.from_series([ints("k", values)])
    frame.columns[0].mark_sorted(Sortedness.ASCENDING)
    assert_routes_agree(frame^, "k", "every row its own group")


def test_runs_of_two_past_the_split() raises:
    # Half the cuts land inside a pair and half between two, which is the case
    # neither of the two above covers.
    comptime rows = 4 * PARALLEL_RUN_ROWS
    var values = List[Scalar[DType.int64]](capacity=rows)
    for i in range(rows):
        values.append(Int64(i // 2))
    var frame = DataFrame.from_series([ints("k", values)])
    frame.columns[0].mark_sorted(Sortedness.ASCENDING)
    assert_routes_agree(frame^, "k", "runs of two")


def test_a_null_is_refused() raises:
    var col = from_list([Int64(1), 1, 4])
    col.set_null(1)
    assert_false(
        Bool(sorted_ordinals(AnyArray(col^))),
        "the flag says nothing about where the nulls went",
    )


def test_a_text_column_is_refused() raises:
    # Not because a sorted text key is a strange thing to have, but because it
    # would match the uint8 arm of the dispatch and group on the first byte of
    # each view. It is a refusal until the walk learns to compare views.
    var text = StringBuilder()
    text.append(String("aa").as_bytes())
    text.append(String("ab").as_bytes())
    assert_false(
        Bool(sorted_ordinals(AnyArray(text^.finish()))),
        "text takes the ordinary route",
    )


def test_the_flag_is_what_chooses_and_a_group_by_agrees_either_way() raises:
    """The whole point, asserted at the level a user sees.

    The same rows, the same aggregate, once with the flag and once without, and
    the two frames have to be the same frame. This is the assertion that would
    fail if `_grouping` picked the walk for a column the walk gets wrong.
    """
    var values = List[Scalar[DType.int64]]()
    var payload = List[Scalar[DType.int64]]()
    for i in range(1000):
        values.append(Int64(i // 7))
        payload.append(Int64(i))

    var specs = List[AggSpec]()
    specs.append(AggSpec("v", AggKind.SUM))

    var plain = DataFrame.from_series([ints("k", values), ints("v", payload)])
    assert_false(plain.columns[0].order.is_known(), "nothing known to start")
    var by_hash = plain.group_by(["k"], specs)

    var flagged = DataFrame.from_series([ints("k", values), ints("v", payload)])
    flagged.columns[0].mark_sorted(Sortedness.ASCENDING)
    var by_walk = flagged.group_by(["k"], specs)

    assert_equal(len(by_walk), len(by_hash), "the same number of groups")
    var walked_keys = by_walk.column("k").as_typed[DType.int64]()
    var hashed_keys = by_hash.column("k").as_typed[DType.int64]()
    var walked_sums = by_walk.column("v_sum").as_typed[DType.int64]()
    var hashed_sums = by_hash.column("v_sum").as_typed[DType.int64]()
    var wrong = -1
    for g in range(len(by_walk)):
        if walked_keys[g] != hashed_keys[g] or walked_sums[g] != hashed_sums[g]:
            wrong = g
            break
    assert_equal(wrong, -1, "the first group that disagreed")


def assert_sums_agree(var left: DataFrame, var right: DataFrame) raises:
    """Asserts two grouped frames hold the same keys and the same sums.

    The two are built by the two routes over the same rows, so they have the
    same shape and the comparison is row by row. The first row that disagrees is
    reported rather than one assertion a row, for the reason
    `assert_routes_agree` gives above.

    Args:
        left: One answer.
        right: The other.

    Raises:
        Error: If the two disagree anywhere.
    """
    assert_equal(len(left), len(right), "the same number of groups")
    var left_keys = left.column("k").as_typed[DType.int64]()
    var right_keys = right.column("k").as_typed[DType.int64]()
    var left_sums = left.column("v_sum").as_typed[DType.int64]()
    var right_sums = right.column("v_sum").as_typed[DType.int64]()
    var wrong = -1
    for g in range(len(left)):
        if left_keys[g] != right_keys[g] or left_sums[g] != right_sums[g]:
            wrong = g
            break
    assert_equal(wrong, -1, "the first group that disagreed")


def assert_both_routes_group(keys: List[Scalar[DType.int64]]) raises:
    """Groups one key column twice, once as it arrives and once forced to hash.

    Marking a copy of the key `UNORDERED` is what forces the other route: the
    flag is then known, so nothing is scanned, and it is not an order, so the
    walk is not offered the column. Both frames are otherwise the same rows.

    Args:
        keys: The key column's values. The payload is derived from the row
            number so that no two groups can accidentally hold the same sum.

    Raises:
        Error: If the two routes disagree, or if the unmarked frame arrives
            carrying a flag, which would mean the test was not testing what it
            says.
    """
    var payload = List[Scalar[DType.int64]](capacity=len(keys))
    for i in range(len(keys)):
        payload.append(Int64(i % 101))

    var specs = List[AggSpec]()
    specs.append(AggSpec("v", AggKind.SUM))

    var plain = DataFrame.from_series([ints("k", keys), ints("v", payload)])
    assert_false(plain.columns[0].order.is_known(), "nothing known to start")
    var as_it_arrives = plain.group_by(["k"], specs)
    # A group by borrows the frame, so there is nowhere to write what the scan
    # found and the column comes back knowing no more than it did. That is the
    # cost of proving from a borrow and it is asserted here so that a later
    # change which does cache the answer has to say so.
    assert_false(plain.columns[0].order.is_known(), "and nothing was written")

    var forced = DataFrame.from_series([ints("k", keys), ints("v", payload)])
    forced.columns[0].mark_sorted(Sortedness.UNORDERED)
    var by_hash = forced.group_by(["k"], specs)

    assert_sums_agree(as_it_arrives^, by_hash^)


def test_an_unmarked_sorted_key_is_found_rather_than_hashed() raises:
    """A key already in order but carrying no flag takes the walk anyway.

    The flag is set by a sort and by a reader and by almost nothing else, so a
    primary key read from a file, a column a join emitted in key order, and
    anything filtered from those are all in order and none of them are marked.
    Those frames used to pay for a hash table they did not need.

    What is asserted is the answer rather than the route, because the route is
    not visible from here and an answer that matches the hash table over sixty
    five thousand rows and ten thousand groups is not something a broken walk
    produces by accident.

    Raises:
        Error: On the first group the two routes disagreed about.
    """
    comptime ROWS = PROVE_SORTED_ROWS + 3
    var keys = List[Scalar[DType.int64]](capacity=ROWS)
    for i in range(ROWS):
        keys.append(Int64(i // 7))
    assert_both_routes_group(keys)


def test_a_key_that_breaks_order_at_the_very_last_row() raises:
    """The case the scan costs the most and buys the least.

    Every row but the last is in order, so `is_sorted` reads the whole column
    before answering no and the group by pays for the scan and then hashes
    anyway. It is the worst case by construction and the answer still has to be
    right, which is the half of it that matters here.

    Raises:
        Error: On the first group the two routes disagreed about.
    """
    comptime ROWS = PROVE_SORTED_ROWS + 3
    var keys = List[Scalar[DType.int64]](capacity=ROWS)
    for i in range(ROWS):
        keys.append(Int64(i // 7))
    keys[ROWS - 1] = Int64(0)
    assert_both_routes_group(keys)


def test_a_key_in_no_order_at_all_past_the_threshold() raises:
    """The common case, where the scan stops within a few rows.

    A key in no order is what the scan is cheap on, since `is_sorted` returns at
    the first pair out of order in each direction. Nothing about that is visible
    from the answer, so what this covers is that the fallthrough is still the
    fallthrough on a frame big enough to be scanned.

    Raises:
        Error: On the first group the two routes disagreed about.
    """
    comptime ROWS = PROVE_SORTED_ROWS + 3
    var keys = List[Scalar[DType.int64]](capacity=ROWS)
    var state = UInt64(0x9E3779B97F4A7C15)
    for _ in range(ROWS):
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        keys.append(Int64(state % 10000))
    assert_both_routes_group(keys)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
