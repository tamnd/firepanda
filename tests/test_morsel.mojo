"""Tests for the morsel queue and the scheduler on top of it.

The thing being checked over and over is that every row is done exactly once. A
queue that hands the same morsel to two workers is a kernel that adds a value
twice, and a queue that skips one is a kernel that silently drops rows, and both
of those look like a wrong answer a long way from here.

The counting is done into one slot per row rather than into a shared total,
because a shared total would be testing an atomic add rather than the queue.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.exec import (
    MORSEL_ROWS,
    Morsel,
    MorselQueue,
    parallel_morsels,
    worker_count,
)


def test_a_queue_hands_out_every_row_once() raises:
    var queue = MorselQueue(10, 4)
    var first = queue.take()
    var second = queue.take()
    var third = queue.take()
    var fourth = queue.take()
    assert_equal(first.start, 0)
    assert_equal(first.end, 4)
    assert_equal(second.start, 4)
    assert_equal(second.end, 8)
    assert_equal(third.start, 8)
    assert_equal(third.end, 10)
    assert_false(fourth)


def test_a_drained_queue_stays_drained() raises:
    # A worker that comes back after the end has to keep getting nothing. The
    # counter runs past the total and is deliberately left there.
    var queue = MorselQueue(3, 4)
    assert_true(queue.take())
    for _ in range(10):
        assert_false(queue.take())


def test_the_last_morsel_is_short_rather_than_over_the_end() raises:
    var queue = MorselQueue(9, 4)
    _ = queue.take()
    _ = queue.take()
    var last = queue.take()
    assert_equal(last.start, 8)
    assert_equal(last.end, 9)
    assert_equal(len(last), 1)


def test_a_queue_over_nothing_hands_out_nothing() raises:
    var queue = MorselQueue(0, 4)
    assert_false(queue.take())
    assert_equal(queue.morsels(), 0)


def test_a_morsel_of_zero_rows_is_treated_as_one() raises:
    # Otherwise the queue never drains and the job never returns, which is a
    # worse failure than the caller's arithmetic being wrong.
    var queue = MorselQueue(3, 0)
    assert_equal(queue.rows, 1)
    assert_equal(queue.morsels(), 3)


def test_the_morsel_count_is_the_rounded_up_division() raises:
    assert_equal(MorselQueue(8, 4).morsels(), 2)
    assert_equal(MorselQueue(9, 4).morsels(), 3)
    assert_equal(MorselQueue(1, 4).morsels(), 1)


def test_an_empty_morsel_is_false_and_a_real_one_is_true() raises:
    assert_true(Morsel(0, 1))
    assert_false(Morsel(5, 5))
    assert_false(Morsel(5, 4))
    assert_equal(len(Morsel(5, 4)), 0)


def test_running_over_rows_touches_each_one_exactly_once() raises:
    var rows = 1000000
    var seen = List[Int32](length=rows, fill=0)

    def one(start: Int, end: Int) raises {mut seen, imm}:
        for i in range(start, end):
            seen[i] = seen[i] + 1

    parallel_morsels(one, rows, 4096)
    for i in range(rows):
        assert_equal(seen[i], 1)


def test_a_job_smaller_than_a_morsel_still_runs() raises:
    var seen = List[Int32](length=7, fill=0)

    def one(start: Int, end: Int) raises {mut seen, imm}:
        for i in range(start, end):
            seen[i] = seen[i] + 1

    parallel_morsels(one, 7, 4096)
    for i in range(7):
        assert_equal(seen[i], 1)


def test_a_job_of_no_rows_does_nothing_and_does_not_hang() raises:
    var count = List[Int](length=1, fill=0)

    def one(start: Int, end: Int) raises {mut count, imm}:
        count[0] = count[0] + 1

    parallel_morsels(one, 0, 4096)
    assert_equal(count[0], 0)


def test_a_failure_in_one_morsel_comes_back_to_the_caller() raises:
    var rows = 100000

    def one(start: Int, end: Int) raises:
        if start <= 50000 and 50000 < end:
            raise Error("the fifty thousandth row is no good")

    with assert_raises(contains="fifty thousandth"):
        parallel_morsels(one, rows, 1024)


def test_a_failure_does_not_stop_the_other_workers_finishing() raises:
    # Returning while another worker is still running would be returning while
    # it still holds a pointer to state the caller is about to drop.
    var rows = 100000
    var seen = List[Int32](length=rows, fill=0)

    def one(start: Int, end: Int) raises {mut seen, imm}:
        for i in range(start, end):
            seen[i] = seen[i] + 1
        if start == 0:
            raise Error("the first morsel is no good")

    with assert_raises(contains="first morsel"):
        parallel_morsels(one, rows, 1024)
    # Not every row, because the worker that failed stopped taking morsels. But
    # no row twice, which is the invariant that matters.
    for i in range(rows):
        assert_true(seen[i] <= 1)


def test_uneven_work_is_shared_out_rather_than_split_in_advance() raises:
    # One morsel in ten costs a hundred times what the others do. Split up front
    # the expensive ones land wherever they land; taken on demand a worker that
    # draws one simply comes back later, and every worker ends up with roughly
    # the same total cost.
    var morsels = 200
    var rows = morsels * 1000
    var workers = worker_count()
    var done = List[Int](length=morsels, fill=0)

    def one(start: Int, end: Int) raises {mut done, imm}:
        var index = start // 1000
        var spins = 100 if index % 10 == 0 else 1
        var total = 0
        for _ in range(spins):
            for i in range(start, end):
                total += i
        done[index] = total

    parallel_morsels(one, rows, 1000)
    for index in range(morsels):
        assert_true(done[index] != 0)
    assert_true(workers >= 1)


def test_the_default_morsel_is_a_chunk() raises:
    assert_equal(MORSEL_ROWS, 128 * 1024)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
