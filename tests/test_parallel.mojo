"""Tests for the one primitive the library has for using more than one core.

`parallel_for` is thirty lines and three of its properties are load bearing, so
each of them has a test rather than a comment.

Every index runs, including the ones after an index that failed. A body that
raises for one input must not leave the outputs of the others unwritten, because
the caller of a failing parallel read is going to look at the error message and
nothing else, and a half finished buffer that is still alive when the task group
goes away is a use after free rather than a wrong answer.

A failure is reported rather than aborted. The version of this in `max` calls
`abort`, and a CSV file with a bad value in it deserves a message.

A count of one runs inline, which is what keeps a small read off the runtime.
That one is not directly observable, so what is tested is the consequence: the
error from a single index is the error the body raised, unchanged.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.exec import parallel_for, worker_count


def test_every_index_runs() raises:
    var seen = List[Int](length=64, fill=0)

    def mark(i: Int) raises {mut seen, imm}:
        seen[i] = i + 1

    parallel_for(mark, 64)
    for i in range(64):
        assert_equal(seen[i], i + 1)


def test_a_count_of_zero_does_nothing() raises:
    var runs = List[Int](length=1, fill=0)

    def never(i: Int) raises {mut runs, imm}:
        runs[0] += 1

    parallel_for(never, 0)
    assert_equal(runs[0], 0)


def test_a_count_of_one_still_runs_the_body() raises:
    var seen = List[Int](length=1, fill=0)

    def mark(i: Int) raises {mut seen, imm}:
        seen[0] = 7

    parallel_for(mark, 1)
    assert_equal(seen[0], 7)


def test_a_failure_is_raised_rather_than_aborted() raises:
    def fail(i: Int) raises:
        raise Error("the body said no")

    with assert_raises(contains="the body said no"):
        parallel_for(fail, 4)


def test_a_single_index_failure_is_raised_unchanged() raises:
    """The inline path has its own return, so it needs its own test."""

    def fail(i: Int) raises:
        raise Error("the body said no")

    with assert_raises(contains="the body said no"):
        parallel_for(fail, 1)


def test_the_indices_that_did_not_fail_still_ran() raises:
    """The group is waited on whether or not anything failed.

    Returning early on the first failure would leave tasks running with borrowed
    state, and the state here is a list that goes away when this function does.
    """
    var seen = List[Int](length=32, fill=0)

    def half(i: Int) raises {mut seen, imm}:
        if i % 2 == 0:
            raise Error(String("index ", i, " is even"))
        seen[i] = i

    with assert_raises(contains="is even"):
        parallel_for(half, 32)
    for i in range(1, 32, 2):
        assert_equal(seen[i], i)


def test_the_first_failure_by_index_is_the_one_reported() raises:
    """Which task finished first is not something a message should depend on."""

    def fail(i: Int) raises:
        if i >= 3:
            raise Error(String("index ", i))

    with assert_raises(contains="index 3"):
        parallel_for(fail, 16)


def test_there_is_at_least_one_worker() raises:
    assert_true(worker_count() >= 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
