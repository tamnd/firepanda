"""Tests for the skewed key generator.

Nothing here checks a number the generator produces, because the numbers are
arbitrary and the seed is what makes them repeatable. What is checked is the two
properties that make the column worth generating at all: the keys sit inside a
small set of networks, so the top of a key carries almost no entropy, and the
frequencies have a head, so a few keys take a large share of the rows. A
generator that lost either of those would still look random and would stop being
the thing `benchmarks/probe_lengths.mojo` is measuring.

The tolerances are wide on purpose. A share of one half drawn a hundred thousand
times lands within a fraction of a percent of a half, so a test that allowed one
percent would be testing the arithmetic rather than the generator, and a test
that allowed forty would pass on a generator that had stopped drawing from the
head at all. These allow enough that a reseed cannot fail them and little enough
that a broken draw cannot pass.
"""

from std.testing import TestSuite, assert_equal, assert_true

from firepanda.array.array import Array
from firepanda.hash import factorize
from firepanda.testing import (
    DEFAULT_NETWORKS,
    HOST_BITS,
    Skewed,
    skewed_int64,
)

comptime HOST_MASK = (1 << HOST_BITS) - 1


def _networks_of(col: Array[DType.int64]) -> Int:
    """Counts the distinct networks a column's addresses came from.

    Args:
        col: The addresses.

    Returns:
        How many distinct values the top of the key takes.
    """
    var seen = List[Int]()
    for i in range(len(col)):
        var net = Int(col[i]) >> HOST_BITS
        var known = False
        for k in range(len(seen)):
            if seen[k] == net:
                known = True
        if not known:
            seen.append(net)
    return len(seen)


def test_every_address_comes_from_one_of_the_networks() raises:
    """The structure the generator exists for.

    A uniform 32-bit key has entropy in every bit. An address does not, and the
    reason is this: the network part takes a few values across the whole column
    however many rows there are.
    """
    var col = skewed_int64(20000)
    assert_equal(
        _networks_of(col),
        DEFAULT_NETWORKS,
        "twenty thousand draws find all eight networks and no ninth",
    )


def test_the_networks_are_not_the_first_eight_numbers() raises:
    """Drawn rather than counted off, so a weak hash is not flattered.

    Networks numbered zero through seven would leave the top of every key at
    almost zero, which is a friendlier key than a real one and would make a hash
    that ignores the high bits look fine.
    """
    var gen = Skewed()
    var largest = 0
    for i in range(len(gen.nets)):
        var net = Int(gen.nets[i]) >> HOST_BITS
        if net > largest:
            largest = net
    assert_true(largest > DEFAULT_NETWORKS, "the networks are spread out")


def test_the_head_takes_the_share_it_was_given() raises:
    """The frequencies, which is the other half of the shape.

    A head host is numbered below `heads` and a tail host is not, so counting
    which side of that line the rows fall on is counting how many came from the
    head.
    """
    var rows = 100000
    var col = skewed_int64(rows, 64, 0.8)
    var from_head = 0
    for i in range(rows):
        if Int(col[i]) & HOST_MASK < 64:
            from_head += 1
    var share = Float64(from_head) / Float64(rows)
    assert_true(share > 0.75, String("share ", share, " is not far below 0.8"))
    assert_true(share < 0.85, String("share ", share, " is not far above 0.8"))


def test_the_head_is_ordered_by_frequency() raises:
    """Zipf and not a uniform draw over the head.

    A head that took half the rows and spread them evenly over its hosts would
    pass the share test above and would not be skewed at all. The first host is
    the most frequent one, and it should be well clear of the last.
    """
    var rows = 200000
    var col = skewed_int64(rows, 64, 1.0)
    var first = 0
    var last = 0
    for i in range(rows):
        var host = Int(col[i]) & HOST_MASK
        if host == 0:
            first += 1
        elif host == 63:
            last += 1
    assert_true(first > last * 4, String(first, " against ", last))


def test_a_column_with_no_head_is_all_distinct() raises:
    """The nearly unique case, which is what q31 actually groups on.

    The tail is a counter, so below the host space every row is its own group.
    """
    var rows = 50000
    var col = skewed_int64(rows, 0, 0.0)
    assert_equal(len(factorize(col).firsts), rows, "every row is its own group")


def test_the_same_seed_replays_the_same_column() raises:
    """The property every generator in `firepanda/testing` has.

    A measurement that cannot be replayed is not a measurement, and a probe
    length distribution nobody else can reproduce is an anecdote.
    """
    var left = skewed_int64(5000, 32, 0.5, 4, 99)
    var right = skewed_int64(5000, 32, 0.5, 4, 99)
    for i in range(5000):
        assert_equal(Int(left[i]), Int(right[i]), "same seed, same row")


def test_a_different_seed_gives_a_different_column() raises:
    """And the seed is doing something, which the test above cannot tell."""
    var left = skewed_int64(5000, 32, 0.5, 4, 99)
    var right = skewed_int64(5000, 32, 0.5, 4, 100)
    var differences = 0
    for i in range(5000):
        if Int(left[i]) != Int(right[i]):
            differences += 1
    assert_true(differences > 4000, String(differences, " rows differ"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
