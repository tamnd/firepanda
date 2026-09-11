"""The window reductions that ask about the order of the values, not the sum.

`median`, `quantile` and `rank` are the three window reductions that are not
folds. There is no number you can carry from one window to the next and correct
for the rows that changed, because the answer is a position in the sorted
window and one row arriving can move that position past any number of values.
So the five reductions in `window.mojo` and the five in `spread.mojo` share a
shape that these three do not, and they are here instead.

They are one piece of work for the same reason. The structure that makes a
window ordered is the whole cost, and once there is one a median is a selection
at the middle, a quantile is a selection at a fraction with five opinions about
what to do between two neighbours, and a rank is a count of how many values one
row beats. Building it three times would be three times the work for one
answer.

## Why this is not a skip list

pandas keeps a skip list per column and inserts and deletes through it, which
costs the logarithm of the width per row and is the right structure for a
stream. This is not a stream. The whole column is in memory before the first
window is formed, which means its values can be ranked once and a window can
then be a count of how many values it holds of each rank.

So the structure is a Fenwick tree over the ranks of the column's own values.
Adding a row is one update, dropping one is another, selecting the k-th
smallest value is one descent of the tree, and counting how many values a row
beats is one prefix sum. Every one of those costs the logarithm of the number
of distinct values in the column, which is worse than a skip list for a two
wide window and better for a wide one, and the pass is n log n whatever the
width where a sort per window is n times the width times the logarithm of the
width.

It also has no rebuild and no error bound, which is the one respect in which
these three are easier than the five that came before them. A count of values
is exact, it stays exact however long the pass is, and it has nothing to lose
when a large value leaves the window.

## Why the window is never cleared

The edges are non decreasing, so the rows to drop are the ones the last window
held and this one does not, and the rows to add are the ones this window holds
and the last one did not. Written that way it covers the case where two
consecutive windows do not overlap at all, which a step wider than the window
produces on every row, without ever walking the tree to zero it. Clearing a
tree costs the number of distinct values in the whole column, and doing that
per row would make a stepped window dearer than sorting each window from
scratch.

## Where these disagree with pandas

In the same place everything else in this section does. pandas replaces every
infinity in the column with a missing value before any window is formed, so a
window holding an infinity is a shorter window to pandas and its median is the
median of the rows either side. Here an infinity is a value, it sorts above
every finite one, and a window holding it has a median.

The median of an even window is the mean of the two middle values, and there
are two ways to write that. Their sum over two is what pandas does and it
overflows when both of them are near the top of the float range, so a window
holding the largest finite double twice has a median of infinity there. Half of
each does not overflow, and it can land one bit away from the sum over two on
ordinary data because the two roundings fall in different places.

So it is written as the sum over two, which agrees with pandas everywhere the
sum is a number, and the halves are used only when that sum came out infinite
and both values were finite. That is the one window where the two differ, it is
the same disagreement the running total has and for the same reason, and the
answer given is the one that is right.
"""

from std.math import floor, isinf

from firepanda.bitmap.bitmap import Bitmap

comptime BETWEEN_LINEAR = 0
"""Weight the two neighbours by how far the position sits between them."""

comptime BETWEEN_LOWER = 1
"""Take the neighbour below."""

comptime BETWEEN_HIGHER = 2
"""Take the neighbour above."""

comptime BETWEEN_MIDPOINT = 3
"""Take the mean of the two neighbours, wherever between them it sits."""

comptime BETWEEN_NEAREST = 4
"""Take the nearer neighbour, and the even one when the position is halfway.

Rounding a half to the even side is numpy's rule and pandas inherits it, so a
quantile of an eighth over five values takes the first of them and a quantile
of three eighths over the same five takes the third. Rounding a half up instead
agrees with pandas on one of those two and not on the other, which is the kind
of difference that shows up on one frame in a suite and nowhere else."""

comptime TIED_AVERAGE = 0
"""Give every tied value the mean of the positions they cover."""

comptime TIED_MIN = 1
"""Give every tied value the first of the positions they cover."""

comptime TIED_MAX = 2
"""Give every tied value the last of the positions they cover."""


def between_named(name: StringSlice) raises -> Int:
    """Turns pandas' interpolation word into a code.

    Args:
        name: One of `linear`, `lower`, `higher`, `midpoint` and `nearest`.

    Returns:
        The matching code.

    Raises:
        Error: If the name is not one of the five. The message names the word
            it was given, because the five are not guessable from each other.
    """
    if name == "linear":
        return BETWEEN_LINEAR
    if name == "lower":
        return BETWEEN_LOWER
    if name == "higher":
        return BETWEEN_HIGHER
    if name == "midpoint":
        return BETWEEN_MIDPOINT
    if name == "nearest":
        return BETWEEN_NEAREST
    raise Error(
        String(
            "window: interpolation '",
            name,
            "' is not one of linear, lower, higher, midpoint and nearest",
        )
    )


def tied_named(name: StringSlice) raises -> Int:
    """Turns pandas' rank method word into a code.

    `Series.rank` takes five of these and the window form takes three, which is
    pandas' own restriction rather than one made here. `dense` and `first` need
    an ordering over the whole column and a window does not have one.

    Args:
        name: One of `average`, `min` and `max`.

    Returns:
        The matching code.

    Raises:
        Error: If the name is not one of the three.
    """
    if name == "average":
        return TIED_AVERAGE
    if name == "min":
        return TIED_MIN
    if name == "max":
        return TIED_MAX
    raise Error(
        String(
            "window: rank method '",
            name,
            "' is not one of average, min and max",
        )
    )


@fieldwise_init
struct Ranks(Movable):
    """The column's values sorted once, and each row's place in that order.

    Built before the first window and read by every one of them. Two rows
    holding the same value get the same rank, which is what lets a tie be
    counted rather than searched for, and a missing row gets no rank at all
    because no window will ever ask about it.
    """

    var values: List[Float64]
    """The distinct values of the column, increasing."""

    var of_row: List[Int32]
    """Each row's rank, or minus one where the row holds no value.

    An int32 rather than an int, because a column tall enough to need more is
    taller than anything else in this package handles and halving this array
    halves what the three order statistics cost over what the ten folds cost.
    """


def ranked[
    origin: ImmOrigin
](
    src: Pointer[Scalar[DType.float64], origin],
    present: Bitmap,
    rows: Int,
) -> Ranks:
    """Ranks a column's values, so a window can hold counts instead of values.

    The sort is over a copy of the values that are present rather than over the
    column, because the column is what the answers are read out of and it stays
    in the order the caller gave. Sorting a copy costs one float per present
    row and gives back both halves of what is needed here, which are the
    distinct values in order for a selection to answer with and somewhere to
    look each row up.

    Args:
        src: The values, already float64.
        present: Which rows hold a value.
        rows: How tall the column is.

    Parameters:
        origin: The origin of the values.

    Returns:
        The distinct values in order and each row's place among them.
    """
    var order = List[Float64](capacity=rows)
    for i in range(rows):
        if present.get(i):
            order.append(src.unsafe_offset(i).unsafe_load())
    if len(order) > 1:
        sort(
            Span[Float64, origin_of(order)](
                unsafe_ptr=order.unsafe_ptr(), length=len(order)
            )
        )
    # Collapsed in place, so a column of one repeated value costs one entry and
    # not one per row. The tree below is as wide as this list, which is what
    # makes selecting out of a low cardinality column cheap.
    var distinct = 0
    for i in range(len(order)):
        if distinct == 0 or order[i] != order[distinct - 1]:
            order[distinct] = order[i]
            distinct += 1
    order.resize(distinct, 0.0)

    var places = List[Int32](length=rows, fill=-1)
    for i in range(rows):
        if present.get(i):
            var value = src.unsafe_offset(i).unsafe_load()
            places[i] = Int32(_place(order, distinct, value))
    return Ranks(order^, places^)


def _place(values: List[Float64], distinct: Int, value: Float64) -> Int:
    """Finds a value's rank among the distinct values.

    Args:
        values: The distinct values, increasing.
        distinct: How many of them there are.
        value: The value to place, which is one of them.

    Returns:
        Its index in the list.
    """
    var low = 0
    var high = distinct
    while low < high:
        var half = low + (high - low) // 2
        if values[half] < value:
            low = half + 1
        else:
            high = half
    return low


struct Ordered(Movable):
    """How many values of each rank the window holds, with the sums to select.

    A Fenwick tree, which is an array where entry `i` holds the total of a run
    of entries ending at `i` whose length is the lowest set bit of `i`. That
    arrangement is what makes both directions cheap at once: a prefix total
    reads one entry per set bit of the position and an update touches one entry
    per position that covers it, and both are the number of bits in the width
    of the tree.

    The count at each rank is kept separately as well. A tie needs to know how
    many values sit at exactly one rank, which the tree answers as the
    difference of two prefix totals and a plain array answers in one load, and
    the array costs one integer per distinct value in a structure that already
    costs one.
    """

    var sums: List[Int32]
    """The tree, one longer than the number of ranks and indexed from one."""

    var tally: List[Int32]
    """How many values the window holds at each rank, indexed from nought."""

    var width: Int
    """How many ranks there are."""

    var stride: Int
    """The largest power of two that is not more than the width.

    Held rather than recomputed, because the descent in `select` starts from it
    and `select` runs once or twice per answered row."""

    def __init__(out self, width: Int):
        """Builds an empty tree over a given number of ranks.

        Args:
            width: How many distinct values the column has.
        """
        self.sums = List[Int32](length=width + 1, fill=0)
        self.tally = List[Int32](length=max(width, 1), fill=0)
        self.width = width
        self.stride = 1
        while self.stride * 2 <= width:
            self.stride *= 2

    def add(mut self, rank: Int):
        """Puts one value of a given rank into the window.

        Args:
            rank: The value's rank.
        """
        self.tally[rank] += 1
        var at = rank + 1
        while at <= self.width:
            self.sums[at] += 1
            at += at & -at

    def drop(mut self, rank: Int):
        """Takes one value of a given rank out of the window.

        Args:
            rank: The value's rank.
        """
        self.tally[rank] -= 1
        var at = rank + 1
        while at <= self.width:
            self.sums[at] -= 1
            at += at & -at

    def below(self, rank: Int) -> Int:
        """Counts the values in the window that are smaller than a rank.

        Args:
            rank: The rank to count below.

        Returns:
            How many values the window holds at a lower rank.
        """
        var total = 0
        var at = rank
        while at > 0:
            total += Int(self.sums[at])
            at -= at & -at
        return total

    def same(self, rank: Int) -> Int:
        """Counts the values in the window that sit at exactly one rank.

        Args:
            rank: The rank to count at.

        Returns:
            How many values the window holds there.
        """
        return Int(self.tally[rank])

    def select(self, wanted: Int) -> Int:
        """Finds the rank of the k-th smallest value in the window.

        One descent from the top of the tree rather than a search that calls
        `below` repeatedly, which would be the square of the logarithm instead
        of the logarithm. At each step the entry being considered is the total
        of a run that starts where the descent has reached, so it can be taken
        whole or skipped.

        Args:
            wanted: Which value, counting from nought.

        Returns:
            Its rank.
        """
        var at = 0
        var remaining = wanted + 1
        var step = self.stride
        while step > 0:
            var ahead = at + step
            if ahead <= self.width and Int(self.sums[ahead]) < remaining:
                at = ahead
                remaining -= Int(self.sums[ahead])
            step //= 2
        return at

    def value(self, ranks: Ranks, wanted: Int) -> Float64:
        """Reads the k-th smallest value of the window out.

        Args:
            ranks: The column's distinct values.
            wanted: Which value, counting from nought.

        Returns:
            The value.
        """
        return ranks.values[self.select(wanted)]


def middle(lower: Float64, upper: Float64) -> Float64:
    """Averages the two middle values of an even window.

    Args:
        lower: The smaller one.
        upper: The larger one.

    Returns:
        Their mean, computed the way pandas computes it unless that overflows.
    """
    var plain = (lower + upper) / 2.0
    if isinf(plain) and not (isinf(lower) or isinf(upper)):
        # Both of them are finite and their sum is not, which is the one window
        # where pandas answers an infinity and there is a real number to give.
        return lower * 0.5 + upper * 0.5
    return plain


def halved(tree: Ordered, ranks: Ranks, found: Int) -> Float64:
    """Reads the middle of the window out.

    Args:
        tree: The window's counts.
        ranks: The column's distinct values.
        found: How many values the window holds, which is one or more.

    Returns:
        The middle value of an odd window, or the mean of the two middle values
        of an even one.
    """
    if found % 2 == 1:
        return tree.value(ranks, (found - 1) // 2)
    return middle(
        tree.value(ranks, found // 2 - 1), tree.value(ranks, found // 2)
    )


def picked(
    tree: Ordered, ranks: Ranks, found: Int, fraction: Float64, between: Int
) -> Float64:
    """Reads the value at a fraction of the way through the window.

    The position is on the values rather than between them, so a fraction of
    nought is the smallest value exactly and a fraction of one is the largest
    exactly, whichever of the five rules was asked for. Where the position
    lands on a value all five agree and the rule never comes up, which is why
    the exact case is settled first.

    Args:
        tree: The window's counts.
        ranks: The column's distinct values.
        found: How many values the window holds, which is one or more.
        fraction: Where to read, from nought to one.
        between: Which of the five rules to use between two values.

    Returns:
        The value.
    """
    var position = fraction * Float64(found - 1)
    var lower = Int(floor(position))
    var part = position - Float64(lower)
    if part == 0.0 or lower + 1 >= found:
        return tree.value(ranks, lower)
    if between == BETWEEN_LOWER:
        return tree.value(ranks, lower)
    if between == BETWEEN_HIGHER:
        return tree.value(ranks, lower + 1)
    if between == BETWEEN_NEAREST:
        if part < 0.5:
            return tree.value(ranks, lower)
        if part > 0.5:
            return tree.value(ranks, lower + 1)
        return tree.value(ranks, lower if lower % 2 == 0 else lower + 1)
    var under = tree.value(ranks, lower)
    var over = tree.value(ranks, lower + 1)
    if between == BETWEEN_MIDPOINT:
        return middle(under, over)
    return under + part * (over - under)


def placed(
    tree: Ordered,
    rank: Int,
    found: Int,
    tied: Int,
    ascending: Bool,
    pct: Bool,
) -> Float64:
    """Says where one value sits among the values of the window.

    Args:
        tree: The window's counts.
        rank: The rank of the value being placed.
        found: How many values the window holds, which is one or more.
        tied: Which of the three tie rules to use.
        ascending: Count from the smallest rather than from the largest.
        pct: Divide by how many values the window holds.

    Returns:
        The position, counting from one, or the fraction of the window it is.
    """
    var under = tree.below(rank)
    var equal = tree.same(rank)
    var ahead = under if ascending else found - under - equal
    var place = Float64(ahead + 1)
    if tied == TIED_MAX:
        place = Float64(ahead + equal)
    elif tied == TIED_AVERAGE:
        # The mean of the positions the tied values cover, which is the first
        # of them plus the last over two, written without the division.
        place = Float64(ahead) + Float64(equal + 1) * 0.5
    return place / Float64(found) if pct else place
