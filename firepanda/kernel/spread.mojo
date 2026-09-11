"""A carried variance is a different argument from a carried total.

A total is one number and the row that leaves is subtracted from it. A variance
cannot work that way, because the mean moves when a row arrives or leaves and
every squared deviation already folded in was measured against the old mean. So
this file holds a state rather than a number, and the whole of it is about one
question, which is when that state stops being worth carrying.

## Why not the sum of squares

The one loop version accumulates the sum and the sum of squares and subtracts at
the end. `group.mojo` already says at length why that is not good enough for a
whole column, and a window makes it worse rather than better: the window is
narrower than the column, so the variance being recovered is smaller, while the
sum of squares it is recovered from is the same size. On a pair of rows holding
ten to the sixteen and ten to the sixteen plus one it answers zero where the
answer is a half.

## Why not the two pass form either

`group.mojo` takes two passes, subtracts each group's own mean and corrects for
the centre it used. That is the accurate answer and it is not available here at
any reasonable cost, because the second pass would run once per window rather
than once per column and a window of five hundred rows would cost five hundred
times the column. Everybody who uses a window reduction is using it because it
is one pass, so one pass is the constraint and the arithmetic has to fit inside
it.

## What it does instead

Welford's method, which carries the mean and the sum of squared deviations from
it and updates both when a row arrives, with the removal being the same update
run backwards. The mean is carried with a compensation term for the same reason
the running total in `window.mojo` carries one, which is that a row is
subtracted again later and the error of the subtraction does not cancel the
error of the addition.

Running an update backwards does not undo it, and the place it goes wrong is
worth naming, because it is not the usual slow drift. A window holding a large
value and several small ones has a large sum of squared deviations, almost all
of it belonging to the large value. When that value leaves, what is subtracted
is nearly the whole of what is there, and the small number left over is the
difference of two large ones. Every digit of it can be wrong. On a column
holding ten to the eight followed by one, two and three, the window over one,
two and three has a variance of one, and carrying the state gives three
quarters.

So the state carries a fourth number, which is a bound on how much error the
additions and the subtractions could have introduced so far. Each update adds
the rounding error it could have made, which is the size of the term it folded
in plus the size of the answer it folded that term into, both times the machine
epsilon. When that bound grows past a hundred millionth of the answer the
carried state is no longer worth carrying and the caller is told to rebuild the
window from its own rows. On ordinary columns the bound grows by a few epsilons
a row and the rebuild fires once in twenty thousand or so, which costs nothing.
On the column above it fires on the rows where it is needed and the answer is
right. This is the same shape as the overflow guard on the running total, with a
real bound in place of a test for infinity, because a variance loses its digits
long before it reaches one.

## The two exact answers

A window whose values are all the same has a variance of exactly zero, and a
rounding error above or below zero is very visible next to a zero. So the state
counts how many of the values it has been handed most recently were the same
value, and a window whose count reaches its own height answers zero without
reading the carried number at all. pandas does this too and for the same reason.
A window holding one value has no spread either, and answers zero when the
degrees of freedom leave it a divisor, which is the other exact case.

## The infinities

The variance of a set holding an infinity is undefined rather than large: the
mean is an infinity, the deviations are an infinity minus an infinity, and there
is no number there. So the infinities are counted rather than folded in, exactly
as the running total counts them, and a window holding any of them answers NaN.
That keeps the finite state clean, which matters more here than it does for a
total, because one infinity folded into a sum of squared deviations turns it to
NaN and no later subtraction brings it back.

pandas answers something else, and not because it disagrees. Its window layer
replaces every infinity with NaN before the kernel sees the column, so an
infinity in a window is a missing value to pandas and the window is reduced over
the rows either side of it.
"""

from std.math import isinf, isnan, nan, sqrt

comptime SPREAD_VAR = 0
"""Form code for the variance itself."""

comptime SPREAD_STD = 1
"""Form code for the square root of the variance."""

comptime SPREAD_SEM = 2
"""Form code for the deviation divided by the root of the count."""

comptime EPSILON = 2.220446049250313e-16
"""The distance from one to the next double, which is the relative error one
rounding can introduce."""

comptime CREDIBLE = 1e-8
"""How much of the answer the accumulated error bound is allowed to be before
the window is rebuilt from its rows. Eight digits is far more than any caller
reads and far less than the digits a double holds, so a rebuild happens long
before the answer is visibly wrong and long after it would be pointless."""


struct Spread(ImplicitlyCopyable, Movable):
    """A running mean and sum of squared deviations, with a bound on its error.

    See the top of the file for why each of the six numbers is here. The short
    version is that the first two are Welford's state, the third keeps the low
    bits of the mean, the fourth is what makes the removal safe, the fifth and
    sixth are the two exact answers, and the last two are the infinities that
    must not be folded in.
    """

    var count: Int
    """How many finite values the window holds."""

    var mean: Float64
    """The mean of those values, near enough that the deviations from it are
    small."""

    var squares: Float64
    """The sum of the squared deviations from that mean."""

    var lost: Float64
    """The low bits the mean's additions have dropped so far."""

    var bound: Float64
    """An upper bound on the error the updates could have put into the sum of
    squared deviations."""

    var runs: Int
    """How many of the values handed over most recently were the same value."""

    var last: Float64
    """The value handed over most recently, which the run length is counted
    against."""

    var highs: Int
    """How many positive infinities the window holds."""

    var lows: Int
    """How many negative infinities the window holds."""

    def __init__(out self):
        """Starts a spread with nothing in it."""
        self.count = 0
        self.mean = 0.0
        self.squares = 0.0
        self.lost = 0.0
        self.bound = 0.0
        self.runs = 0
        self.last = nan[DType.float64]()
        self.highs = 0
        self.lows = 0

    def add(mut self, value: Float64):
        """Folds one row into the spread.

        Args:
            value: The row's value, which is never a NaN because a NaN is not a
                row that holds a value.
        """
        if value == self.last:
            self.runs += 1
        else:
            self.runs = 1
        self.last = value
        if isinf(value):
            if value > 0:
                self.highs += 1
            else:
                self.lows += 1
            return
        self.count += 1
        # The mean before this row, read back through the compensation so that
        # the deviation below is measured from where the mean actually was.
        var before = self.mean - self.lost
        var y = value - self.lost
        var moved = y - self.mean
        self.lost = moved + self.mean - y
        self.mean = self.mean + moved / Float64(self.count)
        var term = (value - before) * (value - self.mean)
        self.squares = self.squares + term
        self.bound = self.bound + EPSILON * (abs(term) + abs(self.squares))

    def drop(mut self, value: Float64):
        """Takes one row back out of the spread.

        The run length is not shortened, and does not need to be. The values in
        the window are always the ones handed over most recently, so a run that
        reaches the height of the window covers the whole of it however long ago
        it started.

        Args:
            value: The row's value.
        """
        if isinf(value):
            if value > 0:
                self.highs -= 1
            else:
                self.lows -= 1
            return
        self.count -= 1
        if self.count == 0:
            self.mean = 0.0
            self.squares = 0.0
            self.lost = 0.0
            self.bound = 0.0
            return
        var before = self.mean - self.lost
        var y = value - self.lost
        var moved = y - self.mean
        self.lost = moved + self.mean - y
        self.mean = self.mean - moved / Float64(self.count)
        var term = (value - before) * (value - self.mean)
        self.squares = self.squares - term
        self.bound = self.bound + EPSILON * (abs(term) + abs(self.squares))

    def settled(self) -> Bool:
        """Says whether the carried state is still worth carrying.

        Returns:
            False once the sum of squared deviations has gone below zero, or
            stopped being a number, or lost more than the eighth digit to the
            error the updates could have made. Any of those means the caller has
            to rebuild from the window's own rows.
        """
        if isnan(self.squares) or isinf(self.squares) or isnan(self.mean):
            return False
        if self.squares < 0.0:
            return False
        return self.bound <= CREDIBLE * self.squares

    def answer(self, found: Int, ddof: Int, form: Int) -> Float64:
        """Returns the spread of the window in one of its three forms.

        Args:
            found: How many rows of the window hold a value, counting the
                infinities, which is what the degrees of freedom are taken from
                and what the standard error divides by.
            ddof: Subtracted from that count to give the divisor.
            form: One of the three form codes at the top of the file.

        Returns:
            A NaN when the count leaves no divisor or the window holds an
            infinity, and the variance, the deviation or the standard error
            otherwise.
        """
        if found < 1 or found <= ddof:
            return nan[DType.float64]()
        if self.highs > 0 or self.lows > 0:
            return nan[DType.float64]()
        var value: Float64
        if found == 1 or self.runs >= found:
            value = 0.0
        else:
            # A sum of squared deviations is not negative, so one that is has
            # been rounded past zero rather than measured there. `group.mojo`
            # takes the same line on the same subtraction.
            var total = self.squares if self.squares > 0.0 else 0.0
            value = total / Float64(found - ddof)
        if form == SPREAD_VAR:
            return value
        var root = sqrt(value)
        if form == SPREAD_STD:
            return root
        return root / sqrt(Float64(found))
