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

## The third and fourth moments are the same argument twice more

A skewness is the third central moment over the three halves power of the
second, and a kurtosis is the fourth over the square of the second. So both
need the same state this file already carries plus two more numbers, and both
need them carried for the same reason: the mean moves, so a cube or a fourth
power folded in earlier was measured against a mean that has since changed.

The recurrence that updates all four together is standard and the removal is it
run backwards, so everything above about the removal applies unchanged and is
worse rather than better. A fourth power of a large value against a small
spread cancels more digits than a second power does. On a column near ten to
the eight the fourth power is near ten to the thirty two and the answer is
about a spread near one, and a double does not hold both. `Moments` therefore
carries a bound for each of the two moments and rebuilds when either has
outgrown its own answer.

Those bounds also count something the variance's bound does not, and it is the
thing that turned out to matter. A deviation is a value take a mean, and when
the mean is much larger than the deviation the subtraction throws away the
digits they agree in before any cube is taken. The variance's bound counts only
the rounding of its own additions, which on a column of four thousand values
let a carried kurtosis drift into the ninth digit while pandas held the
eleventh. Measuring the loss in the deviation as well, and holding the rebuild
to a tighter figure than the variance uses, puts the skewness in the thirteenth
digit and the kurtosis in the twelfth, which is ahead of pandas on the same
column, and costs a rebuild on a few rows in a hundred at the narrowest widths
and on one in a thousand at the wide ones. `TRUSTED` carries the figures.

pandas carries the raw sums of the value, its square, its cube and its fourth
power instead, and reconstructs the central moments at the end. That is the
arrangement the top of this file rejects for the variance, one power further
along, and it was measured against this one on two columns. On four thousand
values spread either side of nought with three far away rows in them, this file
is thirty four times closer to the right answer on a four wide skewness and
level on a four wide kurtosis, and pandas is closer on a three wide skewness,
which is the one window narrow enough that there is almost nothing to carry. On
two hundred values near ten to the eight stepping a thousandth at a time, which
is the case a power sum is worst at, this file is between seven and forty seven
times closer across the three. Both lose digits there and neither can avoid it,
because the deviation itself only has five left before anything is cubed.

## What pandas states, and what it refuses

A window whose values are all the same answers a skewness of zero and a
kurtosis of minus three, for every width. Zero is right, because a constant is
symmetric. Minus three is not a value arithmetic supports: a degenerate
distribution has no kurtosis, the standardized fourth moment of a real one is
never below one, and minus three is what falls out of subtracting the excess
offset from a ratio taken to be nought. It is copied here rather than derived,
because a caller comparing the two libraries on a column of one repeated value
would read a NaN as a bug in this one.

The refusal is the other way round. When the population variance of a window is
at or below ten to the minus fourteen, pandas answers NaN for both. Measured
against pandas 3.0.5, a window whose variance is 1.001e-14 answers and one
whose variance is 9.999999999999998e-15 does not. It is absolute rather than
relative, so whether pandas will tell you the skewness of your readings depends
on the units
you wrote them in: the same four measurements in kilometres and in millimetres
differ by twelve orders of magnitude in variance and pandas answers one and
refuses the other. A one pass power sum cannot compute a skewness down there, so
the threshold is doing real work for pandas. A carried central moment with a
rebuild can, so this answers the number and the difference is registered rather
than copied. A variance of exactly zero with values that are not all the same is
still a NaN in both, because there the ratio has no numerator either.
"""

from std.math import isinf, isnan, nan, sqrt

comptime SPREAD_VAR = 0
"""Form code for the variance itself."""

comptime SPREAD_STD = 1
"""Form code for the square root of the variance."""

comptime SPREAD_SEM = 2
"""Form code for the deviation divided by the root of the count."""

comptime MOMENT_SKEW = 0
"""Form code for the third moment over the second to the three halves."""

comptime MOMENT_KURT = 1
"""Form code for the fourth moment over the square of the second, in the excess
form, which is the one pandas answers."""

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


comptime TRUSTED = 1e-12
"""How much of a moment the carried state is allowed to have lost before it is
thrown away and rebuilt. Four figures tighter than `CREDIBLE`, which the spread
uses, and the reason is measured rather than argued. A spread is a square and a
kurtosis is a fourth power, so the same lost digit in a deviation comes out
twice as far along, and at `CREDIBLE` a rolling kurtosis over four thousand
values drifted into the ninth digit where pandas held the eleventh. At this
figure the skewness holds the thirteenth and the kurtosis the twelfth, both
ahead of pandas, and the rebuild fires on eight rows in a hundred at a width of
three, on four at a width of four, and on one in a thousand from a width of
thirty two up. The cost is bounded whatever the column does, because a rebuild
reads the window once and the row is then answered whether the rebuilt state is
settled or not, so the worst a pass can cost is what recomputing every window
from scratch would."""


def _slipped(value: Float64, centre: Float64, away: Float64) -> Float64:
    """Says how much of a deviation the subtraction that made it threw away.

    A deviation is one value take another, and when the two are close together
    the answer keeps only the digits they differ in. A value of eight hundred
    and a mean of seven hundred and ninety leave a deviation of ten, which a
    double holds to thirteen digits rather than sixteen, and a cube of it to
    eleven. That loss is invisible to a bound that only counts the rounding of
    its own addition, which is why it is measured here and handed to the bounds
    before they are widened.

    Args:
        value: The row.
        centre: The mean it was measured against.
        away: The deviation that came out.

    Returns:
        The error in the deviation as a fraction of the deviation, and nought
        when the row sits exactly on the mean, because a deviation of nought
        contributes nothing for the error to be a fraction of.
    """
    if away == 0.0:
        return 0.0
    return EPSILON * (abs(value) + abs(centre)) / abs(away)


struct Moments(ImplicitlyCopyable, Movable):
    """A running third and fourth central moment, over a running spread.

    The spread is held rather than repeated, so the mean the cube and the fourth
    power are measured around is the same mean the variance is measured around
    and the two cannot drift apart. Each update reads the spread's state from
    before the row was folded in, which is why the order inside `add` and `drop`
    matters and is written down there.
    """

    var spread: Spread
    """The count, the mean and the sum of squared deviations, with its own bound
    and its own run length and its own count of infinities."""

    var third: Float64
    """The sum of the cubed deviations from the mean."""

    var fourth: Float64
    """The sum of the fourth powers of the deviations from the mean."""

    var third_bound: Float64
    """An upper bound on the error the updates could have put into the third
    moment."""

    var fourth_bound: Float64
    """An upper bound on the error the updates could have put into the fourth
    moment. The two are kept apart rather than shared because the two moments do
    not lose their digits at the same row: a skewness at a width of three drifts
    where a kurtosis at the same width is not even answered, and one bound
    standing in for the other was measured to let that drift through."""

    def __init__(out self):
        """Starts a moment state with nothing in it."""
        self.spread = Spread()
        self.third = 0.0
        self.fourth = 0.0
        self.third_bound = 0.0
        self.fourth_bound = 0.0

    def add(mut self, value: Float64):
        """Folds one row into the moments.

        The deviation, the count and the two moments all have to be read from
        before the spread is updated, because the recurrence is written in terms
        of the state the row is arriving at rather than the state it produced.

        Args:
            value: The row's value, which is never a NaN.
        """
        if isinf(value):
            self.spread.add(value)
            return
        var older = Float64(self.spread.count)
        var second = self.spread.squares
        var cubed = self.third
        var centre = self.spread.mean - self.spread.lost
        var away = value - centre
        self.spread.add(value)
        var size = Float64(self.spread.count)
        var share = away / size
        var square = share * share
        var moved = away * away * older / size
        var lift = (
            moved * square * (size * size - 3.0 * size + 3.0)
            + 6.0 * square * second
            - 4.0 * share * cubed
        )
        var tilt = moved * share * (size - 2.0) - 3.0 * share * second
        self.fourth = self.fourth + lift
        self.third = cubed + tilt
        self._widen(_slipped(value, centre, away), tilt, lift)

    def _widen(mut self, slip: Float64, tilt: Float64, lift: Float64):
        """Widens both bounds by what the update it was given could have lost.

        Args:
            slip: How far the deviation could be out, as a fraction of itself.
            tilt: What was added to or taken off the third moment.
            lift: What was added to or taken off the fourth moment.
        """
        self.third_bound = (
            self.third_bound
            + EPSILON * (abs(tilt) + abs(self.third))
            + 3.0 * slip * abs(tilt)
        )
        self.fourth_bound = (
            self.fourth_bound
            + EPSILON * (abs(lift) + abs(self.fourth))
            + 4.0 * slip * abs(lift)
        )

    def drop(mut self, value: Float64):
        """Takes one row back out of the moments.

        Here the order is the other way round. The deviation the recurrence was
        written with is the one from the mean the window had before this row
        arrived, and what is available is the mean it has now, so the first is
        recovered from the second by scaling by the two counts. The third moment
        has to be undone before the fourth, because the fourth's term reads the
        third from before the row was added.

        Args:
            value: The row's value.
        """
        if isinf(value):
            self.spread.drop(value)
            return
        var size = Float64(self.spread.count)
        var centre = self.spread.mean - self.spread.lost
        var short = value - centre
        self.spread.drop(value)
        if self.spread.count == 0:
            self.third = 0.0
            self.fourth = 0.0
            self.third_bound = 0.0
            self.fourth_bound = 0.0
            return
        var older = Float64(self.spread.count)
        var away = size * short / older
        var share = away / size
        var square = share * share
        var moved = away * away * older / size
        var second = self.spread.squares
        var tilt = moved * share * (size - 2.0) - 3.0 * share * second
        self.third = self.third - tilt
        var lift = (
            moved * square * (size * size - 3.0 * size + 3.0)
            + 6.0 * square * second
            - 4.0 * share * self.third
        )
        self.fourth = self.fourth - lift
        self._widen(_slipped(value, centre, short), tilt, lift)

    def settled(self) -> Bool:
        """Says whether the carried state is still worth carrying.

        Returns:
            False if the spread underneath has given up, and False once either
            moment has stopped being a number, and False once the fourth has
            gone below zero, and False once either moment has lost more of
            itself than `TRUSTED` allows to the error the updates could have
            made. A symmetric window has a third moment of nought and that is
            an answer rather than a failure, so the third's bound is not
            measured against the third itself. It is measured against the root
            of the second times the fourth, which no third moment can exceed
            and which only goes to nought when the window holds one value
            repeated.
        """
        if not self.spread.settled():
            return False
        if isnan(self.fourth) or isinf(self.fourth):
            return False
        if isnan(self.third) or isinf(self.third):
            return False
        if self.fourth < 0.0:
            return False
        if self.fourth_bound > TRUSTED * self.fourth:
            return False
        return self.third_bound <= TRUSTED * sqrt(
            self.spread.squares * self.fourth
        )

    def answer(self, found: Int, form: Int) -> Float64:
        """Returns the skewness or the kurtosis of the window.

        Args:
            found: How many rows of the window hold a value, counting the
                infinities.
            form: One of the two form codes at the top of the file.

        Returns:
            A NaN when the window holds fewer values than the moment needs, or
            holds an infinity, or has no spread to standardize by, and the
            skewness or the excess kurtosis otherwise.
        """
        var needed = 3 if form == MOMENT_SKEW else 4
        if found < needed:
            return nan[DType.float64]()
        if self.spread.highs > 0 or self.spread.lows > 0:
            return nan[DType.float64]()
        var size = Float64(found)
        if self.spread.runs >= found:
            # What pandas answers for a window of one repeated value. Zero is
            # right and minus three is copied, for the reasons at the top.
            return 0.0 if form == MOMENT_SKEW else -3.0
        var second = self.spread.squares if self.spread.squares > 0.0 else 0.0
        var spread = second / size
        if spread == 0.0:
            # The values are not all the same and their squared deviations came
            # to nought anyway, which happens when they differ by less than the
            # square root of the smallest double. There is no ratio to take.
            return nan[DType.float64]()
        if form == MOMENT_SKEW:
            var cubed = self.third / size
            return (
                sqrt(size * (size - 1.0))
                * cubed
                / ((size - 2.0) * spread * sqrt(spread))
            )
        var quartic = self.fourth / size
        return (
            (size * size - 1.0) * quartic / (spread * spread)
            - 3.0 * (size - 1.0) * (size - 1.0)
        ) / ((size - 2.0) * (size - 3.0))
