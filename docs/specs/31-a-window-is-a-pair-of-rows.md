# 31. A window is a pair of rows

pandas has three window types, `Rolling`, `Expanding` and `ExponentialMovingWindow`, and puts twenty six reductions on each of the first two. The conformance board had every one of them at zero, which made `windows` the largest section in the suite sitting at nothing, and the obvious way to start on it is to pick a reduction and write it. That is the wrong first move, because it answers the smaller half of the question. What `rolling` and `expanding` are for is not the arithmetic. It is deciding which rows go together, and the arithmetic afterwards does not care which of the two asked.

So this document is about the pair of row numbers, and the reductions are the short part at the end.

## 1. The five parameters are one question

`s.rolling(window, min_periods, center, closed, step)` reads like five independent knobs and is not. Four of the five decide the same thing, which is the first row and the row after the last of the window that row `i` reduces, and the fifth decides which `i` are asked at all.

The base window that ends at row `i` runs from `i + 1 - window` up to `i + 1`, not including the last. Then:

`center` moves the far end forward by `(window - 1) // 2` before the near end is derived from it. That last clause is the part nobody guesses. A centred window of an even width is not symmetric, and it takes one row more from the left than from the right, so `rolling(4, center=True)` at row two covers rows nought through three. The floor division is where that comes from and pandas has the same one.

`closed` moves one or both ends by one row. `right` is the default and is the base window above. `left` moves both ends back, so the window at row `i` is the window at row `i - 1` under the default. `both` moves only the near end back, so it holds one row more than the width asks for. `neither` moves only the far end back, so it holds one row less, and a `neither` window whose `min_periods` is its full width therefore never answers anything. That last one looks like a bug the first time it is measured and is what pandas does.

`min_periods` does not move the window at all. It decides whether the window that was computed is allowed to answer, by comparing how many values it holds against the number given. Its default is where the two window types genuinely differ: a rolling window that is not told needs all of its rows, and an expanding window that is not told needs one. That is why `rolling(5).sum()` has four holes at the top and `expanding().sum()` has none.

`step` does not touch the window either. It decides which rows are asked, so the answer is `ceil(rows / step)` tall and carries the labels of the rows it kept. It is a property of the output rather than of the reduction, which is why it is the one parameter that changes the shape of what comes back.

Clipping happens after all of that and only against the column. The near end is derived from the unclipped far end, so a centred window at the bottom of the column loses rows rather than sliding back up the column to keep its width.

`Shape.edges` in `firepanda/kernel/window.mojo` is those five sentences as ten lines of arithmetic, and it is the only place in the library that knows any of it. Everything else takes a pair of row numbers.

## 2. An expanding window is a rolling one with no near end

Which means there is no second struct, no second resolver and no second loop. `expanding_shape(min_periods, rows)` builds a `Shape` whose width is the height of the column, whose near end therefore clips to row zero for every row, and whose other three parameters are the values that mean nothing. Everything after that is the rolling path.

The alternative is two implementations of the same reduction that agree today, and the way that fails is not that one of them is wrong on the day it is written. It is that the next parameter to arrive gets added to one of them.

The width is resolved against the column rather than at construction, and that is deliberate on the Python side too: `ExpandingMixin` holds `None` for the width and not the height. In pandas a window object holds a reference to a column that can grow before the reduction runs, so filling the width in at construction would be answering a question that has not been asked yet.

## 3. Why the reductions are carried and what that costs

Summing each window separately is the width times the height. Carrying a total, adding the row that arrives and subtracting the row that leaves is the height, and that is the whole reason anybody uses these rather than a loop. The edges are non decreasing in `i` for every combination of the five parameters above, including `step`, which is what makes the carrying legal at all.

Carrying costs something, and pandas pays it visibly.

Its running total is one number, so an infinity that enters the window makes it infinite, and subtracting that infinity when the row leaves gives a NaN rather than giving the total back. Every window after that reads NaN until the window empties on its own. Measured on pandas 3.0.5, `pd.Series([1, 2, inf, 3, 4, 5, 6]).rolling(3).sum()` is `[nan, nan, nan, nan, nan, 12, 15]`, where the last two are right and the two before them are not: the window at row three is `[2, inf, 3]` and its sum is an infinity, not a missing value.

The extremes lose the same two values for a different reason. pandas seeds its running maximum with negative infinity and reads a result equal to that seed as an empty window, so `rolling(3).max()` over a column holding a positive infinity answers NaN there rather than the infinity.

This does not copy either. The two infinities are counted rather than summed, the total carries only the finite rows, and the answer is assembled from the three at the end. A window holding one sign sums to that infinity, a window holding both sums to a NaN, and both survive the infinity leaving the window again. The extremes answer the infinity, because an infinity in a column is an ordinary value.

That leaves the finite rows overflowing on their own, which is rare and real. The guard is to rebuild that one window from its rows, which costs the width of the window on the rows where it fires and nothing anywhere else.

I could not reproduce pandas' exact reset rule, because `pandas/_libs/window/aggregations` ships compiled and its `.pyx` is not in the wheel. Reverse engineering it would have produced a kernel whose specification is another kernel's rounding behaviour, so what is here is the mathematically correct thing and the differences are registered as divergences rather than chased.

## 4. Subtracting a row is not adding one backwards

A running sum that adds and subtracts loses precision in a way a running sum that only adds does not, because the error of the subtraction does not cancel the error of the addition that put the row in. A large value beside small ones is where it shows: after `1e16` and `1.0` have both been added, the `1.0` has been rounded away, and subtracting the `1e16` gives zero rather than one.

So the finite total is compensated, and the compensation is kept beside the total and added back at the end rather than being folded into the next row on the way in. That distinction is the difference between this working and not working. Folding it in is the textbook Kahan loop, and it rounds the incoming row against the compensation before adding it, so on this column the low bit is lost twice and is gone by the time the large row leaves. Keeping it beside the total, which is Neumaier's variant, means the bits are still there to add back. `test_the_low_bits_survive_a_row_leaving_the_window` is that column, and it fails under the textbook loop.

## 5. The extremes are not summable and are not recomputed either

Knowing the maximum of a window and the value that just left does not give the maximum of what remains, so there is nothing to subtract. Recomputing is the width times the height and gives back everything section 3 bought.

The standing answer is a deque of row numbers whose values decrease from front to back. A row arriving evicts every row behind it that it beats, because a row that is both older and smaller can never be the answer again, and the front of the deque is dropped once it falls out of the near end. Every row is pushed once and popped once, so the pass is the height of the column no matter how wide the window is. The deque is an integer array the height of the column with a head and a tail index, rather than a list that grows, because its maximum size is known before the first row is read.

## 6. What a missing row is, and the one place `count` reads differently

Whatever `present_bitmap_any` says, which on a float column means a NaN counts as holding nothing. That is already the library rule and it is also pandas', so none of the reductions here has to think about NaN at all: a missing row is not added to the total, not pushed into the deque and not counted toward `min_periods`.

`count` is the exception, and the difference is not a decision made here. pandas computes it as a rolling sum over the presence indicator, which is a column with nothing missing in it, so what `min_periods` is tested against is how many rows the window covers rather than how many hold a value. Measured on `[1, nan, 3, nan, 5, 6]`, `rolling(3).count()` is `[nan, nan, 2, 1, 2, 2]`: row three holds one value out of three rows and answers one, where `rolling(3).sum()` on the same column is missing everywhere. The kernel gates `count` on the slot count for exactly that reason, and it is the one line in the file that would look like a mistake without this paragraph.

## 7. The answer is always float64

Every window reduction answers float64 in pandas, including `count` and including the extremes over an int64 column, because there is nowhere else to put the holes at the top. So the column is cast once on the way in and there is one loop rather than one per dtype.

That is worth saying out loud because it is the opposite of what `cumulative.mojo` does, which monomorphizes over every dtype and is right to, since a cumulative sum over int64 answers int64. It also sidesteps the compile time hazard document 03 describes, where a helper reached from a dtype dispatch that calls back into the dispatch makes the compiler take unbounded time. There is no dispatch here to call back into.

A missing row in the answer is a NaN with no validity bit behind it, which is what every float column in this package holds and what `nan_over_nulls` exists to say.

## 8. The Python surface, and the four arguments that are declared and refused

`s.rolling(...)` takes nine arguments and this implements six of them. `win_type` asks for a weighted window, which is a different kernel rather than a parameter of this one, and pandas needs scipy for it. `on` says to order the window by another column, which only means anything on a frame. `method` chooses between reducing the columns separately and reducing them together, and a column is one column either way. All three are declared and refused by name, on the rule document 18 section 4 sets: the signature parity check compares the whole parameter list against a running pandas, and a caller who passes one should get a sentence about it rather than a TypeError about an unexpected keyword.

`engine` is the fourth, on the reduction rather than on the constructor. `numba` is refused and `cython` is accepted, because `cython` names the path this library is already on.

`numeric_only` is accepted at both values and changes nothing, which is not the usual treatment. On a column it says to refuse a column that is not a number, and every reduction here already refuses one, so the two values agree everywhere this library has an answer. Refusing True would be inventing a difference rather than reporting one.

The five arguments that do describe a window are checked in the constructor and not at the reduction, because that is where pandas checks them. `s.rolling(-1)` raises out of the `rolling` call and not out of the `.sum()` after it, and a program that catches the wrong line is a program whose error handling does not run. Each of the five messages is pandas' own sentence, and each is an `InvalidArgumentError`, which is a `ValueError`.

There is one check that is not pandas'. pandas accepts `step=0` when the window is built and divides by it when the reduction runs, so what a caller sees is a `ZeroDivisionError` out of `.sum()` with nothing in it naming the argument that caused it. A step of nought asks for the same row forever, so it is refused where the other four are.

The kernel checks the same five again on the way in. That is not duplication worth removing: one of the two checks exists to be reached from Python and the other exists because the Mojo API is a public entry point of its own.

## 9. One door, and the width is what picks the window type

There is one bound method under both classes and one function behind it, and an absent width is how an expanding window is spelled. That is the same shape `text.mojo` uses for a slice bound, and it is chosen for the same reason: the alternative is a flag beside the width saying which of them to believe, which is a parameter that exists to say that another parameter does not, and those are the ones that get out of step.

`min_periods` also crosses as an absence, because its default is not one number. Filling it in on the Python side would mean the layer that does not own the rule owns the rule.

The two Python classes are generated from one table entry each and share a `WindowMixin` that holds the column and the five numbers. They differ in their constructors and in nothing else.

## 10. The five that are here and the twenty one that are not

`sum`, `mean`, `count`, `min` and `max`. They are five rather than some other number because they are the ones a window can be carried through: the answer to a window can be derived from the answer to the window before it and the rows that changed.

`std`, `var`, `sem`, `skew` and `kurt` carry more state than a total and square the error, so a carried variance is a different argument from a carried sum and belongs beside its own tests. `median`, `quantile` and `rank` need the window sorted rather than folded, which is a different data structure again. `apply` needs a Python callable per window. `corr` and `cov` take a second column. `aggregate`, `sum` with a `numeric_only` that means something, and the frame forms are surface rather than kernel. The exponentially weighted window has no edges at all and shares nothing with any of this. A window given as a duration rather than a count needs a datetime index to measure against.

None of them resolves rather than resolving and refusing, for the reason document 07 gives: an absent name reads as unimplemented on the board and a refusing one reads as a failure, and the second is a worse thing to say about work that has not been done.
