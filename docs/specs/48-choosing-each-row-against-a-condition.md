# Choosing each row against a condition

## 1. Two methods and one question

`where` keeps the rows a condition says are true and takes the rest from somewhere else. `mask` is the same method with the condition turned over. pandas documents the second as the first with a `~` in front of it, and that is very nearly how it is implemented here, with one correction that section 4 is about.

The condition is the easy half. It is a column of flags, it is as tall as the rows, and reading it is a pass with no branch in it. The interesting half is the somewhere else, because the somewhere else can be a single value, a column lined up by label, a run of values read by position, a whole frame lined up on both axes, a callable handed the thing it is going to fill, or nothing at all. Document 47 answered five of those six for `fillna`. The sixth is new, and it is the one that decides what type the answer has.

## 2. What is here

`DataFrame.where`, `DataFrame.mask`, `Series.where` and `Series.mask`, with the pandas 3.0 signature, which is `cond` and `other` positional followed by `inplace`, `axis` and `level` as keywords. `other` defaults to a sentinel rather than to a value, because "no other side" and "the other side is None" are different requests in pandas and have to stay different here.

`cond` is taken in every shape pandas takes it. A column lined up by label on either class, a plain sequence read by position on a column, a two dimensional run of values on a frame, a frame lined up on both axes on a frame, a column read across the columns when `axis=1` says so, and a callable handed the frame or column and asked. `other` is taken in every shape as well, plus the empty one.

`inplace` and `level` are refused. `axis` is read, and on a frame it is read properly rather than swallowed, because a column offered to a frame is a value per row or a value per column and pandas will not guess between them.

Three new bindings underneath, all of them over things the core could do and could not be asked. `PySeries.pick` and `PyDataFrame.pick` over the kernel, and `PySeries.missing_row`, which is section 5.

## 3. A false side of one row

The core's pick took two columns as tall as the condition and chose between them row by row. Every call in this family wants something narrower than that. `s.where(cond, 0)` has one value on the false side and not a column of them, and building the column of them would mean writing the same number sixty four times into a buffer that is then read once.

So the kernel learned a false side of one row. `pick_one` in `firepanda/kernel/pick.mojo` takes a condition, a true side as tall as it, and a false side of exactly one row, and spreads that row across every position the condition did not keep. When that row holds a value it is the constant pick that was already there, reached by reading the value out and handing it over. When that row holds nothing it is a pass of its own, because a null cannot be carried in a `Scalar` and the validity bitmap has to be built rather than copied: a row that was kept keeps the true side's bit and a row that was not kept has its bit cleared.

That second case is not a corner. It is what `where` with no other side named means, and it is the single most common way both of these methods are written.

The text side got the same treatment, one row spread rather than one row per row, and the dictionary side reaches it through the same call on the codes, which is the whole of why a category column works here without a line of its own.

## 4. A row the condition says nothing in

A condition can have a gap in it. Not a false, a gap, which is what a comparison against a missing value answers and is therefore what any condition written as `df["a"] > 2` has in it wherever `a` is missing.

pandas replaces that row. Both methods replace it. `where` replaces it because the row was not kept, which is what anyone would guess, and `mask` replaces it too, which is not what anyone would guess, since a row `mask` replaces is a row the condition called true and this row was not called anything. The rule that makes both of them come out right is not "a null is a false" but "a null is not a keep", and the two are the same sentence for `where` and opposite sentences for `mask`.

This is one line in the implementation and it is the line most likely to be written the wrong way round. The condition is turned over first, while its gaps are still gaps, and the gaps are read as falses second, on the answer. Turning over a gap that has already become a false makes it a true and keeps a row pandas replaces. The order is the whole of the difference and there is a test for it in both directions.

It was measured rather than reasoned, and measuring it took a detour worth writing down. pandas cannot answer `s.mask([True, None, True], 0)` at all: a list with a `None` in it becomes an object array, `mask` inverts the condition with a `~`, and `~` on a `None` raises `TypeError: bad operand type for unary ~: 'NoneType'`. The question is only answerable over there in the nullable `boolean` type, which is the type a firepanda column behaves like anyway. So that is where the side by side test asks it.

## 5. The type the answer has

`pd.Series([1, 2, 3]).where([True, False, True])` answers a column of `float64` holding `1.0`, `NaN` and `3.0`. The integers became floats because numpy's `int64` has nowhere to put a missing value, so pandas widened the column to the narrowest type that has somewhere. A column of words widens to object for the same reason, and a column of bools widens to object as well, since there is no bool with a gap in numpy either.

A firepanda column is Arrow's and every one of them carries a validity bitmap, so there is always somewhere to put a missing value and there is never a reason to widen. The answer here is `int64` holding `1`, nothing, and `3`.

That is a divergence and it is the right one, and the argument is not "Arrow can do it" but pandas itself. `pd.Series([1, 2, 3], dtype="Int64").where([True, False, True])` answers `Int64` holding `1`, `<NA>` and `3`. The nullable types do not widen, because they do not have to. `boolean` stays `boolean`, `string` stays `string`. A firepanda column is a masked column, so it follows pandas' rules for masked columns, and document 47 made the same argument for the same reason about what a fill value is allowed to be.

Getting a missing value of the column's own type turned out to be the one thing the Python layer could not build. A one row column of nulls cast to the target type raises on the cast, because a cast to a whole number refuses a gap. A shift by one widens. A reindex to a label that is not there wants an index of the right type to fail to find it in. The core has had the answer the whole time and it belongs to the outer join: a gather from a negative position means a row that is not there, and it answers one row of the right type holding nothing. `missing_row` is a binding over that and nothing else.

## 6. A column that keeps everything is not touched

`pd.Series([1, 2, 3]).where([True, True, True], "a word")` does not raise and does not widen. It answers the column. pandas widens a column when it puts something in one, not when it is offered something, and a condition that keeps every row never puts anything anywhere.

So the same rule runs here, and it runs before the other side has been looked at rather than after. If no row is going to take the other side, the column comes back untouched and the other side is never read, never lined up, and never asked whether this column could hold it. That is the same shape as the rule document 47 section 5 describes for a column with nothing missing, and it is load bearing for the same reason: without it, `df.where(cond, 0)` across a frame would refuse on the frame's text column even when the condition keeps every word in it.

On a frame the question is asked per column, because keeping every row is a property of a column and not of a frame.

## 7. The shapes, and the one pandas cannot run

A condition that carries labels is lined up on them and a label it does not carry is a false. That is worth a second look, because it is pandas making an unusual choice: a reindex that cannot find a label normally leaves a gap, and here it leaves a decision. Extra labels are ignored. A condition that carries no labels is read by position and has to be exactly as tall, and a condition that is not boolean raises `TypeError: Boolean array expected for the condition, not int64`, which is worth knowing because a column of ones and zeros is the obvious thing to write and is the thing pandas will not take.

On a frame, a one dimensional run of values is refused even when it is exactly as tall as the frame, which reads as a gap in pandas until you notice that the thing it would obviously mean is what a column says and a caller who means that can say it. A two dimensional run is a flag per cell. A frame lines up on both axes and a cell it does not carry is a false.

The other side follows the same rules with one addition. A run of values against a frame is one value per column rather than one value per row, which is numpy's broadcasting rule and is not the reading that looks right, and a run of values that is the frame's height and not its width raises `cannot reshape array of size 3 into shape (3,2)`. A column offered to a frame has to be told which way to read, and pandas raises `ValueError: Must specify axis=0 or 1` rather than guessing. A frame offered to a column raises `NotImplementedError: cannot align with a higher dimensional NDFrame`, which is pandas' own wording and is one of the few places pandas raises that class.

One shape in this family cannot be run in pandas at all. `df.where(flags, axis=1)` with `flags` a column labelled by column name raises `TypeError: '_NoDefault' object is not subscriptable` from inside pandas' block manager, and the same call with a scalar other raises the same sentence about an `int`. It is a bug rather than a refusal, the intended meaning is not in doubt, and it is implemented here as one flag per column. It has been reported upstream. A compatibility layer that copies a crash is not more compatible than one that does the thing the crash was trying to do.

There is one shape where this refuses and pandas does not. A mapping as the other side is read by pandas as a single object, and `df.where(cond, {"a": 9})` puts the mapping itself into every cell it replaces, in a column of dtype object. There is no object column here, so it raises, and it raises in the sentence numpy gives for a value of the wrong size. This is the same hole document 47 section 4 describes, seen from a different side.

## 8. What is not here yet

`clip` is the next one and it is the same kernel again. `s.clip(lower, upper)` is two picks against two comparisons, and the reason it belongs in this family rather than in arithmetic is the rule it inherits for free: a comparison against a missing value is not true, so a missing value is not clipped, which is exactly what pandas does and is a special case nobody has to write.

`inplace` is still refused everywhere, for the reason given the first time it came up, which is that every operation here answers a new column and the buffers underneath are shared rather than owned. It is the largest remaining group of divergence cases and it wants a design rather than a method.

`where` on an index is not here either. It is in the list of index methods document 41 section 8 left for later, and it is the same method again once an index can be asked to pick.
