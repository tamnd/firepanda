# Holding every value between two bounds

## 1. The method document 48 said was next

`s.clip(lower, upper)` answers the column with every value below the floor lifted to the floor and every value above the ceiling lowered to the ceiling. Document 48 section 8 said this was two picks against two comparisons and that it belonged in that family rather than in the arithmetic, and that turned out to be right, so there is no new kernel here and no new binding. There is a great deal of reading of what the two bounds are allowed to be, which is the whole of the work and the whole of this document.

The reason it belongs with `where` is not that it can be written with `where`. It is that the rules it inherits are `where`'s rules. A value that is missing is neither above nor below anything, so it is left alone, and nobody has to write that, because a comparison against a missing value answers nothing and a row nothing is known about is a row that keeps what it holds. A column that no bound reaches is handed back untouched with its type, for the reason document 48 section 6 gives. And the answer does not widen, for the reason document 48 section 5 gives.

## 2. What is here

`DataFrame.clip` and `Series.clip` with the pandas 3.0 signature, which is `lower` and `upper` positional followed by `axis`, `inplace` and numpy's leftovers as keywords. Both bounds default to None and either may be left out.

`lower` and `upper` are taken in every shape pandas takes them. A single value, a run of values, a mapping, a column lined up by label, and on a frame a whole frame lined up on both axes. `axis` is read on a frame, since a run of values there is a value per column or a value per row and pandas will not guess between them. `inplace` is refused, which is the fifteenth and sixteenth name on that list.

`**kwargs` is numpy's, and it exists because `np.clip(series, 0, 1)` calls this method with whatever numpy was given. `out=None` is accepted and anything else in `out` raises the sentence pandas raises, and any other keyword raises the `TypeError` an unexpected keyword raises, written out by hand rather than left to Python, because Python's own message would name the wrapper.

## 3. A bound that is not a bound

Three things mean no bound and they are not the same three things.

A bound left out is no bound, which needs no explanation. A bound that is a NaN or an NA is also no bound, which is pandas reading a missing value as "unbounded" rather than as "nothing passes", and it is the reading that makes `s.clip(df["floor"].min(), None)` do something sensible on an empty frame. A bound that carries rows and holds nothing in any of them is no bound either, by the same rule applied to each row.

That last one is the interesting one, because it is per row rather than per bound. A floor of `[None, 2.0, None]` is a floor for the middle row and no floor for the other two, so the other two keep whatever they hold, however small. It is the same sentence as "a missing value is not clipped" pointed at the other side of the comparison, and in the implementation it is the same line: the comparison answers nothing, and a row nothing is known about is kept.

## 4. Two bounds that cross, and a label a bound does not carry

Two things here cannot be guessed and were read out of pandas' own source after being measured.

The first is that `s.clip(8, 2)` answers the same as `s.clip(2, 8)`. pandas puts a pair of bounds in order when they are both single values, and does not when either of them carries rows, so `pd.Series([1, 5, 10]).clip([8, 8, 8], [2, 2, 2])` answers `[8, 2, 2]` rather than `[2, 2, 2]`. The reason that is not `[2, 2, 2]` is the second thing worth knowing: both comparisons are made against the column as it arrived, not against what the other bound made of it. The floor lifts the first row from 1 to 8 and the ceiling still asks whether the original 1 was over 2, and it was not, so the 8 stays. Reading the bounds in the other order gives the same answer, which is the test that says the property is real.

The second is what happens to a label the bound does not carry. `pd.Series([1, 5, 10]).clip(pd.Series([9, 9], index=[0, 2]))` does not answer `[9, 5, 10]`. It answers `[9.0, NaN, 10.0]`, and the row the bound said nothing about has lost its value. That is not a decision, it is an ordering artefact: pandas fills the bound's gaps with an infinity before it lines the bound up against the column, so a gap that was already there means unbounded and a gap the alignment creates means a comparison against NaN, which is false, which replaces the row with the bound it could not find. The two readings of "this bound says nothing here" come out opposite, and which one you get depends on whether the gap was in the bound or made by the lining up.

This is reproduced deliberately and it is the part of the method that costs the most code. A bound that carries labels is asked twice: once for what it holds, which says which rows are unbounded, and once for which labels it has at all, which says which rows are replaced. A mapping is not asked the second question, because pandas builds a mapping against the column's own labels first and so a key it lacks is a gap that was already there. `_covering` is the second question and it exists only for this.

## 5. The type the answer has

The rules are document 48 section 5's, unchanged, and they bite more often here because a bound is usually a number and a column of whole numbers is usually a column of whole numbers on purpose.

`pd.Series([1, 5, 10]).clip(2.5, 8.5)` answers `float64` holding `2.5`, `5.0` and `8.5`. Here it raises, because an `int64` column cannot hold `2.5` and firepanda's columns are nullable columns, which follow pandas' own nullable rules: `pd.Series([1, 5, 10], dtype="Int64").clip(2.5)` raises `TypeError: Invalid value '2.5' for dtype 'Int64'` over there too. It is the same divergence document 47 and document 48 argue and it is registered once.

The rule that a column no bound reaches is not touched matters more here than it did for `where`, because it decides whether that refusal is ever reached. `pd.Series([1, 5, 10]).clip(0.5, 20.5)` answers `int64` in pandas, since nothing moved and nothing was put anywhere, and it answers `int64` here, since the bound is never asked whether the column could hold it. So a bound that is a fraction is only a problem when a row actually takes it, which is the same shape of rule and for once it is the friendly direction.

What that rule does not do is save a column of words from a bound that is a number. Both libraries refuse that before anybody asks whether a row would move, because the comparison is refused first, and the sentence is pandas' `TypeError: Invalid comparison between dtype=str and int`. A category is comparable when it is ordered and not when it is not, and a bound that is not one of its categories is refused for the same reason.

## 6. The frame, which is the column once per column

A frame clips each of its columns and the shapes are the shapes document 47 section 4 and document 48 section 7 already describe, with the axis reading changed.

A single value is a bound for everything. A run of values is a value per column, unless `axis=0` says it is a value per row, which is a real difference from `where`: `where` reads a run against a frame by reshaping it, and `clip` lines a bound up against an axis, so here the axis is a question worth asking and there it was not. A mapping is a bound per column name and has to name every column, which is pandas refusing to guess at a partial mapping in this method although it accepts one elsewhere. A column has to be told which way to read and `ValueError: Must specify axis=0 or 1` is what it gets when it is not. A frame lines up on both axes and a column it does not carry loses every row, which is section 4's rule at the width rather than at the height.

The four refusal sentences are pandas' own, down to which number goes where. A run of the wrong length gets `Unable to coerce to Series, length must be {n}: given {m}` with `n` the width or the height depending on the axis, a grid of the wrong shape gets `Unable to coerce to DataFrame, shape must be (3, 2): given (2, 3)`, a two dimensional run against a column gets `Data must be 1-dimensional, got ndarray of shape (3, 1) instead`, and a run of the wrong length against a column gets `Length of values (2) does not match length of index (3)`.

There is one shape this takes and pandas refuses. A grid written as a list of lists is read here as a grid, the same as an array of the same numbers, and pandas reads it as a run of runs, checks its length against the width, and refuses. That is visible as a refusal in one form and as `TypeError: '>=' not supported between instances of 'int' and 'list'` in the other, which is the shape of a leak rather than of a decision, and `where` takes the same list of lists in pandas without complaint. The rule from document 48 section 7 applies: a compatibility layer that copies a leak is not more compatible than one that does the thing the caller meant.

## 7. Why it is two passes and not four

Each column is read once and both bounds are judged against it. A frame of five hundred columns is five hundred reads of a column and not a thousand, and since neither comparison can see the other's work, the two can be worked out in either order or at the same time. The picks are applied one after the other, and the second one is applied to the answer of the first, which is correct precisely because the conditions were not.

A bound that moves nothing is dropped before any pick is built. That is the check that already existed, asked per bound per column rather than per call, and on a frame where one column is in bounds and one is not it saves half the work and all of the type questions on the half it saves.

## 8. What is not here yet

`inplace` is refused here as everywhere, and the list of names waiting on it is now sixteen long and wants a design rather than a method.

`Index.clip` is not here, for the same reason `Index.where` is not, which is document 41 section 8.

`between` is the obvious neighbour and it is not this method. It answers flags rather than values and it is a pair of comparisons with no pick at all, which makes it the cheaper half of this one, and it is not written yet.
