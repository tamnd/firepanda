# 43. Sorting by value rather than by label

Documents 41 and 42 each opened a door and counted the members that followed from it. This one is a different shape of change and worth writing down for that reason: nothing new was built at all. The sort was already there, in the core, finished, tested and fast, and it had no way of being called from Python.

## 1. What was already written

`firepanda/kernel/sort.mojo` holds five entry points. `argsort` and `argsort_into` are the typed pair, `argsort_multi` takes a list of columns, `argsort_any` and `argsort_any_into` take an erased one. `DataFrame.sort_values` and `DataFrame.argsort` sit on top of the multi column pair at `firepanda/frame/frame.mojo`, `Series.sort_values` and `Series.argsort` sit on top of the erased pair at `firepanda/frame/series.mojo`, and `DataFrame.argsort_limit` is the version that does not build the whole permutation when a limit is going to throw most of it away.

All of that is reachable from Mojo and none of it was reachable from Python. `df.sort_values("a")` is somewhere near the top of the list of things a pandas caller writes, and the answer to it was an `AttributeError` sitting on top of a finished sort.

So this change is two Mojo methods on the two bound types, three entries in the binding table, and about a hundred lines of Python that turn pandas' vocabulary into the core's. There is no kernel work in it, which is the whole point of writing it down: the parity board measures a surface, and a surface can lag an implementation by a long way without anybody noticing, because the thing that is missing does not fail, it is simply absent.

## 2. The vocabulary is where all the work is

pandas and the core describe the same sort with different words, and every one of the differences has to be undone somewhere.

pandas names the direction it sorts in, with `ascending`, and the core names the other one, with `descending`. That is one `not`. pandas takes either a single flag for all the keys or a sequence with one flag per key, and the core takes a list either way, so a single flag is spread across the keys. pandas says where the missing values sit with a word, `first` or `last`, and the core says it with a bool per key, so the word becomes a flag and the flag is copied across the keys because pandas has one word for all of them. pandas takes one key name or a list of them, and a string is one key even though a string is also a sequence, which is the trap in reading that argument and is why the check is for a string before it is for a sequence.

None of that belongs in the binding. The binding takes three lists of the same length and does no interpreting, because the side that knows how many keys there are is the side that read `by`, and that is Python.

## 3. Two arguments that are accepted and never looked at

`kind` is the first, and `_sort_index` already had the argument for it: the four names it takes are numpy's sort algorithms, the sort underneath is stable whichever one is asked for, and a stable order is a correct answer to all four. There is nothing a caller could observe about it.

`Series.argsort` has a second one, `stable`, which is numpy's other way of asking the same question. Accepting two arguments and reading neither looks worse than accepting one, and it is the same fact stated twice rather than a second gap: both of them ask for a sort algorithm, and the sort is stable.

Every other declared argument in this change is either read or refused. `key`, which runs a function over the values before sorting, is refused. `inplace` is refused, in the words document 13 argues for and for the reason it argues them. `order`, on `Series.argsort`, names the fields of a numpy record array and is refused because there is no record array here for it to name.

## 4. `ignore_index` is one line and it is in the right place

pandas numbers the rows again after sorting when `ignore_index` is true. That is `reset_index(drop=True)` on the answer and it is written as exactly that, above the boundary, because it is what the argument means and because the core has no business knowing about it. A sort that also knew how to renumber would be a sort with a second job.

It is worth contrasting with `sort_index`, where `ignore_index` is refused rather than implemented. That refusal is not inconsistency. Sorting rows by their labels and then throwing the labels away discards the thing that was just sorted, so the call is almost certainly a mistake, and pandas' own documentation is lukewarm about it. Sorting rows by a column and then renumbering them is an ordinary thing to want.

## 5. What `argsort` does with a null, and why it is the core's answer and not this layer's

pandas' `Series.argsort` puts a negative position in the row a missing value sat in. It can do that because it is numpy underneath, numpy's `argsort` returns the platform index type, and a negative index is a value that no real position ever takes, so there is a sentinel going spare.

firepanda has no such sentinel, and the reason is in `Series.argsort`'s own docstring in the core. A null here is placed by `nulls_first` rather than removed and marked, so a null gets a real position the same as every other row and nothing ever comes back negative. The permutation the kernel builds is uint32, because it is one number per row and the sort rewrites all of it on every pass, and the core widens it to int64 at this one boundary so that the pandas name answers the pandas width.

This is a divergence and it is recorded rather than hidden. It is also the narrow kind: it only shows up in a column that has a null in it, and the position that comes back for that row is a position rather than a mark, which is a defensible answer to the question and is not the answer pandas gives.

## 6. `Index.sort_values` has two return shapes and that is pandas

`return_indexer` decides whether the caller is handed the sorted index or a pair of it and the permutation that produced it. Two shapes from one method reads badly and it is pandas' method, so it is copied rather than improved on.

The permutation comes back as a list of numbers where pandas gives a numpy array, which is the same shape `Index.duplicated` and the `isna` family already answer in here. Document 41 section 5 has that note and this is another instance of it rather than a new one.

The sorted index goes through `_like`, which document 42 section 6 introduced, so an index of instants that is sorted is still a `DatetimeIndex`. That is the third member to use it and the rule it holds is now written in one place instead of three.

`Index.argsort` is declared `(*args, **kwargs)` in pandas and it is declared that way here too. It is not a shorthand: the index method takes whatever the column method takes and hands it straight down, so this does the same, and a caller who passes something the column refuses gets the column's own refusal rather than a second one written here.

## 7. The one place the column does not go through the frame

Document 42 built ten series members as a frame method with the column put in and taken back out. `Series.sort_values` does not do that, and the reason is that it does not need to. A column is one sort key, the core has a one key sort with its own entry point that exists precisely so that a series does not pay for a list to hold its single key, and going through the frame here would build a frame, sort it on one column, and take the column back, to reach a method that was already sitting on the series.

`Series.argsort` is the same. `Index.sort_values` and `Index.argsort` do go through a column, because an index has no sort of its own in the core and the column it becomes does.

## 8. What this is worth

Five board names: `DataFrame.sort_values`, `Series.sort_values`, `Series.argsort`, `Index.sort_values` and `Index.argsort`, and the last two score twice because `DatetimeIndex` subclasses `Index`. So seven.

The parameter space is where it earns more than that. Several keys, a direction per key, `na_position` on both words, and `ignore_index` are four things a conformance case can vary that the members built in documents 41 and 42 mostly could not, because most of those members have one interesting argument or none.

It also unblocks two cases that were already written and failing. `stats/argsort` asks for `argsort(kind="stable")` and `stats/searchsorted` sorts before it looks anything up. A case written against a door that does not exist does not merely fail to score, it holds down every level above it, which document 42 section 7 found three instances of. These are two more of them and they are now open.

## 9. What is deliberately not here

`axis=1`, which sorts the columns by the values in a row. pandas has it and it is a different operation wearing the same name: the keys are row labels, every column has to be the same type for the comparison to mean anything, and the answer reorders the schema rather than the rows.

`key`, which runs a function over the values before comparing them. It is a real feature rather than a spelling, since it wants a Python callable applied to a column and the result sorted instead, and it is refused rather than guessed at.

`DataFrame.argsort`, which pandas does not have. The core has one and it is what `sort_values` is built on, and binding it would be firepanda offering a method pandas does not, which is the direction of difference this library does not get to have.

`argsort_limit`, for now. It is the right implementation of `nlargest` over a big frame and `nlargest` currently reaches `_top_rows` instead. Moving it is a performance change with no surface in it, which makes it document 39's business rather than this one's.
