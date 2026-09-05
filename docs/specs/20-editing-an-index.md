# 20. Editing an index, and turning labels into slice bounds

Written September 2026, against pandas 3.0.5. This document covers the eight names on `Index` that change which labels it holds or turn a label into a position: `append`, `delete`, `insert`, `drop`, `putmask`, `get_slice_bound`, `slice_locs` and `slice_indexer`. Document 19 is the other half of the same type, the lookups and the set operations, and this one assumes it. What is still missing after both is the level accessors, `asof_locs`, and the Python facing `Index` that would make any of it reachable from a program.

## Two groups that look like one

The eight names arrive together because they are the rest of issue #154, not because they are one idea. They are two.

The first five change the labels. Every one of them is a gather or a concatenation away from something that already existed, and the whole design decision is to write them that way rather than as loops over the labels. `append` is `concat_two_any`. `delete` and `insert` are `take_any` over a list of positions. `drop` is `get_indexer_non_unique` and then the same gather. `putmask` is a gather over the two sides concatenated.

That is not tidiness, it is where the correctness comes from. The gather already knows about the validity bitmap, about the offsets in a string column and about every dtype in the library, so inserting a null into a string index works without a line being written about it and there is one place to fix if it stops working. A hand written loop over the labels would need the null case, the string case and the dtype dispatch written again in five places, and the fifth one would be wrong.

The last three turn a label into a position, and they are the group with the thinking in them.

## What a slice bound is

`df.loc["b":"d"]` includes both ends, which every other range in the library does not. The reconciliation is in the pair of bounds rather than in the slice: the left bound of a label is the first row that is at least it, and the right bound is the first row that is past it. So on `["a", "b", "b", "c"]` the pair for `"b"` on both sides is `(1, 3)`, which is a half open range covering both of the b rows, and the inclusive slice the user asked for and the half open range the gather wants are the same thing seen from two sides.

That is also why the duplicates case is the one to look at first. A search that returns any row holding the label is wrong here even though it is right for `get_loc`. The bound has to be the edge of the run, and the two edges are different questions, which is what `side` selects.

## The three shapes an index can be in

**Ascending.** A binary search. The left bound moves right while the row is less than the label, the right bound moves right while the row is less than or equal to it, and that one difference is the whole of `side`.

**Descending.** The same search with both comparisons turned over. pandas supports this and it is not a curiosity: `sort_values(ascending=False)` produces one, and a `loc` slice on it names the larger label first, because reading the index in order means reading it in its order and not in the numbers' order.

**Neither.** There is nothing to search for, so pandas looks the label up instead, and refuses in two cases. A label that is not there gets an error saying to sort the index or use a label it has, because there is no answer at all: nothing in the index tells you where a label that is absent would have gone. A label that is there twice gets a different error, because with no order there is no run for it to be an edge of. Both messages name the label, which we do too, using the frame printer that already knows how to spell every dtype.

## What the monotonic check costs, and why it is not remembered

The search is logarithmic and the check in front of it is linear, so the bound is a pass over the labels either way. That is worth stating plainly rather than letting the phrase binary search imply otherwise.

Caching the check was considered and not done, for the same reason `is_unique` is not cached. An `Index` is copied into every frame derived from it, `take` and `filter` produce a different index from the same labels, and a remembered answer would have to be invalidated in both. That is a correctness problem in exchange for one sequential pass, and a sequential pass over an int64 column is close to memory bandwidth. What the search buys against the fallback is not the pass; it is the per row dtype check and null check the fallback does.

Measured on 200,000 rows, and the machine was busy enough that only the ratio is worth anything. The quietest of three runs gave 114 us for the sorted bound, 969 us for the unsorted one and 99 us for a thousand lookups against a range. So an unsorted index costs about eight times a sorted one, and a range costs about a thousandth of either. Two other runs put the sorted against unsorted ratio at 14 and at 2.5, with interquartile ranges as high as 272 per cent, so the direction holds and the size does not, and the rows are `index/slice_bound_range`, `index/slice_bound_sorted` and `index/slice_bound_unsorted` for anybody who wants to measure it on a quiet one.

The sorted row is also very close to `index/is_monotonic_increasing` on the same data, which is the arithmetic working out: eighteen comparisons after a pass over two hundred thousand rows is a pass over two hundred thousand rows.

## The range keeps paying

The default index is an arithmetic range, and a bound in a range is `label - start` with two clamps, because a range is sorted, has no duplicates, and holds every integer between its ends. It reads no memory. That is the same shortcut `get_indexer` takes and it is worth roughly a thousand to one here rather than the sixty four to one it was worth there, because the general route for a bound is a whole pass and the general route for an indexer is a factorize that the range also skips.

A range asked for the bound of a label that is not an int64, or of a label that is missing, falls through to the general route rather than growing a second rule. That path materializes the range, which costs an allocation on a case that should be rare, and the alternative was two more branches that would have to agree with the general route forever.

## The name rules

`append` keeps the level name only when every side agrees on it, which is the same rule the set operations use and comes from the same reasoning: a label set drawn from two differently named levels is not either of them. The several index form applies the two index rule down the list, so one dissenter among five leaves the result unnamed.

`delete`, `insert`, `drop` and `putmask` all keep the name. None of them brings a second name into the question, and removing a row is not renaming.

## Divergences recorded rather than fixed

**Inserting a null does not promote, and ours is the better answer.** `pd.Index([1, 2]).insert(0, None)` gives a float64 index holding NaN, 1.0 and 2.0, because a numpy int64 array has nowhere to record absence. An Arrow column has a validity bitmap, so ours stays int64 and the label is simply missing. This is a real difference a user can see and it is written down here rather than smoothed over, but it is not one we intend to fix in this direction.

**`errors` is the pandas spelling and `sort` still is not.** `drop` takes `errors="raise"` or `errors="ignore"` as pandas does, and refuses any other word. The set operations in document 19 take a `Bool` for `sort` where pandas takes three values, and that divergence is unchanged by this work.

**Cross dtype promotion is still missing, and now in more places.** `append` of an int64 index to a float64 one raises where pandas promotes. So does `insert` of a float label into an int64 index, and so does a slice bound of the wrong dtype. Every one of them says so in the same words, which is what makes them one issue rather than eight.

**A slice bound does not take a `kind`.** pandas removed it in 2.0, so there is nothing to diverge from, and it is mentioned only because code written against pandas 1.x passes it.

## What is deliberately not here

`asof_locs` needs a bound and a mask together and belongs with the time series work rather than with this. The level accessors, `get_level_values` and `nlevels` and `names`, belong with `MultiIndex` in issue #155, because on a flat index every one of them has a trivial answer that would have to be rewritten the moment there were two levels.

All eight are reachable from Python. That is the one statement document 19 and this document both ended on that is no longer true, and document 21 is the work that changed it.
