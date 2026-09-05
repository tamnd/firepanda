# 19. Finding a label in an index

Written September 2026, against pandas 3.0.5. This document covers the lookups and the set operations on the `Index` type: the four questions an index answers about itself, the three ways it answers where a label is, the two ways it compares itself against another index, and the four ways two indexes combine. It does not cover the editing operations and the slice locators, which are document 20, nor the level accessors, and it does not cover the Python facing `Index`.

## Why the lookups come before anything that uses them

An index that only stores labels is a field on a struct. What makes it an object is that other code asks it questions, and almost all of that code asks one of the same few questions.

`df.loc["north"]` is `get_loc`. `df.reindex(other.index)` is `get_indexer` followed by a take. `df.merge(other, left_index=True, right_index=True)` is `get_indexer_non_unique`, because an index being merged on is exactly the case where a label appears more than once. `df + other` is `equals` first, to find out whether any alignment is needed at all, and then `get_indexer` when it is. `df.groupby(level=0)` is a factorize of the labels, which is the same machine again.

Writing any of those five before the lookups exist means writing a lookup inside each of them, in five slightly different ways, with five slightly different answers about what a missing label does. Then four of them get deleted. The order here is the cheap one.

## One idea underneath all of them

Put the index's labels and the labels being looked up end to end in a single array. Factorize that array once. Two labels are the same label exactly when they came out of the factorize with the same ordinal.

That is the whole method. It has three properties worth stating.

It is one pass over each side rather than a build phase and a probe phase written separately, because the factorize already does both and has been tuned. There is no second hash table in the codebase to keep correct.

It gets nulls right without any code about nulls. `firepanda.hash.grouping` puts every null in one group, so a null label in the index and a null label in the target come out with the same ordinal and match. pandas matches NaN to NaN in `get_indexer` and nowhere else in the library, which is a rule that has to be written down somewhere, and here it is inherited rather than written. That is the good kind of inheritance and it is also the fragile kind, so `tests/test_index_lookup.mojo` asserts it directly rather than trusting it.

It gets the dtype question right by refusing it. Two arrays can only be concatenated when they have the same dtype, so an int64 index looked up with a string target raises instead of silently finding nothing. pandas would promote both sides to a common type and then compare, which is a real feature and is not written yet, and the error says so in those words rather than pretending the labels were absent.

The cost of the method is that the array being factorized is the two sides joined, so it is the sum of the two lengths rather than the larger of them. `index/get_indexer_labels` in the benchmark suite is 67 ns a row at two hundred thousand rows, and `index/is_unique`, which factorizes one side only, is 25 ns a row. Almost all of the difference is the extra rows. A factorize that took two columns and grouped them together without materializing the join would take most of that back, and when somebody writes it that benchmark row is where it shows up.

## The range does none of that

The default index is an arithmetic range and it stays one until somebody replaces it. The position of a label in a range starting at `start` is `label - start`, and the label is present when that lands in `[0, length)`. No memory is read except the labels being asked about, no array is built, and nothing is hashed.

This is worth sixty four times the general route on the benchmark machine: 210 us against 13.5 ms for the same two hundred thousand lookups. It is the largest ratio between two rows anywhere in the microbenchmark suite, and it is on the operation that a frame nobody set an index on performs every time it is added to, compared with, joined to or reindexed against another frame. Keeping the range unmaterialized is not a memory optimization with a pleasant side effect. The side effect is the point.

The same shortcut applies to the four self questions. A range never repeats a label, so `is_unique` is `True` and `has_duplicates` is `False` without looking. A range steps by one, so `is_monotonic_increasing` is `True`. `is_monotonic_decreasing` is the one that catches people: it is `False` for a range with anything in it, and `True` only for a range of zero or one labels, because there is nothing to descend.

Because the fast paths are a second implementation of a function that already exists, the tests do not only check them against hand counted positions. They materialize the range into a column, build an index over the result, and require the two routes to give the same answer to the same question. Two implementations of one function drift, and the reason they drift is that one of them is cheaper, which is the reason both of them exist.

## What each member does

`is_unique` and `has_duplicates` are one factorize and a comparison of the group count against the length. `has_duplicates` is `not is_unique` and exists because pandas has both spellings and code written against pandas uses both.

`is_monotonic_increasing` and `is_monotonic_decreasing` are one sequential pass, delegated to `is_sorted_any` in the sort kernel so that the comparison rules stay in one place. An index with a null in it is neither, which is what pandas says, and the check is cheap because the column already carries its null count.

`get_indexer` returns one position per label asked for, and `-1` where the label is not there. It refuses a non unique index, raising the way pandas raises `InvalidIndexError`, because a label sitting in two rows has no single position to report and returning either one is a wrong answer rather than an approximate one. On a range index with an int64 target it takes the arithmetic route. Otherwise it factorizes both sides together, builds one table from ordinal to position sized by the group count, and reads the target's ordinals through it.

`get_indexer_non_unique` is the general form and does not refuse anything. Its answer has one row per matching pair rather than one per label, so a target of three labels against an index where the first appears twice produces four rows. It also returns the positions within the target of the labels that were not found, which is the second array pandas returns and is what a caller needs to fill the holes afterwards. Internally it is a counting sort over the ordinals: count how many index rows fall in each ordinal, prefix sum into starts, scatter the row numbers, then walk the target twice, once to size the output and once to fill it. Three sequential passes, no allocation per label.

`get_loc` is the raising form and takes exactly one label. It returns every position that label sits at, so it answers for a repeated label rather than refusing like `get_indexer` does, and it raises when the label is absent. That is the whole difference between it and `get_indexer`: a caller that wants a sentinel already has one, and a caller who wrote `df.loc["nope"]` wants to be told, not handed a row of nulls.

`equals` compares labels and ignores names. Two ranges compare by start and length without materializing either. A range against a column materializes the range, because an index is what its labels are and not how they happen to be stored. `identical` is `equals` plus the name, which is what a frame comparison wants and what an alignment does not.

## The four set operations

They are the same machine with a different rule about which rows survive. Concatenate the two label sets, factorize once, then count how many rows carry each ordinal on each side separately. That pair of counts is the whole answer. A union keeps each label the larger of the two counts times, an intersection keeps the labels with a count above zero on both sides, a difference keeps the ones with no count on the right, and a symmetric difference keeps the ones with no count on the other side.

The rules layered on top of that are where the bugs are, and every one of them was read off a running pandas 3.0.5 rather than off the documentation, which is wrong about at least one.

**Order.** `union` sorts by default and, unsorted, is the left side's order followed by the labels only the right side has. `intersection` and `difference` are the other way round: their default is the left side's order and sorting is what you ask for. That asymmetry is not arbitrary once you look at it. An intersection is a filter of the left side, so there is an order to inherit; a union is not a filter of anything, so there is no order to inherit and one has to be invented. `symmetric_difference` sorts by default, and unsorted it is the left side's leftovers followed by the right side's.

**Duplicates.** `union` is the only one of the four that keeps them, and it keeps each label the larger of the two counts rather than the sum, so `[1, 1, 2, 3]` union `[1, 2, 2, 4]` is `[1, 1, 2, 2, 3, 4]`. The reasoning is that a union of two label sets should be able to label at least as many rows as either of them could. The other three return unique values, including `intersection`, whose docstring in pandas 3.0.5 still says it keeps the smaller of the two counts and which has not done that for several versions.

**Short circuits that are observable.** A union with an empty index, an empty index unioned with one, and an index unioned with itself all return in the original order, because pandas takes an early return before the sort. `Index([3, 1, 2])` union an empty index is `[3, 1, 2]` and not `[1, 2, 3]`. That is not an internal optimization detail, it is the answer, and it matters because unioning an index with an equal one is what aligning two frames with the same labels does and it is the most common call in the whole API. It is also the fast one: the short circuit is worth about nineteen times the general path, and two default ranges do not even compare labels because a range compares by start and length.

**Names.** All four carry the name when both sides agree and drop it when they do not, on the reasoning that a label set drawn from two differently named levels is not either of them. `symmetric_difference` takes a `result_name` that overrides that, and it is the only one of the four that does, because it is the only one where neither side has a better claim.

**Nulls.** A null is an ordinary label. It matches a null and it sorts last, so `[1, null, 3]` union `[null, 3, 5]` is `[1, 3, 5, null]`. Neither half of that is written in the index. Matching comes from the factorize putting every null in one group and the ordering comes from `argsort_any` defaulting to nulls last, so both are asserted in the tests rather than assumed.

pandas can only hold a missing label in a float index, because a numpy int64 array has nowhere to record absence. An Arrow column carries a validity bitmap, so the same cases are written on integers here and mean the same thing.

`unique` and `get_indexer_for` come out of the same machinery. `unique` keeps the first of each label rather than sorting, which is what pandas does and what a caller who wanted them sorted would rather ask for than have done twice. `get_indexer_for` is `get_indexer` when the index is unique and the positions half of `get_indexer_non_unique` when it is not, which is the method the rest of pandas calls internally when it does not know which kind of index it is holding.

## Divergences recorded rather than fixed

`equals` returns `False` between an int64 index and a float64 index holding the same numbers. pandas returns `True`, because it promotes both to a common dtype before comparing. The same promotion is missing in `get_indexer` and in all four set operations, which raise where pandas answers `Index([1, 2]).union(Index([2.0, 3.0]))` with a float64 index. It is one missing piece in six places and it should be written once rather than six times, so all six say the same thing in the same words and none of them guesses.

`sort` is a `Bool` and pandas' third state is not represented. pandas uses `sort=None` to mean sort unless the values cannot be compared, and every dtype firepanda has is comparable, so the distinction has nothing to bite on. The `Bool` default reproduces pandas' `None`, including the short circuits that skip the sort. What is not reachable is pandas' explicit `sort=True`, which sorts even in the short circuit.

`get_indexer` does not take a `method` argument. pandas has `"ffill"`, `"bfill"`, `"nearest"` and a `tolerance`, all of which require the index to be monotonic and all of which are a different search from this one, being a binary search over an ordered column rather than an equality lookup. The monotonicity predicates they need exist now, which is most of what they are made of, and the search itself is not written.

The Python facing `Index` is not here. It arrived in document 21, so `df.index` and `s.index` reach all of this now, and the divergences that only show up at the boundary are recorded there. Inside the library it is reachable from `loc`, `reindex`, `merge` and `align`, which is what it was written for.
