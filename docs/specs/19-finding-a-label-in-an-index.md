# 19. Finding a label in an index

Written September 2026, against pandas 3.0.5. This document covers the lookup half of the `Index` type: the four questions an index answers about itself, the three ways it answers where a label is, and the two ways it compares itself against another index. It does not cover the set operations, which are the other half of issue #154, and it does not cover the Python facing `Index`.

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

## Divergences recorded rather than fixed

`equals` returns `False` between an int64 index and a float64 index holding the same numbers. pandas returns `True`, because it promotes both to a common dtype before comparing. The promotion is the missing piece and it is missing in `get_indexer` too, for the same reason and with the same fix. It is noted in the code at both sites so that whoever writes the promotion finds both.

`get_indexer` does not take a `method` argument. pandas has `"ffill"`, `"bfill"`, `"nearest"` and a `tolerance`, all of which require the index to be monotonic and all of which are a different search from this one, being a binary search over an ordered column rather than an equality lookup. The monotonicity predicates they need exist now, which is most of what they are made of, and the search itself is not written.

The Python facing `Index` is not here. `df.index` and `s.index` return nothing today, so none of this is reachable from Python yet. It is reachable from `loc`, `reindex`, `merge` and `align` inside the library, which is what it was written for.
