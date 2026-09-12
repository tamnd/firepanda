# Dropping a column and dropping a row

## One word for two operations again

`DataFrame.drop`, `Series.drop` and `Index.drop`. The last of those has been here since the index was written and the other two are not, which is two more of the inplace block's missing attributes and two of the commonest calls in pandas.

This is the same split document 45 opens with, one step further along. There, renaming a column was a change to a schema and renaming a row label was a pass over the index, and only the first half could be built. Here both halves can be built, and they are still two different operations wearing one name.

**Dropping a column takes a pointer out of a schema.** `df.drop(columns=["s"])` answers a frame that holds the other columns, and not one value is read, compared or copied. The Arrow buffers underneath the columns that stay are the same buffers, shared rather than copied, because nothing here writes into a frame. The cost is the length of the schema.

**Dropping a row label reads the index.** `df.drop([0, 3])` has to find out which positions carry those labels and then build every column again without them. That is a lookup per label followed by a gather over every column in the frame, and on a wide frame it is the whole frame's width of work.

pandas spells both of them `drop` and decides which was meant from which keyword arrived, the same way it does for `rename`. The difference is that this time the answer is not a refusal.

## What is here

**`DataFrame.drop(columns=...)`**, and `DataFrame.drop(labels, axis=1)`, and `axis="columns"`, all go to the core's `drop`, which resolves every name against the schema in one pass and then selects the positions that were not asked for. One new binding carries it. A name is a string and a list of names is a list of strings, and that is the whole boundary.

**`DataFrame.drop(index=...)`**, and the bare `df.drop(labels)` form that means `axis=0`, take row labels and answer a frame without those rows.

**`Series.drop(labels)`** and **`Series.drop(index=...)`** are the row half with one column under them.

Dropping a column keeps the row labels, dropping a row keeps the level name, and both keep the order of whatever was left. Nothing is sorted and nothing is renumbered.

## The row half is a reindex, and that is a decision

`df.drop(index=labels)` is the frame on the labels its index has minus the labels the caller named, and both halves of that sentence already existed. `Index.drop` knows how to find a set of labels and answer the index without them, `DataFrame.reindex` knows how to build a frame on a set of labels, and the row half of `drop` is those two calls one after the other with nothing in between. No kernel was written for it and no binding was added.

The cost of composing them rather than writing the operation is one extra pass. A purpose built row drop would hash the labels being removed into a set, walk the index once building a boolean mask, and take. This hashes the labels being removed, walks the index once to build the labels being kept, and then hashes every one of those kept labels again on the way back in through `reindex`. On a frame where one row in a million is going, that second hash is over essentially the whole index and the composition is roughly twice the work of the operation.

That is the right trade for now and it is worth being explicit that it is a trade. The extra pass is over the index and not over the data, so it is one column's worth of hashing next to a gather over every column in the frame, which is the part that dominates on anything but a one column frame. If a profile ever says otherwise, the fix is a `drop_labels` on the core that takes the mask route, and the Python above it does not change.

The composition does carry one thing across that a purpose built version would not, and it is in the next section.

## A row label that appears twice

`reindex` refuses an index with a repeated label, because a label that appears twice does not name one row and a reindex has to answer one row per label asked for. The row half of `drop` is a reindex, so it inherits that refusal, and `df.drop([1])` on a frame whose index holds `1` twice raises here where pandas drops both rows.

pandas can do it because its own `drop` is the mask route rather than the lookup route: it builds a boolean array over the index and takes, and a mask does not care how many positions it is true at. So this is the one place where the composition is visibly not the operation, and it is the strongest argument for eventually writing `drop_labels` properly.

It is written down here rather than registered as a divergence because a frame with a repeated row label is already outside what most of the indexing surface accepts, and adding a `drop` shaped entry to the registry would say something narrower than the truth. The truth is that a non-unique index is a piece of work and not a bug in `drop`.

## Which door, and the three ways to get it wrong

pandas works the argument list out in this order, and this copies it exactly, including which class each failure is.

A positional `labels` with `index=` or `columns=` beside it is `ValueError: Cannot specify both 'labels' and 'index'/'columns'`. Passing `index=` or `columns=` together with `axis=1` is `ValueError: Cannot specify both 'axis' and 'index'/'columns'`, and note that `axis=0` beside them is fine, because zero is the default and pandas cannot tell it from nothing. Passing none of the three is `ValueError: Need to specify at least one of 'labels', 'index' or 'columns'`.

`index=` and `columns=` together in one call is not an error and does both, which is the one place `drop` is more generous than `rename`. `rename` refuses both axes at once because one of its two halves refuses on its own. Here neither half refuses, so the pair is two independent steps and there is nothing to decide between them.

**`Series.drop(columns=...)` is accepted and does nothing.** That reads like a bug and it is measured against a running pandas, which takes the keyword, finds a series has no column axis to apply it to, and answers the series unchanged without a word. A compatibility layer that is stricter than the thing it copies is still incompatible with it, so this does the same, and the docstring says so rather than leaving a reader to think it was overlooked. `Series.drop(labels, axis=1)` does raise, with pandas' own `No axis named 1 for object type Series`.

## `errors`, and the two kinds of missing

`errors` defaults to `"raise"` here, which is the pandas default for `drop` and the opposite of the default for `rename`. A name or a label that is not there stops the call with a `KeyError` whose message lists everything that was not found, in the order the caller wrote it, rather than stopping at the first one.

`errors="ignore"` skips whatever was not found and drops the rest.

The two halves get there differently. For the column half the check is done in Python before the core is called, because the core resolves names against the schema and raises on the first one it cannot find, which gives a message about one name when the caller may have mistyped three. Filtering for `"ignore"` and collecting for `"raise"` both need the full list, so both happen above the boundary and the core is only ever handed names the frame has. For the row half `Index.drop` already takes an `errors` and already collects, so the word is passed straight down.

A word that is neither `"ignore"` nor `"raise"` raises here and is silently accepted by pandas, which reads anything that is not `"ignore"` as `"raise"`. That is document 23's rule about a misspelled option being a typo worth catching, it is the same answer `rename` gives, and it is the one deliberate difference in this section.

## The parameters that do no work

`level` selects which level of a multi level index the labels are read against and there is no multi level index, so only `None` is accepted. pandas raises an `AssertionError` here, of all things, saying `axis must be a MultiIndex`, and this raises the `NotImplementedError` the rest of the library raises for the missing MultiIndex instead. Matching an `AssertionError` would mean promising to keep raising one.

`inplace` is refused, for the reason the other thirty six callables that take it refuse it: it does not save a copy, it hides the assignment, and it turns a chained call into a silent no-op. Document 44 is the long version.

## What this does not do

**A frame with no columns left.** `df.drop(columns=df.columns)` in pandas answers a frame of the right number of rows and no columns at all, which is a shape this library's frame can hold but nothing much can be asked about. The call works and what comes back is empty in both dimensions that matter.

**A label that is a tuple.** pandas reads a tuple as one label rather than as a sequence of them, because a tuple is how a multi level label is written. Here a tuple is read as a sequence, since column names are strings and there is no multi level label for a tuple to be. That is the same reading `Index.drop` already uses and it is one more thing waiting on the MultiIndex.

**The mask route.** Covered above. `drop_labels` on the core would remove the extra hashing pass and would make a repeated row label work, and it is a kernel rather than a line, so it is not here.
