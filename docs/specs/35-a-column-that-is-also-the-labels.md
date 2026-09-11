# 35. A column that is also the labels

Status: implemented. Three frame methods, two index methods, and one round trip that has to come back to where it started.

## 1. Why a frame needed this before it needed anything else on the index

`firepanda-compat` has fifty six indexing cases and until this went in not one of them was armed, which made indexing the largest block of unarmed work on the board after the string accessor. The reason was not that the cases were hard. It was that roughly half of them open with `df.set_index("key")` and then ask a question about `loc`, or `xs`, or `reindex`, or `sort_index`, and a case whose first line does not run is a case that cannot be armed no matter how well the rest of it is understood.

So `set_index` is a keystone rather than a feature. It is worth doing on its own, ahead of the much larger `loc` and `iloc` work that most of the indexing section is really about, because it is the line that every one of those cases has to get through first.

## 2. Moving rather than converting

A column and an index hold the same thing in this library. Both are an `AnyArray` of values with a logical type, and the `Index` struct is a small wrapper that adds a name and the ability to stand for a range without materializing one. That is not true in pandas, where an `Index` is a distinct object with its own dtype rules and its own constructor, and it is one of the few places where firepanda's representation is simpler than the thing it is copying.

The consequence is that `set_index` moves rather than converts. The column's values are copied across into the index, the column is taken out of the frame, and nothing is reinterpreted on the way. `drop=False` skips only the taking out, and then the frame holds the same values twice and knows it, which is exactly what pandas does and is the honest reading of what the caller asked for.

Nothing about the new labels is checked. Duplicates are ordinary: an index with two rows labelled `10` is a perfectly reasonable thing, and `loc` answers it with two rows rather than one. pandas does not check either unless it is asked to with `verify_integrity=True`, which is refused here for now because the check is a pass over the labels that nothing else in the library needs.

## 3. The round trip is the specification

`reset_index` is not interesting on its own. What is interesting is that `df.set_index("k").reset_index()` has to give back the frame that went in, and every part of `reset_index` exists to make that true.

The labels come back as a column rather than being thrown away, because throwing them away is what `drop=True` is for. They come back under the index's name, which is why `set_index` bothers to name the level after the column it took. And they come back in the first position rather than the last, which is the part that looks arbitrary and is not: a round trip that put the column back on the far end would give a frame with the same columns in a different order, and a caller comparing two frames would see a difference that nothing in their code asked for.

The name a frame gets when the index has none is the string `index`. That is pandas' rule, it is not derived from anything, and it is why a frame that has never been given an index still gets a column called `index` out of `reset_index`. When a column of that name already exists the operation refuses, which pandas also does, because the alternative is a frame with two columns under one name and this schema does not carry that shape.

## 4. Why `sort_index` has a fast path and why it is narrow

Sorting a frame by its row labels is `argsort` over the materialized labels followed by a `take`, which is three lines. The fast path in front of it is one condition: a range index that is already ascending is already sorted, so the frame is handed straight back.

That condition is narrower than it could be. A descending sort of a range is also decidable without looking at anything, and so is a sort of an index that has already been proved monotonic by something else. Neither is in, because the fast path is not there to be clever. It is there because a range index does not exist as an array until something materializes it, and sorting one would mean building a column of `0, 1, 2, ...` purely in order to discover it is in order. That is the cost worth avoiding, and it only applies to the one case that avoids it.

`kind` is accepted and never looked at, which makes it the only declared argument in the library that is ignored rather than refused. The four names pandas takes are numpy's sort algorithms, the sort underneath is stable whichever is asked for, and a stable order is a correct answer to all four. There is nothing a caller could observe about the difference, and refusing three of the four would refuse a word that makes no difference to the answer.

## 5. Two index methods, and why `searchsorted` does not check

`Index.searchsorted` reports where a label would have to go for the order to hold, and it is one line over the `_searched` helper that `get_slice_bound` and `slice_locs` already use. It does not check that the index is sorted first.

That is deliberate and it is copied. numpy does not check, pandas does not check, and an unsorted index gets an answer that is meaningless in exactly the way it is in both of them. Checking would cost a pass over the labels on every call in order to protect against a mistake that the callers of a binary search do not make, and it would make firepanda refuse a call that pandas answers, which is the kind of divergence that is worse than the bug it prevents.

`Index.isin` is one line over the `is_in_any` kernel, and the interesting part of it is in Python rather than in Mojo.

## 6. Why `isin` has a slow path

The kernel looks up one type in a set of one type. That is the fast answer, it is the right one whenever it applies, and it refuses everything else: an `int64` index against a set of `float64` raises, and a set holding both an integer and a string cannot be built into a column at all.

pandas compares by value. `pd.Index([1, 2]).isin([1.0])` finds the one, and `pd.Index([1, 2]).isin(["a", 2])` finds the two without saying a word about the string. Both of those are ordinary calls and both of them are ones the kernel will not take.

So the Python layer tries the kernel and falls back to comparing in Python when it refuses. That is not a shortcut and it is not a place the work is being avoided: the fast path covers the calls that are shaped like a column, and the slow path covers the calls that are shaped like a Python set, which are the calls that were always going to be a Python loop wherever that loop was written. An empty set never reaches the kernel at all, because a list with nothing in it has no type to build a column from and the answer is all False regardless.

What comes back is a list of bools where pandas gives a numpy array. That is the same divergence `Index.values` and `Index.__eq__` already have and document 21 records it once for all three.

## 7. What each refusal costs to lift

Twelve arguments across the three frame methods are declared and not implemented, and they are not all the same kind of missing.

| Argument | What it would take |
| --- | --- |
| `set_index(keys=[a, b])` | A MultiIndex |
| `set_index(append=True)` | A MultiIndex |
| `set_index(verify_integrity=True)` | A pass over the labels, and a decision about where the check lives |
| `reset_index(level=...)` | A MultiIndex |
| `reset_index(names=...)` | A MultiIndex |
| `reset_index(allow_duplicates=True)` | A schema that carries two columns under one name |
| `sort_index(axis=1)` | Sorting the columns, which is a different operation wearing the same name |
| `sort_index(level=...)` | A MultiIndex |
| `sort_index(ascending=[...])` | A MultiIndex, since a direction per level needs levels |
| `sort_index(key=...)` | Running a Python function over the labels, which is the `key` machinery nothing here has |
| `sort_index(na_position="first")` | The sort kernel taking a direction for nulls, which it has and which is not wired through |
| `inplace=True` everywhere | Owned buffers, which is a decision about the whole library and not about these three methods |

Six of the twelve are one missing thing, and that thing is the MultiIndex. It is issue #155 and it is the largest single lever left on the indexing section, and the shape of this table is the argument for doing it rather than doing six smaller things that each turn out to be it.

`na_position` is the one that could be lifted today and is not. The sort kernel already takes a `nulls_first` flag and `sort_values` already passes one, so wiring it through is a parameter and a line. It is left out of this piece of work because the piece of work is the round trip, and a sort that also grew an option would be two things.

## 8. What it is worth

Ten cases arm on this: `indexing/set-index` over four frames, `indexing/set-index-drop-false`, `indexing/reset-index`, `indexing/reset-index-drop`, `indexing/index-unique`, `indexing/index-is-unique` over three, `indexing/index-monotonic` over three, `indexing/index-get-loc`, `indexing/index-searchsorted` and `indexing/index-isin`. That is roughly seventeen board runs, which is a smaller number than the string work returned for a similar amount of effort.

The number understates it. Forty six of the fifty six indexing cases are still unarmed and roughly half of them open with a `set_index` that now runs, so what this buys is not the seventeen runs. It is that the next piece of indexing work, which is `loc` and `iloc` and is fifteen cases on its own, no longer has to build its own way in.
