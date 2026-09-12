# Filling what is missing with a value that was named

## 1. A method with almost no method in it

`fillna` is one of the most written lines in pandas and the operation under it is the smallest one in this whole section. Take a column, take a second column, and for every row read the first one's validity bit and pick a side. That is a coalesce, it is one pass with no branch that depends on a value, and `firepanda/frame/series.mojo` has had it since the fill family was written. `Series.fill_null` takes a second series and `Frame.fill_null` takes a column name and a second series, both of them accept a fallback one row tall and treat it as the value for every missing row, and the docstring on the series one already said in as many words that a one row fallback is how filling with a scalar is spelled.

So the kernel was there, the broadcast was there, and the method did not exist, because neither call had a binding. It is the only member of the fill family that shipped without one: `ffill` and `bfill` both have bindings and both are reachable from Python, and the one that a caller reaches for first was reachable from Mojo and from nowhere else. That is the third time in five slices that the surface has been found lagging a finished implementation, and it is worth saying out loud that the failure mode is silence. Nothing goes red when a binding is missing. The method is simply not there, and only something that goes looking notices.

What this document is about, then, is not the fill. It is the two questions that have to be answered before the fill can run, and both of them are about types.

## 2. What is here

`DataFrame.fillna` and `Series.fillna`, with the pandas 3.0 signature, which is `value` positional and required followed by `axis`, `inplace` and `limit` as keywords. `method` and `downcast` are gone from pandas 3.0 and are not declared here, because declaring a parameter that the library being copied has deleted would be preserving a mistake.

`value` is taken as a single value, on either class, and as a dict of column name to value on the frame. `axis` is read and not used. `inplace` and `limit` are refused. A dict on a column, and a value that is itself a column or a frame, are refused with a sentence naming what would build them.

Two new bindings over the two core calls, and one more over a thing the frame could not previously be asked. That third one is `null_counts`, which answers how many rows are missing from each column, and section 5 says why the method needs it.

## 3. A typed column stays typed, and that is the whole design

pandas has two kinds of numeric column and they answer this question differently. A `float64` column is numpy's, it spells missing as NaN, and there is no such thing as an integer column with a gap in it, so `pd.DataFrame({"i": [1, None, 3]})` comes back as float. An `Int64` column is pandas' own, it carries a mask beside the values, and it holds whole numbers with gaps in them. firepanda's columns are Arrow's, they carry a validity bitmap beside the values, and every one of them can have a gap. So a firepanda column is `Int64` and not `int64`, and the rules to copy are pandas' rules for its masked types rather than its rules for numpy's.

Those rules were measured rather than guessed, and they are these. An `Int64` filled with `2.0` answers `2`, because a float that is a whole number is a whole number. An `Int64` filled with `True` raises `TypeError: Invalid value 'True' for dtype 'Int64'`, because a bool is an int in Python and is not a number in a column. A `Float64` filled with `2` answers `2.0` and filled with `True` raises. A `boolean` filled with `1` raises. And every one of them raises on text.

This library answers all five the same way, and the sentence it raises is pandas' sentence with this library's own spelling of the type in it. That is not decoration. A caller who has been catching `TypeError` and matching on `Invalid value` is already catching this.

The check is on the kind of the value and not on whether a cast exists, and the difference matters more than it looks. A cast from a number to text exists here and answers `'0'`. A cast from a float to an integer exists and throws the fraction away. `cast(dtype, strict=True)` performs both of them without complaint, because strict there means something narrower than lossless. So a `fillna` written as "cast the value to the column's type and coalesce" would answer `'0'` for `df.fillna(0)` on a text column and `0` for `fillna(0.5)` on an integer one, and both of those are worse than an error, because both of them are a wrong answer that looks like a right one. The kind check sits above the cast, and the cast below it only ever changes a width.

## 4. The one place this cannot follow

There is one row in that table where pandas does not raise, and it is the `string` column. `pd.DataFrame({"s": ["a", None, "c"]}).fillna(0)` answers a column of dtype object holding `'a'`, the integer `0` and `'c'`. It does not raise because it does not have to: pandas has a type that holds anything, so widening is always available as a last resort, and every numpy backed column in pandas widens to it rather than refusing.

firepanda has no such type. An Arrow column is one type all the way down, the object dtype is not implemented and is not planned in this milestone, and there is nothing for a text column filled with a number to widen into. So this raises where pandas widens, and the sentence says which type could not hold which value.

It is worth being clear about what the alternative would have been, because there was one and it was rejected. The library could have cast the number to text and put `'0'` in the gap. Every value in the answer would have been of the column's own type, nothing would have raised, and the call would have looked like it worked. It is the wrong answer. A caller who wrote `df.fillna(0)` across a frame meant zero, and a zero that has become the character zero is a value that will compare wrong, sort wrong and total not at all, and it will do all three quietly. An error the caller sees is better than a value the caller does not.

## 5. A column with nothing to fill is not filled, and that is a rule

The refusal above is narrower than it first reads, and the reason is a rule taken from pandas rather than from convenience.

pandas widens a column's type only when it actually puts something in it. `df.fillna(0)` on a frame whose text column has no gaps leaves that column exactly as it was, dtype and all, because there was no row to write into and therefore no reason to widen. Only a column that actually receives the value changes type.

This does the same, which means the check in section 3 is only ever reached for a column that has something missing. `df.fillna(0)` on a frame with a complete text column and a gappy integer one is fine here and answers what pandas answers. The same call on the same frame with one value taken out of the text column raises. That looks inconsistent written down and it is exactly what pandas does, for exactly the same reason, and a compatibility layer that refused in both cases would be stricter than the thing it is copying and therefore still incompatible with it.

Getting that rule right is what the third binding is for. The method has to know two things about each column before it decides anything, which are the column's type and whether it has a gap. The type was already cheap, because `dtypes` reads the schema. Whether a column has a gap was not reachable without `column(name)`, and `column` copies, and its own docstring says so: it flattens a column into one contiguous array and hands back a new series. Asking a five hundred column frame which of its columns have gaps by copying five hundred columns is the whole frame moved to answer a question the validity bitmaps already hold.

So `PyDataFrame.null_counts` exists, it is shaped exactly like `dtypes`, and it reads validity and nothing else. Both of them are the same principle written twice: knowing a frame's shape should not cost its contents.

## 6. The parameters, and what each one is

`axis` is accepted, checked and not used. It is checked, so `axis=2` gives pandas' sentence about there being no axis named 2 and a column refuses `axis=1` the way a column refuses every second axis. It is not used, because with one value per column the two axes name the same answer: there is nothing about running down a column rather than across a row that changes which rows are missing or what goes in them. pandas has a difference here and it is a difference in the resulting dtype rather than in the values, and this library does not have the dtype either way.

`inplace` is refused, with the sentence the other forty one callables that take it use. Nothing about this one is special.

`limit` is refused, and it is refused after being validated. pandas checks that a limit is a whole number greater than zero before it does anything with it, so `limit=0` and `limit=1.5` raise `ValueError` with pandas' own two sentences here as well, and a limit that would have been usable raises `NotImplementedError` instead. Doing it in that order means a caller who wrote a bad limit is told it is bad rather than being told the parameter is unsupported, which is the more useful of the two things to hear.

The reason it is refused is the coalesce. A limit means stop after so many rows, which means the fill has to count what it has already done as it goes, and a coalesce reads a validity bit and picks a side without ever knowing how many rows came before it. `ffill` and `bfill` do take a limit, and the difference is that they walk: a forward fill already has a notion of how long the current run of gaps is, because that is what it is doing, so a limit is a comparison it was already in a position to make. This one is not walking. Adding a limit here is a different kernel and not a parameter on this one.

## 7. The two shapes of `value` that are not here

pandas accepts four shapes in `value` and this accepts two of them.

A dict on a frame names columns and is here. A dict on a column names row labels, because a column has no columns left to name, and it is not here. So is a `Series` passed to either class, and a `DataFrame` passed to a frame, and all three are the same shape of work: the fallback is not a value, it is a set of labelled values, and before a single one of them can be used it has to be lined up against this object's own labels. That is an alignment, the library has one and it is called `reindex`, and building the fallback out of the mapping and reindexing it onto this object's index is what the refusal message tells the caller to write.

The reason for leaving it out is scope rather than difficulty, and it is worth writing down that the composition is real, since the entry for `drop` in the last slice made the same kind of claim about a kernel that does not exist and this one is about a path that does. A caller who wants `s.fillna({"b": 7.0})` can build a column from the mapping, reindex it onto `s.index`, and pass it to a fill that takes a column, and the only piece of that which the library does not currently hand them is the last step, since `fill_null` is bound but takes only one row or all of them. Making the mapping forms work is one more branch in `_fill_values` and a reindex, and it is the obvious next thing here.

A key the frame does not have is dropped rather than complained about, which is the opposite of what `drop` does with a name that is not there. The difference is what the caller was doing. `drop` was asked to remove something and the something is not there, so the call did not do what it said. `fillna` was offered a value for a column, and a column that does not exist has no gaps to fill, so nothing was left undone.

## 8. The one type whose fallback cannot be named

Everything above treats the fallback as something that can be built from two things, the value the caller wrote and the name of the column's type. That is true of every type here except one. A category column stores codes, a code is a position in a list, and the list is carried by the column rather than by the type, so two category columns are only the same type if they carry the same list. The core says exactly that when it is handed a pair that does not match, in the words "the two columns do not have the same categories, and a code is a position in a category list", and it is right to refuse, because putting one column's code into another column's list would silently answer a different value.

So the fallback for a category column is built as a category of its own and then told to carry this column's list, which is where the code it needs comes from. That is one call to `set_categories`, it keeps the column's own ordering flag, and it is the only branch in this method that has to look at a column rather than at a schema. It is also the one place the method pays for a copy of a column, and it pays only when that column actually has a gap in it, so a frame of a hundred category columns with nothing missing still costs nothing.

The check on the value changes shape here for the same reason. There is no kind to test against, because a category column does not hold a kind, it holds what its list says it holds and nothing else. So the value is checked against the list, and a value that is not on it raises pandas' own sentence, which is that you cannot set a new category on a categorical and have to set the categories first. pandas says that because of the codes, and this says it because of the codes, which is the rarer sort of agreement: the two libraries refuse the same call for the same underlying reason rather than one of them copying the other's message.

## 9. A NaN is missing, and the kernel did not think so

The first run of this against pandas failed on every float frame in the corpus, and the difference was one row. A float column in the suite carries the six float edges at its first six offsets and a NaN is one of them, pandas filled it and this did not.

The rule the library already had is the right one and it was written down in `Series.null_count`, which counts the cleared validity bits plus the NaNs and says in its own docstring that this is the line between the two halves of the library. An `Array` is Arrow and answers what is in the buffers. A `Series` is pandas and answers what pandas would say. `isna` already followed that rule and so did `dropna`, and `fill_null` did not, which made a column that reported two missing rows come back from a fill with one of them still missing. That is worse than either answer on its own, because the two calls disagreed about the same column.

The kernel was not wrong. A coalesce reads a validity bit and nothing else, that is Arrow's question, and it is also SQL's, where `COALESCE` over a NaN answers the NaN because a NaN is a value there. Changing the kernel would have made a fill agree with pandas and made the SQL engine disagree with every other database.

So the fix went where the rule lives. `Series.fill_null` clears the validity bits of the NaN rows before it calls the kernel, using `present_bitmap`, which is the one function in the library that knows what missing means for a float. It costs one pass over the column, the same pass `null_count` was already paying for, and it does not copy the values, because a buffer here is shared until something writes through it and only the bitmap beside them is new. `Frame.fill_null` was rewritten to go through the series call rather than to the kernel, so that there is one place this happens rather than two.

## 10. What this does not do

No `limit`, for the reason in section 6. No mapping onto row labels and no fallback that carries rows, for the reason in section 7. No object dtype, so no answer where pandas widens, for the reason in section 4.

No `inplace`, which is the standing answer and is now the reason eight names on the conformance board stop one level below where their cases would otherwise carry them. That count goes up by two with this change and will keep going up until the divergence family is looked at as a group.

And no fill from a column of the same height, even though the binding takes one and the core has taken one all along. `s.fill_null(other)` where `other` is as tall as `s` is the general form and the one row case is the special one, and the only reason the general form is not reachable from `fillna` is that pandas spells it with a `Series` argument, which is the aligned shape section 7 leaves out. So the capability is bound and the door to it is the same door the mapping forms are waiting behind, which is one more reason to open it next.
