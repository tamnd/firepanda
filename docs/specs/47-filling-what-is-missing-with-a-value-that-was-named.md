# Filling what is missing with a value that was named

## 1. A method with almost no method in it

`fillna` is one of the most written lines in pandas and the operation under it is the smallest one in this whole section. Take a column, take a second column, and for every row read the first one's validity bit and pick a side. That is a coalesce, it is one pass with no branch that depends on a value, and `firepanda/frame/series.mojo` has had it since the fill family was written. `Series.fill_null` takes a second series and `Frame.fill_null` takes a column name and a second series, both of them accept a fallback one row tall and treat it as the value for every missing row, and the docstring on the series one already said in as many words that a one row fallback is how filling with a scalar is spelled.

So the kernel was there, the broadcast was there, and the method did not exist, because neither call had a binding. It is the only member of the fill family that shipped without one: `ffill` and `bfill` both have bindings and both are reachable from Python, and the one that a caller reaches for first was reachable from Mojo and from nowhere else. That is the third time in five slices that the surface has been found lagging a finished implementation, and it is worth saying out loud that the failure mode is silence. Nothing goes red when a binding is missing. The method is simply not there, and only something that goes looking notices.

What this document is about, then, is not the fill. It is the two questions that have to be answered before the fill can run, and both of them are about types.

## 2. What is here

`DataFrame.fillna` and `Series.fillna`, with the pandas 3.0 signature, which is `value` positional and required followed by `axis`, `inplace` and `limit` as keywords. `method` and `downcast` are gone from pandas 3.0 and are not declared here, because declaring a parameter that the library being copied has deleted would be preserving a mistake.

`value` is taken in every shape pandas takes it. A single value on either class, a dict of column name to value or a column read along the columns on a frame, a dict of row label to value or a column read down the rows on a column, and a frame lined up on both axes on a frame. `axis` is read and not used. `inplace` and `limit` are refused. A frame handed to a column is refused in pandas' own words, because there is nothing on a column for a second axis to line up against.

Two new bindings over the two core calls, and one more over a thing the frame could not previously be asked. That third one is `null_counts`, which answers how many rows are missing from each column, and section 5 says why the method needs it. The aligned shapes added one parameter to a binding that already existed rather than a fourth binding, and section 8 is about what that parameter decides.

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

## 7. The four shapes of `value`, and the one that reads backwards

pandas accepts four shapes in `value` and all four are here. Three of them name columns and one of them is read down the rows, and which is which is the part worth reading twice, because it is not the part anyone guesses.

A dict on a frame names columns. A column handed to a frame names columns too, and this is the one that reads backwards: its labels are column names and each of its values is the value for that whole column, so `df.fillna(s)` is the dict written another way and has nothing to do with lining `s` up against the frame's rows. A frame handed to a frame is the one that lines up on both axes, and a cell it does not carry is left missing rather than filled. On a column there are no columns left to name, so a dict there names row labels, and a column handed to a column is that same mapping with a type on it. A frame handed to a column is the one call with no meaning, and pandas says so in a sentence naming the three shapes it would have taken, which is the sentence this raises.

Every one of the aligned shapes is the same work underneath. The fallback is not a value, it is a set of labelled values, and before a single one of them can be used it has to be lined up against this object's own labels. That is an alignment and the library has had one all along, so the shapes cost a call to it rather than a kernel. A row this object has and the fallback does not stays missing, a row the fallback has and this object does not is not read at all, and a row the fallback has and holds nothing in stays missing too, which is the same rule written from the other side.

There is one alignment pandas performs that this refuses, and it is a label that appears twice. pandas answers a duplicated label with the second value it finds. The lookup here refuses, in the sentence pandas itself gives when a repeated label is looked up, which is that you cannot reindex on an axis with duplicate labels. That is the one place in this section where the two libraries answer differently, and it is a refusal rather than a wrong row.

A key the frame does not have is dropped rather than complained about, which is the opposite of what `drop` does with a name that is not there. The difference is what the caller was doing. `drop` was asked to remove something and the something is not there, so the call did not do what it said. `fillna` was offered a value for a column, and a column that does not exist has no gaps to fill, so nothing was left undone.

## 8. An alignment is a reindex with one decision made the other way

The alignment above is `reindex` and it could not be used as it stood, and the reason is a single line in the core that is right where it is and wrong here.

`Series.reindex` widens an integer column to float when a label was not found. That is pandas' answer and it is the correct answer for a reindex, because pandas has one missing value for a number and it is a NaN, and the thing the caller gets back from a reindex is a column they are about to look at. The answer here is not something anybody looks at. It is going straight back into a column whose type nobody changed, through a coalesce that refuses two columns of different types, so a fallback that widened on the way in cannot be used at all, and a fallback that did not widen carries a gap in the type that is already there, which is exactly what an Arrow column has a validity bitmap for.

So `reindex` gained a `widen` flag rather than a sibling method, because it is the same operation with one decision made the other way and splitting it in two would have meant two places to fix the next time the lookup changes. The pandas facing method passes True and is what it always was. The alignment inside a fill passes False. The flag is on the series overload only, since that is the one a fill uses, and the frame's own reindex is untouched.

It is worth naming what the alternative was, because it looked cheaper and it was not. The fallback could have been aligned in the widened type and then cast back, and that fails on the one case the whole thing exists for: a cast from float to a whole number refuses a column with a gap in it, which is `IntCastingNaNError` and is correct, and the aligned fallback is nothing but gaps and values. Building the column row by row out of Python objects would have worked and is a loop over boxed values in the middle of an operation that is otherwise two passes. One flag on a lookup that already existed is the cheaper answer and it is also the more honest one, since the question the flag asks is a real question and the two callers really do answer it differently.

## 9. A fallback is judged by the rows it is read from

A fallback that carries rows has to be the column's type before it can be used, and the check on it is not the check a scalar gets. A scalar is one value and either the column can hold it or it cannot. A column of values is many, and pandas only looks at the ones it actually takes.

That was measured rather than assumed. `pd.Series([1, None], dtype="Int64").fillna(pd.Series([2.5, 2.0]))` answers `[1, 2]`, because the only row it read was the second one and that one is a whole number. Turn the fallback around so the fraction sits in the row that is missing and the same call raises `TypeError: cannot safely cast non-equivalent object to int64`. The same is true of a word that is not a number: unreadable in a row nobody reads is not an error, and unreadable in a row that is read gives `ValueError: invalid literal for int() with base 10`. So the rule is not that a fallback has to be castable, it is that the rows a fill takes out of it have to be.

This follows that rule exactly, and the way it does is one more use of the alignment. The fallback is lined up against the rows being filled before anything is decided about its type, the rows that are not missing are dropped, and what is left is the values the fill is going to use. Those are the rows the check runs on. The cast that actually produces the fallback then runs over all of them and is not the strict one, because a row nobody is going to read is allowed to be nonsense.

The check itself is the same argument section 3 makes, one step along. The kinds are checked above the cast, so a column of numbers still cannot fill a column of words and a column of words can fill a column of numbers, which is pandas both times. Then the cast between two numbers is checked rather than trusted, because a strict cast here still truncates a fraction and still wraps a number too big for the width, and those are the two ways a fill can put a value in a row that is not the value that was offered. The way to catch both without writing a new kernel is to turn the cast around and compare, which is two passes and one comparison over the rows that are going to be used, and it answers in one reduction rather than a loop.

## 10. The one type whose fallback cannot be named

Everything above treats the fallback as something that can be built from two things, the value the caller wrote and the name of the column's type. That is true of every type here except one. A category column stores codes, a code is a position in a list, and the list is carried by the column rather than by the type, so two category columns are only the same type if they carry the same list. The core says exactly that when it is handed a pair that does not match, in the words "the two columns do not have the same categories, and a code is a position in a category list", and it is right to refuse, because putting one column's code into another column's list would silently answer a different value.

So the fallback for a category column is built as a category of its own and then told to carry this column's list, which is where the code it needs comes from. That is one call to `set_categories`, it keeps the column's own ordering flag, and it is the only branch in this method that has to look at a column rather than at a schema. It is also the one place the method pays for a copy of a column, and it pays only when that column actually has a gap in it, so a frame of a hundred category columns with nothing missing still costs nothing.

The check on the value changes shape here for the same reason. There is no kind to test against, because a category column does not hold a kind, it holds what its list says it holds and nothing else. So the value is checked against the list, and a value that is not on it raises pandas' own sentence, which is that you cannot set a new category on a categorical and have to set the categories first. pandas says that because of the codes, and this says it because of the codes, which is the rarer sort of agreement: the two libraries refuse the same call for the same underlying reason rather than one of them copying the other's message.

A fallback that carries rows has all of that and one more question, because it may be a category already. Two category columns are the same type only when they carry the same list, and pandas refuses that pair outright rather than recoding it, in a different sentence which says you cannot set a categorical with another without identical categories. This refuses it too and in those words. A fallback of words is recoded instead, which is what pandas does with one, and the check for a word that fell off the list is read off the count of gaps rather than by looking at the values, since `set_categories` puts a gap wherever a value has no code. That count is taken over the rows the fill is going to read, for the reason section 9 gives, so a word off the list sitting in a row nobody reads is nobody's business here and is nobody's business over there.

## 11. A NaN is missing, and the kernel did not think so

The first run of this against pandas failed on every float frame in the corpus, and the difference was one row. A float column in the suite carries the six float edges at its first six offsets and a NaN is one of them, pandas filled it and this did not.

The rule the library already had is the right one and it was written down in `Series.null_count`, which counts the cleared validity bits plus the NaNs and says in its own docstring that this is the line between the two halves of the library. An `Array` is Arrow and answers what is in the buffers. A `Series` is pandas and answers what pandas would say. `isna` already followed that rule and so did `dropna`, and `fill_null` did not, which made a column that reported two missing rows come back from a fill with one of them still missing. That is worse than either answer on its own, because the two calls disagreed about the same column.

The kernel was not wrong. A coalesce reads a validity bit and nothing else, that is Arrow's question, and it is also SQL's, where `COALESCE` over a NaN answers the NaN because a NaN is a value there. Changing the kernel would have made a fill agree with pandas and made the SQL engine disagree with every other database.

So the fix went where the rule lives. `Series.fill_null` clears the validity bits of the NaN rows before it calls the kernel, using `present_bitmap`, which is the one function in the library that knows what missing means for a float. It costs one pass over the column, the same pass `null_count` was already paying for, and it does not copy the values, because a buffer here is shared until something writes through it and only the bitmap beside them is new. `Frame.fill_null` was rewritten to go through the series call rather than to the kernel, so that there is one place this happens rather than two.

The same disagreement had a second home one level up, and it survived that fix because it never reached the core. Section 5 has the frame's method deciding which columns have work in them from `null_counts`, which reads validity and nothing else, and that decision is made before anything is handed down. A frame whose float column held a NaN and no cleared bit was therefore handed straight back, with the column the core would have filled untouched, while the same column asked on its own was filled. The counts are still where that question starts, because they are free and because they are the whole answer for every type that cannot hold a NaN, and a float column the bits call complete is now asked again before it is skipped. That second question is the one place in the method that reads values before it knows there is work to do, and it is asked for float columns only, only when the bitmap says there is nothing to do, and never for a frame that has no float column in it.

## 12. What this does not do

No `limit`, for the reason in section 6. No object dtype, so no answer where pandas widens, for the reason in section 4. No fallback whose labels repeat, for the reason in section 7, which is the one divergence this method has that is not the object dtype.

No `inplace`, which is the standing answer and is the reason both of these names stop one level below where their cases would otherwise carry them. Ten names on the conformance board are in that position now and it will keep going up until the divergence family is looked at as a group.

There is nothing left in `value` after this. All four shapes are here, the general fill from a column of the same height that the core has taken all along is reachable from Python, and what the method still does not do is a parameter rather than a shape.
