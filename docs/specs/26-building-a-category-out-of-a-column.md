# 26. Building a category out of a column

The two documents before this one taught firepanda to read a dictionary encoded column and to write one back out. Neither of them could make one. Every categorical in the library up to here arrived from somewhere else, out of an Arrow producer or out of the CSV reader, and `astype("category")` was refused with a message saying that building the dictionary is a conversion of its own rather than a change of layout. That message was accurate and it was also the whole of the reason, so this document is about writing the conversion it named.

## Why it is not in the cast kernel with the others

Everything else in `cast.mojo` reads a value and writes the same value in another layout. Row 4 converts without looking at row 3, which is why the loop parallelises over morsels and why the null-is-zero invariant means the validity bitmap comes across untouched.

Dictionary encoding cannot work that way. What code row 4 gets depends on every row before it, because the code is a position in a list that is being built while the column is being read. So it is a kernel of its own, in `firepanda/kernel/dictionary.mojo`, and `cast_any` calls into it rather than growing a special case in the middle of a loop that has no branches in it on purpose.

## The categories come out sorted

Factorize hands back groups in the order they were first seen, which is the cheap order and is not the order that comes out of here. The encoder sorts the distinct values and then rewrites the codes to point at the sorted positions, which costs a sort over the distinct values rather than over the column.

That is worth paying for because the order is not an implementation detail a caller cannot see. `Series.cat.categories` prints it. A groupby over a category column produces its groups in category order. An ordered categorical compares by position, so `min()` on an ordered column is a question about the category order and nothing else. A library that encoded in first appearance order would give a different answer to the same question on the same data depending on how the rows happened to be arranged in the file, which is the kind of difference that is invisible in a test with four rows in it and wrong in production.

## A null is not a category

Arrow says a dictionary column carries its missing values in the codes buffer's validity, exactly as an integer column does. Pandas agrees from the other direction: a `NaN` in a categorical is not a category, does not appear in `.cat.categories`, and is not counted by `value_counts` unless it is asked for. So a null row comes out a null code, and the categories are the distinct values that were actually there.

The arithmetic is the part worth testing rather than reading. Factorize gives the null group ordinal zero when there is one and pushes every real group up by one, so the encoder takes one off before indexing into the sorted ranking, and takes nothing off when there were no nulls. That is one number and it is the number that would go quietly wrong, which is why the test suite asserts the codes themselves and not only the values they stand for.

The empty string is a category. It is a value somebody wrote nothing into rather than a value nobody wrote, the string column already went to some trouble to keep those two apart, and this kernel has no business collapsing them.

## Casting off a category

The other direction matters as much and is easier to get wrong. A dictionary column's physical dtype is its index type, so `astype("int64")` on a category column would have found the int64 source arm in the number path and handed back the codes. Those are integers and they look like an answer.

So both overloads of `cast_any` check for a dictionary source before anything else and decode it first. That is one more pass than a clever version would take, which would cast the categories and then reindex, and it is the version that gives the right answer with no case analysis: whatever the target is, the values are what the codes stand for and not the codes.

A category cast to a category is a copy rather than a decode and a re-encode. That is not only faster, it is different: a re-encode would drop the categories nobody used and would resort what was left, and pandas keeps both across an `astype("category")` on something that already is one.

## Categories that are not text are refused

Firepanda holds categories in a `StringArray`, so a numeric column has nowhere for its categories to go. Rendering the numbers as text first would produce a column that is a category column in every way except the one that matters, which is what `.cat.categories` says it holds: pandas would report int64 there and firepanda would report text, so the values would round trip and the dtype would not, and the caller would find out about it somewhere else entirely.

So it is refused, and the refusal says "not supported", which under document 14's rule makes it a `NotImplementedError` rather than a `ValueError`. Pandas does this, the data is fine, and firepanda has not written the case. That is a gap and it says so. It is the same shape of answer that spec 24 gives to a dictionary arriving over Arrow with categories that are not text, and it has the same fix, which is a category type that can hold something other than strings.

## Naming a type that carries more than a name

`named_type` reads the spellings that are a single word, and its docstring used to say that a dictionary is left out because it carries an index width and an ordering that a name has no room for. Both halves of that are still true and `category` is in the table anyway, because it is how pandas asks for one and because the two things the name cannot say now have answers that do not come from the caller. The width is int32, which is what the encoder builds. The ordering is false, which is what a bare `astype("category")` gives in pandas as well. A caller who wants an ordering is asking for something a type name cannot express in pandas either, and expresses it with a `CategoricalDtype`, which is the next piece of this rather than a decision.

## What this makes possible

`astype("category")` works on a text column, on a series and on a frame, and the result exports over Arrow as a real dictionary encoded column because of spec 25. So a caller can build a categorical in firepanda, hand it to pandas, and get a `Categorical` back.

What is still missing is what a caller does with it after that. There is no `.cat` namespace, no `codes` or `categories` on the Python series, no `CategoricalDtype`, and no way to say ordered. That is the next piece and it is now reachable, which it was not while the conversion did not exist.

## The index width

Arrow permits all eight integer widths for the codes and pandas writes int8 where it can, which saves three bytes a row on a column that is usually being encoded to save memory in the first place. Firepanda writes int32 and that is a real difference rather than an oversight.

It is deliberate for now for two reasons. The width is visible through the Arrow export, so narrowing it later changes what a consumer sees rather than being a private optimisation. And picking the width from the cardinality means the same column encodes to different types on different data, so a program that worked on a sample would produce a differently typed column on the full file. Both of those are conversations about the dtype vocabulary rather than about this kernel, and they belong with the `CategoricalDtype` work that has to happen anyway.
