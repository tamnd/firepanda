# 28. Comparing a category

Document 27 put the eleven names on `Series.cat` and left one thing that a caller reaching for an ordered categorical wants and could not have. `s == "bolt"` raised. So did `s < "bolt"`, and so did comparing two category columns to each other. The message was the promotion refusal, which says that what two categoricals combine to depends on their categories and the categories are held by the column rather than by the type.

That message is right and it is in the right place. What was wrong is that a comparison ever reached it.

## A comparison does not need a promotion

Every other binary operation in firepanda works the same way. The two operand types are promoted to a common type, both sides are converted to it, and then the typed loop runs. That is why `int32 + float32` is float64 and why the answer's type depends on the operand types rather than on the values.

A category column has no common type with anything, including with another category column, and the reason is real rather than an omission. Two dictionary types compare equal whenever their index widths match, and two columns with equal types can hold entirely different categories, so a promotion that answered from the type alone would be answering with half the information. `promote` refuses, and it refuses every mixture including a dictionary against a dictionary, and it says which half is missing.

But a comparison does not need a type that both sides can be read as. It needs to know, for each row, which of two things is larger, and for a categorical the codes already say that. `s == "bolt"` is: find `bolt` in the categories, which gives a number, then compare every code against that number. `left < right` on two categoricals with the same categories is: compare the two code columns. Neither of those promotes anything, and both are integer comparisons over columns that already exist rather than text comparisons over columns that would have to be built.

So `binary_any` and `binary_value_any` grew an arm ahead of the call to `binary_type`, because `binary_type` reaches `promote`. Arithmetic falls through on purpose and still collects the refusal, which is the right answer for it: there is no reading of `category + category`.

## Six rules, all of them measured

Nothing here was designed. Each of these came off a running pandas and was written down.

Equality works whether the categories are ordered or not. An ordering comparison needs them ordered, and refuses with `Unordered Categoricals can only compare equality or not` when they are not. That is what `ordered=True` is for and it is the one thing a caller sets it to get.

A scalar that is not one of the categories is the asymmetry worth staring at. Under equality it is all false, because a value that is not a category is not equal to any row and there is nothing uncertain about that. Under an ordering it is a `TypeError`, because there is no position to compare against and the question has no answer. Both are pandas and both are what a careful reading would have produced anyway.

Two categoricals have to agree about their categories exactly, the same labels in the same order. A code is a position, so two columns whose categories hold the same labels in a different order give the same code to different values, and comparing the codes would answer confidently and wrongly. pandas refuses this and so does firepanda, with the pandas message.

A categorical against an ordinary text column compares by value under equality, which is the one shape in the whole file where a comparison has to decode, because there is no other common ground. Under an ordering it is refused, which is again pandas. Against a number column it is refused under both, because a categorical holds text and a caller comparing one against a number has made a mistake rather than asked a question whose answer happens to be false everywhere.

And the ordering follows the category order rather than the value order. A column of `small`, `medium` and `large` whose categories are in that order answers `medium < large` as true, where the same three words as plain text answer false, since `medium` sorts after `large`. That is the entire point of an ordered categorical and it is the test at the end of both test files.

## The way back in is not the front door

The decoding arm produces two ordinary text columns and a comparison, which is exactly the shape `binary_any` already knows how to answer, so the obvious line to write is a call back into `binary_any`. That line compiles and it is correct and it cost ten minutes.

Two module level functions calling each other is mutual recursion, and the Mojo compiler pays for it in a file that already monomorphizes thirteen operations over every dtype. Written as a call to `binary_any`, the binary test file went from a two second build to one that had not finished after ten minutes, measured directly against the same file on the commit before. Written as a call to `_compare_text_erased`, which is what `binary_any` reaches for two text columns and a comparison anyway, it is two seconds again.

This is worth writing down because the fix looks like a style preference and is not. A helper reached from the dispatch must not call the dispatch, and the shortest way to say that is that the arms go inward. The first guess at the cause was the import surface, on the theory that pulling the encoder's sort and hash table into the binary file was the cost, and splitting those functions into a module of their own changed nothing. Only the recursion did.

## Where the messages come from

All five are pandas' word for word. A program that catches the `TypeError` and matches on its text is a program that exists, and there is nothing to be gained by writing a nicer sentence for it to fail to match.

The one place firepanda's message differs is the type it names when an ordering meets a plain column. pandas names its own internal array class there, which firepanda has no equivalent of and no reason to invent, so the searchable opening of the sentence is kept and the type is firepanda's own. A caller reading it learns the same thing.

## Two orders of precedence

`s < "washer"` on an unordered column with no `washer` category is wrong twice. pandas reports the ordering problem, and so does firepanda, and it is worth having a test for rather than leaving to whichever check happens to be written first. A caller who has not asked for an order has a different problem from a caller whose scalar is a typo, and the first one is the one they need to fix.

## Nulls follow the library rather than pandas

`fp.Series(["bolt", None, "anchor"]) == "bolt"` is already `[True, None, False]` where pandas gives `[True, False, False]`. A comparison against a value that is not there has no answer, firepanda's result column is Arrow and has a validity bitmap to say so, and pandas' is a numpy bool array with nowhere to record it.

The categorical case inherits that whole and does not restate it. A category column answering differently from the text it decodes to would be the surprising thing, and the divergence a caller has to know about is one divergence rather than two.

## What this does not do

`min` and `max` on an ordered categorical, sorting by category order, and `Series.between` all follow from the same codes and none of them is written. Comparing two categoricals whose categories differ only in order is refused here the way pandas refuses it, rather than being reconciled by rewriting one side, which would be a defensible library to build and is not the one being built.
