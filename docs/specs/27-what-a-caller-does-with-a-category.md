# 27. What a caller does with a category

Document 26 taught firepanda to build a category column and left it there. `astype("category")` worked, the result exported over Arrow as a real dictionary encoded column, and a caller who had one could do nothing with it. There was no `cat` namespace, no way to read the categories or the codes, no way to say the order meant something, and no way to change what the categories were. This document is about the eleven names pandas puts on `Series.cat` and the three operations that turn out to be underneath them.

## Eleven names and three doors

The list looks long. `categories`, `codes` and `ordered` answer a value. `as_ordered` and `as_unordered` flip a flag. `add_categories`, `remove_categories`, `remove_unused_categories`, `rename_categories`, `reorder_categories` and `set_categories` all hand back a column. Written out that way it looks like eleven kernels.

It is three, and the way to see which three is to ask what each one does to the codes.

A rename is decided by position. Row 4 held category 2 before and holds category 2 after, and what changed is what category 2 is called. Not one code moves and not one row goes missing, so the whole operation is a new list of labels sitting where the old ones were.

Setting the categories is decided by value. The caller hands over a list, every row keeps the value it had if that value is in the list, and a row whose value is not in the list becomes null. Adding, removing, reordering and setting are all this one operation with the list worked out differently first: adding is the old list with more on the end, removing is the old list with some taken out, reordering is the old list shuffled, and setting is whatever the caller said.

Dropping the unused categories is decided by the codes. That is the third one and it is a door of its own rather than the second one called with the right list, because working out which categories are used is a pass over the column and the second door takes the answer as an argument. A caller doing this over the boundary would have to read every code out into Python to build the list to hand back, on a column that is dictionary encoded precisely because it is too big to want to do that to.

So `firepanda/kernel/dictionary.mojo` gained `rename_categories`, `set_categories` and `drop_unused_categories`, plus `set_ordered`, which is a fourth thing only in the sense that flipping a flag is an operation. The other eight names are arithmetic and the arithmetic is in `python/firepanda/_pandas.py`, where the pandas surface lives.

## Why the arithmetic is on the Python side

The eight are not merely calls with a list computed. Each one has an opinion about what a caller can get wrong, and the opinions are pandas' rather than the kernel's.

Adding a category that is already there is a `ValueError` and not a no op, and the message names the clash. Removing one that is not there is a `ValueError` too, because a typo in a category name is a mistake rather than a request to do nothing. Reordering has to name the same set, or it is a `set_categories` and the caller should have said so. Renaming needs one label per category, except when it does not, which is the next section.

None of that is a question about columns. All of it is a question about what pandas does, and the messages are the pandas messages word for word, because a program that catches the `ValueError` and matches on its text is a program that exists and there is nothing to be gained by writing a nicer sentence.

The same reasoning puts the shapes a caller can write on the Python side. `rename_categories` takes a list, a mapping or a callable, and pandas takes all three. A mapping names the categories that change and leaves the rest, and a key that is not a category is ignored rather than refused, which is worth knowing and is what pandas does. A callable is applied to each label. Two of those three are Python objects that could not cross the boundary anyway, and turning them into a list before the call is one line where a Mojo implementation of it would be a Python interpreter call per category.

## The one place a rename does not need the right number of labels

`rename_categories(["a", "b"])` on a column with three categories is a `ValueError`. `set_categories(["a", "b"], rename=True)` on the same column is not: the third category falls off the end and the rows that were in it become null. A longer list is allowed too, and the extra labels sit there as categories nothing uses.

That is pandas, it is documented there in one sentence, and it is the reason the count check is not in the kernel. Two callers reach the same door and they disagree about whether a mismatch is a mistake, so the check belongs to the one that thinks it is. The kernel takes whatever list it is given: a shorter one nulls the codes that no longer point anywhere, and a longer one keeps the codes and the extra labels.

This is also the one operation in the file where a positional rename can lose rows, which is worth saying plainly because the whole argument for a separate positional door was that it does not.

## Reading codes that are not int32

Document 26 decided that firepanda writes int32 codes and explained why. That decision is about what firepanda builds. It says nothing about what firepanda reads, and everything on this accessor reads.

A pandas categorical of three categories has int8 codes. Handed to firepanda over the Arrow C data interface it arrives with int8 codes, because the importer keeps what the producer wrote rather than normalising, which is right: normalising would copy a buffer that was handed over precisely so it would not have to be copied. So an imported categorical is int8, and an imported categorical is the common case rather than an exotic one, since importing from pandas is what a compatibility layer is for.

Before this document `decode_dictionary` asked for int32 and raised on anything else, which meant `astype("str")` on an imported categorical failed on a column that was perfectly well formed. Everything now reads through `dictionary_codes`, which widens whatever is there into int32 once, and the rest of the file is written against int32 only. The alternative was parametrising every function on the index type for a difference that stops mattering the moment the codes have been read.

The result of a rewrite is int32 regardless of what went in. That is a visible change to a column that arrived as int8 and it is the same decision document 26 already made, arrived at from the other side.

## Three differences from pandas, all of them deliberate

The width of the codes is the first. `Series.cat.codes.dtype` is int8 in pandas and int32 here, for the reasons document 26 gives. The values agree, so a program comparing codes to codes is fine and a program comparing dtypes is not.

The missing code is the second and it is the one worth staring at. pandas writes -1 for a row whose category is missing, because its codes are a numpy integer array and there is nowhere else to record absence. Firepanda's codes are an Arrow column with a validity bitmap, so the code is missing and reads back as `None`. A program that filters on `codes >= 0` is doing the pandas idiom and gets nothing here, and one that takes the mean of the codes gets a different answer in each library. Both of those are better found by a comparison that fails than by a number that is quietly wrong, so the test suite asserts the difference rather than hiding it.

The third is the name on the codes. pandas hands back a series with no name and `None` there; firepanda hands back one with no name and an empty string there. That is not about categories at all, it is that a firepanda series with no name reports an empty string everywhere, and it is filed rather than fixed here, because fixing it in one place would make this accessor disagree with the rest of the library.

## The accessor refuses the column rather than the call

`s.cat` on a column of numbers is an `AttributeError` in pandas and an `AttributeError` here, raised when the accessor is built rather than when a member is used. That is the opposite of what `dt` does, where the accessor is built for any column and the members complain, and the difference is not an inconsistency worth ironing out.

The reason is `hasattr`. A caller who guards with `hasattr(s, "cat")` should be handed a `False` rather than an exception, and that only works if building the accessor is what fails. It also reads better: somebody writing `s.cat` has already decided the column is a categorical, so finding out that it is not at the `.cat` is finding out at the first place they said something wrong.

## What is still missing

`CategoricalDtype` is not here. It is how pandas says both halves of a categorical's type at once, which a name cannot carry, so `astype(CategoricalDtype(["a", "b"], ordered=True))` has no spelling in firepanda yet. Everything it would do is now reachable, since `astype("category").cat.set_categories(names, ordered=True)` is the same thing in three calls, but that is not the same as having the name.

`Categorical` and `CategoricalIndex` are not here either, and both are bigger than a dtype object: one is a column type a caller can build without a series around it and the other is an index. `value_counts` on a category column, a groupby that reports every category rather than the ones with rows in them, and comparing two ordered categoricals are all further out again.

What has changed is that none of those is blocked on anything. The column exists, the codes are readable, the categories are readable and changeable, and the order can be turned on. Every one of the remaining pieces is a surface over what is now here.
