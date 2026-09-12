# 57. How big a frame is and what it holds

## 1. The cheapest members in the library

`empty`, `ndim`, `size`, `shape`, `axes` and `dtypes` are six properties, none of which touches a value, none of which can fail, and every one of which is one line. There is no algorithm here and there is nothing to benchmark. The reason they get a document is that the cheap members are the ones nobody thinks about, and they are the first ones anything else reads.

That is not a figure of speech. A library handed a frame it did not make asks `ndim` to find out what it is, asks `shape` to find out how big, asks `dtypes` to find out what is in each column, and asks `empty` before doing anything at all. Every one of those is a member that does no work, and a frame that cannot answer them is a frame that cannot be passed to anything. Four of the six were missing here, so the answer to all four was an `AttributeError`, which reads to a caller as this not being a dataframe.

`shape` was already here on both classes and `size` was already here on a column. This document is about the other four and about the two rules inside them that are not obvious.

## 2. `empty` is about axes and not about rows

`df.empty` is not `len(df) == 0`. It is whether either axis has nothing in it, so a frame of two columns holding no rows is empty and so is a frame of no columns at all, and both of those have a `shape` with a zero in it.

The name is the problem. `empty` reads as a question about rows, and on a column that is what it is, because a column has one axis and there is only one way for it to be empty. On a frame it is a question about both axes and the answer is the or of them. Writing it as a row count would be right on every frame anybody builds by hand and wrong on the frame that comes out of dropping every column, which is exactly the frame the caller is checking for.

pandas defines it as `any(len(axis) == 0 for axis in self.axes)` and that is the definition copied here, spelt out rather than routed through `axes` because two lengths do not need a list built to hold them.

## 3. `size` counts cells on a frame and rows on a column

`df.size` is rows times columns, so a frame of three rows and three columns answers nine. `s.size` is the number of rows. `len(df)` is the number of rows on both.

A caller who reads `size` as a row count is right on a column, wrong on a frame, and gets a plausible number either way. There is no error available to catch that, so the only thing to do about it is to say it out loud. It is pandas' rule and the same rule numpy has, where `size` is the number of elements and the elements of a two dimensional thing are its cells.

## 4. `axes` answers a list of one on a class with one axis

`s.axes` is `[s.index]`. A list of one, on a class where `index` already answers the thing in it.

That reads like a member that should not exist, and the shape is the reason it does. Code that is handed either class and wants to walk whatever axes it has writes `for axis in obj.axes`, and that loop works on a frame and on a column without asking which one it holds. The list of one is what makes a column answer that loop rather than being a special case in it. A member that is redundant on one class and load bearing on the other is worth having on both, and pandas has it on both for this reason.

The order on a frame is rows first and columns second. That is the order `shape` reads in, the order `axis=0` and `axis=1` name them in, and the order `df.axes[1]` had better mean the columns in. There is only one defensible order and it is this one.

The second entry is a list here where pandas hands back an `Index`. That is the divergence `columns` already carries rather than a new one, and it is now visible in a third place, `columns`, `keys` and here. Document 21 section 7 has the general form of it and fixing it means giving a frame an index of column names, which is a change to what `columns` answers.

## 5. `dtypes` is a column rather than a list

`df.dtypes` hands back a column of the type names, labelled by the name of the column each one belongs to.

A list would have been less code and it would have been the wrong shape. The reason to ask a frame of forty columns what it holds is to read the names off beside the types, and a list makes the caller line the two up by hand against `columns`, which is a zip they will get wrong once. A column carries both halves and prints both halves, and it is also the shape that lets `df.dtypes[name]` work, which is the line people actually write.

The cost is that a column cannot be built with an index on it, so this makes a frame of two columns and moves one of them into the index, which is the route `_labelled` takes and is the same route every labelled answer in this library takes. That route leaves two names behind, `values` on the column and `labels` on the index, and neither of those is a name a caller asked for. Both are taken off. pandas leaves both empty and so does this, and it is worth saying because a name that arrives from an implementation detail is the kind of thing that gets asserted against by accident and then cannot be changed.

`s.dtypes` is `s.dtype`. The plural name on a thing with one type is odd and is pandas', and it is here for the same reason `axes` is: so a caller holding either class asks the same question and gets an answer shaped like the class they have.

## 6. The types are strings and pandas' are objects

`df.dtypes` here gives a column of strings. pandas gives a column of numpy dtypes and extension dtypes, which print as their own names.

This is the divergence `Series.dtype` has carried since it was written rather than a new one, and it is registered where that member is. It is worth naming here because it is more visible in this shape than in the singular one: a column of strings and a column of dtype objects print almost identically, so the difference is invisible until somebody writes `df.dtypes[name] == np.int64` and gets false.

Under `str` the two agree on every type but text, where pandas 3 says `str` and this says `string`. That one is also `dtype`'s and not this member's.

## 7. `nbytes` is not here, and on a frame it is not there either

`Series.nbytes` exists in pandas and `DataFrame.nbytes` does not, which is worth checking rather than assuming, and it was checked: `df.nbytes` raises `AttributeError` in pandas 3. So the frame's half of this is not a gap.

The column's half is a gap and it is left open on purpose. `Index.nbytes` here answers the bytes in the Arrow buffers, which is the honest number for this library. pandas answers the size of the numpy representation, so a column of three strings is 24 bytes over there, which is three pointers, and the strings themselves are not counted. The two numbers are not the same measurement and they never will be, so adding `Series.nbytes` means registering a divergence, and the registry says a divergence entry is its own change reviewed on its own. It is a slice with one property in it and one paragraph of argument, and it is not this one.

## 8. What is refused and what is left

Nothing here refuses anything. There is no parameter to accept and none of the six can fail, which is the property that makes them worth having: a member that cannot raise is a member anything can call.

What is left over, named so it does not have to be found again:

`nbytes` on a column, for the reason section 7 gives, behind a divergence entry.

`values` and `array` on a column, and `values` on a frame. `Index.values` exists already. These are not shape members, they are the door out to numpy, and what comes back through that door is the whole question rather than a line.

`info` and `memory_usage`, which are the two members that report this kind of thing in a shape a person reads rather than a shape a program reads. `memory_usage` is `nbytes` per column plus the index and carries the same divergence. `info` prints and is a formatting problem.

`flags` and `attrs`, which are metadata a frame carries rather than facts about its contents.

One thing found while measuring and not fixed here, because it belongs to display rather than to shape: a column carrying text labels prints positions rather than labels. `df.dtypes` shows `0`, `1`, `2` down the side where its index holds `a`, `b`, `c`, and `.index.tolist()` gives the right answer the whole time. It is not this member's bug, it is every labelled column's bug, and it is filed on its own as issue 719.
