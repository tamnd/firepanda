# 38. Which row of a repeat is the one that stays

Status: implemented. One kernel, one new frame method, one new parameter on a method that already existed, and a bug in the index that the new method found.

## 1. Two methods and one question

`drop_duplicates` has been in the core since the factorize was written. `duplicated` is the same question with the answer handed back instead of applied: which rows repeat a key that another row already carries. A caller who wants the repeats rather than the survivors cannot get them out of `drop_duplicates`, because the rows that went are gone and their positions with them, and reconstructing them by comparing the frame before against the frame after is a join written by hand.

So the two are one piece of work and they are written as one. `_duplicate_mask` computes the mask, `duplicated` returns it and `drop_duplicates` inverts it and filters. There is no path where one of them can be right and the other wrong, which is a property worth having, because a library where the mask and the drop disagree has two bugs and no way to tell which one is the bug.

The same reorganisation added `keep` to `drop_duplicates`, which had only ever kept the first appearance. It had to: the parameter is the whole of what the mask decides, and a `duplicated` that took three rules next to a `drop_duplicates` that took one would have been two methods again.

## 2. The three rules are one table read twice

`keep="first"` marks a row when an equal row came before it. `keep="last"` marks a row when an equal row comes after it. `keep=False`, which the core spells `"none"`, marks every row of a repeated key including the one the other two rules spare.

These are three questions about the same thing and they are one loop over one table rather than three kernels. The table is `groups` entries wide. Under `first` it holds the earliest row each ordinal appears at, under `last` the latest, and under `none` how many rows carry the ordinal. The second pass then asks one question per row: under the first two, whether this row is the row in the table, and under the third, whether the count is more than one.

`first` could be done in one pass, marking each row as it is read, and it is not. The other two cannot be, so a single pass version of `first` would be a special case in the middle of a loop that is otherwise the same three times, and what it would save is one read of an array that is `groups` wide and was in cache from the pass that filled it. A shape that is the same three times is worth more than that.

## 3. The kernel reads the codes and nothing else

`Grouping` already holds `rows_at`, which is the first row of each ordinal, which is exactly what `keep="first"` wants. The kernel does not use it and builds its own table instead.

Two reasons. Only one of the three rules could use it, so the other two build a table regardless and the saving is available in a third of the calls. And a kernel that took a `Grouping` could only be tested by producing a `Grouping`, which means testing the duplicate rule through the factorize. Taking the codes alone means the test writes the ordinals out by hand, so a failure is a failure of the rule rather than a failure of whatever produced the codes.

## 4. Both passes are serial, and that is not an oversight

Under `keep="first"` the answer for a row depends on whether any earlier row carried its ordinal, which is a prefix question. Splitting the rows across workers would need each worker's table merged back in row order afterwards, and the merge is `groups` wide per worker.

What the pass costs is one integer load, one indexed load and one store per row. What has already happened by the time it runs is a factorize, which hashed every key. The parallel version would cost more in the merge than the serial version costs in total, and it would do it in a kernel that only ever runs after something far more expensive.

## 5. A null is a value here, and it is not in `group_by`

`group_by` drops the rows whose key is missing, because that is pandas' rule there and `dropna=True` is its default. This is not that rule. Two rows that are both missing the same field are two rows that say the same thing, so they repeat each other, and one of them goes.

pandas does this and Polars does this, and it is the one place in either library where a missing value compares equal to a missing value. It is worth saying out loud because it contradicts the rule everywhere else, and because a reader who knows the everywhere else rule will assume this method follows it.

The factorize already treats a null as an ordinal of its own rather than skipping it, so nothing had to be written for this. What had to be written is a test, in both the core and the pandas layer, because behaviour that falls out of a dependency is behaviour that can stop falling out.

## 6. The boundary carries a word for all three rules

pandas writes two of the three rules as strings and the third as the bool `False`. That is a spelling and not a distinction: all three answer the same question, and a caller who writes `keep=False` has not asked for something of a different kind from a caller who writes `keep="last"`.

A boundary that carried the pandas spelling would be sending a Python type across to say which of three branches to take, and the core would have to read a bool to find out whether to read a string. So the mixin turns `False` into `"none"` before the crossing, one kind of thing goes across, and the core reads one kind of thing.

What is checked on the way in is membership in a tuple, which is the form pandas uses, and copying the form rather than the intent is deliberate. `0 == False` in Python, so a zero is in that tuple, so pandas takes `keep=0` and reads it as the third rule. Writing the check as `keep is False` would be tidier and would refuse a call that a caller's pandas accepts, which is the wrong trade for a compatibility layer, so the accident is copied along with the rule. The message on the refusal is pandas' own text as well, since a caller reading it is reading it out of a traceback and has no way to know which library wrote it.

## 7. Where the subset default lives, and what it does with a repeat

`subset=None` means every column, and it is resolved in the Python layer rather than sent across as an absence for the core to fill in. Which columns a frame has is a question that side can already ask, and a default that lives in two places is a default that will eventually disagree with itself.

A bare name rather than a list is one column, which is pandas' rule and comes from `is_list_like` answering False for a string. It is worth having, because `subset="key"` is what a caller writes first and iterating the string would ask for one column per letter.

A name written twice is where the two layers part. The core refuses it, on the grounds that a caller writing the subset out by hand and repeating a name meant something else. pandas accepts it and answers as if it were written once. Both are right for their own caller, so the pandas layer drops the repeat before the crossing and the core keeps its refusal for the Mojo caller. Nothing is lost either way, since a key column compared against itself twice tells the same rows apart as a key column compared once.

## 8. The mask has no name

pandas leaves the result of `duplicated` unnamed, and the core does too. The mask is an answer about the rows and not a column of the frame, so there is no name it could carry that would not read like a column somebody had added. This is the one place where the core's name was chosen by what the pandas layer needed rather than by what reads well in Mojo, and the alternative was relabelling at the crossing, which would have put two different names on one array depending on which door it came out of.

## 9. `ignore_index` is honoured and `inplace` is not

The labels of what survives a drop have gaps in them wherever a row went, because they are the positions the rows held in the frame before the drop. That is the one thing a drop leaves behind that a caller may not want, and numbering them again is `reset_index`, which already exists. So `ignore_index=True` is implemented rather than refused, and it is two lines.

`inplace=True` is refused, the way it is refused everywhere else in the library, because every operation here answers a new frame over Arrow buffers that are shared rather than owned. It is listed in document 35 section 7 among the refusals that are a decision about the whole library rather than about any one method.

A frame with no columns is a third case and it is answered rather than refused. The core says a frame with no columns has no rows to tell apart, which is true and is the right thing to tell a Mojo caller. pandas makes the same statement quietly by answering an empty mask, and the pandas layer does that, because raising on a frame where nothing went wrong is not compatibility.

## 10. The bug the mask found

`DataFrame.add_column` updated the frame's height and never its index. A frame read in from Arrow gets its labels when it is read, so nothing noticed. A frame built up by calling `with_column` on an empty frame ended up with a height of three and an index of nothing, and `DataFrame.column` handed that empty index straight on to every series it produced.

`duplicated` is what made it visible, because the mask carries the frame's labels and a mask whose labels are missing cannot be handed back to `filter`, which is the whole of what a caller does with it. The fix is three lines in `add_column`: when the index and the height disagree, build a default index of the height.

It can only fire once, on the first column, because the length check above it refuses a column of the wrong height once there is one to compare against. And it cannot overwrite labels somebody set, because a frame with no rows has no labels to set. It is worth writing down rather than leaving as an obvious guard, since the guard looks like it could clobber an index and the reason it cannot is two steps away.
