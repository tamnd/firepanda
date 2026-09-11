# 42. A column that answers as a frame

Document 41 opened one door, from an index to a column, and counted what it was worth. This is the other door, from a column to a frame, and the argument is the same argument with different numbers. Read that one first if you have not, because everything here follows from it.

## 1. Thirteen methods the frame had and the column did not

The count that started this is one line of Python. Take the public members the generated `DataFrame` has, take the ones the generated `Series` has, and subtract. What comes back is thirteen names: `columns`, `drop_duplicates`, `duplicated`, `filter`, `groupby`, `nlargest`, `nsmallest`, `reset_index`, `select_dtypes`, `set_index`, `sort_index`, `take` and `truncate`.

Three of those are a frame's business and nothing else. `columns` is a list of column names and a column is one column. `select_dtypes` picks columns by type out of several. `set_index` takes one column and makes it the labels of the others, and a series has no others. Those are not gaps.

The other ten are all members of the pandas series, and every one of them was missing here. That is not a coincidence and it is not ten separate oversights. It is one missing thing, which is that a column could not be handed to a frame. Every one of the ten is a question about rows rather than about columns, the frame already answers all of them, and the answer for a column of one is the answer for a frame of one column with the column taken back out.

So this is not ten features. It is one door and ten lines.

## 2. Where the door had to live, again

`PySeries` cannot answer a `PyDataFrame`. `firepanda/py/frame.mojo` already imports `firepanda/py/series.mojo`, because a frame hands out columns, and Mojo will not take the cycle that would let the series import back. The alternative, a method on the frame that takes a column and hands back a frame of that column, reads backwards: it is a question about the column, and the frame it would live on is a frame that has nothing to do with the answer.

So `series_to_frame` is a free function in `frame.mojo`, beside `isocalendar` and `index_to_series`, which are there for exactly this reason and whose docstrings already say so. Three functions now sit in that position and the shape is worth naming: a door between two bound types belongs to neither of them, and the module is where it goes.

None of these three is a user entry point. They are registered under leading underscores, the Python layer is the only caller, and the pandas method a caller actually writes is the mixin method that calls them.

## 3. The labels come across

A frame built out of a column and read back has to be the same column. That means the row labels are copied over rather than left to the range a fresh frame would carry, and it is the reason `series_to_frame` is thirty lines rather than three.

It matters more than it sounds. Every one of the ten methods either removes rows, reorders them, or does both, and every one of them does the same thing to the labels that it does to the values. `sort_index` is the clearest case, since it is entirely about the labels and would sort nothing at all if they had been dropped on the way in. But `take`, `truncate`, `nlargest` and `drop_duplicates` all answer rows that have to keep the labels they had, and none of the ten has a line about the index anywhere, because the frame does all of it and the labels were there to be done to.

## 4. The name that is a number

pandas calls the column of an unnamed series `0`. Not the text `"0"`, the integer. A column name in this library is a `String`, and there is nowhere to put an integer.

So it is the text of it. `fp.Series([1, 2]).to_frame()` has a column called `"0"` where pandas has one called `0`, and a caller who writes `name=None` explicitly gets the same thing, where pandas gives a column literally named `None`. Both of those are the same divergence stated twice and both are recorded here rather than hidden: a column name here is a string, so every name that reaches a frame becomes one.

This is not free to fix and it is not worth fixing now. A column name that can hold any hashable is the same change as a series name that can hold any hashable, which is the change `firepanda/frame/series.mojo` line 136 is waiting for and which the row read across the columns is also waiting for. When that lands, both of these divergences close at once and nothing else has to move.

## 5. Ten members, of which nine are three lines and one is four

`_through` is the three lines: put the column in a frame under the name it already has, run the frame method, take the column back out under the same name. Nine of the ten are a call to it with a lambda.

`duplicated` is the tenth and it is four lines rather than three, because what a frame answers there is a mask rather than a frame. There is no column to take back out, so the mask is relabelled to the series' own name instead. pandas does the same thing: a frame's `duplicated` answers an unnamed mask and a series' `duplicated` answers a mask under the series' name.

`reset_index` is the one with two answers. Dropping the labels leaves a column and keeping them makes a frame of two, because the labels have become values and values in a second column are what a frame is. That is pandas' rule, it is not an invention here, and it is the reason the annotation on that one method is `Any` rather than `Series`.

## 6. And then the index gets three more

`Index.to_frame`, `Index.duplicated` and `Index.drop_duplicates` are two doors end to end. The labels become a column, which is document 41, and the column becomes a frame, which is this one. None of the three needed anything new.

`drop_duplicates` hands back the class the index already was, the same way `dropna` does, so dropping a repeated instant leaves a `DatetimeIndex` rather than a plain index. The rule was written twice in two changes and is now written once, in `_like`, which is where the next member that answers an index should go through.

`duplicated` on an index answers a list of bools where pandas answers a numpy array, which is the same shape `isna` and the rest of that family already have here, and is document 41 section 5's note rather than a new one.

## 7. What this is worth and what comes next

Ten members on the series and three on the index, and the three on the index score twice because `DatetimeIndex` subclasses `Index`. So sixteen board names out of one door and about forty lines of Python.

What it does not reach is as interesting. `Series.groupby` is on the list of thirteen and is not here, because a frame's `groupby` names a column to group by and a series is grouped by something outside it, usually its own labels or an array the caller brings. That is a real piece of work rather than a line. `Series.filter` is also not here, because it filters the labels where a frame's filters the column names, and the frame method's `axis` has to be understood before that is a line rather than a bug.

Two more become reachable with nothing new. `Series.sort_values` is `sort_index` on the frame this column and its own values make, which is the trick `Index.sort_values` also wants, and `DataFrame.sort_values` does not exist yet so neither does. `Series.to_frame` also makes `Series.value_counts` a `groupby` away rather than a kernel away, once the grouping above is written.

## 8. What is deliberately not here

`Series.groupby` and `Series.filter`, for the reasons in section 7.

`set_index`, `select_dtypes` and `columns`, because they are questions about several columns and a series is one.

Anything that changes in place. `inplace` is passed down to the frame method, which refuses it in the words document 13 argues for, and that refusal is the right one for the series too: every operation here answers a new column and the Arrow buffers underneath are shared rather than owned.
