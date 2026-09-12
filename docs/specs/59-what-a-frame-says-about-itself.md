# 59. What a frame says about itself

## 1. A member that is nothing but formatting

`info` prints what a frame or a column holds: the class, the labels, one line per column with its position and its name and how many rows are not missing and its type, then the types counted up and the memory the whole thing weighs.

Every number in that report already existed as a member before this was written. `shape`, `columns`, `count`, `dtypes`, `index` and `memory_usage` are the report and `info` is the layout, which is why it is written in Python and reaches for no kernel at all. It is the last member of the shape and memory group and it is the only one in the group that is a rendering rather than a fact.

That makes it easy to write and easy to get subtly wrong, because a report that is nearly pandas' is worse than one that is obviously not. A caller who diffs the two outputs and finds one unexplained space has to work out whether anything else is different too.

## 2. The layout is pandas' layout, down to the trailing spaces

Every column of the table is as wide as the widest thing in it, the header included, with two spaces between columns and none after the last one. The rule under the header is as long as the header rather than as long as the column, so a column of `float64` values under the heading `Dtype` gets a five character rule over a seven character column. That looks like an oversight and it is what pandas prints, so it is what this prints.

The position column is the one that is not obvious. Its header is `" # "` with a space on either side of the hash, and its width is that header or the widest position with a space in front of it, whichever is larger. A frame of three columns gives a five character column and a frame of a hundred and twenty gives a six character one, because ` 119` is four characters and the header is three.

All of this was read off a running pandas rather than out of its source, and the tests compare the two reports line for line on a frame of numbers, where the only lines allowed to differ are the two that section 3 names.

## 3. Three lines differ and all three are already registered

The class line says `<class 'firepanda.DataFrame'>`. Nothing else could be true and nobody could mistake it for a defect.

The types are spelled the way this library spells them, so a text column says `string` where pandas 3 says `str` and a column of dates says `date32[day]` where pandas says `object`. That is `engine/dtype-spelling` in the compat registry rather than anything this member decides, and it shows up twice in the report, once per column and once in the counted line at the bottom.

The memory number counts Arrow buffers where pandas counts a numpy representation, which is `engine/nbytes` and is document 58. One consequence is worth naming on its own: there is never a `+` after the number here. pandas puts one there when the number it printed left something out, which is the contents of the object columns it did not follow pointers into, and there is nothing here that gets left out.

A fourth difference shows only on an index with a gap in it, where the first or last label is missing. This prints `None` and pandas prints `nan`, because that is the value each library actually holds in that position, and rendering the other library's value would be a lie about what is in the frame.

## 4. The labels line reads its own class name

`RangeIndex: 3 entries, 0 to 2` when the labels are still a range, and `Index: 3 entries, p to rrr` when they are not. An index with nothing in it stops after the count, because there is no first label to name.

The class name is read off the object rather than written into the string. Today every index a frame hands out is an `Index`, so the line always says `Index`, and the day the wrapping gap in issue 495 is closed and a frame hands back a `DatetimeIndex` this line will say so without being edited. Writing the name down would have meant remembering to come back.

## 5. The two forms and the limit between them

A frame wide enough to make the table useless gets one line instead: `Columns: 120 entries, c000 to c119`. `verbose` chooses between the two forms and `max_cols` moves the width at which the choice is made.

pandas reads that width off `display.max_info_columns`, whose default is 100. There is no options system in this library, so it is a constant with pandas' default in it, and `max_cols` is how a caller moves the line. That is not a smaller interface than pandas' in practice, since a caller who wants a different width has to pass something either way once the option stops being the one they want.

## 6. The counts are always taken

pandas turns the missing count off for a frame above `display.max_info_rows`, which is about 1.69 million rows, because over there counting is a pass over every column and a report should not cost a scan.

Here it is not a pass. Arrow keeps a null count against each chunk, so the number is read rather than computed, and there is nothing to save by not asking for it. `show_counts` left alone therefore counts, at any size, and a caller who passes False gets the narrower table anyway. This is the one rule in the member that is deliberately not pandas' rule, and the reason it is safe is that the difference can only ever show on a frame with more than a million rows, where pandas prints less than the truth rather than something else.

## 7. Nothing is refused

`verbose`, `memory_usage` and `show_counts` are read for their truth and are not checked for being booleans, and `memory_usage="deep"` is the one string that means anything, which is the same number because every number here is already the deep one.

That is pandas' behaviour and it was measured rather than assumed. `df.info(memory_usage="bogus")` prints the memory line over there instead of raising, and `verbose="yes"` and `show_counts=1` are simply truthy. Document 58 section 6 has the same rule for `memory_usage` itself and the same reason: a library that refuses what pandas accepts stops a working program for no gain.

## 8. The column's report is the frame's with two columns taken out

A column has no position and no name inside the table, so the table is the count and the type, and the name moves to a `Series name:` line above it. A column with no name prints `Series name: None`.

`verbose` and `max_cols` are accepted on a column and change nothing, which is pandas' behaviour and is right, because there is one column to list and no width at which listing it is too much. Keeping them in the signature is what lets a caller who wrote one call for either class keep writing it.

## 9. What is left

Nothing in the shape and memory group. `info` was the last of it, and `DataFrame.nbytes` is not a gap because it does not exist in pandas 3 either.

What `info` itself does not do is print a `MultiIndex` differently from a plain one, which pandas does, because there is no MultiIndex here yet. When there is, this member gets a line about levels and this section is where to start.
