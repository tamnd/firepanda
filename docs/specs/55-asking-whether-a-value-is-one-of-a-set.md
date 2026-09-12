# Asking whether a value is one of a set

## 1. Nothing in the core was missing

`isin` is the rarest kind of slice, which is one where the hard part was finished before it started. `firepanda/kernel/member.mojo` has held `is_in`, `text_is_in` and `is_in_any` since the indexing work, with a hash table for a set big enough to be worth building one and a scan for a set that is not, a threshold between them, and a test file that runs every case on both sides of it against a scalar twin. `firepanda/frame/series.mojo` has held `Series.is_in` over the top of that kernel for just as long. The core could answer this question the whole time and no Python program could ask it.

What was missing was one binding and a pile of rules about what a set is. The binding took an afternoon. The rules are the rest of this document, and they are most of the method, which is the shape the last few compatibility slices have all had.

## 2. The set crosses as a column

`PySeries.is_in` takes a `Series` and not a list, for the reason `pick` gives in the same file: the kernel wants something typed, and the layer above is the side that knows what type to make it. There is a second reason here that `pick` does not have. A caller who wrote `s.isin(other)` where the other is already a column has a column, and taking it apart into Python objects so that this side could put it back together would be the slowest step in a call whose whole job is to be fast on a lot of rows.

The answer comes back through `masked`, which was `_masked` in `firepanda/py/transform.mojo` until this slice. It puts a bare `Array[DType.bool]` back on the source column's name and labels, and it existed because `isna` and `notna` need exactly that and `is_null` deliberately does not do it. `isin` is the third caller and the first one outside that file, so the helper lost its underscore. A name with an underscore on it is a name that says not to, and this one now means the opposite.

`isin` lives in `series.mojo` rather than in `transform.mojo` because everything in `transform.mojo` reads one column and this reads two.

## 3. The kernel refuses what pandas answers

`is_in_any` compares one type against one type and raises on a mismatch, which is right for a kernel and is not what `Series.isin` can do. pandas compares by value and never refuses. `pd.Series([1, 2]).isin(["1"])` finds nothing and says nothing about the string. `pd.Series([1, 2]).isin([2, "b", 9.5, None])` finds the two and ignores the other three, and that mixed shape is what real code passes, because a set written by hand is written out of whatever the program had lying about.

The gap is closed by `_isin_wanted`, which drops the values the column cannot hold before the set is built. That is not an approximation and it is not a shortcut. A value a column cannot hold is a value none of its rows can equal, so dropping it and answering false is the same answer as comparing it and finding nothing, arrived at without the comparison. What decides is `_holds`, the same predicate `fillna` uses to decide whether a fill value belongs in a column, which is worth noticing: the question `fillna` asks is whether a value can go into a column and the question `isin` asks is whether a value could already be in one, and those turn out to be the same question.

Document 35 section 6 describes the other way of closing this gap. `Index.isin` tries the kernel, catches the refusal and falls back to comparing in Python, which reaches the right answer for both of the examples that document gives. Dropping reaches the same two answers without the fallback: `[1.0]` against an int64 column is a whole number, so it is kept and cast and finds the one, and `["a", 2]` keeps the two and drops the word. The difference is what it costs when it is reached. An index answers a Python list anyway, so a Python loop over its labels is the same order of work as the answer it is building, while a column answers a column and may be ten million rows, where a Python loop would be the slowest thing in the library. `Index.isin` could be rewritten on the partition and lose its slow path, and section 9 names that.

## 4. Two places where the kinds meet

Dropping everything the column cannot hold would be wrong twice, and both times it is because Python thinks a flag is a number.

`pd.Series([True, False]).isin([1])` finds the True row, because `True == 1`. So a number that is zero or one becomes a flag when the column is flags. `pd.Series([1, 2]).isin([True])` finds the one for the same reason read the other way, so a flag becomes a number when the column is numbers. Nothing else crosses: `pd.Series(["a", "b"]).isin([True])` finds nothing, which is `_holds` refusing a flag for a column of words and is pandas' answer as well.

Those two conversions are three lines and they are the only place in this method where a value is changed rather than kept or dropped. Everything else about the kind boundary falls out of `_holds`, including the rule that a whole float matches an integer column and a fractional one does not, which `_holds` already had because `fillna` already needed it.

## 5. Four missing values and one null

pandas has a different missing value per dtype and matches each column against its own. Measured against pandas 3.0.5: an `object` or `string` column finds `nan`, `None` and `pd.NA`; a `float64` column finds only `nan`; a nullable `Int64` column finds only `pd.NA`; a `datetime64` column finds only `NaT`; a categorical finds `nan` and `None`; and a nullable `boolean` column finds neither `None` nor a number.

firepanda has one missing value underneath all of those. A null in an int64 column and a null in a column of words are the same null, so the distinction pandas draws cannot be read off the column. `_isin_nulls` reads it off the set instead: it asks whether the set holds `None` and whether it holds a nan, and then applies the column's own rule to those two answers. A column of words is true if either is there, a column of numbers is true only for the nan, and everything else is false. That lands on pandas' answer for every case pandas has an answer for, and it does it without needing four missing values, which is the sort of trade this project keeps finding.

The kernel's own answer for a null row is null, not false, and that is deliberate: `Series.is_in` is describing SQL's `IN`, where a comparison against an unknown value is unknown, and the docstring on it says so. pandas answers false. The boundary is where the two meet, so the boundary is where the fill happens, and by the time a `Series` comes back there are no nulls left in the mask at all.

## 6. A nan is not equal to itself

There is one row a fill cannot reach. A float column can hold a real nan as well as a row with nothing in it, and pandas finds both of them with the same nan in the set. The kernel finds neither, but for two different reasons: the missing row compares unknown, and the nan compares false, because IEEE says a nan is not equal to anything including another nan.

So the nan row is not a null and filling the nulls leaves it alone. What catches both is `notna`, which is the one question that is false in both places: firepanda's `isna` counts a nan float as missing even though the validity bit says it is present, which is a decision made long before this slice and is exactly what is wanted here. The answer takes its own value where `notna` holds and true where it does not, through `pick`, and the two rows come out together.

This is the only part of the method that would be easy to get wrong and hard to notice, since it needs a column holding both a nan and a null and a set holding a nan, which is three unusual things at once.

## 7. A category is looked up through its codes

The kernel refuses a category column outright, and the message it refuses with is the reason: a category column stores positions into its categories rather than values, so a lookup against it would be comparing positions against values. Building a second category column carrying the same categories and comparing the two would work, and it is more work than the question needs.

`_isin_codes` turns the set into a set of positions instead. The categories are a short list by construction, so scanning them for the words the set names costs nothing next to scanning the column, and the codes are then looked up in those positions through the ordinary integer path. A value that is not one of the categories has no position, so it is simply not in the list and the rows answer false, which is pandas' answer and arrives here without a branch for it.

The answer is a plain boolean column and not a category of two, which is the obvious thing to get wrong and is tested.

## 8. A frame is put back together through the constructor

`DataFrame.isin` asks each column its own question, which is what pandas does for a mapping and is also what it does for a list, since one set asked of every column is the same loop with the same set. A column a mapping does not name answers false all the way down rather than being left out, and that is pandas' answer too.

Putting the columns back together is the part worth writing down, because it is slow and the slowness is structural rather than careless. There is no `__setitem__` on a frame, there is no `assign`, and there is no concatenation, which document 44 section 1 argues for and which this slice is not going to relitigate. The constructor is the only way to make a frame in this library and it reads Python values, so each column's mask crosses back into Python as a list of bools and then crosses into the extension again as a column. For a frame of ten columns that is ten round trips that a kernel would not make.

A frame the constructor makes carries a range starting at zero, which is already the right index most of the time, so `_isin_framed` checks for that and hands the frame straight back when it holds. When it does not, the labels go in as one more column and come back out through `set_index`, which is the same route `_labelled` takes for a single column and is there for the same reason: an index is built by the constructor or it is not built at all. The name of that extra column is grown with underscores until it does not collide with a real one.

## 9. What is refused, and what it would cost

**A frame or a series as the argument to `DataFrame.isin`.** pandas answers both and does not answer them as membership tests. It lines the argument up against the frame by label and compares cell against cell, so `df.isin(other_frame)` is an equality test wearing this method's name. Reading it as a set would produce a wrong answer that looks like a right one, which is worse than a refusal, so it raises. Lifting it is an aligned comparison and not a set operation at all, and it belongs with `align` rather than here.

**A timestamp against a temporal column.** Everything else the column cannot hold is dropped, on the argument in section 3. A timestamp is the one value where that argument fails, because pandas does find those rows and answering false everywhere would be silently wrong. It cannot be answered here yet for a concrete reason: the set crosses as a column and a column of timestamps cannot be built out of Python objects, which `Series([datetime(2020, 1, 1)])` says when it is asked. Document 18 section 2 is where that ends, with five outcomes from inference and no temporal one among them, so lifting this needs a sixth and is its own slice. Until then it raises and says why. A temporal column asked about anything else still answers, and answers false, which is pandas' answer.

**`Index.isin` still has the Python fallback** described in document 35 section 6. It gives the right answers, so this slice did not touch it. Section 3 argues that the partition would replace it and be faster, and that is a small self contained change for whoever wants it.

**A kernel for the frame.** Section 8 describes ten round trips through Python for a frame of ten columns. A `PyDataFrame.is_in` taking the names and the per column sets would remove all of them and would keep the index without the `set_index` detour. Nothing has asked for it yet, and a frame wide `isin` over a lot of rows is the call that would.

**`min_count`, `skipna` and the rest** are not arguments this method has. `Series.isin` and `DataFrame.isin` take one parameter each in pandas, and both of them take exactly that, so there is no signature gap here to record.
