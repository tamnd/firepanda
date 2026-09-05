# The column you can hold

`df["a"]` is the most written expression in pandas, and until this work there was no answer to it. The bound `DataFrame` could report its shape, print itself and hand out an Arrow array, and the one thing it could not do was give you a column. A frame whose only output is another frame is a demonstration rather than a library, because there is no pandas program that does not take a column out of a frame at some point in its first ten lines.

This is the second bound type, the expression that reaches it, and the two things adding a second type changed about the generator that writes both of them.

## 1. What a series is on each side

On the Mojo side, `PySeries` in `firepanda/py/series.mojo` is `PyDataFrame` with a different payload. It holds an `ArcPointer[Series]` rather than an `ArcPointer[DataFrame]`, for the reason document 15 gives: a share is what lets an Arrow export hand out pointers into this memory and keep it alive after the Python object is gone. Nothing in this milestone exports a series yet, and holding it as a value now would mean changing the type later rather than changing one call site.

On the Python side, `Series` is generated from the same table as `DataFrame` and delegates the same way. What it exposes is `name`, `dtype`, `size`, `shape`, `count`, `hasnans`, `head`, `tail`, `tolist`, `len`, a repr, and the two Arrow capsule dunders section 5 is about. That is a small surface and it is chosen rather than arbitrary: it is what a person checks when they have just pulled a column out and want to know whether it is the column they meant, plus the way out to every other library.

## 2. `df[key]` is two operations wearing one name

A string key takes a column out and gives back a series. A list of strings takes several and gives back a frame. Those are different operations and the only thing they share is the bracket, which is why this member could not be a row in the binding table.

The table in `tools/bindings.py` describes each member as one expression written against `self._inner`, and that shape is the whole reason the table is reviewable: you can read a row and know what it does. The header of that file already said what to do about the first member that needs real logic, which is to put it in a hand written base class rather than to grow the expression, because a table that carries code stops being a table. `python/firepanda/_pandas.py` is that file and `DataFrameMixin.__getitem__` is its first and so far only member. The generated `DataFrame` inherits from it.

The rule for what goes there is narrow on purpose: a member belongs in the mixin when what it does depends on its argument, and nowhere else. Anything that is one call under a different name belongs in the table, where the parity test can check its signature against pandas.

pandas reads several other kinds of key here, including a boolean mask, a slice, a callable and a tuple for a MultiIndex. Those raise a `TypeError` that says what is read today. Approximating them would be worse than refusing: a key that quietly means something else is a wrong answer rather than an error, and a wrong answer found in a report is much more expensive than one found at the call.

## 3. A missing integer stays missing

`tolist` on an integer column with a hole in it gives back `[10, None, 25]`. pandas gives back `[10.0, nan, 25.0]`, because a numpy int64 array has nowhere to record absence, so the column has to be widened to float64 and the hole filled with a NaN.

This is a real divergence and it is the right one. A firepanda column is Arrow and carries a validity bitmap, so the value is missing rather than approximated, and `None` is what says so. It is also what `pyarrow.Array.to_pylist` and `polars.Series.to_list` both return, so of the two available answers the pandas one is now the unusual one. pandas itself is moving this way with its nullable dtypes, and a library built on Arrow that widened a column to imitate the older behaviour would be adding a lossy step in order to be bug compatible.

The same argument does not extend to arithmetic, where pandas semantics have to be matched exactly because that is where silent differences hide. This is a formatting decision at the edge of the library rather than a semantic one in the middle of it, and it is documented rather than inferred.

## 4. What `tolist` costs, and why it exists anyway

It copies every value, one Python object at a time, which is the slowest thing in the binding layer by a wide margin. It is here because it is currently the only way to read a value from Python, and because a test that cannot read a value can only assert about shapes, which is the kind of test that passes for a column that is right and for one that is merely the right length.

It also costs binary size. Reading a column of unknown dtype means dispatching over the twelve physical dtypes, which compiles twelve copies of the loop, and the extension went from 760,688 bytes to 917,968. That is 157 kilobytes for one method, against a budget of eight megabytes, and it is the clearest single example of the cost model document 03 states: a dispatch list is a multiplier on binary size, so the lists stay narrow and a kernel that does not need a dtype parameter does not take one.

The way out is not to make `tolist` cheaper, it is to make it unnecessary, and section 5 is that. `__arrow_c_array__` on a series hands the whole column over with no copy and no per element Python object, so `tolist` is now the thing you reach for when you want Python values in particular rather than the only way to get anything at all.

Three kinds of column take different paths out. Text is not dispatched over the numeric dtypes, because a string column is physically uint8 and would come back as a list of byte values, which is the kind of bug that looks like a formatting problem. A column of the null type has no buffer at all, so there is nothing to read and every row is missing. Everything else is one typed pass.

## 5. A column goes out as a column

A series answers `__arrow_c_schema__` and `__arrow_c_array__`, the same two names the frame answers, and the difference is what comes out of them. A frame is a struct in Arrow's type system with one child per column, so its export is one struct array and a consumer reaches into it to find a column. A series is not a struct of one thing, it is the thing, so its export is the column's own array and `pa.array(df["qty"])` gives back an `int64` array rather than something to unpack.

Two things fall out of that which are worth saying rather than leaving to be discovered.

**There is nothing to refuse.** The frame's export has to check that no column is stored in more than one chunk, because a struct array has no way to express one, and it refuses before allocating anything so that the failure is clean. A series cannot be in that state: taking a column out of a frame flattens it, so there is exactly one array by construction, and the check would be dead code rather than caution.

**The name travels.** Arrow allows a top level array to have no name and pyarrow exports its own arrays that way, so passing the name is a choice rather than an obligation. It is worth making, because a series always knows its name and `polars.Series` picks it straight up off the field, and a consumer that has to put the name back by hand has been handed less than was available for free.

What this closes is the array direction of the P5 issue, and what it costs is nothing new: the buffers are the column's own memory and the export takes a share of the series, so the array outlives the Python object it came from. That is the ownership argument document 15 section 2 makes for the frame, unchanged, because it was never about frames. It cost 24,240 bytes of binary, against the 157 kilobytes section 4 records for `tolist`, which is the cost model working the way it is supposed to: the fast path is the small one because it does not dispatch over dtypes at all.

`requested_schema` is still refused on both types. Converting on the way out is not written, and a consumer is entitled to assume that what it asked for is what it got, so answering with the wrong type and no complaint would be worse than not answering.

## 6. Two things a second type changed about the generator

**A method can return a type other than its own.** `wraps` used to be a flag, because the only thing a frame method returned was a frame. `DataFrame.__getitem__` returns a series, so it is a class name now.

**An empty constructor reaches the refusal.** The generated `__init__` takes the extension object, which is not a public entry point, so `firepanda.DataFrame()` used to answer with a complaint about a missing argument named `inner` that no user was ever meant to pass. The parameter now defaults, and an empty call reaches the Mojo `py_init`, which refuses with a message saying to use `read_csv`. That was a pre-existing wart on `DataFrame` and it was worth fixing here rather than later, because the second type would otherwise have doubled it.

## 7. `dtype` is a string, and says so

pandas returns a numpy dtype object from `Series.dtype` and this returns a string. The names agree for every type both libraries have, so `str(s.dtype)` reads the same on both sides, and code that compares against `numpy.int64` fails immediately and visibly rather than subtly. Returning something that pretends to be a numpy dtype without being one would be the worse choice, since the failure would then be at a distance from the cause.

## 8. What is left

**`__arrow_c_stream__` on a series.** A series is one array and pyarrow's own `Array` offers only the array half for the same reason, so this is a smaller gap than it looks. It matters for the consumers that read only the stream, which is DuckDB and anything holding a chunked column, and it is the shape document 16 section 9 already built for the frame.

**`to_numpy`.** The array export is what it needs and it is now there, so this is a Python side call through pyarrow rather than new Mojo, but it wants a decision about what happens to nulls, which is section 3 again in a place where the answer may have to be different.

**Positional and label access.** `s[0]`, `s.iloc[0]` and `s.loc[label]` are all unwritten. They want the index work rather than the binding work, so they are not simply more rows in the table.

**Construction from Python.** `pd.Series([1, 2, 3])` still refuses, and it needs the Python to Arrow conversion that also blocks `pd.DataFrame({...})`.

**Everything in document 06 section 3.** The surface here is thirteen members against a pandas `Series` of several hundred. What it buys is not coverage, it is that a column can now be held at all, which is what every one of those several hundred is a method on.
