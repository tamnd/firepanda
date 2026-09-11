# Renaming, which moves a name and not a row

## What pandas calls renaming

Six callables and a property, and they do two different things that share a word.

`DataFrame.rename`, `Series.rename`, `DataFrame.rename_axis`, `Series.rename_axis`, `Index.rename` and `Index.set_names`, plus the `Index.names` property that `set_names` is the setter for. `Index.rename` is already here and has been since the index was written. The other five are not, and neither is `names`, which is why four of the cases in the compat suite's inplace block report a missing attribute rather than the refusal they were registered for.

The two things are worth separating before anything else, because the whole shape of this document follows from the split.

**Renaming a name moves metadata.** A column called `b` becomes a column called `B`, or the index level called `a` becomes the one called `row`, and not one value in the frame is read, compared or copied. It is a field in a schema and a field on an index, and changing either is a handful of bytes.

**Renaming a label moves data.** `df.rename(index={1: 99})` maps every row label through a dictionary, and `df.rename(index=str.upper)` maps every row label through a function. That is a pass over the index column with a lookup or a call per element, and the result is a new column of labels. It is the same shape of work as `map`, and it is not free on a hundred million rows.

pandas spells both of them `rename` and decides which one it meant from which keyword the caller used. That is a fair design for a library where everything is in memory and mostly small, and it is the reason the parameter list is as long as it is.

## What is here and what is refused

The metadata half is here, all of it. The label half is not, and it refuses rather than being written as a loop over boxed objects.

**`DataFrame.rename(columns=...)`** takes a dictionary or a callable and answers a new frame whose schema says something different. A dictionary names the columns it wants changed and leaves the rest alone. A callable is applied to every column name. Underneath it is the core's `rename`, which copies the frame's column pointers and edits one field of the schema, so the cost is the schema and not the data.

**`DataFrame.rename(mapper, axis=1)`** and **`axis="columns"`** are the same door with the argument in a different place, and they go to the same code.

**`DataFrame.rename(index=...)`**, and the bare `df.rename(mapper)` form that means `axis=0`, raise. The message says that mapping row labels is a pass over the index rather than a change to a schema, that doing it from Python would be a lookup per row through a dictionary the interpreter owns, and that `set_index` on a column computed the way the caller wants is the expression that does the same work with the loop in the right place.

**`Series.rename(name)`** with a scalar changes what the column is called, which is the metadata half, and is here. **`Series.rename(mapping)`** with a dictionary or a callable is the label half and raises for the frame's reasons. pandas decides between the two by asking whether the argument is callable or dict like, and so does this, which means the two doors of one pandas method land on opposite sides of the line this document draws. That is worth saying out loud rather than leaving for somebody to discover from an error message.

**`Series.rename(None)`** clears the name in pandas and it clears it here too, which needed no new machinery, because the core holds a column's name as a `String` and an empty one is already how it spells having none. A column built without a name is in that state from the start. So the difference a caller sees is not in `rename` at all, it is that `Series.name` answers `""` where pandas answers `None`, and that is true whether or not anything was ever renamed. Making it answer `None` means a name that can be absent in the core, which is the same change the row as a series wants, and it is on that piece of work rather than this one.

**`DataFrame.rename_axis(name)`** and **`Series.rename_axis(name)`** change the index level name and nothing else. `rename_axis(None)` clears it, which is a real operation and not the same as not passing an argument, so the parameter's default is the no default sentinel this library already keeps for exactly this problem rather than `None`.

**`rename_axis(..., axis=1)`** raises. In pandas it names the column axis, and a pandas frame's `columns` is an `Index` with a name field to put it in. Here `columns` is a list of strings, because a frame's columns are its schema and a schema is not a column of data. There is nowhere to write the name, and inventing somewhere to write it would mean giving every frame a second index that nothing else in the library reads.

**`Index.set_names(names)`** takes either a name or a one element sequence of names, because pandas takes both and the sequence form is the one that generalises to a multi level index. **`Index.names`** answers the level names as a list, which for a flat index is one long.

## `level`, which has to be None

`DataFrame.rename`, `Series.rename` and `Index.set_names` all take a `level`, and it selects which level of a multi level index the renaming applies to. There is no multi level index here yet, so the only accepted value is `None`.

pandas refuses a level on a flat index too, and it refuses `level=0` as well as `level="a"`, with `ValueError: Level must be None for non-MultiIndex`. That is not obvious, since zero is the only level a flat index has and asking for it is not wrong in any interesting sense, but matching it is free and diverging from it would be a difference nobody asked for. So this refuses the same values with the same class, and the message names the milestone the multi level index is on rather than pretending the parameter is meaningless.

## `errors`, which is the one parameter that changes an answer

`errors="ignore"` is the default and it means a name in the mapping that is not a column is skipped. `errors="raise"` means it is a `KeyError` listing every name that was not found. Both are here, because the default is the surprising one: `df.rename(columns={"colour": "color"})` on a frame whose column is spelled `color` already does nothing at all and says nothing about it, and a caller who wants to know has to ask.

The `KeyError` collects the missing names rather than stopping at the first, and it lists them in the order the caller wrote them, so a mapping with three typos in it reports three typos.

## `inplace`, and why the index is still the exception

`inplace` is refused on `DataFrame.rename`, `Series.rename`, `DataFrame.rename_axis` and `Series.rename_axis`, which is what it is on the other thirty six callables that take it, and for the same reason: it does not save a copy, it hides the assignment, and it makes a chained call silently do nothing.

It is honoured on `Index.rename` and on `Index.set_names`, which is the decision `Index.rename` already made and this follows. An index is immutable in every respect that concerns the rows it holds, and a level name is not one of those. Renaming a level moves no labels, changes no lookup and invalidates nothing, and pandas treats an index as mutable in that one respect for the same reason. There are exactly two of these in the library and they are both on the index.

## `copy`, accepted and not read

`DataFrame.rename`, `Series.rename` and both `rename_axis` take a `copy`, and all four ignore it, which is what document 44 says about every `copy` parameter in this library and for the same reason. Nothing here writes into a frame, so there is no later write for a copy to protect an original from. pandas has been deprecating the parameter across the whole API for two versions and it defaults to the no default sentinel in every one of these signatures, which is what a parameter looks like on its way out.

## What this does not do

**Duplicate column names.** pandas lets a rename produce two columns with the same name and then makes `df["a"]` answer a frame rather than a column. firepanda's schema will not hold two columns of one name and the core says so, so a mapping that collides raises rather than producing a frame that half the rest of the library cannot describe. This is the existing schema decision surfacing through a new method rather than a new decision.

**`Index.set_names` on more than one level.** The sequence form is accepted, and a sequence longer than one raises with the length that was given and the one level the index has.

**`DataFrame.columns` as an index.** Noted above under `rename_axis`. It is the same missing piece that makes `df.columns` a list rather than something with a `name`, a `dtype` and a `rename` of its own, and it is not renaming's to fix.

**The label half, again.** `rename(index=...)` and the mapping form of `Series.rename` want an index `map`, which wants a kernel that can apply a dictionary or a Python callable across a column without the interpreter owning the loop. That is the same kernel `Series.map`, `Index.map` and `factorize` want, and it should be written once for all of them rather than four times badly.
