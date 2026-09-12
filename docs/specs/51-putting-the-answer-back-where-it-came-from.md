# Putting the answer back where it came from

## 1. The argument that was refused for five slices

`inplace=True` has been refused on every callable in this library that declares it, and the sentence it was refused with said that every operation here answers a new frame and the Arrow buffers underneath are shared rather than owned. That sentence has been repeated in six specs and it has capped eighteen otherwise complete names at L2 on the conformance board for five slices running. The comment that closed the last slice said the next one should be a design for owned buffers rather than the nineteenth name added to the list.

There is no design for owned buffers here, because the measurement that should have been made first showed that none is needed. The refusal was answering a question pandas stopped asking in version 3.0, and the library had already been doing the right thing in four places for months without anybody noticing that it was the general answer.

## 2. What pandas actually does now

Under copy on write, which is the only mode pandas 3.0 has, an inplace call does not reach anything except the object it was called on. This is measurable in four lines:

```python
a = pd.DataFrame({"x": [1.0, None, 3.0]})
s = a["x"]
a.fillna(0, inplace=True)
s.tolist()          # [1.0, nan, 3.0]
a["x"].tolist()     # [1.0, 0.0, 3.0]
```

A column taken out of the frame before the call does not see the fill. Neither does a copy, and neither does the frame when the call is made on a column taken from it, and in 3.0.5 that last one does not even warn any more. The only thing that can see an inplace call is a second name bound to the same object, because it is the same object.

So pandas is not writing into shared memory and letting everything that shares it observe the write. It copies before it writes whenever anything else refers to the same block, which is what copy on write means, and then it swaps the result into the object that was asked. What `inplace=True` buys a pandas user in 3.0 is one fewer assignment statement. It does not buy them a saved copy and it has not since the mode became the default.

## 3. What an object here is

A `DataFrame` in this library is a Python object with one slot in it, which holds the extension object that holds the Arrow buffers. `Series` and `Index` are the same. Every method builds a new extension object and wraps it, and `_wrap` is `object.__new__` followed by writing that one slot.

Which means putting an answer in place is writing that one slot. No buffer is touched, nothing that already holds a reference to the old extension object can tell, and the frame the caller is holding now carries the new rows. That is observationally identical to what pandas does, right down to which other objects can see it, which is none of them.

The library has been doing exactly this since the index slice. `Index.rename` and `Index.set_names` honour `inplace` on both `Index` and `DatetimeIndex` by rebinding that slot, and their docstring says so and explains why a level name is safe to change. The explanation was narrower than it needed to be. It is not that a name is safe where values are not. It is that nothing here is ever written, so there is nothing for the flag to be unsafe about.

## 4. The two halves of pandas' return

pandas does not answer the same thing from every inplace call and the split is not documented anywhere. Measured on 3.0.5, these answer `None`:

`drop`, `dropna`, `drop_duplicates`, `sort_values`, `sort_index`, `reset_index`, `set_index`, `rename_axis`, and `DataFrame.rename`, on both a frame and a column where both have them, plus `Index.rename` and `Index.set_names`.

And these answer the object itself:

`fillna`, `ffill`, `bfill`, `where`, `mask`, `clip`, `replace`, on both a frame and a column, plus `Series.rename`.

The second group is a wart rather than a decision, and `Series.rename` sitting in it while `DataFrame.rename` sits in the first one is the clearest sign of that. It is matched anyway, because `df.fillna(0, inplace=True).sum()` works in pandas and somebody has written it. The two halves are two functions here, `_settled` and `_kept`, which differ in one line, and which one a method calls is the whole of the difference. The methods in the first group are typed as answering their own type or `None`, and the methods in the second group are typed as always answering, which is true of them.

## 5. What a flag is allowed to be

pandas validates this argument and will not widen it, which is worth matching because the rule surprises people. A one is refused even though `1 == True` in Python, and so is a zero, and a `numpy.int64` holding a one, all with `For argument "inplace" expected type bool, received type int.` naming the type that arrived. A word is refused the same way. A `numpy.bool_` is accepted, because pandas asks whether something is a boolean rather than whether it is a `bool`. And `None` is accepted and means False, which nobody would guess.

All four rules are in one function here, which is read by every method that takes the flag. The numpy case is recognised by the type's name and module rather than by importing numpy to ask, since nothing else in this library needs numpy to be installed and this is not the place to start.

`Index.rename` and `Index.set_names` do not validate the flag in pandas at all. They read it for truth, so `inplace=1` works there and works here, which is a difference between two methods of the same library rather than between two libraries.

## 6. The one refusal that stays

`Series.reset_index()` that keeps its labels answers a frame, because the labels have become a second column of values. There is nowhere to put a frame inside a column, so pandas refuses with `TypeError("Cannot reset_index inplace on a Series to create a DataFrame")` rather than quietly changing what the name refers to. That refusal is here with the same sentence and the same class, and it is the only place in this slice where `inplace` raises for a reason that is about the operation rather than about the library.

## 7. What this does not buy

It is worth being blunt about this, because the flag reads like a performance argument and is not one, in either library.

An inplace call here allocates the same answer an ordinary call allocates, and then writes a pointer. It does not avoid building the new column, it does not reuse the old buffers, and it is not faster than writing `df = df.fillna(0)` by any amount that can be measured. The saving is one name binding in the caller's source. Under copy on write pandas is in the same position, and the pandas documentation now says as much, which is why the argument is deprecated in spirit if not in the signature.

What it does buy is that code written against pandas runs. That is the whole of the value and it is a large value, because `inplace=True` is in a great deal of existing code and every line of it was a hard stop at this library's front door.

## 8. What is not here yet

Four callables still refuse the flag and all four refuse it because the method itself is not here: `DataFrame.eval`, `DataFrame.query`, `interpolate` on a frame and on a column, and `pandas.eval`. `MultiIndex.rename` and `MultiIndex.set_names` refuse it because there is no `MultiIndex`. None of those is about `inplace` and each will get the flag for free when its method arrives, since the flag is now four lines at the top of a method and one call at the bottom.

The six specs that carry the old refusal sentence are left as they were written, since each of them is the record of what its slice did and was true when it was written. Document 44, on copying, is the exception and has been corrected, because its argument for ignoring `deep` rested on the refusal. The argument survives the correction and comes out stronger: nothing here writes into a buffer, so a deep copy still has nothing to protect anything from, and `inplace` rebinding a slot is not a write.
