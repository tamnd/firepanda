# The index you can hold

`df.index` is the second thing a person asks a frame for, after its columns, and until this work there was nothing to hand back from it. Documents 19 and 20 built the whole of the flat index in Mojo, thirty odd methods covering lookups, set operations, editing and slice bounds, and both of them end on the same sentence: none of it is reachable from Python. This is the work that makes it reachable.

It is the third bound type, after `DataFrame` and `Series`, and it is the first one that changes something about how the other two are built rather than only adding to them.

## 1. Why it is a type and not a list of labels

The cheap version of `df.index` returns a Python list. It is one line, it is what a user sees when they print the answer, and it is the wrong answer for a reason that only shows up later: a list cannot be asked whether it is unique, cannot be unioned with another, cannot be handed to `reindex`, and cannot carry a level name. Everything documents 19 and 20 built would stay unreachable, and the day the real type arrived every caller written against the list would break.

So `df.index` returns an `Index` and `tolist` is the one member on it that gives the list. That is also what pandas does, which matters here more than usual: `df.index.is_unique` and `df.index.get_loc("x")` are ordinary lines in ordinary pandas code, and a firepanda that answers a list to `df.index` fails them all at once.

## 2. The shape of the binding

`firepanda/py/index.mojo` holds `PyIndex`, which is `PySeries` with a different payload. It carries an `ArcPointer[Index]` for the reason document 15 gives, which is that an Arrow export hands out pointers into this memory and takes its own share, so a consumer may outlive the Python object and still be reading live memory. A range index shares two integers and no memory at all, and that survives being wrapped: `df.index` on a frame nobody reindexed allocates nothing except the Python object.

The methods are flat and plainly named, as in the other two files, and the pandas API is on the Python side. `PyIndex` has no dunders, no properties and no argument that means three different things.

Two new problems came with the type and both are about arguments.

The first is that its methods take other bound objects. `union`, `intersection`, `difference`, `symmetric_difference`, `append`, `equals` and `identical` all take another index, and neither of the first two types ever did. That costs one helper, `_other`, which downcasts and complains by name. Note that it raises where the frame's equivalent aborts: `PyDataFrame._frame` can only fail through a bug in the binding, but a user reaches `_other` by writing `index.union(7)`, so a wrong type there is a `TypeError` and not a crash.

The second is that several methods take a label, and a label is a one row Arrow column. `_one_label` builds one through `array_from` in `build.mojo`, the same reader the frame constructor uses, so an index infers its dtype by exactly the rules document 18 wrote and there is no second inference to keep in step. The one thing it has to do differently is wrap the value in a Python list first, because a bare string is a scalar here and `array_from` would otherwise read it as a column of characters.

## 3. Where the shared helpers went

Three copies of the same helper is where a private convention drifts, and this was the third type. Two things moved out.

`_int`, which reads a Python integer and complains about anything else, existed in `frame.mojo` and again in `series.mojo` with a comment saying it was written twice rather than shared. It is now `firepanda/py/args.mojo`, alongside `flag` and `words`, which do the same job for a bool and a string. `flag` refuses a truthy non bool on purpose, because `sort=None` means a third thing in pandas and a helper that accepted it would quietly turn it into `False`.

`_numbers`, and the loop in `Series.to_list` around it, is now `firepanda/py/values.mojo`, the pair of `build.mojo`. `python_list` reads a column back out into Python objects and `python_value` reads one row, implemented as `python_list(column.slice(i, i + 1))[0]`, so the null rule and the string rule are written once and not twice.

## 4. What the table generates and what it does not

The rule from document 17 has not moved: a member belongs in the table when it is one expression written against `self._inner`, and in the hand written mixin when what it does depends on its argument.

Twenty six members are in the table. They are the properties, `dtype` and `inferred_type` and `size` and `shape` and `nbytes` and the rest, the two list conversions, `unique`, `rename`, `take`, `insert`, `get_slice_bound`, `slice_locs`, and the two Arrow dunders.

Eighteen are in `IndexMixin`. The clearest case is `__getitem__`, which is three operations wearing one name: an integer takes a label out and gives a Python value, a slice gives an index, and a list gives an index and is read as positions or as a mask depending on what is in it. A slice is resolved with `slice.indices` on the Python side rather than in Mojo, because that function is the definition of what a Python slice means and a second implementation that agreed with it would be work with nothing to gain.

`get_loc` is the other one worth naming. pandas returns three different types from it and which one it returns depends on the labels rather than on the argument: one hit is an integer, several hits in a row on a sorted index are a slice, and anything else is a boolean mask. The Mojo side returns positions, which is what all three are made of, and the Python side picks among them, so the rule is written once. Note that a run of duplicates is a slice only when the index is monotonic. `Index(["c", "b", "b", "a"]).get_loc("b")` is a mask in pandas despite its two hits being adjacent, and it is a mask here.

The four set operations are in the mixin for a duller reason, which is that `sort` has three values in pandas and a `Bool` in the core. `None` means sort for `union`, `difference` and `symmetric_difference` and means do not sort for `intersection`, and the mixin is where that is turned into a bool so the core takes a bool and means it.

`__eq__` is in the mixin and defining it there has one consequence worth stating: a class that defines `__eq__` and not `__hash__` is unhashable, so an index cannot be a dictionary key. pandas has exactly that property for exactly that reason.

## 5. A missing label needs a type to be missing in

`array_from` infers a dtype from the values, and a list of nothing but `None` has no dtype in it. `empty_column` answers float64 there, which document 18 argues for and which is right when a column is being built out of its own values.

It is wrong when the value is a label going into an index that already has a type. `Index([3, 1]).insert(1, None)` would build a float64 null, hand it to a concatenation that refuses two columns of different types, and tell the user about a float64 they never mentioned.

So `_one_label` takes the index's own logical type and, when the label is missing, builds the null in that type instead of inferring one. Every call site passes it, through a small `_type` helper, and the effect is that `insert` of a null keeps the index's dtype and `None in index` finds a real absence rather than failing a type check on the way in.

## 6. The errors

One new kind crosses the boundary. `firepanda:position:` becomes `firepanda.errors.OutOfBoundsError`, which is an `IndexError`, and it exists because `except IndexError` around `index[i]` is an idiom that predates this project and has to keep working. It is `value` narrowed to one shape rather than a new idea, and it is separate for the same reason Python separates them.

The table in `python/firepanda/errors.py` gains one row and `python/tests/test_errors.py` holds both halves to it, by asking the Mojo side to raise one of each kind and checking what comes out.

## 7. Divergences recorded rather than fixed

**A frame and its columns do not share one index object.** pandas answers `True` to `df.index.is_(df["a"].index)`, because a frame and its columns hold one index between them. Here each access wraps a copy, so the answer is `False` while `equals` is `True`. Making it true means the frame holding a shared index rather than an owned one, which is a change to the core and not to the binding.

**`values` and `__eq__` return Python lists where pandas returns numpy arrays.** This is the same divergence document 17 recorded for `Series`, and it is the whole of what a numpy dependency would buy at this point.

**`nbytes` is zero for a range where pandas says 132.** The pandas number is the size of the Python object and is the same for three rows and for three hundred million. A range stores no labels, so this reports no bytes, and the honest answer was preferred to the compatible one.

**A missing label keeps its dtype.** `Index([3, 1]).insert(1, None)` is an int64 index with a hole in it where pandas gives float64 with a NaN. Document 20 recorded this at the core and section 5 above is what makes it hold from Python. Its consequence is that `None in Index([3, None, 1])` is `True` here and `False` in pandas, since the label pandas holds is NaN and NaN is not `None`.

**The multi line repr is not reproduced.** A long index prints on one line with an ellipsis in the middle where pandas wraps to the terminal width. The labels shown are the same ones.

**`Index.__arrow_c_array__` is an addition rather than a compatibility.** pandas has no Arrow export on an index at all, so this is the one place the surface is wider than the one it copies. A range has to be materialised to be exported, which is the only allocation on that path.

**Cross dtype promotion is still missing.** Everything document 19 and document 20 said about it is still true and is now visible from Python. It is one missing piece in nine places, all of which raise in the same words.

## 8. What is deliberately not here

`MultiIndex` and the level accessors are issue #155. `nlevels` and `ndim` are constants on this type for that reason, and the constructor's `tupleize_cols` is refused rather than ignored, because what it turns on is the type that does not exist.

`df.index = ...` does not assign. An index comes out of a frame and cannot be put back into one yet, which is the same reason the `DataFrame` and `Series` constructors still refuse `index=`. Their refusal message changed with this work, from saying there is no `Index` type to saying that putting labels on a frame as it is built is not written, because the first of those stopped being true.

`asof_locs` and the filling arguments to `get_indexer` are unchanged from document 19. `method=`, `limit=` and `tolerance=` are declared on the Python side and refused by name, so a caller is told what is missing rather than that the keyword was unexpected.
