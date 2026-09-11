# 44. Copying when nothing can be written

`DataFrame.copy` and `Series.copy` are near the top of the list of pandas members by how often they are written, and neither of them existed here. This document is about what they mean in a library that has no way of writing into a frame, which turns out to be a different thing from what they mean in pandas, and about the one place where that reasoning does not hold.

## 1. What `copy` is for in pandas, and why that reason is gone

pandas' `copy` exists so that a later assignment into one object is not seen by the other. `df2 = df.copy()` and then `df2["a"] = 0`, and `df` still has its old column. Without the copy, whether `df` changes depends on how the slice was taken and on what the block manager did with it, which is the whole of the `SettingWithCopyWarning` and is why the defensive `.copy()` is so common in pandas code that it is nearly punctuation.

None of that applies here. There is no `__setitem__` on a frame or a series, there is no `assign`, and `inplace` is refused on all 42 pandas callables that take it, which is a registered engine level divergence rather than an oversight. So a frame in this library cannot be written into at all, no object can ever observe a write to another object, and there is no later assignment for a deep copy to protect the original from.

That means `deep` is unobservable. Both values of it are satisfied by the same answer, and not by accident: `deep=False` asks to share the data, which is what happens, and `deep=True` asks for isolation from later writes, which is already guaranteed by there being no writes. So it is accepted and never read, which `Index.copy` was already doing for its own narrower version of the same argument.

## 2. So the implementation is a wrapper and that is the whole of it

`DataFrame.copy` allocates a new Python wrapper around the extension object the original holds and hands it back. The Arrow buffers are not touched, the schema is not cloned, and the index is not rebuilt. `Series.copy` is the same one line.

It is worth saying what this costs against what pandas charges for the same call, because it is the largest ratio in the library and it is not the result of anything clever. `df.copy()` in pandas duplicates every column, which is the whole frame in memory a second time and a full pass over it, and a function that takes a frame and copies it defensively at the top pays that on every call. Here it is an allocation of one object with one field. A pandas user who brings their defensive copies across with them unchanged will find they have stopped costing anything, which is the pleasant shape of this kind of difference: the code does not have to change for the cost to go away.

## 3. The one place the reasoning does not hold, which is the index

`Index.copy` existed already and it does build a new index underneath rather than sharing the old one. That is not inconsistency, it is the one question in the library that can tell a copy from the original.

`Index.is_` asks whether two indexes are the same object underneath, by comparing the address of the shared index rather than anything visible on the Python side. pandas has it, pandas' own `idx.copy().is_(idx)` is false, and so a copy here has to be a different object underneath for that to stay true. A frame and a series have no such method, in pandas or here, so nothing can ask them the same question and sharing the inner object is unobservable for them in a way it is not for an index.

The cost of the index doing real work is the index, which is one column of labels rather than the whole frame, so the asymmetry is cheap as well as correct.

## 4. `type(self)` rather than `Index`

`Index.copy` was building its answer with `Index._wrap`, which meant a copy of an index of instants came back as a plain index and the calendar members stopped resolving. That is the wrapping bug #495 records, the same one document 42 introduced `_like` for, and it is fixed here by taking the class from `type(self)`.

It is not `_like`, because `_like` turns a column into an index and this already has an index. Two lines that do the same thing in two situations is the right amount of duplication when the alternative is a helper with a branch in it, and the rule they both hold is written down in one place, which is here.

## 5. The copy protocol

`copy.copy(df)` and `copy.deepcopy(df)` reach `__copy__` and `__deepcopy__`, which pandas defines on all three types and answers from `copy`. They are defined here for the same reason: without them, `copy.copy` on a class with empty `__slots__` goes through the pickling protocol and produces something that is not a frame, which is a worse answer than an `AttributeError` because it does not say anything.

`memo` is the dictionary `copy.deepcopy` threads through an object graph so that a node reached twice is copied once. Nothing is recursed into here, so there is nothing to put in it, and it is accepted and ignored the way pandas accepts and ignores it.

They are not board names, because the conformance board counts pandas' public callables and dunders are not in that list. They are here because they are correct, not because they score.

## 6. What this unblocks on the conformance side

`DataFrame.copy` was attaining nothing at all despite `basics/copy` passing at L2, because the L0 case asks the module whether the name resolves and the answer was no. The L2 case passes through the driver, which calls the core directly and never goes near the Python surface, so the board was in the odd position of holding evidence that the behaviour is right for a member that could not be called. That is the same lag document 43 found in the other direction and it is worth noticing that it can happen this way round too.

The larger thing it unblocks is the `inplace` divergence family. Thirty four cases in `divergences/inplace/` each take a copy of a frame or a column, run a mutating call on it, and hand back the object that was mutated, and every one of them was reported absent because the copy at the top could not be taken. A divergence case that does not run is a divergence that is asserted nowhere, which is exactly the failure mode the divergence registry exists to prevent, so those cases were claiming to hold the engine to its word while holding it to nothing.

They do not lift any level. A registered divergence stops a name's climb the same way a gap does, which is right, because a deliberate difference from pandas is still a difference. What changes is that the registry's claim is now tested: if `inplace` ever quietly starts working, those cases start agreeing with pandas and the build goes red.

## 7. What is deliberately not here

`DataFrame.equals` and `Series.equals` are the obvious neighbours and they are not in this change. `Index.equals` exists, the frame and series versions compare shape, dtypes and values with the rule that two missing values are equal, and doing that from Python over `tolist()` would be a loop over boxed objects in a library whose whole argument is about not doing that. It belongs with a kernel behind it.

`attrs`, `flags` and `set_flags`, which are pandas' places to hang metadata on a frame and have it survive operations. The propagation rules are the hard part rather than the storage, and they are their own piece of work.

`Series.view` and the rest of the reinterpretation family, which ask for the same buffer read as another type and are a numpy idea rather than an Arrow one.
