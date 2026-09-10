# 29. A category that survives being moved

Document 27 gave a caller the eleven names on `Series.cat` and document 28 taught a category column to be compared. Both of them are about operations that know they are looking at a categorical. This document is about the ones that do not, which turned out to be where the bug was.

`s.dropna()` on a category column raised. Not with a message about categories, but from the Arrow writer, several layers away, saying the column was a categorical with no categories behind it. That is the shape of an error that means something went wrong earlier and quietly.

## What a kernel does with a type it is not thinking about

Every kernel that moves rows around is written against the physical layout rather than the logical type. A filter over a date column runs through `Array[DType.int32]`, because that is how a date is stored, and the erased result would say it was an int32. So the kernel puts the input's logical type back on the output, and `retyped` is the one line that does it. It is checked, in the sense that the physical layouts have to match, and for every type in the library that check is the whole of the correctness argument. The bytes moved, the meaning did not.

A dictionary column breaks that argument, because a dictionary is the one type whose meaning is not all in the buffer. The codes are in the buffer. The categories are beside it, in a field of their own, and `retyped` neither knows nor could know to bring them along. So `filter`, `take`, `fill_forward`, `fill_backward`, `concat` and `coalesce` each produced a column whose type said `category` and which had an empty category list, and every one of them had been doing it since the day dictionary columns were added.

That list is six of eight. `slice`, which is what `head` and `tail` are, does not go near any of the code the other six share: a slice copies bytes and never looks at them, so it takes the element width as a runtime number and needs no dispatch at all. It built its result the same way and lost the categories the same way. `shift` is the eighth and the odd one, because it is a run of gap stacked onto a slice, and the gap is built from the type alone, so it is the one operation where the side that has the categories is the second argument rather than the first.

The last two were not found by reading. `dropna` was the case that failed, and fixing the six that the filter shares code with is what a person following the bug does. Finding the other two took walking the whole `Series` surface, calling every member on a category column and asking each answer for its categories, which is a fifteen line program and the only way to know the list is complete rather than long.

What makes this worth a document rather than a line in the changelog is how it failed. A column in that state is not obviously broken. `len` works, the codes are all there and correct, and the type prints as `category`. It fails at whatever eventually asks for the categories, which is the Arrow writer, or `decode_dictionary`, or the accessor, and the message names that place rather than the filter three steps back. Two of those readers do not fail at all: anything reading the buffer as the integers it is stored as gets numbers that look like data.

## Carrying it across, and when that is not enough

For the one input kernels there is nothing to decide. A row moving cannot change what a code means, so the result names the same categories the input did, and `with_categories` is that sentence. It is safe to call on any column, because a source that is not a dictionary has no categories to carry, which keeps the call sites to one line and keeps the rest of the library out of it. A shift counts as one of these even though it stacks two things together, since the gap it stacks on is missing rather than in any category and has nothing to contribute to the list.

One thing it deliberately does not do is drop the categories a filter emptied. A filter that removes every row in a category leaves that category in the list, unused. That is what pandas does and it is what `drop_unused_categories` exists to undo, so a kernel deciding otherwise would be quietly making a choice the caller has a name for.

The kernels with more than one input are a different question, and it is not a question about carrying a field. A code is a position in a list. Two columns whose categories differ give the same code to different values, so stacking their code buffers produces a column where half the rows say something nobody wrote. That is worse than the empty list this document is about, because there is nothing left to notice.

So `concat`, `coalesce` and `pick` require that every side names the same categories in the same order, and refuse otherwise. The refusal names the operation, says why a code is not portable between two lists, and says to use `set_categories` to bring them onto one list first, because that is the fix and there is no reason to make the caller find it.

Unifying the lists instead would be a defensible library to build. It would mean rewriting the codes of whichever side did not already agree, which is a pass over the data for an operation whose entire point is that it is not one, and it would turn a caller's mistake into a slow success rather than a message. pandas refuses the same shapes, so refusing is the smaller surprise as well as the cheaper one, and the door to the other behaviour is open if somebody wants it later.

## Pick had a different bug, which was not a bug

`pick` is what `where` and `mask` are underneath. It never corrupted anything, because it reads its two sides through `as_typed_view`, and that refuses a dictionary outright with a message saying the column stores positions rather than values. So the protective check that the rest of the library relies on was doing its job, and the result was that `where` on a category column did not work at all.

It works now, by reading the codes through `dictionary_codes` the way everything else in the categorical files does, picking between the two code columns, and rebuilding a dictionary around the answer. The categories are the shared list, which the check above has already established there is one of.

The result is int32 coded whatever went in, because `dictionary_codes` widens. That is not a new decision, it is the one document 26 made and document 27 restated: every rewrite in this library produces int32 codes, and a column that arrived over Arrow at int8 comes back wider. It is worth naming here only because `pick` is the first operation to widen a column that nobody asked to rewrite.

## Why the compat suite found it and the test suite did not

The Mojo test files for dictionary columns are thorough about the operations that are about categories. There was no test that a filtered categorical was still a categorical, because filtering is not a categorical operation and the person writing the filter tests was not thinking about dictionaries, which is exactly the gap.

The conformance driver found it in one run, and it found it the moment the driver could ask the question at all. `categorical/dropna` is a case that existed before any of this work: it had been reporting absent, because the Mojo `Series` had no category surface for the driver to reach, so nobody had ever run a filter over a category column and then written it to Arrow. The first day the case could run, it failed.

That is the argument for a conformance suite that goes through the same front door a user does, stated more cheaply than an argument usually is. A unit test asks the question its author thought of. A case list copied from another library's surface asks the questions that library's users ask, and `dropna` on a categorical is an ordinary thing to want.
