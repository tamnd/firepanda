# 36. A position and a label are different questions

Status: implemented on the frame and on the series, apart from one shape of answer. Four accessors on each, one method, square brackets on a series, `get` and `squeeze` on both, and eight new ways across the boundary.

## 1. Why `loc` and `iloc` are one piece of work

They are the same code with two readers in front of it. Both take a key that names rows and optionally a second key that names columns, both decide the shape of the answer from the shape of the key rather than from the data, and both end up asking the core for a range of rows, a gather of rows, or the rows a mask is true at. The only thing they do differently is turn a key into positions, and that is about forty lines on each side.

Writing one of them first and the other afterwards would have meant writing the shared half twice, or writing it once and then discovering on the second pass which parts of it had quietly assumed a position. So they went in together, and the shared half is a base class with two abstract readers on it, which is the shape the code wanted rather than a pattern applied to it.

The difference between them is worth stating once, because everything downstream follows from it. `iloc` counts and `loc` names. A slice of positions excludes the position it stops at, because that is what a Python slice does and `iloc` is numpy's model. A slice of labels includes the label it stops at, because a caller who writes `df.loc["b":"d"]` means through d rather than up to it, and there is no position arithmetic in their head to make the half open reading natural. That one difference is the most common off by one in code written by somebody who came to pandas from numpy, and it is the reason the conformance suite has a case called `loc-slice-closed` that asserts five rows where the same slice through `iloc` gives four.

## 2. The shape of the key decides the shape of the answer

pandas' rule is that an axis named by one thing collapses and an axis named by a set of things does not. Two collapsed axes are a value, one collapsed axis is a column, and no collapsed axis is a frame. That is the whole rule, it is about the key and not about the data, and `df.iloc[0:1, 0:1]` is a one by one frame while `df.iloc[0, 0]` is a number.

This is the reason `__getitem__` on the accessor cannot be a row in the generated table. The table holds members that are one expression written against the extension object, and this one is a decision about what to build before there is anything to build it out of. Document 17 drew that line for `Series` and it falls in the same place here.

## 3. Where a key is read and where it is applied

A key is read into one of five descriptions: every row, a half open range, a gather of positions, a mask that is already a column, or a single row. Reading is the half the two accessors do differently and applying is the half they share, and putting the boundary there means the shared half never has to ask which accessor it was reached through.

The five are not an abstraction over the core, they are the core's four row operations plus the one that has no operation because it collapses an axis. `slice` shares buffers and copies nothing that is not asked for, `take` gathers, `filter` reads a boolean column, and a frame with no row key at all is handed straight back. A description that did not correspond to something the core already does would be a description this layer had to implement, and there are none.

## 4. Which axis is narrowed first

The columns, always. A key that names both axes could be applied in either order and the answer is the same, and the work is not: narrowing the columns first means the row gather copies only the columns that were asked for. On the conformance suite's widest frame that is the difference between copying two columns and copying sixteen, and on a frame with five hundred columns and a two column selection it is two orders of magnitude.

The exception is the single value, which skips both. `df.iloc[0, 0]` does not narrow anything, because there is a binding that reads one cell by a pair of coordinates and neither narrowing would be on the way to it.

## 5. Reading one cell, and why it is a binding rather than two calls

`at` and `iat` could have been `column(name)` followed by a read at a position, and that is what the first sketch did. It is wrong by a factor of the height. `column` copies, because a `Series` here is one contiguous array and a column in pieces has to be stacked on the way out, so asking for one cell of a million row frame through a column copies a million values to answer with one of them.

So `cell(row, at)` is its own crossing. It finds the chunk holding the row by binary search over the chunk starts, which the chunked array already does for its own reasons, and reads the value out of that chunk. Nothing is copied, nothing is stacked, and the cost is the logarithm of the number of chunks.

That is also what makes `at` worth having at all. `df.loc[label, name]` reaches the same value through the same lookup, and the only reason to write `df.at[label, name]` instead is that it promises never to answer anything except a value, so it can skip every branch that decides what shape the answer has. A version of `at` that went through `loc` would be an alias, and an alias for a fast path is not a fast path.

## 6. The row as a series, which is not here

`df.iloc[0]` and `df.loc[label]` on a unique index both collapse the row axis and leave the column axis alone, so both answer a series whose labels are the column names. That is refused rather than approximated, and the refusal is the one piece of this that is a decision rather than a gap.

A frame is stored as columns and each column has its own type. A row read across them is a transpose of one row, and a series has one type, so the answer has to be a type every column fits. pandas computes that type with a rule that is not obvious from the outside: a frame of int64 and float64 gives a float64 row, a frame with any string column gives an object row, and a frame of one type gives that type. Nothing in this library computes a common type over a set of columns, because nothing until now needed one, and inventing it inside an indexing accessor would put a type rule in the last place anybody would look for it.

The cost of leaving it out is two of the fifteen indexing cases in the conformance suite and a message that says what is missing. The cost of guessing at it is a row whose dtype is right for the frames in the test suite and wrong for the first frame a caller brings.

There is a second thing in the way that was found by measuring rather than by reading, and it matters because it does not go away when the type rule is written. The row as a series is named after the row label. pandas carries that label as whatever it is, usually an integer, and a series here is named by a string, so a row built correctly and typed correctly would still come back under a name that is the label spelled out rather than the label. That is why building the type rule is worth doing for its own sake and is not by itself enough to pass the two cases.

## 7. A negative position, which the core reads as something else

`DataFrame.take` in the core answers a null row for a negative index, and that is not an accident to be worked around. It is what an outer join needs: a join that found no match on the right has to produce a row of nulls for it, and encoding that as a negative index into the right hand frame means the gather and the null filling are one pass instead of two.

pandas' `take` counts a negative from the end. Both are correct readings of the same argument for different callers, so the counting back happens in the binding, which is also where the bounds check goes. Every position is resolved and checked against the height before the core sees it, and what that check is really doing is making sure no negative index survives to be read as a null row. A gather that a caller wrote as a selection cannot produce a row that was not in the frame.

## 8. The messages, and which side raises them

Document 22's rule is that a message leads with pandas' sentence. Five of the messages here are pandas' word for word: `positional indexers are out-of-bounds`, `Too many indexers`, `Boolean index has wrong length: N instead of M`, `[label] not in index`, and a missing single label, which in pandas is the label on its own and nothing else.

The last two are raised as a plain `KeyError` rather than as one of the named classes in `firepanda.errors`, and that is the one deliberate exception in this file. The named classes exist because a Mojo error carries a kind tag across the boundary and the tag has to become a class on this side. A missing row label never crosses the boundary at all: it is found here, by asking the index where a label is and being told there is no such label, so there is no tag to translate and nothing for a named class to carry. pandas raises a plain `KeyError` too.

The missing single label is worth one more sentence, because the message it is raised with is thrown away and rewritten. The index does answer with a sentence, and the sentence names the index rather than the label, which is the right way round for somebody who called `get_loc` and already has the label in their hand. It is the wrong way round for somebody who wrote `df.loc[99999]` and is now reading a traceback, because the thing they are going to search their own source for is the number. pandas raises `KeyError(99999)` and so does this.

There is one message in the list that cannot be raised with the class pandas raises it with. `Too many indexers` is a `pandas.errors.IndexingError`, which is a class pandas defines rather than a builtin, and firepanda does not import pandas, so a firepanda exception cannot be a subclass of it and an `except pandas.errors.IndexingError` clause will not catch one. That is a consequence of the library not depending on the library it replaces, it applies to every pandas defined exception class and not only to this one, and it is registered as a divergence rather than left to be rediscovered.

## 9. What is refused, and what each would cost

| Argument or shape | What it would take |
| --- | --- |
| `df.iloc[0]` and `df.loc[label]` | A common type over a set of columns, which is section 6 |
| `df.at[label, name]` on a repeated label | The same, since pandas answers an array there |
| `df.loc[label] = value` | Assignment through an accessor, which means an owned buffer |
| `df.iloc[cond]` with a callable | Running a Python function over the frame, which is the `key` machinery nothing here has |
| A key with three or more axes | A MultiIndex, since that is the only thing a third axis can mean |
| `take(**kwargs)` | Nothing. pandas accepts them only to ignore them, and this refuses them |
| `except pandas.errors.IndexingError` | Importing pandas, which document 14 section 8 rules out |

Only one of the seven is a missing kernel and it is the same one twice. Assignment is a decision about the whole library rather than about these accessors, and it is the same decision `inplace=True` is waiting on everywhere else.

## 10. What it is worth

Thirteen of the conformance suite's fifty six indexing cases arm on this, and they are the largest block in the section: eight `iloc` cases, six `loc` cases and two each of `at`, `iat` and `take`, less the two that are the row as a series. That is more board runs than any single piece of work in the indexing section is going to return again, because `loc` and `iloc` are where the section's weight is.

What it buys beyond the runs is that the frame is now addressable. Until this went in there was no way to ask a frame for a row, and a library that can only hand back a whole frame or a whole column is one a caller cannot walk through. Everything left in the section, which is `reindex`, the `duplicated` family, `nlargest`, `query`, `filter` and `truncate`, is a rule for computing a set of rows, and every one of them ends in the gather that is now written.

## 11. The same two questions about a series

A series has one axis, so everything in section 2 about the shape of the key deciding the shape of the answer collapses to one decision: a key that names one row is a value and a key that names a set of them is a series, a set of one included. That is a smaller problem than the frame's and it is the same problem, which is why `s.loc` and `s.iloc` are one class with a flag rather than two classes, and why the two key readers moved out of the frame's accessors and became functions the four accessors share.

Moving them was the whole of the work. The five descriptions in section 3 are the core's row operations and `Series` has all four of them already, so `slice_rows`, `take`, `filter_rows` and `cell` on the binding are the same four crossings the frame has, written against a column instead of against a table. Nothing new was computed. What was missing was a door.

The part that is not shared is `s[key]`, and it is worth writing down because it is indefensible and it is not ours. Square brackets on a series read one key as a label and a slice of numbers as positions, on the same series, in the same expression. `s[2]` is the label two even on an index whose labels are strings, where it raises. `s[2:5]` is the rows two to five counting from the front even on an index whose labels are those same numbers in a different order, where it answers different rows than `s.loc[2:5]` does. Nobody would design that. Code that depends on it is everywhere, and a library that reads the slice by label is not the library people have, so this reads it the way pandas reads it.

What decides is the slice's own bounds rather than the index's type, which is the detail that makes the rule implementable rather than a special case per index. A bound that is a whole number means positions, anything else means labels, and a bound that is not written says nothing either way, so `s["a":"c"]` on a string index stays a closed slice of labels while `s[0:2]` on the same index is the first two rows. Everything that is not a slice goes to `loc` unchanged, which is exactly what pandas does with it.

The two messages about a position past the end are different sentences in pandas and they stay different here. `s.iloc[9]` says there is a single positional indexer out of bounds and `s.iat[9]` names the axis and the size, and since the two reach the same binding, the binding raises the `iat` sentence and the `iloc` path checks the bound itself before it asks. Two callers being told two things about the same mistake is not a design, it is pandas, and the cheapest way to match it is to let each caller raise its own.

`Too many indexers` on a series is `s.loc[a, b]`, where the second key names an axis that does not exist. pandas raises its own `IndexingError` there, which is the divergence section 8 already registered for the frame, and it applies here unchanged.

## 12. `get` and `squeeze`, which are square brackets with the failure removed and a shape read off the data

These two are in this document rather than in one of their own because they are indexing wearing other names, and because both of them answer a shape rather than a value, which is the property everything else here has.

`get` is square brackets with the lookup failure turned into a value. That is the whole method on both objects, and it is the only difference between the two, which is the only reason pandas has it: a caller who already has a default in hand does not want a traceback on the way to it. The frame's reads a column name and the series' reads a label, exactly as square brackets do on each, including the slice rule from section 11, so there is one rule to learn rather than two.

The set of failures it catches is one wider than the set pandas catches, and the difference is invisible from outside. pandas reads `df[0]` as a column it does not have and raises a lookup error, and this reads it as a key of a type square brackets do not take and raises a type error, because square brackets here want a name or a list of names and say so. `get` catches both kinds and answers the default, so a caller of `get` sees the same thing from both libraries and the exception neither of them raises is where they differ. Narrowing the catch to match pandas exactly would mean making square brackets answer a lookup error for a key that is not a name, which is a worse message for the larger number of callers who did not write `get`.

`squeeze` is the one that reads a shape off the data. Three answers come out of one name: a frame of one column is that column, a frame of one row and one column is the value in it, and a frame that is neither is itself. The axis parameter names which of the two may go, and the default of `None` lets either. A series has one axis, so it answers its one value when it has one row and itself otherwise, and an axis that names the column axis is refused in pandas' words, which is worth having because a frame and a series get passed to the same function often enough that an `axis=1` arriving at a series is a real mistake.

Two details are worth writing down. The first is that a frame with nothing to drop comes back as a new object rather than as the frame it was given, which is what pandas does, and a caller who tests that is testing something real, so `squeeze` builds a fresh wrapper around the same columns rather than returning itself. The second is that the answer where the row axis goes and the column axis stays is refused, and that refusal is section 6 arriving through a different door. It covers the one column shape as well as the several column shape, even though a single column has a type the row could have used, because the name is still the row label and a series here is still named by a string.
