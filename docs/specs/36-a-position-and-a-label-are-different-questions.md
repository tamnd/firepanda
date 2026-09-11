# 36. A position and a label are different questions

Status: implemented, apart from one shape of answer. Two accessors, two more beside them, one method, and four new ways across the boundary.

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
