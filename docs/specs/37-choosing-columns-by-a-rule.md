# 37. Choosing columns by a rule

Status: implemented. Three methods, one new binding, and a type tree that is copied from numpy rather than designed.

## 1. Why these three are one piece of work

`filter`, `select_dtypes` and `truncate` look like three unrelated methods and they are the same method three times. Each one computes a set of labels from a rule the caller described rather than from labels the caller wrote down, and each one then hands that set to a narrowing that already existed. Document 36 built the narrowing; this is three rules on top of it and nothing else.

Writing them separately would have meant three copies of the same last two lines and three separate decisions about what a rule is allowed to see. Written together, the shape is obvious: a rule reads labels and answers labels, and nothing in a rule touches a value. That is why none of the three has a kernel, and why `select_dtypes` reads the schema and never the columns.

## 2. Asking a frame what it holds without reading what it holds

`select_dtypes` needs the type of every column. Until now the only way to get one was `column(name).dtype`, and `column` copies, because a `Series` is one contiguous array and a column in pieces has to be stacked on the way out. So asking a frame of five hundred columns which of them are numbers would have copied the whole frame in order to look at five hundred strings.

The types are in the schema, which is the frame's shape and is already in memory. `PyDataFrame.dtypes` reads them there and answers a list of strings. Knowing the shape should not cost the contents, and a method whose whole job is to look at the shape is exactly where that rule shows.

It is a binding rather than a loop in Python for the same reason `cell` is. The loop would be one crossing per column and this is one crossing.

## 3. The type tree is numpy's and it is copied without tidying

`select_dtypes(include="number")` takes a column of spans as well as the integers and the floats. That is not a bug here and it is not really a bug in pandas either. numpy's scalar type hierarchy makes `timedelta64` a subclass of `signedinteger`, pandas asks the question with `issubclass` against that hierarchy, and the answer falls out. A caller who wrote `include="number"` on a frame with a duration column gets the duration column from pandas, and a compatibility layer that decided this was untidy and left it out would be answering a different question from the one their pandas answers.

Two more rows of the tree are worth naming because they are pandas correcting numpy rather than inheriting from it. `include="int"` takes int32 and int64 and not the unsigned widths, because numpy resolves a bare `int` to one concrete type whose width depends on the platform and pandas widens it back out to both by hand so that the same code selects the same columns on Windows and on Linux. `include="float"` is widened the same way to float32 and float64. Meanwhile `include="integer"` is the abstract class and does take the unsigned widths. So `int` is narrower than `integer`, which reads backwards and is what both libraries do.

The tree is written out in a table in the pandas layer rather than computed, because firepanda does not import numpy and there is nothing here to ask. The table is thirteen rows and the branches are `signed`, `unsigned`, `floating`, `span`, `naive` and `aware`. A column's type carries the branch names above it, a caller's word resolves to a set of branch names, and a column is selected when the two sets meet. That is `issubclass` with the tree spelled out, which is the only form available to a library that does not have the tree.

The zone is a branch and not a flag, which is why `datetime64` does not select a column that has one. firepanda writes the zone into the printed name after a comma and so does pandas, and `datetimetz` is the word for the other branch. Two columns of instants that differ only in whether they are anchored are different types on both sides.

## 4. What `include="object"` does, which is nothing

pandas 3 still selects string columns for `include="object"`, under a deprecation warning saying that it will stop. firepanda refuses the word outright, with the message document 22's table already carries: Arrow has no type that holds anything at all, so there is no column for an object dtype to be, and the type that holds text is `str`.

This is the same refusal `astype("object")` already makes and it is deliberately not softened here. Accepting the word in one method and refusing it in another would mean a caller learns that firepanda has an object dtype from `select_dtypes` and learns that it does not from `astype`, which is worse than being told the same thing twice.

## 5. The three rules of `filter`, and why two of them keep the frame's order

`items` keeps the order it was written in. `like` and `regex` keep the frame's own order. That is pandas' rule and it is the right one: a list of items is a sequence the caller wrote and a substring or a pattern describes a set, and a set has no order to inherit except the frame's.

The three are exclusive and passing two of them is a `TypeError` rather than a preference. pandas says so and the reason is worth stating: a call that passed both `items` and `like` meant something the signature cannot express, and answering one of them would answer a question nobody asked.

An item that is not there is dropped rather than refused, which is the one place in this file where a missing label is not an error. It is not an inconsistency with `loc`, where a missing label is a `KeyError`. `loc[["a", "b"]]` is a request for two things and getting one of them back is a failure. `filter(items=["a", "b"])` is a request for whichever of two things the frame has, which is what the word filter means.

Along the rows, `filter` compares the labels as text, because two of its three rules are text rules. A frame with integer labels and a `like` of `"1"` selects the rows labelled 1, 10 and 11, which is what pandas answers and is worth knowing before writing it.

## 6. `truncate` is `loc` with two rules in front of it

`df.truncate(before=x, after=y)` is `df.loc[x:y]`, which document 36 already built, plus a check that the index is sorted and a check that the pair is the right way round.

The sorted check is the interesting one. A label slice on an unsorted index is not meaningless: `slice_locs` will find the first label and the last label and answer the rows between those two positions. It is just not what the caller asked for, because a caller who wrote two labels meant the values between them and got the rows that happen to lie between two positions instead. pandas refuses rather than answering that, and so does this.

The pair check happens before the direction is worked out, which matters on a falling index. On an index that runs downwards the row nearer the top of the frame carries the larger label, so the two arguments are swapped before the slice, and the check is still written against the unswapped pair. The upshot is that a caller writes the smaller label as `before` on both kinds of index, and `truncate(before=6, after=3)` is an error on both. That is pandas' order and it reads as arbitrary until the alternative is written out, which is that the same call means an error on one index and a full answer on another.

`truncate` takes an axis and the column names get the same two rules, since the column names are an ordered set of labels and the question is the same one.

## 7. What is refused, and what each would cost

| Argument or shape | What it would take |
| --- | --- |
| `select_dtypes(include="object")` | An Arrow type that holds anything at all, which is section 4 |
| `select_dtypes(include="period")` | A period dtype, which is its own piece of work |
| `truncate(copy=True)` | An owned buffer, which is the decision every `inplace=` is waiting on |
| `filter(items=...)` on a MultiIndex | A MultiIndex, which is issue 155 |

Nothing in the first column is a kernel. Three of the four are decisions made elsewhere in the library and showing up here, which is the usual shape once the primitives are written.

## 8. What it is worth

Seven runs of the conformance suite and three names off the unimplemented list. That is a smaller return than document 36's thirteen cases, and the reason is that these three are rules and rules are cheap once the thing they feed is written. The expensive part was `loc`, and this is what having it looks like afterwards.

The `dtypes` binding is worth more than the seven runs. Every question about a frame's shape that does not need its contents now has somewhere to be asked from, and `select_dtypes` is the first of several: `infer_objects`, `convert_dtypes` and the whole of `info` are the same read.
