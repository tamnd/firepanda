# What a plan is

Before anything can be optimized there has to be something to optimize. firepanda has no such thing today: `df.filter(...).select(...).group_by(...)` runs the filter, materializes it, runs the select, materializes that, and so on, and by the time the group by is called the information that a filter happened is gone.

## The three representations

Polars keeps three and the separation is worth copying exactly.

**The DSL** is what the user's calls build. A tree that mirrors what was written, with expressions as their own tree hanging off the nodes. No schemas resolved, no columns bound, no checking beyond what can be done syntactically. Building one is free.

**The IR** is the DSL after conversion and optimization. Schemas resolved, column references bound to positions rather than names, casts inserted explicitly rather than left implicit, and every rewrite applied. This is what a physical planner consumes.

**The physical plan** is engine specific and is where the operator choices live.

The reason to keep the DSL and the IR apart, rather than optimizing in place, is that the DSL is what you print when the user asks what they wrote and the IR is what you print when they ask what will run. Polars exposes both through `explain(optimized=False)` and `explain()`, and every hour spent debugging a plan is spent looking at those two side by side.

## Nodes

A logical plan node is one of a short list. Polars' `PhysNodeKind`, quoted in `engine/polar/02-lazy-ir.md`, has about sixteen entries for a complete engine, and the logical list is shorter than the physical one because several physical nodes are alternative implementations of one logical node.

The logical list firepanda needs:

**Scan.** A table, a file, or an in memory frame. Carries a column list and, once pushdown has run, a predicate and a row limit.

**Filter.** A predicate over its input.

**Project.** A list of output expressions. `select` and `with_columns` are both this, and merging adjacent ones is a rewrite.

**Aggregate.** Group keys and aggregate expressions. An empty key list is a whole frame reduction.

**Join.** Two inputs, a key pair list, and a kind. Inner, left, right, outer, semi, anti, cross, and the mark join that an `IN` subquery becomes.

**Sort.** Keys, directions, null placement.

**Limit.** Offset and length. Separate from sort because a limit on an unsorted input is a different operator and because slice pushdown treats it separately.

**Distinct.** Keys, or the whole row.

**Union** and **Concat**, which are cheap and get forgotten until a query needs them.

Nine node kinds. That is the whole logical language, and every TPC-H query is expressible in it.

## Expressions

An expression tree is separate from the plan tree and it is where most of the code goes.

The node kinds are column reference, literal, unary and binary operation, cast, function call, aggregate, conditional, and window. Each carries the logical type it produces, computed once during binding rather than rediscovered at every use.

Two analyses over expressions do most of the work downstream, both taken from Polars:

**Elementwise.** The value at row `i` depends only on row `i`. `a + b * 2` is elementwise, `a.sum()` is not, `a.rank()` is not. An elementwise expression can run on a morsel with no state carried between morsels, which is what makes projections free in a streaming engine. The analysis is a recursive walk and the answer is statically obvious for everything in firepanda's kernel set.

**Input independent.** The expression does not reference the input at all. A literal, or arithmetic over literals, or `date '1994-01-01' + interval '1 year'`. Evaluate once at plan time and replace with a constant. This sounds trivial and it is not: TPC-H q1 has `date '1998-12-01' - interval '90 days'` in its predicate, and an engine without this analysis computes it six million times.

A third analysis we need that Polars does not spell out the same way:

**Table set.** Which input relations an expression touches. This is the analysis that predicate pushdown and predicate transfer both run on, because a predicate can only be pushed to a subtree that provides every column it references.

## Binding

Binding turns a name into a position and a type. It runs once, over the DSL, producing the IR.

This is where firepanda gets a fix it has needed for a long time for free. `DataFrame.column(name)` copies and flattens the column, and `DataFrame.__getitem__(i)` borrows. Every hand written query in the TPC-H driver has helper functions whose only job is to look a name up and then use the borrowing accessor, because the by name one costs ninety six megabytes on a text column of six million rows. A bound plan holds positions, so there is no by name lookup at execution time at all and the question stops existing.

## What the eager surface does with all this

Nothing, at first, and that is the point.

`df.filter(...)` on an eager frame still runs a filter and returns a frame. The plan layer sits behind a lazy frame, which is milestone M3, and the eager surface is a thin wrapper that builds a one node plan and collects it immediately. That keeps two things true: the eager API does not change, and there is exactly one execution path rather than two that drift.

There is one thing an eager API can do with a plan and it is worth doing early. A single eager call that is internally several operations, `read_parquet` with a column list and a predicate being the obvious one, is a plan of two or three nodes and it can be optimized before it runs. That is projection and predicate pushdown into the reader, which `engine/duckdb/05-optimizer.md` already flags as the part of an optimizer that pays for itself before there is an optimizer.

## What we should take from this document

Three representations, kept apart, with both of the first two printable.

Nine logical node kinds and nine expression kinds. Write the list down and refuse to add to it without an argument.

The elementwise, input independent and table set analyses, all three, early, because every later pass is written in terms of them.

Binding to positions, which retires the copying by name accessor rather than working around it.
