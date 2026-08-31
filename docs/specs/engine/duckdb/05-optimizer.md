# DuckDB optimizer

firepanda has no optimizer and will not have one for a long time, because a dataframe API is eager and there is nothing to optimize between one call and the next. This document is here for two reasons. One is that M3 in the roadmap is a lazy frame, and the moment there is a lazy frame there is a plan and the question comes up. The other is that some of what the optimizer does is the reason DuckDB beats us on queries where our kernels are individually competitive.

## The shape

A logical plan comes out of the binder. The optimizer runs a fixed sequence of passes over it, each of which takes a logical plan and returns a logical plan. Then a physical planner turns the result into operators.

The passes are not a search. There is no cost model choosing between rewrite orders. It is a pipeline of rewrites in a hardcoded order, with one exception, which is join ordering.

## The rewrites

**Expression rewriting.** Constant folding, arithmetic simplification, comparison simplification, moving constants to one side, collapsing nested conjunctions. Applied repeatedly until the expression stops changing.

**Filter pushdown.** The important one. A filter is moved as close to the scan as it can go, so that rows are eliminated before they are joined, aggregated or projected. Pushing a filter through a join also generates transitive predicates: if `a.x = b.x` and there is a filter on `a.x`, the same filter can be applied to `b.x`, which the user never wrote.

**Projection pushdown.** Columns that nothing reads are never scanned. In a column store this is the difference between reading two columns and reading forty.

**Common subexpression elimination**, **unused column removal**, **empty result pruning** where a provably false filter collapses the subtree, and a set of specific rewrites like turning `IN` with a large constant list into a join against a materialized list, as a MARK join or an INNER join depending on the context.

## Join ordering

This is the part that is a search rather than a rewrite.

The joins in a query form a hypergraph. Nodes are relations, edges are join conditions, and the edge is a hyperedge when a condition touches more than two relations. The problem is to find the order of joins with the lowest total intermediate cardinality.

DuckDB uses DPhyp, from the paper "Dynamic Programming Strikes Back" by Moerkotte and Neumann. It enumerates connected subgraphs and their complements in an order that never generates the same pair twice, which is what makes dynamic programming over the hypergraph tractable. Below that there is DPccp for the simpler case. Above a size threshold the search space stops being enumerable and it falls back to a greedy algorithm.

Cardinality estimation feeds this. DuckDB uses the samples and the min/max statistics it already keeps per row group, plus HyperLogLog for distinct counts.

**Build side selection is a separate pass.** Once the order is fixed, each individual hash join still has to decide which side to build the table from. That is `BUILD_SIDE_PROBE_SIDE`, distinct from `JOIN_ORDER`, and the two can be disabled independently. The rule is to build from the smaller side, since the hash table is what has to be resident.

That second one is directly relevant to firepanda. `firepanda/join/pairs.mojo` builds from the right side because the API says right, not because the right side is smaller. In db-benchmark j4 and j5, which join ten million to ten million, that does not matter. In a query where the user wrote the tables in the unhelpful order it matters a great deal, and it is a cheap thing to fix: compare the two lengths and swap, flipping the output pair order to compensate.

## What a dataframe library can and cannot do here

An eager API sees one operation at a time. `df.filter(...).group_by(...).agg(...)` in an eager library runs the filter, materializes, then groups. There is no plan and no opportunity for pushdown.

That is exactly what a lazy frame buys. Polars' whole design is built around this and `polar/02-lazy-ir.md` covers it. The pushdowns are worth more than any kernel level optimization, because not reading a column at all beats reading it fast.

There is one thing an eager API can still do, which is defer within a single call. `read_parquet` with a column list is projection pushdown by another name. `read_csv` with a predicate is filter pushdown by another name. M2 is about readers and both of those belong in it.

## What we should take from this document

Build side selection by size rather than by argument order. Small, local, and we should do it now.

Projection and predicate pushdown into the readers in M2, since a reader that takes a column list is doing the useful half of an optimizer without needing a plan.

The full pass pipeline and DPhyp are M3 material and belong in the lazy frame spec, not here. Note for then: DPhyp is the right algorithm and the fallback threshold matters more than the algorithm, because most dataframe queries join fewer than ten relations and every algorithm is exact at that size.
