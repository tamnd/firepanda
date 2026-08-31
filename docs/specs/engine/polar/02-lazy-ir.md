# Polars lazy plans and the IR

## Three representations

**The DSL.** What `pl.scan_parquet(...).filter(...).group_by(...).agg(...)` builds. A tree of nodes that mirrors what the user wrote, with expressions as their own tree. No schema resolution has happened. `LazyFrame` holds one of these and building it does no work.

**The IR.** The result of running the DSL through the conversion and optimization passes. Schemas are resolved, column references are bound, and the rewrites have been applied. This is the thing the physical planners consume, and both the streaming engine and the distributed planner consume the same IR, which is why a new backend is a new physical layer and not a new frontend.

**The physical plan.** Engine specific. Document 03 covers the streaming engine's.

`explain()` prints the optimized plan. `explain(optimized=False)` prints the DSL.

## The optimizer passes

Roughly the same list as DuckDB's, arrived at independently, which is a reasonable signal that the list is right.

**Projection pushdown.** The single largest win in a column store. If the query selects three columns out of forty, the scan reads three. Users notice this one because it turns a Parquet read from ten seconds into one.

**Predicate pushdown.** Filters move toward the scan. Parquet gets them as row group statistics checks, so whole row groups are skipped without decoding. Hive partitioned datasets get them as directory pruning, so files are never opened.

**Slice pushdown.** A `head(10)` at the end tells the scan to stop early. This is the one that makes `.head()` on a lazy scan of a huge file instant, and it is a thing eager libraries cannot do at all.

**Common subplan elimination.** If the same subtree appears twice, it is computed once and shared. Turned on by default in `collect()`.

**Simplify expressions**, constant folding, type coercion inserted as explicit cast nodes rather than done implicitly in the kernels, and **cluster with columns**, which merges adjacent `with_columns` calls so that ten chained calls become one projection node.

Join reordering exists but is much less developed than DuckDB's. There is no DPhyp. This is the clearest place Polars is behind, and it shows up on the PDS-H queries with many joins.

## Streaming lowering

The IR is not directly executable by the streaming engine. `lower_ir()`, in `crates/polars-stream/src/physical_plan/lower_ir.rs`, walks the IR and produces a graph of `PhysNode`s held in a `SlotMap<PhysNodeKey, PhysNode>`. A `PhysStream` is a handle to one output port of one node, which is how a node with multiple consumers is expressed.

The important property of `lower_ir` is that it is partial. When it meets an IR node with no streaming implementation, it wraps the subtree in `InMemoryMap` or `InMemoryJoin`, which run the old engine on a materialized input and hand the result back into the stream. So a plan is streaming where it can be and not where it cannot, on a node by node basis, and adding a streaming operator is a local change that removes one fallback.

That design is why the migration was survivable, and it is the single most important structural idea for firepanda to copy. A rewrite that requires every operator to be ported before anything works never ships.

## Expression lowering

Expressions become `StreamExpr`. Two analyses decide how:

`is_elementwise_rec_cached()` asks whether an expression is row wise, meaning the value at row i depends only on row i. Elementwise expressions can run on a morsel with no state at all. `a + b * 2` is elementwise. `a.sum()` is not. `a.rank()` is not.

`is_input_independent_rec()` asks whether an expression depends on the input at all. A literal or a computation over literals does not, so it is evaluated once and becomes a constant rather than being recomputed per morsel.

The elementwise analysis is what makes a `select` or a `with_columns` cheap in the streaming engine. It applies per morsel with nothing carried between them.

## The node kinds

From `PhysNodeKind`, and the list is short enough to be worth writing out because it is a good specification of the minimum operator set for a streaming engine:

Sources: `InMemorySource`, `MultiScan`.

Elementwise: `Select`, `SimpleProjection`, `WithRowIndex`, `Filter`.

Slicing: `StreamingSlice`, `NegativeSlice`. The negative case is separate because taking the last n rows needs a buffer of n rows and the streaming case does not.

Aggregation: `Reduce` for a whole frame reduction, `GroupBy`, `SortedGroupBy`. The sorted variant is chosen by `try_build_sorted_group_by()` in `lower_group_by.rs` when the key is known sorted, in which case grouping is a scan with no hash table at all.

Joins: `EquiJoin`, `MergeJoin`, `CrossJoin`. Semi and anti live in `nodes/joins/semi_anti_join.rs`.

Fallbacks: `InMemoryMap`, `InMemoryJoin`.

That is about sixteen node kinds for a complete engine. It is not a small amount of work but it is a countable amount, and roughly half of them firepanda already has as kernels.

## What we should take from this document

The partial lowering with an in memory fallback node. This makes a streaming engine something we can ship incrementally, one operator at a time, with a working system after every step.

The elementwise analysis, because it is what makes projections free.

The node kind list as the definition of done for a streaming engine.

Projection and predicate pushdown at the reader, which is the part of a lazy plan that pays for itself before there is a lazy plan.
