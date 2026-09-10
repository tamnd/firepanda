# The plan

What firepanda builds, in what order, and what each step is expected to be worth. Every entry is one pull request or a small number of them.

The order is by benefit divided by cost, and the first section needs no planner at all.

## Where we actually are

Worth stating precisely, because it changes what comes first.

`firepanda/exec/node.mojo` already contains a physical operator set: `Filter`, `Project`, `Limit`, `Compute`, `Cast`, `Join`, `Group`, `Reduce` and a `Materialize` fallback, with `node_bind` resolving schemas and a closed union for dispatch. `firepanda/exec/pipeline.mojo` has `Scan`, `Collect` and a `Pipeline` driver that pushes chunks through them. That is a real chunked execution engine and it works.

Nothing in the public API reaches it. The only callers of `Pipeline` in the whole repository are in `tests/test_group_node.mojo`. Every `DataFrame` method runs a kernel over a whole column and materializes.

So the missing piece is not the engine and not the operators. It is the layer that turns what a user asked for into a graph of nodes, and the passes that improve that graph on the way. That is what makes the physical engine reachable and it is what this folder is about.

## Stage zero: things that need no plan

These should ship as ordinary pull requests before anything else here starts.

**Build side by size.** `firepanda/join/pairs.mojo` builds from the right side because the parameter is called right. Compare the lengths, build from the smaller, flip the output pair order. About twenty lines. Nothing on the symmetric db-benchmark joins, a large win the first time a user writes the tables in the unhelpful order.

**Column metadata.** A distinct count, a sortedness flag, a minimum and maximum, and an all valid flag, cached on a column when a kernel computes one anyway and invalidated on write. `factorize` produces a group count as a side effect and throws it away; the compare kernels know whether a mask came out all true. Everything in document 07 depends on this and none of it is new computation.

**Dense integer key detection at the join.** When the build side key is an integer whose range is dense enough, index directly instead of hashing. We already detect a unique build side and build a direct table; this removes the factorize pass in front of it. On db-benchmark j4 that pass is 395 milliseconds inside a 635 millisecond query.

**Sortedness flag, then sorted group by.** A group by whose key is known sorted needs no hash table. `sort_values` sets the flag, readers that know set it, and everything that reorders rows clears it.

Expected: j4 and j5 substantially, and a class of user query where the tables were written in the wrong order.

## Stage one: the plan layer

**The IR.** Nine logical node kinds and nine expression kinds, from document 01. A builder that produces one, a printer that prints it, and a lowering pass that turns it into the existing `Node` union. At the end of this the physical engine is reachable and the eager API is a one node plan collected immediately.

**Binding.** Names to positions and types, once. This retires `DataFrame.column`'s copy as a problem rather than working around it, which every helper in the TPC-H driver currently does by hand.

**The three analyses.** Elementwise, input independent, table set. Everything after this is written in terms of them.

**Projection pushdown.** The largest single pass. Ours by hand was most of the distance from 6.671 seconds to about 2 across the twenty two TPC-H queries.

**Predicate pushdown, with transitive predicates.** Second largest. Ours by hand: q19 83 milliseconds to 70, q21 285 to 169.

**Projection merging.** Adjacent `with_columns` become one node. Removes the copy that `DataFrame.add_column` currently exists to let a caller avoid by hand: q1's two additions were 93 milliseconds and are 37.

**Expression simplification and constant folding.** Cheap, and it stops `date '1998-12-01' - interval '90 days'` being evaluated six million times.

**Slice pushdown and top n recognition.** A limit above a sort is a bounded heap, and we already have the heap.

Expected: this is where the TPC-H set stops needing a hand written driver. The target is that a plain `LazyFrame` translation of each query performs within a few per cent of the hand tuned Mojo we have now.

## Stage two: predicate transfer

Document 04, and the reason document 03 is short.

**The filter type.** Blocked Bloom over the 64 bit hashes the join already computes, sized from the exact build count, one block resident in L1. An IN list below about sixty four distinct keys, which is `firepanda/kernel/member.mojo` and already exists. Minimum and maximum always.

**The transfer graph.** Nodes are scans, edges are equality conditions, built after predicate pushdown so the pushed predicates are the seeds.

**LargestRoot.** Maximum spanning tree weighted by the smaller endpoint's cardinality, rooted at the largest relation. Forward pass leaves to root, backward pass root to leaves.

**The escape hatch.** Fewer than two equality joins, skip.

Expected, from the paper's DuckDB integration across TPC-H, JOB and TPC-DS: the ratio between the worst and best random join order on an acyclic query drops to 1.6, and end to end improves by 1.5 times as a geometric mean. Our own hand versions of two special cases of this were worth a factor of two on q7 and a fifth on q19.

## Stage three: runtime

**Runtime join filters.** Document 06. Build the filter as a side effect of the hash join build and hand it to the probe side. Shares everything with stage two except who decides and when.

**Late operator selection.** Document 07. The broadcast versus group and join choice, made from the measured input rather than hardcoded per query. Between a factor of three and a factor of eight, in opposite directions, depending on group count.

**Estimate bail out.** When a join's output exceeds its estimate by a large factor, re-decide downstream from the measured number.

## Stage four: join ordering

Greedy first, from exact row counts. DPccp for fewer than ten relations only if a measured query still needs it after stage two. DPhyp last and possibly never.

## What this is expected to add up to

The current four engine TPC-H sf1 result on the 13900K is firepanda 1.389 seconds against Polars 0.995, DuckDB 1.259 and pandas 5.553, with every firepanda query hand planned.

Stage one should get a machine planned query to where the hand planned one is now, which is the point at which the result stops depending on a person having read the query. Stage two is what closes the remaining gap on the join heavy queries, which are q5, q8, q9, q10, q20 and q21, together about 0.6 seconds of the 1.389. Stage three is the group by heavy ones.

The 2x goal against all three rivals is not reachable from plan work alone, because q1 and q6 have no joins and no plan to improve, and their gap is memory traffic, which is `engine/02-execution-model.md`. Plan work and the streaming engine are the two halves and neither one finishes the job alone.

## What we should take from this document

Stage zero now, because none of it needs a plan and one item of it is 395 milliseconds on a benchmark we already run.

Stage one is the milestone. It is what makes the engine we already wrote reachable.

Stage two is where the differentiated performance is, and it is a smaller amount of code than stage one.
