# M2c, the planner

This is the milestone. It is numbered M2c rather than getting a number of its own for the same reason M2b was: it slots in without renumbering the nine milestones after it, and it is partly a pull forward of work that was sitting in M3.

## Why it exists

We just measured what a planner is worth by doing one by hand. The TPC-H driver in firepanda-bench went from 6.671 seconds for the twenty two queries at sf1 to about 1.4 seconds, and one kernel changed in that time. Everything else was projection pushdown, predicate pushdown, join reordering and operator selection, written out by a person, query by query.

That is a factor of four point eight from decisions firepanda does not make. It is larger than every kernel optimization of the last several releases put together, and it will not transfer to any query a user writes, because the hand tuning lives in a benchmark driver rather than in the engine.

There is a second reason. `firepanda/exec/` already contains a working chunked engine with nine operators and a pipeline driver, and the only thing in the repository that calls it is a test file. The planner is the layer that makes it reachable.

## What it moves from where

M3 was the lazy frame. M2c takes the plan, the binder and the optimizer passes from it and does them now, because the plan is the thing the passes rewrite and building the lazy surface first means building it against an API that does not exist yet.

M3 keeps the lazy user surface, `LazyFrame`, `scan_parquet`, `collect`, `explain` and the streaming collect variants.

## Scope

### Stage zero, no plan required

- [ ] Join build side chosen by size rather than by parameter name, with the output pair order flipped to compensate
- [ ] A sortedness flag on a column, set by sort and by readers that know, cleared by everything that reorders
- [ ] A cached distinct count on a column, written by `factorize` which already computes it
- [ ] A cached minimum and maximum, and an all valid flag, written by the kernels that already know
- [ ] Dense integer key detection at the join, indexing directly rather than hashing to a code first
- [ ] Sorted group by, chosen when the key carries the sortedness flag

### The IR

- [ ] Nine logical node kinds: Scan, Filter, Project, Aggregate, Join, Sort, Limit, Distinct, Union
- [ ] Nine expression kinds: column, literal, unary, binary, cast, call, aggregate, conditional, window
- [ ] Binding, resolving names to positions and types once, with the logical type computed at bind time
- [ ] The elementwise analysis
- [ ] The input independent analysis
- [ ] The table set analysis
- [ ] `explain(optimized=False)` printing the unoptimized form
- [ ] `explain()` printing the optimized form
- [ ] Lowering into the existing `exec` node union, with `Materialize` as the fallback for anything not yet lowerable

### The passes

- [ ] Expression simplification, constant folding, comparison simplification, conjunction flattening, applied to a fixed point
- [ ] Type coercion inserted as explicit cast nodes
- [ ] Projection pushdown
- [ ] Predicate pushdown
- [ ] Transitive predicates across equality join conditions
- [ ] Common subplan elimination
- [ ] Common subexpression elimination
- [ ] Projection merging, adjacent projections into one node
- [ ] Slice pushdown
- [ ] Top n recognition, a limit above a sort becoming the bounded heap we already have
- [ ] Empty and constant pruning
- [ ] `IN` against a large constant list rewritten as a join
- [ ] A semi join whose right side is distinct on the key rewritten as an inner join

### Predicate transfer

- [ ] A blocked Bloom filter over the join's existing 64 bit hashes, sized from the exact build count, one block resident in L1
- [ ] Filter type chosen by distinct count: IN list below about sixty four keys, Bloom above, min and max always
- [ ] The transfer graph, built from the bound plan after predicate pushdown
- [ ] LargestRoot, a maximum spanning tree weighted by the smaller endpoint's cardinality and rooted at the largest relation
- [ ] The forward pass, leaves to root
- [ ] The backward pass, root to leaves
- [ ] The cyclic case, taking the spanning tree and accepting the weaker guarantee
- [ ] The escape hatch, skipping when the query has fewer than two equality joins

### Runtime

- [ ] Join filters built as a side effect of the hash join build and handed to the probe side
- [ ] A filter applied inside the probe loop when the consumer is adjacent, as a mask when it is not
- [ ] Late operator selection for the correlated aggregate, broadcast against group by plus semi join, decided from the measured input
- [ ] The crossover for that choice measured and written down, with the method recorded in the docstring
- [ ] Estimate bail out, re-deciding downstream operators when a join's output exceeds its estimate by a large factor

### Join ordering

- [ ] Greedy ordering from exact row counts, joining the pair whose result is estimated smallest
- [ ] Join output size estimated as the size of the larger side, with the foreign key assumption documented
- [ ] DPccp for fewer than ten relations, only if a measured query still needs it after predicate transfer

### Operators the plan needs and `exec` does not have

- [ ] `Sort`
- [ ] `Distinct`
- [ ] `Union`
- [ ] Semi and anti join kinds as first class nodes rather than through `Materialize`

### Quality

- [ ] Plan tests written as a plan in and a printed plan out, compared as text
- [ ] Every TPC-H query expressible without a hand written plan, and within a few per cent of the hand tuned driver
- [ ] The hand written helpers in the benchmark driver deleted, since binding makes them unnecessary
- [ ] A differential test running each TPC-H query with the optimizer on and off and comparing answers

## Exit criteria

A plain translation of all twenty two TPC-H queries, written against the frame API with no hand planning, performs within a few per cent of the hand planned driver that currently totals about 1.4 seconds at sf1.

The ratio between the fastest and slowest execution of a query over randomly permuted join orders is under 2, measured on the six join heavy TPC-H queries.

`explain()` prints something a person can read, for every query in the benchmark suite.

No user visible behaviour change: the full test suite passes unchanged, and the differential test agrees with the optimizer on and off.

## What this does not do

It does not close the gap on q1 and q6. Those have no joins and nothing to reorder, and their gap is memory traffic, which is M2b. The planner and the streaming engine are two halves and neither finishes the job alone.
