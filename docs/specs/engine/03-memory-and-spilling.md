# Memory and spilling

## The current position

Every firepanda kernel allocates an output as long as its input and assumes that works. A group by builds one hash table over the whole column. A join materializes both sides. There is no memory limit, no accounting, and no behaviour other than allocating until the allocator fails.

For the workloads we benchmark this is fine. For the workload a user actually has, which is a Parquet dataset larger than their laptop, it is the difference between a tool and a demo. DuckDB's own headline example is fifty gigabytes producing a billion groups on a sixteen gigabyte machine, finishing in 264 seconds. Thirty times slower than fitting, and it finished.

## The choice

Two designs, described in `duckdb/00-overview.md` and `polar/04-out-of-core.md`.

**Buffer manager.** Everything large lives in blocks. Operators pin and unpin. Unpinned blocks can be written to a temp directory. Spilling is not a mode an operator enters, it is what happens under memory pressure. DuckDB.

**Operator controlled.** A memory manager that operators consult, and each spillable operator implements its own policy. Polars.

**We take the second.** Three reasons.

It is smaller. A buffer manager is a subsystem every allocation in the library has to go through, and it changes how every kernel is written. Operator controlled spilling is three operators.

It keeps control where the knowledge is. The Kuiper VLDB paper makes this point about DuckDB's own hash join: a page the operator is about to read can be evicted, and the only way to prevent it is to pin, which is to say to manage memory yourself after all.

It fits the scheduler from `02-execution-model.md`. Because the scheduler owns the morsel queue, an operator that is flushing to disk declines to take a morsel and the morsel waits. That is exactly the argument Polars gives for why the scheduler holds the queue rather than the producer pushing, and we get it for free from the design we were going to build anyway.

## The memory manager

Small. A configured limit, an atomic current total, and a registration call.

```
reserve(bytes) -> Bool     take it if there is room, otherwise say no
release(bytes)             give it back
pressure() -> Float64      current over limit, for operators that want to act early
```

Operators that can spill call `reserve` before growing and spill when it says no. Operators that cannot spill do not call it at all, which keeps the change contained. The limit defaults to a fraction of physical memory and is settable.

Deliberately not a page cache. No pinning, no eviction, no block identity. It counts.

## The three operators

### Group by

Partition on the group hash. Enough partitions that a partition is individually small, more partitions than workers on purpose, following `duckdb/04-operators.md`, so that only a few need to be resident.

A worker accumulates into a thread local table. When `reserve` says no, the table's partitions are written out and a fresh table starts. At merge time, partitions are read back one at a time and merged, and because two rows with different hashes cannot be the same group, a partition merges with no coordination.

This is also the change that makes group by parallel in a way it currently is not. Today `firepanda/hash/grouping.mojo` factorizes the whole key into codes and then aggregates by code. Partitioning replaces the global factorize with per partition ones.

The one caution, from `duckdb/04-operators.md`: the 2025 paper "Global Hash Tables Strike Back" argues a good global concurrent table beats the partitioned design in many cases. Our global table is fast. Measure the partitioned version against it in memory before assuming partitioning is a win for the fitting case as well as the spilling one.

### Join

Build side partitioned, partitions spilled when they do not fit, probe rows routed by the same radix bits to the resident partitions and spilled otherwise, then replayed. Both rival documents describe the same thing.

Two details from `duckdb/04-operators.md` worth writing down now. Right, outer and anti joins need the probe side partitioned rather than rescanned, because they need to know which build rows never matched. And if a single key is a large fraction of the input, no number of radix bits separates it, so there has to be a bail out that stops increasing the bit count and falls back to a nested loop for that partition.

### Sort

Runs built in memory, spilled, merged. `firepanda/kernel/sort.mojo` is a good in memory sort and this wraps it.

## The edges

From `polar/04-out-of-core.md`. When the plan is a graph rather than a chain, a node with two consumers has to buffer for the slower one, and that buffer is unbounded. Polars needed a separate pull request to make the multiplexer out of core, after the operators were done.

We do not have a graph yet. `02-execution-model.md` describes chains. Writing this down so that the day a node gets two consumers, the buffer is accounted for from the start.

## Order

The memory manager first, doing nothing but counting, so that we can see what a query actually uses before deciding what to spill.

Then sort, because it is the simplest and the pattern is textbook.

Then group by, because it is the one users hit.

Then join.

None of this is on the critical path for the speed numbers. It is the capability half of M2b and it can trail the execution model work by a milestone if the schedule demands it.
