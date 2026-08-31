# The firepanda execution model

This is the design. `01-what-we-take.md` is the argument for it and `05-m2b.md` is the checklist.

## Where we start from

`Frame` holds a `List[AnyArray]`, one contiguous array per column. Every kernel takes an `Array[dt]` and returns a new `Array[dt]`. `parallel_for(body, count)` in `firepanda/exec/parallel.mojo` runs a body once per index and the caller slices the column into `worker_count()` equal pieces itself.

`ChunkedArray` exists in `firepanda/array/chunked.mojo` and is used by nothing except its own test. Its docstring already says the right thing, which is that reading Parquet gives one array per row group and that scan, filter and aggregate are happy per chunk. It was written for this and then never wired up.

## The four changes

### 1. Chunks become the unit

`ChunkedArray` becomes what a `Frame` column is, and the kernels grow a chunked entry point.

Concretely, for each kernel there are two functions. The inner one keeps today's signature, one contiguous array in, one out, and does not change. The outer one walks chunks and calls the inner one per chunk. Nothing gets slower, because a single chunk column is the same work it is today plus one loop iteration.

This has to be first because everything else assumes it.

Chunk size when we create chunks ourselves is 128 thousand rows, from `duckdb/02-storage.md` and `polar/03-streaming-engine.md`, which agree. When a reader gives us chunks we keep the reader's boundaries.

### 2. Morsels, and a scheduler that hands them out

Replace the fixed split with a work queue.

`parallel_for` stays as it is. It is the right primitive and the docstring correctly says a scheduler nothing feeds is a liability. On top of it goes a `MorselQueue` holding an atomic cursor, and a `parallel_morsels(body, total, morsel_rows)` that starts `worker_count()` tasks, each of which loops taking the next morsel from the cursor until it is empty.

The measurement that motivates this is our own. On a machine with sixteen physical and thirty two logical cores, the optimal worker count for a low cardinality group by came out at twenty four. That is not a property of the machine, it is what a fixed equal split does when some slices cost more than others: the whole thing waits for the slowest slice, and fewer, larger slices happened to balance better. A work queue makes the number twenty four disappear.

This change is independently benchmarkable and does not depend on anything else. It could ship before change 1.

### 3. Pipelines

A `Pipeline` is a source, a list of operators, and a sink. An operator takes a chunk and produces a chunk. The source produces chunks, the pipeline pushes each one through every operator, the sink consumes them.

A `Node` interface with three methods, taken from `ComputeNode` in `polar/03-streaming-engine.md`:

```
update_state()   what this node wants next, and whether it is finished
process(chunk)   transform one chunk, called per morsel per worker
finish()         produce the node's result, for nodes that have one
```

Pipelines are cut at breakers. A breaker is any node that cannot emit until it has seen all its input: the build side of a join, a group by, a sort, a top n. Cutting is one traversal of the operator list.

**The fallback node is not optional.** There is a `Materialize` node that collects every chunk into one contiguous array, calls today's whole column kernel, and emits the result as chunks. Any operation with no chunked implementation goes through it. That is what makes this a sequence of small pull requests instead of a rewrite: on day one every operator is a `Materialize` and the system is exactly as fast as it is today, and each subsequent pull request removes one.

Polars shipped `InMemoryMap` and `InMemoryJoin` for this and still has them at 1.39, eighteen months in. We should expect the same and not treat the fallback as temporary.

### 4. Selection vectors

A chunk carries an optional selection vector, which is a list of indices into its arrays. `filter` writes a selection vector instead of copying. `take` writes a selection vector instead of copying. A join output is two selection vectors, one per side.

Kernels read through it. The shape is DuckDB's unified format from `duckdb/01-data-model.md`: a data pointer, a validity bitmap and a selection, with `data[sel[i]]` correct whether or not there is a real selection.

A `flatten()` materializes when something genuinely needs contiguous memory, and the rule for when to flatten is the one place this design has a real tuning question. A selection that keeps one row in a thousand makes every subsequent read a random access, and materializing is better. A selection that keeps nine in ten should stay a selection. Start with a fixed threshold and measure.

This is last because it is the one that touches every kernel signature.

## What this does not change

The kernels. `firepanda/kernel/agg.mojo`, `arith.mojo`, `compare.mojo` and the rest are the inner loops and they stay. This is a change to what calls them and how often, not to what they do.

The hash tables. `firepanda/hash/` is good. The salt byte from `duckdb/04-operators.md` is a separate small improvement, not part of this.

The public API. `df.filter(...)` still returns a `Frame` and still runs eagerly from the user's point of view. The pipeline is built and executed inside the call. A lazy frame that lets a user build a plan across calls is M3, and this design is what makes that possible without a second engine, which is exactly the layering point from `polar/05-distributed-and-gpu.md`.

## What we expect to get

Polars measured three to seven times from this change, growing with data size, on a benchmark much heavier than db-benchmark.

Our situation differs in one way that matters. db-benchmark queries are short chains: scan, group by, aggregate. There are not five operator boundaries to remove, there are one or two. So the honest prediction at 0.5GB is modest, maybe twenty to forty percent on the group bys, and more on the joins because a join is where we currently do the most materializing.

The larger number arrives at larger scale, and the first thing `05-m2b.md` should do is get firepanda-bench running at 5GB so the effect is visible at all.

There is a second thing we get that is not a speed number, which is that a chunked engine can process a file larger than memory. Today every kernel allocates an output as long as its input. That is a capability gap, not a performance gap, and `03-memory-and-spilling.md` covers it.
