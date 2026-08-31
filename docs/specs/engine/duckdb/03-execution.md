# DuckDB execution

This is the document that matters most.

## Push, not pull

The textbook query engine is a pull model. The root operator calls `next()` on its child, which calls `next()` on its child, and tuples come back up the chain. Volcano. Every dataframe library that grew out of one kernel per operation is doing a degenerate version of this, where `next()` returns the entire column.

DuckDB moved to a push model in 2021. The plan is cut into pipelines. Each pipeline is a source, then a chain of operators that transform a chunk into a chunk, then a sink. The source produces a chunk, the chunk is pushed through every operator in the chain, and the result is pushed into the sink. Nothing calls back down.

A pipeline ends wherever an operator cannot produce output until it has seen all of its input. Those are pipeline breakers: the build side of a hash join, a group by, a sort, a top N. Pipelines are found by one traversal of the plan tree, cutting at every breaker.

For a join of two tables with a filter and an aggregate, that gives roughly:

```
pipeline 1   scan build side  ->  filter  ->  sink: hash table build
pipeline 2   scan probe side  ->  filter  ->  probe hash table  ->  sink: aggregate
pipeline 3   source: aggregate  ->  sink: result
```

Pipeline 1 must finish before pipeline 2 starts, because the hash table has to exist. Within a pipeline everything runs at once.

## Why push is better here

Two reasons that are usually given, and one that is more important than either.

The usual ones are that a push model has no per tuple virtual call and that it makes parallelism easier to express, since a pipeline is a unit of work you can hand to a thread.

The one that matters more is cache. In the push model a chunk of 2048 values enters at the source and does not leave the cache until it reaches the sink. Scan, filter, project, probe and aggregate all happen while those values are hot. In firepanda's current model, a filter reads ten million values, writes ten million values, and then the next kernel reads them back from memory. Every operator boundary is a full round trip through DRAM.

That is the whole gap. At ten million int64 rows a column is eighty megabytes, which is far outside any cache, so a five operator query does five write and five read passes over eighty megabytes that a chunked engine does not do at all.

## Morsel driven parallelism

The scheduling model comes from the HyPer paper by Leis and others, and DuckDB, Polars and the Polars GPU backend all implement it.

The idea is that you do not divide the work up front. You chop the input into small fragments, called morsels, put them in a queue, and let threads take one when they are free. A thread that finishes early takes another. Skew in the data or in the machine gets absorbed automatically, and there is no partitioning decision to get wrong.

In DuckDB the morsel is a row group at the scan, and chunks of 2048 within it as data flows. Pipelines are broken into tasks, tasks go on a queue, and the executor runs them. `PipelineExecutor::Execute` and `FetchFromSource` are the entry points to read if you want the source.

Two properties of the model are worth stating because they explain why it stays simple.

Contention is only at the two ends. The scan hands out morsels under a lock or an atomic, and the sink merges thread local state. Everything in between, the filters, the projections, the probes, has no shared state and does not need to know parallelism exists.

Scan parallelism is optional per source. A table function that does not implement it just runs on one thread, and the rest of the pipeline still parallelizes downstream of the sink of the previous pipeline. That is a useful escape hatch for a library that has to support formats it has not parallelized yet.

## Spilling and the buffer manager

DuckDB is often called an in memory database and it is not one. Every intermediate that can be large lives in blocks owned by the buffer manager, and a block that is unpinned can be written to the temp directory and read back.

The pattern operators follow is: materialize into a row layout in blocks, swizzle pointers to variable length data into offsets, unpin, and recompute pointers on the way back in. That last step is why DuckDB does not pay a serialization cost to spill. The bytes on disk are the bytes in memory with pointers turned into offsets.

The consequence for a user is that a query that does not fit gets slower rather than failing. The external aggregation post reports a fifty gigabyte input producing a billion groups completing in 264 seconds on a sixteen gigabyte laptop, against about eight seconds on a machine that could hold it. Thirty times slower and it finished.

## Where the design is contested

Buffer manager driven spilling takes the decision away from the operator. The operator says "I am not reading this right now" and the buffer manager decides whether that means it goes to disk. The Kuiper VLDB paper notes the failure mode: a page the operator is about to need can be evicted, and the operator has no way to say "not this one" short of pinning it, at which point it is back to managing memory itself.

Polars went the other way and put the spill policy in the sink. `polar/04-out-of-core.md` covers that. Neither is clearly right and the difference matters for us, because firepanda has no buffer manager and building one is a much bigger commitment than putting a spill path in three operators.

## What we should take from this document

The push based pipeline, cut at breakers, with chunks flowing through a chain of operators. This is the single change with the largest expected effect on the benchmarks.

Morsel driven scheduling with a work queue, rather than the fixed `parallel_for` over equal slices that every firepanda kernel does today. Our own measurements already show the problem: the optimal worker count for a low cardinality group by came out at 24 on a machine with 32 logical cores, which is exactly the signature of a fixed split where some slices cost more than others.

Not the buffer manager, not yet. `03-memory-and-spilling.md` argues for operator controlled spilling in the three operators that need it instead.
