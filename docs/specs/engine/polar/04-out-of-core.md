# Polars out of core

## The claim

A streaming engine that only bounds the memory of the data in flight is not out of core. A group by with a billion distinct keys has a hash table larger than memory no matter how small the morsels are. The three operators that accumulate state proportional to the input are group by, equi join and sort, and all three now spill.

## The memory manager

A lock free memory manager sits under the spillable operators. Operators register their memory use with it and are told when to release. There is no page level buffer manager, so no operator loses control of a buffer it is about to read. The trade is that every spillable operator has to implement its own spill policy, which is more code than DuckDB writes for the same capability.

## Group by

Partitioned, as everywhere. The partition count is chosen so that partitions are individually small, and a partition that grows past its share is written out and read back at merge time. The mechanism is the same one DuckDB uses and the difference is who decides: the operator asks the memory manager whether it should spill, rather than unpinning and finding out later.

## Equi join

Build side partitioned, partitions spilled, probe side routed to the resident partitions and spilled otherwise. Structurally the same as `duckdb/04-operators.md` describes.

## Sort

Runs built in memory, spilled, merged. The classic external merge sort.

## The multiplexer

Pull request #26774 made the multiplexer fully out of core, and it is worth understanding why a multiplexer needs to spill at all.

A multiplexer is the node that feeds one stream to several consumers. It exists because the plan is a graph and not a tree: common subplan elimination produces one node with two parents, and a self join reads the same source twice. If the two consumers run at different speeds, the multiplexer has to buffer everything the fast consumer has taken and the slow one has not. That buffer is unbounded in the worst case, and until it could spill, the whole engine had a hole in its memory guarantee that had nothing to do with any operator being large.

The general lesson is that in a graph engine the memory bound has to cover the edges as well as the nodes.

## Sources and sinks

In the out of core design, sources are iterators feeding a central task queue and sinks are consumers with their own spill policy. `sink_parquet`, `sink_csv`, `sink_ipc`, `sink_ndjson` and the partitioned variants write as they go and never hold the result. `pl.PartitionBy` and friends express a partitioned sink, and partitioned sinks are the one place where the streaming engine is already the default.

That is the shape of the canonical out of core job: scan a dataset, transform, sink partitioned, with the result never resident. It is also the shape of most ETL, which is why this is the feature that made the streaming engine worth building for their users independent of speed.

## What we should take from this document

Spill in three operators, group by, join and sort, and nothing else. Not a buffer manager. This is the cheaper design and it is the one that keeps operators in control.

A memory manager that operators ask, rather than a page cache that acts behind their backs.

The multiplexer point, as a warning to write down now and act on later: when firepanda has a graph, the buffering between nodes is part of the memory bound.

Streaming sinks as a first class feature and not a side effect of streaming execution. `write_parquet` that never materializes is a user visible capability firepanda can offer in M2 even before there is a streaming engine, because a writer that consumes chunks is much easier than an engine that produces them.
