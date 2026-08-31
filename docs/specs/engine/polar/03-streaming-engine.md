# The Polars streaming engine

This is the document to read alongside `duckdb/03-execution.md`. Two engines built for the same reason with different mechanisms.

## What "streaming" means here

Not streaming in the Kafka sense. There is no continuous ingestion and no incremental view maintenance. It means the query runs on bounded memory over an unbounded input, by moving small pieces of data through the operator graph instead of materializing whole columns between operators.

Polars had an earlier thing also called the streaming engine, which was a different and more limited design. That one is gone. When their docs say streaming engine now they mean the one described here, present since 1.31.1.

## The core idea, in one paragraph

Morsel driven parallelism, as in HyPer and DuckDB, combined with Rust async state machines. Work is chopped into morsels. Workers pull morsels from a scheduler. Each operator is compiled into an async state machine that can suspend at any await point, so an operator that cannot accept more input right now simply does not resume, and that is the backpressure mechanism. Their own description is a hybrid push pull model: pull at the scheduler boundary, push within a pipeline.

## Morsels

A `Morsel` is a small DataFrame plus a `MorselSeq`, which is a sequence number.

Typical size is 128 thousand rows. Compare DuckDB's row group at 122,880. Two teams landed within five percent of each other from opposite directions, which says the number is a property of the hardware rather than of either design.

The sequence number is how order is preserved without serializing execution. Operators that must produce ordered output use it to reassemble, through `OrderedUnion`. Operators that do not care ignore it. That means a query on an unordered result gets full parallelism and a query that needs order pays only for the reassembly.

## The compute node

Every operator implements `ComputeNode`, which is three methods.

`update_state()` is called between phases. The node is told which of its input and output ports are still live and says what it wants. This is where a node declares it is done, or that it no longer needs a particular input, and it is how the graph learns that a `head(10)` upstream has been satisfied so the scan can stop.

`spawn()` is given a `TaskScope` and a set of morsel receivers and senders, and spawns the async tasks that do the work. Multiple tasks per node is normal, one per worker.

`get_output()` collects the result for the nodes that produce one.

The graph is built by `to_graph_rec()` in `to_graph.rs`, which walks the `PhysNode` graph from document 02 and instantiates a `ComputeNode` per node.

## Backpressure

This is the part that differs most from DuckDB.

Each morsel carries a token, a `WaitToken` or a `SemaphorePermit`, held for as long as the morsel is in flight. A source cannot produce a new morsel until a token is available. So the number of morsels alive in the system is bounded by construction, and a slow sink stops the source without any explicit rate limiting.

Their design notes make a point about why the scheduler owns the queue rather than the operators pushing directly. If a sink is flushing a partition to disk, it needs to refuse work for a moment. In a pure push model there is nowhere for the refused morsel to go. With the scheduler holding the queue, the operator just does not take one, and the morsel waits.

That is the operator controlled spilling design, and it is the concrete reason Polars did not build a buffer manager. The operator knows it is spilling, so the operator says when it can take work. In DuckDB the buffer manager knows and the operator does not, which is the failure mode the Kuiper paper describes.

## Visualization

`visualize_plan()` emits DOT. Fallback nodes, meaning the subtrees that dropped back to the in memory engine, are red. Memory intensive nodes are yellow.

Shipping the tool that shows users which parts of their query are not streaming, in red, is a good instinct and firepanda should have the equivalent when it has a graph to draw.

## Measured effect

Polars' own PDS-H numbers put the streaming engine at three to seven times the in memory engine, with the multiple growing as the data grows. It is roughly at parity on small inputs and pulls away on large ones, which is exactly what you expect from removing per operator materialization: the cost you remove is proportional to the data, and the cost you add, which is scheduling, is not.

That shape is worth holding onto because it predicts what firepanda would see. At the db-benchmark 0.5GB scale we should expect a modest gain. The gain at 5GB and 50GB is the point.

## What we should take from this document

The morsel with a sequence number, so that order is opt in rather than a constraint on the scheduler.

The token per morsel as the backpressure mechanism, because it bounds memory by construction instead of by a limit that has to be checked.

Operator controlled spilling, with the scheduler holding the queue so an operator can decline work while it flushes.

A three method node interface, since it is small enough that porting one kernel to it is an afternoon rather than a project.

128 thousand rows as the morsel size, corroborated by DuckDB's 122,880.
