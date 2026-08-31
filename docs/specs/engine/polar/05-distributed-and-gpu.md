# Polars beyond one machine

Both of these reuse the optimized logical plan and replace the physical layer. That is the architectural point of the document and the reason it is worth reading even though neither is in firepanda's roadmap.

## Polars Cloud, distributed

**The pipeline.** DSL, then logical plan, then optimized logical plan, then a distributed planner that produces a stage graph. A stage graph is a DAG of stages separated by shuffles, where a stage is the work that can happen without moving data between workers.

**The scheduler is bulk synchronous.** It assigns partitions to workers, waits for every worker to finish the stage, then starts the next. Simple, and it means the wall clock of a stage is its slowest worker. Their own writing says stragglers dominate, which is the known cost of bulk synchronous and the reason systems eventually move off it.

**Workers get the logical plan, not the physical plan.** Each worker receives the optimized logical plan plus its partitions and does its own physical planning, running the ordinary single node streaming engine underneath. So the distributed layer is genuinely a layer, and every improvement to the local engine improves the distributed one for free.

**Hive awareness.** Since Polars Cloud 0.10.0 the planner knows about Hive partitioning in the source, so a group by on a partition key needs no shuffle at all. Knowing the physical layout of the input lets the planner delete the most expensive stage boundary there is.

**Cost based planning** was added and reported at about ten percent average improvement. Not a large number, and worth remembering when we are tempted by an optimizer.

**Partitioned mode** returns per partition results, so `n_unique(key)` gives you a result per key value. This is a different execution mode from distributing one query, and it is the embarrassingly parallel case.

**A query profiler** with per stage metrics ships with it.

**"Diagonal scaling"** is their marketing term for choosing between a bigger machine and more machines automatically.

**Numbers**, June 2026: up to 7.7 times faster than Spark with a 3.2 times average distributed, and up to 38 times with a 6.4 times average on a single node. Public beta.

## cudf-polars, GPU

The 26.06 release replaced the Dask based executor with a streaming backend built on RapidsMPF.

**The model is CSP.** Each operation is a long lived actor coroutine. Coroutines are connected by bounded capacity channels. A full channel blocks its producer, which is backpressure by the same mechanism as the CPU engine's tokens, expressed differently. Spilling goes to host memory rather than disk. Shuffles are implemented as streaming collectives.

The convergence is the thing to notice. The CPU streaming engine, the GPU engine and DuckDB's executor all arrived at bounded units of work moving through a graph of stateful operators with backpressure at the edges. Three teams, three languages, three targets, one design.

**Numbers.** PDS-H at scale factor 1000 on a single GPU, 3.2 times CPU. PDS-DS at the same scale, 2.2 times. Scale factor 3000 on eight GPUs, 23.2 times on PDS-H and 11.0 times on PDS-DS.

A single GPU at 3.2 times a good CPU engine is a much smaller multiple than GPU marketing usually suggests, and it is the honest number for a memory bandwidth bound workload.

## What we should take from this document

The layering. Optimized logical plan as the interface, physical planning below it. If firepanda ever grows a lazy frame, keeping that boundary clean is what makes a GPU or distributed backend possible later without a second frontend.

The convergence on bounded work units with backpressure at the edges. Three independent implementations agreeing is the strongest evidence in this whole folder that `02-execution-model.md` is proposing the right thing.

The GPU number, as a corrective. firepanda has a Mojo GPU story available to it and the ceiling on a bandwidth bound dataframe workload is a small single digit multiple, not an order of magnitude. Worth doing eventually, not worth doing before the CPU engine is right.

Nothing operational. Distributed and GPU are not on the roadmap and this document does not add anything to it.
