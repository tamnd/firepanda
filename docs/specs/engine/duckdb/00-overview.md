# DuckDB, the shape of the whole thing

## What it is

An analytical database that runs inside the calling process. No server, no network hop, one library. SQLite's deployment model with a column store and a vectorized engine behind it.

Current versions in August 2026 are 1.4 LTS, which is supported until September 2026, and 1.5.5 from July 2026. Version 2.0 is expected in the autumn.

## The five parts

DuckDB is usually described as three pillars, which are columnar storage, vectorized execution and morsel driven parallelism. That is a good summary and it leaves out two parts that matter as much.

**Storage.** A single file, block structured, holding row groups of 122,880 rows. Inside a row group each column is stored as its own chain of compressed segments with min and max statistics attached. Document 02.

**The type and vector system.** A logical type lattice on top of a small set of physical types, and a vector abstraction that can be flat, constant, dictionary encoded or an arithmetic sequence, so that compression survives into execution rather than being undone at the scan. Document 01.

**Execution.** Push based pipelines of operators, each pipeline made of a source, a chain of stateless operators, and a sink. Work is chopped into morsels and handed to threads by a scheduler. Document 03.

**Operators.** Radix partitioned hash aggregation and hash join, both of which spill to disk when they do not fit, plus external sort, top N and window. Document 04.

**The optimizer.** A sequence of logical rewrites, then a join order search using dynamic programming over the join hypergraph, then a physical plan. Document 05.

The pillars story is about the middle three. The reason DuckDB wins on a laptop with sixteen gigabytes of memory and a fifty gigabyte input is the first and the fourth, and that is the part a dataframe library normally has no answer to at all.

## The buffer manager is the thing that ties it together

Almost everything in DuckDB that touches more memory than it has goes through one buffer manager. The storage layer pins and unpins blocks through it. The hash join materializes its build side into blocks it can unpin. The hash aggregate unpins pages when a thread's local table fills. External sort writes runs through it.

That means spilling is not a special mode that operators opt into, it is what happens when a page is unpinned and memory is tight. Operators are written to unpin what they are not currently reading and to be able to recompute pointers when a page comes back, and then the buffer manager decides.

There is a real cost to that design, and DuckDB's own people have written about it. The Kuiper VLDB paper on the hash join notes that fine grained buffer manager driven spilling takes the decision away from the operator, so an arbitrary page can be evicted just before the operator needed it, unless the operator pins it. The alternative, which is operator controlled spilling, is what Polars chose. Neither is obviously right.

## What DuckDB is not

It is not lazy in the dataframe sense. There is no user visible plan object that you build up and collect later. You hand it SQL or a relational API call and it runs.

It is not distributed. There is one process. MotherDuck sells the hosted and hybrid version and Quack, which landed in 1.5.3, turns DuckDB into a client server database, but the engine itself is single node.

It is not incremental. There is no streaming ingestion model and no materialized view maintenance.

## Why we care

Two reasons, and they are different.

The first is that DuckDB is the number we are trying to beat. On db-benchmark at ten million rows it is between five and eight times faster than firepanda on joins and about five times faster on the widest group by. Everything in `04-operators.md` is an answer to why.

The second is that DuckDB has solved a problem we have not thought about at all, which is what happens when the data does not fit. Every firepanda kernel today allocates an output as long as its input and assumes that works. Documents 03 and 04 describe an engine where that assumption is never made, and `03-memory-and-spilling.md` in the parent folder is where we decide how much of that to take.
