# What we take, and what we do not

The two rival documents are long. This one is the summary of the argument, and it is the document to read if you only read one.

## Where firepanda actually is

Twenty thousand lines of Mojo across ten packages. The kernels are good. On db-benchmark at ten million rows we are roughly two times pandas across the board, which was the goal we set, and we hit it by writing careful SIMD kernels with good hash tables.

Against the engines we are not close.

```
                             firepanda   pandas   polars   duckdb
q6   group by two keys           0.727    1.293    0.346    0.493
q7   range by key                0.120    0.298    0.126    0.051
q10  group by six keys           1.068    3.438    0.736    0.216
j1   10M to 10K                  0.155    0.301    0.027    0.017
j2   10M to 100K                 0.191    0.350    0.029    0.021
j3   10M to 100K, left           0.190    0.347    0.041    0.015
j4   10M to 10M                  0.734    1.428    0.209    0.097
j5   10M to 10M, aggregated      0.719    1.391    0.237    0.089
```

Seconds, 0.5GB db-benchmark, gamingpc, August 2026.

The joins are the story. j1 has us nine times slower than duckdb and six times slower than polars, on a query where our join kernel is a good join kernel. That gap is not a kernel gap.

## The diagnosis

firepanda has pandas' execution model. One kernel runs over one whole column and writes a whole new column. Then the next kernel reads it back.

At ten million int64 rows a column is eighty megabytes. Nothing about that fits in any cache. So a five step query does five write passes and five read passes over eighty megabytes of DRAM that a chunked engine does not do at all. The kernel can be perfect and the memory traffic is still there.

Both rivals abandoned this model. DuckDB never had it. Polars had it, built a streaming engine to replace it, and measures three to seven times on their own benchmark from the change alone.

This is why kernel level optimization has been giving diminishing returns. PR #75 was a real improvement, correctly measured, and it moved j4 by twenty one percent. Twenty more of those does not close a factor of six.

## What we take

**Chunked columns as the thing kernels operate on.** `firepanda/array/chunked.mojo` exists and almost nothing uses it. Everything downstream needs this first.

**Push based pipelines cut at breakers.** A source, a chain of operators that transform a chunk into a chunk, a sink. From `duckdb/03-execution.md`.

**Morsels of about 128 thousand rows, handed out by a scheduler, not divided up front.** DuckDB says 122,880 and Polars says 128 thousand, which is agreement. Our `firepanda/exec/parallel.mojo` divides the input into equal slices per worker, and our own measurement that 24 workers beat 32 on a 32 thread machine is the signature of that being wrong.

**A morsel sequence number so that ordering is opt in.** From `polar/03-streaming-engine.md`.

**A small node interface, three methods, with an in memory fallback node.** This is the thing that makes the migration incremental. From `polar/02-lazy-ir.md`. Without it, this is a rewrite that never lands.

**A selection vector layer, so filter and take do not copy.** From `duckdb/01-data-model.md`.

**Operator controlled spilling in three operators, group by, join and sort.** From `polar/04-out-of-core.md`, and specifically not DuckDB's buffer manager.

**Two cheap things we should do immediately, ahead of all of the above.** A salt byte beside the pointer in our hash table, and build side selection by size rather than by argument order.

## What we do not take

**A buffer manager.** It is a large commitment and it takes control away from the operators. Both rival documents say the design is contested and Polars chose the other side.

**A storage engine.** No file format, no MVCC, no compression. firepanda reads and writes other people's formats. This was already the position and nothing in the research changes it.

**An optimizer with join reordering.** Polars measured about ten percent from cost based planning. DPhyp is a real algorithm and it belongs in the lazy frame milestone, not this one.

**VARIANT and GEOMETRY.** Out of scope for a dataframe library.

**Distributed and GPU.** The honest single GPU number on a bandwidth bound workload is about three times, and that is with a mature engine underneath. Not before the CPU engine is right.

## The order

1. The two cheap hash and join fixes, now, as ordinary pull requests.
2. M2 as planned, which is Arrow interchange and the readers. A reader that takes a column list is projection pushdown without a plan, and readers are the thing that produces chunks naturally, so M2 makes the engine work easier rather than competing with it.
3. M2b, the engine, described in `02-execution-model.md` through `05-m2b.md`.

M2 first is deliberate. A streaming engine with no streaming source to feed it is a benchmark harness. The Parquet reader is the first real source, and Polars' own release history says sinks and sources first for the same reason.
