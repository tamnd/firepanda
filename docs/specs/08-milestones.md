# Milestones

This document commits to an order, not to a calendar. There are no durations in it, deliberately: sizing a milestone in weeks invites a reader to treat the total as a delivery date, and the only useful thing to say about a project at this stage is what has to happen before what, and where the decision points are.

Where relative size matters it is stated in relative terms. M6 is the largest by volume and the smallest by risk. M4 is the hardest. M0 is the smallest and the one whose mistakes are most expensive later.

The engine track is M0 through M11 and it is what produces a defensible 1.0. There is no separate bindings track, because the bindings are M3 and they are in the middle of the engine work rather than beside it.

M0 through M3 produces something a Python user can `pip install` and form an opinion about. That is the point at which the project either proves itself or should be stopped.

Every milestone has an exit criterion you can check mechanically. "Feels done" is not an exit criterion.

## The two things that changed relative to the Go sibling spec

**There is no SIMD milestone.** In Go, vectorized kernels are an entire milestone of build tags, three implementations per kernel, runtime feature detection and a fuzz suite whose main job is catching tail bugs. In Mojo a kernel is one generic function over `DType` and `vectorize` writes the tail. That milestone dissolves into M1, which absorbs a fraction of its cost and none of its risk.

**The Python bindings moved from the end to M3.** In the Go spec they are B2, after the engine, because Go programmers already have Go. Here the audience is Python programmers who have never installed a Mojo toolchain and are not going to. The front door has to exist before there is anything impressive behind it, for two reasons: it is the cheapest possible test of demand, and the distribution problem in document 07 is genuinely risky and needs to fail early if it is going to fail.

The cost of that reordering is real. At M3 there is no lazy engine, no optimizer and no parallel execution, so the thing being shipped is an eager columnar frame with fast CSV, fast Parquet and fast group by. That is enough to be interesting and it is not enough to be impressive, and the milestone has to be framed that way internally so that a lukewarm reception is read correctly.

# The engine track

## M0, Foundations

The layer everything else sits on. Get it wrong and every later milestone pays.

Build the type lattice from document 02, `Schema`, `Field`, `LogicalType` and the coercion rules. Build `firepanda/bitmap` with get, set, popcount, the boolean operations, append, slice and a builder. Build `Buffer` with 64 byte aligned allocation and a size class pool. Build `Array[dt]`, `AnyArray` and `ChunkedArray` with construction, slicing, appending and null counting. Implement the StringView layout with its 16 byte inline prefix representation. Build the `comptime` dtype lists and the `dispatch` bridge from document 03 section 3.

**Stand up the compile time and binary size graph in CI in this milestone.** Not a threshold, just a graph, per document 03 section 6. The shape of that curve over the next year is the thing that tells us whether the monomorphization strategy holds, and starting it later means having no baseline.

Done when every dtype round trips through construction, slice, append and read. Done when the bitmap operations survive ten million fuzz cases against a `List[Bool]` reference model. Done when `as_typed` and `dispatch` have a test asserting that a dtype outside a kernel's list raises rather than compiling. Done when the CI graph exists.

The StringView representation is the risk here. Getting the short versus long discriminant wrong is silent data corruption rather than a crash. Write the fuzz test before the implementation.

## M1, Eager frame, kernels, hash table, CSV

Correctness first, and the kernels are vectorized from the first one because in this language that is not a separate decision.

Build `DataFrame` and `Series`, then select, cast, filter, sort with multiple keys and null placement, and `head`, `tail`, `slice`, `take`. Build `firepanda/hash` from document 05 section 3, because group by and join both sit on it and Mojo's `Dict` is not usable for this. Add group by with the basic aggregations, meaning sum, mean, min, max, count, first, last, std, var, median, quantile and distinct count. Add the joins: inner, left, right, outer, semi, anti and cross. Add `concat`, and the null handling functions. Write the CSV reader and writer with schema inference, and the display layer from document 04 section 9.

Write the scalar twin for every kernel and the differential fuzz harness. This is the milestone where that discipline is established or lost.

**Stand up `tamnd/firepanda-bench` here.** It can only measure CSV reading and a handful of aggregations at this point and that is fine. A benchmark harness created at the end of a project only ever proves what its author already believed.

**Differential testing against pandas starts here, not later.** Mojo can import pandas in the same process through Python interop, so the conformance harness works from M1 with no bindings in place. In the Go sibling spec this required an entire bindings milestone to exist first. Use it.

Done when the db-benchmark group by and join queries produce results identical to DuckDB at one and ten million rows, with the harness forcing materialization the way the official suite does. Done when CSV reading beats pandas `read_csv` with the pyarrow engine. Done when the basic aggregations beat pandas at ten million rows. Done when every kernel has a twin and the differential fuzz passes a hundred million cases. Done when the group by beats `Dict` based implementations by enough to have justified writing our own, which is the assumption from the MojoFrame paper and should be measured rather than trusted.

## M2, Arrow interop and Parquet

Deliberately early, because this is the milestone that makes every later gap survivable.

Implement the Arrow C Data Interface in both directions with no copying, using `abi("C")` function types, `Optional[UnsafePointer[T]]` for nullable C pointers and the external origins for foreign memory. Implement the IPC file and stream formats.

For Parquet, bind an existing reader through the C Data Interface rather than writing one. This is the decision that keeps this milestone at roughly half the size it would otherwise be. A native Mojo Parquet reader is a performance project for M8, and a bound one is a capability that unblocks everything now.

The library bound is DuckDB, not Arrow C++ as this paragraph said when it was written. Arrow C++ exports no unmangled C symbols. `libparquet.so` is a C++ library with a C++ API, so there is nothing for `dlsym` to find and nothing an `abi("C")` function type can describe, and the pyarrow package that everybody actually uses is a Cython wrapper around that C++ rather than a C shim we could borrow. DuckDB ships `duckdb.h`, a stable C API that has an Arrow C Data Interface export built into it, so a Parquet read is `SELECT * FROM read_parquet(path)` followed by handing the result chunks straight out as Arrow arrays that our existing importer already knows how to take. It costs a run time dependency we do not otherwise want, so it is a soft one: firepanda loads it with `dlopen` on first use, and a build with no libduckdb anywhere is a build where `read_parquet` raises a message saying so and nothing else changes. It also brings globbing, Hive partition discovery and predicate pushdown for free, which the Arrow C++ path would have made us write.

Add Hive partitioned dataset scanning: parse the paths, expose partition columns as virtual columns, and carry the partition metadata on the scan node even though nothing uses it until M10. Add NDJSON.

Done when every dtype round trips through pyarrow via the C Data Interface with zero copies, verified by buffer pointer identity rather than by eye. Done when TPC-H SF1 loads from Parquet. Done when a firepanda aggregation over TPC-H matches DuckDB exactly. Done when Parquet read through the binding is within 2x of Polars, which is a low bar and is the right bar for a bound reader.

From here on, anything we have not implemented can be handed to DuckDB or pyarrow without copying. That converts incompleteness from a blocker into an inconvenience and it is the highest leverage risk reduction in the plan.

## M2b, Chunked execution engine

Added after the first several releases of M1 kernel work, because the kernel work was optimizing the wrong shape. The full argument and the design are in [`engine/`](engine/00-README.md), and the short version is here.

firepanda runs one kernel over one whole column and writes a whole new column, which is pandas' execution model. At ten million int64 rows a column is eighty megabytes, so every operator boundary is a full round trip through DRAM that a chunked engine does not do at all. DuckDB never had this model and Polars replaced it, measuring three to seven times on PDS-H from the change alone.

Make `ChunkedArray` what a `DataFrame` column is, and give every kernel a chunked entry point that walks chunks and calls the existing contiguous kernel per chunk. Add a morsel queue with an atomic cursor on top of `parallel_for`, and convert every kernel that currently slices the input into equal per worker pieces. Add a node interface of three methods, a pipeline of source, operators and sink cut at breakers, and a materializing fallback node so that anything not yet ported still works. Add selection vectors so that filter, take and join output stop copying. Add a memory manager that counts, and operator controlled spilling in sort, group by and join.

This pulls the morsel scheduler forward from M5 and the spilling forward from M10, because they are one design and doing them separately means building the scheduler twice. M5 and M10 keep the wider operator coverage and the distributed pieces.

Done when every existing test passes with the engine on, when group by queries at 5GB are at least 1.5x their pre M2b selves and joins at least 2x, when a group by and a join over an input larger than the memory limit complete rather than fail, when q8 is answered, and when the worker count sweep shows no interior optimum.

Deliberately not in it: a buffer manager, a lazy frame, an optimizer, window functions, and anything distributed or GPU.

## M3, The Python front door

Everything in document 07.

The first week is not code. It is answering whether the Mojo runtime can be redistributed inside a wheel, because a negative answer changes the distribution strategy and there is no point writing four hundred bindings before knowing.

Then: the `BINDINGS` table, `PyInit_firepanda`, the PyCapsule protocol in both directions, GIL release around execution, `SIGINT` to cancellation, the error mapping table, generated `.pyi` stubs, the thin Python `__init__.py`, and `cibuildwheel` across four platform wheels plus a free threaded 3.14 build.

Then publish it. Actually publish it, to PyPI and to `modular-community`, with a README that says what it does and does not do yet.

Done when the exit criteria in document 07 section 8 all pass.

This is the cheapest possible test of whether anyone wants this, and it reaches an audience several orders of magnitude larger than the Mojo one.

## M4, Lazy engine, expressions and optimizer

The milestone that defines the product, and the most complex one.

Build the expression node types, immutable and structurally shared so that common subexpression elimination is a pointer comparison. Build `LazyFrame` with the operations from document 02. Build the logical plan, plan time type checking and validation.

Write the optimizer passes: projection pushdown, predicate pushdown, slice pushdown, common subexpression elimination, constant folding, type coercion insertion, dictionary preservation and expression fusion.

**Rewire the eager surface to build plans instead of materializing**, per document 04 section 3. This is the change that makes the naive pandas idiom fast, and it is the single highest leverage item in this milestone. It also means every M1 test now exercises the lazy path, which is the differential coverage that makes the optimizer safe.

Ship `explain()` and `profile()` here, not later.

Build the error model: poisoned nodes, did-you-mean suggestions, plan position in the message.

Done when every optimizer pass has a test asserting on `explain()` output, because the plan shape is the assertion. Done when pushdown is verified by instrumented counters for columns decoded and row groups read, not by timing. Done when every eager operation from M1 produces identical results through the new deferred path, checked by the M1 suite unchanged. Done when `__repr__` on an unmaterialized frame over a hundred million rows returns in under 200 milliseconds.

The risk is that a wrong query plan still returns a plausible looking result. The defence is that the M1 test suite becomes the differential oracle, which is only true if it was written well, which is why M1 is deliberately oversized for what it delivers.

## M5, Parallel execution

Build the morsel driven scheduler over `parallelize` with a worker pool sized to the physical core count and work stealing. Build partitioned hash aggregation where each worker owns a private table over a radix partition of the key space. Parallelize the joins and the sort. Add late materialization through selection vectors. Thread the cancellation flag through to morsel boundaries.

Keep every piece of shared mutable state in the library in `firepanda/exec/morsel.mojo`, each with its invariant. There is no race detector in this language and one named file is the substitute, per document 02.

Declare the operator interface additions that M9 and M10 need: a device affinity and a streamable flag, both ignored for now.

Determine the chunk size by benchmark in this milestone rather than guessing earlier.

Done when group by scales at least 12x on 16 cores at 100 million rows, and at better than 60 percent efficiency on 8 cores. Done when the concurrency stress suite from document 09 passes ten thousand iterations under load. Done when cancellation latency is under 50 milliseconds at 100 million rows. Done when `morsel.mojo` accounts for every atomic and every mutable global in the library, checked by a grep in CI.

## M6, pandas parity, first tier

The largest milestone by volume and the smallest by risk. It parallelizes well across contributors.

Everything marked M6 in document 06: the complete string namespace at around fifty methods including the RE2 engine, categoricals, the nested data accessors, rolling and expanding and exponentially weighted windows, the reshaping operations, the optional explicit index and everything that depends on it including `loc` and `iloc`, `rank`, `cut`, `qcut`, `factorize`, `interpolate`, `replace`, `describe`, `corr`, `cov`, `duplicated` and the rest.

Done when every M6 checkbox in document 06 is ticked with a runnable example behind each. Done when the differential test against pandas 3.0.5 passes for every one of them, with the documented divergences being the only permitted differences. Done when the full db-benchmark suite runs green at 0.5 GB and 5 GB and the results are published whatever they say.

The string namespace is where a partial implementation reads as a toy, so ship it whole. The RE2 engine is the largest single item in it and there is no Mojo regex library to start from.

## M7, Time series

Build `resample`, `asfreq` and upsampling. Build `merge_asof` with backward, forward and nearest directions, `by` grouping and tolerance, plus `merge_ordered`. Build time based rolling windows. Implement full IANA timezone support with DST aware arithmetic, `date_range` and `bdate_range`, the roughly forty pandas date offsets plus business days and holiday calendars, `at_time`, `between_time` and `infer_freq`, the period and interval types, and the sorted key fast paths.

Done when every M7 checkbox is ticked. Done when `merge_asof` matches pandas across a generated suite covering every combination of direction, tolerance and `by`. Done when the DST correctness suite passes across at least ten zones, with the explicit earliest, latest or raise policy asserted for ambiguous and nonexistent local times.

This is the milestone that earns adoption from finance and observability users specifically, because as of joins and DST correct resampling are simultaneously where pandas is most used and most error prone.

## M8, Optimization pass

The first milestone whose entire purpose is making existing things faster, scheduled here because by now there is a year of benchmark history saying where the time actually goes.

Build the compile time fused expression path from document 02, so that expressions written literally in Mojo source lower into a single loop with no intermediate buffers. Write the native Parquet reader to replace the Arrow C++ binding from M2, with projection pushdown, predicate pushdown, row group skipping and bloom filters. Do the dictionary encoding work end to end so that group bys on string columns never hash a string. Profile driven kernel specialization per document 03 section 5.

Also in this milestone: whatever the compile time and binary size graph from M0 says needs doing. If the curve has gone badly, narrowing the dtype lists happens here.

Done when the fused path is at least 2x the interpreted path on a five operation elementwise chain. Done when the native Parquet reader beats the Arrow C++ binding and matches Polars within 20 percent. Done when a group by on a dictionary encoded string column is within 30 percent of the same group by on an integer column. Done when compile time and binary size are inside their thresholds.

## M9, GPU backend

Add device affinity to the operator interface, `DeviceContext` acquisition guarded by `has_accelerator()`, device resident buffers, transfer nodes visible in `explain()`, and a cost model that refuses to move data for work that will not pay for the transfer. Port the aggregation, join, filter and sort kernels, which is mostly a launch and indexing change rather than a rewrite.

Done when the db-benchmark group by queries run on a GPU and the results are published next to cuDF's. Done when the whole library builds, tests and passes on a machine with no accelerator, with every GPU test skipped rather than failed. Done when the cost model's threshold is derived from measurement and is in a table somebody can read.

The honest framing: cuDF got here first and is mature. What we have that it does not is that a user defined function written once runs on both devices in the same language. If the benchmarks say we lose, they get published saying we lose, per document 10.

## M10, Streaming and out of core

Polars took roughly three years to get here and shipped it incrementally behind per operator fallback. Copy that approach.

Add a streamable declaration to the operator interface with transparent fallback to the in memory engine. Build spillable sinks for group by, join and sort over a memory manager with a disk budget. Build sinks for Parquet, CSV, IPC and NDJSON that never materialize the whole result. Add partition aware planning: prune Hive partitions, pre partition group bys and joins on partition keys, and rewrite inner joins as unions of partition filtered joins.

Done when db-benchmark at 50 GB completes within a memory budget smaller than the dataset, and when every operator either streams or falls back with the choice visible in `explain()`.

Highest effort, least parity value. Ship 1.0 without it if anything is pressing, since nothing earlier depends on it.

## M11, SQL, remaining IO, and 1.0

Build `firepanda.sql()`, parsing into the same logical plan so the optimizer is shared, plus `query()` and `eval()` for pandas parity, and an ADBC driver. Add the remaining IO formats marked M11 in document 06: SQL, ORC, fixed width, Excel, HTML, XML and clipboard.

Then the API review and freeze, the complete docstring coverage with runnable examples, and the migration guide for pandas users organized by pandas function name.

Done when all 22 TPC-H queries run correctly at SF10 with published timings against DuckDB, Polars and MojoFrame. Done when the public API is frozen and every exported symbol is documented. Done when every checkbox in document 06 outside the explicitly post 1.0 sections is ticked.

# Dependencies

```
M0 -> M1 -> M2 -> M3 (publish)
             |
             +--> M4 -> M5 -> M8
                   |     |     |
                   |     +-----+---> M9
                   |     |
                   |     +---------> M10
                   |
                   +--> M6 -> M7 -----------> M11

firepanda-bench: continuous from M1
pandas differential testing: continuous from M1
```

M6 and M7 are surface area. M8, M9 and M10 are performance. They are independent of each other after M5, they need different skills, and with more than one contributor that is the natural split.

# Four points to stop and reassess

**After M2: does the Arrow bridge work well enough that being incomplete is survivable?** If handing a frame to DuckDB is not genuinely zero copy and genuinely easy, the whole plan is built on a false assumption and every later milestone gets harder rather than easier.

**After M3: does `pip install firepanda` work, and does anyone do it?** Two questions and both are gates. If the wheel cannot be made self contained, the project has a distribution problem that no amount of engine work fixes. If it works and nobody installs it, that is the cheapest signal about demand this project will ever get, and it costs four milestones to obtain rather than twelve.

**After M5: is the engine within 4x of Polars on db-benchmark?** By here the kernels are vectorized, the executor is parallel and the optimizer exists. If the answer is no, the problem is architectural, and M8 will not rescue it. The numbers to answer it with are already in `firepanda-bench` because it has been running since M1.

**After M6: is anyone actually using it?** Parity is expensive, and without users M7 through M11 is speculative work on a library nobody has adopted.

# Ordering principles

Correctness before speed. The scalar twins from M1 are the specification that everything after is checked against, and reversing that order leaves nothing to check against.

Interop early. M2 comes before the lazy engine because it makes every subsequent gap survivable.

**Distribution before capability.** M3 is early because the risky part of shipping a Mojo library to Python users is the shipping, not the library, and finding that out at M9 would be fatal.

Benchmarks and pandas differential testing from M1, continuously, so a regression is attributable to a commit instead of being discovered at M8.

Measure the thing most likely to be wrong from the first week. That is compile time and binary size under monomorphization, and the graph starts at M0.

Surface area parallelizes and engine work does not. Schedule contributors accordingly.
