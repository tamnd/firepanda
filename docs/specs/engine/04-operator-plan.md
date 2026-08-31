# Operator plan

Which operators get a chunked implementation, in what order, and what each one is expected to buy. The order is by benefit divided by cost, not by how the code is laid out.

Every operator starts life as a `Materialize` fallback from `02-execution-model.md`, so this is a list of fallbacks to remove, and each entry is one pull request.

## Before any of it: two cheap wins

These do not need the engine and should ship as ordinary pull requests now.

**Salt bytes in the hash table.** Store one or two bytes of the hash beside the pointer in `firepanda/hash/table.mojo`. A probe compares the salt first and only follows the pointer on a match, so a miss costs one cache line instead of two. From `duckdb/04-operators.md`. Expected to show on q3 and q10, the high cardinality group bys, where probe misses dominate.

**Build side by size.** `firepanda/join/pairs.mojo` builds from the right side because the argument is called right. DuckDB has a whole optimizer pass for this, `BUILD_SIDE_PROBE_SIDE`, separate from join ordering. Compare the two lengths, build from the smaller, flip the output pair order to compensate. Does nothing for j4 and j5, which are symmetric, and is a large win the first time a user writes the tables in the unhelpful order.

## Tier one, elementwise

`filter`, `select`, arithmetic, comparison, cast, `with_column`.

These are the easy ones. The value at row i depends only on row i, so a chunked implementation is a loop over chunks calling the existing kernel, with no state carried between them. This is Polars' `is_elementwise_rec_cached()` analysis and the answer is statically obvious for all of these.

Cheap, and they are what makes a pipeline have more than one stage in it. Until several of these are chunked there is nothing to chain.

`filter` is the one that also wants a selection vector rather than a copy, but do the chunked version first and the selection vector after, as two changes.

## Tier two, the group by

The one that matters most for the benchmark numbers.

Chunked group by means: each worker takes morsels, accumulates into a thread local partitioned table, and the partitions merge at the end. That is the phase one and phase two structure from `duckdb/04-operators.md`, and it replaces the current design where `firepanda/hash/grouping.mojo` factorizes the whole key column and then `firepanda/kernel/group.mojo` aggregates by code.

Expected effect: this is where the cache argument bites hardest, because today the codes array is a full extra eighty megabyte column written and read. Removing it is a real pass removed.

Do the in memory version first and leave spilling to `03-memory-and-spilling.md`.

The sorted group by is worth a note. Polars has `SortedGroupBy`, chosen when the key is known sorted, and it needs no hash table at all. We have no sortedness flag on a column. Adding one, set by sort and by readers that know, is cheap and it makes the group by after a sort free.

## Tier three, the join

Where the largest gap is. j1 has us at nine times DuckDB.

Chunked join means the build side consumes chunks into a partitioned table, and the probe side is a pipeline stage that emits matched chunks as it goes rather than building whole index arrays first.

The current profile of j4, measured on gamingpc, is: frame_join 635 ms, of which join_indices 590, of which bucket build 384 and `group_ordinals` 395 and emit 300, then concat 8 and take 30. `group_ordinals` at 395 milliseconds inside a 635 millisecond query is the thing to attack, and it exists because we factorize the key into codes as a separate whole column pass before joining.

Two things fold in here:

**PR #75 generalized.** We already detect a unique build side and build a direct code to row table. DuckDB's perfect hash join detects a dense integer key and skips the hash entirely, decided at runtime after min and max are known. Ours still hashes to get the code and then indexes. Detecting a dense integer key at the join and skipping `factorize` altogether removes most of `group_ordinals` for the case that is most joins.

**Merge join.** Two sorted inputs join with no hash table. Polars shipped it in 1.38. Needs the sortedness flag from tier two.

## Tier four, the holes

Things firepanda cannot do at all, which are correctness items rather than performance ones.

**Top n per group.** db-benchmark q8 asks for the two largest values of a column per group and we have no answer. DuckDB makes it a separate operator with a bounded heap, not a sort followed by a limit. The place to build it is beside `group_ordinals` in `firepanda/kernel/group.mojo`, reusing `_slab_bounds` and `_fill_slab`.

**Sort as a pipeline breaker.** The existing sort works. It needs to consume chunks and emit chunks so it can sit in a pipeline.

**Window functions.** Nothing started. Partitioned and ordered, with special cases for the common frames. Not M2b.

## Tier five, the sinks

From `polar/06-whats-new.md`, the observation that Polars did sinks before joins. Their reason was that streaming ETL is a workload that cannot be done any other way, whereas a slow join is merely slow.

A writer that consumes chunks and never holds the whole result is much easier than an engine that produces them, and it can be built during M2 alongside the readers. `write_parquet` on a frame larger than memory is a user visible capability we could have before the engine exists.

## Definition of done for M2b

The `Materialize` fallback is still present and still correct, and these are no longer using it: filter, select, arithmetic, comparison, cast, group by, inner and left join, sort.

Measured on gamingpc at 0.5GB and at 5GB, with the 5GB numbers being the ones that matter, since `polar/03-streaming-engine.md` says the multiple grows with size and 0.5GB is small enough that it may show almost nothing.
