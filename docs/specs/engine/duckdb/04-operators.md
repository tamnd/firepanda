# DuckDB operators

The two that decide the benchmarks are hash aggregation and hash join. Both are radix partitioned, both are two phase, and both spill.

## Hash aggregation

**Phase one, thread local.** Every thread builds its own hash table over the morsels it gets. Nothing is shared, so nothing is contended.

**Phase two, partition wise merge.** Groups are radix partitioned on the group hash. Two groups with different hashes cannot be the same group, so a partition can be merged by one thread with no coordination at all. Each thread takes whole partitions and combines every thread's contribution to them.

The important detail is that this is not thirty two hash tables per thread. The older design had each thread build one table per partition, which meant thread count times partition count allocations and did not scale. The current design is one hash table per thread that is internally partitioned, so the partitioning is a property of the layout rather than of the number of objects.

**Layout.** Two parts. A pointer array, and payload blocks holding the group values and the aggregate states. Resizing rebuilds the pointer array and leaves the payload alone, which is what makes growth cheap.

**Probing.** Linear probing rather than chaining. Collisions walk forward in an array, which the prefetcher handles, instead of following pointers into unrelated cache lines.

**Salt.** One or two bytes of the hash are stored beside the pointer in the pointer array. A probe compares the salt first and only dereferences into the payload block when the salt matches, so a miss costs one cache line instead of two.

**When it starts partitioning.** A thread does not partition until its table passes roughly ten thousand entries. Below that the partitioning is pure overhead and a plain table wins.

**Spilling.** Morsels of about a hundred thousand rows go into a thread local table of fixed size. When it fills, the thread unpins the pages and starts a fresh table, and the buffer manager can now write those pages out. In phase two, many more partitions than threads are created on purpose, so that only a few partitions need to be resident at a time. With eight threads and thirty two partitions, eight are in memory.

Compare that to what firepanda does now. `firepanda/hash/factorize.mojo` builds one global table, or one direct addressed table, over the whole column, and `grouping.mojo` packs multi key groups into an integer and factorizes that. There is no partitioning, no thread local phase, and no spilling. It is fast because the tables are good and because it is Mojo, and it has no answer at all above the memory limit.

There is a 2025 paper, "Global Hash Tables Strike Back", arguing that a well built global concurrent hash table beats the partitioned two phase design in a good number of cases. Worth reading before we commit to partitioning, because our current global table is not obviously the wrong choice.

## Hash join

`PhysicalHashJoin` over `JoinHashTable`. Build side is materialized into a row layout in blocks that can be unpinned.

**Build.** Each thread materializes its share of the build side into thread local blocks. If the total stays under a threshold, the thread local tables merge into one global table and it is an ordinary in memory hash join. If not, each thread radix partitions its own data, and then hash tables are built over as many partitions as the memory limit allows.

**Probe, external.** The radix bits of a probe row's hash say which partition it belongs to. If that partition's table is currently resident, probe it now. If not, accumulate the row and spill it. When the resident partitions are done, build the next set and replay the spilled probe rows against them. Repeat.

**Probe side partitioning.** A later change partitions the probe side rather than rescanning it, which is what makes external right, outer, mark and anti joins possible, since those need to know which build rows were never matched. If probing finishes in two rounds the probe side is not partitioned at all.

**Recursion.** If a partition still does not fit after partitioning, the radix bit count goes up and it partitions again. The known failure mode is heavy key skew: if a single key is a large fraction of the input, all its rows have the same hash and no number of radix bits separates them, so the loop spins.

**Perfect hash join.** When the build side key is a dense integer range, there is no hash table at all, just an array indexed by key minus minimum. Originally decided at planning time, now decided at runtime, because the build already computes min and max, so it triggers far more often. The cost is that it can no longer be shown in the printed plan, since the decision happens after the plan is formatted.

That last one is worth dwelling on. firepanda merged exactly this optimization in PR #75 two days ago, arriving at it from the other end: we noticed that most joins are onto a unique key and built a single table from code to row, abandoning it on the first duplicate. DuckDB's version keys on density rather than uniqueness and skips the hash entirely. Ours still hashes to get the code and then indexes. Combining the two, which is to say detecting a dense integer key at the join and skipping `factorize` altogether, is on the list in `04-operator-plan.md`.

## Sort

External merge sort through the buffer manager. Runs are built in memory, spilled, and merged. Sorting is also how DuckDB implements some window functions and ordered aggregates.

firepanda's `firepanda/kernel/sort.mojo` is an in memory sort with no spill path, which is the correct scope for now.

## Top N

A separate operator rather than a sort followed by a limit. It keeps a bounded heap, so the memory is the limit rather than the input. This is db-benchmark q8, which asks for the largest two values of one column per group, and it is one of the two queries firepanda cannot answer at all.

## Window

Partitioned and sorted, with specialized paths for the common frame shapes. Not something firepanda has started.

## What we should take from this document

Radix partitioning on the hash as the way to get a parallel aggregate and join that can also spill, with the caveat that the global table paper says to measure before assuming.

The salt byte in the pointer array. That is a small, local change to our existing hash table and it should show up on q3 and q10, which are the high cardinality group bys where a probe miss costs a cache line.

The runtime decision to skip hashing entirely for a dense integer key, which generalizes the unique build side path we already have.

A top N per group kernel, because it is a real hole and not a performance question.
