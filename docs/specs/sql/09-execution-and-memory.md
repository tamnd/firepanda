# Execution and memory

Optimized plan in, pipelines out, frames back. Most of what this needs already exists: `firepanda/exec/` has the chunk, the morsel queue, the node interface and the driver, and the whole point of document 02's layering is that SQL does not get to build a second engine beside it.

The second half of the document is memory, which is the part of the brief that is easiest to agree with and hardest to hold, because memory is not a feature you add. It is a property that every operator either preserves or destroys, and one materializing operator undoes the discipline of ten streaming ones.

## 1. What is already there

`firepanda/exec/chunk.mojo` gives 128K rows, one array per column, no schema and no names on the chunk because the plan already knows them, owned outright with no sharing and no reference counting. `firepanda/exec/morsel.mojo` is a single counter morsel queue at the same 128K, one atomic per morsel rather than per row, with no work stealing because the work is a range. `firepanda/exec/node.mojo` has the three method interface of `update_state`, `process` and `finish`, a closed union rather than trait objects because Mojo 1.0 cannot hold a `List` of trait objects, dispatched by a chain of type tests at one branch per chunk.

The existing node set is `Filter`, `Project`, `Compute`, `Cast`, `Group`, `Reduce` and a `Materialize` fallback. The interface already earns its keep: a `Limit` that says `FINISHED` in `update_state` makes everything upstream useless and the driver stops reading the source, which is limit pushdown falling out of the interface rather than out of an optimizer pass.

## 2. What SQL adds

Physical operators, all of which the dataframe surface also needs, all of which go in `firepanda/exec/`:

| operator | note |
| --- | --- |
| `HashJoin` | build and probe, partitioned, algorithms exist in `firepanda/join/` |
| `NestedLoopJoin` | non equi conditions, `ASOF`, `LATERAL` fallback |
| `Sort` | a breaker, multi key, with the null placement from document 06 |
| `TopN` | `ORDER BY ... LIMIT` without a full sort, a bounded heap |
| `Limit` | already implied by `update_state`, made explicit |
| `Window` | partitioned, with the three frame strategies from document 07 |
| `Distinct` | hash based, plus the `DISTINCT ON` variant |
| `SetOp` | union, except and intersect, plus the by name variant |
| `Unnest` | row multiplying |
| `RecursiveCTE` | fixed point driver |

Every one of them is a node under the existing interface. None of them is reachable only from SQL.

## 3. Pipelines and breakers

The plan is cut into pipelines at the breakers. A pipeline is a source, a line of non blocking operators, and a sink. `Filter`, `Project`, `Compute`, `Cast` and `Unnest` are non blocking. `Sort`, the hash join's build side, `Group`, `Distinct` and `Window` are breakers, meaning they consume a pipeline entirely and start a new one.

Dependencies between pipelines form a DAG, because a join's build pipeline must complete before its probe pipeline starts, and the driver executes it in dependency order. Independent pipelines can run concurrently, which on a query with several build sides is real parallelism that morsel level parallelism alone does not provide.

Within a pipeline the model is morsel driven parallelism: N workers each pull a morsel from the source and push it the whole length of the pipeline. A chunk stays in one core's L2 from source to sink, which is the argument in `chunk.mojo`'s own header and the reason a chunked engine beats a whole column one.

The gap today is the parallel breaker. `pipeline.mojo` has a parallel prefix for the elementwise family and a single threaded tail. A parallel `Group` needs per worker partial hash tables and a `combine`, which is exactly the aggregate protocol from document 07 and is why `combine` is mandatory at registration. A parallel `Sort` needs per worker runs and a merge. Both are on the critical path for TPC-H at SF10 and both are engine work rather than SQL work, and document 12 sequences them.

## 4. The expression executor, and a design decision to make deliberately

firepanda's current model lowers an expression tree into a line of nodes. `(a + b) < c` is one `Compute` that appends `a + b`, a second that compares against `c`, and a `Project` that drops the intermediate. The comment in `node.mojo` is explicit that this is why `Compute` appends rather than replaces and why `Filter` names its mask by position.

That model is clean and it has a cost that grows with SQL. Every intermediate is a materialized column of 128K values with its own allocation, and a TPC-H expression like q1's four arithmetic terms inside four aggregates produces a lot of intermediates that exist for one node's lifetime.

DuckDB's alternative is an expression executor, an interpreter over the expression tree that evaluates into a small pool of reusable intermediate vectors, so the intermediates are recycled rather than allocated and a chain evaluates without a node boundary between each step.

The position for 1.0 is to keep the node chain lowering, add a per pipeline intermediate vector pool so the allocations are recycled, and monomorphize fused chains where the shape is known at compile time. Mojo is ahead of time, so there is no JIT to compile a chain at query time, but a `comptime` parameter is a compile time fusion mechanism: the common shapes, compare and filter, or arithmetic then aggregate, can be monomorphized into single kernels at build time and pattern matched by the physical planner. That is a smaller change than an interpreter and it captures most of the win on the shapes that dominate the benchmarks.

If measurement shows the residual intermediate traffic dominating, the executor is a bounded follow up that does not change the plan or the node interface. Document 13 holds it as an open question with a decision criterion rather than a preference.

Two things the executor must do regardless of which model wins. Selection vectors rather than compaction, so that a filter produces a list of surviving indices and downstream operators respect it, and a one per cent selective filter does not copy ninety nine per cent of the data to remove it. And constant and dictionary vectors, so that a literal operand is one value rather than 128K copies and a low cardinality string column computes on the dictionary rather than on the rows, which is what makes string comparison cheap and is worth more on real data than any arithmetic optimization.

## 5. Adaptivity

Two run time adaptations, both cheap, both from DuckDB.

**Conjunct reordering.** A `WHERE a AND b AND c` evaluates conjuncts in the order the optimizer guessed. The executor tracks selectivity and cost per conjunct over the first few chunks and reorders. This catches the case the optimizer cannot know, which is a predicate whose selectivity depends on data the statistics did not describe.

**Join build side correction** is not done at run time, because by the time you know, you have built. It stays an optimizer decision, informed by the exact row counts we have and a database does not.

## 6. Strings

Called out because document 06 makes it a real cost. SQL string indexing is by character rather than by byte, and firepanda's string kernels are byte oriented.

The resolution is to keep the byte oriented representation and make the character indexed operations, meaning `substring`, `length`, `strpos` and slicing, take an ASCII fast path with a UTF-8 scan fallback, decided per chunk by a vectorized check for whether the chunk is all ASCII that costs one pass over the bytes with SIMD. Real data is overwhelmingly ASCII in the columns people slice, and the check is cheaper than the scan it avoids. `length` on an all ASCII chunk is the byte length.

Short string inlining, meaning a twelve to sixteen byte prefix stored inline with the pointer, which is DuckDB's and Velox's representation, makes comparison and equality resolve without chasing a pointer in the common case. It is a storage layer change rather than a SQL one, it benefits every existing kernel, and it is worth its own issue.

## 7. The output

A physical plan's sink builds a `DataFrame`. Not an Arrow buffer to be converted, not a result set to be fetched, but the same frame type the eager API returns, with the same column arrays, handed back with no copy.

That is the sentence document 01's latency axis rests on, and it is worth checking that nothing in this document breaks it. The last operator writes chunks into the columns of the result frame, and `sql()` returns it. There is no serialization step to remove because there was never one to add.

## 8. The memory targets, restated as tests

From document 01, with the instrument attached.

**Peak RSS within 1.2x of DuckDB on every TPC-H query at SF10.** Measured by the harness, sampling RSS at 10 ms and taking the maximum, per query, in a fresh process. Reported as a column beside every timing, never as a footnote.

**No query in the suite fails for memory that DuckDB completes.** A query that does not fit gets slower. DuckDB's external aggregation completes a 50 GB input on a 16 GB machine roughly thirty times slower than in memory, and finishing thirty times slower is a different category from raising, and it is the category we have to be in.

**Working memory bounded by operator, not by hope.** The number of live chunks in a pipeline is bounded structurally, and every breaker declares its memory behaviour at construction.

## 9. Where the memory actually goes

Six places, and only three of them are large.

**Pipeline intermediates.** 128K rows times columns times workers. At ten columns of eight bytes and ten workers that is about 100 MB live. Bounded by construction, and this is the number that the intermediate vector pool in section 4 keeps from multiplying.

**Hash tables.** The build side of every join plus every `Group`. This is the largest single consumer on TPC-H and it is data dependent, because a group by on a high cardinality key holds one state per group and there is no bound on the number of groups.

**Sort buffers.** The whole input, by definition, unless the sort is external or is a `TopN`.

**The input frames.** Ours, already in memory, not counted against the query, but counted against the process, which is why the harness reports absolute RSS rather than a delta.

**The result.** Handed back to the user, so it lives as long as they hold it.

**The front end.** Tokens, parse tree, AST, plan. Microseconds of work and kilobytes of memory, and it is on the latency path rather than the memory path. It gets an arena that is reset per statement rather than freed.

## 10. Bounded by construction

The strongest memory guarantee is the one that does not require a check.

**Backpressure.** Polars' streaming engine bounds live morsels with a token: a worker takes a token to start a morsel and returns it at the sink, so the number of chunks in flight is the token count regardless of how fast the source can produce. firepanda's morsel queue already has the shape, since a worker pulls one morsel at a time, and the addition is that a sink which cannot accept must stop the pull rather than buffer. Without this, a slow sink behind a fast scan buffers the whole input, which is the classic streaming engine memory bug.

**Folding aggregates rather than holding rows.** `node.mojo` already makes this distinction, because `Group` holds one row per group and merges each chunk as it passes rather than holding every row until the last one. A sum of sums is a sum and a median of medians is not, so the foldable set is a list rather than a judgement and the rest goes through the materializing fallback. Extending that list is directly extending the set of queries that run in bounded memory, which makes it the highest value memory work there is.

**`TopN` instead of sort then limit.** `ORDER BY x LIMIT 10` over a billion rows should hold ten rows and not a billion. This is an optimizer rewrite from document 08 and a physical operator from section 2, and it converts the single most common unbounded operator into a bounded one.

**Late materialization.** Sorting one key column plus a row id instead of fifty columns is a memory win before it is a speed win.

## 11. Spilling

For the operators that genuinely cannot be bounded, the answer is disk, and the design decision is where the policy lives.

**Operator controlled, not allocator controlled.** An allocator that fails at a limit produces an out of memory error at an arbitrary point with no way to recover. An operator that knows it is a hash aggregate over a partitioned table can decide to write partition 7 to disk and carry on. DuckDB moved from the former to the latter and it is the difference between a limit that raises and a limit that degrades.

**A memory manager with reservations.** Each operator reserves before it grows and is told to shrink when the budget is tight. The budget is a session setting, `memory_limit`, defaulting to a fraction of physical RAM. DuckDB's default on this machine is 19.1 GiB of a 24 GiB system, so about eighty per cent.

**Radix partitioning as the spilling mechanism.** A hash aggregate or hash join partitioned by the high bits of the key hash can evict whole partitions independently, and rebuild processes them one at a time. This is why the partitioning is a design requirement rather than only a parallelism device: it is what makes spilling possible at all without a full re-read.

**External sort** is runs plus a k-way merge, which is standard, and the run size is the budget.

**Fixed size, relocatable aggregate state** is what makes spilling cheap, because writing a partition is writing bytes. Document 07 makes this a registration time property, and an aggregate whose state is a heap allocated list, such as `list`, `string_agg` or exact `quantile`, cannot spill and must say so. Those get the honest answer: they can raise, and the error names the aggregate and suggests the approximate variant.

## 12. The allocator

Three rules, all of which are about the interactive path rather than the analytical one.

**Arena per statement for the front end.** Tokens, parse nodes, AST nodes and plan nodes all come from one arena that is reset rather than freed. This is what makes document 01's 3 us small statement target reachable, because a per node malloc makes it unreachable regardless of how fast the matcher is.

**Chunk buffer pool.** 128K row buffers are recycled across pipelines and across statements rather than allocated per chunk. A REPL running a thousand small queries should allocate its chunk buffers once.

**No hidden copies at the boundary.** Registering a frame is a borrow. Returning a result is a move. Document 10 owns the Python side reference counting that makes both safe, and any place that copies is a place where document 01's latency claim quietly stops being true.

## 13. What we measure and publish

Peak RSS per TPC-H query at SF10 against DuckDB, Polars and pandas. Peak RSS as a multiple of input size for a `GROUP BY` at cardinalities from ten to a hundred million, which is the curve that shows whether the aggregate is folding or holding. Time and peak for a query at twice available memory, to demonstrate that spilling works rather than to claim it. Allocation count and bytes for the small statement loop, which is the front end arena's test.

The reporting rule from document 01 applies without exception: no timing is published without its peak memory beside it. A benchmark that wins on time and loses on memory has not won, and a table that omits the column is hiding that.

## 14. The failure mode to design against

Not running out of memory. Running out of memory at ninety five per cent completion of a five minute query, which is the experience that makes users leave.

Three mitigations, in order of value. Reserve early, so an operator that knows it will need a large table asks before it starts and the failure is at second one rather than minute five. Degrade rather than fail, because spilling exists so that the answer is slower and not absent. And report honestly, so that when a query spills, `EXPLAIN ANALYZE` says so, per operator, with bytes, because a user whose query took thirty times longer deserves to be told why rather than left to guess.
