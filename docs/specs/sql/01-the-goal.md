# The goal

Read this one first. It is the document that decides whether the project is honest, and it exists because the brief that produced it, one hundred per cent compatible with DuckDB's syntax and an engine ten times faster than any rival, contains one claim that is achievable exactly as stated and one that is not. The difference has to be written down before any code is.

## 1. The claim that is achievable as stated

Syntax compatibility with DuckDB can be one hundred per cent, and it can be proven rather than asserted.

Not because we are good, but because since August 2026 the dialect is a file. Forty `.gram` files, 1,087 rules, 61,190 bytes, MIT licensed, executed by the reference implementation itself. Generate a matcher from those files and every string DuckDB's parser accepts, ours accepts, and every string it rejects, ours rejects, up to the fidelity of the tokenizer and the matcher, which is a bounded and mechanically diffable surface rather than an open ended one.

So the syntax axis gets an absolute target and a test that can fail. For every statement in DuckDB's 4,046 `.test` files, and for every statement in a corpus of generated SQL, firepanda's parser and DuckDB's parser agree on accept against reject. Disagreements are bugs with a fixed cost, not a percentage we negotiate.

Note carefully what this does and does not promise. It promises we never emit a syntax error for something DuckDB accepts. It does not promise we execute it. Those are different claims and document 05 keeps them apart on purpose, because conflating them is how one hundred per cent compatible becomes a lie. A system that parses `CREATE INDEX` and then ignores it is worse than one that parses it and says firepanda has no indexes and points at the issue tracking that.

## 2. The claim that is not achievable as stated

Ten times DuckDB on analytical query throughput is not a target, it is a fantasy, and saying so now is cheaper than discovering it at SF10.

DuckDB is a mature vectorized engine with morsel driven parallelism, a radix partitioned hash aggregate, adaptive expression reordering, statistics driven join ordering, and years of profiling behind every operator. Our own measurements in `docs/specs/engine/01-what-we-take.md` have us at nine times slower than DuckDB on db-benchmark j1 today. The path from there is the engine work in M2b and M5, and the honest ceiling on a shared design running on the same hardware over the same data is parity plus whatever we win on specific operators. Anybody promising ten times on TPC-H against DuckDB is either changing the hardware, changing the data, or changing the question.

There are three places where a multiple of ten is real, and they are all places where the rival is not slow but is doing extra work that we can structurally avoid. That is what the performance half of this specification is actually about.

**The boundary.** A Python user with data in memory who writes `duckdb.sql("SELECT ...").df()` pays to register the frames and pays again to materialize the result into pandas. firepanda's query is over frames it already owns, in the layout the kernels already read, and its result is a frame. Nothing is converted at either end because there is no other end. On a query whose compute is a few milliseconds, which is most interactive queries, the conversion is the query.

**Latency.** DuckDB is a database. Every statement goes through a catalog lookup, a transaction, a binder that resolves against persistent schemas, an optimizer pass list and a physical planner. Measured on an M4, a trivial statement costs about six microseconds to parse alone. We have no catalog to persist, no transaction to begin, and a prepared statement cache that can key on the exact text. A dataframe REPL runs thousands of small statements, and this is the axis nobody optimizes because databases cannot.

**Memory on the small to middling query.** DuckDB reserves a buffer pool. Polars 2.0 runs the streaming engine by default and is explicit that ordering is the price. A library that owns its inputs and emits its outputs into the caller's memory has fewer copies available to make, and document 09 turns that into a measured target rather than an intuition.

So the goal is four axes, each falsifiable, each with the instrument named.

## 3. Axis one, dialect compatibility

**Parse agreement one hundred per cent, no exceptions, measured continuously.** The differential parser harness runs every statement in DuckDB's corpus plus generated SQL through both parsers and compares accept against reject. This is a hard gate in CI from the first week the matcher exists.

**Execution conformance as a published pass rate over `test/sql/`, per directory.** The target is that the analytical SELECT surface passes, meaning `select`, `filter`, `aggregate`, `join`, `order`, `window`, `subquery`, `cast`, `types` and `function`, and that everything outside it refuses by name. We publish the per directory table in the README, including the directories that are at zero, because a compatibility number that hides its denominator is marketing.

**Semantic agreement on the things that are not in the grammar.** Every rule in document 06 has a test that runs the same expression through DuckDB in process and through firepanda and compares the value and the type. A `sum()` over an INTEGER column returning BIGINT where DuckDB returns HUGEINT is a compatibility failure even though every row is numerically identical.

## 4. Axis two, query performance

The target is the one issue #299 already set: twice DuckDB, Polars and pandas on every TPC-H query at SF10, or a written reason on each one that is not. SQL inherits it and adds nothing, because the SQL path lowers into the same logical plan the dataframe path uses. If `sql("SELECT ...")` is slower than the equivalent chain of frame calls, the difference is planner overhead and it is a bug with a number attached.

Two rules make this measurable rather than rhetorical.

The same query, expressed both ways, must produce the same physical plan. There is a test that asserts the plan from `sql()` and the plan from the dataframe chain are structurally equal for each of the twenty two queries. That test is worth more than any benchmark, because it is what stops the SQL surface quietly becoming a second and worse engine.

Every published timing carries peak RSS. Not a footnote, a column.

## 5. Axis three, latency

This is our axis and it deserves an absolute number rather than a multiple, because a multiple against an engine doing something else is not informative.

`sql("SELECT a, sum(b) FROM t GROUP BY a")` over a registered frame of ten thousand rows, warm, returning a frame, under one hundred microseconds end to end, of which under twenty is parse, bind, plan and optimize.

And the second order one, which is the one users actually feel. The second execution of a statement whose text has been seen before skips parsing and binding entirely, and a cache hit reaches the physical plan in under two microseconds.

Against the rivals this is measured end to end and in the shape a user would actually write, which for DuckDB means `.df()` or `.arrow()` on the result and for Polars means `SQLContext.execute(...).collect()`. The report says what each system is doing, because ten times DuckDB without the sentence on a query where DuckDB spends most of its time converting to pandas is a dishonest headline even when the number is right.

## 6. Axis four, memory

**Peak RSS within 1.2x of DuckDB on every TPC-H query at SF10**, measured by the harness, reported per query.

**No query in the suite may fail for memory that DuckDB completes.** A query that does not fit gets slower, not fatal. DuckDB's external aggregation completes a 50 GB input on a 16 GB machine roughly thirty times slower than in memory, and finishing thirty times slower is a different category from raising. Document 09 specifies which operators spill and which are allowed to refuse.

**Bounded by construction where possible rather than by a limit that is checked.** Polars' token per morsel backpressure bounds the number of live morsels structurally. Our engine already pushes chunks through a pipeline, and the work is making the SQL operators respect that instead of materializing.

## 7. What we are willing to lose

**Ordering.** Polars 2.0 made unordered the default for `join`, `group_by` and `unpivot`, and it is the right call for a streaming engine. DuckDB keeps `preserve_insertion_order` true by default. We follow DuckDB, because the compatibility number is measured against DuckDB's expected outputs and half the corpus would fail otherwise, and we expose the same setting so a user can trade it away deliberately.

**Breadth of the function catalog.** 948 names is not a target and pretending otherwise is how a milestone becomes a year. Document 07 tiers them and the refusal message names the function.

**DDL and transactions.** No `ATTACH`, no `BEGIN`, no indexes, no constraints, no triggers. Document 05 lists them and each refusal says what firepanda is instead of pretending.

**VARIANT, GEOMETRY, ENUM, UNION and BIGNUM.** Out of scope for the reason `docs/specs/engine/duckdb/06-whats-new.md` already gave for the first two: this is a dataframe library and its type system is Arrow's.

## 8. The one sentence version of each axis

Syntax: everything DuckDB parses, we parse, and the diff is empty.

Semantics: everything we execute, we execute with DuckDB's types and DuckDB's answer, and the ones we do not execute say so by name.

Speed: parity or better with DuckDB on TPC-H at SF10, and an order of magnitude on the interactive path where the rivals are paying for a boundary we do not have.

Memory: within 1.2x of DuckDB at peak, and nothing in the suite dies.
