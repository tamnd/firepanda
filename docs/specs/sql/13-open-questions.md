# Open questions

Ranked by how expensive the answer gets if it is deferred. Each has the stage from document 12 by which it must be answered, and the evidence that would answer it. A question with no decision criterion is not an open question, it is an unexamined assumption.

## 1. Does M4 absorb this, or does M4b follow it? Before M4

The only question here that changes the project's plan rather than this specification's content, and document 12 makes the case for M4b.

The decision depends on evidence that does not exist yet, which is the does anyone install it gate after M3. If early users ask for SQL, M4b. If they ask for the pandas surface, M6 first and SQL after M7 as originally planned.

What does not depend on the answer, and should be settled either way, is that the logical plan built in M4 is the one specified in document 08, designed against DuckDB's dialect. That costs nothing extra and it is what prevents a rewrite in either ordering.

## 2. Node chain lowering or an expression executor? By S6

Document 09 makes a position and it is a position, not a conclusion. firepanda lowers expression trees into chains of `Compute` nodes, each materializing a 128K value intermediate. DuckDB uses an interpreter over a recycled vector pool.

**The criterion.** Instrument intermediate allocation and bytes touched on TPC-H q1 and q6 at SF10. If intermediates exceed roughly twenty per cent of total bytes touched, build the executor. If not, the vector pool plus comptime fused chains is enough and the interpreter is complexity we do not need.

Deferring past S6 is expensive because the optimizer's expression rewrites and the fusion patterns are written against whichever model wins.

## 3. How much of the type system is really required? By S3

Document 06 asserts that DECIMAL128 and INT128 are hard requirements, because `1.1 + 2.2` is `DECIMAL(3,1)` and `sum(INTEGER)` is HUGEINT. Neither exists in firepanda today and both are substantial: fixed point arithmetic with derived precision, and 128 bit integers in every arithmetic and aggregate kernel.

**The criterion.** Count the corpus files under the target directories whose expected output changes without each. The prediction is that decimals are unavoidable and hugeint sums nearly so. If decimals turn out to affect only a small slice, a DOUBLE backed approximation with a documented divergence becomes arguable, but only with the count in hand, because computing `1.1 + 2.2` as `3.3000000000000003` is a compatibility failure a user will hit in their first session.

## 4. `UPDATE` and `DELETE`, implement or refuse? By S5

Frames are immutable. `UPDATE t SET a = a + 1` is expressible as a projection producing a new frame, and `DELETE` as a filter. Both are mechanical.

The problem is that SQL's version mutates in place and a user who writes `DELETE FROM t WHERE ...` expects `t` to change afterwards. Rebinding a session name to a new frame is a plausible semantics and it diverges the moment anything else holds a reference to the old frame, which in Python is normal.

**Three options.** Refuse both by name, which is the current position and the safest. Implement with rebinding semantics and document the aliasing behaviour loudly. Or implement returning a new frame and refuse to rebind, which is coherent and will surprise everyone.

**The criterion.** How much of the corpus outside the target directories depends on them, and whether users ask. Refusing is cheap to reverse, and the wrong semantics shipped is not.

## 5. How far does time zone support go? By S3

`now()` is `TIMESTAMP WITH TIME ZONE`. Full `TIMESTAMPTZ` means a session time zone, the IANA database, DST transition rules, and ICU equivalent behaviour for `date_trunc` and interval arithmetic across a transition. In DuckDB much of this lives in an extension.

**The 1.0 position** in document 06 is that naive `TIMESTAMP` is fully supported, `TIMESTAMPTZ` is supported for UTC, and anything else is a named refusal. **The criterion for going further** is corpus coverage under `test/sql/timestamptz` plus whether M7, the time series milestone, needs the tz database anyway. If it does, this question is answered by that milestone and not by this one.

## 6. Statistics, exact counts only or sketches? By S6

Document 08 starts with exact row counts and per column minimum, maximum and null count, which is more than a database has at plan time and is enough for TPC-H.

It is not enough for correlated predicates, for non equality selectivity, or for join key distinctness on skewed data. HyperLogLog for distinct counts and a small sample per column are the standard answers.

**The criterion.** When a query in the benchmark suite picks a plan more than twice as bad as the best available plan and the cause traces to an estimate rather than to a missing rewrite. That is a specific, observable trigger, and building sketches before it fires is speculative.

## 7. What is `enable_external_access` inside a user defined function? By S7

Document 10 sets defaults per door: permissive for the library call, restrictive for the CLI and ADBC. The gap is `df.sql()` called from a user defined function that is itself executing inside a query that arrived through a restrictive door.

The principle is that capability should not be regained by nesting. The mechanism is presumably that the setting is per context and inherited, and that a nested `sql()` inherits rather than resets. Worth settling before UDFs exist rather than after, because retrofitting a capability model is how the interesting vulnerabilities happen.

## 8. Do we ship grammar extension hooks? Post 1.0

DuckDB v2.0 ships hooks so extensions can add syntax. It is the right design for a system with an extension ecosystem, and it is a promise about a stable interface.

Not at 1.0, per document 02, because the interface would be a public API before we know what it is for. Revisit if a plausible extension appears, such as a domain specific function set or a custom table function, and note that the underlying mechanism costs little given a generated matcher, so the question is about the API commitment rather than the implementation.

## 9. `WITH RECURSIVE ... USING KEY`. By S5

DuckDB's keyed recursion variant. It is in the grammar, so it parses. Whether it executes at 1.0 depends on corpus coverage and on whether the fixed point operator needs a different shape to support it. The default is to refuse by name and file the issue.

## 10. Do we vendor the conformance corpus, or fetch it? By S1

Document 12 says vendor. 4,046 files is a real weight in the repository, and the alternative is fetching at the pinned tag in CI.

Vendoring wins on reproducibility and on offline builds, and it means a bisect over a year old commit runs the corpus that commit was tested against. Fetching wins on repository size. The tie breaker is whether the corpus is large enough to matter to clone time, and if it is, a shallow fetch of the pinned tag in CI plus a documented local cache path is the compromise. Measure the size before deciding.

## 11. String representation, when does short string inlining land? By S6

Document 09 notes that a twelve to sixteen byte inline prefix beside the pointer makes comparison and equality resolve without a dereference, and that it benefits every existing kernel rather than only SQL.

It is a storage layer change in `array/strview.mojo`, which already reserves a sixteen byte view layout, so the groundwork may already be there. The question is sequencing, because SQL string work in S3 through S5 will be written against whatever representation exists and doing it twice is waste. Settle before S3 whether the layout is final.

## 12. Questions that are settled and should stay settled

Recorded so they are not reopened by each new contributor.

**We do not wrap libduckdb for query execution.** Document 00 gives three reasons and they do not weaken over time.

**We do not extend the dialect.** Not one convenience, not one relaxation. The compatibility number stops meaning anything the moment we do.

**We do not fall back to DuckDB for unsupported queries.** A fallback would make the conformance number a measurement of DuckDB.

**We do not JIT.** Mojo is ahead of time. `comptime` monomorphization is the substitute and document 09 says what it buys.

**We follow DuckDB on ordering, not Polars.** `preserve_insertion_order = true`. The corpus is the oracle and the corpus assumes order.

**We publish losses.** Every benchmark table includes the queries where we lose, with the reason.
