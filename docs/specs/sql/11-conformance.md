# Conformance

The document that turns one hundred per cent DuckDB compatible from a claim into a number. DuckDB ships its own test suite under an MIT license: 4,046 `.test` files under `test/sql/`, in sqllogictest format, which is a plain text format designed to be run by any engine. That corpus is the oracle, and everything here is about how to run it honestly.

## 1. The format

```
statement ok
CREATE TABLE t (a INTEGER, b VARCHAR);

query IR
SELECT a, b FROM t ORDER BY a
----
1	one
2	two
```

`statement ok` and `statement error` for statements, `query <typestring>` for results, where the type string is one character per column: `I` for integer, `R` for real, `T` for text. Results come after `----`, tab separated. Larger results are given as an MD5 hash of the values with a row count, which keeps the files small.

Directives that matter for us: `require <extension>` skips a file we do not have, `mode skip` and `mode unskip` bracket sections, `loop` and `endloop` parameterize, `sort` and `rowsort` control comparison, and `onlyif` and `skipif` gate by engine name. That last one is how DuckDB's own files handle divergence, and it is why running the corpus unmodified is possible at all.

## 2. The rules

**The files are run unmodified.** No local edits, no patched expectations, no per file skips added to make a number look better. The corpus is vendored at a pinned DuckDB tag alongside the grammar, per document 03, and refreshed with it.

**Every failure is classified automatically into one of five buckets:** unsupported feature, meaning a document 05 refusal; missing function, meaning a document 07 tier 2 or tier 3 name; semantic divergence, meaning the right shape with a wrong value or type; crash; and wrong answer without an error. The last two are bugs at the top of the queue. The first two are the roadmap.

**The number published is per directory, with the denominator.** For example, `test/sql/aggregate: 412/431 (95.6%)`. A single blended percentage over 4,046 files is not information, because it is dominated by directories we have deliberately not implemented and it can be moved by adding files rather than by adding features.

**Nothing is rounded up, and the README is generated from the harness output** rather than edited by hand. This is the mechanism and not the aspiration, because a number a human types is a number that drifts.

## 3. The three harnesses

**The parse differential harness.** Every statement in the corpus, plus generated SQL, through both parsers, comparing accept against reject. This is the hard gate from document 01, running from the first week the matcher exists, long before anything executes. Its output is a list of strings where we disagree, and an empty list is the requirement.

It runs against DuckDB in process through the same `libduckdb` that `firepanda/io/duckdb.mojo` already opens with `dlopen`, using `json_serialize_sql` for the parse only path, which is also how the parse cost numbers in document 04 were measured.

**The execution harness.** Runs `.test` files against firepanda, compares values, classifies failures. It drives the CLI from document 10, so it is testing the whole stack the way a user gets it.

**The semantics harness.** One case per rule in document 06, comparing value and `typeof()` against DuckDB in process, plus the expression fuzzer and the overload resolution fuzzer from document 07. This one catches what the corpus does not, because the corpus was written to test DuckDB's features rather than to pin down its type lattice.

## 4. Fuzzing

firepanda already fuzzes the bitmap, the kernels, the hash and the join against scalar twins. `pixi run fuzz` runs four fuzzers, and the twins in `firepanda/kernel/scalar.mojo`, `hash/scalar.mojo` and `join/scalar.mojo` exist to be the specification. SQL adds four more, and the pattern is the same: an oracle that is obviously correct, and a fast thing checked against it.

**Grammar directed generation.** The vendored `.gram` files are a generator as well as a recognizer, so walk the rules, choose alternatives and emit tokens. Every string produced is valid DuckDB SQL by construction, and both parsers must accept it. This is the fuzzer that makes the parse agreement claim mean something beyond the corpus, and it exists only because the grammar is declarative, which is the same reason document 03 exists.

**Round trip.** Parse, print with the printer from document 05, reparse, compare ASTs. Catches precedence and associativity bugs that accept against reject cannot see.

**Query generation with a differential oracle.** SQLancer style: generate random queries over random small tables, run through DuckDB and firepanda, compare results. This is what finds wrong answers, and wrong answers are the failures that matter most because everything else is loud.

**Pathological input.** The cases from document 04, meaning deep nesting, long lists and unmatched parentheses, each with a wall clock ceiling in CI, because a PEG parser's failure mode is exponential and a front door that takes untrusted SQL turns that into a denial of service.

## 5. The plan equality test

The single most valuable test in this specification, and it is not about compatibility at all.

For each of the twenty two TPC-H queries, the physical plan produced by `fp.sql(q)` and the physical plan produced by the equivalent dataframe chain must be structurally equal.

It costs almost nothing, because the plans already print and round trip per document 08, and it is what enforces document 02's rule that SQL and dataframes are one engine. Without it, the SQL path grows its own lowering for one operator, then another, and a year later there are two engines with different bugs and different performance, which is precisely the outcome that issue #13's line about parsing into the same logical plan so the optimizer is shared exists to prevent.

## 6. The optimizer equivalence test

From document 08, restated because it belongs to conformance as much as to the optimizer: every query in the corpus runs twice, once with all optimizer passes disabled and once with all enabled, and the results must be identical including order.

This is the test that catches a filter pushed through a node that does not preserve its meaning, a join reordered across an outer join that is not reorderable, or a decorrelation that changed null semantics. Those bugs produce plausible wrong answers on real queries and are nearly impossible to find any other way.

## 7. What runs when

**Every commit:** the parse differential over the full corpus, the semantics cases, unit tests, and the pathological input ceilings. Minutes.

**Every commit:** the execution harness over the target directories with a pass rate floor that ratchets, so the number may go up and a commit that lowers it fails. Ratcheting rather than a fixed threshold is what stops the number quietly sliding while everyone is busy.

**Nightly:** the full 4,046 files with the classification report, all four fuzzers with a time budget, optimizer equivalence, plan equality, and the benchmark suite with peak memory.

**Weekly:** the grammar bump check from document 03 against the latest upstream tag.

## 8. What the corpus does not cover

Stated so that the published number is read correctly.

It does not cover performance, because a query that takes an hour passes.

It does not cover memory, because a query that uses 40 GB passes. That is document 09.

It does not cover the Python boundary, meaning capture rules, parameter binding, GIL behaviour and exception types. Those are firepanda's own tests, per document 10.

And it does not cover concurrency. `pixi run stress` is firepanda's substitute for a race detector, and SQL adds cases to it: the same query from several threads, registration racing execution, and cache invalidation under concurrent registration. The prepared statement cache in document 10 is shared mutable state on the hot path, which makes it the most likely place in this whole specification for a race, and it is the one that a correctness test will never find.
