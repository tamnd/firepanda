# SQL

DuckDB's SQL dialect, parsed by firepanda, bound to firepanda's own logical plan, executed by firepanda's own engine. No DuckDB process, no DuckDB planner, no DuckDB execution. The compatibility claim is measured against DuckDB's own test corpus and published as a number rather than asserted. The performance claim is measured against DuckDB, Polars and pandas on TPC-H and on a latency suite, with peak memory beside every timing.

Tracking issue: [#304](https://github.com/tamnd/firepanda/issues/304). Written against DuckDB 1.5.5 and the `v2.0-cyanoptera` branch, Polars 2.0rc1, pandas 3.0, ADBC 1.1.0 and firepanda 0.6.52.

## The one thing that changed this year

DuckDB threw away its parser. Since the beginning it parsed SQL with a vendored fork of PostgreSQL's Bison grammar under `third_party/libpg_query`. In August 2026 they published the replacement, a hand written PEG parser, default in v2.0, living in `src/parser/peg/`. The old third party directory is gone from the v2.0 branch entirely.

The reason they give is that the dialect outgrew the grammar. `GROUP BY ALL`, `SELECT * EXCLUDE (...)`, `QUALIFY`, `PIVOT` and the rest each fought the LALR tables, and every addition risked a shift reduce conflict somewhere unrelated.

The consequence for us is the whole reason this specification is cheap enough to write. DuckDB's SQL dialect is now a declarative artifact: forty `.gram` files, 1,421 lines, 1,087 rules, 61,190 bytes, plus five keyword lists, all MIT licensed, in a syntax whose entire semantics is ordered choice. That is not a parser we have to reverse engineer from behaviour. It is a specification of the dialect that the reference implementation itself executes, which means anything we build from it is compatible by construction on syntax, and any divergence is a diff we can print.

Eighteen months ago, compatible with DuckDB SQL meant reading a Bison grammar and a hand written transformer and hoping. Today it means vendoring 61 KB of text and writing a matcher for it.

## Why this is not a wrapper

firepanda already links DuckDB at run time. `firepanda/io/duckdb.mojo` opens `libduckdb` with `dlopen` and reads Parquet by handing DuckDB a `SELECT * FROM read_parquet(path)` and taking the Arrow chunks back. The laziest possible implementation of `firepanda.sql()` is three more lines in that file.

It would be a dead end for three reasons, and they are worth stating once at the top because the temptation recurs.

It cannot be faster than DuckDB, ever, by construction. The whole product argument in `docs/specs/00-README.md` is that user code and engine code are the same language, and a wrapper puts a C API boundary between the user's Mojo function and the query that should have inlined it.

It cannot see the frames. The interesting query is over data the user already has in memory. A wrapper has to register those frames with DuckDB, which is a copy or an Arrow view, and then convert the result back, which is where the latency axis is won or lost.

And it makes the compatibility claim vacuous. One hundred per cent compatible with DuckDB is not an achievement when the answer is computed by DuckDB.

## Settled decisions

**Vendor DuckDB's `.gram` files verbatim and generate the matcher from them.** They are MIT, they are the reference implementation's own definition of the dialect, and they change by about 78 lines across a patch series and about 700 across a major release. We check them in, we pin the tag they came from, and CI fails when upstream moves and we have not looked. Document 03.

**The grammar is total and the transformer is partial.** Everything the grammar accepts, parses. What we cannot execute is refused after parsing, by name, with the source position, and with a pointer at the issue tracking it. A user must never see a syntax error for valid DuckDB SQL. Document 05.

**Selective packrat memoization, not full packrat.** DuckDB's own numbers: nineteen unmatched parentheses took 10.640 seconds without memoization and 0.001 seconds with it. Their memoized rule list is deliberately short, because caching every rule costs more than it saves. We copy the list and the reasoning. Document 04.

**SQL binds to the same logical plan as everything else, and that plan does not exist yet, so this work builds it.** `firepanda/frame/frame.mojo` is eager and `firepanda/exec/pipeline.mojo` is a push driver with no planner above it. Document 08 specifies the plan, and document 12 argues that this is the reason to pull SQL ahead of M11 rather than the reason to leave it there. The plan is load bearing for the lazy frame, the optimizer and `query()`, and SQL is the forcing function that gets it written.

**No JIT, because Mojo is ahead of time.** The expression executor is a vectorized interpreter over comptime monomorphized kernels, which is DuckDB's design and not Neumann's. Document 09 says what we get instead of compilation, and it is not nothing: a fused chain is a comptime parameter, so the shapes that matter can be monomorphized at build time rather than at query time.

**Compatibility is a measured number in the README, never a claim.** Document 11 defines the harness, and it runs DuckDB's `.test` files unmodified. The day the number is 94.1 per cent the README says 94.1 per cent.

**Nothing in the SQL path reaches the filesystem or the network unless the caller asked for it.** `read_csv` and `read_parquet` inside a query string are table functions that open files. DuckDB has `enable_external_access` and we ship the same flag on day one, permissive for the library API and restrictive for anything that takes SQL from a socket. Document 10.

## The documents

| | | |
| --- | --- | --- |
| 00 | this file | the pitch, the settled decisions, what to read first |
| 01 | `01-the-goal.md` | the four axes, where ten times is real and where it is not |
| 02 | `02-architecture.md` | the stages, where SQL enters, what it may not touch |
| 03 | `03-the-grammar.md` | vendoring the `.gram` files, the codegen, the churn measurements |
| 04 | `04-the-parser.md` | tokens, the PEG matcher, the first token filter, memoization, errors, the budget |
| 05 | `05-ast-and-binder.md` | parse tree to AST, the refusal, name resolution, the catalog |
| 06 | `06-types-and-semantics.md` | the type lattice, decimals, nulls, the divergences that bite |
| 07 | `07-functions.md` | 948 names, the tiers, overload resolution, the protocols |
| 08 | `08-plan-and-optimizer.md` | the shared IR, decorrelation, the pass pipeline |
| 09 | `09-execution-and-memory.md` | physical planning onto the morsel engine, the memory budget |
| 10 | `10-front-doors.md` | `sql()`, `query()`, the CLI, ADBC, parameters, sandboxing |
| 11 | `11-conformance.md` | the 4,046 test files as oracle, fuzzing, the published score |
| 12 | `12-stages.md` | S0 to S8, exit criteria, why this moves ahead of M11 |
| 13 | `13-open-questions.md` | the ranked list, and when each has to be answered |

Read 01 first, then 03. Document 01 decides whether the project is honest, and document 03 is the mechanism that makes the compatibility half of it cheap.

## What this is not

**Not a database.** No storage, no catalog persistence, no transactions, no MVCC, no `ATTACH`. `CREATE TABLE` creates a frame in a session scoped namespace and nothing survives the process.

**Not a server.** DuckDB v2.0 becomes one, with a wire protocol and a `CONNECT` statement. That is a coherent direction for a database and it is not one for a dataframe library.

**Not a SQL dialect of our own.** There is no firepanda extension to the dialect, no relaxation, no convenience we invented. Everything in the surface is something DuckDB does, because the moment we add one thing DuckDB does not have, the compatibility number stops meaning what it says.

**Not a transpiler to another engine.** No Substrait emission, no shelling out to DuckDB for the queries we cannot run. The refusal in document 05 is a refusal and not a fallback, because a fallback would make the compatibility number a measurement of DuckDB.

## Honesty about scope

The parser is the cheap half and the reason to start. Grammar, tokenizer, matcher and the transformer for the SELECT surface is a few thousand lines of Mojo plus a generator, and it is testable against DuckDB's corpus from the first week.

The expensive half is everything after it, and two numbers set the size. DuckDB's hand written transformer is about 500 KB of C++ across 45 files, of which one file for expressions is 132 KB. Its function catalog is 2,951 overloads under 948 distinct names. We are not implementing 948 functions, and document 07 says which tier we are implementing and what the rest do instead.

The thing most likely to kill it is neither the parser nor the function count. It is the semantics. `1/2` is `0.5` and a DOUBLE, `1.1 + 2.2` is `DECIMAL(3,1)`, `sum()` over an INTEGER column is HUGEINT, `'a' || NULL` is NULL while `concat('a', NULL)` is `'a'`, `127::TINYINT + 1` raises rather than wrapping, and `[1,2,3][1]` is `1` because lists are one based and slices are inclusive. Every one of those is measured in document 06, every one disagrees with either pandas or Arrow or both, and each is a silent wrong answer rather than a crash. That document exists because the parser is what people think the hard part is and it is not.
