# Stages

Nine stages, an argument for where they go in firepanda's existing roadmap, and the package layout they land in. The argument comes first, because moving SQL from M11 to just after M4 is the one thing in this specification that changes the project's plan rather than adding to it.

## 1. Why this moves ahead of M11

`docs/specs/08-milestones.md` puts SQL at M11, last, on the path M4 to M6 to M7 to M11. That ordering was right when it was written and three things have changed.

**The parser stopped being the expensive part.** M11's placement assumed SQL meant a hand written parser for a dialect defined by another project's C++ and a two thousand line Bison grammar. Since August 2026 the dialect is 61 KB of MIT licensed declarative text, per document 03. The compatibility half of the work fell by an order of magnitude, and a milestone whose cost estimate is stale should be re-placed rather than left.

**M4 and this specification are largely the same milestone.** M4 is the lazy engine, expressions and optimizer: build the expression nodes, the logical plan, plan time type checking, and the optimizer passes for projection pushdown, predicate pushdown, constant folding, type coercion and common subexpression elimination. Documents 06, 08 and 09 here specify the same artifacts, in more detail, with a demanding consumer attached. Building them twice is the actual risk of the current ordering.

**SQL is the forcing function that makes the plan right.** A logical plan designed against only the dataframe API is designed against a consumer that can be changed to suit it. SQL cannot be changed, because the dialect is fixed by an external artifact, so it exercises subqueries, correlation, grouping sets, window frames and set operations from the start. A plan that survives DuckDB's dialect will survive anything the dataframe API asks of it later, and the reverse is not true. Discovering at M11 that the plan cannot express a correlated subquery means rewriting M4 through M8.

**And the compatibility number is a distribution asset, available early.** After M2 the does anyone install it gate in `08-milestones.md` becomes real. Runs DuckDB SQL, here is the measured conformance table, is a far stronger answer to that gate than a partial pandas surface, and it is available a milestone or two after M3 rather than nine.

The proposal is that SQL becomes M4b, immediately after M4, sharing M4's plan and optimizer work, before M5's parallelism and before M6's pandas surface. The dependency graph becomes:

```
M2 -> M3 (publish)
       |
       +--> M4 -> M4b (SQL) -> M5 -> M8
             |                  |
             +--> M6 -> M7      +---> M9, M10
```

M11 keeps the remaining IO formats, the API freeze and the 1.0 review, and loses only the SQL work.

**The counter argument, stated fairly.** M4b delays M6, and M6 is pandas parity, which is the product's headline claim. If the does anyone use it signal after M3 says users want the pandas surface and do not care about SQL, this ordering is wrong and should be reverted. That is a real gate with a real answer, and it is worth asking rather than assuming. What does not change either way is that the plan and optimizer belong in M4 and should be designed with this specification's requirements in view, because that costs nothing and prevents the rewrite.

## 2. The stages

**S0, grammar and codegen.** Vendor the `.gram` and `.list` files with checksums. Write the PEG notation parser and `tools/gen_grammar.py`. Emit the rule table and keyword hashes. Done when the generator round trips the full grammar and the checked in output is reproducible byte for byte in CI.

**S1, tokenizer and matcher.** The tokenizer from document 04, the rule table interpreter, selective memoization with DuckDB's rule list, and furthest failure errors. Done when the parse differential harness agrees with DuckDB on accept against reject across all 4,046 corpus files and on grammar directed generated SQL, and the pathological cases are inside their wall clock ceilings. This is the stage that makes the compatibility claim testable, and it lands before anything executes.

**S2, AST and transformer.** The AST arenas, the transformer for the tier 1 `SELECT` surface, the refusal table, and the printer. Done when round trip parse, print and parse is structurally stable across the whole corpus and every non tier 1 statement refuses by name.

**S3, binder, catalog and types.** Scopes, resolution, star expansion, aggregate and window classification, subquery correlation, the type lattice and implicit casts from document 06, and the tier 1 function registry. Done when the semantics harness passes every enumerated case on value and `typeof()`, and the expression fuzzer finds no type disagreement in an overnight run.

**S4, the logical plan and the first executions.** The fourteen operators, decorrelation, and enough of the physical planner to run `SELECT`, `WHERE`, `GROUP BY`, `ORDER BY`, `LIMIT` and inner joins on the existing engine. Done when TPC-H q1, q3 and q6 return correct results at SF1, and when the plan equality test passes for those three.

**S5, full SELECT.** Windows, set operations, `DISTINCT ON`, `QUALIFY`, grouping sets, `PIVOT` and `UNPIVOT`, all join types, recursive CTEs, and `unnest`. Done when all twenty two TPC-H queries return correct results at SF1 and the conformance rate over the target directories is published for the first time.

**S6, optimizer.** The pass list in document 08. Done when all twenty two queries complete at SF10, when optimizer off equivalence passes over the corpus, and when the plan snapshots are in the repository.

**S7, front doors.** `sql()`, `df.sql()`, `query()`, `eval()`, parameters, the prepared statement cache, the CLI, and the capability flag. Done when the latency suite hits document 01's numbers, meaning under 100 us end to end and under 2 us on a cache hit, and when the GIL split from document 10 gives real parallelism across threads.

**S8, memory and ADBC.** Backpressure, the extended folding aggregate set, `TopN`, spilling for hash aggregate and sort, external sort, and the ADBC driver. Done when peak RSS is within 1.2x of DuckDB on every SF10 query, when a query at twice available memory completes, and when a third party ADBC client can query firepanda.

## 3. Ordering, and what is deliberately early

**S1 before anything executes.** The parse differential harness running against the full corpus in week one is the cheapest confidence in the project, and it is available before there is an engine to be confident about.

**Types before operators.** S3 precedes S4 because the decimal rules, the HUGEINT promotion and the null semantics from document 06 change what every operator computes. Building operators first means rebuilding them.

**Decorrelation in S4, not S6.** It is in the plan layer rather than the optimizer, it is not optional, and a version of S4 that runs correlated subqueries per row is a version that has to be thrown away.

**The refusal table in S2**, before any real coverage exists. The whole compatibility story depends on the difference between we do not support this and syntax error, and a refusal mechanism retrofitted after a hundred features is a hundred scattered `raise` statements that cannot be enumerated.

**Memory last, and that is a risk.** S8 is where the low memory half of the brief actually lands, and putting it last means measuring it late. The mitigation is that the memory suite runs from S4 with published numbers, so the curve is visible long before the work to fix it starts, and nobody should be surprised at S8.

## 4. What is explicitly not in these stages

Everything document 01 said we would lose: DDL beyond `CREATE TABLE AS` and `CREATE VIEW`, transactions, indexes, `ATTACH`, extensions, the tier 2 and tier 3 function catalog, VARIANT and GEOMETRY and ENUM, the server protocol, and `UPDATE` and `DELETE` pending document 13.

Each of them parses. Each refuses by name. The list is generated from the refusal table and published, so the gap between what firepanda parses and what it runs is a document rather than a discovery.

## 5. The constraint that shapes the package layout

Mojo 1.0's rule, restated because it decides the tree: a package submodule is reachable only if `__init__.mojo` re-exports it. There is no `internal/` and no `pub(crate)`. A symbol re-exported from a package's `__init__.mojo` is public API and a symbol that is not is package private.

That makes `sql/__init__.mojo` the whole of the SQL surface's access control, and it should export exactly three things: `sql(query)`, `SQLContext` which holds the catalog and settings, and `sql_support()` which is the refusal enumeration from document 05. Everything else, meaning the tokenizer, matcher, transformer and binder, is package private, which is what lets all of it be rewritten without a compatibility discussion.

## 6. The tree

```
firepanda/
  sql/
    __init__.mojo          sql(), SQLContext, sql_support(), the entire surface

    grammar/
      VENDOR               upstream ref, commit, date, per file SHA-256
      statements/*.gram    vendored verbatim, never edited          document 03
      keywords/*.list      vendored verbatim
      memoized_rules.list  vendored, DuckDB's packrat rule list     document 04
      matcher_overrides.list  vendored, the rules the matcher owns  document 04

    generated/
      rules.mojo           the rule table, checked in, CI verified
      keywords.mojo        the sorted keyword table

    token.mojo             Token, TokenKind, the tokenizer          document 04
    matcher.mojo           the rule table interpreter, memo table   document 04
    parse_result.mojo      ParseNode arena
    ast.mojo               statement, tableref and expr arenas      document 05
    transform.mojo         ParseResult to AST                       document 05
    print.mojo             AST to SQL, for round trip and EXPLAIN   document 05
    unsupported.mojo       the refusal table                        document 05
    bind/
      context.mojo         BindContext, scopes, correlation
      catalog.mojo         name to frame or view, generation counter
      expr.mojo            expression binding, implicit casts
      star.mojo            *, EXCLUDE, REPLACE, RENAME, COLUMNS()   document 05
      aggregate.mojo       aggregate and window classification      document 05
      subquery.mojo        the four kinds, correlation recording
    function/
      registry.mojo        the overload table, comptime built       document 07
      resolve.mojo         overload resolution, the cast lattice    document 07
      scalar.mojo          tier 1 scalar registrations
      aggregate.mojo       the four operation protocol              document 07
      window.mojo          ranking, navigation, frame strategies    document 07
      table.mojo           read_csv, read_parquet, range, unnest    document 07
    pandas_expr.mojo       the query() and eval() grammar           document 10
    cache.mojo             the prepared statement cache             document 10

  plan/                    shared with the dataframe surface, not under sql/
    node.mojo              the fourteen logical operators           document 08
    bound_expr.mojo        BoundExpr                                document 08
    schema.mojo            per node schema computation
    decorrelate.mojo       dependent join pushdown                  document 08
    serialize.mojo         JSON in and out, EXPLAIN printing        document 08
    optimize/
      rewrite.mojo  pushdown.mojo  project.mojo  stats.mojo
      join_order.mojo  late_materialize.mojo  passes.mojo

  exec/                    existing, SQL adds operators and not a layer
    +hash_join.mojo  nested_loop.mojo  sort.mojo  topn.mojo
    +window.mojo  distinct.mojo  setop.mojo  unnest.mojo  recursive.mojo

  py/
    +sql.mojo              the sql(), df.sql() and query() bindings document 10
  adbc/
    driver.mojo            AdbcDatabase, Connection, Statement      document 10
```

Two placements are load bearing and both follow from document 02.

**`plan/` is not under `sql/`.** It is a sibling, because the dataframe surface and `query()` build the same plans. Putting it under `sql/` would make the plan look like a SQL artifact and would invite a second one for the dataframe path, which is the exact failure this specification is arranged to prevent.

**The new physical operators go in `exec/`, not `sql/`.** Same reason. `HashJoin` is an engine operator that SQL happens to use.

## 7. The dependency rules

One direction only, enforced by a CI grep in the spirit of the existing `check-shared-state` task:

```
sql/  ->  plan/  ->  exec/  ->  kernel/, join/, hash/  ->  array/, bitmap/, buffer/
```

`sql/` may not import `exec/`, `kernel/`, `join/` or `hash/`. Its only output is a `LogicalPlan`. A SQL feature that needs a new kernel gets a kernel that the dataframe surface can also reach.

`plan/` may not import `sql/`. The plan does not know a parser exists.

`exec/` may not import `plan/` or `sql/`. The physical planner sits in `plan/` and constructs `exec/` nodes, and the operators themselves know nothing about where they came from.

Nothing outside `sql/grammar/generated/` may import from it. The rule table is an implementation detail of the matcher.

These are four rules a grep can check and they are worth a CI check, because dependency inversion happens one convenient import at a time and is expensive to unwind later.

## 8. Build cost

`tools/compile_budget.py` already graphs compile time and binary size from M0, because monomorphization is where those blow up. SQL adds two pressures and both need to be on that graph from S0.

**The function registry.** 150 tier 1 names times the dtype set is a large `comptime` expansion, and it is the single most likely thing in this specification to double the build. The mitigation, if the graph says so, is to monomorphize only the dtypes that appear in practice and route the rest through a `Value` based slow path, which is a decision to make with the measurement rather than in advance.

**The rule table.** 1,087 rules as data is a few tens of kilobytes and costs nothing. That is the argument from document 03 for emitting data rather than 1,087 functions, and the graph is what would catch it if the argument were wrong.

## 9. Tests and benchmarks

```
tests/
  sql/               unit tests per stage: token, match, transform, bind, plan
  conformance/
    corpus/          vendored DuckDB .test files, pinned with the grammar
    runner.mojo      the sqllogictest runner                    document 11
    report.py        the per directory table, generated into the README
  differential/
    sql_semantics.mojo    value and typeof() against libduckdb  document 06
    sql_parse.mojo        accept and reject against libduckdb   document 11
  fuzz/
    sql_grammar.mojo      grammar directed generation           document 11
    sql_roundtrip.mojo    parse, print, parse
    sql_query.mojo        random queries, differential oracle
    sql_pathological.mojo the wall clock ceilings
  stress/
    sql_concurrent.mojo   the cache under concurrent registration
```

The conformance corpus is vendored at the same tag as the grammar and refreshed by the same procedure, because a test corpus from one version checking a grammar from another is a source of failures that are nobody's bug.

Benchmarks live in `firepanda-bench` per `docs/specs/10-benchmarks.md`, not here. The SQL suites are added to that repository, and there is no separate SQL benchmark harness, for the same reason there is no separate SQL engine.
