# The AST and the binder

Two stages. The transformer turns a generic parse tree into something with SQL meaning, and it is where the grammar's totality meets our partiality. The binder turns names into references and an AST into a logical plan, and it is where most of DuckDB's behavioural surface that is not in the grammar actually lives.

## 1. Why there is a transformer at all

The `ParseResult` tree from document 04 has one node per grammar rule, which for the ordinary `SELECT` is dozens of nodes of pure syntax: the rule that names a keyword, the rule that wraps a list, the rule that exists so another rule could reference it. Binding against that directly would tie the binder to grammar rule names, and a grammar bump would then break the binder rather than a small translation layer. The transformer is the shock absorber that keeps document 03's bump procedure a one day job.

DuckDB's equivalent layer is 46 files and 2,817,088 bytes, of which roughly 2.3 MB is generated boilerplate for serialization, copy and equality, leaving about 500 KB hand written, with `transform_expression.cpp` alone at 132 KB. That is the honest size signal for this stage. We write less because we transform less, and section 5 says exactly how much less.

## 2. The AST

Arena allocated, index referenced, three arenas: statements, table references, expressions.

```
struct Expr:
    var kind: ExprKind
    var a: UInt32          # operand or child index, meaning is kind specific
    var b: UInt32
    var children: UInt32   # index into a side list for variadic nodes
    var token: UInt32      # position, for errors, every node carries one
    var payload: UInt32    # literal index, function name index, and so on
```

Fixed size nodes in a `List`, variadic children in a side list of index runs. The alternative, a variant with an inline `List` per node, allocates per node and fights ownership for no benefit.

`ExprKind` covers literal, column reference, star with `EXCLUDE`, `REPLACE` and `RENAME` payloads, function call with `DISTINCT`, `ORDER BY`, `FILTER` and window spec, operator, cast, case, subquery, exists, in, between, collate, lambda, list, struct and map constructors, slice, extract, and parameter.

Statement kinds at 1.0 are `SELECT`, `CREATE TABLE AS`, `CREATE VIEW`, `INSERT`, `EXPLAIN`, `DESCRIBE`, `SUMMARIZE`, `PRAGMA` and `SET`, `COPY`, and `PREPARE` and `EXECUTE`. Everything else in the grammar parses into an `Unsupported` node carrying the rule name and the token span, which is the mechanism in section 4.

Every node has a token index. Not for debugging: every binder and runtime error in document 02's taxonomy renders a caret, and a node with no position produces the error message nobody can act on.

## 3. The transformer

A recursive walk over `ParseResult`, dispatching on rule index. The rule index is a small integer from the generated table, so the dispatch is a jump table and not a string comparison.

Three properties it must have.

**Total over the grammar.** Every rule index has a case. Most cases are descend into the single child or collect children into a list, and can be generated. The interesting ones, which are expression precedence, table references and the `SELECT` clauses, are hand written. A missing case is a compile error and not a runtime fallthrough, and this is enforced by generating the dispatch table's shape from the same rule table the matcher uses.

**Ignorant of the catalog.** It resolves nothing. `foo.bar` becomes a two part name, not a column. `count(x)` becomes a function call with a name, not a resolved overload. Every decision that needs to know what exists belongs to the binder. This split is what lets the transformer be a pure function of the parse tree, which is what lets it be fuzzed with round trip as the invariant.

**Precedence flattening.** The grammar expresses the sixteen precedence levels in `expression.gram` as a chain of rules, so `1 + 2 * 3` arrives as a deeply nested tree of single child pass throughs. The transformer collapses those to a binary tree with the right shape. Getting this wrong is a wrong answer bug that no syntax test catches, so document 11 round trips: AST printed back to SQL, reparsed, compared structurally, over the whole corpus.

## 4. The refusal

This is the single most important section in the compatibility story.

When the transformer hits a rule it cannot represent, or the binder hits a feature it cannot execute, the result is not a syntax error and not a silent no-op. It is:

```
Not Implemented Error: firepanda does not support ALTER TABLE.
LINE 1: ALTER TABLE t ADD COLUMN c INTEGER;
        ^
firepanda has no persistent catalog; tables are frames in a session
namespace. See https://github.com/tamnd/firepanda/issues/13
```

Three components, all mandatory: the feature by name, the position, and what firepanda is instead, with a tracking link. The third component is the one that gets dropped and it is the one that stops the user filing the bug.

Every refusal is a table entry, not a scattered `raise`. `firepanda/sql/unsupported.mojo` holds `(rule_or_feature, message, explanation, issue)`, which means the refusal set can be enumerated. `firepanda.sql_support()` returns it, the README's compatibility table is generated from it, and document 11's harness can assert that every corpus failure in a non target directory failed for a refusal rather than a crash or, worse, a wrong answer.

The prohibition that makes it work: no statement may be parsed and then ignored. A `CREATE INDEX` that succeeds and does nothing is worse than one that refuses, because the user's next query is silently slow and they have no way to find out why. The same goes for `BEGIN`, for isolation hints and for storage options. If we do not do it, we say so.

## 5. What transforms at 1.0

Ordered by what the four axes in document 01 actually require.

**Tier 1, the analytical `SELECT` surface.** Full `SELECT` with `WITH` including `RECURSIVE`, `FROM` with all join types, `WHERE`, `GROUP BY` including `ALL`, `GROUPING SETS`, `CUBE` and `ROLLUP`, `HAVING`, `QUALIFY`, window specs, `ORDER BY` including `ALL` and `NULLS FIRST` or `LAST`, `LIMIT` and `OFFSET`, set operations including `UNION BY NAME`, `DISTINCT ON`, subqueries in every position, `LATERAL`, `VALUES`, and table functions. The whole expression grammar. `SELECT * EXCLUDE`, `REPLACE` and `RENAME`, and `COLUMNS()`. Lateral column aliases. Trailing commas. This is the tier that document 01's conformance target is measured over, and it is most of the work.

**Tier 2, the frame shaped statements.** `CREATE TABLE AS`, `CREATE OR REPLACE VIEW`, `INSERT INTO ... SELECT`, `COPY ... TO`, `DESCRIBE`, `SUMMARIZE`, `EXPLAIN`, `PREPARE` and `EXECUTE`, `SET` and `RESET` for the settings we honour, and `PIVOT` and `UNPIVOT`. Each of these maps onto something a dataframe already does.

**Tier 3, refused by name.** `ATTACH` and `DETACH`, `BEGIN`, `COMMIT` and `ROLLBACK`, `CREATE INDEX`, constraints, `ALTER`, `CREATE SEQUENCE`, `CREATE MACRO`, `CREATE SECRET`, `INSTALL` and `LOAD`, `CALL`, `CHECKPOINT`, `EXPORT` and `IMPORT DATABASE`, and `CONNECT`. All parse, all refuse, all listed.

`UPDATE` and `DELETE` sit awkwardly and document 13 keeps them open. They are expressible over an immutable frame as a rewrite, they are not what a dataframe user reaches for, and implementing them half way is worse than refusing.

## 6. The printer

An AST to SQL printer, written at the same time as the transformer rather than after.

It pays for itself three times. Round trip is the transformer's main test: parse, print, reparse, compare, over 4,046 files, and it catches precedence bugs that nothing else catches. It is the fuzzer's oracle: generate a random AST, print it, parse it, compare. And it is how `EXPLAIN` shows filter and projection expressions, which is how a user debugs a pushdown that did not happen.

It is not a formatter and does not aim to preserve the user's text. It prints fully parenthesized where precedence is involved, because a printer that tries to minimize parentheses is a second implementation of precedence and therefore a second place to get it wrong.

## 7. The catalog is a namespace, not a database

firepanda has no storage, so the catalog is a map from name to something already in memory or reachable as a file.

```
struct Catalog:
    var frames: Dict[String, DataFrame]        # registered, by name
    var views: Dict[String, LogicalPlan]       # CREATE VIEW, unexpanded
    var ctes: ...                              # per statement, not here
    var functions: FunctionRegistry            # document 07
    var settings: Settings
```

Session scoped, dies with the process, no persistence, no schemas, no `ATTACH`. Names are case insensitive and fold down, matching the tokenizer.

Three ways a name resolves, in order:

1. A CTE in scope for this statement. Innermost wins.
2. A registered frame or view. `register("t", df)` explicitly, or implicitly, where `sql()` at module scope makes locals and globals of the calling Python frame visible by name, which is what makes `duckdb.sql("SELECT * FROM my_df")` feel like magic and is worth copying. Document 10 specifies the capture rules and the ambiguity errors.
3. A replacement scan. `FROM 'data/*.parquet'` and `FROM 'x.csv'` rewrite to the corresponding table function. This is DuckDB behaviour, it is the most loved thing in the dialect, and it is the one that reaches the filesystem, so it is gated by `enable_external_access` per document 02.

If none matches, the error is `Binder Error: Table with name t does not exist!` followed by DuckDB's own "Did you mean ...?" over the registered names, because the corpus matches error text by substring and because the suggestion is genuinely most of the value.

## 8. Binding scopes

A `BindContext` per query level, with a parent pointer for correlated subqueries.

Each entry is a binding: a name, a column list with types, and a source. Sources are base frame, table function, subquery, CTE, join, and `VALUES`. Column resolution walks the current level's bindings first and then parents, and a resolution that crosses a level boundary marks the subquery correlated, which is the flag document 08's decorrelation pass keys on.

The rules that have to match DuckDB exactly, each of which is a compatibility bug if it does not:

**Unqualified column ambiguity.** A bare `x` matching two bindings is an error, unless the join was `USING` or `NATURAL`, in which case the merged column is unambiguous and refers to the coalesced value.

**Qualified resolution** is `column`, `table.column`, `schema.table.column` and `database.schema.table.column`, disambiguated by trying the longest interpretation first, which means `a.b` where `a` is a struct column is a field access, and where `a` is a table is a column reference, and the order of those attempts is observable.

**Alias visibility.** `SELECT a + 1 AS b, b * 2 FROM t` works in DuckDB, because lateral column aliases resolve left to right within the select list, and `WHERE b > 1` also works, referring to the select list alias, which is a Postgres divergence and appears constantly in the corpus. `GROUP BY b` and `ORDER BY b` likewise, with `ORDER BY` additionally accepting a positional integer.

**`USING` and `NATURAL` column merging**, including which side survives in `SELECT *`, and `NATURAL` matching on name with the join type applied.

## 9. Star expansion

`SELECT *` is not one feature, it is six, and they compose. All of them expand here, before the plan exists, so the plan never contains a star.

`*` expands over the bindings in order. `t.*` over one binding. `* EXCLUDE (a, b)` removes by name and errors if a name is not present. `* REPLACE (expr AS a)` substitutes in place, keeping position. `* RENAME (a AS x)` renames in place. `COLUMNS('regex')` and `COLUMNS(lambda)` select by pattern, and, the part people miss, `COLUMNS()` inside a function call distributes, so `min(COLUMNS(*))` becomes one aggregate per column. `STRUCT.*` unpacks a struct column into its fields.

Expansion happens after the `FROM` is bound and before the select list is bound, and the resulting order is observable in every corpus test that uses `*`, so it is not a detail.

## 10. Aggregate and window binding

The part of binding that most implementations get subtly wrong, because it requires classifying expressions before you can build the plan.

Walk the select list, `HAVING`, `QUALIFY` and `ORDER BY`, and classify each expression as scalar, aggregate or windowed. Then:

- Aggregates in the select list with no `GROUP BY` mean one group over everything, and an empty input still produces one row, where `count(*)` is `0` and `sum(x)` is `NULL`. That asymmetry is in the corpus.
- A non aggregated column in the select list alongside an aggregate is an error naming the column. DuckDB's message is good and worth matching.
- `GROUP BY ALL` collects every non aggregate select list expression, in order. `ORDER BY ALL` orders by every output column.
- `GROUPING SETS`, `CUBE` and `ROLLUP` expand into a set of grouping set masks on one aggregate node, with `GROUPING()` reading the mask, and not into a `UNION` of separate aggregates, which is the naive lowering and is quadratically slower.
- `HAVING` binds against grouped output. `QUALIFY` binds against window output, which is why it needs a separate filter node above the window node.
- Windows bind their `PARTITION BY`, `ORDER BY` and frame. The frame's `RANGE`, `ROWS` and `GROUPS` modes, its `EXCLUDE` clause and its offset expressions are all part of the spec and all appear in `test/sql/window`.
- `FILTER (WHERE ...)` on an aggregate is a per aggregate predicate, not a `WHERE`.
- `DISTINCT` inside an aggregate, and `ORDER BY` inside `string_agg` and `array_agg`, are per aggregate modifiers.

Nested aggregates are an error. Aggregates inside a window's arguments are legal and bind at the level below.

## 11. Subqueries

Four shapes, and the binder tags each because the plan node differs. Scalar, which must produce one column and raises at runtime on more than one row. `EXISTS`. `IN` and `NOT IN`, with the null semantics from document 06, which is the classic wrong answer trap. And quantified `ANY` and `ALL`.

Each is either uncorrelated, meaning bind independently and plan as a separate subtree, or correlated, which is recorded with the exact set of outer columns referenced. Document 08 owns the decorrelation, and the binder's only job is to make the correlation explicit rather than leaving it to be discovered by a pass walking for outer references.

`LATERAL` is a correlated table reference rather than a correlated expression, and it binds against the bindings to its left in the same `FROM`.

## 12. CTEs

Non recursive CTEs bind in order, each visible to those after it. On materialization, DuckDB defaults to inlining, offers `MATERIALIZED` and `NOT MATERIALIZED` hints, and inlines by default unless the CTE is referenced more than once. We follow, and document 08 owns the cost decision.

`WITH RECURSIVE` binds the anchor first, uses its schema as the recursive binding's schema, then binds the recursive term. It becomes a fixed point node in the plan. `USING KEY` is DuckDB's variant for keyed recursion, and document 13 keeps it open at 1.0.

## 13. What the binder produces

A `LogicalPlan`, specified in document 08, whose expressions are `BoundExpr`. Every column is a `(binding_index, column_index)` pair, every function is a resolved overload from document 07, every cast is explicit including the implicit ones the binder inserted per document 06, and every literal has a concrete type. No names, no ambiguity, no strings to resolve later.

That property is what makes the optimizer safe to write. A pass that reorders or duplicates nodes cannot change what a column means, because the column is an index rather than a name, and there is no scope for it to be reinterpreted in.

## 14. Where this stage is measured

Binding is on the latency path and it is the stage most likely to blow the 20 us front end budget from document 01, because it is the first one that touches dictionaries and allocates per column.

Three rules. The binding context allocation is pooled across statements. Name lookup on a small catalog uses a linear scan over a small vector rather than a hash map, because for the eight frame catalog a REPL actually has, the scan wins. And the prepared statement cache in document 10 caches the bound plan and not just the AST, so the repeat execution number is measured against a path that skips this stage entirely, which is only sound because the catalog generation counter is part of the cache key.
