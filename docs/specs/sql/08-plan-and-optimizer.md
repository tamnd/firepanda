# The plan and the optimizer

The contract between the two halves of firepanda. Document 12 argues that this is the real reason to pull SQL forward. The plan is load bearing for the lazy frame, for the optimizer, for `query()` and for issue #299's TPC-H work, and SQL is the forcing function that gets it written with a real consumer instead of a hypothetical one.

## 1. What exists today, and what does not

This section was written when there was no planning layer at all. There is one now, built alongside the binder and specified in `docs/specs/planner/`, so what follows is the state of it rather than the absence of it.

`firepanda/plan/` is the arena, the expression tree, the binder, the printer and the pass list: `node.mojo` holds the plan and its builders, `expr.mojo` the expressions, `bind.mojo` turns every name into a position against a list of source schemas, `print.mojo` writes the indented tree, and the rest are the passes. `firepanda/sql/plan.mojo` is what lowers a `SELECT` into it, and `docs/specs/planner/01-what-a-plan-is.md` is the document that describes it.

`firepanda/frame/frame.mojo` is still eager: a method computes and returns. `firepanda/exec/` has `chunk`, `morsel`, `node`, `parallel` and `pipeline`, which together are a push based driver with a three method node interface of `update_state`, `process` and `finish`, plus a morsel queue. `firepanda/join/` and `firepanda/hash/` have the algorithms. So the plan exists and the road from a plan to the morsel engine does not, which is the gap this stage closes rather than the one it opens.

### 1.1 Nine node kinds against the fourteen below

`docs/specs/planner/01-what-a-plan-is.md` names nine node kinds and says every TPC-H query is expressible in them. Section 3 of this document names fourteen. Both are right about their own surface and the difference is exactly the part of the DuckDB dialect that TPC-H does not use.

The nine are scan, filter, project, aggregate, join, sort, limit, distinct and union. The five this document adds are `TableFunction`, `Values`, `Window`, `Unnest` and `RecursiveCTE`, and they are additions rather than disagreements: `read_parquet` and `range` are sources with no frame behind them, `VALUES` is a source with no table at all, a window is the one clause that is neither elementwise nor a fold, `UNNEST` multiplies rows, and a recursive CTE is a fixed point. There is no way to write any of the five with the nine, which is why they are nodes and not lowerings.

Three smaller gaps sit inside the nine rather than beside them. `SetOp` is `union` widened to carry `EXCEPT` and `INTERSECT` and the `BY NAME` variant. The join kinds have to grow semi, anti and mark before decorrelation has anywhere to land. And `DependentJoin` is a node the binder emits and the optimizer must remove, so it is in the plan for one pass and then never again.

The rule stays what section 6 says. Each of these is a gap to close in `firepanda/plan/` where the dataframe front end can reach it too, and not a node `firepanda/sql/` builds on the side.

### 1.2 Two type sets, and which one the plan holds

`firepanda/sql/types.mojo` is DuckDB's type set and `firepanda/dtype/logical.mojo` is the engine's, and they do not line up: DuckDB has `DECIMAL(p, s)` and `HUGEINT` and the engine has neither. That is a real problem for this stage and not a naming one, because section 12 requires constant folding to answer `1.1 + 2.2` with exactly `3.3`, and a plan that can only hold a double cannot.

Until the plan carries an exact decimal, `firepanda/sql/plan.mojo` refuses a decimal literal by name rather than lowering it to a double. A refusal is visible in `pixi run sql-support` and in the conformance harness, and a double that answers `3.3000000000000003` is not visible anywhere. Adding `DECIMAL` and `INT128` to `LogicalType` is the work that removes the refusal, and it is engine work rather than SQL work for the same reason as section 1.1.

## 2. The shape

```
struct LogicalPlan:
    var nodes: List[PlanNode]     # arena, index referenced
    var exprs: List[BoundExpr]    # arena, index referenced
    var root: UInt32
```

Same arena discipline as the AST, for the same reason, plus one that is specific to this layer. Optimizer passes rewrite trees, and a rewrite over indices is a cheap structural edit while a rewrite over owning pointers is a fight with Mojo's ownership model on every pass.

Every node carries the schema it produces, meaning column names and types, computed at construction and validated in debug builds. A pass that produces a node whose schema disagrees with its children is caught at the pass boundary, not three stages later as a wrong answer.

## 3. The operators

Fourteen, which is the number that covers the whole `SELECT` surface without inventing anything.

| node | produces |
| --- | --- |
| `Get` | scan of a registered frame, with projection and filter pushdown slots |
| `TableFunction` | `read_parquet`, `range` and `unnest` sources |
| `Values` | inline constant rows |
| `Projection` | expressions over the child's schema |
| `Filter` | boolean predicate |
| `Aggregate` | grouping keys, aggregate expressions, grouping set masks |
| `Window` | window expressions with partition, order and frame |
| `Join` | inner, left, right, full, semi, anti, cross, positional and asof, plus mark |
| `Order` | sort keys with direction and null placement |
| `Limit` | offset and count, possibly expressions |
| `Distinct` | with an optional `ON` key list |
| `SetOp` | union, except and intersect, all or distinct, plus the by name variant |
| `Unnest` | list expansion, row multiplying |
| `RecursiveCTE` | fixed point over an anchor and a recursive term |

Everything in the dialect lowers to these. `QUALIFY` is a `Filter` above a `Window`. `GROUPING SETS`, `CUBE` and `ROLLUP` are masks on one `Aggregate` and not a `SetOp` of several. `PIVOT` is an `Aggregate` with a generated grouping set and conditional aggregates. `DISTINCT ON` is `Distinct` with a key list under an `Order`. Semi, anti and mark joins are join types rather than subquery nodes, which is the whole point of decorrelation.

Two node types exist for the optimizer rather than for the binder. `Mark` join produces a boolean column rather than filtering, and is what `IN` and `EXISTS` become. `DependentJoin` is the temporary node a correlated subquery binds to before decorrelation removes it.

## 4. Expressions

```
BoundExpr:  ColumnRef(binding, column) | Constant(value)
          | Function(overload, args)   | Cast(target, try)
          | Case(when_then, else)      | Conjunction(and/or, args)
          | Comparison(op, l, r)       | Operator(op, args)
          | Aggregate(overload, args, distinct, filter, order)
          | Window(overload, args, partition, order, frame)
          | Subquery(kind, plan, correlated_columns)
          | Lambda(params, body)       | Parameter(index)
```

`ColumnRef` is `(binding_index, column_index)`. No names anywhere after binding. This is the property that makes rewrites safe, and it is worth restating because it is easy to break: a pass may duplicate, reorder, push down or hoist a `Projection` without any risk of changing what a column means, because there is no scope to reinterpret it in.

Constants carry their exact type from document 06, so a `DECIMAL(3,1)` literal is a decimal in the plan and stays one through constant folding.

## 5. Decorrelation

The single algorithmically interesting thing in this layer, and the one that separates an engine from a query loop.

A correlated subquery evaluated naively is one execution per outer row. On TPC-H q17 and q20 at SF10 that is millions of executions and it turns a two second query into an unfinished one. The fix, from Neumann and Kemper's "Unnesting Arbitrary Queries", is a rewrite that makes it a single join:

1. The binder emits a `DependentJoin` with the outer columns it references made explicit, per document 05.
2. Push the dependent join down through the subquery's plan, one node type at a time, until it reaches the point where the correlated columns are actually used.
3. At that point it becomes an ordinary join against the distinct set of outer correlation values.
4. Assert that no `DependentJoin` survives.

That last step is the whole discipline. A leftover `DependentJoin` means a fallback to per row execution, and a fallback that works but is a thousand times slower is how a benchmark quietly fails. It is an assertion in debug builds and a logged plan warning in release.

The subquery kinds each get a rewrite. Scalar becomes a left join with a single row check, `EXISTS` becomes a semi join, `NOT EXISTS` an anti join, `IN` a mark join, and `NOT IN` a null aware anti join, because of the three valued semantics measured in document 06. A plain anti join for `NOT IN` is the classic wrong answer and it is silent.

## 6. The dataframe surface uses this too

The rule from document 02, restated as an obligation on this document: no node may have only a SQL constructor.

L3's eager `DataFrame` becomes a facade that builds a one node at a time plan and executes it, and a `LazyFrame` becomes the same builder without the execute. `df.filter(...).groupby(...).agg(...)` builds `Aggregate(Filter(Get))`, which is what `SELECT ... FROM df WHERE ... GROUP BY ...` builds, and document 11's plan equality test asserts exactly that for the twenty two TPC-H queries.

This is a real constraint on the API design and not a slogan. It means `df.query("a > 1")`, in document 10, is a parser producing a `Filter`, and it means an eager frame operation that cannot be expressed as a node is a gap to close in the plan rather than a special case to keep on the side.

## 7. Serialization and printing

Every plan prints, in two forms: a compact tree for `EXPLAIN`, and a JSON form for tests and tooling. Both round trip, because the JSON parses back to an identical plan, and that is what makes optimizer passes unit testable by writing the input and expected output plans as text rather than as construction code.

The `EXPLAIN` output deliberately mirrors DuckDB's shape, and `EXPLAIN ANALYZE` adds per operator row counts and timings from the execution layer. Not for compatibility, since nothing in the corpus checks plan text, but because a user comparing firepanda to DuckDB on a slow query wants to diff two plans, and two different notations make that a research project.

## 8. What the plan does not contain

**No physical choices.** No join algorithm, no build side, no chunk size, no parallelism degree. Those are document 09.

**No statistics.** Cardinality estimates live beside the plan in the optimizer's own structures and not on the nodes, so that a plan is comparable by structure across runs. Document 11's equality test depends on this.

**No engine specific hints.** Nothing that says use the hash join. If a user needs that, it is a setting, and settings are session state rather than plan content.

**No unbound names.** If a plan contains a string that has to be resolved later, the binder did not finish its job.

## 9. The optimizer, and its shape

Plan in, better plan out, same answer. This is where the query performance axis in document 01 is actually won or lost, because on TPC-H the difference between a good plan and a naive one is two orders of magnitude and no amount of kernel tuning closes it.

It is also the place where a project like this most often overreaches, so the rest of this document is ordered by return on effort, and it names what waits.

A fixed sequence of passes over the plan. Not a cost based rewrite search, not a Cascades style memo. DuckDB's optimizer is a pass list and it is competitive with anything, and a memo framework is a year of work whose payoff arrives only after the pass list is exhausted.

```
1  expression rewriter        constant folding, simplification, CNF
2  filter pushdown            with predicate inference
3  projection pushdown        column pruning to the scans
4  decorrelation              section 5, runs before join order
5  statistics propagation     from scans upward
6  join order                 DP over the join graph
7  build side selection       smaller side builds
8  common subexpression       within an expression tree
9  limit pushdown             into order and scan
10 late materialization       defer wide payload columns
11 sampling, regex and like    specialization rewrites
```

Every pass takes a plan and returns a plan. Every pass is independently testable against a written input and expected output in the JSON form from section 7. Every pass can be disabled by name from a setting, which is not a convenience: it is how a wrong answer gets bisected to a pass in one minute rather than one day, and DuckDB has exactly this for exactly this reason.

## 10. Pushdown, which is most of the win

**Filter pushdown** moves predicates toward the scans, through projections, through joins on the appropriate side, and into the `Get` node's filter slot where the scan can apply it before materializing. On a Parquet scan it becomes a row group skip, which is the difference between reading the file and not.

The refinements that matter, in order:

**Predicate inference across equi joins.** `a.x = b.x AND a.x > 5` implies `b.x > 5`, and the derived predicate pushes to the other side. On TPC-H this is worth more than any single other rewrite because it turns a filter on one table into a filter on both.

**Conjunction splitting before pushdown.** A predicate is pushed conjunct by conjunct, because an `AND` that is pushed as a unit gets stuck at the first node that cannot take all of it.

**Null rejecting predicate detection**, which converts an outer join to an inner join when a predicate above it would discard the null extended rows. This unlocks join reordering that is otherwise blocked, and outer joins are common in generated SQL.

**Projection pushdown** prunes columns to the scan. On a wide frame or a Parquet file this is the largest single reduction in bytes touched, and it is nearly free to implement because the plan already carries schemas.

## 11. Join order

The pass with the highest variance in outcome. Get it wrong on TPC-H q9 or q21 and the query does not finish.

**Algorithm.** Extract the join graph, with relations as vertices and equi join conditions as edges, then run dynamic programming over connected subgraphs, in DuckDB's case a `DPhyp` style enumeration. Above a vertex threshold, fall back to a greedy heuristic. TPC-H's largest query joins eight relations, which DP handles in microseconds, and the threshold exists for generated SQL with thirty joins.

**Cardinality estimation.** DP is only as good as its estimates. The sources, in order of reliability: exact row counts, which we have because our inputs are in memory frames and which is a real advantage over a database estimating from a stale sample; per column minimum, maximum and null counts, computed on registration or lazily on first use; distinct count estimates via HyperLogLog for join key selectivity; and fixed selectivity constants for everything else, which is what every engine actually does for non equality predicates.

**The honest limitation.** Correlated predicates are estimated as independent and the estimate is therefore wrong, sometimes by orders of magnitude. DuckDB has samples and sketches, and we start with exact counts plus minimum and maximum, which is enough for TPC-H and not enough in general. Document 13 keeps the sketch question open.

**Build side selection** is separate and is nearly free: the smaller estimated side builds the hash table. Getting this backwards on a large against small join is a tenfold error, and it is one comparison.

## 12. The expression rewriter

Runs first and again after other passes, because pushdown and decorrelation both create foldable expressions.

Constant folding, with document 06's exact semantics, so folding `1.1 + 2.2` must produce `DECIMAL(3,1) 3.3` and not a double, or the optimizer changes the answer. Comparison simplification. Constant condition `CASE` elimination. `IN` over a short constant list becomes a chain of `OR`, and over a long one a hash set. `LIKE` with no wildcards becomes equality, with a trailing `%` becomes `starts_with`, and with leading and trailing becomes `contains`, all of which lower directly onto the kernels already in `firepanda/kernel/pattern.mojo`, where they are documented as LIKE lowerings. Regex becomes `LIKE` where the pattern is anchored and literal. Common subexpression elimination within a tree means `x*2` appearing three times is computed once.

Two rules on this pass. It may never change a type, because a rewrite that produces a different `typeof()` is a compatibility bug under document 01. And it may never turn a raising expression into a non raising one or the reverse, because folding `1/0` at plan time when the branch would never have executed is a wrong answer, so folding stops at anything that can raise under document 06's overflow rules unless the operands are known safe.

## 13. Late materialization

Worth its own section because it is the pass with the best ratio of value to complexity for a dataframe engine specifically, and because Polars and DuckDB both arrived at it independently.

Sort, join and top-N on a wide frame move payload columns they never inspect. The rewrite carries a row identifier through the operator and joins the payload back afterwards. On `ORDER BY x LIMIT 10` over a fifty column frame it is the difference between sorting fifty columns and sorting one plus a gather of ten rows.

Our version is stronger than a database's, and this is one of the few places where the structural advantage from document 01 shows up in the optimizer rather than at the boundary. Our inputs are frames that are already in memory and already columnar, so a gather the payload later step is a real gather against real memory rather than a re-read.

## 14. What waits

**A cost model with sketches.** Exact counts plus minimum and maximum first. Sampling and HyperLogLog when a measured query picks the wrong plan and the estimate is why.

**Subquery caching and materialized CTE cost decisions.** Inline by default, honour the explicit `MATERIALIZED` hints, and revisit when a benchmark shows a repeated CTE dominating.

**Adaptive reordering of filter conjuncts at run time.** DuckDB does this in the expression executor and it is a genuine win on unselective first predicates. It belongs in document 09 rather than here, and it is post 1.0.

**Semi join reduction, bloom filters and sideways information passing.** Real wins on star schemas, and none of them matters until the pass list above is done and measured.

**Any form of plan caching beyond the prepared statement cache** in document 10.

## 15. How the optimizer is verified

Three layers, because the optimizer made it faster is not a correctness argument.

**Pass unit tests.** Input plan and expected output plan, both as JSON text. Cheap to write, cheap to read in a diff, and they document the pass better than prose.

**Optimizer off equivalence.** Every query in the conformance corpus runs with all passes disabled and with all passes enabled, and the results must be identical including row order where the query specifies one. This is the test that catches a pushdown through a node that does not preserve the predicate's meaning, and it is the single highest value test in this document.

**Plan stability.** The twenty two TPC-H plans are snapshotted. A change to a pass that changes a plan shows up in a diff and has to be justified in the commit, which is what stops a plausible looking rewrite from silently regressing q9 while fixing q3.
