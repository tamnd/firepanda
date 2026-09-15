# The pass pipeline

A logical plan goes in, a logical plan comes out, and in between is a fixed sequence of rewrites. DuckDB's is a hardcoded order with no search in it except join ordering. Polars' is roughly the same list, arrived at independently, which is a decent signal that the list is right.

What follows is the list, in the order firepanda should run it, with what each pass is worth. Where a number is attached it is ours, measured by doing the rewrite by hand in the TPC-H driver at sf1 on a 13900K.

## 1. Expression simplification

Constant folding, arithmetic simplification, comparison simplification, moving constants to one side of a comparison, collapsing nested conjunctions and disjunctions. Applied repeatedly until nothing changes.

Cheap and unglamorous. The one that matters is folding, because date arithmetic in a predicate is otherwise evaluated per row. Also here: an `AND` of two comparisons on the same column with the same direction collapses to one, and a comparison against a value outside the column's type range collapses to a constant.

## 2. Type coercion

Insert casts explicitly as plan nodes rather than letting kernels promote implicitly. Polars does this and the reason is not tidiness: an implicit promotion inside a kernel is invisible to every later pass, so a pass that wants to know whether a predicate can be pushed into a Parquet reader cannot tell whether the comparison is against the column's own type or against something that will be widened first.

## 3. Projection pushdown

Columns nothing reads are never read. In a column store this is the difference between reading three columns and reading forty, and it is the pass users notice first because it turns a ten second Parquet read into a one second one.

Ours, by hand: the first version of the TPC-H driver filtered all sixteen columns of `lineitem` to produce answers that read two, and carried every column of both sides of a join through the probe. Introducing `_keep`, which filters and projects in one step, and naming the needed columns before each join rather than after, is most of the distance from 6.671 seconds to about 2 seconds across the twenty two queries. It is the single largest pass by a wide margin.

## 4. Predicate pushdown

A filter moves toward the scan until it cannot go further, so rows are eliminated before they are joined, aggregated or projected. The rule is the table set analysis: a predicate can be pushed into a subtree if that subtree provides every column it references.

Three things happen at the bottom. A Parquet scan turns a pushed predicate into row group statistics checks and skips whole row groups without decoding them. A Hive partitioned dataset turns it into directory pruning and never opens the file. An in memory scan turns it into an ordinary filter, which is what firepanda has today.

Which sides a join passes a predicate into depends on its kind, and the argument is about what happens to a row the predicate drops. An inner join passes into both, because a row dropped on either side takes with it every pair it would have made. A left, semi, anti or mark join passes into the left side only: all four hand out left rows, so a left row dropped below drops exactly the output rows the predicate would have dropped above, while a right row dropped below a left join gets its partner null extended rather than removed. A right or a full outer join passes into neither, which is the left join's argument in both directions at once. A cross join passes into the left side, and the right side is left alone for a reason about firepanda's operator rather than about the answer: the lowering pairs a whole frame against a right side of a single row, and a predicate pushed into that side can leave it with no row at all.

Which side provides a column is a question about the column and not only about its name. `FROM nation n1, nation n2` puts an `n_nationkey` on each side of a join, so a search by name finds the name on both and can only answer that it does not know, and a query that qualified every mention of it gets nothing pushed anywhere. The qualifier is not lost. The binder writes the relation it picked onto the reference and binding writes the relation each column came from onto the node, so the two can be compared and `n1.n_nationkey` placed on the side it was written about. A column that no single relation produced stays a maybe, which keeps this no stricter than the search by name. TPC-H q7 is the case: its `WHERE` pairs `s_nationkey` with `n1.n_nationkey` and `c_nationkey` with `n2.n_nationkey`, and neither equality could become a join key until the qualifiers were read.

A predicate splits at its `and` nodes and not at its `or` nodes, but a disjunction is still worth reading. A condition written into every branch of an `or` holds whenever the `or` does, so it is carried alongside it as a conjunct of its own and can then be pushed like any other. `(a AND b) OR (a AND c)` carries `a`. The `or` is left as it was rather than having `a` taken out of it, because the copy left behind runs on rows a pushed `a` has already thinned and the same rows come out either way. TPC-H q19 is the case: its whole `WHERE` is one disjunction of three branches, each of which writes out `p_partkey = l_partkey` for itself, so nothing reaches the product under it until the shared equality is carried out.

An equality carried over a cross join is a join condition rather than a filter, and this is the pass that puts it back where it belongs. `FROM a, b WHERE a.x = b.y` and `FROM a JOIN b ON a.x = b.y` are the same query and the first one lowers to a product, which is the one shape that cannot be run at all once either table is real, so this rewrite is what makes the older spelling of a join work rather than what makes it faster. Both halves have to be a plain column and each has to belong to one side, which is the same table set analysis the rest of the pass does.

Transitive predicates are the part that is easy to miss and is worth the most. If the plan has `a.x = b.x` and a filter on `a.x`, the same filter applies to `b.x` and the user never wrote it. This is the first step toward `04-predicate-transfer.md` and the two should be built as one thing.

Ours, by hand: q19's three disjuncts between them only ever ask for one of three brands, one of twelve containers, a size up to fifteen and a quantity up to thirty. Running those four on `part` and on `lineitem` before the join, rather than running the disjunction over every shipped line, took q19 from 83 to 70 milliseconds, and combined with projecting `part` from nine columns to four it went from 0.087 to 0.070 in the four engine run. q21 was filtering after its joins rather than before them and went from 285 to 169 milliseconds.

## 5. Common subplan and subexpression elimination

If the same subtree appears twice it is computed once. Polars turns this on by default in `collect()`.

For a dataframe library this matters more than it does for SQL, because a user writing Python naturally repeats themselves: `df.filter(cond).select(a)` and `df.filter(cond).select(b)` in adjacent lines is a common subplan and a person would never write the SQL that way.

## 6. Projection merging

Adjacent `with_columns` calls become one projection node. Polars calls this cluster with columns. Ten chained calls become one pass over the data instead of ten.

Ours, by hand, and it is the one place where a rewrite and a data structure decision met. `with_column` returns a new frame and therefore deep copies the old one. TPC-H q1 filters six columns of `lineitem` down to five point nine million rows and then adds two computed expressions, and written as two `with_column` calls that is two deep copies, around seven hundred megabytes moved in order to write two new columns. Ninety three milliseconds became thirty seven with the copies removed. A projection merging pass makes this structural rather than a thing a caller has to remember, because one projection node produces one frame.

## 7. Slice pushdown

A `head(10)` at the end tells the scan to stop early. Eager libraries cannot do this at all and it is the difference between `head()` on a lazy scan of a large file being instant and being a full read.

Also here: a limit above a sort becomes a top n operator, which is a bounded heap rather than a full sort. firepanda already has `group_nlargest` and a top n route in the sort kernels, so this pass is connecting things that exist.

## 8. Empty and constant pruning

A provably false filter collapses its subtree to an empty frame with the right schema. A join against an empty side collapses by the same rule, respecting the join kind. Rare in benchmarks, common in generated queries, and cheap.

## 9. Specific rewrites

The bag of tricks. `IN` against a large constant list becomes a join against a materialized list, as a mark join or an inner join depending on context. A semi join whose right side is distinct on the key becomes an inner join. An aggregate over a join whose other side contributes nothing to the result pushes below the join. A count over a scan with no predicate reads the row group metadata and nothing else.

Ours, by hand, and the largest single one: q7 joined the full `supplier`, `orders` and `customer` tables into a shipping window of four and a half million lines and only then narrowed to the two nations the query asks about. Two nations out of twenty five is eight per cent of the suppliers, so doing the nation joins on `supplier` and `customer` first drops the line count to a third of a million before anything else runs. 155 milliseconds became 76. That one is join ordering, which is document 03, and it is here as well because from the rewrite side it looks like pushing a filter through two joins.

## 10. Join ordering

The only pass that is a search rather than a rewrite. Document 03.

## 11. Build side selection

Once the order is fixed, each hash join still chooses which side to build the table from. DuckDB has this as a separate pass from join ordering, `BUILD_SIDE_PROBE_SIDE`, and the two can be disabled independently.

The rule is to build from the smaller side, because the hash table is the thing that has to be resident. `firepanda/join/pairs.mojo` builds from the right side because the parameter is named right. That does nothing on the db-benchmark joins, which all put the larger table on the left and so happen to want the right side bucketed anyway, and it is a large win the first time a user writes the tables in the other order. It is also about twenty lines: compare the lengths, build from the smaller, flip the output pair order to compensate.

## What order, and why fixed

Fixed, not searched. There is no cost model choosing between rewrite orders in either DuckDB or Polars, and building one would be the second hardest thing in this folder for a benefit nobody has demonstrated.

The order above is not arbitrary. Simplification runs first so the later passes see the simplest form of every expression. Projection pushdown runs before predicate pushdown so that a predicate arriving at a scan sees a column list that already exists. Elimination runs after both so that subtrees that were made identical by pushdown are recognized as identical. Join ordering runs late because it wants the cardinalities that the pushed filters imply, and build side selection runs after it because it is a decision about a join whose position is already fixed.

Run the whole pipeline twice if the second run changes anything, up to a small bound. DuckDB applies expression rewriting repeatedly for the same reason and it costs microseconds on a plan of tens of nodes.

## What we should take from this document

The list, in this order, as the definition of done for the logical optimizer.

Projection pushdown first, because it is the biggest single pass and it is the one that needs the least machinery.

Predicate pushdown with transitive predicates second, and built with document 04 in mind rather than as a separate thing.

Projection merging early even though it looks minor, because it removes a class of copy that a caller currently has to work around by hand.
