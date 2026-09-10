# Join ordering

The classical hard problem, and the one we are deliberately not starting with. This document says what it is, what the best known answers are, and why the next document is where the effort should go instead.

## The problem

The joins in a query form a hypergraph. Nodes are relations, edges are join conditions, and an edge is a hyperedge when a condition touches more than two relations. The task is to find the order of joins with the lowest total intermediate cardinality, because the cost of a join plan is dominated by how many rows flow between the joins rather than by the joins themselves.

The search space is large. For `n` relations in a chain the number of left deep plans is `n!`, and allowing bushy plans makes it worse. Twelve relations is beyond enumeration by brute force and TPC-DS has queries with more than that.

## The algorithms

**DPccp**, Moerkotte and Neumann, enumerates connected subgraphs and their complementary pairs over a simple join graph, in an order that generates each pair exactly once. That last property is what makes dynamic programming over the graph tractable rather than exponential in a way you notice.

**DPhyp**, the same authors, "Dynamic Programming Strikes Back", generalizes it to hypergraphs so that non inner joins and conditions spanning more than two relations are handled. This is what DuckDB uses.

Above a size threshold the search space stops being enumerable and DuckDB falls back to a greedy algorithm. That threshold matters more than the algorithm does for our purposes, because most dataframe queries join fewer than ten relations and every algorithm is exact at that size.

**Left deep versus bushy.** Left deep plans keep one build side small and pipeline the rest, which suits a hash join engine. Bushy plans can be dramatically better on snowflake schemas where two independent selective subtrees should both be reduced before they meet. DuckDB searches bushy.

## What it needs, and why that is the problem

All of it runs on cardinality estimates. The estimate for a join of two subtrees comes from the estimates of the subtrees and an assumed correlation between the keys, and the errors compound multiplicatively as the plan gets deeper.

Leis and colleagues made this concrete in "How Good Are Query Optimizers, Really?" with the Join Order Benchmark, which is a set of queries over the real IMDB dataset chosen so that correlations between columns are the norm rather than the exception. The result that stuck is that estimates for joins of three or more relations are routinely wrong by orders of magnitude in every system tested, and that the cost model matters far less than the estimates do. A follow up from the same group made the argument that a simple cost model over good estimates beats a sophisticated one over bad estimates.

The 2025 framing in "Debunking the Myth of Join Ordering" is blunter still: despite decades of research and practice, modern query optimizers could still generate inferior join plans that are orders of magnitude slower than optimal.

## The contrarian thread

There is a line of recent work arguing that cardinality estimation matters less than the literature assumes for modern main memory analytical engines. DuckDB and similar systems operate with limited estimation and still perform competitively, which is examined in "Analyzing Query Optimizer Performance in the Presence and Absence of Cardinality Estimates".

The explanation is not that ordering does not matter. It is that vectorized main memory execution has a much flatter cost curve than a disk based system does, so a mediocre order costs a factor rather than a catastrophe, and that runtime techniques such as the ones in documents 04 and 06 absorb a lot of what a bad order would otherwise cost.

## Our own evidence

We reordered joins by hand in exactly one TPC-H query where it mattered a great deal and several where it did not.

q7 went from 155 to 76 milliseconds by doing the two nation joins on `supplier` and `customer` before either one met `lineitem`. Two nations out of twenty five is eight per cent of the suppliers, so the shipping window drops from four and a half million lines to a third of a million before anything else runs. That is a factor of two on one query out of twenty two, and it is the whole of what join reordering bought us across the set.

Compare that with projection pushdown, which was worth several seconds across the set, and predicate pushdown, which was worth several hundred milliseconds. Join ordering was third by a distance.

That ranking is specific to TPC-H, whose schema is a well behaved snowflake with declared keys and whose queries were written by people who knew the schema. It would not hold on JOB, which was constructed precisely to break it. It probably does hold for the queries a dataframe user writes, which are usually a fact table and a few dimensions.

## What firepanda does

Not DPhyp, not yet, and possibly not ever in the form DuckDB has it.

**Now.** Nothing. There is no plan to reorder.

**With the plan layer.** A greedy order: repeatedly join the pair whose result is estimated smallest, starting from the smallest relation. Greedy with even rough estimates captures the q7 case, because the difference between eight per cent of the suppliers and all of them is not a subtle estimation problem. Greedy is about fifty lines and it is the fallback DuckDB uses above its own threshold anyway.

**Then predicate transfer**, document 04, which is where the effort actually goes and which makes the order matter much less.

**Then, if it is still the bottleneck**, DPccp over the simple graph for queries below ten relations, with the greedy order as the fallback above that. DPhyp over the full hypergraph is the last step and should not be taken until a measured query needs it.

The order of that list is the point of this folder. Every engine got here by writing the search first and then spending a decade fixing the estimates. We have the advantage of reading the ending first.

## What we should take from this document

Join ordering is real and it is third in line.

Greedy is enough to capture the case we actually hit, and it is cheap.

Do not build a cardinality estimator in order to build a join orderer. Build document 04 instead, and revisit this when a query is slow for a reason document 04 did not fix.

Sources: Moerkotte and Neumann, "Dynamic Programming Strikes Back", SIGMOD 2008. Leis et al., "How Good Are Query Optimizers, Really?", PVLDB 2015. Zhao et al., "Debunking the Myth of Join Ordering: Toward Robust SQL Analytics", SIGMOD 2025, arXiv:2502.15181. Datta and Rusu, "Analyzing Query Optimizer Performance in the Presence and Absence of Cardinality Estimates", arXiv:2311.17293.
