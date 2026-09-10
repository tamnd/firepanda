# Predicate transfer

This is the strategy document. If only one thing in this folder gets built, build this.

## The idea in one paragraph

Before running any join, filter every table by what every other table in the query implies about it. A predicate on `nation` implies a set of suppliers, which implies a set of line items, which implies a set of orders, which implies a set of customers. Push all of that around the join graph first, as cheap approximate semi joins, and only then run the joins. Every relation arrives at its join already reduced to roughly the rows that will survive, so the intermediate results are small no matter which order the joins run in.

## Where it comes from

**Yannakakis, 1981.** For an acyclic query, a bottom up pass of semi joins followed by a top down pass fully reduces every relation to exactly the tuples that contribute to the output. The join then costs `O(N + OUT)`, linear in the input plus the output, which is instance optimal. This is a forty five year old theorem that essentially no production system implements.

The reason nobody implements it is that a semi join is a hash table build and probe, and doing two extra passes of those over every relation costs more than it saves in the common case. The theory is right and the constant factor kills it.

**Predicate Transfer, Yang, Zhao, Yu and Koutris, CIDR 2024.** Replace the exact semi joins with Bloom filters. A Bloom filter is a few kilobytes and a probe is a couple of bit tests, so the constant factor collapses. It is approximate, so a relation is not fully reduced, but it is reduced by most of what the semi join would have removed. The paper reports 3.1 times over Bloom join on TPC-H, where Bloom join means filtering within a single join operation rather than across the whole graph.

Mechanically: build a predicate transfer graph at planning time, build local filters on the leaf relations, then transfer along the graph in topological order, forward and then backward.

**Robust Predicate Transfer, Zhao, Su, Yang, Yu, Koutris and Zhang, SIGMOD 2025.** Predicate Transfer was inspired by Yannakakis but does not inherit the guarantee, because it does not ensure full reduction. RPT adds two algorithms, LargestRoot and SafeSubjoin, that restore it for acyclic queries. LargestRoot builds a maximum spanning tree over the weighted join graph, rooted at the largest relation rather than chosen by a small to large heuristic, which is what guarantees full reduction in the transfer phase. It also handles cyclic queries, where no full reduction guarantee is available but every predicate still reaches every relation at least once, and that works well in practice.

The headline result is the one that matters for us. Integrated with DuckDB and evaluated on every query in TPC-H, JOB and TPC-DS: the largest ratio between the slowest and fastest execution over random join orders of a single acyclic query drops to 1.6, and is close to 1 for most queries. End to end performance improves by 1.5 times as a geometric mean per query.

Read that again. With predicate transfer in place, a randomly chosen join order is within 1.6 times of the best one. The forty year problem in document 03 stops being a problem you have to solve well.

**Yannakakis+, 2025**, and the column store engineering paper by Bekkers and colleagues in PVLDB 18(8), are the two follow ups worth tracking. Yannakakis+ maximizes the filtering power of a single semi join round by enumerating join trees, estimating intermediate sizes after a bottom up pass, and picking the tree that minimizes them. The column store paper is the one closest to our situation because it is about making this work in an engine shaped like ours rather than in a row store.

## Why this fits firepanda specifically

**It does not need good cardinality estimates.** The whole apparatus of document 05 becomes optional. RPT's spanning tree needs relation sizes, which we know exactly, and edge weights, which can be as crude as the smaller endpoint's row count.

**It is robust, and robustness is what a library needs more than peak performance.** A database serves queries a DBA tuned. A dataframe library serves whatever a user typed at two in the morning, and the failure mode that loses users is not being twenty per cent slower than Polars, it is being a hundred times slower on one query out of fifty. A 1.6 times worst case ratio across join orders is exactly the property that makes a library feel reliable.

**We already have every piece.** `firepanda/hash/table.mojo` is a hash table with a probe. `firepanda/kernel/member.mojo` is a set membership kernel with both a linear route and a table route. The join code builds and probes. A Bloom filter over the existing 64 bit hashes is a small amount of new code, not a new subsystem.

**We have already validated it by hand, twice, without knowing that is what we were doing.** q7's rewrite is a manual forward transfer: filter `nation`, transfer to `supplier` and `customer`, and only then meet `lineitem`. 155 milliseconds to 76. q19's rewrite is a manual local filter plus transfer: derive from the three disjuncts that only three brands, twelve containers, a size up to fifteen and a quantity up to thirty can ever match, and apply those to `part` and `lineitem` before the join. 83 to 70. Those are the two queries where a person looked at the join graph and pushed information around it, and they are the two largest join related wins we got.

## The design for firepanda

**The filter.** A blocked Bloom filter over the 64 bit hash the join already computes. The experimental DuckDB work quoted in the Parachute paper caps the filter at eight kilobytes so it stays in L1, derives two bit positions from the existing hash for `m = 2^16` and `k = 2`, and supports about five thousand distinct keys at a two per cent false positive rate, for 1.26 times end to end on JOB. Size the filter from the build side's row count rather than fixing it, since we know the count exactly, and keep the L1 residency rule as the upper bound on a single block.

**The graph.** Nodes are scans, edges are equality join conditions. Build it from the bound plan after predicate pushdown has run, because the local filters that seed the transfer are the pushed predicates.

**The schedule.** LargestRoot. Maximum spanning tree over the join graph weighted by the smaller endpoint's cardinality, rooted at the largest relation. Forward pass from leaves to root, backward pass from root to leaves. For a cyclic graph, take the spanning tree and accept the weaker guarantee.

**The application.** A transferred filter becomes a mask over the receiving scan, which is exactly the operation `is_in` already performs and which the newly parallel filter kernel then compacts. Nothing about the execution layer has to change.

**The escape hatch.** Skip the whole thing when the query has fewer than two equality joins, which is what the DuckDB `robust` community extension does, since there is nothing to transfer around a single join that the join itself will not do.

## What it costs

Two extra passes over each relation's key column, in exchange for every relation after the first being smaller. The papers say this is a win on essentially all of TPC-H, JOB and TPC-DS, and that is the strongest evidence available. Where it will not be a win is a two table join of two small tables, which is what the escape hatch is for.

There is one cost that the papers do not carry and we do. Our scans are in memory frames, not Parquet readers, so a transferred filter cannot prune a row group. It produces a mask and a compaction, and the compaction moves bytes. That makes the escape hatch matter more for us than for DuckDB, and it makes document 06's runtime variant, where the filter is applied inside the probe rather than as a separate pass, worth having as well.

## What we should take from this document

Build predicate transfer. It is the highest value item in this folder and it is the reason document 03 is short.

Use RPT's LargestRoot rather than PT's heuristic, because the extra code is small and the guarantee is the whole point.

Bloom filters over the hashes we already compute, sized from the build side count, one block resident in L1.

Do not build a cardinality estimator first. This works with the exact row counts we already have.

Sources: Yannakakis, "Algorithms for Acyclic Database Schemes", VLDB 1981. Yang, Zhao, Yu and Koutris, "Predicate Transfer: Efficient Pre-Filtering on Multi-Join Queries", CIDR 2024, arXiv:2307.15255. Zhao, Su, Yang, Yu, Koutris and Zhang, "Debunking the Myth of Join Ordering: Toward Robust SQL Analytics", SIGMOD 2025, arXiv:2502.15181. Wang, Chen, Dai, Yi, Li and Lin, "Yannakakis+: Practical Acyclic Query Evaluation with Theoretical Guarantees", PACMMOD 3(3), 2025. Bekkers, Neven, Vansummeren and Wang, "Instance-Optimal Acyclic Join Processing Without Regret", PVLDB 18(8), 2025. "Database Theory in Action: Yannakakis' Algorithm", ICDT 2026.
