# Cardinality and cost

How much the planner needs to know about the data, how wrong that knowledge is in every system, and how little of it firepanda is going to build.

## What the estimate is for

Two consumers. Join ordering needs to compare two candidate subplans, and operator selection needs to choose between implementations that win at different sizes. The second one is real for us today and the first one is document 03, which is third in line.

## The four families

**Histograms.** Accurate for a single column, and they do not capture correlation between columns. A multidimensional histogram grows storage dramatically with the number of dimensions, and nothing about a histogram carries across a join, which is where the errors that matter come from.

**Sampling.** Detects arbitrary correlations among common values, which histograms cannot. Sensitive to skew and to sparsity: once a chain of joins has cut the cardinality down, few sampled tuples survive, and the estimate for the next join is built on almost nothing.

**Sketches.** Space efficient and scalable. Good for distinct counts and for equi join cardinality. AGMS and its Fast variant are the classical answers for multi way join estimation, and bound sketches give a provable upper bound rather than a point estimate. The catch stated in the literature is that building sketches online does not scale.

**Learned models.** Effective on single table estimation, awkward across joins, and they need training. Not appropriate for a library that has to be correct on the first query it ever sees on a user's data.

## How wrong they are

Leis and colleagues measured this on the Join Order Benchmark, which is real IMDB data chosen so that correlations are the norm. Estimates for joins of three or more relations are routinely wrong by orders of magnitude in every system tested. They also showed the cost model matters far less than the estimates do, which is the reason nobody should build an elaborate cost model early.

## The contrarian thread, and why it is our thread

There is a recent line of work observing that modern main memory analytical systems, DuckDB among them, operate with limited or no cardinality estimation and remain competitive. The explanation is that vectorized main memory execution has a flat enough cost curve that a mediocre plan costs a factor rather than a catastrophe, and that runtime techniques absorb what a bad plan would otherwise cost.

Document 04 is the strongest form of that argument. With robust predicate transfer in place the ratio between the worst and best random join order on an acyclic query is 1.6. A cardinality estimator exists to close a gap that predicate transfer has already closed.

## What firepanda actually builds

**Exact row counts.** We have them. Every frame knows its height and every scan of an in memory frame knows exactly how many rows it will produce. This alone is enough for greedy join ordering, for build side selection, and for RPT's spanning tree weights.

**Exact distinct counts where they are already computed.** `factorize` produces a group count as a side effect of doing its job. When an operator has already factorized a column, cache the count on the column. This is free information we currently throw away, and it is what operator selection needs.

**Selectivity of a pushed predicate, after the fact.** Once a filter has run, its output row count is known exactly and every decision downstream of it can use the true number rather than an estimate. In an eager library that runs the plan node by node this is not a small point: for most of a plan, the cardinality is not an estimate at all, it is a measurement. That is a real structural advantage over a system that must plan the whole query before running any of it, and it is worth designing for rather than designing around.

**Nothing else.** No histograms, no samples, no sketches for estimation, no model. If a decision needs a number we do not have, the decision is made at runtime after the number exists, or it is made by a rule that does not need the number.

## Where an estimate is unavoidable

Join output size, before the join runs. Two exact input counts and no idea how the keys correlate.

The rule for that one case: assume the join is a foreign key join, so the output is the size of the larger side. That is correct for every join in TPC-H and for the overwhelming majority of joins a dataframe user writes, because that is what joining a fact table to a dimension is. It is wrong for a many to many join, and the way it is wrong is that it underestimates, which is the dangerous direction. So pair it with the runtime bail out in document 06: if a join's actual output exceeds its estimate by a large factor, the estimate was wrong and the remaining decisions that depended on it should be revisited rather than trusted.

## The cost model

A cost model turns cardinalities into a number to compare plans by. Given how bad the inputs are, keep it as simple as it can be.

Rows moved. One unit per row per column materialized, with a text column counting its view plus its bytes. That is a memory traffic model and memory traffic is genuinely what our execution costs, which we measured directly: a 48 megabyte column copy runs at about 29 gigabytes a second on the 13900K, and the gap between firepanda and Polars on TPC-H q6 is that we move about 470 megabytes where Polars streams about 120.

Nothing about CPU, nothing about cache levels, no calibration constants. When a decision comes out close under this model, the two plans genuinely are close and either is fine.

## What we should take from this document

Build no estimator. Use exact counts, cache distinct counts we already compute, and measure rather than estimate wherever the plan runs node by node.

The one estimate we cannot avoid is join output size, and the rule is the size of the larger side, backed by a runtime check for when that is wrong.

The cost model is rows moved, and that is the whole model.

Sources: Lan, Bao and Peng, "A Survey on Advancing the DBMS Query Optimizer", arXiv:2101.01507. Leis et al., "How Good Are Query Optimizers, Really?", PVLDB 2015. Datta and Rusu, "Analyzing Query Optimizer Performance in the Presence and Absence of Cardinality Estimates", arXiv:2311.17293. Cormode, Garofalakis, Haas and Jermaine, "Synopses for Massive Data".
