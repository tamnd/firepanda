# Operator selection

One logical operation, several implementations, and a rule for choosing. This is the part of a planner firepanda can build first, because the implementations already exist and all that is missing is the choosing.

It is also the part where we have the most evidence, because every one of the following was found by measuring two implementations of the same query against each other on real data.

## The choices we already have and do not make

**Correlated aggregate: broadcast or group and join.**

`group_broadcast` computes an aggregate per group and returns it aligned to the input rows, in one pass. `group_by` followed by a semi join computes the same thing in two passes with a join in between. Which is faster depends entirely on the number of groups.

Measured at sf1: TPC-H q17 has about two hundred thousand groups and the broadcast is 0.011 seconds against DuckDB's 0.104, a factor of eight. q18 has about one and a half million groups, one per order, and there the broadcast is 0.146 against 0.048 for group by plus a semi join, a factor of three the other way.

Same logical operation, same engine, and the right choice differs by a factor of three in one direction and eight in the other. The driver currently has the choice hardcoded per query with a docstring explaining which is which, which is exactly the thing a planner should be doing.

The rule: broadcast when the group count is small relative to the row count, group and join when it approaches it. The crossover is somewhere between two hundred thousand and one and a half million groups on this data and needs measuring properly. The group count is not known before the operation runs, but it is estimable from the key's distinct count, which `factorize` computes and which document 05 says to cache.

**Join build side.**

`firepanda/join/pairs.mojo` builds from the right side because the parameter is called right. It should build from the smaller side, because the hash table is what has to be resident. Compare the two lengths, build from the smaller, flip the output pair order to compensate. DuckDB has this as a distinct optimizer pass from join ordering.

**Dense integer keys.**

DuckDB's perfect hash join detects a dense integer key at runtime, after the minimum and maximum are known, and skips hashing entirely: the key minus the minimum is the slot. We have half of this already, since PR #75 detects a unique build side and builds a direct code to row table, but it still hashes to get the code first.

This matters more than it sounds. On db-benchmark j4 the profile is `frame_join` 635 milliseconds, of which `join_indices` is 590, of which `group_ordinals` is 395. That 395 exists because we factorize the key into codes as a separate whole column pass before joining. Detecting a dense integer key and indexing directly removes most of it, for the case that is most joins in practice, because most join keys are identifiers.

**Sorted input.**

Polars 2.0 ships a streaming sort merge join, chosen when the key is known sorted, and a `SortedGroupBy` chosen the same way, which needs no hash table at all. The critical detail from their own documentation is that known to be sorted means known to the optimizer, not known to you, and `set_sorted` is a declaration rather than a sort.

We have no sortedness flag on a column. Adding one, set by `sort_values` and by readers that know, is cheap and it makes a group by after a sort nearly free, and it is the precondition for a merge join.

**Top n.**

A limit above a sort is a bounded heap rather than a full sort. We already have `group_nlargest` and a top n route in the sort kernels. What is missing is the plan node that recognizes the pattern.

**Distinct.**

`drop_duplicates` on the whole row against `drop_duplicates` on a subset are different problems, and we measured them: at a key of a thousand values we are 0.618 nanoseconds a row against Polars' 4.281, at a key of a million we are 17.142 against 23.776, and on a whole row we are 35.535 against 31.593. The whole row case is the one we lose, and it loses for the same reason the wide group by loses, which is the high cardinality hashed factorize at 12.739 nanoseconds a row.

## The rule for choosing

Late, and from measurements rather than estimates wherever possible.

In an engine that runs a plan node by node, the input to the node being chosen has usually already been computed, so its row count is exact. The only genuinely unknown quantity at choice time is the output, and for the choices above the relevant one is a distinct count.

So the mechanism is:

**Cache what we compute.** A distinct count, a sortedness flag, a minimum and maximum, an all valid flag. These are things kernels already discover and then discard. Attach them to the column and invalidate them on write.

**Decide at the operator, not in the planner.** The planner emits a logical node. The physical choice happens when the node runs and the input is in hand. This is not how a database does it and it is the right answer for a library, because it converts most of the estimation problem into a measurement problem.

**Write the crossover down and measure it.** Every threshold in firepanda's kernels has a docstring recording how it was measured, and there are a lot of them: `LINEAR_MAX` at 32, `TEXT_LINEAR_MAX` at 2, `PARALLEL_TAKE_ROWS` and `PARALLEL_FILTER_ROWS` at 65536, `DIRECT_LIMIT` at 65536, and a dozen more in `firepanda/hash/factorize.mojo`. That discipline is right and the operator level choices should follow it.

## The threshold problem

There are now enough of these constants that they are worth naming as a design debt rather than a set of individually reasonable decisions.

Every one is a row count at which one implementation overtakes another, measured on one machine, on one shape of data, at one point in time. They are not portable across machines with different core counts or cache sizes, and several were measured on a 13900K with thirty two logical cores and a thirty six megabyte L3, which is not a typical user's machine.

The eventual answer is a cost model that produces these numbers from machine properties rather than from measurement, which document 05 says should be rows moved and nothing else. That is a later project. The interim answer is that a threshold should be expressed in terms of something that varies with the machine where one is available, worker count and cache size being the two we can query, rather than as a bare number.

## What we should take from this document

Build side by size, now, twenty lines.

The broadcast versus group and join choice, next, because we have both implementations and the gap is a factor of three to eight.

Column metadata: distinct count, sortedness, min and max, all valid. Cache what kernels already compute, invalidate on write. Everything else in this document depends on it.

Dense integer key detection at the join, which is the largest single item here by measured time.

Sortedness flag, then sorted group by, then merge join, in that order, because each needs the one before it.
