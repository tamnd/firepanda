# Runtime filters

Decisions made after the query has started, using information that did not exist at planning time. This is the half of modern query optimization that has nothing to do with the optimizer, and it is where a lot of the recent engineering effort in every engine has gone.

## The idea

A hash join builds a table from one side before it probes with the other. At the moment the build finishes, facts exist that nobody knew when the plan was made: how many distinct keys there are, what the minimum and maximum are, and which keys are present. Any of those can be turned into a filter and handed to the probe side's scan, which then never reads rows that cannot match.

This is sideways information passing, and it is different from predicate pushdown in one important way. Predicate pushdown moves something the user wrote. Sideways information passing invents a predicate out of the data.

## What DuckDB does

Join Filter Pushdown. When the build side of a hash join has a subset of the join keys, DuckDB tracks the minimum and maximum key and pushes a table filter into the probe side, eliminating keys outside that range. The example in their own writing is an orders table of a hundred million rows probing a filtered parts table of about twenty thousand: parts becomes the build side, orders becomes the probe side, and the min and max of the surviving part keys prunes orders before it is read.

The Parachute paper documents the behaviour more precisely as of v1.2 and later. In the general case DuckDB transfers the min and max of the key columns for coarse pruning. If there is exactly one distinct build side value it generates an equality predicate instead. If the number of distinct build keys is below a threshold, fifty by default, it pushes an IN list, which is used for partition level pruning but not row level filtering. Above that threshold it is min and max only.

Notably DuckDB does not natively implement Bloom filter joins. Researchers added one experimentally and got 1.26 times end to end on JOB, with an eight kilobyte filter sized to stay in L1, reusing the join's own 64 bit hashes to derive two bit positions.

## What everyone else does

Apache Spark has had Bloom filter joins for years. Apache DataFusion built dynamic filter infrastructure in 2025 and reported up to 25 times on some queries, with the remaining work being join kinds beyond inner and pushing whole hash tables to the scan rather than just min and max.

The DuckDB `robust` community extension is the interesting one because it sits between this document and document 04. It chooses among Bloom filters, min and max ranges and IN lists based on the build side's distinct count, and unlike native join filter pushdown, which forwards filters down a linear join spine with no backward propagation, it sees the whole join graph and runs a backward pass to shrink the early scans. It engages when there are at least two equality joins on an acyclic graph and passes single join queries through unchanged.

That is predicate transfer implemented as a runtime feature rather than a planning feature, and the fact that it can be an extension rather than a core change is encouraging for us.

## The relationship to document 04

Predicate transfer decides at plan time which filters to build and in what order to move them, and applies them before any join runs. Runtime filtering builds one filter as a side effect of a join it was going to run anyway and applies it to the very next operator.

They overlap and they are not the same. Predicate transfer reaches relations that are several joins away, which a runtime filter cannot do because those joins have not started. A runtime filter is free, because the build side hash table was being built regardless, whereas predicate transfer pays for extra passes.

The right structure is to build both and have them share machinery. The filter representation, the sizing rule, and the code that applies a filter to a column are the same in both cases. What differs is who decides to build one and when it gets applied.

## The design for firepanda

**One filter type, chosen by distinct count.** Below about sixty four distinct build keys, an IN list, which is `firepanda/kernel/member.mojo` and which already has a linear route and a table route with measured thresholds. Above that, a blocked Bloom filter over the 64 bit hashes the join already computes. Always, in addition, the minimum and maximum, because they cost two comparisons to maintain during the build and they are what allows a whole morsel to be skipped rather than each row tested.

**Sized from the exact build count.** We know it. One block, sized to stay inside L1, and multiple blocks selected by the high bits of the hash when the key count needs them.

**Applied as a mask.** A filter produces a boolean mask over the probe column, which the now parallel filter kernel compacts. That is an operation that already exists and is already fast.

**Applied inside the probe when the probe is the next thing that runs.** Producing a mask and compacting moves bytes; testing the filter inside the probe loop before the hash table lookup does not. Both forms are needed, because the first one is what lets a filter reach a scan two operators away and the second is what makes a filter free when the consumer is right there.

**Escape hatch.** Skip when the build side is not meaningfully smaller than the probe side, since the filter will not eliminate anything and the probe was going to happen anyway.

## The other runtime decisions

Runtime filtering is the largest of a family. The others worth naming:

**Adaptive operator selection.** Document 07 chooses between physical implementations by a rule over cardinalities. Several of those cardinalities are known exactly by the time the choice has to be made, because the input has already been computed. Make the choice as late as possible.

**Estimate bail out.** Document 05 admits one unavoidable estimate, the output size of a join. When the actual output exceeds the estimate by a large factor, the decisions downstream that depended on it are suspect. At minimum, do not compound it: re-decide the next operator from the measured number rather than from the estimated one.

**Skew detection.** A partitioned hash aggregate or join whose partitions come out badly unbalanced has a skewed key, and the response is to split the heavy partition further rather than to let one worker carry the query. We already see this shape in `firepanda/hash/factorize.mojo`, which has a crowded share constant and a partitioned route, so the detection exists in one place and could become a general property.

## What we should take from this document

Build the filter machinery once and use it for both document 04 and this one.

Min and max always, because they are two comparisons and they allow morsel level skipping rather than row level testing.

IN list below about sixty four distinct keys, Bloom above, both over hashes the join already computes.

Apply inside the probe when the consumer is adjacent, as a mask when it is not.

Sources: "Optimizers: The Low-Key MVP", DuckDB blog, November 2024. "Parachute: Single-Pass Bi-Directional Information Passing", arXiv:2506.13670. "Dynamic Filters: Passing Information Between Operators During Execution", Apache DataFusion blog, September 2025. DuckDB community extension `robust`.
