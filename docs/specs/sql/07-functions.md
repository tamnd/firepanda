# Functions

The part of the work that is large rather than hard, and therefore the part most likely to be underestimated in a plan and then to consume a year. This document counts it, tiers it, and says what the ones outside the tier do.

## 1. The size of the catalog

Measured from `duckdb_functions()` on DuckDB 1.5.5:

| kind | overloads | distinct names |
| --- | --- | --- |
| scalar | 1,465 | 617 |
| aggregate | 1,177 | 88 |
| macro | 131 | 118 |
| table | 129 | 89 |
| pragma | 45 | 45 |
| table macro | 4 | 4 |
| total | 2,951 | 948 distinct overall |

The shape of that table is the useful part. There are 617 scalar names against 88 aggregate names, but the aggregates have almost as many overloads, at 1,177 across 88 names, which is thirteen per name, because every aggregate is instantiated for every numeric type and most also exist in `_if`, `_distinct`, ordered and windowed forms. Aggregates are few, deep and expensive per name. Scalars are many, shallow and cheap per name.

That asymmetry sets the strategy: write aggregates by hand and generate scalars.

By area, counting distinct names: list and array 115, datetime 34, JSON 29, string 20 by prefix and far more by meaning, struct 12, map 11, regexp 8, spatial 8, and everything else 711.

## 2. The registry

```
struct FunctionOverload:
    var name: UInt32              # interned
    var kind: FunctionKind        # scalar, aggregate, window capable, table
    var params: List[TypeId]      # or a variadic marker
    var varargs: TypeId
    var return_type: ReturnTypeRule   # fixed, or a function of the arguments
    var impl: UInt32              # index into the kernel table
```

Interned names, a flat table, and a lookup that groups overloads by name. The return type is a rule and not a type, because DuckDB's return types are computed. `sum(INTEGER)` is HUGEINT, decimal arithmetic derives precision and scale per document 06, and `list_value(...)` returns a list of the resolved element type. A table of fixed return types is wrong for a large fraction of the catalog and the wrongness is silent.

Registration is at build time via `comptime`, so the table is static data with no startup cost and no dictionary construction, which matters because document 01's latency budget is measured from a cold `sql()` call and a registry built at import time is a fixed tax on every process.

## 3. Overload resolution

DuckDB's rule, matched exactly because the corpus depends on it:

1. Collect all overloads with the name. An unknown name is an error, with edit distance suggestions over the catalog.
2. For each, check arity, honouring variadics and defaults.
3. Score each candidate. An exact type match is free, an implicit cast costs what that cast costs, and a cast that does not exist disqualifies the candidate outright.
4. Lowest total cost wins. A tie is an error listing the candidates that tied, and only those.

The cast costs are the whole of it and DuckDB publishes none of them, so firepanda measures them instead. `tools/gen_casts.py` reads which casts exist out of `can_cast_implicitly`, then puts every one, two and three argument tier 1 call over real typed columns to a live DuckDB and reads the answer back out of the plan and out of `typeof`. Each choice is an inequality and each refusal to choose is an equality, and the costs are what a linear program makes of the pile. The numbers that come out are not DuckDB's own and do not need to be, because what has to match is the ordering, and the generator checks that by replaying all 2550 decisions through the table it has just solved.

Three things measurement said that guessing would not have.

A cost very nearly belongs to the target type alone. One number per target satisfies almost every inequality there is, so widening within a family being cheap and crossing families being expensive is not the rule: `TINYINT` to `SMALLINT` and `UBIGINT` to `SMALLINT` cost the same, because both of them end at `SMALLINT`. What depends on the source is whether the cast exists at all, and that is where the surprises are. `UTINYINT` reaches `SMALLINT` and not `TINYINT`, and no integer reaches `VARCHAR`.

The almost is two pairs, and both of them are a 128 bit integer. `mod` is declared over `DECIMAL` and over `DOUBLE`, and DuckDB takes the `DOUBLE` one for a `HUGEINT` argument, which one number per target cannot express because everything else prefers the decimal. A decimal stops at 38 digits and a `HUGEINT` needs 39, so the widening a decimal parameter does for a `BIGINT` it cannot do here, and DuckDB charges accordingly. So the generator starts with one number per target and gives a pair a number of its own only where no set of numbers fits otherwise, then takes each of those back out again to check it was needed. Which pairs come out of that is not unique, because a discount on one cast and a surcharge on the cast it competes against say the same thing about which overload wins, and the catalog holds no call that tells the two apart. The generated header says so rather than presenting the list as a design.

A `NULL` is priced by a rule of its own. It is not a value being converted, and its column of costs orders differently from the rest of the table, which is why `abs(NULL)` is a `BIGINT` rather than an error and `century(NULL)` is an error rather than a `DATE`.

A tie is a real case and DuckDB's answer to one is to refuse the call, in a sentence of its own with a candidate list holding the overloads that tied and nothing else. 68 of the 2550 decisions are refusals, and they are the most useful readings in the pile, because that candidate list is the only place DuckDB ever says out loud that two signatures cost the same.

Getting one cost wrong picks a different overload for some argument combination and produces a different type, then a different answer, so the resolution fuzzer in document 11 section 5 calls each name in the catalog with every combination of a small set of typed arguments and compares the resolved return type against DuckDB's. It is not the same check as the replay above and it has already paid for itself twice over: the replay only says the solved table agrees with the calls it was solved from, and the fuzzer is what noticed that some of those calls had been read wrong in the first place.

A signature is not always the whole rule. `greatest` and `least` are declared over `ANY` with a variadic, and DuckDB then insists the arguments share a common type, refusing a `TINYINT` against a `VARCHAR` with the same sentence `CASE` gives for the same pair. The catalog cannot say that, so `resolve.mojo` says it, and the fuzzer is what found it.

## 4. The tiers

**Tier 1, required for TPC-H and for the conformance target.** This is the 1.0 commitment.

Arithmetic and comparison over every numeric type including decimal and hugeint, with the overflow behaviour from document 06. `CASE`, `COALESCE`, `NULLIF`, `IFNULL`, `IS DISTINCT FROM`. Casts between every pair in the supported type set, plus `TRY_CAST`. The string set the corpus and the benchmarks actually use: `substring`, `length`, `upper`, `lower`, `trim`, `ltrim`, `rtrim`, `concat`, `||`, `replace`, `strpos` and `position`, `like`, `ilike` and `similar to`, `left`, `right`, `lpad`, `rpad`, `split_part`, `starts_with`, `contains`, `repeat`, `reverse`, `md5`, `regexp_matches`, `regexp_replace` and `regexp_extract`. Date and time: `extract` and its shorthands, `date_part`, `date_trunc`, `date_diff`, `strftime`, `strptime`, `age`, interval arithmetic, `current_date` and `now`. Numeric: `abs`, `round`, `ceil`, `floor`, `sign`, `sqrt`, `pow`, `exp`, `ln`, `log`, the trig set, `greatest`, `least`, `mod`. Aggregates: `count`, `sum`, `avg`, `min`, `max`, `stddev` and `var` in both forms, `median`, `quantile` and `approx_quantile`, `string_agg`, `list` and `array_agg`, `bool_and` and `bool_or`, `first`, `last` and `any_value`, `count(DISTINCT)`, `arg_min` and `arg_max`. Windows: `row_number`, `rank`, `dense_rank`, `percent_rank`, `cume_dist`, `ntile`, `lag`, `lead`, `first_value`, `last_value`, `nth_value`, plus every tier 1 aggregate used as a window function.

Roughly 150 names. That is the honest 1.0 scope and it covers TPC-H completely, db-benchmark completely, and the large majority of what the `test/sql` analytical directories exercise.

**Tier 2, the list, struct and map surface.** 138 names by the count above. These are what makes DuckDB pleasant and they are what a pandas user reaches for when a column holds a list. Post 1.0, except that the constructors `list_value`, `struct_pack`, `[...]` and `{...}` and the accessors are pulled forward into tier 1 because the grammar makes them syntax rather than functions.

**Tier 3, refused by name.** JSON, spatial, `bit_*`, encryption, the `duckdb_*` introspection functions beyond `duckdb_functions()` and `duckdb_settings()`, full text search, and the long tail of the 711 other names. Each refusal names the function and points at the tracking issue, per document 05, and `firepanda.sql_support()` enumerates them.

## 5. Implementation, and why generation is the answer

Every scalar function is a kernel over Arrow arrays with a validity bitmap, and ninety per cent of them are the same three shapes:

- unary elementwise over `T` producing `U`, with null in giving null out
- binary elementwise over `(T, T)` producing `U`, with broadcast for constants
- n-ary with a custom loop

Mojo's `comptime` monomorphizes these across the dtype set from one source, which is exactly how `firepanda/kernel/` already works, so a scalar function is a few lines plus a registry entry. The registry table itself is generated from a declaration list, which is what keeps 150 names from being 150 hand written registration blocks.

Three things must not be generated, because generating them produces something wrong.

**Anything with derived return types.** Decimal arithmetic and `sum`'s promotion to HUGEINT are rules, written once and shared.

**Anything with a null rule that does not propagate.** `concat` ignores nulls, `coalesce` short circuits, `count` counts non nulls, and `||` propagates. Each is stated explicitly at its registration site and tested, because the default is null propagating and a silently defaulted function is a wrong answer.

**Aggregates.** They are stateful and their protocol is the performance critical one.

## 6. The aggregate protocol

Four operations, which is DuckDB's design and the only one that supports both the hash aggregate and the window path:

```
init(state)                          # zero a state
update(states, sel, input_vector)    # vectorized, many states at once
combine(target, source)              # merge partials, for parallelism and spilling
finalize(states) -> result_vector    # produce values
```

`update` taking a selection vector and a whole input vector at once is what makes the hash aggregate fast: one call per chunk per aggregate, not one call per row. `combine` is what makes it parallel across morsels and what makes it spillable, and an aggregate without a `combine` forces a single threaded, unbounded memory path. So `combine` is mandatory at registration, and the ones that genuinely cannot have it, such as `string_agg` with `ORDER BY`, declare themselves ordered and get the fallback path explicitly, matching DuckDB's own `ordered_aggregate_threshold` of 262,144.

State must be fixed size and trivially relocatable wherever possible, because that is what lets the hash table store states inline and lets document 09 spill a partition by writing bytes. `sum`, `count`, `avg`, `min` and `max` on fixed width types, and the moment based statistics, all qualify. `list`, `string_agg`, `quantile` and the distinct variants do not, and each carries a note about what its memory does as cardinality grows, which is the question document 09 actually asks.

## 7. Window functions

Windows share the aggregate protocol and add a frame. The three implementation strategies, chosen per function:

**Naive per row** for small frames or non decomposable aggregates.

**Segment tree** over the partition for decomposable aggregates with arbitrary frames. This is DuckDB's approach, it is `O(n log n)`, and it is the only thing that makes `RANGE BETWEEN` frames tractable.

**Streaming** for the running total case, `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, which is by far the most common and is `O(n)` with one state.

The ranking functions `row_number`, `rank`, `dense_rank` and `ntile`, and the navigation functions `lag`, `lead`, `first_value` and `nth_value`, are not aggregates and get direct implementations over the sorted partition.

`EXCLUDE CURRENT ROW`, `GROUP` and `TIES`, and `GROUPS` framing, are part of the grammar and part of `test/sql/window`, so they are part of 1.0 rather than deferred. The frame boundary computation is shared and the exclusion is a modification to it, not a separate path.

## 8. Table functions

Nine names at 1.0: `read_csv`, `read_csv_auto`, `read_parquet`, `read_json`, `read_json_auto`, `glob`, `range`, `generate_series` and `unnest`. All but the last three touch the filesystem and all of them are gated by `enable_external_access` per document 02.

They bind to firepanda's existing readers. `read_parquet` today goes through `firepanda/io/duckdb.mojo`, which is an honest dependency for reading a file format and not the wrapper that document 00 rejected. The query is still planned and executed by firepanda, and the dependency is on a decoder rather than on an engine.

`unnest` is different from the rest and deserves its own note. It is a table function in the grammar but a plan node in practice, because `SELECT unnest(l) FROM t` expands rows, and document 08 gives it a dedicated operator rather than pretending it is a scalar.
