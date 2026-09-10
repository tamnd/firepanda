# Types and semantics

The document that decides whether the compatibility claim is true. The parser is what people think the hard part is, and it is not. The hard part is that every rule below produces a silently different answer rather than an error, and every one of them disagrees with pandas, or with Arrow, or with both.

Everything in this document was measured against DuckDB 1.5.5 on this machine rather than read from documentation.

## 1. The type set

Thirty nine logical types. The ones that matter and their firepanda mapping:

| DuckDB | firepanda | note |
| --- | --- | --- |
| BOOLEAN | BOOL | |
| TINYINT, SMALLINT, INTEGER, BIGINT | INT8, 16, 32, 64 | |
| UTINYINT through UBIGINT | UINT8 through 64 | |
| HUGEINT, UHUGEINT | INT128, UINT128 | required, see section 4 |
| FLOAT, DOUBLE | FLOAT32, FLOAT64 | |
| DECIMAL(p,s) | DECIMAL128 | required, see section 3 |
| VARCHAR | STRING | UTF-8, no length semantics |
| BLOB | BINARY | |
| DATE, TIME, TIMESTAMP, TIMESTAMPTZ | date32, time64, ts us | microsecond precision |
| INTERVAL | months, days and microseconds triple | not a duration |
| LIST, STRUCT, MAP | nested Arrow | |
| UUID, BIT, ENUM, UNION, VARIANT, GEOMETRY, BIGNUM | none | out of scope, document 01 |

Two of these are not optional and both are absent from firepanda today.

**`VARCHAR(n)` carries no semantics.** `typeof('a'::VARCHAR(3))` is `VARCHAR` and `'abcd'::VARCHAR(3)` is `'abcd'`. No truncation, no error, the length is parsed and discarded. Copy this exactly, because a helpful truncation would be a wrong answer.

**`TIMESTAMP` is microsecond.** `'2020-01-01 00:00:00.123456789'::TIMESTAMP` yields `...123456`. Nanoseconds are truncated and not rounded, and pandas defaults to nanoseconds, so the boundary in document 10 is a conversion with a documented loss.

## 2. Integer division is not integer division

```
1/2        ->  0.5        DOUBLE
1//0       ->  NULL
1/0        ->  inf        DOUBLE
5%3        ->  2          INTEGER
2^3        ->  8.0        DOUBLE
```

`/` on two integers is floating point division producing a DOUBLE. Integer division is `//`. The `integer_division` setting flips `/` back and defaults to false.

Division by zero follows `ieee_floating_point_ops`, which defaults to true, so `1/0` is `inf`, not an error and not null. `1//0` is `NULL`. Postgres raises on both. pandas gives `inf` for the float case and raises for the integer one.

Three different systems, three different answers, no error in any of them. This is the archetype of what this document is for.

## 3. Decimals, and why `1.1 + 2.2` is exactly `3.3`

```
1.1 + 2.2                        ->  DECIMAL(3,1)   3.3
DECIMAL(4,2) * DECIMAL(5,3)      ->  DECIMAL(9,5)
CASE WHEN true THEN 1 ELSE 2.5   ->  DECIMAL(11,1)
coalesce(1, 2.0)                 ->  DECIMAL(11,1)
SELECT 1 UNION ALL SELECT 2.5    ->  DECIMAL(11,1)
```

An unsuffixed decimal literal is DECIMAL, not DOUBLE. This one decision propagates through the whole type system: any expression mixing an integer with a decimal literal produces a decimal, and the width is derived rather than fixed, so INTEGER widens to `DECIMAL(10,0)` and combined with a one place scale that becomes `DECIMAL(11,1)`.

The consequence is that `1.1 + 2.2 = 3.3` is true in DuckDB and false in pandas, NumPy, and every language whose literals are binary floats. Fixed point arithmetic with derived precision is therefore not an optional feature we can defer to a later milestone. Without it, arithmetic over literals gives different answers everywhere, and a large fraction of `test/sql/` fails on values that look correct.

Scale and precision rules must match DuckDB's exactly. Addition takes the maximum scale and widens precision by one, multiplication adds both precisions and both scales, division has its own rule, and overflowing `DECIMAL(38, s)` promotes to DOUBLE at some operations and raises at others. `sum()` over `DECIMAL(5,2)` is `DECIMAL(38,2)`, so scale is preserved and precision is maxed.

## 4. Aggregate result types

```
sum(INTEGER)        ->  HUGEINT
sum(BIGINT)         ->  HUGEINT
sum(DOUBLE)         ->  DOUBLE
sum(DECIMAL(5,2))   ->  DECIMAL(38,2)
avg(INTEGER)        ->  DOUBLE
min(INTEGER)        ->  INTEGER
count(*)            ->  BIGINT
count(DISTINCT x)   ->  BIGINT
bool_and(BOOLEAN)   ->  BOOLEAN
any_value(INTEGER)  ->  INTEGER
```

`sum` over any integer type is HUGEINT, which is 128 bit, and that is why INT128 is a hard requirement and not a nicety. pandas gives int64 and overflows silently, and Arrow gives int64 and raises. A `sum` that returns BIGINT is a compatibility failure under document 01 even when every value fits, because `typeof()` is observable and the corpus checks it.

`sum` over an empty group is `NULL`. `count` over an empty group is `0`. An aggregate with no `GROUP BY` over an empty input still emits one row.

## 5. Overflow raises

```
127::TINYINT + 1  ->  Out of Range Error: Overflow in addition of INT8
```

Integer overflow is an error, at every width, for every arithmetic operation. Not wraparound, which is what C, NumPy and pandas do, and not promotion. Every arithmetic kernel therefore needs an overflow checked path, and the cost of that check is a design constraint on the kernels in `firepanda/kernel/` rather than an afterthought. The checked path has to be vectorized, using the compiler's overflow intrinsics on the whole vector with a single branch on the aggregated flag, not a branch per element.

Casts raise too. `'abc'::INTEGER` is a `Conversion Error` naming the value. `try_cast('abc' AS INTEGER)` is `NULL`, and `TRY_CAST` is the escape hatch users are expected to reach for.

## 6. Nulls

The rules that produce wrong answers rather than errors.

```
'a' || NULL         ->  NULL
concat('a', NULL)   ->  'a'
NULL = NULL         ->  NULL
NULL IS NOT DISTINCT FROM NULL -> true
3 IN (1,2,NULL)     ->  NULL
3 NOT IN (1,2,NULL) ->  NULL
1 IN (1,2,NULL)     ->  true
```

The `||` against `concat` split is the one that catches people. The operator is null propagating and the function is null ignoring, and they are otherwise the same operation.

Three valued `IN` is the classic. A non match against a list containing null is `NULL` and not `false`, so `NOT IN` over a nullable subquery returns nothing, which is correct SQL, surprising to everyone, and load bearing in TPC-H q16 and q21. The plan must not rewrite `NOT IN` to an anti join without a null aware anti join, which document 08 owns.

`IS DISTINCT FROM` is the null safe comparison and is what `USING` joins and `GROUP BY` use internally. Grouping treats nulls as equal and comparison does not.

## 7. Ordering

```
default_null_order       =  NULLS_LAST
default_order            =  ASCENDING
preserve_insertion_order =  true
```

Measured: with three values `1, NULL, 2`, both `ORDER BY x` and `ORDER BY x DESC` put `NULL` last. `NULLS_LAST` is absolute and not relative to direction. Postgres puts nulls last on `ASC` and first on `DESC`, and pandas puts them last by default in both. We match DuckDB, and we expose the same setting.

`preserve_insertion_order = true` is why we differ from Polars 2.0's unordered default, and document 01 already made that call: the corpus is the oracle and the corpus assumes order.

Strings compare bytewise on UTF-8, so `'A' < 'a'` is true, and `'ab' = 'ab '` is false, with no trailing space equivalence, unlike `CHAR(n)` in the standard. `upper('ß')` is `'ẞ'`, which means case folding is Unicode aware rather than ASCII and cannot be a byte table.

## 8. Implicit casts

The type resolution order for comparisons and for `UNION`, `CASE`, `COALESCE` and function overloads is a lattice with an implicit cast cost. Two measured behaviours pin it:

```
1 = '1'   ->  true      (VARCHAR is cast to INTEGER, not the reverse)
'2020-01-01' = DATE '2020-01-01'  ->  true
```

String literals are cast to the other side's type, and if the cast fails at runtime it raises. This is more permissive than Postgres and much more permissive than Arrow, which refuses cross type comparison outright.

The binder inserts every cast explicitly, per document 05, so the plan contains no implicit conversion. That property is what makes an execution time type surprise impossible: if the plan says the operands are INTEGER, they are.

## 9. Indexing is one based and slices are inclusive

```
[1,2,3][1]        ->  1
[1,2,3][1:2]      ->  [1,2]
'hello'[2:3]      ->  'el'
substring('hello',1,3) ->  'hel'
substring('hello',0,3) ->  'he'
strpos('hello','l')    ->  3
```

Lists and strings are one based. Slices include both endpoints. `substring` with start `0` is not an error, because the range is clipped and one character is lost, which is a real and much reported behaviour that must be reproduced rather than fixed.

Every one of these disagrees with Python and therefore with the mental model of the user firepanda is built for. That is not a reason to change it. The moment we make lists zero based, one hundred per cent compatible becomes false and the number stops meaning anything.

String indexing is by character and not by byte, so it needs UTF-8 aware offsets, which is a real cost against firepanda's byte oriented string kernels and is called out in document 09.

## 10. Dates, times and intervals

```
DATE '2020-01-01' + 1                ->  DATE 2020-01-02
TIMESTAMP - TIMESTAMP                ->  INTERVAL
now()                                ->  TIMESTAMP WITH TIME ZONE
current_date                         ->  DATE
```

Integer addition to a date is days. `INTERVAL` is a triple of months, days and microseconds rather than a duration, because months and days are not fixed length, which is why interval arithmetic is not commutative with respect to daylight saving and why an interval cannot be normalized.

`now()` is `TIMESTAMP WITH TIME ZONE`, which drags in a session time zone and the ICU rules behind it. Document 13 keeps the extent of time zone support open. The 1.0 position is that naive `TIMESTAMP` is fully supported and `TIMESTAMPTZ` is supported for UTC with a named refusal otherwise, because a half implemented time zone is the worst of the three options.

## 11. How this is tested

Not by reading this document. `tests/differential/sql_semantics.mojo` holds one case per rule above, and each case runs the expression through DuckDB in process, using the same `libduckdb` that `firepanda/io/duckdb.mojo` already opens with `dlopen`, and through firepanda, and compares the value and `typeof()`.

Beyond the enumerated cases, an expression fuzzer generates random typed expression trees from the operator and function tables and compares both engines. Type disagreement is a failure even when the values match, because a wrong type is a wrong answer one operator later.

This suite runs before any of the performance work, because a fast wrong answer is not a result.
