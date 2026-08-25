# The two front doors

The requirement is "same DX, so easy to learn from a Python background". This document is about what that is allowed to mean, because taken literally it is both impossible and undesirable, and taken loosely it is an excuse for shipping something that merely resembles pandas.

## 1. The contract

**If you know pandas, you can use firepanda without reading anything.** Method names, argument names, argument order and return shapes match pandas 3.0 wherever a match is possible. When you type `df.groupby("region").agg(...)` it works, and it works the same way.

**Where we differ, you get an error, never a surprise.** A divergence that raises is a five minute annoyance. A divergence that silently returns different numbers is a corrupted report and a lost user. Every divergence in section 6 either fails loudly or does not exist.

**The Mojo source and the Python source are the same text.** Not similar, the same. This is possible because Mojo's syntax is Python's for the subset a dataframe API uses, and it is the reason the two front doors are not two APIs.

**Nothing in the Python surface is a second class citizen.** No "for full performance, drop into Mojo". A Python user gets the engine.

## 2. What it looks like

Python:

```python
import firepanda as pd

df = pd.read_parquet("trades/*.parquet")
out = (
    df[df["price"] > 100]
    .groupby(["symbol", pd.Grouper(key="ts", freq="1min")])
    .agg(volume=("qty", "sum"), p99=("price", lambda s: s.quantile(0.99)))
    .sort_values("volume", ascending=False)
    .head(20)
)
print(out)
```

Mojo:

```mojo
from firepanda import read_parquet, col, Grouper

var df = read_parquet("trades/*.parquet")
var out = (
    df.filter(col("price") > 100)
      .groupby(["symbol", Grouper(key="ts", freq="1min")])
      .agg(volume=col("qty").sum(), p99=col("price").quantile(0.99))
      .sort_values("volume", ascending=False)
      .head(20)
)
print(out)
```

Three differences and each one is forced.

`df[df["price"] > 100]` is boolean mask indexing through `__getitem__`. It works in Python and it works in Mojo, and in Mojo it is not what anyone should write, because `df.filter(col("price") > 100)` is the same thing without the intermediate mask materialization. Both exist. The pandas spelling is supported so pandas code runs; the expression spelling is documented as the one to use.

`lambda` in the aggregation is a Python callable and it stays one. Mojo 1.0 added Python style lambda syntax for inline closures, so the equivalent compiles, but a Python lambda arriving through the binding is invoked per group through CPython. Section 7.

Keyword arguments in `agg` work in both, because Mojo supports them and the binding layer supports `OwnedKwargsDict[PythonObject]`.

## 3. The eager surface is a facade and that is the whole trick

`df[df["price"] > 100].groupby(...).agg(...)` in pandas materializes an intermediate frame at every step. In firepanda it does not.

Every eager method builds a plan node. Materialization happens when a terminal operation demands it: `print`, `__len__`, `to_pandas`, `to_arrow`, `sum()` on a Series, iteration, `__repr__`, or an explicit `.collect()`.

So the pandas spelling gets predicate pushdown, projection pushdown and kernel fusion, without the user knowing what any of those words mean. `df[df.price > 100][["symbol", "qty"]]` reads two columns from Parquet and skips row groups, because by the time anything runs, the optimizer has seen the whole chain.

This is the single highest leverage decision in the document. It means the naive pandas idiom, the one every tutorial teaches and every user types, is fast here. Not "fast if you rewrite it in the lazy API", fast as written.

Two things make it honest rather than a trick.

**The deferral has to be invisible in the failure case too.** A bad column name in step one must raise when step one is written, not five lines later at `print`. So the plan is type checked eagerly at each step against the known schema, and only execution is deferred. The error arrives where the mistake is. This is why the schema has to be known at scan time, which is why `read_csv` on a file with no header has to read a sample rather than deferring inference.

**`repr` in a notebook must not run the whole query.** `__repr__` collects with an implicit `head(N)` and a limit pushdown, and prints the shape from metadata where the plan permits it. Getting this wrong makes interactive use unbearable, and it is the failure mode every lazy library ships with at least once.

An escape hatch exists for the cases where deferral confuses somebody: `firepanda.set_option("eager", True)` collects at every step. It is documented as a debugging aid and it is slower.

## 4. The lazy surface, for people who want it

`firepanda.scan_parquet(...)` returns a `LazyFrame` with the Polars shaped API, and `.collect()` runs it. Same plan, same optimizer, same execution; different spelling for people who prefer explicit.

```python
import firepanda as fp

out = (
    fp.scan_parquet("trades/*.parquet")
      .filter(fp.col("price") > 100)
      .group_by("symbol")
      .agg(fp.col("qty").sum().alias("volume"))
      .sort("volume", descending=True)
      .head(20)
      .collect()
)
```

pandas 3.0 shipping `pd.col()` is what makes having both surfaces coherent rather than confused. The two communities have been converging for two years, and a library starting in 2026 gets to serve both without picking a side.

## 5. `explain()`, which pandas does not have

```python
>>> out.explain()
SORT volume DESC, limit 20
  AGGREGATE by [symbol] -> [sum(qty) as volume]
    FILTER price > 100
      SCAN trades/*.parquet
        project: [symbol, qty, price]     3 of 14 columns
        predicate: price > 100            pushed to row group statistics
        estimated row groups: 12 of 340
```

`profile()` adds wall time, rows in and out, and bytes read per operator.

This ships at M4, not as a debugging feature later. Two reasons. A user whose query is slow currently has nothing at all in pandas and this is the single largest DX advantage on offer. And internally, every optimizer pass we claim becomes a test asserting on this output rather than a claim nobody can check.

## 6. Every divergence from pandas

This is the complete list of places where a working pandas program does not do the same thing in firepanda. If it is not here, it is a bug.

| Divergence | Behaviour | Why |
|---|---|---|
| No automatic index alignment | `a + b` is positional. Aligning requires `a.align(b)` first. | pandas' largest source of silent wrong answers. |
| No implicit index | An index exists only if you call `set_index`. Then `loc`, `reindex`, `sort_index` and the rest work normally. | Same reason. The capability is kept, the surprise is not. |
| No `inplace=` | The keyword raises `TypeError` with the reassignment form in the message. | Frames are values. pandas 3.0 deprecated most of these anyway. |
| Chained assignment | `df[cond]["c"] = v` raises. | pandas 3.0 raises too. Identical behaviour. |
| No `dtype=object` | Constructing one raises, naming the column and the offending Python type. | There is no boxed column, ever. |
| No silent upcasting | `int64 + float64` raises at plan time, with the two dtypes, the column names and the exact cast to write. | pandas' silent upcast is a correctness hazard. This is the divergence users will hit most. |
| NaN is not null | `NaN` is a valid float and does not count as missing. `isna()` reports the validity bitmap. | Arrow semantics. pandas 3.0 moved this way. |
| Regex dialect | RE2 semantics. No backreferences, no lookaround. | Linear time guarantee. Documented prominently, because a regex that works in pandas and raises here is a bad afternoon. |
| Float aggregation is not bit identical | Vectorized summation associates differently and is more accurate. | Unavoidable and true of Polars and DuckDB as well. |
| `apply` with a Python callable | Works. Warns once per session with the timing and the Mojo alternative. | Section 7. |
| No `.plot` | `.to_pandas().plot()` or hand the Arrow frame to any plotting library. | Not our job. Zero copy makes it a non issue. |
| No pickle IO | `read_pickle` and `to_pickle` raise, pointing at Parquet and IPC. | Pickling a Mojo struct is not a thing, and pickle as an interchange format should die anyway. |
| `.style` | Absent until post 1.0. | Jupyter HTML styling is a real feature and it is not on the critical path. |
| Negative index normalization | `iloc[-1]` and `s[-3:]` work. | Mojo 1.0 removed negative indexing from `List` and `Span`, so the API layer normalizes. Listed here because it is a bug waiting to happen internally, not because a user sees it. |

Seven of those are genuine capability omissions: `plot`, pickle, `.style`, `object` dtype, `inplace=`, index alignment, and lookaround regex. The rest are behaviour differences that raise.

## 7. The `apply` problem, stated honestly

`df.apply(f)` where `f` is a Python function has to cross the CPython boundary once per row or once per group. That is slow for exactly the reason it is slow in pandas, and no amount of engineering on our side changes it.

Three responses, in order.

**Make the fast path reachable.** Most `apply` calls in real code are expressible as expressions. `df.apply(lambda r: r.a * r.b)` is `col("a") * col("b")`. The library recognizes a set of common lambda shapes by bytecode inspection and rewrites them into expressions. This is not a general solution and it does not need to be; a handful of shapes covers a large fraction of real usage.

**Make the cost visible.** The first Python callable `apply` in a session emits a one line warning with the measured per row cost and the expression equivalent when one is inferable. Once per session, not once per call, because a warning that fires in a loop gets suppressed and then ignored.

**Make the alternative easy.** A Mojo user writes the function in Mojo and it compiles into the pipeline at full speed, per document 03 section 7. A Python user who cares can put one function in a `.mojo` file next to their script and pass it by name. That is a real step up in effort and it is the honest boundary of what this design can do.

What we do not do is silently accept a Python callable and let the user believe the library is slow. The whole positioning is about that boundary, so hiding it here would be dishonest in the one place it matters most.

## 8. Errors

pandas error messages are famously bad and this is free ground to take.

```
>>> df.groupby("regoin")
KeyError: column 'regoin' not found in frame

  did you mean 'region'?

  available: region, symbol, qty, price, ts
```

```
>>> df["qty"] + df["price"]
DTypeError: cannot add int64 and float64

  qty   is int64
  price is float64

  firepanda does not upcast silently. write one of:
      df["qty"].astype("float64") + df["price"]
      df["qty"] + df["price"].astype("int64")

  at plan position 2 of 4
```

Four requirements, all enforced in review.

Name the column and the dtype involved, never just the operation. Offer the correction when edit distance makes one obvious. Say what to write, not just what is wrong. Give the plan position, because in a fifteen step chain "which line" is the actual question.

Mojo has a single `Error` type carrying a string, and Python users expect `KeyError`, `TypeError` and `ValueError`. The binding layer maps our error taxonomy onto the CPython exception classes, so `except KeyError` in user code works. That mapping is a table in document 07 and it is tested, because an error type that changes between versions breaks user code as surely as a signature does.

## 9. Display

The `repr` is the thing a notebook user looks at more than any other output, so it is a feature.

```
shape: (4, 3)
┌────────┬──────────┬───────────┐
│ symbol │ volume   │ p99       │
│ str    │ i64      │ f64       │
├────────┼──────────┼───────────┤
│ AAPL   │ 18234100 │ 234.51    │
│ MSFT   │  9120400 │ 411.09    │
│ NVDA   │  8004250 │ 1204.88   │
│ null   │    12000 │ null      │
└────────┴──────────┴───────────┘
```

Dtypes in the header, because the single most common pandas debugging question is "what type is this column". Null printed as `null` and distinct from `NaN`, because they are different things here. Numbers right aligned, strings left aligned, long values truncated with an ellipsis at a terminal aware width. Shape first, so the answer to "how many rows" does not require scrolling.

`_repr_html_` for Jupyter, with the same information.

## 10. Installation, which is part of DX

```
pip install firepanda
```

That has to work, on macOS and Linux, with no Mojo toolchain, no conda, and no compiler. If it does not, the audience is people who already have Mojo installed, and that audience is small enough that the project is not worth doing.

This is the hardest engineering problem in the specification and it is not a dataframe problem. Document 07.

`pixi add firepanda` from the `modular-community` channel is the Mojo path and it is the easy one.

The README says Linux and macOS, Windows via WSL, in the first screen. Discovering that at `pip install` time is a bad first impression and it is entirely avoidable by saying so.
