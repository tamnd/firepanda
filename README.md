<h1>firepanda</h1>

<p>
  <a href="https://github.com/tamnd/firepanda/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/tamnd/firepanda/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
  <a href="docs/specs"><img alt="Specification" src="https://img.shields.io/badge/spec-12%20documents-informational"></a>
  <img alt="Status" src="https://img.shields.io/badge/status-M1%20group%20by-orange">
</p>

A dataframe library for [Mojo](https://mojolang.org) with the pandas API.

> **Status: M0 landed and M1 is under way. There is a `DataFrame` now, it groups by and joins, and there is still nothing to install.** What exists is the layer underneath one, plus the first thing built on it: validity bitmaps, aligned buffers with a size class pool, typed and type erased columns, the StringView layout, the logical type lattice with its promotion rules, the `comptime` dtype dispatch bridge, the compute kernels that run over a column, a stable radix sort with multi-key and null placement, the hash table and `factorize` that group by and join are both built on, thirteen grouped reductions from `sum` through `median` and distinct count, all seven join kinds from inner through anti and cross, an eager `Series` and `DataFrame` over all of it that can select, cast, filter, take, slice, sort, group by and join, and a display layer that renders either of them as a table. It cannot read a file. It is tested, fuzzed against a reference model, and checked against numpy and pyarrow in the same process. The specification is twelve documents in [`docs/specs/`](docs/specs), written against Mojo 1.0, pandas 3.0, Polars 1.43 and Arrow 25.0 as of August 2026, and the milestone issues track the rest of the way to a frame you can actually use. If you are looking for a working Mojo dataframe today, you want [MojoFrame](https://arxiv.org/abs/2505.04080), which is a research prototype, or [Polars](https://pola.rs), which is not in Mojo but is excellent.

## What it is meant to be

```python
import firepanda as pd

df = pd.read_parquet("trades.parquet")
big = df[df["qty"] > 1000].groupby("symbol")["notional"].sum()
```

That is pandas. It is also, underneath, a lazy columnar query engine that pushed the projection and the predicate down into the Parquet reader, never decoded the columns you did not ask for, and ran the aggregation across every core in your machine.

And this is the same library, from Mojo, with the user's own function compiled into the pipeline rather than called through an interpreter:

```mojo
from firepanda import read_parquet, col

fn score(price: Float64, qty: Float64) -> Float64:
    return price * qty * 0.997

var df = read_parquet("trades.parquet")
        .filter(col("qty") > 1000)
        .with_column(map2[score](col("price"), col("qty")).alias("net"))
        .group_by("symbol").agg(col("net").sum())
        .collect()
```

## What actually runs today

The block above is the target. This is the part that works now, and the output is copied from a real run rather than written by hand.

```mojo
var df = DataFrame.from_series(columns^)
print(df)

var big = df.filter(greater(qty, threshold)).sort_by("price", descending=True)
print(big)

var widened = big.cast("qty", DType.float64)
print(Series("notional", multiply(
    widened.column("qty").as_typed[DType.float64](),
    widened.column("price").as_typed[DType.float64](),
)))
```

```
    qty    price
0   400   101.25
1  1200     99.5
2  2500    100.0
3    80    98.75
4  1750     <NA>
5  3000  100.125

[6 rows x 2 columns]

    qty    price
0  3000  100.125
1  2500    100.0
2  1200     99.5
3  1750     <NA>

[4 rows x 2 columns]

0    300375.0
1    250000.0
2    119400.0
3        <NA>
Name: notional, dtype: float64
```

Group by works too, on one key or several, with a reduction per output column.

```mojo
var specs = List[AggSpec]()
specs.append(AggSpec("qty", AggKind.SUM, "qty"))
specs.append(AggSpec("price", AggKind.MEAN, "avg_price"))
specs.append(AggSpec("qty", AggKind.COUNT, "trades"))
print(df.group_by(["symbol"], specs))
```

```
   symbol   qty    price
0       1   400   101.25
1       2  1200     99.5
2       1  2500    100.0
3       2    80    98.75
4       1  1750     <NA>
5       3  3000  100.125

[6 rows x 3 columns]

   symbol   qty  avg_price  trades
0       1  4650    100.625       3
1       2  1280     99.125       2
2       3  3000    100.125       1

[3 rows x 4 columns]
```

Symbol 1 averages 100.625 over two prices rather than three, because row 4 is null and a mean divides by what is there. `trades` counts three, because it counts `qty` and `qty` has no nulls.

No Parquet, no expression API and no strings. Columns are built by hand, the mask comes from a kernel rather than from `df["qty"] > 1000`, group by takes a list of specs rather than a chained `.groupby("symbol").sum()`, and a null and a `NaN` print differently because in an Arrow layout they are different things.

## The argument

Every fast dataframe library today is a fast engine in one language with a Python veneer on top. pandas is C and Cython. Polars is Rust. DuckDB is C++. The veneer is where user code lives, and it is why `df.apply(lambda ...)` falls off a cliff: the moment you write a function the library did not anticipate, you leave the fast language and enter the slow one.

Mojo removes the boundary. The library, the kernels and the user's own hot loop are all the same language, and they all compile. That is the one thing firepanda can offer that a mature library in another language structurally cannot, and it is the reason to build this rather than use Polars.

The honest counterweight is in [`docs/specs/00-README.md`](docs/specs/00-README.md): the argument only pays off for Mojo users. For Python users arriving through `pip install firepanda`, a Python callable is still a Python callable, and the benchmark tables say so in their own column.

## Design in one screen

| | |
|---|---|
| **Memory** | Apache Arrow layout throughout. Validity bitmaps, StringView, dictionary encoding. Interchange with pandas, Polars, DuckDB and pyarrow is zero copy through the PyCapsule protocol. |
| **Kernels** | One generic function per kernel, `comptime`-parameterized over `DType`. `vectorize` writes the remainder loop. No build tags, no runtime feature detection, no hand-written tails. |
| **Dispatch** | Columns are type-erased with a runtime `DType` tag; a `comptime for` over the dtype list generates the dispatch chain. Mojo has no sum types, so this replaces the `enum` a Rust design would use. |
| **Execution** | Morsel-driven, work-stealing, radix-partitioned hash aggregation. Workers own their partitions exclusively, because there is no race detector in this language. |
| **Planning** | Lazy underneath, always. The eager pandas surface builds plans too, so the naive idiom gets projection and predicate pushdown for free without anyone learning an expression API. |
| **Front doors** | Mojo and Python, neither second class. `pip install firepanda` must work on a machine with no Mojo toolchain — otherwise the audience is people who already have Mojo, and that audience is too small. |

## The specification

Read [`docs/specs/00-README.md`](docs/specs/00-README.md) first; it is the index and it says what was already decided and why.

| | |
|---|---|
| [00](docs/specs/00-README.md) | Index, settled decisions, prerequisites, honesty about scope |
| [01](docs/specs/01-research-2026.md) | State of Mojo 1.0, pandas 3.0, Polars, Arrow and the Mojo ecosystem, with sources |
| [02](docs/specs/02-architecture.md) | Layers, memory, types, plan, optimizer, execution |
| [03](docs/specs/03-dtype-dispatch.md) | Compile-time monomorphization and the generated dispatch table |
| [04](docs/specs/04-python-dx.md) | The two front doors, and every deliberate divergence from pandas |
| [05](docs/specs/05-kernels.md) | Kernel shape, the hash table, strings, the GPU path |
| [06](docs/specs/06-pandas-parity.md) | The full pandas 3.0 conformance checklist, milestone-tagged |
| [07](docs/specs/07-python-bindings.md) | `PythonModuleBuilder`, Arrow PyCapsule, wheels, and the ABI problem |
| [08](docs/specs/08-milestones.md) | M0 through M11, exit criteria, and four points to stop and reassess |
| [09](docs/specs/09-quality-bar.md) | Testing, and living without a race detector |
| [10](docs/specs/10-benchmarks.md) | What gets measured and against whom |
| [11](docs/specs/11-package-layout.md) | The tree, and why Mojo 1.0's import rules decide it |

There are no time estimates in these documents, deliberately. Milestones are ordered by dependency and by risk; a week count invites a reader to add them up and treat the total as a delivery date.

## The two risks worth naming up front

**Distribution.** Mojo 1.0 guarantees source compatibility within 1.x and explicitly does not guarantee ABI stability, and a wheel is a binary artifact. The plan is to vendor the runtime and pin hard — which first requires confirming the runtime may be redistributed at all. That question is the first task of M3, before any binding code gets written, because a negative answer changes the entire distribution strategy.

**Code size.** Monomorphizing every kernel over every dtype produces on the order of a few thousand instantiated function bodies. Compile time and stripped binary size are graphed in CI from the first commit, with named thresholds and three graduated responses, because finding this out late is how the strategy fails.

## Related

- [tamnd/firepanda-bench](https://github.com/tamnd/firepanda-bench) — the performance comparison against pandas, Polars, DuckDB, cuDF and MojoFrame. Losses get published next to the wins; for a project whose pitch is performance, the credibility of the numbers is the asset.
- [tamnd/kuma](https://github.com/tamnd/kuma) — the sibling specification for the same problem in Go. Several documents here are written against it as a contrast.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). At specification stage the most useful contribution is disagreement: if something in `docs/specs/` is wrong about Mojo 1.0, about pandas 3.0, or about what an engine of this shape costs to build, an issue saying so is worth more than any amount of code written against a bad premise.

Claims in the specification that could not be confirmed against Modular's own documentation are marked **[verify]**. Confirming or refuting one of those is a genuinely valuable pull request.

## License

Apache-2.0. See [LICENSE](LICENSE).
