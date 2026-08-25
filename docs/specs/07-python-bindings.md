# The Python front door

`pip install firepanda` has to work on a machine with no Mojo toolchain, no conda and no compiler. If it does not, the audience is people who already have Mojo installed, and that audience is too small for this project to be worth doing.

This is the hardest engineering problem in the specification and none of it is about dataframes.

## 1. The mechanism

Python loads a compiled extension module by finding a dynamic library that exports `PyInit_<name>()`. This is the same protocol C, C++ and Rust extensions use, and Mojo supports it.

```mojo
from std.python.bindings import PythonModuleBuilder
from std.python import PythonObject
from os import abort

@export
fn PyInit_firepanda() -> PythonObject:
    try:
        var m = PythonModuleBuilder("firepanda")
        m.def_function[read_csv]("read_csv", docstring="Read a CSV file.")
        m.def_function[read_parquet]("read_parquet", docstring="Read Parquet.")
        _ = m.add_type[DataFrame]("DataFrame")
        _ = m.add_type[Series]("Series")
        _ = m.add_type[Expr]("Expr")
        return m.finalize()
    except e:
        return abort[PythonObject](String("firepanda init failed: ", e))
```

The import path for `PythonModuleBuilder` has moved at least once, from `python.bindings` to `std.python.bindings`. **Verify against the toolchain you install** rather than against this document.

Keyword arguments arrive as `OwnedKwargsDict[PythonObject]`, which is what makes `df.agg(volume=..., p99=...)` possible. Static methods on exposed types go through `PythonTypeBuilder.def_staticmethod()`.

## 2. The distribution problem, which is the real one

Two facts collide.

**Mojo 1.0 guarantees source compatibility within 1.x and explicitly does not guarantee ABI stability.**

**A wheel is a binary artifact.**

A shared library built against Mojo 1.0.0 is not guaranteed to load correctly against a 1.1.0 runtime. That is not a hypothetical, it is a stated non guarantee.

Three approaches, and the plan is to take the third.

**Ship the Mojo runtime inside the wheel.** Static linking, or vendoring the runtime shared objects and fixing the load paths with `auditwheel` on Linux and `delocate` on macOS. The wheel then depends on nothing external and the ABI question does not arise, because both sides of the boundary shipped together. The costs are wheel size, and a licensing question that has to be answered before M3 rather than at release, because the Mojo compiler is not open source and the runtime redistribution terms are the gating fact. **This is the item to check first at M3, before writing any binding code**, since a negative answer changes the entire distribution strategy.

**Require a Mojo runtime from conda and ship a thin wheel.** Correct, cheap, and it means `pip install firepanda` does not work standalone, which fails the requirement at the top of this document.

**Ship self contained wheels and pin hard.** Vendor the runtime, build one wheel per platform per Mojo minor version, and declare the toolchain version in the wheel metadata. When Mojo 1.1 lands, rebuild. CI tests against the pinned version and against nightly, so a break is caught on the day it lands rather than on the day a user reports it.

Platforms: `manylinux` x86-64, `manylinux` aarch64, macOS arm64, macOS x86-64. Four wheels. No Windows, because Mojo has no native Windows support; WSL users install the Linux wheel.

`cibuildwheel` drives it. The Mojo toolchain comes from the pixi environment inside the build container.

## 3. The binding table

Every public API entry point appears exactly once, in one declarative table, and both the Mojo registration and the Python type stubs are generated from it.

```mojo
comptime BINDINGS = [
    Binding("read_csv",      read_csv,      Sig.file_and_options),
    Binding("read_parquet",  read_parquet,  Sig.file_and_options),
    Binding("DataFrame.head", DataFrame.head, Sig.int_arg),
    ...
]
```

The reason is stated plainly in document 01: Modular documents calling Mojo from Python as being in early development with a lot of expected change to the API and ergonomics. That instability sits underneath the product's main feature. Generating the registration from a table means an upstream change is one file rather than four hundred call sites.

Two things fall out of the table for free.

**Type stubs.** A `.pyi` generated from the same table, so autocomplete and mypy work. A pandas replacement with no type stubs is a worse developer experience than pandas, which now has them.

**The surface parity test.** A test asserts that every public method on the Mojo `DataFrame` appears in the table and vice versa. Without it, the Python surface silently drifts behind the Mojo one, which would break the promise in document 04 section 1 that neither front door is second class.

## 4. Data crosses as Arrow, never as objects

The one rule that keeps the boundary from becoming the bottleneck.

`PythonObject` traffic per element is exactly what makes pandas slow at the C-to-Python boundary, and it would make firepanda slow for the same reason. So:

**Plans and scalars cross as objects.** A column name, a threshold, an aggregation spec. Small, infrequent, and the 12x interop speedup in Mojo 1.0 applies here.

**Data crosses as an Arrow C stream, once.** `__arrow_c_array__` and `__arrow_c_stream__`, the PyCapsule protocol that pandas 3.0, Polars, DuckDB and pyarrow have all agreed on. Implementing that pair of methods once gives zero copy interchange with all of them and requires no library specific code path for any of them.

```python
import firepanda as fp, polars as pl, duckdb, pyarrow as pa

df = fp.read_parquet("trades.parquet")

pl.DataFrame(df)          # zero copy
pa.table(df)              # zero copy
duckdb.sql("select * from df")   # zero copy
df.to_pandas()            # zero copy for primitives, copy for object-requiring dtypes
```

Verification is by pointer identity, not by eye. A test asserts that the buffer address on the pyarrow side matches the address on the firepanda side. "It seemed fast" is not evidence of zero copy.

The reverse direction is the same protocol: anything exposing `__arrow_c_stream__` is constructible into a firepanda frame without a copy, which means pandas, Polars and DuckDB results all arrive for free.

## 5. Error mapping

Mojo has one `Error` type carrying a string. Python users write `except KeyError`.

The binding layer maps our error taxonomy onto CPython exception classes:

| firepanda error | Python exception |
|---|---|
| unknown column | `KeyError` |
| dtype mismatch, bad cast, `inplace=` | `TypeError` |
| bad argument value, unparseable format | `ValueError` |
| missing file, IO failure | `OSError` |
| unsupported operation, `object` dtype | `NotImplementedError` |
| cancellation | `KeyboardInterrupt` |
| everything else | `RuntimeError` |

This table is tested. An error class that changes between versions breaks user code as surely as a signature does, so it is part of the public API and it is in the compatibility policy.

The message text keeps the structure from document 04 section 8: the column, the dtypes, the suggestion, the plan position.

## 6. The GIL, threads and Ctrl-C

**Release the GIL around execution.** The whole point is parallel execution in a language that does not have one, and holding the GIL while running a 30 second query would block every other thread in the user's process. Acquire it again to construct the return value.

**Do not touch a `PythonObject` from a worker thread.** Any Python callable in a UDF is invoked from the calling thread, serially, or under an explicit re-acquire. Getting this wrong produces an interpreter crash rather than an exception, and it is the single most likely source of a hard-to-diagnose bug in this layer.

**`SIGINT` sets the cancellation flag.** Ctrl-C interrupts a running query at the next morsel boundary and raises `KeyboardInterrupt`. pandas cannot do this reliably and everybody who has killed a notebook kernel to escape a runaway `groupby` will notice.

**Free threaded Python works unmodified.** The extension is compiled and its correctness under a free threaded interpreter depends on its own locking, not on the GIL. Since data crosses as Arrow and the engine does not hold Python objects across threads, this should hold, and the CI matrix includes a free threaded 3.14 build to prove it rather than assume it.

## 7. Packaging layout

```
firepanda/                  the Mojo package
  __init__.mojo
  py/
    module.mojo             PyInit_firepanda, the builder calls
    bindings.mojo           the BINDINGS table
    convert.mojo            PyCapsule, error mapping, scalar conversion
python/
  firepanda/
    __init__.py             re-exports from the extension, plus pure Python sugar
    __init__.pyi            generated stubs
    _firepanda.so           the built extension
pyproject.toml
pixi.toml
```

The Python package has a thin `__init__.py` rather than being the raw extension, for three reasons: `__repr__` and `_repr_html_` are more pleasant to write in Python than to bind; deprecation shims for pandas signatures live there without touching Mojo; and the import error when a wheel does not match the platform can be a readable message instead of a linker error.

That file is the only Python in the project and it stays under a few hundred lines. It must never contain anything on a data path.

## 8. Exit criteria for M3

The example in document 04 section 2 runs from a clean virtualenv on macOS arm64 and manylinux x86-64, installed with `pip install firepanda`, with no Mojo toolchain present.

`to_pandas`, `to_polars` and passing the frame to DuckDB all work with no copy, verified by buffer pointer identity.

Ctrl-C interrupts a running query and raises `KeyboardInterrupt`.

The surface parity test passes, meaning every public Mojo method is exposed and vice versa.

The error mapping table is exercised by a test per row.

The free threaded 3.14 wheel passes the same suite as the standard build.

Wheel size is recorded in CI with a threshold, because vendoring a runtime makes this easy to lose track of.
