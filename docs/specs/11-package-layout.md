# Package layout

## The constraint that shapes this

Mojo 1.0 changed the import rules, and the change decides the tree.

**Package submodules are only accessible if they are re-exported in `__init__.mojo`.** Intra-package access without an explicit import is deprecated. Relative imports must use the `from` form; `import .foo` no longer works.

So `__init__.mojo` is not a convenience file listing a few nice names. It is the access control mechanism, and it is the only one the language provides. There is no `internal/` convention and no `pub(crate)`.

That is the entire visibility story: a symbol re-exported from a package's `__init__.mojo` is public API, and a symbol that is not is reachable only from inside that package. The tree below is arranged so that this distinction lands in the right places on its own.

## The tree

```
firepanda/
  __init__.mojo            the public surface: DataFrame, Series, col, read_*, options
  version.mojo

  dtype/                   DType lists, LogicalType, Schema, Field, coercion rules
    __init__.mojo
  buffer/                  aligned allocation, the size class pool
  bitmap/                  validity bitmaps, the boolean ops, popcount
  array/                   Array[dt], AnyArray, ChunkedArray, as_typed, dispatch
    strview.mojo           the 16 byte view layout
  hash/                    the open addressing table, batch probe, radix partitioning

  kernel/                  every kernel, comptime parameterized over DType
    __init__.mojo
    compare.mojo  filter.mojo  aggregate.mojo  arith.mojo
    string.mojo   regex.mojo   sort.mojo       cast.mojo
    scalar/                the twins, @no_inline, never called in production

  frame/                   DataFrame, Series, the eager surface
  expr/                    Expr, the node types, the builders
  plan/                    logical plan, optimizer passes, explain, profile
  exec/                    morsel scheduler, operators, selection vectors
    parallel.mojo          parallel_for, one task per index, no shared state
    morsel.mojo            the morsel queue, and every atomic in the library
    device.mojo            GPU affinity, transfer nodes                     M9
    stream.mojo            spillable sinks, the memory manager              M10

  io/
    csv.mojo  ndjson.mojo  ipc.mojo  parquet.mojo
    arrow_c.mojo           the C Data Interface, abi("C") declarations
    hive.mojo              partitioned dataset scanning
  temporal/                timezones, offsets, calendars, the DST rules     M7
  sql/                     parser into the same logical plan                M11

  py/
    module.mojo            PyInit_firepanda
    bindings.mojo          the BINDINGS table
    convert.mojo           PyCapsule, error mapping, scalar conversion

  testing/                 assert_frame_equal, the generated corpus, diffing

python/
  firepanda/
    __init__.py            re-exports, __repr__ sugar, the readable import error
    __init__.pyi           generated from BINDINGS
    py.typed

pixi.toml
pyproject.toml
```

## Why these boundaries

**`dtype`, `buffer`, `bitmap`, `array` and `hash` have no dependencies on anything above them.** They are independently useful and independently testable, and a bug in the bitmap has nowhere to hide. `hash` in particular is a general purpose open addressing table that somebody could reasonably want on its own, and keeping it usable that way costs nothing and keeps its interface honest.

**`kernel` depends on `array` and nothing else.** It never sees a `DataFrame`, a `Schema` or a plan node. A kernel takes buffers and produces buffers, which is what makes the differential fuzz against `kernel/scalar` possible and what makes the GPU port at M9 a launch change rather than a rewrite.

**`kernel/scalar` is a sibling, not a fallback.** Nothing in production calls it. It exists so that every kernel has a specification the tests can compare against, and it lives in the tree rather than in the test directory so that it is compiled and type checked with everything else.

**`frame` depends on `plan`, not the other way round.** The eager surface builds plans, per document 04 section 3. If that dependency ever points the other way, the facade has leaked and the optimizer has stopped seeing whole queries.

**Every atomic in the library is in `exec/morsel.mojo`** because one named file is the substitute for a race detector. A CI grep asserts that `Atomic` and mutable globals appear nowhere else. The earlier plan was a `shared.mojo` holding nothing but declarations, and that turned out to be worse. There is exactly one atomic, the morsel queue's cursor, and a file that keeps a struct's own field away from the struct hides the thing it was meant to make obvious. Document 09 section 3.

**`py` depends on everything and nothing depends on `py`.** The Python front door is a leaf. Nothing in the engine may import from it, and a test asserts that, because the moment a kernel knows about `PythonObject` the whole no-boundary argument is gone.

**`testing` is public.** Users writing tests over firepanda frames need `assert_frame_equal`, and shipping it means their tests fail the same way ours do.

## Stability tiers

Since there is no `internal/`, the tiers are documented and enforced by review rather than by the compiler.

| Tier | Packages | Promise |
|---|---|---|
| Stable | `firepanda`, `frame`, `expr`, `testing` | Semantic versioning from 1.0. Breaking changes are major only. |
| Stable, smaller surface | `dtype`, `array`, `bitmap`, `hash`, `io` | Same promise, deliberately narrow public surface so there is less to promise. |
| Unstable, documented | `kernel`, `plan`, `exec` | May change in a minor release. Documented so that a contributor can work in them, not so that a user can depend on them. |
| Private | `kernel/scalar`, `py`, anything not re-exported | No promise at all. |

Every package's `__init__.mojo` carries its tier in the module docstring. A symbol's tier is the tier of the package that re-exports it.

## Two repositories

`tamnd/firepanda` is the library, the Python module and the wheels. Everything above.

`tamnd/firepanda-bench` is the performance comparison: pandas, Polars, DuckDB, cuDF, MojoFrame, the datasets, the Docker images and the result history. Separate because building the library must never require installing any of them, and because its release cadence is driven by other projects' releases rather than by ours. Document 10.

The pandas *correctness* comparison stays in the main repository, in `testing`, because Mojo imports pandas in the same process and there is no reason to move it. That is a genuine difference from the Go sibling spec, where a separate bindings repository was required before pandas and the library could be loaded together at all.

## Naming

Module names are lowercase, single word where possible. Public API names match pandas 3.0 exactly, per document 09, and where pandas is inconsistent we are inconsistent in the same way, because the point is somebody's muscle memory rather than our aesthetics.

Internal names follow Mojo 1.0 conventions: `var` for every declaration, `comptime` for compile time values with `alias` accepted but not written in new code, `imm` rather than `read` for read-only arguments, `__deinit__` rather than `__del__`.

An underscore prefix marks something not re-exported. It is a signal to a reader, not a mechanism; the mechanism is the `__init__.mojo`.

## The single rule for adding a file

If a new file needs a symbol from a package that does not re-export it, the correct response is almost never to add the re-export. It is to ask why a dependency is pointing that direction, because under Mojo 1.0's import rules the visibility list and the architecture are the same document, and the moment they stop agreeing the tree above stops meaning anything.
