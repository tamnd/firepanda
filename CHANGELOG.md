# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

Pre-1.0, minor versions may break the API. Every break appears here with the migration, not just with a note that it happened.

The Mojo toolchain version is part of a release's identity and is recorded with each entry, because the Mojo ABI is not stable within 1.x and a binary artifact built against one runtime is not guaranteed to load against another.

## [Unreleased]

### Added

- The specification: twelve documents in `docs/specs/`, written against Mojo 1.0, pandas 3.0.5, Polars 1.43 and Arrow 25.0.0 as of August 2026.
- Milestone issues M0 through M11 covering the work to a defensible 1.0.
- CI: spec conformance checks, a guarded build and test matrix, wheel builds with clean-install verification, a nightly Mojo canary that files an issue on failure, a benchmark regression gate, and workflow, dependency and Scorecard auditing.
- Release: PyPI publishing through Trusted Publishing with build provenance attestations. No PyPI token exists in this repository.
- M0, the foundation layer, built against Mojo 1.0.0 (ed45d567):
  - `firepanda.bitmap`: an Arrow validity bitmap with word at a time popcount, boolean operators, ranged set, and both aligned and unaligned slicing.
  - `firepanda.buffer`: 64 byte aligned allocation and a size class pool.
  - `firepanda.array`: `Array[dt]`, the type erased `AnyArray`, `ChunkedArray`, and the StringView layout with its 16 byte inline prefix representation.
  - `firepanda.dtype`: the logical type lattice, `Schema` and `Field`, promotion that agrees with numpy on all 144 pairs, the `comptime` dtype lists, and the `dispatch` bridge from a runtime dtype tag to a compiled instantiation.
  - Tests: 90 unit tests, a ten million case bitmap fuzz against a `List[Bool]` reference, a concurrency stress harness, and a differential suite that runs against numpy and pyarrow in process.
  - Tools: a microbenchmark suite with a median and IQR report, a comparison tool that will not call anything a regression unless it clears the measured spread, and the compile time and binary size probes that make the monomorphization cost per dtype a number rather than a worry.

- The compute kernel layer, the first part of M1:
  - `firepanda.kernel`: sum, count, min, max and mean reductions; add, subtract, multiply and divide; the six comparisons; casts between any two numeric dtypes; validity masking; and take and filter.
  - A null holds a zero in the values buffer, which is what lets `sum` and `mean` run without reading the validity bitmap at all. The invariant, and the one way to break it, are written down in `firepanda/kernel/__init__.mojo`.
  - `firepanda/kernel/scalar.mojo`: a one element at a time twin of every kernel, never called in production, which is what the kernels are checked against.
  - Tests: 26 unit tests and a second fuzz harness that runs every kernel against its twin over six dtypes and four null shapes.
  - `Bitmap.slice` on an unaligned start now shifts a byte at a time instead of a bit at a time, and `Array.slice` copies its values with one memcpy. Unaligned bitmap slicing went from 1.326 to 0.051 ns per bit.

- The hash layer, the second part of M1:
  - `firepanda.hash`: `factorize`, which rewrites a column as dense group ordinals plus the keys those ordinals name. This is what group by, join, unique, value counts and the categorical dtype are all going to be built on.
  - An open addressing table with linear probing at a load factor of one half. It stores the hash rather than the key, which is exact rather than a shortcut, because the mixing function is a bijection on 64 bits.
  - Columns whose integer range is small enough skip the table entirely and index an array with the value. That route runs at 0.74 ns per row against 2.68 for a `Dict`, and it covers most of the categorical columns anyone actually has.
  - Sizing is measured, not asked for. The build watches its own group discovery rate at two checkpoints and extrapolates. Presizing to the row count instead was tried and was 1.7x slower on a ten thousand group column, because the table stops fitting in cache.
  - `firepanda/hash/scalar.mojo`: a quadratic twin to check against, and a `Dict` based factorize kept in the library so the comparison stays runnable.
  - Tests: 41 unit tests and a fuzz harness that checks the table, the partitioning and both factorize routes against the twin.

There is no dataframe yet. See the status notice in the README.
