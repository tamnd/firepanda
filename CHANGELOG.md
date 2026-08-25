# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

Pre-1.0, minor versions may break the API. Every break appears here with the migration, not just with a note that it happened.

The Mojo toolchain version is part of a release's identity and is recorded with each entry, because the Mojo ABI is not stable within 1.x and a binary artifact built against one runtime is not guaranteed to load against another.

## [Unreleased]

## [0.5.0] - 2026-08-26

Built against Mojo 1.0.0 (ed45d567).

The rest of group by. `std`, `var`, `median`, `quantile` and distinct count join the eight reductions from 0.4.0, which closes the group by line on the M1 issue.

The minor bump is for the five new reductions and for `AggKind` growing a `param` field to carry a delta degrees of freedom or a quantile. Anything that constructed an `AggKind` from a bare code still works and now gets the reduction's documented default rather than zero.

### Added

- `firepanda.kernel.group`: `group_var`, `group_std`, `group_median`, `group_quantile` and `group_nunique`, plus `AggKind.VAR`, `STD`, `MEDIAN`, `QUANTILE` and `NUNIQUE` so all five run through `aggregate_group` and `DataFrame.group_by` as well.
- `AggKind.var_with`, `AggKind.std_with` and `AggKind.quantile_at`, for the three reductions that take a number as well as a name.
- Tests: 24 more in `tests/test_group.mojo`, including the large value variance case that the one pass formula gets wrong.
- Benchmarks: seven `group/*` rows covering the two dispersions, the order statistics dense and sparse, the distinct count, and one erased quantile.

### Changed

- `AggKind` carries a `Float64` beside its code. `VAR` and `STD` read it as a delta degrees of freedom, `QUANTILE` and `MEDIAN` read it as the quantile, and everything else leaves it at zero. Two kinds compare equal on the code alone, so `kind == AggKind.QUANTILE` is true for the ninetieth percentile as well as for the median, which is what the dispatch chain needs.
- `tests/fuzz/kernel.mojo` rotates through all thirteen reductions rather than eight, with a fresh quantile position and a fresh degrees of freedom each time round. Its tolerance is relative above one, because a variance of a column near a million lands near 1e12 and an absolute tolerance there is asking floating point addition to be associative.

### Notes on the numbers

Measured on the reference machine, a 16 core x86_64 with 32 byte SIMD, at 1,048,576 rows and 1,000 groups.

- `group/var_sparse` is 4.413 ms against 1.609 ms for `group/mean_sparse`. Variance takes two passes over the column where a mean takes one, and the first of those two passes is the mean.
- `group/median` is 4.075 ms and `group/nunique` is 3.994 ms. Both build a slab of the non-null values grouped contiguously and sort each group's run, so they cost close to the same thing and the distinct count is the sort plus a scan for runs.
- `group/median_cardinality_10` is 18.601 ms against 4.075 ms for the same rows over a thousand groups. Ten groups means each slab run is a hundred times longer and the sort inside it is `n log n` on that length, which is the whole difference.
- `group/quantile_dispatched` is 3.792 ms, in line with the typed `group/median`, so the erased path costs nothing measurable once the reduction is this expensive.

### Known limitations

- The two dispersions read validity a bit at a time, like `mean`, `min` and the rest of the null-aware reductions. Reading a word at a time is still the open item in this file.
- `median`, `quantile` and `nunique` allocate a slab the size of the non-null values. A group by that computes several order statistics of the same column builds and sorts that slab once per reduction rather than sharing it.
- `nunique` counts by sorting rather than by hashing. That is the right call while the slab is being built anyway and it is `n log n` where a hash set would be linear.
- A float column containing NaN sorts it in an unspecified position, so a quantile over one is unspecified. Nulls are a separate thing and are excluded properly.

## [0.4.0] - 2026-08-26

Built against Mojo 1.0.0 (ed45d567).

Group by. `DataFrame.group_by` takes one or more key columns and a list of reductions and gives back one row per distinct key tuple, with pandas' answers for null keys, empty results and group ordering. This is the operation `firepanda/hash` was written for and it is the largest piece of M1.

The minor bump is for the new `firepanda.kernel.group` and `firepanda.hash.grouping` modules and for `AggSpec` in `firepanda.frame`. Nothing that existed in 0.3.0 changed shape.

### Added

- `firepanda.kernel.group`: `AggKind` and eight grouped reductions, `group_sum`, `group_mean`, `group_min`, `group_max`, `group_count`, `group_first`, `group_last` and `group_size`, plus the erased entry points `aggregate_group` and `aggregate_group_any` that a frame calls when the dtype is a runtime value. All eight share five pointer level cores, so the typed and erased spellings run the same code rather than two copies of it.
- `firepanda.hash.grouping`: `Grouping` and `group_ordinals`, which turn one or more key columns into a dense ordinal per row by composing `factorize` rather than hashing the tuple. Each additional key packs the running ordinal against the new one and refactorizes, which keeps the ordinal space bounded by the key tuples actually observed rather than by the cross product.
- `firepanda.frame`: `DataFrame.group_by`, `DataFrame.group_agg` and `DataFrame.group_count`, plus `AggSpec` to say which column, which reduction and what to call the result.
- `Factorized.into_codes`, which gives up the ordinals without copying them when the keys are not wanted.
- `firepanda.kernel.scalar.group_scalar`, the twin, which materializes each group's values into a list and reduces it with a plain loop. `tests/fuzz/kernel.mojo` now checks all eight reductions against it, one kind per case.
- Tests: 40 unit tests covering the null policy of each reduction one at a time, the multi-key ordinal combination, and the frame level behaviour of `dropna`, `sort` and output naming.
- Benchmarks: sixteen `group/*` rows, arranged so the interesting numbers are subtractions between neighbouring rows.

### Changed

- `tools/bench_compare.py` requires a fixed cost benchmark to move by an absolute margin as well as a percentage. `dispatch/call_1_row` measured 4.0 ns and 7.2 ns on two CI runs of unrelated changes, +82%, while the same two binaries measured 2.817 ns and 2.796 ns against each other on a dedicated machine. A percentage is the wrong instrument on a benchmark that reports a few nanoseconds, and the item count is what separates those from throughput rows, where an absolute margin would mask a genuine doubling.
- CI runs `tools/bench_compare.py` with a new `--advisory` flag, which prints the table and the verdicts and exits zero. `benchmarks/main.mojo` is one compilation unit and the benchmark bodies inline what they measure, so appending the sixteen `group/*` rows to the end of it changed which of the loops above them get vectorized: `array/sum_scalar` went from 386 us to 740 us and `bitmap/or_with` from 58 us to 92 us, in files this release does not touch, reproducibly to within half a percent across two runners. Every row measuring through a boundary the compiler will not inline across held still, and on the reference machine `kernel/sum_dense` and `kernel/sum_sparse` came out at 110.776 us and 110.855 us, which is the invariant those rows exist to check. The numbers are real, the attribution is not, and no gate can tell the two apart. The performance gate with teeth is the reference machine run recorded in each pull request, where the same tool runs without the flag.
- `_check_codes` in `firepanda.kernel.group` and the ordinal scan in `firepanda.hash.grouping` both use `max_of` rather than a scalar loop. A scalar scan of a million codes measured at roughly 350 us on the reference machine, which was more than the reduction it was guarding: `group/sum_dispatched` went from 960 us to 453 us and `group/frame_two_keys` from 11.1 ms to 8.8 ms.

### Notes on the numbers

Measured on the reference machine, a 16 core x86_64 with 32 byte SIMD, at 1,048,576 rows and 1,000 groups.

- `group/sum` is 348 us against 105 us for the ungrouped `kernel/sum_dense`. The 3.3x is what the scatter costs: a grouped sum reads a code, indexes an accumulator and writes it back, where an ungrouped one accumulates into a register.
- `group/sum_cardinality_10` is 374 us and `group/sum_cardinality_100k` is 805 us over the same rows and the same loop. The 2.2x is the accumulator array leaving cache, and it is the reason `firepanda/hash/partition.mojo` exists.
- `group/min` is 627 us dense and `group/min_sparse` is 1.654 ms. The difference is reading the validity bitmap a bit at a time, which `group/sum` never has to do because a null holds a zero. `mean`, `min`, `max`, `first` and `last` all pay it and all could be reading a validity word at a time instead. That is the next thing worth fixing in this file.
- `group/ordinals_one_key` is 2.534 ms of the 3.678 ms that `group/frame_one_key` takes, so on one integer key the grouping is 69% of the work and the reduction is the rest.
- `group/ordinals_two_keys` is 8.553 ms against 2.534 ms for one key, which is more than 2x because a second key costs two factorize passes rather than one plus a pass to pack them.

### Known limitations

- `std`, `var`, `median`, `quantile` and distinct count are not implemented. They need either a second pass or a buffer per group and they are the second half of the group by scope on the M1 issue.
- No joins, no IO, no strings as group keys.
- `group_ordinals` copies each key column once, because `factorize` reduces its input with `min_of` and `max_of` and those take an `Array` rather than a pointer. Rewriting that path to pointer form would remove a full column copy per key.
- The chained `df.groupby("k").sum()` spelling does not exist. `group_by` takes the keys and the reductions in one call, because the intermediate object needs either a borrow that outlives the expression or a copy of the frame, and neither is available until the plan layer at M4.

## [0.3.0] - 2026-08-26

Built against Mojo 1.0.0 (ed45d567).

The dataframe. `DataFrame` and `Series` exist, they hold real columns, they do the dozen operations that everything else is built out of, and they print themselves. This is the first release where the package does something a pandas user would recognise as the point of it.

The minor bump is for the new `firepanda.frame` package and for four new public functions in `firepanda.kernel`. Nothing that existed in 0.2.0 changed shape.

### Added

- `firepanda.frame`: `Series` and `DataFrame`, eager, positional and immutable. Column access, `select`, `drop`, `rename`, `with_column`, `cast`, `filter`, `take`, `slice`, `head`, `tail`, `argsort`, `sort_by` and `sort_values`.
- `firepanda.frame.display`: `render_table`, `render_column`, `render_value`, `format_float` and `DisplayOptions`. A frame prints as a table with a header, an integer index, right aligned cells and a shape line, and elides the middle of both axes when there is too much to print. `DataFrame.describe` reports the shape and the schema without rendering any values, which is what `write_to` used to do.
- `firepanda.kernel`: `take_any`, `filter_any`, `cast_any` and `argsort_any`, the type erased entry points a frame needs because its columns have different dtypes from each other. They share a body with the typed kernels rather than duplicating one, so the scalar twin still covers both.
- `AnyArray.slice`, which needs no dtype dispatch at all because a slice moves bytes without looking at them.
- `tools/probes/cast_matrix.mojo`, a compile budget probe for the first two sided dispatch in the package. `cast_any` instantiates 144 copies of its loop and they cost 72 KB and 1.3 seconds of compile time over a probe that dispatches over nothing.
- Tests: 61 unit tests covering the frame invariants, the error paths, the erased dispatch over all twelve dtypes, and the rendered output compared whole rather than probed at.
- Benchmarks: eleven `frame/*` rows, each paired with the kernel row underneath it so the frame layer's overhead is a number rather than an assertion.

### Changed

- Printing a `Series` or a `DataFrame` now writes the values. Both used to write a one line summary, which was a placeholder until this release.

### Notes on the numbers

- Rendering costs what it prints and not what it holds. `frame/render` measures 5.25 us on a frame of 1,024 rows and 5.43 us on the same frame at 1,048,576 rows, a thousandfold increase in height for three percent more work, because only the cells that appear are ever built.
- A null and a `NaN` print differently, as `<NA>` and `NaN`. pandas spells a null either way depending on the dtype backing the column. Every firepanda dtype is nullable through the validity bitmap and a float column can genuinely hold a `NaN`, so the two have to be distinguishable.

### Known limitations

- An operation that changes one column still copies the ones it did not touch, because a frame owns its columns outright. `frame/cast_one` measures 4.6 ms against 0.5 ms for the same cast at the kernel layer, and the difference is copying the other two columns. Sharing immutable columns by reference count is the fix.
- Fetching a column by name copies it, at 431 us on a million rows, where fetching by position borrows it at 0.9 ns. A borrowing accessor needs the column index in its return type, which a name lookup cannot supply until the plan layer resolves column references ahead of time.
- No group by, no joins, no IO.
- Strings still have a layout and no kernels, so a string column cannot be sorted or hashed, and the renderer prints `<string>` in place of a value rather than the value.

## [0.2.0] - 2026-08-26

Built against Mojo 1.0.0 (ed45d567).

Sorting. A dataframe still does not exist, but the operation that a dataframe spends the most time on outside of group by now does, and it is faster than the standard library's sort on the only comparison that can be made today.

The version is a minor bump rather than a patch because `firepanda.kernel` gained public functions. Nothing that existed in 0.1.0 changed shape.

### Added

- `firepanda.kernel.sort`: `argsort`, `argsort_into`, `argsort_multi`, `sort_rows` and `is_sorted`. A least significant digit radix sort on eight bit digits, stable, with null placement at either end and a direction per key.
- `sort_key` maps every numeric dtype and `bool` onto an unsigned integer whose ordering is the dtype's ordering, so a float is radix sorted exactly rather than approximately. Negative zero sorts below positive zero and NaN sorts above every finite value, both matching numpy.
- Every digit's histogram is counted in a single read of the keys, and a digit whose values are all identical is skipped, so an int64 column of small positive values costs two passes rather than eight.
- `firepanda/kernel/scalar.mojo`: `argsort_scalar`, an insertion sort comparing values with `<`. It does not go through `sort_key`, so the transform is checked rather than assumed.
- Tests: 35 unit tests, and the kernel fuzz harness now checks `argsort` against its twin on the permutation itself rather than on the sorted values, which makes it a stability check as well.
- Benchmarks: eight `sort/*` rows covering the pass count, null handling, multi-key sort and the standard library as a reference point.

### Known limitations

- A column that arrives already sorted is the slowest input the sort has, at 32.7 ms against 8.7 ms for a random column of the same value range and the same three radix passes. The cause is the scatter, not the pass count: sequential input visits the 256 write cursors in strict round robin and each one is evicted before it comes round again. Staging the writes through a per bucket buffer is the fix and it is not in this release.
- There is no comparison sort, so only the numeric dtypes and `bool` can be sorted. Strings arrive with the string kernels.
- Still no `DataFrame`, no `Series`, no IO.

## [0.1.0] - 2026-08-26

Built against Mojo 1.0.0 (ed45d567).

The first tagged release. There is no dataframe in it. What it contains is the layer a dataframe is built out of, plus the parts of the compute layer that sit directly on top: bitmaps, buffers, columns, the type lattice, the kernels, and the hash table that group by and join will use. It is tagged because the pieces underneath are now stable enough that changing them would be a break worth writing down, not because any of it is usable as a dataframe yet.

Install it and you get a library with no public API to speak of. The point of the tag is the version identity: the Mojo ABI is not stable within 1.x, so a build is identified by its own version and by the toolchain that produced it, and that pairing needs somewhere to start.

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

### Known limitations

- No `DataFrame`, no `Series`, no IO. See the status notice in the README.
- `factorize` loses to a `Dict` based implementation by about 1.3x on columns with a hundred or ten thousand groups, and beats it by 2.6x when every row is distinct and by 3.6x when the integer range is small enough to skip hashing. The tracking issue for M1 has the numbers and the reasoning.
- The string layout exists but no string kernels do, so a hash table keyed on strings is not possible yet.

[Unreleased]: https://github.com/tamnd/firepanda/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/tamnd/firepanda/releases/tag/v0.5.0
[0.4.0]: https://github.com/tamnd/firepanda/releases/tag/v0.4.0
[0.3.0]: https://github.com/tamnd/firepanda/releases/tag/v0.3.0
[0.2.0]: https://github.com/tamnd/firepanda/releases/tag/v0.2.0
[0.1.0]: https://github.com/tamnd/firepanda/releases/tag/v0.1.0
