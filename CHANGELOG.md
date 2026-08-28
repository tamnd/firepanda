# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

Pre-1.0, minor versions may break the API. Every break appears here with the migration, not just with a note that it happened.

The Mojo toolchain version is part of a release's identity and is recorded with each entry, because the Mojo ABI is not stable within 1.x and a binary artifact built against one runtime is not guaranteed to load against another.

## [Unreleased]

## [0.6.3] - 2026-08-28

Built against Mojo 1.0.0 (ed45d567).

The variable width string column, which is the last thing a CSV reader was missing and the reason 0.6.2 shipped a scanner with nowhere to put a text field. Plus the continuous integration work that took the pull request pipeline from about twelve minutes to about three.

A patch bump. The column is a new module and the pipeline changes are not API.

### Added

- `firepanda/array/strings.mojo`, holding `StringArray` and `StringBuilder`. A column is a views buffer of sixteen bytes per element, a payload buffer holding the bytes of the long elements, and a validity bitmap. An element of twelve bytes or fewer lives entirely inside its own view, so a column of country codes, status labels or short names is one flat array with no second buffer touched at all. A longer element keeps its length and its first four bytes in the view and its bytes in the payload.
- `StringBuilder`, which is the only way to make a column. Append a field, append a null, ask for the result. `finish` consumes the builder and hands its payload buffer to the column rather than copying it again.
- `StringArray.unsafe_bytes`, which returns a `Span` into whichever buffer the element lives in, so a kernel can read an element without allocating a `String` for it. The span is tied to the column's origin, so nothing it produces can outlive the column.
- `equals` for comparing an element against a run of bytes, and `element_equals` for comparing two elements of the same column. Both are the operations a join key or a group by needs.
- `slice`, `take`, `filter`, `to_list` and a deep copy, all of which produce an independent column.
- Twenty four tests in `tests/test_strings.mojo`, including every length from zero to thirty, the inline limit approached from both sides, a null after a long element, and two thousand random elements with nulls mixed in.
- `tests/fuzz/strings.mojo`, a fifth fuzzer, which checks the column against a `List[String]` reference through random rounds of slice, take, filter, copy and rebuild. A third of the lengths it draws land within two bytes of the inline limit.
- Ten benchmark rows under `strings/`, every one of them measured at eight bytes and at thirty two so the two storage paths are always side by side.

### How it works, and what follows from it

The classic Arrow string layout gives every element an offset into one data buffer, so reading any string is two dependent loads and knowing its length is two more. This layout puts the length and a four byte prefix in the element itself. A length costs one load whatever the element is, and two long elements that differ in their first four bytes are settled without either payload being read.

The cost is sixteen bytes per element against Arrow's four or eight. On a column of paragraphs that is a loss. On the short repeated text that dataframes are actually full of it is a large win, and short is the case worth optimizing for.

A finished column has exactly one payload block, so every long view carries block index zero. That field is not wasted. It is what will let a chunked string column share payload across chunks later without rewriting a single view.

Slicing, taking and filtering copy rather than pointing into the source column's payload. This is the same decision `Array` and `Bitmap.slice` already made, for the same reason: a view into another column's payload would make every column's lifetime depend on every column it was ever cut from.

### Notes on the numbers

Measured on gamingpc, sixteen physical cores, thirty two byte registers, at 262144 elements, ten repetitions, median reported.

```
strings/build_short          566.130 us     4.2%     8.638 ns   115.76 Mrows/s
strings/build_long           959.114 us     2.9%    14.635 ns    68.33 Mrows/s
strings/length_short          12.987 us     1.8%     0.198 ns     5.05 Grows/s
strings/length_long           12.385 us     3.0%     0.189 ns     5.29 Grows/s
strings/bytes_short           53.789 us     0.8%     0.821 ns     1.22 Grows/s
strings/bytes_long            63.092 us     1.7%     0.963 ns     1.04 Grows/s
strings/equals_prefix        148.520 us     0.9%     2.266 ns  441.25 Mpairs/s
strings/equals_payload       240.393 us     0.6%     3.668 ns  272.62 Mpairs/s
strings/take                   1.250 ms     6.3%    19.075 ns    52.42 Mrows/s
strings/filter               455.942 us     0.6%     6.957 ns  143.74 Mrows/s
```

`length_short` and `length_long` land on top of each other at a fifth of a nanosecond, which is the layout's first claim and the one the offsets layout cannot make. Asking a thirty two byte element how long it is costs exactly what asking an eight byte element costs.

`bytes_short` against `bytes_long` does not show the gap it looks like it should. Both walks are in order and a prefetcher handles two sequential streams as easily as one, so reading the payload is nearly free here. The inline layout is worth nothing to a scan. It is worth something to everything that jumps.

`equals_prefix` against `equals_payload` is where it pays. Both compare adjacent long elements that are not equal and both answer false, and the only difference is whether the difference is in the four bytes the view already holds. It is 1.6x on this machine and 1.9x on the eight core server, on elements of thirty two bytes with the payload in cache. Neither of those is the interesting case. The interesting case is a join key column that does not fit in cache, where settling a comparison in the view is a cache miss that does not happen.

Two things were slower before they were measured. The payload started as a `List[UInt8]` appended one byte at a time, which cost 143 ns per row on a long build, and `take` of a quarter of a million rows took 42 ms. Replacing the loop with one copy per field and a `resize` made it worse, not better, because `resize` allocates exactly what it was asked for and turns a growing payload into a quadratic copy: the same `take` went to 3.2 seconds. The payload is now a buffer that doubles, and the build is 14.6 ns per row.

### Changed

- The test runner runs several files at once rather than one after another. Each test file is a separate `mojo run` that compiles the library again, so the step was spending most of its wall clock in the compiler with one core busy. Output is still collected per file and printed in filename order, so the log reads the same. `FIREPANDA_TEST_JOBS=1` restores the serial run. On an eight core machine the suite went from 70.7 s to 26.9 s.
- The fuzzers run at once for the same reason, through `tools/run_fuzz.sh`. `--max-total-time=N` still means N seconds per fuzzer rather than N in total.
- Continuous integration no longer builds against the Mojo nightly toolchain on every pull request. `.github/workflows/nightly.yml` already does that every morning and opens a tracking issue when it breaks, and the duplicate was the slowest job in the pull request pipeline.
- The microbenchmark job measures the previous commit on pushes to main rather than the merge base on every pull request. Measuring both sides doubled the job, and what it bought was a comparison the job already prints as advisory. The performance gate with teeth is the reference machine run recorded in each pull request.
- The benchmark harness takes a `--max-time` flag, in milliseconds, for the wall clock ceiling on one repetition. It defaults to what the harness has always used, three times the minimum plus a quarter of a second, so nothing changes for a developer who does not pass it. It exists because that ceiling is what the suite's runtime is actually made of: `run` keeps sampling until the ceiling rather than stopping at the minimum, so the total is roughly the ceiling times the benchmark count times the repetitions, and the row count barely enters into it. Halving the rows moved a full run from 310 s to 276 s. Halving the ceiling halves it.
- The link check retries and accepts a rate limit response rather than failing the build on it. It failed a pull request because one host reset one connection, which is a flaky check rather than a broken link.
- Continuous integration runs the suite at five repetitions with a twenty millisecond minimum and a forty millisecond ceiling, which is 80 s against 316 s for the settings it used before. The measured spread hardly moves: median interquartile range 13.7% against 12.1%, ninetieth percentile identical at 50%. On a shared runner the noise floor is the machine rather than the sampling budget.

### Known limitations

- The column is not wired into `AnyArray`, `Series`, `DataFrame` or the display layer, so nothing that takes a frame can hold one yet. That is the next change and it is what actually unblocks `read_csv`.
- There is no comparison other than equality. Ordering strings needs a lexicographic compare that uses the prefix, and sorting or grouping by a string column needs that first.
- `element_equals` on two long elements with the same prefix walks the payload a word at a time. A column of URLs, where thousands of elements share their first four bytes, gets nothing from the prefix and pays the full walk.
- The bytes are not validated as UTF-8, deliberately. A CSV field is bytes and a column that refuses to hold what the file contains cannot read the file.

## [0.6.2] - 2026-08-28

Built against Mojo 1.0.0 (ed45d567).

The two halves of a CSV reader that do not need a frame: a scanner that finds where every field in a buffer is, and parsers that turn a field's bytes into an integer, a float or a boolean. There is no `read_csv` yet, because there is still no variable width string column for a text field to land in, and building that column and the reader and these kernels in one change would have made a pull request nobody could review.

A patch bump. Everything here is a new module and nothing that existed changed shape or behaviour.

### Added

- `firepanda/io/scan.mojo`, an RFC 4180 field scanner. `scan_csv` takes a span of bytes and a `Dialect` and returns a `Scan`, which is one flat list of field spans plus a list of row offsets into it, the same shape as an Arrow offsets buffer and for the same reason: one allocation for the file rather than one per row. A field span carries its start, its end and a flag saying whether it contains a doubled quote, so the reader only pays for unescaping on the fields that need it.
- `firepanda/io/parse.mojo`, the field parsers. `parse_int` handles an optional sign and digits with the range check done against the target dtype rather than against `Int64`, so `parse_int[DType.int8]` refuses 128 and `parse_int[DType.uint32]` refuses a negative. `parse_float` handles a sign, a decimal point, an exponent and the words `nan`, `inf` and `infinity` in any case. `parse_bool` takes `true` and `false` in the three usual capitalisations and nothing else. `is_missing` recognises the empty field along with `-`, `na`, `n/a`, `nan`, `nil`, `null` and `none`.
- `firepanda/io/scalar.mojo`, a byte at a time scanner that produces byte identical output to the vectorized one, so a block boundary landing in the middle of a field cannot pass a test unnoticed.
- `field_bytes` and `unescape` in `firepanda/io/scan.mojo`, for turning a span back into bytes with and without the doubled quotes collapsed.
- Forty six tests across `tests/test_parse.mojo` and `tests/test_scan.mojo`, including four hundred random buffers drawn from an alphabet loaded with commas, quotes, newlines and carriage returns, on which the two scanners have to agree on whether the buffer is legal and on every field boundary in it.
- Eight benchmark rows under `csv/`, covering a narrow file, a wide one, a quoted one, a long text one, and the two scanners run against each other on both the narrow and the long shapes.

### How it works, and what follows from it

A parse failure is a value and not an exception. `parse_int` and friends return a `Parsed[dt]` holding a value and an `ok` flag, because a reader with a bad field in row four million wants to record which row it was and carry on, and an exception per bad field would cost more than the parse.

Nothing is guessed. A field with trailing bytes after the number is a failure rather than a truncated value, so `12abc` does not become 12. A quoted field that never closes is an error naming the row and the byte, and so are bytes sitting between a closing quote and the next delimiter. Readers that accept those files do it by inventing a value, and an invented value in a data file is worse than a failed read.

The integer path is exact by construction and the float path is exact in the common case. The integer accumulator checks for overflow before the multiply rather than after it, so it never relies on wraparound. The float path builds the mantissa as a `UInt64` and applies the decimal exponent in one multiply or divide when the mantissa fits in 53 bits and the exponent is within 22, which is the range where a single rounding is provably correct, and falls back to scaling in exact steps of 1e22 outside it.

The scan is a separate pass over the buffer rather than a parse as it goes loop. That is one more pass over the text, and it buys three things: the second pass walks a compact offsets array instead of text, each column's parse knows its dtype and can be a tight typed loop instead of a switch, and the row count is known before a single column is allocated so nothing has to grow.

Blank lines are skipped, which is what pandas does and what stops a file ending in a newline from producing a phantom last row. Ragged rows are reported through `Scan.is_ragged()` rather than refused, because what to do about them is the reader's policy question and not the scanner's.

### Notes on the numbers

Measured on the reference machine, 16 physical cores, 32 byte registers, at 1,048,576 rows for the narrow shapes and 262,144 for the long one, 10 repetitions, median with the interquartile range next to it.

```
csv/scan_narrow               10.293 ms    34.8%    39.263 ns    25.47 Mrows/s
csv/scan_scalar_twin          10.419 ms     3.8%    39.745 ns    25.16 Mrows/s
csv/scan_long_text             3.049 ms     2.7%    46.523 ns    21.49 Mrows/s
csv/scan_long_twin             6.045 ms     5.9%    92.238 ns    10.84 Mrows/s
csv/parse_int                  3.165 ms     2.9%     6.037 ns 165.65 Mfields/s
csv/parse_float                3.061 ms     3.9%    11.676 ns  85.64 Mfields/s
```

The first version of the scanner was slower than its own byte at a time twin, and the benchmark is what caught it. A register can be tested against the delimiter, the newline and the carriage return in three instructions, but there is no packed movemask reachable from this stdlib, so a register that hits still has to be walked byte by byte to find the lane. In a file whose fields are five bytes long every register hits, which means the vector compare is added to the byte walk rather than replacing it.

The fix is that a search walks the first eight bytes one at a time and only starts testing registers once a field has proved it is longer than a word. Narrow files now scan at the same speed as the scalar scanner, 10.293 ms against 10.419 ms, and long text fields scan at twice its speed, 3.049 ms against 6.045 ms. Eight was measured rather than assumed; a whole register as the threshold gave up most of the win on the long fields and bought nothing on the short ones.

Both parsers are far cheaper than fetching the field they parse, 6.0 ns and 11.7 ns per field, which is the right shape for what comes next: once there is a reader, the cost will be in the scan and the allocation, not in turning digits into numbers.

### Known limitations

There is no `read_csv`, no schema inference and no writer. Those need a variable width string column, which is the next change.

The scanner assumes the buffer is one whole file or a block already split on a row boundary. Splitting a file into blocks for parallel scanning has to respect quoted newlines and is not done here.

Quoting is RFC 4180 only. There is no backslash escape mode and no comment character.

## [0.6.1] - 2026-08-28

Built against Mojo 1.0.0 (ed45d567).

Concat and the null handling functions. Stacking columns, series and frames, and the five things everybody does with a missing value: ask where they are, replace them from somewhere else, carry the last value forward, carry the next one back, and drop the rows.

A patch bump. Everything here is new surface and nothing that existed changed shape or behaviour.

### Added

- `firepanda.kernel.concat`: `concat_arrays` for a list of typed columns, `concat_any` for a list of erased ones, and `concat_two_any` for the two argument case that borrows rather than owns.
- `firepanda.kernel.nulls`: `is_null` and `is_not_null` in typed and erased spellings, `coalesce`, `fill_forward`, `fill_backward` with an optional run limit, and `all_valid_mask` for the intersection of several columns' validity.
- `firepanda.frame.concat`: the free functions `concat` for frames and `concat_series` for series, both re-exported from `firepanda`.
- `Series.is_null`, `Series.is_not_null`, `Series.drop_nulls`, `Series.fill_null`, `Series.fill_forward` and `Series.fill_backward`.
- `DataFrame.drop_nulls`, taking an optional subset of column names, and `DataFrame.fill_null`.
- Scalar twins in `firepanda.kernel.scalar`: `concat_scalar`, `coalesce_scalar`, `fill_scalar` and `is_null_scalar`, on the same terms as the twins already there.
- Tests: 21 in `tests/test_concat.mojo` and 32 in `tests/test_nulls.mojo`.
- Benchmarks: seven `nulls/*` rows and three `concat/*` rows.

### Changed

- `firepanda.join.pairs` no longer carries its own private concatenation. It keeps its own dtype error message, which is about key columns rather than about columns in general, and then calls `concat_two_any`. The kernel copies values a SIMD register at a time where the join's copy went element by element, so the join benchmarks came out level to slightly better: `join/indices_1000` 12.046 ms against 11.773 before, `join/inner_1000` 19.337 against 19.562, `join/many_to_many` 1.741 against 1.870.

### How it works, and what follows from it

- A fresh `Array` is zeroed and marked all present, so a part with no nulls costs concat nothing beyond the value copy. Only a part that actually has nulls pays a validity pass, and that pass runs a word at a time and skips any word that is all ones.
- The validity repair in concat runs a bit at a time rather than a word at a time, because the destination offset is a running total of the earlier parts' heights and is almost never a multiple of 64. Shifting and masking a source word into a destination that straddles two words is the faster spelling and is not written yet.
- Filling with a scalar and filling from a column are the same operation. A fallback of exactly one row broadcasts, so `fill_null` takes a `Series` in both cases rather than growing an erased scalar type that would have to carry its own dtype tag. That is also what SQL `COALESCE` already means.
- `fill_forward` and `fill_backward` are one core with a `comptime` direction rather than two loops, so the two directions cannot drift apart. A `limit` of zero means no limit, and the run counter resets at every present value rather than at the start of the column.
- Nothing here promotes. `coalesce` of an int32 column and a float64 fallback is refused, for the same reason concat and join refuse it.
- Frame concat matches columns by name, not by position, so a frame whose columns are in a different order still stacks. Stacking by position would put two different meanings in one column with nothing at the call site to catch it.
- `drop_nulls` implements only pandas' `how="any"` rule. `subset` is the control that matters and `how="all"` is a filter anybody can write in one line against `is_null`.
- There is no horizontal concat. Putting two frames side by side means deciding which row of one lines up with which row of the other, and without an index the only answer available is position. It waits until there is something to align on.

### Notes on the numbers

Measured on the reference machine, a 16 core x86_64 with 32 byte SIMD, at 1,048,576 rows over 10 repetitions. Sparse means one row in eight is null.

- `nulls/is_null` is 34.524 us on a column with no nulls and `nulls/is_null_sparse` is 380.886 us on one that has them. The 11x gap is the whole value of the word at a time scan: an all ones or all zeros validity word turns into a block store of the same answer, and only a mixed word is walked bit by bit.
- `nulls/coalesce` is 447.310 us and `nulls/coalesce_sparse` is 1.951 ms. `nulls/ffill` is 441.804 us and `nulls/ffill_sparse` is 1.955 ms. Both operations show the same 4.4x, which says the cost is the repair pass over the missing rows and not the copy that precedes it, and that the two operations are doing the same amount of work per missing row.
- `concat/eight_parts` is 1.723 ms against `concat/two_parts` at 1.694 ms for the same total height. Within 2%, so per part overhead is not measurable and the operation is memory bandwidth and nothing else.
- `concat/frame_three_columns` is 3.820 ms, a shade over twice the single column figure for three columns, so the schema walk and the name lookups cost nothing worth naming.
- `nulls/drop_nulls` is 6.885 ms, which is `is_null` plus a `filter` over three columns, and the filter is all of it.

### Known limitations

- Frame concat of three or more frames copies every column into a `List[AnyArray]` before concatenating, because the list spelling owns its parts. The two frame call, which is the common one, borrows and does not. Reference counted columns would remove the copy from both this and `DataFrame.drop_nulls`.
- `fill_forward` and `fill_backward` are serial by construction, since a filled value can depend on one arbitrarily far back. Splitting the column into blocks and resolving the carry between them is the parallel form and is not written.
- `coalesce` reads the fallback for every missing row rather than gathering the missing rows first, so a column that is mostly null pays a scattered read.

## [0.6.0] - 2026-08-26

Built against Mojo 1.0.0 (ed45d567).

Joins. All seven kinds, on one or more key columns, from `DataFrame.join` down to the row pairing underneath it.

The minor bump is for a new top level package, `firepanda.join`, and for three new methods on `DataFrame`. Nothing that existed changed shape.

### Added

- `firepanda.join`: `JoinKind` with `INNER`, `LEFT`, `RIGHT`, `OUTER`, `SEMI`, `ANTI` and `CROSS`, `JoinIndices` holding the paired row numbers, and `join_indices` producing them from two column lists and a set of keys.
- `DataFrame.join` for keys named the same on both sides, `DataFrame.join_on` for keys named differently, and `DataFrame.cross_join`.
- `firepanda.join.scalar.join_nested`, the nested loop twin, on the same terms as the twins in `firepanda/kernel` and `firepanda/hash`: it is never called in production and it is what the fast path is checked against.
- `tests/fuzz/join.mojo` and the `fuzz-join` pixi task, wired into `pixi run fuzz`. Two million cases pass in forty seconds.
- Tests: 34 in `tests/test_join.mojo`, covering each kind's row set, the null rule, the column naming rules and the row order.
- Benchmarks: nine `join/*` rows covering the pairing on its own, four of the kinds, dimension size, two keys, and the many to many case.

### How it works, and what follows from it

- The two frames are aligned by concatenating each key column with its opposite number and handing the result to `group_ordinals`. Two rows share a code exactly when they share a key tuple, whichever side they came from, and the multi-key packing and the small integer fast path come along unchanged. The cost is one extra pass over each key column plus the memory to hold the copy.
- A row whose key contains a null matches nothing, including another null. That is SQL's rule and Polars' default. pandas `merge` joins NaN keys together and this deliberately does not, because firepanda has a validity bitmap and does not need to overload a float value to mean missing. The rows are not dropped: an unmatched left row still survives a left join or an anti join.
- The result is in left row order, and within a left row, in right row order. A right join is the same operation with the sides exchanged, so it comes out in right row order. This is fixed rather than incidental, because a join whose row order moves between runs cannot be compared against another engine.
- A key column that both frames call by the same name appears once in the output and is filled from whichever side had the row. That only matters for a right or outer join, where an output row can have no left row at all, and taking the key from the left would put a null in the column the row was matched on.
- Key columns must have the same dtype on both sides. Promoting them here would mean a join silently finding fewer matches than either side expected, with nothing on screen to say why, so the cast is the caller's to write.

### Notes on the numbers

Measured on the reference machine, a 16 core x86_64 with 32 byte SIMD, at 1,048,576 fact rows against a 1,000 row dimension unless stated.

- `join/indices_1000` is 11.773 ms and `join/inner_1000` is 19.562 ms. The first is the pairing alone and the second is the whole operation, so building the output columns is the larger half at roughly 2 ms per gathered column.
- `join/inner_100k` is 30.528 ms against 19.562 ms for the same fact rows using the same thousand keys. The result is identical and only the dimension's unused rows differ. The extra 11 ms is the key alignment: a dimension spanning a hundred thousand values puts `factorize` on a 400 KB direct table where the thousand row dimension fits in 4 KB, and every one of the 1.1 million concatenated rows pays the difference.
- `join/semi` is 19.185 ms and `join/anti` is 18.723 ms, both against `join/inner_1000` at 19.562 ms. The filtering joins gather nothing from the right and stop at the first match, and they are barely cheaper, because the key alignment dominates and they pay all of it.
- `join/outer` is 30.241 ms. The extra over inner is the bitmap write per matched row and the coalesced key column, which is a two source gather rather than a straight one.
- `join/two_keys` is 28.652 ms. The second key adds a factorize, a pack and a refactorize, which is the same 9 ms it adds to a two key group by.
- `join/many_to_many` is 1.870 ms for 262,144 output rows out of two 4,096 row frames, or 7.13 ns per output row, the cheapest per row in the set because both sources are cache resident.

### Known limitations

- The right side is bucketed with a counting sort over the group ordinals, which is one array as wide as the number of distinct keys in both frames together. For a join between two large frames with high cardinality that array is the working set, and radix partitioning the probe is what `firepanda/hash/partition.mojo` exists for.
- The key alignment copies each key column. Rewriting `factorize` to take a pointer rather than an `Array` removes that copy and is the same open item `group_ordinals` already has.
- A cross join materializes the full product and nothing refuses a large one. Any threshold would be arbitrary and would be in the way of the case the operation exists for.
- Joining on a float column keys NaN to NaN and negative zero to zero, which is `key_bits` and is the same rule group by uses.

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

[Unreleased]: https://github.com/tamnd/firepanda/compare/v0.6.3...HEAD
[0.6.3]: https://github.com/tamnd/firepanda/releases/tag/v0.6.3
[0.6.2]: https://github.com/tamnd/firepanda/releases/tag/v0.6.2
[0.6.1]: https://github.com/tamnd/firepanda/releases/tag/v0.6.1
[0.6.0]: https://github.com/tamnd/firepanda/releases/tag/v0.6.0
[0.5.0]: https://github.com/tamnd/firepanda/releases/tag/v0.5.0
[0.4.0]: https://github.com/tamnd/firepanda/releases/tag/v0.4.0
[0.3.0]: https://github.com/tamnd/firepanda/releases/tag/v0.3.0
[0.2.0]: https://github.com/tamnd/firepanda/releases/tag/v0.2.0
[0.1.0]: https://github.com/tamnd/firepanda/releases/tag/v0.1.0
