# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

Pre-1.0, minor versions may break the API. Every break appears here with the migration, not just with a note that it happened.

The Mojo toolchain version is part of a release's identity and is recorded with each entry, because the Mojo ABI is not stable within 1.x and a binary artifact built against one runtime is not guaranteed to load against another.

## [Unreleased]

### Added

- `firepanda/io/arrow_c.mojo`, the Arrow C Data Interface declared: `ArrowSchema`, `ArrowArray`, the flag constants, the format string in both directions, and the release protocol. No producer and no consumer yet, which is deliberate. This is the layer where a mistake is invisible until it is a wrong pointer read inside pyarrow, so it lands on its own with the layout pinned by tests before anything is built on it. The tests write a distinct value into every field and read the struct back as an array of eight byte words, which catches two fields swapping places, and they call an actual `abi("C")` function pointer through the release field, which is the mechanism the whole interface rests on and was the part in most doubt. `ArrowSchema` is seventy two bytes and `ArrowArray` is eighty, as the specification says.
- Three facts about Mojo 1.0 came out of building it and are written down in that file rather than rediscovered later. A C function pointer usable as a struct field is `def (args) thin abi("C") -> None`, where `thin` is what makes it a bare pointer instead of a trait a struct cannot hold. `Pointer` is not nullable, so every C field that may be null is `Optional[Pointer[...]]`, which is still eight bytes because a null pointer is the niche the discriminant packs into. That does not extend to function pointers, where `Optional` is sixteen bytes and would move every field after it, so the release fields are nullable void pointers with the reinterpretation done at the two ends.
- `firepanda/io/arrow_export.mojo`, the export half: `export_schema` and `export_array` hand a firepanda column to a C consumer without copying a value. `buffers[1]` is the column's own values pointer and `buffers[0]` is its own validity pointer, and the tests assert that by comparing addresses rather than contents, because contents compare equal for a copy too. Two coincidences make it possible and both are now asserted rather than assumed: firepanda's validity bitmap is one bit per row, least significant bit first, with one meaning present, which is exactly Arrow's, and a fixed width values buffer is the Arrow values buffer with no header on either side.
- Ownership is a heap box behind `private_data`, holding the column and the buffer pointer array, allocated with `malloc` and destroyed by the release callback. C's allocator rather than Mojo's, because the free happens inside a callback a foreign runtime invokes on a thread firepanda knows nothing about. `export_array` consumes its column, which is not a convenience choice: firepanda columns are deep copied rather than refcounted, so sharing one between a `DataFrame` and a consumer is not something this layer can honestly offer. Verified leak free under valgrind, zero bytes lost in any category and zero errors.
- Strings and binary export without copying either, which was not the expectation going in. A firepanda string column is already Arrow's view layout byte for byte: sixteen bytes per element, a little endian uint32 length, then the data inline when it fits in twelve bytes or a four byte prefix plus a uint32 buffer index and a uint32 offset when it does not. firepanda picked that layout for short string inlining and prefix comparison and landing on Arrow's was not one of the reasons, so a test reads the actual bytes of a short view and a long one rather than trusting the coincidence to keep holding. This is the case that would have hurt most to copy, because the payload of a text column is usually the largest buffer in a frame. Such a column exports as four buffers, validity, views, one payload block and the sizes, and the block is emitted even when every string inlines so that the count is a constant a consumer can rely on.
- Bool is the one type that is copied, and it is a packing pass rather than a copy in the usual sense. firepanda stores a bool as a byte because that is what a kernel wants to load and Arrow stores it as a bit, so the exported values buffer is a freshly packed bitmap that lives in the ownership box like everything else. It comes out eight times smaller than the column it was built from. A test asserts the exported pointer is not the column's byte buffer, which is the failure that would otherwise pass every test that only reads element zero.
- The null type is the only thing the exporter refuses, and it raises with the reason: firepanda has it as a `LogicalType` but has no column that carries it at run time, so there is nothing to hand over.
- `type_for_format` refuses `u`, `U`, `z` and `Z`, the offset based string formats, rather than quietly reading them as views. They are different memory from the view layout firepanda uses, and accepting one as the other is exactly the failure this file exists to prevent. Converting them is real work and belongs in the import path where it can allocate.
- `firepanda/io/arrow_import.mojo`, the other direction: `import_array` takes a schema and an array from a C producer and returns a firepanda column. It copies, and unlike the export that is not a temporary state of affairs. A firepanda buffer is 64-byte aligned and allocated in whole 64-byte blocks so that a kernel can read a full register past the last value it cares about, and Arrow requires eight bytes of alignment and says nothing about what follows the last byte. An Arrow array carries a row offset, so what arrives is often a slice of somebody else's column, and firepanda columns start at element zero. And a foreign view column may spread its long elements over any number of data buffers where firepanda has exactly one, so the views are rewritten regardless. Any one of the three would be enough on its own.
- The import accepts `u`, `U`, `z` and `Z` as well as the view formats, which is the promise the export side made when it refused them. That matters more than the symmetry does, because `u` is what pyarrow produces unless it is asked for views, so an import that took only `vu` would be an import that works with almost nothing. An element short enough to inline never reaches the payload at all, so a column of country codes arrives as a data buffer plus an offset per element and leaves as views alone.
- Both string paths are two passes rather than a `StringBuilder`. The lengths are all known before the first byte is read, which is exactly what a builder cannot assume, so the payload is sized once and each view is written straight into place. A builder doubles its payload as it goes and then copies every view a second time on the way out of `finish`, which on a ten million row column is a second pass over a hundred and sixty megabytes for nothing. Measured on an i9-13900K over a million elements of fourteen bytes: 3.58 ns an element for the offset layout against the 14.36 ns `strings/build_long` spends per element going through the builder.
- A producer with one data buffer and no row offset has handed over a column already shaped like ours, and that case is two memcpys: 3.50 ns an element against 7.07 for the same column split over two data buffers. The check pass still runs, because it is also where every view is tested against the length of the buffer it names. A view is three numbers a stranger chose and following one unchecked is an out of bounds read waiting for a malformed file. The offset layout gets the one check it allows, which is that the offsets go forwards, since a backwards pair is a negative element length and a negative length reaching a memcpy is an enormous one.
- Fixed width imports at 0.23 ns a row and bool at 0.36, the second being a bit at a time unpack into firepanda's byte per value. Two thirds of the fixed width number is the allocation rather than the copy: `buffer/alloc_fresh` alone is 156 microseconds for the eight megabytes that `arrow/import_int64` moves in 238.
- `import_array` releases both structures before it returns, including when it raises. A producer that has handed a structure over has no way to reclaim it, so a consumer that refuses the type still owes it the release call, and a refusal that leaked would make every unsupported column a slow leak in a loop over a directory of files. Since everything is copied the release happens immediately rather than being deferred, and nothing that comes out points into the producer's memory. Leak checked under valgrind alongside the export.
- Five benchmark rows under `arrow/`, covering the fixed width copy, the bool unpack, a view column in the shape our own exporter produces, the same column split over two data buffers, and the offset layout. The producers are built by hand rather than by the exporter, partly because the exporter consumes its column and a benchmark needs to hand over the same bytes every iteration, and partly because three of those shapes are ones firepanda can never produce.
- `AggKind.CORR` and `AggKind.COV`, the first two reductions that read a pair of columns rather than one. `AggSpec` gains a second column name and a constructor that takes it, so a correlation is a spec like any other and composes with the rest in one call: asking for a correlation and three sums over the same keys groups the rows once. The kernel centres both columns on their pairwise means before accumulating, the way `group_var` does and for the same reason, and pairwise means it: a row where either value is null contributes to neither mean, because a covariance is a statement about rows in which both were observed. A group with fewer than two such rows is null, and so is a correlation whose denominator is a zero. The erased entry point casts both columns to float64 and calls one instantiation rather than dispatching on both dtypes, which would be a hundred and forty four instantiations of a loop that reads its inputs as float64 in either case; the typed spellings stay generic and copy nothing.

## [0.6.23] - 2026-08-31

Built against Mojo 1.0.0 (ed45d567).

Two changes, one to the group by and one to the join, and both of them are the same observation from different ends: the library was choosing a general shape in places where it already knew the specific one.

The group by on several keys packs its keys into one integer per row and factorizes that at the end. Factorize decides between a direct table indexed by value and a hash table by scanning the column for its range, and it declines the table above sixty five thousand, because a scan is a measurement of data the library did not construct and a span of ten million says nothing about whether ten values or ten million occupy it. A packed key is not in that position. Its range is `g0 * g1 * g2 * ...`, computed on the way down out of ordinals that are dense by construction, and a range built that way out of dense parts is itself densely occupied. So there is now a `factorize_dense` for callers who can name the range, and its rule is the table against the column rather than the table against the cache: index it when the span fits in what the column already costs, which caps the direct route at four bytes a row. Two keys of a hundred by a hundred thousand is a span of exactly ten million on ten million rows, and it now indexes instead of hashing.

The join builds its right side by bucketing rows by code: count per code, prefix sum, scatter, undo the cursor. That is the general answer and it stays, because a right key can repeat and then a left row pairs with several. But a join onto a primary key has one right row per code, and there the counts, the prefix sum, the cursor walk and the bucket array all exist to say "one". The build now assumes the right key is unique and fills a single table from code to row in one pass, and the first code it finds already taken abandons that and runs the general build from the top. Being wrong costs part of one scan of the right side. Being right saves two walks of a table as long as the frame plus an array as long as it again, and the table is `int32` rather than `Int`, so the widest join in the suite carries forty megabytes where the general shape carried a hundred and sixty.

Ten million rows on an i9-13900K, v0.6.22 against this release, alternating builds, one process per measurement, median of the per round paired ratios.

| query | shape | 0.6.22 | 0.6.23 | ratio |
| --- | --- | --- | --- | --- |
| j5 | ten million to ten million, then aggregated | 777.5 ms | 625.8 ms | 1.28 |
| j4 | ten million to ten million | 753.3 ms | 618.5 ms | 1.21 |
| q6 | group by two keys, six million out | 627.8 ms | 522.0 ms | 1.19 |
| q2 | group by two keys, ten thousand out | 75.2 ms | 72.6 ms | 1.06 |
| j2 | ten million to a hundred thousand | 204.8 ms | 199.3 ms | 1.02 |
| q3 | group by one key, hundred thousand out | 138.3 ms | 135.6 ms | 1.02 |
| j1 | ten million to ten thousand | 155.8 ms | 152.7 ms | 1.01 |
| j3 | left join, ten million to a hundred thousand | 205.0 ms | 206.9 ms | 1.00 |
| q10 | group by six keys, ten million out | 1009.0 ms | 1015.7 ms | 0.99 |

The two halves land on disjoint queries, which is what their code predicts. q6 is the query the dense table was written for and j4 and j5 are the joins the unique table was written for, and neither change touches the other's shape. q10 is a control for the first and a control it stays: a space of ten to the eighteen declines the table on either route. j3 is a control for the second in the same sense, since its build side is a hundred thousand rows and there was never much there to save.

Two things learned in the measuring are worth carrying forward. The unique join route was first written as an `if` inside the two hot loops rather than as its own pair of loops, and in that form j4 and j5 gained about what they gain now while j3 read 0.898 and j1 read 0.994. The predicate is loop invariant and answers the same way on all ten million rows, and it is still a compare and a jump on each of them, so on the joins with a small build side it was pure cost. Splitting the loops put j3 back to 1.000 with nothing lost at the other end. And q10 first read 0.927 on eight rounds, which looked like a regression on the widest query and is not one: it is bimodal on this machine with modes near 980 and 1090 ms that both builds visit, and twenty two rounds put it at 0.974 with the two ranges overlapping across their whole width. A reading that contradicts the mechanism needs more samples before it gets a hypothesis.

### Added

- `factorize_dense`, for a caller that already knows the range its values are in. `factorize` learns the range by scanning, and what a scan finds is a bound on data the library did not construct, so it declines a direct table above `DIRECT_LIMIT` because a span of ten million says nothing about whether ten values or ten million occupy it. A group by on several keys is not in that position. Its packed key has a range of `g0 * g1 * g2 * ...`, computed on the way down out of ordinals that are dense by construction, and the shapes where that product is large are the shapes where most of it is occupied. So the bound for a caller who can name the range is the table against the column rather than the table against the cache, which caps the direct route at four bytes a row and never more. Two keys of a hundred by a hundred thousand is a span of exactly ten million on ten million rows, which now indexes a table instead of hashing.

### Changed

- The join build side takes one table from code to row when the right key is unique. Bucketing the right rows by code is the general answer, and it has to exist, because a key can repeat on the right and then a left row pairs with several. But a join onto a primary key has one right row per code, and there the counts, the prefix sum, the cursor walk and the bucket array are all machinery for saying "one". So the build assumes uniqueness and fills a single table in one pass, and the first code it finds already taken abandons that and runs the general build from the top. Being wrong costs part of one scan of the right side. Being right saves two walks of a table as long as the frame plus an array as long as it again, and the table is `int32` rather than `Int`, so the widest join in the suite carries forty megabytes where the general shape carried a hundred and sixty. The count and emit loops are written twice rather than branching on which build ran, because the branch answers the same way on all ten million rows and is still a compare and a jump on each of them: with the branch inside the loops j3 read 0.898, and with the loops split nothing goes backwards. Ten million rows on an i9-13900K, alternating builds, one process per measurement, eight rounds, median of the per round paired ratios: j5 777.5 ms to 625.8 ms for 1.278, j4 753.3 ms to 618.5 ms for 1.207, j2 1.018, j1 1.013, j3 1.000. j4 and j5 join ten million to ten million on a key that is one through n, which is the shape the table was written for. j1 through j3 join to a dimension of ten thousand or a hundred thousand, so their build side was small either way and the whole of the change there is the emit reading one `int32` instead of two `Int`s. The group by controls q6 and q10 sit at 1.003 and 0.989.
- The multi key group by asks for `factorize_dense` at the end and inside `_condense`, passing the space it already computed. Ten million rows on an i9-13900K, alternating builds, one process per measurement, median of the per round paired ratios: q6 627.8 ms to 522.0 ms for a ratio of 1.19 over eight rounds, and q2 75.2 ms to 72.6 ms for 1.06 over twenty two. q6 is the query the table route is for. q2 is a smaller and different win, since its space of ten thousand was under `DIRECT_LIMIT` and took the direct route already, and what it saves is the scan that used to walk ten million int64 values to learn a range the group by had computed. q10 is unchanged at 0.974 over twenty two rounds with the two distributions overlapping across their whole width, which is what its code predicts: a space of ten to the eighteen declines the table on either route, and the scan that no longer runs was stopping on its first few rows anyway. q3, q1 and j4 sit at 1.02, 1.01 and 0.99.

## [0.6.22] - 2026-08-31

Built against Mojo 1.0.0 (ed45d567).

The group by on several keys. Two changes, both of them work that was being done and then discarded, and on the widest shape in db-benchmark they take a third off the query.

Grouping on more than one key works by packing the keys into a single integer per row. Group on the first key and every row has an ordinal in `[0, g0)`; group on the second and it has another in `[0, g1)`; then `c0 * g1 + c1` names the pair exactly. Fold in a third key by multiplying by `g2` and adding, and so on down the list. Factorize the packed value at the end and the group by is done.

That packed value was being factorized after every single key it folded in, not just at the end. The reason was overflow: without the intermediate passes the running space is the product of all the group counts rather than the number of tuples actually present, and the concern was that the product would leave an int64. It does not, for anything anyone has. Nineteen digits is twenty six columns of a hundred distinct values, or six columns of a thousand. So the packing now happens in place and the factorize happens once, and `_condense` renumbers the running key into the tuples it actually holds if a column list ever does approach the bound. The bound is still checked before every multiply. A six key group by at ten million rows was eleven hashed passes over the full column, six for the keys and five to redensify after each fold, and it is now seven.

Separately, every hashed factorize was building a column of the distinct key values and returning it, and nothing in the library ever read it. A group by does not need it, because it gathers all of its key columns at once at the end by the rows that introduced each group. A join does not need it either, because it reads the ordinals and nothing else. So `Factorized` now carries those representative rows, which every route already had, and building the key values is something a caller asks for. On the six key query the old code produced five of those columns at eighty megabytes each and dropped all five.

Ten million rows on an i9-13900K, v0.6.21 against this release, alternating builds, six rounds of five runs each, one process per measurement, median of the per round paired ratios.

| query | shape | 0.6.21 | 0.6.22 | ratio |
| --- | --- | --- | --- | --- |
| q10 | group by six keys, ten million out | 1747.8 ms | 1132.0 ms | 1.56 |
| q6 | group by two keys, six million out | 811.3 ms | 663.4 ms | 1.21 |
| q2 | group by two keys, ten thousand out | 83.2 ms | 70.8 ms | 1.18 |
| j5 | ten million to ten million, then aggregated | 807.7 ms | 747.5 ms | 1.07 |
| j1 | ten million to ten thousand | 156.8 ms | 150.9 ms | 1.03 |
| j4 | ten million to ten million | 793.9 ms | 770.5 ms | 1.03 |
| q1 | group by one key, hundred out | 20.3 ms | 19.8 ms | 1.02 |
| q7 | group by one key, then a subtraction | 126.6 ms | 126.0 ms | 1.01 |
| j2 | ten million to hundred thousand | 206.5 ms | 204.8 ms | 1.01 |
| q5 | group by one key, hundred thousand out | 78.4 ms | 82.5 ms | 0.99 |
| q3 | group by one key, hundred thousand out | 135.2 ms | 138.3 ms | 0.98 |

The last two are inside the noise band this setup shows on changes known to do nothing, which is about three percent. q10 is not: its two distributions do not overlap, 1705 to 1784 ms before and 1022 to 1152 ms after.

Where the credit goes is worth being precise about, because the two changes help different shapes. The packing change is the whole of q10 and none of q6, since two keys never had a redundant fold to remove. The discarded key column is the whole of q6 and q2, since both produce enough groups for that column to be large. One key queries were never paying for either.

A note for anyone measuring this themselves: q6 on this machine is bimodal, with a fast mode near 600 ms and a slow one near 770 ms that every build visits, so its median moves by twenty percent depending on which mode the samples land in. Paired ratios are stable there and raw medians are not.

### Changed

- `Factorized` no longer carries a `keys` column. Read the key values with `Factorized.keys(col)`, passing the column that was factorized, which gathers them from the representative rows. Callers who were reading `.keys` as a field want `.keys(col)` as a call; callers who were ignoring it, which was everyone inside the library, want nothing.

## [0.6.21] - 2026-08-31

Built against Mojo 1.0.0 (ed45d567).

The join. Three changes, each taking a different pass off it, and together they roughly double it.

The pairing was doing all of its work on one thread. Deciding which left row goes with which right rows is a walk over the left side that carries no state between rows: what a row emits depends on that row and on the right side buckets, which are finished before the walk starts. It is now cut into slices, one per core, with a prefix sum of the per slice output counts in between telling each worker where its slice begins in the result. That prefix sum is why the counting walk, which already existed so the two index lists could be allocated once at the right size, had to be cut the same way. An outer join is left on one thread deliberately, because it has to record which right rows it paired so it can emit the rest afterwards, and that record is a bitmap whose set is a read modify write of a word eight rows share.

`take_rows` got the same treatment for the same reason. It is what actually builds the output columns of a join, once per column, and a gathered row depends on its own index and nothing else in the output. The only part of that loop which is not per row is the validity bitmap, which is built in a register and stored once every sixty four rows, so the slice boundaries round up to a multiple of sixty four and no two workers write the same word.

Then two passes that did not need to happen. The pairing built a per row flag over both frames saying whether that row's key tuple contained a null, one branch per key per row, and answered no every time on a frame that has no nulls; asking each key column for its null count first is a popcount per validity word. And the gather read the source's validity bit for every row, a second random read into a different array from the values, which a column with no nulls does not need either. Between them these are the largest single wins here relative to the code they replace, and neither is clever.

Last, the bucket build stopped allocating a second copy of the group table. The scatter needs a cursor per group, and on a join between two frames that are mostly one to one that table is as long as the frames. The offsets can be their own cursor: group `g` is written from `starts[g]` to `starts[g + 1]`, so afterwards each entry holds what its successor held, and one backwards pass puts them back.

Ten million rows on an i9-13900K, v0.6.20 against this release, alternating builds, six rounds of five runs each, one process per measurement, median of the per round paired ratios.

| query | shape | 0.6.20 | 0.6.21 | ratio |
| --- | --- | --- | --- | --- |
| j5 | 10M to 10M, then aggregated | 1745.6 ms | 758.5 ms | 2.30 |
| j4 | 10M to 10M, one match a row | 1818.4 ms | 746.8 ms | 2.18 |
| j1 | 10M to 10k, one match a row | 304.0 ms | 155.2 ms | 1.99 |
| j2 | 10M to 100k, inner join | 341.6 ms | 202.5 ms | 1.75 |
| j3 | 10M to 100k, left join | 343.6 ms | 209.2 ms | 1.56 |
| q10 | six key group by, no join | 1752.7 ms | 1728.1 ms | 1.01 |
| q3 | string key group by, no join | 136.8 ms | 139.7 ms | 1.00 |
| q1 | one key group by, no join | 19.8 ms | 20.7 ms | 0.95 |

The three group by queries are controls and none of them touches a join.

What is left in the j4 pairing is the factorize of the concatenated twenty million row key column, which is around 390 ms of the 747, and the scatter itself, which is ten million random writes into a table that does not fit in cache. The first of those is the more interesting one, because it is the same code path the group by queries spend their time in. DuckDB does these five queries in 15 to 95 ms, so this is progress rather than arrival.

### Changed

`take` gathers on every core and stops probing a validity bitmap the column has none of.

A gather's output row depends on its own index and on nothing else in the output, so it splits by output row. The one thing in there that is not per row is the validity bitmap, which is built a word at a time in a register and stored once every sixty four rows, so the slice boundaries are rounded up to a multiple of sixty four and no two workers touch the same word.

The other half is the probe. The loop read the source's validity bit for every row, which is a second random read into a different array from the values, and doubles the number of cache misses a gather takes. A column with no nulls does not need it. The negative index check has to stay, because that is how a left join reports a row the right side did not have, but the two halves of that condition are separate questions and only one of them was avoidable.

Ten million rows on an i9-13900K, eight paired rounds of five runs each. These are joins because that is where the big gathers are, one per output column.

| query | before | after | ratio |
| --- | --- | --- | --- |
| j1 | 235.7 ms | 181.4 ms | 1.34 |
| j2 | 282.2 ms | 223.7 ms | 1.28 |
| j3 | 266.6 ms | 226.8 ms | 1.21 |
| j5 | 961.4 ms | 806.1 ms | 1.18 |
| j4 | 958.7 ms | 822.2 ms | 1.18 |
| q1 | 22.6 ms | 21.7 ms | 1.00 |

`filter_rows` is deliberately not split the same way. Where a filtered row lands depends on how many rows before it survived, which is a prefix sum the gather does not need.

The join's bucket build stops allocating a second copy of the group table.

Bucketing the right side is a count, a prefix sum and a scatter, and the scatter needs a cursor per group saying where the next row of that group goes. It was a separate array, filled from the offsets. On a join between two frames that are mostly one to one the group table is as long as the frames, so that is an allocation and a copy the size of the input, plus the zero fill of the bucket array on top, which the scatter overwrites in full anyway.

The offsets can be their own cursor. Group `g` is written from `starts[g]` up to `starts[g + 1]`, so when the scatter finishes each entry holds what its successor held, and one backwards pass over the table puts them back. A group nothing was scattered into needs no special case, because its offset was already equal to its successor's.

Ten million rows on an i9-13900K, eight paired rounds of five runs each, on a quiet machine this time.

| query | groups | before | after | ratio |
| --- | --- | --- | --- | --- |
| j5 | 10M | 1094.7 ms | 909.9 ms | 1.17 |
| j4 | 10M | 1035.6 ms | 918.8 ms | 1.14 |
| j2 | 100k | 249.3 ms | 247.5 ms | 1.04 |
| j1 | 10k | 206.1 ms | 206.2 ms | 1.01 |
| j3 | 100k | 260.1 ms | 261.5 ms | 0.98 |
| q1 | control | 20.1 ms | 20.2 ms | 0.98 |

j1 through j3 join against a small right frame, so their group tables are ten and a hundred thousand entries and there was nothing there to save. The gain is the whole point of the change: it scales with the number of distinct keys, not with the number of rows.

The join emits its row pairs on every core, and stops looking for null keys in frames that have none.

`join_indices` was two serial walks over the left side, one counting the output rows and one writing them, and on db-benchmark's j queries the writing walk alone was half of the pairing. It is a walk with no carried state: what a left row emits depends on that row and on the right side buckets, which are finished before the walk starts. So both walks now split by left row across cores, with a prefix sum of the per slice counts in between telling each worker where in the output its slice begins. That is what replaces the append, and it is why the counting walk had to split the same way.

An outer join is left on one thread on purpose. It has to remember which right rows it paired so it can emit the leftovers afterwards, and that memory is a validity bitmap whose set is a read modify write of a word that eight rows share, so two workers marking at once would drop marks and invent unmatched rows.

The other half is a pass that did not need to run at all. Before touching any of the above, the pairing built a `List[Bool]` over both frames saying whether each row's key tuple contained a null, one branch per key per row, and on a frame with no nulls anywhere the answer was False every time. Asking each key column for its null count first is a popcount per validity word, a sixty fourth of the pass, and db-benchmark's keys have no nulls. That was 14 ms of the 97 ms pairing on j1 and 27 ms on j4.

Ten million rows on an i9-13900K, alternating builds, eight rounds of five runs each, one process per measurement, reported as the median of the per round paired ratios with the full range of those ratios in the last two columns. The machine was under an unrelated load again, which is what the pairing is for.

| query | shape | before | after | ratio | low | high |
| --- | --- | --- | --- | --- | --- | --- |
| j4 | 10M to 10M, one match a row | 2280.7 ms | 1335.8 ms | 1.72 | 1.26 | 1.90 |
| j5 | 10M to 10M, then aggregated | 2232.3 ms | 1342.6 ms | 1.67 | 1.37 | 2.07 |
| j1 | 10M to 10k, one match a row | 386.5 ms | 255.3 ms | 1.49 | 1.13 | 1.71 |
| j3 | 10M to 100k, left join | 417.2 ms | 321.8 ms | 1.32 | 0.97 | 1.66 |
| j2 | 10M to 100k, inner join | 413.3 ms | 318.0 ms | 1.30 | 1.01 | 1.46 |
| q1 | group by, no join | 26.7 ms | 25.8 ms | 1.03 | 0.79 | 1.37 |

q1 is the control and touches none of this. The before column is higher across the board than the numbers quoted in the 0.6.20 notes because the machine was busier during this run, which is exactly the reason the ratios are paired rather than divided at the end.

What is left in the j4 pairing is the scatter that fills the buckets, which is 354 ms of random writes into a 10M entry table, and the factorize of the 20M row concatenated key column, which is 390 ms. Neither is touched here.

## [0.6.20] - 2026-08-31

Built against Mojo 1.0.0 (ed45d567).

The numeric factorize routes hand back a representative row per group, which is the last thing that was keeping the renumbering pass on the critical path.

The change below took the pass off a string key with no nulls, because the string merge already produced that list on its way to comparing candidate keys. The numeric routes had the same row for the same reason and were throwing it away. All three of them recognize a new group by the row that introduced it: the direct one appends to `keys` on first sight, the hashed serial one already gets the list back from `HashTable.build` and reads the key values out of it, and the parallel merge picks its representative when a worker's local group turns out to be new. So `Factorized` grows a `firsts` and each route fills it where it already had the row in hand, which is one append per group rather than a pass.

That covers the first key. The other place the pass was running unconditionally is the combine step, once per key after the first. What it renumbers there is the packed column, which is written a row at a time from two code arrays and therefore has no nulls at all, so its factorize's ordinals are already in first-appearance order and its representative rows are already the ones the pass would have collected. That is five passes over every row on a six key group by.

What is left calling `_densify` is a first key with nulls, of any dtype. All four routes put the null group at ordinal zero wherever its first null actually is, and `sort=False` at the frame layer means first appearance, so something has to move it.

Ten million rows on an i9-13900K, alternating builds, twenty rounds of five runs each, one process per measurement. The machine was running an unrelated benchmark throughout, which is why this is reported as the median of the per round paired ratios rather than as a ratio of medians: the pairing is what makes the contention cancel. The full range of those per round ratios is in the last two columns.

| query | keys | before | after | ratio | low | high |
| --- | --- | --- | --- | --- | --- | --- |
| q4 | id4, integer | 51.6 ms | 41.9 ms | 1.22 | 1.14 | 1.38 |
| q5 | id6, integer | 94.7 ms | 80.1 ms | 1.17 | 0.97 | 1.31 |
| q6 | id4 and id5, integer | 898.3 ms | 813.4 ms | 1.11 | 1.08 | 1.15 |
| q10 | id1 through id6, mixed | 1904.0 ms | 1736.2 ms | 1.11 | 1.02 | 1.19 |
| q3 | id3, string | 130.1 ms | 133.5 ms | 0.98 | 0.91 | 1.07 |
| j1 | id1, join | 329.2 ms | 308.0 ms | 1.05 | 0.95 | 1.17 |

q4 and q5 are single integer keys and they get the first key pass removed. q6 gets that plus one combine, and q10 gets five combines on top of a string first key that was already free. q3 is the control: it went through the change below and there was nothing left here for it to gain, and it did not. j1 is the other control and its spread covers one, so the five percent is not a claim.

A group by no longer renumbers ordinals it has no reason to renumber.

`group_ordinals` ran `_densify` over every row of every key. That pass was written because `factorize` did not promise that every ordinal it can produce belongs to some row, and an ordinal nothing carries becomes an aggregation row nobody asked for. It does promise that now, on all three routes, and a fuzz over three thousand random int64 columns covering both numeric routes found no sparse ordinal on either, so the density is not what the pass is still buying. Two other things are. It puts the null group where its first null appears instead of at ordinal zero, which is what `sort=False` means at the frame layer, and it fills the representative row table the frame layer gathers key values with.

Neither is needed as often as the pass was being run. A string key with no nulls already has both: the merge hands out ordinals in first-appearance order, and the row list it built to compare candidate keys with is exactly the table `_densify` would have produced. A key that is not the first one needs neither either, whatever its dtype and whatever its nulls, because only its group count is read at that point, and the packed column is factorized again afterwards, which is what fixes the ordinals for the result.

So `_factorize_any` now reports what its route knows in a `KeyCodes`, and `group_ordinals` decides from that rather than paying unconditionally. What is left paying is a numeric first key and a string first key with nulls. The numeric routes could record representative rows during the build and skip the pass too, which is the obvious next step and is a separate change.

Ten million rows on an i9-13900K, alternating builds, five runs each, three rounds, medians of the per round medians, one process per measurement. The queries are db-benchmark's at the 0.5GB scale.

| query | key | before | after | ratio |
| --- | --- | --- | --- | --- |
| q1 | id1, string, 100 groups | 28.7 ms | 19.9 ms | 0.69 |
| q2 | id1 and id2, string | 115.2 ms | 90.0 ms | 0.78 |
| q3 | id3, string, 100k groups | 139.8 ms | 116.3 ms | 0.83 |
| q7 | id3, string, 100k groups | 130.7 ms | 110.4 ms | 0.84 |
| q10 | id1 through id6, mixed | 1968.1 ms | 1878.9 ms | 0.95 |
| q4 | id4, integer | 52.0 ms | 50.0 ms | 0.96 |
| q5 | id6, integer | 91.9 ms | 89.3 ms | 0.97 |
| q6 | id4 and id5, integer | 853.9 ms | 835.5 ms | 0.98 |
| j1 | id1, join | 326.3 ms | 319.7 ms | 0.98 |
| j4 | id1 through id3, join | 1775.0 ms | 1741.9 ms | 0.98 |

The first five are the string keyed queries, which is the set the change targets. The last five are controls and none of them moved beyond the run to run spread, which is what a change that only removes work should look like.

### Changed

- `_factorize_any` returns a `KeyCodes` carrying the ordinals, the group count and the representative rows, instead of the ordinals alone, and `group_ordinals` skips `_densify` for any key whose `KeyCodes` describes every group.
- `group_ordinals` skips it for the packed column too, which is one pass per key after the first.
- `Factorized` carries a `firsts`, so its constructor and `_finish` take a fourth argument. Both are internal to `firepanda/hash`, and `factorize` itself is unchanged at the call site.
- `_densify`'s docstring records that the sparse ordinal case it was written for no longer happens, and what it is still for.

### Added

- `FactorizedStrings.into_parts`, `Factorized.into_parts` and `KeyCodes.into_parts`, which give up the ordinals and the representative rows together, because a struct cannot have two of its fields moved out one at a time and unpacking a returned tuple copies.
- Tests that a numeric key without nulls keeps the factorize's ordinals on both the hashed and the direct route, that the packed column keeps them too, and that a null in the first of two keys still lands where its first null is.
- `test_no_shape_of_column_factorizes_to_a_sparse_ordinal`, a swept property test over a hundred and twenty random int64 columns with varying length, value span and null count, which holds the promise the skipping now depends on across both numeric routes.
- Tests that a numeric key is still renumbered, that a null key still lands where its first null is, that a text key is grouped without renumbering its ordinals, and that a text key with nulls is renumbered after all.

## [0.6.19] - 2026-08-31

Built against Mojo 1.0.0 (ed45d567).

A group by now chooses how many cores to use, rather than choosing between one and all of them.

Splitting a factorize across workers buys a shorter build and pays for it with a merge no thread can help with, and the merge grows with every worker added, because each one rediscovers whatever groups fall in its own slice. So the two curves cross, and the best worker count is at the crossing. Until now the code could only ask for all of them or none, and it decided with a guard that refused anything projecting more than half a slice of groups. That guard was reading the wrong number. `project_groups` exists to size a hash table, where guessing high costs memory and guessing low costs a rehash, so it extrapolates the discovery rate flat and overshoots on purpose. On ten million rows with a hundred thousand groups it answers six million, and a column that a split wins two and a third times on was going to one thread.

There are two pieces. `_estimate_groups` fits the coupon collector curve those two sample counts actually lie on instead of the tangent, which recovers ninety nine thousand for that column and nine hundred and thirty six thousand for a genuinely high cardinality one, so the refusals that should happen still happen. `_parallel_workers` then costs the route at every worker count the machine can offer, in rows touched, and takes the cheapest if it beats the serial cost by a quarter. The two weights in that cost, what a remap row costs against a build row and what a merged group costs against one, were measured rather than guessed: the route was timed at every worker count from two to thirty two across four cardinalities, and `MERGE_COST` is the weight that puts the model's answer on the measured minimum.

This supersedes the cardinality guard described below, which shipped in the same release cycle and never reached a version of its own.

Ten million rows on an i9-13900K, alternating builds, five reps each, four rounds, medians, one process per measurement. Fifteen untouched benchmarks ran as controls and fourteen of them sat within seven percent.

| benchmark | groups | before | after | ratio |
| --- | --- | --- | --- | --- |
| hash/factorize_100k | 100000 | 88.63 ms | 38.06 ms | 0.43 |
| text/group_medium | 100000 | 49.90 ms | 34.02 ms | 0.68 |
| hash/factorize_10k | 10000 | 11.56 ms | 9.80 ms | 0.85 |
| hash/factorize_nulls | 10000 | 10.64 ms | 9.40 ms | 0.88 |
| hash/factorize_100 | 100 | 6.10 ms | 6.23 ms | 1.02 |
| hash/factorize_all_distinct | 10000000 | 218.20 ms | 214.95 ms | 0.99 |

The first two are the columns this is for and neither was being split at all before. The next two were already parallel and gain from being given twenty five workers instead of thirty two, which is the model declining the last seven because their share of the merge costs more than their share of the build saves. The last two are the ends of the range, where the answer was already right and the point is that it did not change.

The string `factorize` runs on every core too.

Text was the one key type left on a single thread after 0.6.18, and it is the type that matters most for the group by queries anyone benchmarks. It now takes the same route the numeric one does: one contiguous slice per worker, a private table per slice, and a sequential merge that renumbers the local ordinals into global ones in workers-then-ordinals order, which is what preserves first-appearance ordering.

The merge is where the two differ. The numeric one probes on the hash alone, because for it the hash is the key. A string does not fit in a hash, so a match there is a candidate and the rows behind the two keys have to be compared, which is the new `HashTable.insert_string`. That comparison needs a representative row per group and produces one, so the list the merge builds is not scratch that gets discarded, it is the result `factorize_strings` returns. The workers feed it: each records the absolute row of every key that was new to its own table, and the merge visits those in an order that makes the surviving representative the earliest row in the column with that key, which is the row one thread would have picked.

The same cardinality guard applies. A column of distinct strings gets nothing from being split, because the merge would rebuild all of it after the workers already had, so a sample is built first and anything projecting more than half a slice of groups goes to the serial route.

Ten million rows on an i9-13900K, alternating builds in one session, five reps each, four rounds, medians. Fourteen untouched `text/*` and `hash/*` benchmarks ran alongside as controls and all but two sat within four percent.

| benchmark | before | after | ratio |
| --- | --- | --- | --- |
| text/group_repeated | 27.16 ms | 4.19 ms | 0.15 |
| text/group_distinct | 50.59 ms | 51.90 ms | 1.03 |
| text/group_prefixed | 55.74 ms | 55.55 ms | 1.00 |

Only the first of those three is meant to move. The other two are columns of distinct keys, which is exactly what the guard is there to keep on one thread, and their staying flat is the guard working rather than the change failing.

### Changed

- Both hashed routes ask `_parallel_workers` how many workers to use instead of testing a projected group count against half a slice, and `_projected_groups` and `_projected_groups_strings` answer with `_estimate_groups` instead of `project_groups`.
- `factorize_strings` is `raises`, and picks between `_factorize_strings_serial` and `_factorize_strings_parallel` the way the numeric one picks between its two.

### Added

- `_parallel_workers`, which costs the parallel route at every available worker count and returns the cheapest, or one to stay serial, with `REMAP_SHARE`, `MERGE_COST` and `SPLIT_MARGIN` as its weights.
- `_estimate_groups`, which reads a cardinality off two sample counts by fitting the curve they lie on rather than the tangent, replacing `MERGE_HEADROOM`.
- `_factorize_strings_parallel`, the multi threaded string route, `_factorize_strings_serial`, the single threaded one, and `_projected_groups_strings`, the sample build that chooses between them.
- `HashTable.insert_string`, which is `insert` with the key comparison `build_strings` does, for the merge at the end of a parallel string build.
- `hash/factorize_100k` and `text/group_medium` benchmarks, a numeric and a text column of a hundred thousand randomly drawn keys, which is the cardinality band the suite had nothing in.

## [0.6.18] - 2026-08-29

Built against Mojo 1.0.0 (ed45d567).

`factorize` decides which route to take in one pass that can give up early, instead of two full passes that always finish.

The choice is between a direct table indexed by the value and a hash table, and it is made from the column's minimum and maximum. Those came from a `min_of` and a `max_of`, which is two complete reductions over the column before any grouping starts. Both are thrown away on every column that ends up hashed, which is every float column, every wide-ranging integer column, and every join key that is an identifier rather than a category. On ten million rows a standalone probe puts them at 8.5 ms of the 61 ms `factorize` takes at a hundred groups.

Nothing about the decision needs the exact bounds. The bounds only ever widen as the scan goes, so once the span has passed what a direct table is allowed to be, or once a value has landed outside the window the subtraction is safe in, no later row can bring it back. So the two reductions are fused into one and the test that was waiting for them is moved inside the loop, checked once per validity word. A column of scattered keys now answers in the first few thousand rows rather than in ten million.

The scan is the same shape the aggregation kernels use, which is what makes it cheap on the columns that do take the direct route. A validity word that is all ones and covers a whole block is read as SIMD lanes and reduced with `min` and `max`, and only a partial or gappy word falls back to a row at a time. The all-null column is the one case that has no bounds at all, and it gets a single slot it never reads rather than a hash table.

Measured against a build of the previous release, the two run alternately in one session, five reps each, six rounds, ten million rows on an i9-13900K. Medians. The machine was running other people's benchmark suites throughout, so the `hash/dict_*` rows are carried as a control: they call none of this code and should not move.

| benchmark | before | after | ratio |
| --- | --- | --- | --- |
| factorize_nulls | 65.1 ms | 44.3 ms | 0.68 |
| factorize_direct | 17.8 ms | 14.1 ms | 0.79 |
| factorize_100 | 53.8 ms | 43.1 ms | 0.80 |
| factorize_10k | 61.2 ms | 50.0 ms | 0.82 |
| factorize_all_distinct | 305.9 ms | 299.6 ms | 0.98 |
| dict_100 (control) | 36.2 ms | 35.9 ms | 0.99 |
| dict_10k (control) | 41.9 ms | 41.4 ms | 0.99 |
| dict_all_distinct (control) | 724.9 ms | 779.0 ms | 1.07 |

Every row that runs this code improves and the two quiet controls do not move. The third control is a heavy allocator benchmark whose variance is wide enough on a contended machine that its ratio says nothing either way, which is worth stating rather than dropping.

The size of each gain is the fraction of the work the bounds were. `all_distinct` barely moves because three hundred milliseconds of hashing swamps eight of scanning. `nulls` moves most because its bounds pass was the slowest of the five and the new one skips whole validity words of nulls without reading a value.

What this does not do is win the milestone's group by criterion. Against the language's own `Dict` at ten million rows on this machine, `factorize` is now 43.1 ms against 35.9 at a hundred groups and 50.0 against 41.4 at ten thousand, so `Dict` is still ahead at low and medium cardinality, and 299.6 against 779.0 at all distinct, where the table is ahead by two and a half times. Removing the wasted passes closed part of the gap and did not close it. What is left is in the probe and the build, not in the decision.

The hashed `factorize` runs on every core.

A group by spends most of its time turning a key column into ordinals, and until now that happened on one thread while every engine we are measured against used all of them. The column is now cut into one contiguous slice per worker, each worker builds a private table over its own slice, and a sequential merge afterwards renumbers what they all found into one set of ordinals.

Cutting a column into private tables is easy and getting first-appearance order out the other side is not, because that order is what pandas returns and what every test here compares against. It falls out of two facts. The slices are contiguous and taken in order, so a group's first row is in the earliest slice that contains the group at all. And within a slice the local ordinals are already in first-appearance order, because that is what a build produces. So the merge walking workers in order, and within a worker walking local ordinals in order, visits every group's first row before any later one, and assigning global ordinals in that visiting order gives exactly the sequence one thread would have produced.

The merge costs one probe per group per worker and it is the one part that does not parallelize, so the route is only worth taking when a column's groups are far fewer than a slice is long. A column where every row is its own key would spend the merge rebuilding the whole column serially, having already built it once in parallel, and that measured as a clear loss before this was guarded. So a sample of the column is built first, at most sixty five thousand rows, and the discovery rate over it is extrapolated by the same `project_groups` the table already sizes itself with. A projection of more than half a slice sends the column to the serial route.

Nothing is hashed twice. The workers' tables already hold the hashes, which is what this table calls a key, and `HashTable.keys_by_ordinal` scans the slots to write them out in ordinal order so the merge can probe with them directly. That scan is twice the group count and it replaces hashing the group count over again.

Measured against a build of the previous commit, the two run alternately in one session, five reps each, four rounds, ten million rows on an i9-13900K with thirty two logical cores. Medians. The machine was busy, so three untouched benchmarks are carried as controls.

| benchmark | before | after | ratio |
| --- | --- | --- | --- |
| factorize_100 | 45.5 ms | 8.2 ms | 0.18 |
| factorize_nulls | 46.1 ms | 14.1 ms | 0.31 |
| factorize_10k | 41.5 ms | 14.1 ms | 0.34 |
| factorize_all_distinct | 298.7 ms | 291.9 ms | 0.98 |
| dict_10k (control) | 45.4 ms | 42.0 ms | 0.93 |
| dict_100 (control) | 36.2 ms | 36.6 ms | 1.01 |
| table_probe (control) | 176.3 ms | 219.3 ms | 1.24 |

The three shapes a group by actually meets get between three and five and a half times faster, and `all_distinct` is left alone by the guard as intended. The controls are the honest part of this table: two of them sit within seven percent of where they started and the third moved twenty four percent against us on code neither build touched, which is the size of the noise on this machine on this day and is the reason nothing in the five to ten percent range is claimed here as a result either way.

This clears the milestone's group by criterion. Against the language's own `Dict` at ten million rows, `factorize` is now 8.2 ms against 36.2 at a hundred groups, 14.1 against 45.4 at ten thousand, and 291.9 against 739.5 at all distinct, so the table is ahead by four and a half, three, and two and a half times respectively. Before this it was behind at low and medium cardinality.

Strings are still on one thread. `build_strings` takes the new argument the parallel route needs and nothing calls it that way yet, which is the next piece of work, because four of the ten db-benchmark group by queries key on text.

### Changed

- `factorize` gets its route from `_direct_plan`, which fuses the minimum and the maximum into one scan and returns as soon as the answer is settled. `_direct_span` and the `min_of` and `max_of` pair it called are gone.
- `HashTable.build` and `HashTable.build_strings` take a `rank` alongside `base`. `base` stays the absolute row a chunk starts at, which is what the validity bitmap and the output are indexed by, and `rank` is how many rows this table has already seen, which is what the sizing schedule reads. They are the same number for a column built front to back on one thread, which is every existing caller.
- `factorize` and `_factorize_hashed` are `raises`, because the parallel route calls `parallel_for`.

### Added

- `DirectPlan`, the result of that decision, carrying the number of slots a direct table would need and the value that indexes slot zero.
- `_factorize_hashed_parallel`, the multi threaded hashed route, and `_factorize_hashed_serial`, the single threaded one it is chosen against.
- `_projected_groups`, which builds a sample of a column and extrapolates its group count, so the choice between those two can be made before either has run.
- `HashTable.keys_by_ordinal`, which writes every key a table holds into a buffer indexed by its ordinal.

## [0.6.17] - 2026-08-29

Built against Mojo 1.0.0 (ed45d567).

Two changes to the CSV reader, one to each half of it. The fill stopped zeroing memory it was about to overwrite, and the scan stopped growing its index by doubling. Together they take a ten million row narrow file from 98.7 ms to 59.3 and a nine column numeric file from 197.9 to 75.6, on an i9-13900K reading off a warm mapping.

The fixed width columns are no longer zeroed before they are filled. The 0.6.16 entry ended with the fill being the larger half of a read for the first time and the numeric columns being where to look, so this is the measurement of that. A single integer column of ten million rows read in 27 ms, and a file of one digit integers read in the same time as a file of seven digit ones, which says the digits are not what is being paid for. What is being paid for is the allocation: `Array[int64](10_000_000)` takes 5.2 ms on its own, all of it a memset, and it runs on one thread before the parallel fill it is for can be handed out.

Nothing needs it. The sweep visits every row of every fixed width column, so the only slot it was leaving to the memset was one holding a missing or unparseable field, and writing a zero there is one store on a path that was already clearing a validity bit. So the columns are allocated with the `Buffer(overwritten=)` the string fill has used since 0.6.15, and the sweep writes the zero itself.

The rest of the entry is the sweep's inner loop, which the change made worth rewriting. It asked on every value which of three kinds of value it was about to parse, though the answer is settled per column before the rows are walked, and it reached the destination through a list index per value. Both are now hoisted into a `fill_tile` parameterized on the dtype, so a tile of rows is a straight loop with its destination in a register. A failed parse also stores now rather than branching around the store, since the parse leaves a zero to store either way.

Measured against a build of the previous release, the two run alternately in one session, five reps each, three rounds, on an i9-13900K, warm cache, reading off a mapping. Medians.

| file | fixed width columns | fill before | fill after | ratio | read before | read after | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nulls | 9 | 139.2 ms | 44.8 ms | 0.32 | 197.9 ms | 105.5 ms | 0.53 |
| narrow | 3 | 62.0 ms | 39.9 ms | 0.64 | 98.7 ms | 76.9 ms | 0.78 |
| wide | 40 | 102.3 ms | 75.4 ms | 0.74 | 145.6 ms | 116.5 ms | 0.80 |
| quoted | 1 | 77.9 ms | 73.5 ms | 0.94 | 111.4 ms | 106.9 ms | 0.96 |

The order of that table is the number of fixed width columns per row of the file, which is the amount of memset removed, and it is also the order of the gains. Nulls is nine columns of ten million values, so it was zeroing seven hundred and twenty megabytes before reading anything. Quoted is one, and it barely moves.

Peak resident memory is unchanged, within a couple of percent either way across repeated runs. It would be: the pages are touched by the fill whether or not they were touched by a memset first. What the change removes is a pass, not a page.

The four ingestion files give byte identical schemas and null counts before and after.

The field index is sized once instead of grown by doubling. With the fill halved the scan is the larger half again, and a scan does two things: it finds the boundaries and it records them. Finding them is a SIMD compare over the block, which is memory bound and near the limit. Recording them was an append to a list that started at nothing, so a ten million row file's index reached three hundred and twenty megabytes through twenty five reallocations that copied about as many bytes as the index ends up holding, on the memory system the compare is already saturating.

So the scan now guesses the size first. A quarter megabyte of the block is counted with the same SIMD compare, the count is extrapolated over the block, an eighth is added on top, and the result is clamped to the one field per two bytes that is the arithmetic ceiling. The guess does not have to be right, only close and on the high side: short by a little costs one reallocation of a nearly finished list, and long costs address space that is never written and so never becomes a page. A delimiter inside a quoted field is counted as a boundary it is not, which pushes the guess up, which is the safe direction.

Measured against a build of the previous release, the two run alternately in one session, five reps each, three rounds, on an i9-13900K, warm cache, reading off a mapping. Medians.

| file | fields | scan before | scan after | ratio | read before | read after | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nulls | 90.0M | 58.3 ms | 30.5 ms | 0.52 | 102.8 ms | 75.6 ms | 0.74 |
| narrow | 40.0M | 35.3 ms | 21.5 ms | 0.61 | 75.3 ms | 59.3 ms | 0.79 |
| wide | 50.0M | 39.0 ms | 25.4 ms | 0.65 | 115.0 ms | 100.1 ms | 0.87 |
| quoted | 30.0M | 32.1 ms | 25.9 ms | 0.81 | 102.1 ms | 97.4 ms | 0.95 |

The fill is flat on all four to within noise, which is the control: nothing outside the scan was touched.

How far off the guess is, whole file, estimated against actual: wide 56.3M against 50.0M, nulls 108.8M against 90.0M, narrow 53.7M against 40.0M, quoted 70.7M against 30.0M. Narrow is high because its early rows have the short ids and short labels and the rest of the file does not, and quoted is high because most of its commas are inside quotes. Both are the harmless direction, and the memory numbers say so.

Peak resident memory for one read in one process, the two builds run alternately, three repeats each, medians in megabytes: narrow 1167 to 1090, quoted 1611 to 1577, nulls 1788 to 1655, wide 1305 to 1277. Every file is lower, because the doubling had to hold the old buffer and the new one at once at every step and this does not. A reserved page that is never written costs nothing resident, which a standalone probe confirms: reserving eight hundred megabytes and touching none of it leaves the process at fourteen.

One caveat on that measurement, because an earlier version of this entry had it wrong. A process that reads the same file several times in a row has a peak resident set well above what one read costs, and it climbs with each read on both builds, because the allocator does not return every freed block to the system and the reads do not ask for the same sizes in the same order. Read once per process and the numbers above are what a read costs. Read four times and the high water mark is several hundred megabytes above that on both builds, which is a property of the arena rather than of the reader.

### Changed

- The sweep allocates its fixed width columns unzeroed and writes every slot itself, including a zero where the field is missing and where it did not parse. The declared type path in `fill_column` still allocates zeroed, because a block there stops at the first value that does not fit and so does not write every slot.
- The sweep's per value three way test on which group a column is in is hoisted out of the row loop into `fill_tile`, which takes the dtype as a parameter and the destination as a pointer.
- `scan_csv` reserves its field index from a sample of the block instead of growing it from nothing. The index still grows if the guess was short, so a file that defeats the sample is slower than it would have been and not wrong.

### Added

- `Array.__init__(overwritten=)` and `ColumnData.__init__(overwritten_bytes=)`, the column level form of the `Buffer` constructor of the same name, for a caller that will write every element.
- `Scan.__init__(capacity=)`, which sizes the field index up front, and `estimate_fields`, which says what to size it to.

## [0.6.16] - 2026-08-29

Built against Mojo 1.0.0 (ed45d567).

The scan's row offsets are gone. The 0.6.15 notes said the next place to look was the fill, on the grounds that the scan is not search bound. That was right about the scan not being search bound and wrong about where its time went. Lined up against each other the four ingestion files say it plainly: narrow is ten million rows and forty million fields and scans in 60 ms, wide is one million rows and fifty million fields and scans in 42 ms. More fields, more bytes, less time. The scan's cost tracked the row count.

What costs a row is the offset recorded for it. The index is a flat list of packed fields plus a list saying where each row begins in it, which is the Arrow offsets shape and the obvious one. But almost every CSV file is rectangular, and for a rectangular file the offset of row r is r times the width and the list holding it is eight bytes a row of pure redundancy. On a ten million row file that is eighty megabytes written during the scan and read back during every fill.

So the width is kept as a number and the offsets are built only when a row turns up that disagrees, which is a file this reader refuses anyway. `at` multiplies instead of loading, `width` returns a field, and `is_ragged` went from a walk over every row to reading whether the offsets exist, which matters because a read called it once per block and once more for inference.

Measured against a build of the previous release, the two run alternately in one session, five reps each, three rounds, ten million rows on an i9-13900K, warm cache, reading off a mapping. Medians.

| file | rows | scan before | scan after | ratio | read before | read after | ratio | peak RSS after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| narrow | 10M | 59.9 ms | 35.3 ms | 0.59 | 125 ms | 96 ms | 0.77 | 0.93 |
| quoted | 10M | 55.5 ms | 33.6 ms | 0.61 | 141 ms | 117 ms | 0.83 | 0.95 |
| nulls | 10M | 82.2 ms | 58.0 ms | 0.71 | 230 ms | 208 ms | 0.91 | 0.93 |
| wide | 1M | 41.9 ms | 39.1 ms | 0.93 | 147 ms | 150 ms | 1.00 | 0.98 |

Wide is flat, which is the expected answer and the confirmation: one million rows had one million offsets to skip writing, and the file has fifty times as many fields as it has rows.

The four ingestion files give byte identical schemas and null counts before and after. A patch bump, since nothing changes shape.

### Changed

- `Scan` no longer keeps a row offset per row while the file is rectangular. It keeps the width and the row count, and `at(row, column)` indexes at `row * width + column`. A row that disagrees with the ones before it fills the offsets in for every row up to that point, since they all had the same width, and appends from then on, so a ragged file behaves exactly as it did.
- `Scan.is_ragged` is now a constant time test rather than a walk over every row.
- `Scan.end_row` replaces appending to the offsets directly, and both the vectorized scanner and the scalar reference one call it.

## [0.6.15] - 2026-08-29

Built against Mojo 1.0.0 (ed45d567).

The string concat is gone. The 0.6.14 entry ended by saying that what remained of it was close to the cost of first touching the output pages, so the next change should remove it rather than speed it up, by sizing the string column up front from the field index and letting each block write into its own slice. That is this change.

Sizing it up front means knowing the payload before reading the file, and the field index already holds it. An element costs payload bytes only when it is longer than twelve, the index records every field's start and end, and the one length the index gets wrong is an escaped field's, whose doubled quotes collapse. So the block payload sizes are added up, prefix summed, and the column is allocated once at its full height; each block then writes its own slice of views and its own slice of payload at absolute offsets, so nothing is stacked and no offset is rebased afterwards.

Measuring is itself a second walk over the index, though, and a column whose elements all fit inside their views has no payload for that walk to find. That is the ordinary case, and it is what narrow and wide are: `"row9999999"` and `"s615"` both inline. So the fill is tried first on the guess that nothing reaches the payload, and a block that meets an element too long to inline stops where it stands and reports it; only then is the column measured and filled again. The guess is the whole read when it holds, and the block that disproves it usually does so within a few rows, because a column with long elements rarely hides them at the end. It is the bargain the type ladder already makes.

Measured against a build of the previous release, the two run alternately in one session, five reps each, three rounds, ten million rows on an i9-13900K, warm cache, reading off a mapping.

| file | columns | before | after | ratio | peak RSS before | peak RSS after | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| narrow | 4 | 127 ms | 128 ms | 1.01 | 1.65 GB | 1.50 GB | 0.90 |
| quoted | 3 | 215 ms | 151 ms | 0.70 | 2.70 GB | 1.75 GB | 0.65 |
| nulls | 9 | 240 ms | 207 ms | 0.86 | 2.17 GB | 2.19 GB | 1.01 |
| wide | 50 | 165 ms | 151 ms | 0.92 | 1.46 GB | 1.38 GB | 0.95 |

The nulls row should be read as flat, not as a gain. That file has no text column at all, so it does not reach any of this, and its 0.86 is the machine drifting between the two builds. The honest results are quoted, which is what the change was aimed at, and wide. Narrow is a wash on time and a tenth better on memory: its one text column is entirely inline, so it takes the guessed path, writes its views once instead of writing them into a per block piece and copying them into the column, and never allocates a payload.

The four ingestion files give byte identical schemas and null counts before and after. A patch bump, since nothing changes shape.

### Changed

- A string column is now filled in place rather than built per block and concatenated. The payload size is derived from the field index, the block sizes are prefix summed, and each block writes its views and its payload bytes into its own slice of one column. The concat of string columns is still there and still parallel, it is simply no longer on the path a read takes.
- The fill of a text column speculates that every element fits inside its view, which needs no payload and so no measuring pass. A block that meets a longer element stops and the column is measured and filled again. On a column that is entirely inline this removes the index walk that 0.6.14's design would have added.

### Added

- `collapse_into` in `firepanda.array.strings`, which copies a field's bytes to a destination and collapses doubled quotes as it goes, returning what it wrote. `StringBuilder.append_escaped` now delegates to it, and the reader calls it to write straight into a column's payload.
- `collapsed_length` in `firepanda.io.scan`, which reports what an escaped field measures once its doubled quotes collapse, for a reader sizing a column before it fills it.

## [0.6.14] - 2026-08-29

Built against Mojo 1.0.0 (ed45d567).

The reader's text path. The 0.6.13 entry ended by saying the quoted file was flat because its time is in unescaping rather than in the index walk, and that this was where to look next. It was half right. Unescaping was costing something, but the larger cost was not the parse at all, it was the concat that stitches the per block columns back into one column at the end of a read. On the quoted file that concat was 175 ms against 68 ms for the parse of the same column.

So this release is three changes on the join rather than on the parse: unescape into the payload instead of into a temporary, do not zero a buffer that is about to be completely overwritten, and paste the parts in parallel now that every part's destination is a prefix sum known in advance. A patch bump, since nothing changes shape and the four ingestion files give byte identical schemas and null counts before and after.

Measured against a build of the previous release, the two run alternately in one session, ten million rows on an i9-13900K, warm cache, reading off a mapping.

| file | columns | before | after | ratio |
| --- | --- | --- | --- | --- |
| narrow | 4 | 157 ms | 130 ms | 0.83 |
| quoted | 3 | 292 ms | 197 ms | 0.68 |
| nulls | 9 | 225 ms | 221 ms | 0.98 |
| wide | 50 | 180 ms | 164 ms | 0.91 |

Per column, on the quoted file and in the same session, the `note` column's concat went from 159 and 152 ms to 29 and 35 ms, and `label`'s from 14.5 and 41 ms to 5.7 and 5.6 ms. Narrow and wide move as well because they have text columns of their own. The nulls file is flat, which is the expected answer: its columns are mostly fixed width and 0.6.13 already took that path.

What remains of the string concat is close to the cost of first touching the output pages, so the next change on this path is to remove the concat rather than speed it up, by sizing the string column up front from the field index and letting each block write into its own slice.

### Changed

- A quoted field is now unescaped straight into the string column's payload instead of into a temporary `String` that is then appended. Collapsing a doubled quote only ever shortens a field, so the builder reserves the raw length, copies the runs between the doubled pairs, and builds the view from what it actually wrote. A field that shortens past twelve bytes ends up inline and the payload offset does not move.
- Concatenating string columns now writes the parts in parallel when the output is at least 65536 rows and there is more than one part. Every part's destination is known before any of it is written, since the row and payload offsets are prefix sums over the parts, so each part memcpys its views and its payload into its own slice and then rebases the offsets in the views it just wrote. Validity is still merged serially, which costs nothing next to the payload copy. Below the threshold the old sequential paste runs unchanged.
- `concat_any` no longer has its own paste loop for the string case. It builds the same part descriptors the typed path builds and takes the same parallel route, which is what a read actually reaches.

### Added

- `Buffer(overwritten=n)`, for a caller that will write every one of the n bytes. It skips the zeroing pass an ordinary `Buffer` does and still zeroes the pad up to the 64-byte capacity, so a vectorized kernel reading one register past the end still sees zeroes. The string concat output is allocated this way.
- `make_inline_at` and `make_long_at` in `firepanda.array.strview`, which build a view from a pointer and a length rather than from a span, for a caller that has already written the bytes.

## [0.6.13] - 2026-08-29

Built against Mojo 1.0.0 (ed45d567).

One change, to how a read walks the field index. A column at a time meant walking the whole index once per column, and a column's fields sit one row stride apart in it, so a wide file used one word of every cache line it fetched and then came back for the next column and fetched them all again. Filling every fixed width column of a block together, a tile of rows at a time, reads the index once and uses all of it. The wide file, fifty columns, halves. A patch bump: no API changes shape, and the four ingestion files give byte identical schemas and null counts before and after.

Measured against a build of the previous commit, the two run alternately in one session, ten million rows on an i9-13900K, warm cache, reading off a mapping.

| file | columns | before | after | ratio |
| --- | --- | --- | --- | --- |
| narrow | 4 | 167 ms | 151 ms | 0.90 |
| quoted | 3 | 291 ms | 297 ms | 1.02 |
| nulls | 9 | 317 ms | 219 ms | 0.69 |
| wide | 50 | 412 ms | 198 ms | 0.48 |

Peak RSS is unchanged, within a tenth on every file, which is the expected answer: the same buffers are allocated, in a different order.

The quoted file is flat because almost none of its time is in this loop. Two of its three columns are text and go through unescaping, which this change does not touch, and that is where the next reader change should go.

### Changed

- `read_csv` fills every fixed width column of a block in one pass over the block, a tile of rows at a time, instead of walking the whole file once per column. The tile is sized to keep its slice of the field index in the data cache, so every read after the first in a tile comes out of cache, and the columns are still done one at a time within a tile, so a column's running state stays in a register rather than in a list. The two have to be traded off against each other and the tile is where the trade is made.
- The fixed width columns of a frame are grouped by type before they are filled, so the branch that picks a parser is resolved once per column group instead of once per value.

### Added

- `TILE_BYTES`, how much of the field index one tile of a fixed width sweep works over, `sweep_fixed`, which does the sweeping, and `wanted_of`, which names a rung for an error message.

### Notes on the numbers

Three other shapes were built and measured the same way before this one was kept, because the first two were slower than what they replaced.

| shape | narrow | quoted | nulls | wide |
| --- | --- | --- | --- | --- |
| a row at a time, type chosen per value | 1.19 | 1.00 | 0.75 | 0.55 |
| a row at a time, columns grouped by type | 1.12 | 1.03 | 0.87 | 0.54 |
| one parallel region, still a column at a time | 1.07 | 1.10 | 1.01 | 0.97 |
| a tile of rows at a time, columns grouped | 0.90 | 1.02 | 0.69 | 0.48 |

The first two cost narrow more than they saved it. Going row major turns a column's accepted flag, its first bad row and its validity bitmap from things the compiler keeps in registers into list elements indexed by column, and on a four column file that is most of the loop. The third shape is the control: it keeps the column at a time walk and only removes the barrier between columns, and it moves nothing, which is what says the wide file's win is the index traversal and not the scheduling.

## [0.6.12] - 2026-08-29

Built against Mojo 1.0.0 (ed45d567).

Two changes to the reader, both about not doing work twice. The scanned index is packed into one word per field instead of four, and the pass that decided every column's type before a single value was parsed is gone, replaced by a guess from a sample that corrects itself when it is wrong. A patch bump: nothing in the API changes shape and the reader gives the same answer on every file it gave before.

Each change was measured against a build of the commit before it, the two builds run alternately in one session, because this machine drifts by tens of percent over an afternoon and two numbers taken hours apart are not a comparison. That means the two are not on one scale and cannot be subtracted from each other, so what follows is each one's ratio against its own baseline, ten million rows on an i9-13900K, warm cache, reading off a mapping.

| file | packed index | speculative types | the two together |
| --- | --- | --- | --- |
| narrow | 0.81 | 0.83 | 0.67 |
| quoted | 0.87 | 0.95 | 0.83 |
| nulls | 0.70 | 0.76 | 0.54 |
| wide | 0.94 | 0.66 | 0.62 |

So a third off narrow, a sixth off quoted, not quite half off nulls and just under two fifths off wide. The last column is the product of the two before it, which assumes they do not interact, and they should not: one shrinks the index and the other removes a pass over it.

Peak RSS fell by about a third with the packed index and is unchanged by speculation, which is the expected answer, since speculation allocates the same buffers one pass earlier.

### Added

- `read_csv_as(path, schema)` and `read_csv_as(path, schema, options)`, which read a file with types the caller already knows. `read_csv_bytes_as` has always existed, so the only way to declare a schema was to open and copy the file yourself, which gave up the mapping and read the file twice as slowly for the trouble.
- `Scan.push` and `Scan.field`, the two functions that know how a scanned field is packed, and `Scan.long`, where the length of a field too long to pack is kept.
- `LongField`, the position and length of a field of four megabytes or more.
- `SPECULATE_ROWS`, how many rows of each block a read looks at before it picks a type, and `sample_columns`, which does the looking. `rung_of` maps a declared type back to its rung on the ladder.

### Changed

- A scanned field is one 64-bit word rather than a struct of two offsets and two flags. Forty bits address the buffer, twenty two hold the length and two are the flags. The index a scan produces is the largest thing a read allocates and it is written once and read once, so its size is very nearly all of its cost: for a four column file of ten million rows it was 960 MB over a 357 MB input and it is now 320 MB. `Scan.at` still returns a `FieldSpan`, so nothing above the scanner changed.
- A buffer larger than a terabyte is refused by the scanner with a message that says so, rather than recording offsets that do not fit. A field of four megabytes or more keeps its length in `Scan.long` and is not refused, because a file with one in it is a real file.
- Types are guessed from a sample and the guess is corrected if it turns out wrong, rather than decided by a pass over every value in the file. Each block looks at its first `SPECULATE_ROWS` rows, the column is filled at the rung that sample reached, and a value that does not fit moves the column one rung up and fills it again. A rung read off a sample is never higher than the rung the whole column needs and the ladder only climbs, so the type that comes out is the type the old full pass gave, on every file, and the schemas and null counts of the four ingestion files were compared build against build to check it. Refilling costs one more pass over one column, not over the file.
- `fill_block` and `fill_column` report a value that does not fit rather than raising on it, because whether it is an error at all is now the caller's question: a declared type says it is and a guessed one says the guess was too narrow. When it is an error, the row named is the lowest failing row in the file rather than whichever block reached one first, so the same malformed file always produces the same message.
- A bounded `infer_rows` still decides types first and fills second. A bound is a promise about what gets looked at, and speculating would quietly look past it.

### Notes on the numbers

Ten million rows on an i9-13900K, five repetitions per round, four rounds, warm cache, reading off a mapping. The two changes were measured separately and each against a build of the commit before it, run alternately in one session, because the machine drifts by tens of percent over an afternoon and numbers taken hours apart are not comparable.

| file | 0.6.11 | packed index | speculative types | peak RSS before | peak RSS after |
| --- | --- | --- | --- | --- | --- |
| narrow | 460 ms | 372 ms | 192 ms | 2.10 GB | 1.72 GB |
| quoted | 655 ms | 570 ms | 333 ms | 2.80 GB | 2.57 GB |
| nulls | 940 ms | 660 ms | 381 ms | 3.10 GB | 2.32 GB |
| wide | 830 ms | 780 ms | 473 ms | 2.03 GB | 1.43 GB |

The two middle columns are from different sessions and the machine was slower for the first, so read down a column rather than across the whole row. Against its own baseline the packed index took nulls from 940 to 660 and speculation took it from 499 to 381, and the wide file, which has fifty columns and the most inference to skip, went from 720 to 473.

Peak RSS is unchanged by speculation, within two percent either way, which is the expected answer: the same buffers are allocated, one pass earlier.

## [0.6.11] - 2026-08-29

Built against Mojo 1.0.0 (ed45d567).

The reader stops copying. A file is mapped rather than read into a buffer, and a fixed width column is parsed straight into the column rather than into a per block piece that is joined afterwards. Two copies of the whole file's worth of data, both of them removed. A patch bump: nothing in the API changes shape, and the one new module is additive.

Reading the four ingestion files on an i9-13900K, ten million rows, warm cache, against 0.6.10: narrow 613 ms to 281 ms, quoted 948 ms to 402 ms, nulls 822 ms to 631 ms, wide 852 ms to 701 ms. Cold cache, which is what the ingestion suite measures, narrow 1045 ms to 625 ms and quoted 1977 ms to 822 ms.

### Added

- `firepanda/io/mapped.mojo`, which maps a file into memory instead of copying it. `MappedFile` owns the mapping and hands out a `Span` that borrows from it, so the span cannot outlive the mapping and the origin system is what enforces that rather than a comment. `map_file` is the same thing with the failure returned as a value, which is the shape a caller with a fallback wants.
- `tests/test_mapped.mojo`, the first tests in the repository that write a real file and read it back. Everything else in the reader is tested against a byte span built in memory, which is the right way to test a parser and no way at all to find out whether the file ever arrives.

### Changed

- A fixed width column is parsed straight into the finished column. The column is allocated at its full height before any block runs and each block writes its own contiguous range of it, so the per block pieces are gone and so is the copy that joined them. Validity is still per block, because two blocks meeting inside a byte would both have to read, modify and write the same word, and the bitmaps are pasted in afterwards, which is a pass over a bit per row rather than over a value per row. A block with no nulls in it is not pasted at all. String columns are unchanged and still build a piece per block, because a block's payload size is not known until the block has been read and there is nothing to write into.
- `read_csv` maps the file rather than copying it. The kernel already has the bytes and a mapping points at them where they are, so nothing is copied, and because the parser touches its blocks on every core at once the page faults are taken in parallel rather than serialised behind one `read` call. A file that cannot be mapped, and an empty one counts, is read the old way.

### Notes on the numbers

Ten million rows on an i9-13900K, five repetitions, warm page cache, reading off a mapping. The direct write is measured against the mapping alone, so the two columns below are the two changes in this release in the order they were made.

| file | mapped | mapped and written direct |
| --- | --- | --- |
| narrow | 348 ms | 281 ms |
| quoted | 431 ms | 402 ms |
| nulls | 791 ms | 631 ms |
| wide | 701 ms | 701 ms |

The narrow file is four columns of nothing but numbers over ten million rows, so all of the copy that went away was on its critical path, and it gains a fifth. The nulls file gains the most in absolute terms: nine values in ten are missing there, and the join was copying a validity bitmap along with the values on every one of eight columns. The wide file does not move, which is the expected answer rather than a disappointing one. It is fifty columns of a million rows, so a column is eight megabytes against the narrow file's eighty, and what a wide read spends its time on is fifty million fields rather than the bytes that come out of them.

Peak resident memory on the quoted file goes from 2.80 GB to 2.77 GB. That it barely moves is the honest result and it is worth saying why. The peak is not the columns, it is the scan index, which is a twenty four byte span per field plus eight bytes per row and comes to 960 MB on the narrow file for a 357 MB input. That is the next thing to go.

Ten million rows on an i9-13900K, five repetitions, warm page cache, both paths in the same process back to back so they share a cache and a heap.

| file | size | copy | map |
| --- | --- | --- | --- |
| narrow | 357 MB | 613 ms | 348 ms |
| quoted | 577 MB | 948 ms | 431 ms |
| wide | 383 MB | 852 ms | 701 ms |
| nulls | 207 MB | 822 ms | 791 ms |

The getting the bytes step itself goes from 199 ms to 0.026 ms on the narrow file and from 523 ms to 0.036 ms on the quoted one, because a map is three system calls and no work. The whole read saves more than that, which is the destination pages of the copy no longer being faulted in and zeroed.

Cold cache, with the page cache dropped before every single run, is the case the ingestion suite measures and it is better rather than worse: 1045 ms to 625 ms on the narrow file and 1977 ms to 822 ms on the quoted one. Thirty two workers faulting in parallel pull from the device harder than one sequential `read` does.

The same measurement on an eight core AMD EPYC, where the copy is a larger share of everything, warm: narrow 17311 ms to 8804 ms, quoted 12500 ms to 5687 ms, wide 21551 ms to 13096 ms, nulls 14639 ms to 12145 ms. Cold there, narrow 8753 ms to 4049 ms and quoted 14931 ms to 11230 ms.

Peak resident memory on the quoted file falls from 3.48 GB to 2.80 GB, which is the copy that no longer exists.

## [0.6.10] - 2026-08-29

Built against Mojo 1.0.0 (ed45d567).

Stacking string columns stops touching elements. A patch bump: no API changes shape, one kernel gets between four and nine times faster, and two functions are added that exist because that kernel needed them.

### Added

- `concat_strings`, the typed spelling of a concat of string columns, beside the `concat_arrays` that has always been there for the fixed width ones.
- `Bitmap.paste`, which writes a run of bits into a bitmap at a bit offset. It is the mirror of `slice` and it is harder, because the destination is shared: the bits either side of the run belong to a different part and have to survive, so every word is a read, a mask and a write, and a run that does not start on a word boundary is written as two halves. Tested against a loop over bits at every offset from 0 to 70 for five run lengths.
- `StringView.shift_offset`, which moves a long element's payload offset along by a fixed amount. That is the whole of what stacking two payloads costs a view.
- Two benchmark rows, `strings/concat_short` and `strings/concat_long`, stacking eight parts of a quarter million elements each, which is the shape `read_csv` hands the kernel.

### Changed

- A concat of string columns copies blocks rather than elements. It walked every element into a `StringBuilder`, which copied the bytes into a growing payload, appended a view to one `List` and a flag to another, and then copied both `List`s again in `finish`. Four copies per element on a path `read_csv` runs once per column. The views of a part are now one memcpy and its payload is another, and the only per element work left is adding the part's payload base to the offset field of the views long enough to have one. Short elements are not touched at all.
- The validity of a part with nulls goes across through `Bitmap.paste` rather than one bit at a time.

### Notes on the numbers

Measured on an i9-13900K, eight parts of 262,144 elements, five repetitions: 2.339 ms to 0.265 ms on a column of eight byte elements, and 4.897 ms to 1.218 ms on a column of thirty two byte ones, which is 8.8 and 4.0 times. The gap between the two is the payload, which still has to be copied and is now the whole cost of the long case.

Inside `read_csv` at ten million rows, the phase that stacks the per block pieces: 833 ms to 83 ms on a file of two text columns, 268 ms to 43 ms on a four column one, 176 ms to 55 ms on a fifty column one. A file with no text column does not move, which is expected, since its stack was already a memcpy.

The end to end read does not move by as much as that, and it is worth saying why rather than quoting the phase number and stopping. Reading the file into memory is 400 ms of a narrow read and 760 ms of a quoted one, which is now the largest single part of it, and at these sizes the wall clock of a whole read on this machine varies by twenty to thirty per cent between runs. The file read is the next piece of work.

## [0.6.9] - 2026-08-29

Built against Mojo 1.0.0 (ed45d567).

`read_csv` runs on every core. A patch bump: nothing changes shape, one thing gets faster, and the library gains its first piece of threading.

### Added

- `firepanda/exec`, which is the whole of the library's threading. One function, `parallel_for`, which runs a body once per index and returns when all of them have finished. It is built on `TaskGroup` from `std.runtime.asyncrt` rather than on `parallelize`, which is no longer in the standard library and now lives behind a GPU dependency in `max.algorithm`.
- `firepanda/io/split.mojo`, which cuts a CSV buffer into blocks that each begin on a row boundary. A newline inside a quoted field is data rather than a separator, so the quote state at an offset is what decides, and that state is the parity of the quote bytes before it. Counting bytes is parallel and a prefix sum over one number per block is free, so every block learns its starting state without reading what came before it.
- `scan_block`, which scans a byte range of a buffer while keeping the offsets absolute, and `scan_blocks`, which runs one per block and checks the result.
- A CSV round in `tests/stress/main.mojo`. It builds a file whose every value is a function of its row index, cuts it into a randomized number of blocks between one and thirty three, checks the blocks against a single pass field span by field span, then reads it the way a caller would and checks the values against the generator rather than against another run of the reader.
- `tests/test_split.mojo`, twenty two tests, and `tests/test_parallel.mojo`, eight.

### Changed

- `read_csv` runs on every core for a file large enough to be worth it, which is a quarter of a megabyte per core. The scan, the inference and the conversion all run one task per block. Nothing is shared between blocks: each scans into its own `Scan`, guesses its own types, and fills its own piece of each column.
- `Scan` records how many quote bytes it read as structure. That number is what makes the split checkable rather than merely plausible; see below.
- Columns are stacked one at a time rather than all at once. The parallelism is over blocks within a column, so every core is still busy, and the peak holds two copies of one column instead of two copies of the whole frame.

### Why a wrong split cannot pass silently

The parity argument holds under RFC 4180 and this reader is deliberately looser than RFC 4180: it accepts a bare quote in the middle of an unquoted field, because pandas does. Such a quote is data, it does not come in a pair, and it flips the parity of every offset after it.

So the split is checked. A block's structural quote count can never exceed the quote bytes in its own byte range, and the blocks partition the buffer, so if the totals agree over the whole file they agree over every block, which means no quote byte was read as data. Block zero starts at offset zero, which is a row boundary outside quotes by definition, so block zero's parse is correct, so the next block's start is a row boundary too, and the induction runs to the end.

The other half is that a boundary with the wrong parity cuts the block before it inside a quoted field, and a quoted field with no closing quote is an error the scanner already raises. When either fires the whole read is done again on one thread, which is also what produces the right file row number in the message for a file that really is malformed.

### Notes on the numbers

Measured on an i9-13900K, sixteen physical cores and thirty two logical, against one million rows of four columns, 28.2 MB, the same file for every reader and seven runs each.

firepanda reads it in 50.3 ms median, against 145.9 ms for the same code before this change. That is 2.9 times, on a machine with thirty two logical cores, and the reason it is not more is that only part of the work moved. The scan went from 45.2 ms to 7.5 ms, which is six times and close to what the hardware allows. The rest of the read, inference and conversion and stacking, is 32 ms of the remaining 39 ms and did not improve nearly as much. That is where the next piece of work is.

On the same file `pandas.read_csv` takes 237.5 ms with the C engine and 11.1 ms with the pyarrow engine, and `polars.read_csv` takes 6.2 ms. So firepanda is now four and a half times faster than the C engine and still four and a half times behind the pyarrow one. The M1 exit criterion asks for a reader that beats the pyarrow engine and this is not yet that reader.

## [0.6.8] - 2026-08-29

Built against Mojo 1.0.0 (ed45d567).

firepanda reads and writes CSV files, and a text column converts to a number and back. Those are the last two things on M1's scope list that were missing rather than merely slow.

A patch bump. Two fixes, and the rest is addition.

### Added

- A text column casts to a number and a number column casts to text. `cast_strings_to` reads bytes through the same parser the CSV reader uses, and `cast_to_strings` writes them the way the CSV writer writes them, so a value that survives a file round trip survives this one.
- `cast_any` and `Series.cast` and `DataFrame.cast` take a `LogicalType` as well as a `DType`. The dtype form cannot name text, text having no dtype of its own, so `series.cast(LogicalType.STRING)` is the way to ask for it.
- All three take a `strict` flag, defaulting to true. Strict raises on a text value that is not a number and names the row and the value. Not strict writes a null, which is what `to_numeric(errors="coerce")` does.
- `tests/test_text_cast.mojo`, twenty three tests.
- `read_csv`, which turns a file or a buffer of bytes into a `DataFrame`, inferring the schema from the first rows or taking one you hand it. Names come from the header when there is one and are `column_0` upward when there is not.
- `write_csv`, which turns a `DataFrame` back into a file or a buffer of bytes, quoting a field only when leaving it bare would move a boundary.
- `ReadOptions` and `WriteOptions`, carrying the dialect, whether there is a header, and how many rows the inference is allowed to look at. `INFER_ALL` reads the whole file before deciding.
- `benchmarks/read_file.mojo`, a standalone timed read of a file that already exists, so the comparison against `pandas.read_csv` can be run on byte identical input rather than on two files of the same description.
- Three benchmark rows, `csv/read_inferred`, `csv/read_declared` and `csv/write`.
- `tests/test_csv.mojo`, twenty eight tests.

### Changed

- `FieldSpan` records whether the field was written inside quotes. An empty field and a quoted empty field are the two ways a CSV file has of writing a missing value and the empty string, and without the flag the reader has to guess, and whichever way it guesses one of the two becomes unrepresentable.

### Fixed

- `parse_float` was an ulp or two off from a correctly rounded `strtod` outside the exponent range where a single multiply is exact, because it scaled in steps of 1e22 and rounded once per step. Five steps of that put `1.2345678901234567e100` one ulp away, so a float written at seventeen digits did not read back as itself. The fast path is unchanged and still one multiply. Everything outside it now goes to the platform's `strtod`, which is correctly rounded at every exponent, and the grammar is still checked here first, so the fallback answers the question and does not get to widen it.
- Casting a text column raised rather than converting, because the numeric path would have found its uint8 physical dtype and converted the first byte of every sixteen byte view. It converts now.

### Why the empty string is not a number

Coercing it to a null quietly would throw away the distinction the string column went to some trouble to keep. An empty field and a quoted empty field are different values in a CSV file, the reader keeps them apart, and a cast that collapses them undoes that two lines later.

### How the type of a column is decided

Inference climbs a ladder, bool to int64 to float64 to string, and never descends. A value that does not fit the current rung moves the column up one rung and the column keeps the widest rung any value forced. A missing value decides nothing, so a column of numbers with a gap in it is still a column of numbers, and a column with nothing in it at all is text, because text is the only rung that can hold whatever eventually turns up. A quoted field is not promoted to text for being quoted, since quoting says where a field ends and not what is in it, but a quoted empty string is the empty string and not a null, which is the one place quoting does change a value.

### Notes on the numbers

Measured on an i9-13900K against one million rows of four columns, 20.7 MB, the same file for every reader. firepanda reads it in 139.7 ms median. `pandas.read_csv` with the C engine takes 137.1 ms and with the pyarrow engine 28.8 ms from the file and 12.4 ms from memory. So this is at parity with the engine pandas has had for fifteen years and roughly eleven times behind the threaded Arrow reader. The gap is threads. The scan and the conversion both run on one core here, and Arrow runs both across all of them. The M1 exit criterion asks for a reader that beats the pyarrow engine and this is not that reader yet.

## [0.6.7] - 2026-08-28

Built against Mojo 1.0.0 (ed45d567).

A text column can be aggregated, not only grouped by. With 0.6.5 sorting one and 0.6.6 grouping by one, a string column is now a first class column everywhere a group by can reach it.

A patch bump. One fix, and the rest is addition.

### Added

- A text column can be aggregated. `count`, `size`, `nunique`, `first`, `last`, `min` and `max` all work on a string column, per group, through the same `group_by` call the number columns go through.
- `aggregate_group_strings`, the kernel behind it, and `group_text_scalar`, the obvious twin the tests compare it against.
- `tests/test_text_agg.mojo`, eighteen tests, and a round in the string fuzzer that checks the four value reductions against an inline reference.

### Fixed

- A string column reaching the aggregation dispatch matched `uint8` and was summed as bytes, so `sum` over a column of names returned a number rather than an error. The dispatch now asks whether the column is text first, the same way the sort, group by and join dispatches do. That is the last of the three places the hazard lived.

### Why nunique and min take different routes

`nunique` factorizes the column and then runs the number kernel over the ordinals, because two rows hold the same bytes exactly when they share an ordinal, and the number kernel already knows how to count distinct values per group. `first`, `last`, `min` and `max` never copy a value during the scan. They keep one row number per group and gather once at the end through a `StringBuilder`, so a column of long values costs the same to reduce as a column of short ones and only the surviving values are ever written.

### Notes on the numbers

Measured on an i9-13900K, two hundred thousand rows over a hundred groups, fifteen repetitions, interquartile range around 1 percent. `min` over a text column runs at 6.0 ns a row and `nunique` at 16.9, the gap being the factorize pass the second one pays for. On the AMD EPYC VPS the same rows are 31.3 and 64.5 ns.

## [0.6.6] - 2026-08-28

Built against Mojo 1.0.0 (ed45d567).

A text column can be a group by key and a join key. 0.6.5 made one sortable and this makes one groupable, which together are what db-benchmark's group by queries need from a string column.

A patch bump. One fix, and the rest is addition.

### Added

- A text column can be a group by key and a join key. `group_by` accepts one on its own or alongside number keys, with or without `dropna`, sorted or in first seen order, and the joins take one on either side.
- `factorize_strings`, which assigns a group ordinal to every element of a `StringArray`, and `FactorizedStrings`, which carries the codes, the row that first showed each group and the null group.
- `hash_bytes` and `hash_strings_chunk`, a length seeded hash over a run of bytes and the chunked form the table build consumes.
- `HashTable.build_strings`, the probe loop for keys that do not fit in the hash.
- `factorize_strings_linear`, the scalar twin, which compares bytes against every group it has seen and never hashes.
- `tests/test_text_group.mojo`, twenty one tests, and a factorize round in the string fuzzer against a reference that shares no code with the kernel.

### Fixed

- A string column reaching the group by and join dispatch matched `uint8` and was grouped on the first byte of each view, which put every value starting with the same letter in one group. Both dispatches now ask whether the column is text before the type walk. The same hazard was fixed in the sort in 0.6.5 and this is the rest of it.

### Why text needs its own table build

The existing table stores a 64 bit hash and treats hash equality as key equality. That is exact for a fixed width key, because the mix is a bijection on 64 bits, so two keys that hash alike are the same key. Text does not fit in 64 bits and the hash is a real hash, so `build_strings` compares the bytes against the row that first showed the group and keeps probing when they differ. The two builds are kept apart rather than merged behind a flag, so the fixed width one stays a loop with nothing extra in it.

### Notes on the numbers

Measured on an i9-13900K, one million rows, fifteen repetitions, interquartile range under 4 percent on every row. Distinct text keys group at 14.9 ns a row against 14.3 for an int64 column of the same height that also goes through the hash. A column where every element shares a nine byte prefix costs 16.8 ns a row, which is the case that costs the sort 307.6, because a group only has to find its bucket and does not have to order anything. A hundred repeated values cost 10.6 ns a row against 3.6 for the int64 column, which takes a direct route that text has no equivalent of.

## [0.6.5] - 2026-08-28

Built against Mojo 1.0.0 (ed45d567).

A text column can be sorted. 0.6.4 got one into the frame layer and left it unsortable, and this is the ordering that db-benchmark q1, q2, q3, q7 and q10 need.

A patch bump. Everything here is addition apart from one fix to a round trip that was returning mojibake.

### Added

- A text column can be ordered. `sort_values`, `argsort` and a frame sort all accept one, ascending or descending, nulls first or last, and the order is stable.
- `StringArray.sort_prefix`, `compare_elements` and `compare`, which are the three pieces the sort needs: a radix key, an ordering between two elements of a column, and an ordering between an element and a run of bytes for a future search.
- `tests/test_text_sort.mojo`, fourteen tests that separate the radix half from the tie break half so that neither can pass on the other's behalf.
- A differential sort round in the string fuzzer, against a reference insertion sort that shares no code with the kernel. It compares the permutation position by position rather than the sorted elements, because two permutations can produce the same elements and only one of them is stable.

### Fixed

- `StringArray.__getitem__` returned mojibake for any element with a byte above 127. It appended `chr(byte)` per byte, which treats every byte as a code point, so a column holding "ábove" read back as "Ã¡bove". It now hands the bytes over whole. The column stores bytes and does not interpret them, and this is the one place that has to say what they mean.

### How the sort works

The first eight bytes of an element pack into a `UInt64` most significant byte first, so comparing two of those integers gives the same answer as comparing the first eight bytes of the elements, and a shorter element sorts before one that extends it. That means the existing radix sort runs over text unchanged, at eight digits. Only the runs whose keys came out identical need a comparison, and those are finished by a stable insertion sort below sixteen rows and a stable bottom-up merge above it.

A run whose elements are all at most eight bytes and all the same length is already in its final order, because the key held the whole element, and it is skipped without a single comparison. That is not a corner case, it is the shape of the columns a dataframe most often sorts. Without it a column of a hundred distinct two byte labels cost 403 ns a row against 27 for an int64 column of the same height.

### Notes on the numbers

Measured on an i9-13900K, 262144 rows. Distinct keys 23.2 ns a row against 17.2 for an int64 column of the same height through the same entry point. A hundred repeated short values 14.8 ns a row, which is faster than the int64 column because the settled check skips every comparison and the low entropy keys let the radix passes skip digits. The pathological case is a column where every element shares a nine byte prefix, at 424 ns a row, where the radix does no work at all and the merge does all of it. A wider key would move that case rather than fix it.

## [0.6.4] - 2026-08-28

Built against Mojo 1.0.0 (ed45d567).

A `DataFrame` can hold a text column. 0.6.3 shipped `StringArray` as a standalone module with nothing above it able to hold one, and this connects it to `AnyArray`, which is the type-erased column a `Series` and a frame are made of.

A patch bump. Everything here is addition, and the four operations that now refuse a text column previously could not be handed one at all.

### Added

- `AnyArray` carries a string column. New `AnyArray(StringArray)` constructor, plus `is_string()`, `strings()` for a borrow and `into_strings()` for a move.
- `Series` takes a `StringArray`, and reports `is_string()`, `as_strings()` and `text(i)`.
- Take, filter, slice, concat, coalesce, fill forward and fill backward all carry text through the erased path, which means a `DataFrame` can select, reorder, cut and stack frames with text columns in them.
- The display layer prints a text value rather than its type, and `<NA>` for a null. Values are not quoted and not escaped, which is what pandas does too, because a table is read by a person and `to_csv` is where escaping belongs.
- `tests/test_text.mojo`, twenty one tests, one per operation that reads values.
- A `text/` benchmark group measuring the erased path, with every text row paired against an int64 column of the same height through the same entry point.

### Fixed

- Every kernel that dispatches by matching on `col.dtype()` now asks `is_string()` first. `LogicalType.STRING` has physical dtype uint8 and uint8 is a member of both `ALL` and `ORDERED`, so a string column reaching one of those dispatches would have matched the uint8 arm and read the first byte of a sixteen byte view as the value. That is not a crash and not an obviously wrong answer, it is a plausible number, and a sum over a column of country codes would have returned a total. This was unreachable before 0.6.3 because no frame could hold a text column, and it is closed in the same release that makes it reachable.
- `AnyArray.check_dtype` refuses a string column for every dtype, so `as_typed[DType.uint8]()` on a column of names raises instead of handing back a column over the views buffer.
- `concat` compares `is_string()` as well as `dtype()`. Both a text column and a column of bytes are uint8 physically, so the dtype check on its own let the pair through.

### Not yet

Cast, ordering comparison and group by aggregation refuse a text column, each with a message naming what is missing rather than a dtype the caller did not choose. Ordering is next and is what db-benchmark q1, q2, q3, q7 and q10 need.

### The validity duplication

`AnyArray` keeps the validity bitmap in `data` as well as inside the `StringArray`. `is_null`, `is_not_null` and the all-present mask that a join and a group by both build read `data.validity` and never look at a value, so keeping the bitmap where they already look means those three need no text case. A column is immutable once constructed, so both copies are written from the same source in the same call and cannot drift. The cost is one bit per row per text column and it goes away when columns are refcounted rather than deep copied.

### Notes on the numbers

Measured on an i9-13900K at 262144 rows, ten repetitions, median. Per row: take 78.5 ns for text against 2.0 ns for int64, filter 8.8 against 1.4, concat 25.1 against 0.4, slice 15.5 against 0.2. The `is_string` guard that is now on the front of every erased kernel costs 0.14 ns.

The 40x on the gather is mostly not inherent. A standalone probe separates it: at 32 byte elements a sequential take is 48 ns/row and a scattered one is 247, which is two dependent cache misses per row, the view and then the payload it points at. Every one of those addresses is derivable before the loop starts because the index list is the input, and none of them are prefetched today. That is worth fixing before string keys reach a join.

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

[Unreleased]: https://github.com/tamnd/firepanda/compare/v0.6.23...HEAD
[0.6.23]: https://github.com/tamnd/firepanda/releases/tag/v0.6.23
[0.6.22]: https://github.com/tamnd/firepanda/releases/tag/v0.6.22
[0.6.21]: https://github.com/tamnd/firepanda/releases/tag/v0.6.21
[0.6.20]: https://github.com/tamnd/firepanda/releases/tag/v0.6.20
[0.6.19]: https://github.com/tamnd/firepanda/releases/tag/v0.6.19
[0.6.18]: https://github.com/tamnd/firepanda/releases/tag/v0.6.18
[0.6.17]: https://github.com/tamnd/firepanda/releases/tag/v0.6.17
[0.6.16]: https://github.com/tamnd/firepanda/releases/tag/v0.6.16
[0.6.15]: https://github.com/tamnd/firepanda/releases/tag/v0.6.15
[0.6.14]: https://github.com/tamnd/firepanda/releases/tag/v0.6.14
[0.6.13]: https://github.com/tamnd/firepanda/releases/tag/v0.6.13
[0.6.12]: https://github.com/tamnd/firepanda/releases/tag/v0.6.12
[0.6.11]: https://github.com/tamnd/firepanda/releases/tag/v0.6.11
[0.6.10]: https://github.com/tamnd/firepanda/releases/tag/v0.6.10
[0.6.9]: https://github.com/tamnd/firepanda/releases/tag/v0.6.9
[0.6.8]: https://github.com/tamnd/firepanda/releases/tag/v0.6.8
[0.6.7]: https://github.com/tamnd/firepanda/releases/tag/v0.6.7
[0.6.6]: https://github.com/tamnd/firepanda/releases/tag/v0.6.6
[0.6.5]: https://github.com/tamnd/firepanda/releases/tag/v0.6.5
[0.6.4]: https://github.com/tamnd/firepanda/releases/tag/v0.6.4
[0.6.3]: https://github.com/tamnd/firepanda/releases/tag/v0.6.3
[0.6.2]: https://github.com/tamnd/firepanda/releases/tag/v0.6.2
[0.6.1]: https://github.com/tamnd/firepanda/releases/tag/v0.6.1
[0.6.0]: https://github.com/tamnd/firepanda/releases/tag/v0.6.0
[0.5.0]: https://github.com/tamnd/firepanda/releases/tag/v0.5.0
[0.4.0]: https://github.com/tamnd/firepanda/releases/tag/v0.4.0
[0.3.0]: https://github.com/tamnd/firepanda/releases/tag/v0.3.0
[0.2.0]: https://github.com/tamnd/firepanda/releases/tag/v0.2.0
[0.1.0]: https://github.com/tamnd/firepanda/releases/tag/v0.1.0
