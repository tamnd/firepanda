# DuckDB storage

## The file

One file. It opens with a 64 bit checksum over the main header, then the four bytes `DUCK`, then the storage version. After that it is fixed size blocks, and every structure in the database is a chain of block pointers.

Compression is applied to persistent databases and not to in memory ones, which is worth knowing when reading benchmarks: an in memory DuckDB table is uncompressed and a file backed one is not.

Reported sizes are roughly a quarter of the equivalent CSV and roughly one and a fifth of the equivalent Parquet. DuckDB is not trying to beat Parquet on size, it is trying to be fast to scan and updateable.

## Row groups

A table is a `RowGroupCollection`, which is a horizontally partitioned list of `RowGroup`s held in a segment tree. A row group holds up to 122,880 rows.

That number is 60 times 2048. It is the unit of three separate things and that is why it matters.

It is the unit of parallelism. Scan parallelism starts at the row group, so a query needs at least `threads * 122880` rows before every thread has work. Ten million rows is 82 row groups, which is enough for 32 threads. A hundred thousand rows is one row group and runs on one thread no matter what you set.

It is the unit of statistics. Each row group carries per column min, max, count and null count, so a predicate can skip a whole row group without reading it.

It is the unit of checkpointing and compression selection. Compression is chosen per column segment at checkpoint time, and the analyze phase samples across the row group.

`ROW_GROUP_SIZE` is settable on `ATTACH`, which matters for small tables where the default leaves you single threaded.

## Column segments

Inside a row group each column is a `ColumnData` holding a tree of `ColumnSegment`s. A segment is a block of compressed values with its own statistics. Segments typically cover 2048 rows, which lines up with the vector size, though compression and updates can change that.

`ColumnData` has subclasses for the shapes that need children: `StandardColumnData` for primitives with validity, `StructColumnData`, `ListColumnData`, `ArrayColumnData`, and `VariantColumnData` which implements shredded storage for the new variant type.

Columns are loaded lazily from metadata pointers, with an atomic per column flag saying whether they are in memory. Opening a wide table does not read all its columns.

## Compression

Chosen per segment by a sampling analyzer that tries the candidates and picks the smallest. The candidates are constant, run length, bit packing, frame of reference, dictionary, FSST, ALP, ALPRD, Chimp, Patas, roaring bitmaps for validity, zstd, and uncompressed as the fallback.

The ones worth understanding:

**Bit packing** tracks the maximum over each group of 1024 values and packs to the width that maximum needs. **Frame of reference** adds a minimum, so a column of dates around a common epoch packs to the bits its range needs rather than the bits its magnitude needs.

**Dictionary** for text with repeats, and **FSST** on top of it. FSST builds a symbol table of up to 255 frequently occurring byte sequences and replaces occurrences with single byte codes, which catches repetition inside strings and not just repetition of whole strings. URL columns and log lines are the case it was designed for.

**ALP** is for floating point. It multiplies doubles by a power of ten to turn them back into the integers they were before somebody stored a price as a double, then bit packs those. **ALPRD** handles the doubles that really are doubles by splitting the bit pattern into a left part with few distinct values and a right part that is incompressible. DuckDB's own measurement is that ALP decompresses two to four times faster than Patas at twice the ratio.

The analyze phase runs per vector, per 2048 values, and the per value methods like Patas and Chimp do a dry run there to estimate the compressed size. ALP needs a sample of the whole row group so it only collects during that step and decides later.

`PRAGMA storage_info` prints the row group id, column, segment type, count, compression and block for every segment, which is the way to check what actually happened.

## Zone maps

The min, max, count and null count on each segment and each row group. A filter on a column with any ordering at all skips most of the table. This is why DuckDB is fast on a range predicate over a sorted or clustered column and merely normal on a random one.

## Updates and transactions

MVCC. Each row group carries version info tracking inserts and deletes. Updates go to an update segment attached to the column rather than rewriting the base data. Appends during a transaction live in `LocalStorage`, and an `OptimisticDataWriter` writes parallel bulk inserts to temporary blocks before commit so a large load does not have to be buffered in memory.

## What we should take from this document

Not much directly, because firepanda is not a storage engine and document 00-README of the parent spec says so explicitly.

Two things transfer anyway.

The first is the row group as a unit of parallelism and statistics. Even for in memory frames, carrying min, max and null count per chunk of a column would let a filter skip chunks, and firepanda's `ChunkedArray` already has the place to put it.

The second is the number 122,880 as a chunk size for parallel scan, and the observation that it is 60 vectors. When `02-execution-model.md` picks a morsel size, this is one of the two data points, and Polars' 128k is the other. They agree to within five percent, which is not a coincidence.
