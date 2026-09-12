# pandas conformance checklist

The target is pandas 3.0, released 21 January 2026, currently 3.0.5, not the 2.x series most people are still running. That distinction matters: 3.0 made Copy-on-Write the only mode, made the Arrow backed string dtype the default, shipped `pd.col()` expressions and moved datetimes to microsecond resolution. The API we are matching has already moved toward the design in this spec.

The rule for this document is that everything pandas can do, firepanda can do, under the same name. Unlike the Go sibling spec, we do not get to change the shape to suit the host language's idiom, because the host language's idiom is Python. Where the name cannot carry over, the note says why, and there are far fewer of those here than there.

Tags:

- `(M1)` through `(M11)` is the milestone from document 08
- `same` means the pandas name and signature carry over unchanged, which is the default and is therefore not written
- `adapted` means same capability, different behaviour, with a note
- `omitted` means genuinely not provided, with what to use instead

## A note on the index

firepanda frames have **no implicit index and never align automatically**. A frame may carry an **optional, explicitly declared index**, and when it does, every label based operation works: `loc`, `reindex`, `align`, `sort_index`, `asof`, `reset_index`, `set_index`, the lot.

What never happens is pandas' behaviour of silently aligning two frames on their indexes during arithmetic. You get the capability without the surprise.

```python
df = df.set_index("ts")     # explicit, opt in
df.loc["2026-01-01"]        # now available
a + b                       # still positional, never aligned behind your back
a.align(b) ; a + b          # align explicitly if that is what you meant
```

A second note, internal rather than user facing: Mojo 1.0 removed negative indexing from `List`, `Span` and `String`. Every row below involving positional indexing normalizes negative indices at the API boundary. `iloc[-1]` works. This is written down once here and not repeated on forty rows.

---

## 1. Top level functions

### IO

- [ ] `read_csv` (M1)
- [ ] `read_table` (M1)
- [ ] `read_fwf` (M11)
- [ ] `read_parquet` (M2)
- [ ] `read_feather` -> Arrow IPC (M2)
- [ ] `read_orc` (M11)
- [ ] `read_json` (M2)
- [ ] `read_json` with `lines=True` -> NDJSON (M2)
- [ ] `json_normalize` -> flattening of nested struct columns (M6)
- [ ] `read_sql`, `read_sql_query`, `read_sql_table` (M11)
- [ ] `read_clipboard` (M11)
- [ ] `read_excel`, `ExcelFile`, `ExcelWriter` (M11)
- [ ] `read_html` (M11)
- [ ] `read_xml` (M11)
- [ ] `read_hdf` — `omitted`, no HDF5 dependency. Use Parquet or IPC.
- [ ] `read_pickle`, `to_pickle` — `omitted`, raises with a pointer to Parquet and IPC
- [ ] `read_sas`, `read_spss`, `read_stata` (post 1.0)

### Reshaping and combining

- [ ] `concat` (M1)
- [ ] `merge` (M1)
- [ ] `merge_ordered` (M7)
- [ ] `merge_asof` (M7)
- [ ] `melt` (M6)
- [ ] `pivot` (M6)
- [ ] `pivot_table` (M6)
- [ ] `crosstab` (M6)
- [ ] `cut` (M6)
- [ ] `qcut` (M6)
- [ ] `get_dummies` (M6)
- [ ] `from_dummies` (M6)
- [ ] `factorize` (M6)
- [ ] `unique` (M6)
- [ ] `lreshape` (M6)
- [ ] `wide_to_long` (M6)

### Missing data and conversion

- [ ] `isna`, `isnull` (M1)
- [ ] `notna`, `notnull` (M1)
- [ ] `to_numeric` with `errors=` (M1)
- [ ] `to_datetime` with format and inference (M7)
- [ ] `to_timedelta` (M7)
- [ ] `array`, `Series`, `DataFrame` constructors (M1)

### Date ranges and offsets

- [ ] `date_range` (M7)
- [ ] `bdate_range` (M7)
- [ ] `timedelta_range` (M7)
- [ ] `period_range` (M7)
- [ ] `interval_range` (M7)
- [ ] `infer_freq` (M7)
- [ ] `offsets.*`, roughly forty offset classes plus business day and holiday calendars (M7)
- [ ] `Timestamp`, `Timedelta`, `Period`, `Interval` scalar types (M7)
- [ ] `Grouper` (M7)

### Expressions, evaluation and options

- [ ] `col` — pandas 3.0's `pd.col()` expression builder, and firepanda's native spelling (M4)
- [ ] `eval` (M4b)
- [ ] `query` on `DataFrame` (M4b)
- [ ] `set_option`, `get_option`, `reset_option`, `describe_option`, `option_context` (M1)
- [ ] `show_versions` (M1)
- [ ] `testing.assert_frame_equal`, `assert_series_equal`, `assert_index_equal` (M1)

---

## 2. DataFrame

### Attributes

- [ ] `index`, `columns`, `dtypes`, `shape`, `size`, `ndim`, `empty`, `values`, `axes`, `flags`, `attrs` (M1)
- [ ] `T` — `adapted`, transpose of a columnar frame requires a common dtype and raises otherwise (M6)

### Selection and indexing

- [ ] `__getitem__` for column, list of columns, boolean mask, slice (M1)
- [ ] `__setitem__` for column assignment (M1)
- [x] `loc`, `iloc`, `at`, `iat` (M6), apart from a key that collapses the row axis and leaves the columns, which answers a row as a series and needs one type every column fits
- [ ] `head`, `tail`, `sample`, `nlargest`, `nsmallest` (M1)
- [ ] `filter` with `items`, `like`, `regex` (M6)
- [x] `take` (M6), on either axis and counting a negative position from the end
- [ ] `xs`, `get`, `pop`, `insert` (M6)
- [ ] `where`, `mask`, `query` (M6, `query` M4b)
- [x] `isin` on Series (M1) and on DataFrame (M6), apart from a frame or a series as the argument, which pandas lines up by label and compares cell against cell rather than reading as a set
- [ ] `between` on Series (M1)
- [ ] `first`, `last` (M7)

### Reshaping and sorting

- [ ] `sort_values` with multiple keys, `ascending` list, `na_position` (M1)
- [x] `sort_index` (M6, `level`, `key` and a direction per level still to come)
- [ ] `rank` with all `method=` values (M6)
- [x] `set_index`, `reset_index` (M6, one column only until there is a MultiIndex)
- [ ] `reindex`, `reindex_like`, `rename`, `rename_axis`, `set_axis` (M6)
- [ ] `stack`, `unstack`, `melt`, `pivot`, `pivot_table`, `explode`, `squeeze` (M6)
- [ ] `transpose` — see `T` above
- [ ] `droplevel`, `swaplevel`, `reorder_levels` (M6, compound index only)
- [ ] `assign`, `eval` (M6, `eval` M4b)

### Combining

- [ ] `merge`, `join` (M1)
- [ ] `concat` via top level (M1)
- [ ] `combine`, `combine_first`, `update` (M6)
- [ ] `compare` (M6)
- [ ] `align` — explicit only, per the index note (M6)

### Missing data

- [ ] `isna`, `notna`, `dropna`, `fillna`, `ffill`, `bfill`, `interpolate`, `replace` (M1 for the first four, M6 for the rest)
- [ ] `dropna` with `how`, `thresh`, `subset` (M1)
- [ ] `fillna` with a scalar, a dict, a Series (M1)
- [ ] `interpolate` with `linear`, `time`, `nearest`, `polynomial`, `spline` (M6)

### Computation and descriptive statistics

- [ ] `sum`, `mean`, `median`, `min`, `max`, `prod`, `count`, `std`, `var`, `sem`, `mad` (M1)
- [ ] `skew`, `kurt`, `quantile`, `mode` (M6)
- [ ] `cumsum`, `cumprod`, `cummax`, `cummin` (M6)
- [ ] `diff`, `pct_change`, `shift` (M6)
- [ ] `abs`, `clip`, `round` (M1)
- [ ] `corr`, `cov`, `corrwith` — pearson, kendall, spearman (M6)
- [ ] `describe` with `include` and `exclude` (M6)
- [ ] `value_counts`, `nunique`, `idxmax`, `idxmin` (M6)
- [ ] `any`, `all` (M1)
- [x] `drop_duplicates` (M6) — with and without a subset, keeping the first of each repeat and keeping null keys
- [ ] `duplicated` (M6)
- [ ] `apply`, `map`, `pipe`, `agg`, `transform` (M6, and see document 04 section 7 on Python callables)
- [ ] `applymap` — `adapted`, `map` in pandas 3.0
- [ ] `groupby` (M1 for the basic aggregations, M6 for the rest)
- [x] `rolling`, `expanding`, `ewm` (M6)
- [ ] `resample` (M7)
- [ ] `dot`, `add`, `sub`, `mul`, `div`, `mod`, `pow` and the `r`-prefixed reflected forms (M1)
- [ ] `add` with `fill_value=` (M6)
- [ ] `mode`, `nunique` per axis (M6)

### Type conversion and metadata

- [ ] `astype`, `convert_dtypes`, `infer_objects`, `copy` (M1)
- [ ] `select_dtypes` (M6)
- [ ] `info`, `memory_usage` (M6)
- [ ] `to_numpy`, `to_records`, `to_dict`, `to_arrow`, `to_pandas`, `to_polars` (M2 and M3)
- [ ] `equals` (M1)

### Output

- [ ] `to_csv`, `to_parquet`, `to_feather`, `to_json`, `to_string`, `to_markdown` (M1 and M2)
- [ ] `to_sql`, `to_orc`, `to_excel`, `to_html`, `to_xml`, `to_latex`, `to_clipboard` (M11)
- [ ] `style` — `omitted` until post 1.0
- [ ] `plot` — `omitted`. `df.to_pandas().plot()`, or hand the Arrow frame to any plotting library at zero copy.

---

## 3. Series

Everything in section 2 that makes sense on one dimension, plus:

- [ ] `values`, `array`, `name`, `dtype`, `hasnans`, `is_unique`, `is_monotonic_increasing`, `is_monotonic_decreasing` (M1)
- [ ] `item`, `tolist`, `to_frame`, `to_list`, `unique`, `nunique`, `value_counts` (M1)
- [ ] `argsort`, `argmin`, `argmax`, `searchsorted` (M6)
- [ ] `repeat`, `reindex`, `rename`, `reset_index` (M6)
- [ ] `autocorr`, `between`, `clip`, `divmod` (M6)
- [ ] `str`, `dt`, `cat`, `list`, `struct` accessors (sections 4 to 8)
- [ ] `map` with a dict, a Series, or a callable (M6)
- [ ] `combine`, `combine_first`, `update` (M6)
- [ ] `nlargest`, `nsmallest` (M1)
- [ ] `to_numpy` zero copy where the dtype allows (M3)

---

## 4. The `.str` accessor

Ships whole at M6. A partial string namespace reads as a toy, and this is where most real data work happens.

RE2 semantics throughout: linear time, no backreferences, no lookaround. That is a documented divergence and it is prominent in the migration guide, because a regex that works in pandas and raises here is a bad afternoon.

`len` counts characters and not bytes, which is worth writing down because there are two kernels underneath and only one of them is this. `str.len` is `Series.chars_length`, which runs `text_character_length` in `chars.mojo`, walking the payload and counting anything that is not a continuation byte, and it is the one that answers `'café'` with 4 the way pandas does. `text_byte_length` in `substr.mojo` is SQL's `STRLEN` and answers 5, and it is a read of the length field in each view rather than a pass over anything, which is thirteen times cheaper on a column of thirty two byte elements. DuckDB spells the two apart as well, `length` against `strlen`, so this is a difference between two questions rather than between two libraries. The accessor never reaches the byte length one: it is reached from SQL and from the kernel package, and a caller who wants it in Python asks for it by its own name rather than by a flag on `len`. Written up in document 30.

- [ ] `len` (M6), which ships as `chars_length` and stays unticked until there is a differential test for it in firepanda-bench, per the rule at the bottom of this document
- [ ] `lower`, `upper`, `title`, `capitalize`, `casefold`, `swapcase` (M6)
- [x] `strip`, `lstrip`, `rstrip`, `pad`, `center`, `ljust`, `rjust`, `zfill` (M6), and `wrap` is still to come
- [x] `slice`, `slice_replace`, `get`, `repeat` (M6), and `repeat` takes one count for the whole column rather than one per row
- [ ] `cat` with `sep` and `others` (M6)
- [ ] `split`, `rsplit`, `partition`, `rpartition`, with `expand=` (M6)
- [ ] `join` (M6)
- [ ] `contains`, `startswith`, `endswith`, `match`, `fullmatch` (M6)
- [ ] `find`, `rfind`, `index`, `rindex`, `count` (M6)
- [ ] `replace` with regex and literal, `removeprefix`, `removesuffix` (M6)
- [ ] `extract`, `extractall`, `findall` (M6)
- [ ] `get_dummies` (M6)
- [ ] `encode`, `decode`, `normalize` (M6)
- [ ] `isalnum`, `isalpha`, `isdigit`, `isspace`, `islower`, `isupper`, `istitle`, `isnumeric`, `isdecimal` (M6)
- [ ] `translate`, `casefold` (M6)

## 5. The `.dt` accessor

- [ ] `year`, `month`, `day`, `hour`, `minute`, `second`, `microsecond`, `nanosecond` (M7)
- [ ] `dayofweek`, `day_of_week`, `weekday`, `dayofyear`, `day_of_year`, `quarter`, `week`, `isocalendar` (M7)
- [ ] `days_in_month`, `daysinmonth`, `is_month_start`, `is_month_end`, `is_quarter_start`, `is_quarter_end`, `is_year_start`, `is_year_end`, `is_leap_year` (M7)
- [ ] `date`, `time`, `timetz`, `day_name`, `month_name` (M7)
- [ ] `floor`, `ceil`, `round`, `normalize` (M7)
- [ ] `tz`, `tz_localize`, `tz_convert` with an explicit ambiguous and nonexistent policy (M7)
- [ ] `to_period`, `to_pydatetime`, `strftime` (M7)
- [ ] `total_seconds`, `components` on durations (M7)

## 6. The `.cat` accessor

- [ ] `categories`, `ordered`, `codes` (M6)
- [ ] `rename_categories`, `reorder_categories`, `add_categories`, `remove_categories`, `remove_unused_categories`, `set_categories` (M6)
- [ ] `as_ordered`, `as_unordered` (M6)
- [ ] `CategoricalDtype` (M6)

Backed by Arrow dictionary encoding, which is the same representation the group by fast path in document 05 section 3 exploits.

## 7. The `.list` and `.struct` accessors

pandas 3.0 covers nested data thinly and Polars covers it well. We follow Polars here and expose the pandas spellings where they exist.

- [ ] `list.len`, `list.get`, `list.slice`, `list.contains`, `list.join` (M6)
- [ ] `list.sum`, `list.mean`, `list.min`, `list.max`, `list.sort`, `list.unique` (M6)
- [ ] `explode` on a list column (M6)
- [ ] `struct.field`, `struct.rename_fields`, `struct.unnest` (M6)
- [ ] `json_normalize` on a struct column (M6)

---

## 8. GroupBy

- [ ] `sum`, `mean`, `min`, `max`, `count`, `size`, `first`, `last`, `nth` (M1)
- [x] `std`, `var`, `sem`, `median`, `quantile`, `nunique`, `prod`, `any`, `all` (M1)
- [ ] `agg` with a string, a list, a dict, named aggregation kwargs (M1 basic, M6 complete)
- [x] `transform` with a built in aggregate, which is `group_broadcast` (M6)
- [ ] `transform` with a Python callable (M6, and see document 04 section 7)
- [ ] `apply` (M6, and see document 04 section 7)
- [ ] `filter` (M6)
- [ ] `cumsum`, `cumcount`, `cumprod`, `cummax`, `cummin`, `rank`, `shift`, `diff`, `pct_change` (M6)
- [ ] `head`, `tail` (M6)
- [x] `nlargest`, `nsmallest` (M6)
- [ ] `idxmin`, `idxmax`, `value_counts` (M6)
- [ ] `get_group`, `groups`, `ngroup`, `ngroups`, `indices` (M6)
- [ ] `pipe`, `describe` (M6)
- [ ] `as_index=False`, `sort=`, `dropna=`, `observed=` (M1)
- [ ] `Grouper` with `key`, `freq`, `level` (M7)
- [ ] `resample` on a grouped frame (M7)
- [ ] `rolling` and `expanding` on a grouped frame (M6)

## 9. Windows

- [x] `rolling` on a column, with all five of `window`, `min_periods`, `center`, `closed` and `step`, and with the answer stepped and relabelled (M6)
- [ ] `rolling` with a time based window, meaning an offset string on a datetime index (M7)
- [x] `expanding` on a column, with `min_periods` (M6)
- [x] `ewm` with `com`, `span`, `halflife`, `alpha`, `adjust`, `ignore_na` and `min_periods`, on a column and on a frame (M6)
- [ ] `ewm` with `times`, `method="table"` and `online` (M7)
- [ ] All of `sum`, `mean`, `median`, `min`, `max`, `std`, `var`, `count`, `quantile`, `skew`, `kurt`, `sem`, `corr`, `cov`, `rank` on each of the three (M6), of which all but `corr` and `cov` are written on `rolling` and `expanding`, and `mean`, `sum`, `var` and `std` are written on `ewm`, which is every reduction pandas puts on it apart from those same two
- [ ] `rolling.apply` with a callable (M6)
- [ ] `win_type` weighted windows: triang, gaussian, boxcar, and the rest (M6)
- [x] `over` with an aggregate and no ordering, which is the Polars spelling of `group_broadcast` (M6)
- [ ] `over` with an ordering or a window frame (M6)

## 10. Time series

The milestone that earns finance and observability users, because as of joins and DST correct resampling are simultaneously where pandas is most used and most error prone.

- [x] `DatetimeIndex`, an index whose labels are instants, with the calendar fields, the three roundings, `normalize`, `as_unit`, `day_name`, `month_name`, `strftime` and the two clock moves (M6), documented in `33-an-index-of-instants.md`, whose section 6 names what is still absent from it
- [ ] `resample` with all closed and label conventions, `origin`, `offset` (M7)
- [ ] `asfreq`, upsampling with `fill_method` and `limit` (M7)
- [ ] `merge_asof` with `direction` backward, forward, nearest, plus `by`, `tolerance`, `allow_exact_matches` (M7)
- [ ] `merge_ordered` with `fill_method` (M7)
- [ ] Full IANA timezone support with DST aware arithmetic (M7)
- [ ] Ambiguous and nonexistent local times with an explicit `earliest`, `latest`, `raise` policy (M7)
- [ ] `at_time`, `between_time`, `first`, `last`, `truncate` (M7)
- [ ] `tshift`, `shift` with `freq` (M7)
- [ ] `Period` and `PeriodIndex` arithmetic (M7)
- [ ] `Interval` and `IntervalIndex`, `IntervalDtype` (M7)
- [ ] Business day and custom holiday calendars (M7)
- [ ] Sorted key fast paths for as of join and resample on already ordered data (M7)

## 11. Types

- [ ] All integer widths including `Int128`, all float widths (M0)
- [ ] `boolean` (M0)
- [ ] `str`, the pandas 3.0 default string dtype, Arrow backed (M0)
- [ ] `category` (M6)
- [ ] `datetime64` at second, millisecond, microsecond and nanosecond, with and without timezone (M0 physical, M7 semantic)
- [ ] `timedelta64` (M0 physical, M7 semantic)
- [ ] `period`, `interval` (M7)
- [ ] `Decimal128` (M0)
- [ ] `list`, `struct`, `map`, `fixed_size_list` (M6)
- [ ] `object` — `omitted`, raises at construction naming the column and the Python type
- [ ] Nullable extension dtypes `Int64`, `Float64`, `boolean` — `adapted`, every dtype is nullable, so the capitalized variants are aliases

## 12. Interoperability

Not a pandas feature, and more important than most of them.

- [ ] Arrow C Data Interface, import and export, zero copy (M2)
- [ ] `__arrow_c_array__` and `__arrow_c_stream__` PyCapsule protocol (M3)
- [ ] `to_pandas`, `from_pandas` zero copy where the dtype allows (M3)
- [ ] `to_polars`, `from_polars` (M3)
- [ ] DuckDB: registering a firepanda frame as a scannable table (M3)
- [ ] `to_numpy` zero copy for primitive dtypes with no nulls (M3)
- [ ] Arrow IPC file and stream (M2)
- [ ] ADBC driver (M4b)

This section is the escape hatch that makes every gap elsewhere survivable. Anything firepanda has not implemented yet can be handed to pandas, Polars or DuckDB without a copy, which converts incompleteness from a blocker into an inconvenience. It is also why document 08 puts M2 before the lazy engine.

---

## The genuine omissions

Seven, and each is a decision rather than an oversight.

**`plot`.** Zero copy Arrow means matplotlib, plotly, altair and seaborn all work through `to_pandas()` or directly. Reimplementing a plotting stack is not a dataframe project.

**Pickle IO.** `read_pickle` and `to_pickle` raise. Pickling a Mojo struct is not a coherent operation, and pickle as an interchange format is a security problem that the ecosystem should stop teaching.

**`.style`.** Jupyter HTML styling is a real feature used by real people and it is post 1.0. `_repr_html_` exists from M3; the styling API does not.

**`dtype=object`.** There is no column of boxed anything. Raises at construction, naming the column and the offending Python type, and pointing at a struct column or `to_pandas()`.

**`inplace=`.** Raises a `TypeError` whose message contains the reassignment form. pandas 3.0 deprecated most of these upstream.

**Automatic index alignment.** Covered at the top. The capability is available through `align`; the implicit behaviour is not.

**Regex lookaround and backreferences.** RE2 semantics. This is the omission most likely to actually block someone, and it buys a linear time guarantee that Python's `re` cannot make. Documented at the top of the string namespace page rather than in a footnote.

## How this document is used

Every checkbox is a test in `firepanda-bench`, run against pandas 3.0.5 in the same process, comparing outputs on generated frames covering every dtype, null shape and edge case. Document 09 section 4.

The pass rate per section is published per commit. A section at 90 percent is reported as 90 percent, not as done.

An unticked box is not a bug report. A ticked box whose differential test does not exist is.
