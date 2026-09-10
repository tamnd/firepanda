# Front doors

Six ways a query string reaches the engine. They differ in who wrote the string, what names are in scope, and how much the caller should be trusted, and that last one is why the capability flag from document 02 is specified here rather than added later.

## 1. `firepanda.sql(query, **params)`

The main door. Takes SQL, returns a `DataFrame`.

```python
import firepanda as fp
customers = fp.read_parquet("customers.parquet")
orders    = fp.read_parquet("orders.parquet")

fp.sql("""
    SELECT c.name, sum(o.total) AS spend
    FROM customers c JOIN orders o ON o.cust_id = c.id
    GROUP BY ALL ORDER BY spend DESC LIMIT 10
""")
```

`customers` and `orders` are not registered. They are Python locals in the calling frame, found by name, which is DuckDB's behaviour and is most of why `duckdb.sql()` feels good. The rules, stated because implicit capture that is not specified becomes a bug report:

Resolution order is explicit registrations, then the caller's locals, then the caller's globals. Only `DataFrame` and `Series` are captured, and a name bound to something else is skipped rather than raising, so a local called `orders` that is a list does not shadow a registered frame. Capture is by borrow and not by copy, so the frame's buffers are used in place and the Python object is kept alive for the duration of the call. And capture is opt out via `fp.sql(..., capture=False)`, because a library calling `sql()` on behalf of its own caller does not want that caller's locals in scope.

Two frames of the same name, one in locals and one in a registration, is not ambiguous, because registration wins, explicitly. Two names differing only in case is ambiguous and raises, because identifiers fold down per document 04.

## 2. `df.sql(query)`

The frame itself, bound to the name `self`, matching DuckDB's relational API:

```python
df.sql("SELECT a, sum(b) FROM self GROUP BY a")
```

Other names still resolve by the rules above. This is the door that composes with method chaining and it is the one a pandas user will reach for, so it gets the same latency budget as `sql()` and no extra copy.

## 3. `df.query(expr)` and `df.eval(expr)`, which are pandas compatibility

These are pandas API surface and therefore governed by `docs/specs/06-pandas-parity.md` rather than by DuckDB compatibility, which makes them the one place in this specification where we implement a dialect that is not DuckDB's.

pandas' expression language is Python syntax and not SQL: `df.query("a > 1 and b < 2")`, with `@` for locals, backticks for column names that are not identifiers, `and`, `or` and `not`, `in` and `not in`, chained comparisons, and `parser="pandas"` against `parser="python"` changing operator precedence for `&` and `|`. numexpr is used above a row threshold and pure Python below it.

We cannot route this through the SQL parser, and pretending otherwise would break both. So we write a small dedicated parser for the pandas expression grammar, producing the same `BoundExpr` from document 08 and lowering to `Filter` and `Compute` nodes. A few hundred lines, a closed grammar that has not changed in years, and it shares everything below the AST.

The one deliberate difference from pandas, documented rather than hidden: pandas' `query` with `engine="numexpr"` computes in float64 in places where the dtypes would suggest otherwise, and it silently falls back to Python for unsupported expressions. We compute with firepanda's own type rules throughout. `docs/specs/06-pandas-parity.md` already has the mechanism for recording a divergence and this is one.

## 4. Parameters

Three forms, all from the grammar: `?` positional, `$1` numbered, and `$name` or `:name` named.

```python
fp.sql("SELECT * FROM t WHERE a > ? AND b = ?", 5, "x")
fp.sql("SELECT * FROM t WHERE a > $lo", lo=5)
```

A parameter is a value, never text. It is bound after parsing, into a `Parameter` node in the plan, and substituted at execution. There is no string interpolation anywhere in the implementation and no API that accepts one, which makes SQL injection structurally impossible on this path rather than discouraged. That property is worth more than any documentation about it.

Parameter types are inferred from the plan where the parameter is used, matching DuckDB, so `a > ?` against an INTEGER column binds an integer and a string argument raises a conversion error naming the parameter.

## 5. The prepared statement cache

Document 01 promises a repeat execution under 2 us to the physical plan, and this is the mechanism.

The key is the exact query text, plus a catalog generation counter, plus the values of any settings that affect binding. The value is the bound and optimized plan. Parameters are not part of the key, which is the entire point, because a loop over a parameterized query parses once.

Soundness rests on document 04's rule that parsing is a pure function of text and grammar, and on document 05's rule that binding depends only on the catalog. The generation counter increments on any registration, deregistration or schema change, which invalidates the cache wholesale. Wholesale rather than selectively, because a REPL registers frames rarely and runs queries constantly, and a precise invalidation scheme is a correctness risk bought with no measurable benefit.

`PREPARE` and `EXECUTE` are the explicit form of the same thing and are in tier 2 of document 05.

## 6. The CLI

`firepanda -c "SELECT ..."` and an interactive REPL, because a SQL engine without a shell is hard to explore and harder to demo.

The REPL gets history, multi line continuation until a semicolon, `.mode` output formats matching DuckDB's box drawing default, tab completion over the registered names and the function catalog, and `.timer`. `-c` and a piped stdin are non interactive and get the restrictive capability default from section 8.

It is also the fastest path to a conformance harness, because document 11's runner drives the CLI the way DuckDB's `sqllogictest` runner drives DuckDB.

## 7. ADBC

The one door that makes firepanda usable from something that is not Python or Mojo, and it is cheap because ADBC 1.1.0 is a C API over the Arrow C Data Interface, which firepanda already exports per `docs/specs/15-the-arrow-capsule-boundary.md`.

A driver implementing `AdbcDatabase`, `AdbcConnection` and `AdbcStatement`, with `ExecuteQuery` returning an `ArrowArrayStream` over the result's chunks. No serialization, because the stream is the chunks. This gets firepanda into anything that speaks ADBC, meaning notebooks, BI tools and other languages, for a few hundred lines and no new engine surface.

It is also the door where the query text comes from somewhere else, which is section 8.

## 8. The capability flag

`enable_external_access`, from document 02, specified here.

When false, the table functions in document 07 refuse: `read_csv`, `read_parquet`, `read_json`, `glob`, `COPY ... TO`, and the replacement scan that turns `FROM 'file.parquet'` into one of them. The refusal is a clear error naming the setting, not a file not found.

Defaults by door. `fp.sql()` and `df.sql()` are permissive, because the caller is the program and the program can already open files. CLI `-c` and piped stdin are restrictive, because the string may have come from a script assembling it. ADBC is restrictive, because by construction the query came from another process.

The setting is one way within a session, so it can be turned off and not back on. DuckDB has the same lock and the reason is the same: a query that could turn it back on would make it decorative.

Document 13 keeps one case open, which is `df.sql()` called inside a user defined function that is itself running inside a query, where the caller is the program argument gets weaker.

## 9. Errors at the boundary

`docs/specs/14-errors-across-the-boundary.md` governs the translation, and SQL adds the requirement from document 02: the message includes the query text, the line, and a caret at the offending position.

Exception types map so that Python code can catch usefully rather than matching strings. Syntax and binder errors become a `SQLError` subclassing `ValueError`, unsupported features become `NotImplementedError`, and conversion and overflow become the same exceptions the dataframe path already raises for the same conditions.

The traceback should point at the user's `fp.sql(...)` call and not into the binder. A five frame Mojo traceback above a one line SQL error is noise, and the caret is the information.

## 10. The GIL

Issue #204 tracks releasing the GIL around execution. A SQL call is the best case for it: one long unit of work, no Python objects touched between the frame capture at the start and the result construction at the end.

The release goes around planning and execution, not around capture. Capture reads the caller's frame dictionaries and must hold the GIL, and execution touches only firepanda's own memory and must not. That split is what makes `concurrent.futures` over several `fp.sql()` calls actually parallel, which is a thing users try immediately and which fails silently, as a performance non result rather than an error, if the boundary is drawn in the wrong place.
