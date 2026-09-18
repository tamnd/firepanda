# Conformance

The document that turns one hundred per cent DuckDB compatible from a claim into a number. DuckDB ships its own test suite under an MIT license: 4,796 `.test` files under `test/sql/` at the commit the grammar is pinned to, in sqllogictest format, which is a plain text format designed to be run by any engine. That corpus is the oracle, and everything here is about how to run it honestly.

## 1. The format

```
statement ok
CREATE TABLE t (a INTEGER, b VARCHAR);

query IR
SELECT a, b FROM t ORDER BY a
----
1	one
2	two
```

`statement ok` and `statement error` for statements, `query <typestring>` for results, where the type string is one character per column: `I` for integer, `R` for real, `T` for text. Results come after `----`, tab separated. Larger results are given as an MD5 hash of the values with a row count, which keeps the files small.

Directives that matter for us: `require <extension>` skips a file we do not have, `mode skip` and `mode unskip` bracket sections, `loop` and `endloop` parameterize, `sort` and `rowsort` control comparison, and `onlyif` and `skipif` gate by engine name. That last one is how DuckDB's own files handle divergence, and it is why running the corpus unmodified is possible at all.

## 2. The rules

**The files are run unmodified.** No local edits, no patched expectations, no per file skips added to make a number look better. The corpus is fetched at exactly the commit the grammar was vendored from, which document 13 question 10 settles and `tools/fetch_corpus.sh` implements, so it moves when the grammar moves and never on its own.

**Every failure is classified automatically into one of five buckets:** unsupported feature, meaning a document 05 refusal; missing function, meaning a document 07 tier 2 or tier 3 name; semantic divergence, meaning the right shape with a wrong value or type; crash; and wrong answer without an error. The last two are bugs at the top of the queue. The first two are the roadmap.

**The number published is per directory, with the denominator.** For example, `test/sql/aggregate: 412/431 (95.6%)`. A single blended percentage over the whole corpus is not information, because it is dominated by directories we have deliberately not implemented and it can be moved by adding files rather than by adding features.

**Nothing is rounded up, and the README is generated from the harness output** rather than edited by hand. This is the mechanism and not the aspiration, because a number a human types is a number that drifts.

## 3. The three harnesses

**The parse differential harness.** Every statement in the corpus through both parsers, comparing accept against reject. This is the hard gate from document 01, running from the first week the matcher exists, long before anything executes. It is `tests/differential/sql.mojo` and `pixi run differential-sql`. `tests/differential/sql_generated.mojo` and `pixi run differential-sql-generated` are the same comparison over statements the grammar wrote rather than statements people wrote, and section 4 covers why that is a different question and not the same one with more input.

`tools/corpus.py` flattens the `.test` files into 71,438 statements, taking the SQL out of every `statement` and `query` block and ignoring the expected output, because the annotation in a test file is not the oracle. A `statement error` is very often a binder error, which means the parser was perfectly happy with it. Statements inside a `mode skip` region are left out, because those are ones DuckDB has turned off itself, and so are the hundred or so that carry a `${name}` loop variable, because substituting one means running the loop.

The oracle is DuckDB's Python module, which the differential environment already has for the pandas comparisons, and the call is `extract_statements`. That splits a string into statements and builds nothing else, so a column that does not exist and a function with no overload are not errors on that side. That is exactly where the line has to be drawn, because firepanda refuses those in the binder and a binder refusal is not a compatibility failure. The whole corpus takes about two seconds on that side, so it runs live rather than from a recorded answer file, and a DuckDB upgrade shows up as a change in the numbers rather than as nothing at all.

**The two directions are not the same failure.** DuckDB parsing something firepanda rejects is a compatibility failure and the target for it is zero. firepanda parsing something DuckDB rejects usually is not, for two reasons that will not go away soon. The grammar is vendored from DuckDB's development branch and the oracle is the last release, so `CREATE TRIGGER` and `JOIN BY (TYPE semi)` and DML inside a CTE are syntax the oracle has never heard of. And replacing Bison with a PEG grammar moved a pile of errors out of the parser and into the binder upstream, so `SELECT * EXCLUDE (i, i)` and `SELECT GROUPING()` now parse there too and fail later. Both directions are counted and both have a ceiling in the harness, but only the first one has a target.

The first reading, at the commit the matcher landed on, was 2 and 1,130 out of 71,300 compared, which is 98.41 per cent agreement. The 2 are both the recursion depth guard in the matcher firing on input DuckDB's explicit stack handles, and they are the reason document 04 wants that stack machine.

**The execution harness.** Runs `.test` files against firepanda, compares values, classifies failures. It drives the CLI from document 10, so it is testing the whole stack the way a user gets it.

**The semantics harness.** One case per rule in document 06, comparing value and `typeof()` against DuckDB, plus the expression fuzzer and the overload resolution fuzzer from document 07. This one catches what the corpus does not, because the corpus was written to test DuckDB's features rather than to pin down its type lattice. The half of it that asks about types is `tests/differential/semantics.mojo` and `pixi run differential-semantics`, and section 5 is what it does and what it found. The half that asks about values is `tests/differential/answers.mojo` and `pixi run differential-answers`, and section 6 is the same for that one.

## 4. Fuzzing

firepanda already fuzzes the bitmap, the kernels, the hash and the join against scalar twins. `pixi run fuzz` runs four fuzzers, and the twins in `firepanda/kernel/scalar.mojo`, `hash/scalar.mojo` and `join/scalar.mojo` exist to be the specification. SQL adds four more, and the pattern is the same: an oracle that is obviously correct, and a fast thing checked against it.

**Grammar directed generation.** The vendored `.gram` files are a generator as well as a recognizer, so walk the rules, choose alternatives and emit tokens. This is the fuzzer that makes the parse agreement claim mean something beyond the corpus, and it exists only because the grammar is declarative, which is the same reason document 03 exists. It is `firepanda/sql/generator.mojo`, and `pixi run differential-sql-generated` is the harness that points it at DuckDB.

Three things make the walk terminate. Every node carries a cost, meaning the fewest tokens that finish it, computed once by fixpoint before any generation. A token budget counts down, and once it is spent every choice takes its cheapest alternative. And a depth cap stops the walk far below the matcher's own guard, because a string the generator wrote and the matcher then refused for running out of stack tells nobody anything. The budget is where the walk stops spending rather than where it stops writing, so a finished statement runs past it by whatever closing the open frames costs.

The plan said every string produced would be valid DuckDB SQL by construction. That is not quite true and the reason is worth writing down. Satisfying a negative lookahead in general means solving the problem the parser exists to solve, so the generator ignores `!X` and can write a statement the grammar itself would refuse. About four in ten of what comes out is refused somewhere, mostly there. That does not weaken the harness, because the assertion is not that the generator writes good SQL, it is that the two parsers agree about whatever it writes.

The two directions are read the same way they are in the corpus differential and the numbers are very different. DuckDB accepting something firepanda rejects has a ceiling of zero and means a bug. firepanda accepting something DuckDB rejects runs at about 55 per cent, because the generator samples the grammar rather than the language people write and so spends most of its time in rules the released oracle has never been asked about, where the corpus spends almost none. `DROP EXTENSION REPOSITORY` and `DISCONNECT` are both in the vendored development grammar and neither is in the oracle. That side is a rate ceiling, so it holds at any case count, and it exists to catch the number moving rather than to be driven down.

The first run found one thing, and it was in the grammar rather than in the matcher. `CopyFileName` lists a bare `Identifier` ahead of `Identifier '.' ColId`, and PEG choice is ordered, so the bare one always wins and the qualified alternative can never be reached. `COPY t TO a.b` is a statement DuckDB's own parser takes and DuckDB's own grammar cannot. The grammar is vendored byte for byte and stays that way, so the harness carries the case in a `known` list with the reason, which is what lets the ceiling above stay at zero and mean something. See document 13, question 12.

**Round trip.** Parse, print with the printer from document 05, reparse, compare ASTs. Catches precedence and associativity bugs that accept against reject cannot see.

**Query generation with a differential oracle.** SQLancer style: generate random queries over random small tables, run through DuckDB and firepanda, compare results. This is what finds wrong answers, and wrong answers are the failures that matter most because everything else is loud.

**Pathological input.** The cases from document 04, meaning deep nesting, long lists and unmatched parentheses, each with a wall clock ceiling in CI, because a PEG parser's failure mode is exponential and a front door that takes untrusted SQL turns that into a denial of service.

## 5. The type differential harness

The two parse harnesses ask whether a statement parses. This one asks the question after that, and it is the one the compatibility claim actually turns on: given columns of known types, what type does an expression over them come out as. A parser that agrees with DuckDB everywhere and a binder that makes `sum` a `BIGINT` where DuckDB makes it a `HUGEINT` is a library that returns wrong answers with no error attached to them, and `typeof()` is itself in the corpus, so the difference is not even hidden. It is `tests/differential/semantics.mojo` and `pixi run differential-semantics`.

The matrix is 27 columns, every scalar type the binder knows plus six decimals chosen to sit on the edges of DuckDB's width rules, including `DECIMAL(18,18)` and `DECIMAL(38,10)`. Four things are compared over it. Arithmetic, over every ordered pair and all six operators, which is the one place firepanda derives a type rather than looking one up. Negation, which is a short list and a different rule from subtraction. The lattice, through `CASE`, over every ordered pair, which is the type two branches of one expression agree on. And calls, over the tier 1 catalog at every arity up to two, which is the overload resolution fuzzer document 07 asks for. Both halves of a call are compared, which signature won and what it comes out as, the second being derived by `result.mojo` wherever the catalog writes a rule in place of a type. That comes to 15,001 expressions and takes about four seconds.

The two ways a probe can agree are counted separately, because they are not the same claim. 14,630 of the expressions have their type compared and that is the claim the harness is for. 12 of them, all of them `histogram`, are calls returning a `MAP`, which needs a key type and a value type where `SqlType` carries one element, so what is compared is which overload won and not what it returns. Folding the two into one figure would give a number that improves when the binder is taught less, so the report prints both and names what is in the second. Issue #780 is what is left; it started at 932 across 29 names, and the template letters, the bare decimals, the `sum`, `avg` and `median` rules and the list elements have since been derived.

The oracle is `tools/semantics.py`, which builds a table with one column per type and asks DuckDB for `typeof` of each expression over it. Both halves of that are needed. A column rather than a literal, because DuckDB folds an expression whose arguments are all constants before anything can be read off it, and one row rather than none, because `typeof` over an empty table returns no rows to read. The script runs DuckDB in a child process for the same reason `tools/corpus.py` does: importing the module inside the CPython embedded in a Mojo binary registers exit handlers that run after that interpreter has been torn down, and the process then dies in a destructor with the report already printed.

It earned its keep on the first run, at 348 disagreements over five separate defects, none of which the unit tests or the generators had caught. DuckDB narrows an addition or a multiplication back to 18 digits when both sides already fit in 18, which is where `DECIMAL(9,4) * DECIMAL(9,4)` stops being `DECIMAL(18,8)` and stays there, except that a multiplication whose scale reaches 18 is left alone. The lattice gives up scale rather than digits when two decimals do not fit in 38, and does not do that when one side is an integer. `greatest` and `least` are declared over `ANY` and then insist their arguments share a common type, which no signature can say. And two of them were in the cast cost generator rather than in the binder: `EXPLAIN` wraps its output to the box width and had been splitting long casts across lines where the generator read them, which left the measured cost of reaching a `DECIMAL` wrong relative to reaching a `DOUBLE`. The harness now runs at zero disagreements over 15,001 expressions, with 359 cases in a `known` list, each carrying the reason it is there.

## 6. The value differential harness

The question after the type question, and the one nothing above it can reach. `strlen` shipped counting characters where DuckDB counts bytes, and everything agreed with it: the type was `BIGINT` on both sides, the plan printed the way it was meant to, the parse harnesses never look at a result, the type harness saw two `BIGINT`s and moved on, and the unit tests asserted the wrong number because they had been written from the same misreading. A kernel that returns the right type and the wrong value is invisible to every other harness here, and this is the one that sees it. It is `tests/differential/answers.mojo` and `pixi run differential-answers`.

The probe is one table of eight rows, described once as SQL literal text and built twice from that description, so the two sides cannot drift the way two fixtures maintained separately do. firepanda parses the literals into a frame and DuckDB inserts them. The rows are chosen so that every reading a kernel can get wrong is in the table: a row of ASCII, a row that is not ASCII, a row whose character count and byte count differ, an empty row, a row with spaces on its ends, a null in every column, a negative number, a zero, a date before the epoch and a leap day. A row number goes in front of the columns so that both sides order by the same thing and the answers line up.

Every answer is compared as text, because text is the one rendering both sides can be asked for without either of them having an opinion about formatting, and a whole number and a boolean render identically either way. Floating point does not, so nothing in the list answers one, and the decimals and the timestamps are the obvious next thing to add rather than something the harness pretends to cover. A null is a separate character rather than an empty field, because an empty string is a value an expression can really answer and the two have to stay apart.

The expressions are written by hand rather than generated. What is worth asking is the surface firepanda actually runs, and a generator over the tier 1 catalog would spend the whole run on names nothing is wired to. The order is the order the functions were wired in, so a diff against a later version of the list reads as the list of what was added.

Three things can happen to an expression. Both sides answer and the answers agree, which is the point. Both sides refuse, which is agreement of a weaker kind and is counted rather than failed. Or they disagree, and there are two ways to do that. The same rows with different values in them is the defect the harness exists to catch and its ceiling is zero. One side answering where the other refuses is the gap list in executable form, and its ceiling is zero too, because the list was written to the surface firepanda covers and a new entry that refuses is either a gap worth recording on the stage issue or an expression that does not belong in the list yet.

A ceiling of zero only means anything if the disagreements that are allowed to stand carry the reason they are, which is what the recorded list in the harness is for. An expression on it is either a decision, meaning firepanda answers differently on purpose and the reason says why, or a gap with an issue number against it. Either way the expression stays in the list and its two answers are printed every run with the reason underneath, rather than being deleted from the list, because an expression quietly dropped is a comparison nobody makes again. The same shape as the type harness's list, for the same reason.

The oracle is `tools/answers.py`, built the same way `tools/semantics.py` is and running DuckDB in a child process for the same reason.

It has paid for itself three times. Its first run found `n // 3` and `n % 3` answering Python's rounding where DuckDB uses C's, which is issue #770, and `s LIKE '_b_'` refused where DuckDB answered, which is issue #776. Both are fixed and both took the shape this section predicts: the expressions sat on the recorded list with an issue number against them until the kernels existed, and came off it when they did.

The third arrived with the double literals. `CAST(2.6 AS BIGINT)` was 3 in DuckDB and 2 in firepanda, because DuckDB rounds to the nearest and firepanda truncated the way the machine instruction does. That one is issue #786 and it took longer to come off the list than the other two, because the fix was a decision rather than a kernel: `astype` on the dataframe side has to keep truncating, since that is what pandas does, so the two front ends want different answers out of one conversion. Where the difference ended up is a flag on the plan's cast, set by the SQL lowering whenever the target is an integer and left off everywhere else, and the reason it is on the plan rather than in the kernel is that the plan is the one place that knows which front end asked.

The recorded list is empty as of that fix and the ceiling of zero is doing the work on its own. The list stays in the harness for the next disagreement that is a decision rather than a patch.

## 7. The query differential harness

Everything above asks about one expression at a time, which is the right size for a kernel and the wrong size for a query. A query has a plan in it, and a plan has a join order, a group, a sort and a limit in it, and every one of those is a place an answer can go wrong in a way no expression comparison reaches. A join that drops a row, a group that keys on the wrong column, a sort that is not stable where the query needed it to be: all four harnesses above pass while any of those is true. This one asks whole TPC-H queries. It is `tests/differential/tpch.mojo` and `pixi run tpch`.

The data is DuckDB's own `tpch` generator, exported to Parquet, and both engines then read the same files. Reading the same bytes is the whole point. Two generators seeded the same way is a claim about two programs, and a claim about two programs is what a differential harness exists to stop making. The queries come from DuckDB's `tpch_queries()` for the same reason, written out beside the data rather than checked in, so there is no second copy of the query text to drift from the first.

One thing here is not the reference data and has to be said plainly. TPC-H declares eight columns as `DECIMAL(15,2)` and firepanda has no decimal type, so those columns are cast to `DOUBLE` on the way out. DuckDB is then asked the same question over the same doubles, which keeps the comparison exact, but it is a comparison against DuckDB over this data rather than against the published answer set: a sum of doubles is not a sum of decimals and the last digits differ. The published answers come back when there is a decimal type. Within that, a double is compared with a relative tolerance and everything else exactly, because neither engine adds a column of doubles in the same order as the other and an exact comparison would be a test of the summation order rather than of the answer.

A query firepanda refuses is reported rather than failed when the refusal has a reason written down beside it, which is the same recorded list the two harnesses above keep and for the same reason. A refusal with nothing written against it fails, so a query that stops running is noticed the run after it stops. A query that answers where a refusal was recorded fails too, because a stale record is how a gap that has closed goes on looking open.

The scale factor is `FIREPANDA_TPCH_SCALE` and defaults to 0.01, which is about sixty thousand lineitem rows and three megabytes and takes a few seconds, so it is cheap enough to run on a push. The exit criterion for stage S4 is written at scale 1, which is a hundred times that in both, and is a number to run by hand or nightly.

All twenty two are asked. Twenty one agree with DuckDB row for row today, which is every one of them but q21. q21 correlates an `EXISTS` through an inequality as well as an equality, which is the dependent join a decorrelation pass removes rather than one the rewrite can, and issue #816 has it written out with the refusal it comes back with.

## 8. The plan equality test

The single most valuable test in this specification, and it is not about compatibility at all.

For each of the twenty two TPC-H queries, the physical plan produced by `fp.sql(q)` and the physical plan produced by the equivalent dataframe chain must be structurally equal.

One expression is allowed to print differently on the two sides and it is the cast of a float to an integer, because the two front ends mean different conversions by it and the plan says which. A dataframe chain that wants the SQL answer asks the plan builder for it, and the equality test compares the plan the chain built rather than the spelling a caller reached for, so this is a difference the test can still see rather than one it has to ignore.

It costs almost nothing, because the plans already print and round trip per document 08, and it is what enforces document 02's rule that SQL and dataframes are one engine. Without it, the SQL path grows its own lowering for one operator, then another, and a year later there are two engines with different bugs and different performance, which is precisely the outcome that issue #13's line about parsing into the same logical plan so the optimizer is shared exists to prevent.

## 9. The optimizer equivalence test

From document 08, restated because it belongs to conformance as much as to the optimizer: every query in the corpus runs twice, once with all optimizer passes disabled and once with all enabled, and the results must be identical including order.

This is the test that catches a filter pushed through a node that does not preserve its meaning, a join reordered across an outer join that is not reorderable, or a decorrelation that changed null semantics. Those bugs produce plausible wrong answers on real queries and are nearly impossible to find any other way.

## 10. What runs when

**Every commit:** the parse differential over the full corpus, the semantics cases, the value cases, the query comparison at scale 0.01, unit tests, and the pathological input ceilings. Minutes.

**Every commit:** the execution harness over the target directories with a pass rate floor that ratchets, so the number may go up and a commit that lowers it fails. Ratcheting rather than a fixed threshold is what stops the number quietly sliding while everyone is busy.

**Nightly:** the full 4,796 files with the classification report, all four fuzzers with a time budget, optimizer equivalence, plan equality, the query comparison at scale 1, and the benchmark suite with peak memory.

**Weekly:** the grammar bump check from document 03 against the latest upstream tag.

## 11. What the corpus does not cover

Stated so that the published number is read correctly.

It does not cover performance, because a query that takes an hour passes.

It does not cover memory, because a query that uses 40 GB passes. That is document 09.

It does not cover the Python boundary, meaning capture rules, parameter binding, GIL behaviour and exception types. Those are firepanda's own tests, per document 10.

And it does not cover concurrency. `pixi run stress` is firepanda's substitute for a race detector, and SQL adds cases to it: the same query from several threads, registration racing execution, and cache invalidation under concurrent registration. The prepared statement cache in document 10 is shared mutable state on the hot path, which makes it the most likely place in this whole specification for a race, and it is the one that a correctness test will never find.
