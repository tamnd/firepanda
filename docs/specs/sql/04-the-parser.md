# The parser

The two components between a query string and a `ParseResult` tree. Together they are the whole of what one hundred per cent DuckDB syntax compatible means operationally, and one of them is generated from a vendored file while the other is hand written and is therefore where the bugs will be.

## 1. The budget

Measured on this machine, DuckDB 1.5.5, Apple M4, by parsing unique statements in a loop and dividing:

| | DuckDB |
| --- | --- |
| TPC-H q1, parse only | 156 us |
| trivial `SELECT 1`, parse only | 6 us |

Those numbers include DuckDB's JSON serialization of the parse tree, so the true parse cost is lower. They are an upper bound on what we must beat, and they were measured again after the first attempt returned an impossible 0.8 us because DuckDB constant folded the identical input out of the loop.

Our targets, which document 01's latency axis depends on:

| | firepanda target | measured |
| --- | --- | --- |
| tokenize, TPC-H q1 | | 3.2 us |
| tokenize and match, TPC-H q1 | | 181 us |
| tokenize, match and transform, TPC-H q1 | under 60 us | 449 us |
| tokenize, `SELECT 1` | | 101 ns |
| tokenize and match, `SELECT 1` | | 9.4 us |
| tokenize, match and transform, `SELECT 1` | under 3 us | 13.9 us |
| allocations for a small statement | one arena block, no per node malloc | met |
| memoization table | reused across statements, not reallocated | not allocated at all unless a memoized rule finishes |

The allocation line is the one that matters. A REPL loop over small statements spends its time in the allocator, not the matcher, and this is the axis where a library beats a database.

Those are the `sql/` rows of `benchmarks/main.mojo`, ten repetitions, the median, on Apple M4, Mojo 1.0.0 (ed45d567), firepanda 0.6.53, with the grammar and the jump table built once outside the timed section because that is what a process does. Interquartile range was under five per cent on every row. The machine had other work on it, so these read high rather than low, and they are here because a number anybody can reproduce with `pixi run bench -- --filter=sql/` is worth more than a better one nobody can.

The split is the useful part. Tokenizing is under one per cent of a parse either way, so the tokenizer is done. The matcher is 40 per cent of q1 and 68 per cent of `SELECT 1`, and the transformer is the rest. The last time this table was written its measured column had no transformer in it, because there was no transformer, so it was comparing two thirds of the work against a budget for all of it. Against the whole of it, q1 is seven and a half times over and `SELECT 1` is four and a half times over, and the work to close that is on both sides rather than only in the matcher.

Two changes have already been made to the matcher and both are still worth what they were: the first token filter in section 4, worth about three times on its own, and memoizing successes as well as failures in section 5, worth about one and a half times on top of it. Both were measured back to back in one binary against the matcher as it was before them, so they are speedups against a build that no longer exists and the absolute readings that went with them are not comparable to the table above.

Where the time went when the matcher landed: not the tokenizer, and not the allocator, which is five per cent of a parse. It was 27,814 node visits and 10,504 rule entries to parse one hundred tokens, which is the interpreter doing an honest amount of work an honest number of times. Closing the gap means visiting fewer nodes rather than shaving the visit, so what is left after sections 4 and 5 is the recursion itself, and turning the matcher into an explicit stack machine is the remaining lever on that side.

## 2. The tokenizer

Hand written, matching DuckDB's `src/parser/peg/` tokenizer by behaviour rather than by code, because there is no declarative artifact for it. One pass, no backtracking beyond a two byte peek, producing a flat `List[Token]` of `(kind, flags, keyword_id, byte_start, byte_len)`, twelve bytes each.

Nothing is decoded. A string keeps its quotes and its escapes, a number keeps its underscores, an identifier keeps its case. Decoding is the transformer's job in document 05, and leaving it there is what keeps a token at twelve bytes and the whole vector in cache. The flags carry the handful of facts that are cheap to record while the bytes are under the cursor and expensive to work out again later, which is the literal's type, which quoting form a string used, and which parameter form a parameter was.

Kinds: identifier, quoted identifier, keyword, number, string, operator, punctuation, parameter, end.

The details that have to be right, each of which is a silent compatibility bug if it is not:

**Identifiers are case insensitive and fold to lowercase, and quoted identifiers do not fold.** `SELECT A` and `select a` are the same column and `"A"` is a different one. DuckDB folds down, unlike standard SQL's fold up, which matters for error messages and for `information_schema` style output.

**Keyword classification comes from the vendored `.list` files**, five classes, resolved by bisecting one sorted table on the folded text. The longest keyword is fifteen bytes, so the fold happens in a fixed stack buffer and a longer word skips the lookup entirely. A word in the reserved list cannot be a bare identifier, and a word in the unreserved list can be, in the positions the grammar allows. Getting a word into the wrong class produces exactly the failure mode we promised never to have, which is a syntax error on valid DuckDB SQL.

**String literals** are single quoted with `''` as the escape. Dollar quoting, `$tag$ ... $tag$`, has no escapes at all and is how macro bodies and regexes are written.

Four letters are string prefixes, `E`, `X`, `B` and `N` in either case, and only when the quote is the very next byte, so `SELECT e 'a'` is an identifier followed by a string. Only `E` changes how the body is read, where a backslash swallows whatever byte comes after it, including a quote. The other three are ordinary strings whose prefix the transformer reads back off the token text, which is what DuckDB does with them too. There is no `U&'...'` form, contrary to what an earlier draft of this document said: `SELECT U&'a'` is the identifier `U`, the operator `&` and the string `'a'`, and it fails on the operator rather than on the string.

And two literals separated by whitespace containing a newline are one literal. `SELECT 'a' 'b'` is a syntax error, `SELECT 'a'` then a newline then `'b'` is `'ab'`, a line comment in between keeps the join and a block comment breaks it. That is Postgres's rule and DuckDB kept it, and it is the sort of thing nobody writes down until a corpus file uses it.

**Numeric literals decide types**, and this is the tokenizer reaching into document 06: an unsuffixed literal with a decimal point is DECIMAL, not DOUBLE, which is why `1.1 + 2.2` is `DECIMAL(3,1)` and exactly `3.3`. An exponent overrides that and makes it DOUBLE. `1.` and `.5` are both valid and both DECIMAL.

Underscore digit separators are accepted, but only between two digits: `1_000` is a thousand while `SELECT 1_` is `1` aliased `_` and `SELECT 1__0` is `1` aliased `__0`. The same shape of rule governs the exponent, because `SELECT 1e` is `1` aliased `e`, so an `e` with no digits after it has to be handed back rather than reported as an error.

There are no hex or binary literals, contrary to what an earlier draft of this document said. `SELECT 0x1F` returns 0 in a column named `x1F`, which is DuckDB reading the number `0` and then the identifier `x1F`, and `0b101` and `0o17` behave the same way.

**Comments** are `--` to end of line and `/* */` with nesting.

**Parameters** are `?` anonymous, `?1` and `$1` numbered, and `$name` named. There is no `:name` form, contrary to another line in an earlier draft: `SELECT :name` is a syntax error and `expression.gram` lists exactly the four rules above. `$` is also dollar quoting, and the disambiguation is that a dollar quote tag is a word that does not start with a digit. `SELECT $1$a$1$` settles it, because DuckDB fails there with `unterminated dollar-quoted string`, which only happens if `$1` was a parameter and the quote started at `$a$`.

The marker is a token and the number or the name after it is another one, because all four grammar rules are two nodes and a matcher that was handed `$1` whole would have no node to spend it on. The cost of that is one divergence in a corner: DuckDB rejects `SELECT $ 1` and `SELECT ? 1` and accepts `SELECT $ name`, and we accept all three. Accepting a string DuckDB rejects is the cheap direction of that trade, and the differential harness reports it rather than hiding it.

**Trailing commas** are legal in most list positions. This is grammar, not tokenizer, but it is the single most used Friendly SQL nicety and it belongs on the same checklist.

**Operators** are a maximal run of operator characters, with three rules on top that came out of `src/parser/peg/tokenizer/base_tokenizer.cpp` rather than out of Postgres. This is the one place where copying Postgres's answer is wrong, and it produced two real bugs before the corpus found them.

A dozen characters are their own token always and never join a run: `( ) { } [ ] , ? $ - #`. So `SELECT #1+#2` is five tokens and not three, and `SELECT 1 =- 1` is `1 = -1` because the minus left on its own. `?` is out because DuckDB spends it on parameters, and `-` and `#` are out because the tokenizer says so.

Six sequences are checked before that, `->>`, `::`, `:=`, `->`, `**` and `//`, which is how `->` survives the rule that a minus never joins anything. If the byte after one of them is itself an operator character then it was a run after all, so `a ->>= b` asks for an operator named `->>=` rather than for a JSON extract and a comparison.

And a run of more than one character gives back a trailing `+` unless the run also contains one of ``~ ! @ # % ^ & | ` ?``. `SELECT 1 =+ 1` is `1 = +1` and `SELECT 1 !=+ 1` goes looking for an operator named `!=+`. Only `+` is given back, because the minus never got into the run to begin with.

Every one of these came from running the query against DuckDB, or from reading its tokenizer where there was no query that could tell the difference. There is nothing declarative to read: `NumberLiteral`, `StringLiteral`, `Identifier` and `OperatorLiteral` are all in the rule override block, which is DuckDB stating that its own matcher ignores their bodies. Each behaviour above has a test in `tests/test_sql_token.mojo` naming the query it came from.

The tokenizer gets a dedicated differential fuzzer from week one: random bytes and structured random SQL through both tokenizers, comparing the token stream. It is a few hundred lines of hand written state machine standing between us and the compatibility claim, and it is the cheapest place in the whole project to buy confidence.

## 3. The matcher

A recursive descent interpreter over the rule table from document 03. Input is the token vector and a start rule. Output is a `ParseResult` tree, or a failure with a position.

```
struct ParseNode:
    var rule: UInt16        # index into the generated rule table
    var token_start: UInt32
    var token_end: UInt32
    var first_child: UInt32 # index into the node arena, 0 means none
    var next_sibling: UInt32
```

Twenty bytes, arena allocated, index linked. No owning pointers, for the same reason document 02 gave for the AST: an ownership tree in Mojo is a fight with no payoff.

Twenty four rules are the exception, and they are where the tokenizer and the matcher meet. Their bodies in the grammar text are placeholders, `OperatorLiteral <- Identifier` being the one that shows why it matters, and upstream matches them from code instead. The list is the generated rule override block in `compiled_grammar.cpp`, it is vendored the same way the memoized list is, and the generated table carries it as a short section naming the rule, its matcher and the suggestion it was built with. When the matcher reaches one it asks the token at the current position whether it is that kind of token, in constant time, and never looks at the body.

Twenty one of the twenty four are identifier rules and they split twelve to nine. The twelve run `IdentifierMatcher`, which rejects a word that is a keyword unless the keyword is unreserved or is in the one class the position tolerates. The nine run `ReservedIdentifierMatcher`, which drops that check entirely, and that is what makes `db.select` legal after the dot while a bare `select` is not a column name. Which class a position tolerates comes from the suggestion, not from the matcher: a type name takes a type name keyword, the two function name rules take a type or function name keyword, and everything else takes a column name keyword. So this is also the whole of how the grammar's five keyword classes turn into a decision about whether a word can be a name in this position.

Both identifier matchers accept a double quoted identifier and reject a token that starts like a number. Both accept a single quoted string only where the suggestion is a table name, which is how `FROM 'data.parquet'` parses without a rule for it. Neither one looks past the first byte of the token otherwise, so `E'x'` and `U&"x"` are identifiers to the matcher and fail later in the transformer, which is what DuckDB does and is worth knowing before reading a confusing error.

The matcher is generic over the rule kinds and knows nothing about SQL. Sequence matches children in order and fails as a unit. Ordered choice tries alternatives left to right, resetting the token position on each failure, and takes the first success, with no longest match, no ambiguity and no conflict. Repetition is greedy with no backtracking into it, which is standard PEG and is a real semantic difference from a regex or a context free grammar that the grammar is written to expect. Lookahead matches without consuming.

Because it recurses, it needs a depth cap that DuckDB does not, and the cap is five hundred rule frames. A statement costs forty frames before any nesting and then twenty one per nested parenthesis, twenty one per nested `CASE` and sixteen per nested subquery, so five hundred is about twenty two parentheses or twenty eight subqueries. An unoptimized build runs out of native stack at around eight hundred and twenty frames, which is where the number came from. Past the cap it raises DuckDB's own `memory exhausted at or near`, so the error text agrees even though the limit does not: DuckDB takes five thousand parentheses, because its matcher keeps an explicit `MatcherStack` rather than recursing. Turning this into a stack machine is the same change the budget above wants, which is why the two are one piece of work and not two.

The matcher never reaches a character level node, and that is checked rather than assumed. Character classes and captures appear only inside `%whitespace`, `NumberLiteral`, `StringLiteral`, `PlainIdentifier` and `QuotedIdentifier`. The first is never referenced, because whitespace is applied between tokens rather than called; the next two are overridden; and the last two are reachable only through `Identifier`, which is overridden as well. So reaching one means the grammar grew a shape the tokenizer does not cover, and the matcher stops loudly rather than guessing.

## 4. The first token filter

The ordered choices in this grammar are long, fifty alternatives is ordinary, and the token in hand rules out nearly all of them. Without help the matcher finds that out one recursion at a time, which is what the budget above is spent on.

So every node in the generated table carries a sixty four bit word saying which tokens it can start with, and the matcher tests it against the token in hand before it walks the node. A node whose word does not have the token's bit cannot match, and fails without recursing.

The word is a set of token keys, one bit each, at bit `key % 64`. A keyword is its own key, which is its index in the keyword table, so telling `SELECT` from `INSERT` is one bit out of sixty four. Identifiers, quoted identifiers, numbers, strings and end of input get a key each. Punctuation, operators and parameters are keyed on the byte they start with, which loses nothing, because a literal made of punctuation has to match every byte of itself anyway. Sixty four bits over seven hundred odd keys makes this a Bloom filter rather than an exact set, so it can say yes where it should have said no, and that costs one recursion that fails.

The words are FIRST sets over the rule graph, computed at generation time to a fixed point because the graph has cycles in it. Three rules govern the computation and all three exist to keep the filter loose rather than tight:

- A node that can match the empty string gets every bit. It can succeed without reading a token at all, so filtering it on the token would be wrong. This is a third of the nodes.
- A negative lookahead never consumes, so it contributes nothing to what the node around it can start with. That is what lets `!X Y` filter on `Y` rather than on the union of the two.
- The twenty four code matched rules get their words read off the matcher in section 3, not off their bodies, for the same reason the matcher does not walk those bodies.

Anything else unclear resolves to every bit. A word that is too generous costs time and a word that is too tight rejects a valid query, so the two failures are not symmetric and the computation is written to fail in one direction.

2,972 of the 4,422 nodes filter, and a node that filters has 1.8 of its sixty four bits set on average. The effect on the numbers in section 1 is that TPC-H q1 goes from 27,814 node visits and 10,504 rule entries to 9,918 and 3,197, of which the filter throws 2,870 out before they cost anything, and `SELECT 1` goes from 1,436 visits and 541 rule entries to 451 and 140.

The words are stored as a palette. There are 237 distinct words across the 4,422 nodes, so each node line in the generated table names one by index, which keeps a sixty four bit column from being written out four thousand times.

Correctness is by construction and then by test, because construction is not enough here: the generator computes the words and the matcher computes the keys, they are two pieces of code in two languages, and a disagreement between them shows up as a valid query that no longer parses rather than as a crash. So `parse_unfiltered` runs the same matcher with the check compiled out, and the whole corpus goes through both and has to come back with the same tree node for node and the same error text word for word.

## 5. Memoization

Full packrat memoizes every rule at every position, gets linear time, and pays for it with a table proportional to rules times positions, so 1,087 times the token count, plus a lookup on every rule entry. For SQL, where most rules match or fail immediately, that overhead exceeds what it saves on almost every real query.

DuckDB measured the pathology. A query with nineteen unmatched parentheses took 10.640 seconds unmemoized and 0.001 seconds memoized. That is exponential backtracking on the expression rules, which is where the deep ordered choices are.

Their answer, and ours, is a short explicit list of memoized rules. `packrat_memoized_rules` in the upstream source names them. We vendor the list alongside the grammar and treat any change to it as a grammar change. The rules on it are the expression precedence chain and a handful of the deepest statement level choices, which are the places where the same position is genuinely retried by many alternatives.

Implementation is denser than the open addressed table this document first described. The generated table says which rules are memoized, so each one gets a slot number and the memo table is one four byte word per slot per token position, which is eighty eight bytes a token. Zero means the rule has not been tried there, one means it failed there, and anything else is the arena index of the node it produced, plus one so that a real node can never read as either of the other two. The table is allocated on the first memoized rule to finish and dropped with the parse, so a `SELECT 1` never touches it and there is nothing to keep or clear between statements.

Both outcomes are recorded, and they are worth recording for two different reasons. A PEG rule is a pure function of the grammar and the position, so whatever it did once at a position it does every time.

The failures are DuckDB's pathology, which is nineteen unmatched parentheses and is a query nobody meant to write.

The successes are `SELECT f(f(f(1)))`, which is a query somebody did. `TypeModifiers <- Parens(List(Expression)?)` means `f(x)` parses as a parameterized type before it parses as a call, so `SingleExpression` reaches `TypeLiteral` first, walks the whole argument as a type, fails on the string literal that a type literal needs, and then reaches `FunctionExpression` and walks the same bytes again as an expression. Two full walks per level of nesting is a factor of two per level, and twelve levels took a fifth of a second, unoptimized, against under a millisecond now. Failure memoization cannot see any of it, because both walks succeed.

That is what the first draft of this section got wrong. It said a success cannot be memoized because a memoized success is a subtree in the node arena and a failing ancestor may have truncated that subtree away since. The premise was right and the conclusion was not: the fix is for the arena to stop truncating. A failed attempt now unwinds only the pending stack, and the nodes it built stay where they are, unreachable from the tree but reachable from the memo table. The cost is an arena about half again the size of the tree it holds, which for TPC-H q1 is 1,376 nodes against a tree of 838.

Nothing is written to the table from inside a negative lookahead. A terminal that fails in there failed on purpose and does not move the furthest position, so an entry written from in there would hand that silence to a later walk that meant it, and the error would point earlier than it should. Reading from in there is fine, because the answer is the answer either way.

One thing a memo hit must not do is hand back the node itself. A node's `next_sibling` is written by whichever parent adopts it, and the same subtree can be adopted more than once, so a hit copies the root and shares everything under it. The copy is one node and it is the only node in the subtree that a new parent ever writes to.

The pathological input tests are in the suite from the first week, generated rather than collected, in `tests/test_sql_pathological.mojo`: N unmatched parens, N nested calls, N nested list literals, N nested `CASE`, N deep parenthesized expressions, N nested subqueries, long `IN` lists, long operator chains, wide select lists, long qualified names and long scripts. Each has a wall clock ceiling, and the shapes that could go exponential are also asked at n and at 2n, because an absolute ceiling on a shared runner is a blunt instrument while a shape that takes sixty four times as long for twice the size is exponential however slow the machine was. A PEG parser's failure mode is exponential blowup on adversarial input, it is a denial of service if any front door takes untrusted SQL, and the only defence that works is a test that fails loudly. This suite earned its place immediately, because the nested call blowup is the first thing it found.

## 6. Errors

PEG error reporting is genuinely bad by default. The failure surfaces at the top level choice, having discarded everything it learned, and the naive message is syntax error at position 0.

The standard fix, and DuckDB's, is to track the furthest position reached across all attempts, together with the set of terminals that were expected there. That position is almost always where a human would point. The furthest position is one field updated on every terminal failure, which is cheap and allocates nothing on the success path, and it is enough on its own to produce DuckDB's message. The expected set is not carried at all. A run takes a compile time flag that says whether to collect it, every ordinary parse runs with that flag off, and the set is built by a second run that only starts once a parse is already known to have failed. The second run also has the first token filter off, because the filter works by refusing to walk a node that cannot match the token in hand, so a filtered run never reaches the terminals that would have said what they wanted. That is the right way round: an error is on its way to a person who is about to read it, and a success is on its way to a hot loop. One thing that is not optional is the quiet counter, because a terminal that fails inside a negative lookahead failed on purpose, and letting it move the furthest position makes the error name whatever the grammar was checking was absent.

```
Parser Error: syntax error at or near "form"

LINE 1: SELECT * form t
                 ^
```

matching DuckDB's shape, blank line and all, because document 11's corpus matches error text by substring and because the shape is good. When the furthest failure is at the end of the token vector there is no token to name and no caret to draw, and DuckDB says `Parser Error: syntax error at end of input` on one line with nothing after it, which is what `SELECT 1 FROM` gives.

Keyword typo suggestions are built on top of that set, and they are the one place in this document where firepanda says more than DuckDB does. DuckDB's parser error stops at the caret. The candidate machinery upstream feeds autocomplete and binder errors and is never reached from the parser, so `SELCT 1` gets `syntax error at or near "SELCT"` there and nothing more. Here the first line and the caret block are still DuckDB's byte for byte, and the suggestion goes on its own line between them, which is where DuckDB puts candidate bindings on a binder error, so an error with one still reads like an error from the same program.

```
Parser Error: syntax error at or near "a"
Did you mean "WHERE"?

LINE 1: SELECT * FROM t WEHRE a = 1
                              ^
```

A word in the expected set one edit from what the query wrote is almost certainly what was meant, where one edit is a substitution, an insertion, a deletion or a swap of two neighbours. The swap is in there because it is the typo people actually make on a keyboard, and WEHRE for WHERE is two edits to anything that counts them the plain way.

The hard part is not the distance, it is that a misspelled keyword usually does not fail at its own token. `SELCT 1` reads SELCT as an identifier and dies at the `1`, and `SELECT * FROM t WEHRE a = 1` reads WEHRE as an alias for t and dies at the `a`. So there are three places to look, in order, and the first one that has an answer wins. The word being blamed, against the set collected there, which is what catches `GROUP BYY a`. The word before it, against the set collected at that position, which is what catches `SELCT 1`, because a statement whose first keyword is misspelled never gets a second alternative tried at token 0. And when that set has nothing either, one more parse over the same tokens with the suspect word replaced by a token no rule can match, which forces the parse to stop exactly there and to collect everything the grammar would have accepted in its place. That last one is what catches WEHRE, and it exists because PEG makes the alternative permanent: an optional that matched is never given back, so no amount of backtracking will go and ask what else could have stood where the alias went. Three parses of a query that has already failed, and only when the first two had nothing to say.

The guards matter as much as the search. A word shorter than three bytes gets no suggestion, because every two letter word is one edit from a dozen keywords and means none of them. More than three candidates gets no suggestion, because that is a list rather than a hint. A keyword the query spelled right is not a candidate for itself. And the suggestions come out in the order the grammar tried them, which is the order the alternatives are written in, which is upstream's own opinion about what is likely.

One refinement is still worth its cost and is not built: the rule stack at the furthest position, behind a debug flag, because when a grammar bump breaks something this is the only tool that finds it quickly.

## 7. What the matcher must not do

**No semantic decisions.** Not whether an identifier is a table or a column, not whether a function exists, not whether a cast is valid. The matcher's only output is shape. Every temptation to sneak semantics in here, and the classic one is resolving whether `foo(x)` is a function call or a type constructor, makes the parser impossible to regenerate and breaks the property that document 03 exists to buy.

**No lookups outside the token vector and the rule table.** No catalog, no settings, no session state. A statement's parse must be a pure function of its text and the vendored grammar, which is what makes the prepared statement cache in document 10 sound: cache on the text, because nothing else was an input.

**No error recovery.** The first failure stops the parse. Recovery is for editors and it changes which strings are accepted, which would put a hole in the compatibility claim. If firepanda ever wants a language server, that is a second entry point over the same rule table, and its behaviour is explicitly not what `sql()` does.
