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

| | firepanda target |
| --- | --- |
| tokenize, match and transform, TPC-H q1 | under 60 us |
| tokenize, match and transform, `SELECT 1` | under 3 us |
| allocations for a small statement | one arena block, no per node malloc |
| memoization table | reused across statements, not reallocated |

The allocation line is the one that matters. A REPL loop over small statements spends its time in the allocator, not the matcher, and this is the axis where a library beats a database.

## 2. The tokenizer

Hand written, matching DuckDB's `src/parser/peg/` tokenizer by behaviour rather than by code, because there is no declarative artifact for it. One pass, no backtracking beyond a two byte peek, producing a flat `List[Token]` of `(kind, flags, keyword_id, byte_start, byte_len)`, twelve bytes each.

Nothing is decoded. A string keeps its quotes and its escapes, a number keeps its underscores, an identifier keeps its case. Decoding is the transformer's job in document 05, and leaving it there is what keeps a token at twelve bytes and the whole vector in cache. The flags carry the handful of facts that are cheap to record while the bytes are under the cursor and expensive to work out again later, which is the literal's type, which quoting form a string used, and which parameter form a parameter was.

Kinds: identifier, quoted identifier, keyword, number, string, operator, punctuation, parameter, end.

The details that have to be right, each of which is a silent compatibility bug if it is not:

**Identifiers are case insensitive and fold to lowercase, and quoted identifiers do not fold.** `SELECT A` and `select a` are the same column and `"A"` is a different one. DuckDB folds down, unlike standard SQL's fold up, which matters for error messages and for `information_schema` style output.

**Keyword classification comes from the vendored `.list` files**, five classes, resolved by bisecting one sorted table on the folded text. The longest keyword is fifteen bytes, so the fold happens in a fixed stack buffer and a longer word skips the lookup entirely. A word in the reserved list cannot be a bare identifier, and a word in the unreserved list can be, in the positions the grammar allows. Getting a word into the wrong class produces exactly the failure mode we promised never to have, which is a syntax error on valid DuckDB SQL.

**String literals** are single quoted with `''` as the escape. `E'...'` handles backslash escapes. Dollar quoting, `$tag$ ... $tag$`, has no escapes at all and is how macro bodies and regexes are written. `U&'...'` takes unicode escapes. All four forms appear in the corpus.

And two literals separated by whitespace containing a newline are one literal. `SELECT 'a' 'b'` is a syntax error, `SELECT 'a'` then a newline then `'b'` is `'ab'`, a line comment in between keeps the join and a block comment breaks it. That is Postgres's rule and DuckDB kept it, and it is the sort of thing nobody writes down until a corpus file uses it.

**Numeric literals decide types**, and this is the tokenizer reaching into document 06: an unsuffixed literal with a decimal point is DECIMAL, not DOUBLE, which is why `1.1 + 2.2` is `DECIMAL(3,1)` and exactly `3.3`. An exponent overrides that and makes it DOUBLE. `1.` and `.5` are both valid and both DECIMAL.

Underscore digit separators are accepted, but only between two digits: `1_000` is a thousand while `SELECT 1_` is `1` aliased `_` and `SELECT 1__0` is `1` aliased `__0`. The same shape of rule governs the exponent, because `SELECT 1e` is `1` aliased `e`, so an `e` with no digits after it has to be handed back rather than reported as an error.

There are no hex or binary literals, contrary to what an earlier draft of this document said. `SELECT 0x1F` returns 0 in a column named `x1F`, which is DuckDB reading the number `0` and then the identifier `x1F`, and `0b101` and `0o17` behave the same way.

**Comments** are `--` to end of line and `/* */` with nesting.

**Parameters** are `?` anonymous, `?1` and `$1` numbered, and `$name` named. There is no `:name` form, contrary to another line in an earlier draft: `SELECT :name` is a syntax error and `expression.gram` lists exactly the four rules above. `$` is also dollar quoting, and the disambiguation is that a dollar quote tag is a word that does not start with a digit. `SELECT $1$a$1$` settles it, because DuckDB fails there with `unterminated dollar-quoted string`, which only happens if `$1` was a parameter and the quote started at `$a$`.

**Trailing commas** are legal in most list positions. This is grammar, not tokenizer, but it is the single most used Friendly SQL nicety and it belongs on the same checklist.

**Operators** are a maximal run of operator characters with one exception, which is Postgres's and which DuckDB inherited. A run of more than one character that ends in `+` or `-` keeps those characters only if the run also contains one of ``~ ! @ # ^ & | ` ``. `SELECT 1 =- 1` is `1 = -1` and `SELECT 1 !=- 1` goes looking for an operator named `!=-`. Without the rule, `x=-1` calls an operator nobody defined. `?` is not in the set, because DuckDB spends it on parameters.

Every one of these came from running the query against DuckDB rather than from reading anything, because there is nothing to read: `NumberLiteral`, `StringLiteral`, `Identifier` and `OperatorLiteral` are all in `matcher_rule_overrides`, which is DuckDB stating that its own matcher ignores their bodies. Each behaviour above has a test in `tests/test_sql_token.mojo` naming the query it came from.

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

Twenty four rules are the exception, and they are where the tokenizer and the matcher meet. Their bodies in the grammar text are placeholders, `OperatorLiteral <- Identifier` being the one that shows why it matters, and upstream matches them from code instead. The list is `matcher_rule_overrides` in `grammar_types.yml`, it is vendored the same way the memoized list is, and the generated table carries it as a column on each rule. When the matcher reaches one it asks the token at the current position whether it is that kind of token, in constant time, and never looks at the body. Twenty of the twenty four are identifier rules, twelve that accept an unreserved keyword in place of a name and eight that do not, so this is also the whole of how the grammar's five keyword classes turn into a decision about whether a word can be a name in this position.

The matcher is generic over the rule kinds and knows nothing about SQL. Sequence matches children in order and fails as a unit. Ordered choice tries alternatives left to right, resetting the token position on each failure, and takes the first success, with no longest match, no ambiguity and no conflict. Repetition is greedy with no backtracking into it, which is standard PEG and is a real semantic difference from a regex or a context free grammar that the grammar is written to expect. Lookahead matches without consuming.

## 4. Memoization

Full packrat memoizes every rule at every position, gets linear time, and pays for it with a table proportional to rules times positions, so 1,087 times the token count, plus a lookup on every rule entry. For SQL, where most rules match or fail immediately, that overhead exceeds what it saves on almost every real query.

DuckDB measured the pathology. A query with nineteen unmatched parentheses took 10.640 seconds unmemoized and 0.001 seconds memoized. That is exponential backtracking on the expression rules, which is where the deep ordered choices are.

Their answer, and ours, is a short explicit list of memoized rules. `packrat_memoized_rules` in the upstream source names them. We vendor the list alongside the grammar and treat any change to it as a grammar change. The rules on it are the expression precedence chain and a handful of the deepest statement level choices, which are the places where the same position is genuinely retried by many alternatives.

Implementation is an open addressed table keyed on `(rule, token_pos)`, sized to the token count, allocated once per parser instance and cleared by generation counter rather than by memset. A `SELECT 1` must not pay for a table it does not use, and clearing by generation is what makes the 3 us target reachable.

The pathological input tests go in the suite from day one, generated rather than collected: N unmatched parens, N nested `CASE`, N deep parenthesized expressions, long `IN` lists, deeply nested subqueries. Each has a wall clock ceiling in CI. A PEG parser's failure mode is exponential blowup on adversarial input, it is a denial of service if any front door takes untrusted SQL, and the only defence that works is a test that fails loudly.

## 5. Errors

PEG error reporting is genuinely bad by default. The failure surfaces at the top level choice, having discarded everything it learned, and the naive message is syntax error at position 0.

The standard fix, and DuckDB's, is to track the furthest position reached across all attempts, together with the set of terminals that were expected there. That position is almost always where a human would point. We track `(furthest_token, expected_set)` as two fields updated on every terminal failure, which is cheap and allocates nothing on the success path, and render:

```
Parser Error: syntax error at or near "form"
LINE 1: SELECT * form t
                 ^
```

matching DuckDB's shape, because document 11's corpus matches error text by substring and because the shape is good.

Two refinements are worth their cost. Keyword typo suggestions: when the furthest failure expected a keyword set and the actual token is an identifier within edit distance one of one of them, say so. DuckDB does this and it is most of the perceived quality of a SQL error message. And the rule stack at the furthest position, behind a debug flag, because when a grammar bump breaks something this is the only tool that finds it quickly.

## 6. What the matcher must not do

**No semantic decisions.** Not whether an identifier is a table or a column, not whether a function exists, not whether a cast is valid. The matcher's only output is shape. Every temptation to sneak semantics in here, and the classic one is resolving whether `foo(x)` is a function call or a type constructor, makes the parser impossible to regenerate and breaks the property that document 03 exists to buy.

**No lookups outside the token vector and the rule table.** No catalog, no settings, no session state. A statement's parse must be a pure function of its text and the vendored grammar, which is what makes the prepared statement cache in document 10 sound: cache on the text, because nothing else was an input.

**No error recovery.** The first failure stops the parse. Recovery is for editors and it changes which strings are accepted, which would put a hole in the compatibility claim. If firepanda ever wants a language server, that is a second entry point over the same rule table, and its behaviour is explicitly not what `sql()` does.
