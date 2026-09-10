# The grammar

This is the mechanism that makes the compatibility half of document 01 cheap. Everything else in this specification is work we would have to do anyway for a lazy dataframe engine. This document is the part that is only possible because of what DuckDB shipped in August 2026, and it is the reason to start now rather than at M11.

## 1. The artifact

On the `v2.0-cyanoptera` branch, `src/parser/peg/grammar/` contains:

| | |
| --- | --- |
| `.gram` files | 40, all in `statements/` |
| total bytes | 61,190 |
| lines | 1,421, of which 1,222 are neither blank nor comment |
| rules (`Name <- ...`) | 1,087 |
| keyword lists | 5 files in `keywords/`: 75 reserved, 339 unreserved, 55 column name, 30 function name, 32 type name |

`v2.0-cyanoptera` is DuckDB's default branch rather than a tag, because 2.0 has not been released yet and the last tag is v1.5.5. So the pin is a commit SHA, and the bump procedure below takes either a tag or a branch and records the SHA it resolved to.

Sixty one kilobytes of declarative text is the entire syntactic surface of the dialect that beat Postgres compatibility at its own game. For comparison, DuckDB's hand written transformer, which is the layer after the parser and which we still have to write ourselves, is 46 files and 2,817,088 bytes, of which about 2.3 MB is generated. The grammar is two per cent of the front end and it is the two per cent that defines compatibility.

The license is MIT. We vendor it.

## 2. The notation

PEG, with the ordinary operators and two conveniences.

```
SelectStatement <- WithClause? SelectOrParens SetopClause* OrderByClause? LimitClause?
```

`<-` defines a rule. Juxtaposition is sequence. `/` is ordered choice, meaning the first alternative that matches wins, and this is the whole reason the dialect stopped fighting its parser, because there are no conflicts to resolve, only an order to get right. `?`, `*` and `+` are the usual. `!` is negative lookahead, used in exactly one place, `PlainIdentifier <- !ReservedKeyword <[a-z_]i[a-z0-9_]i*>`. Positive lookahead `&` is in the notation and is never used, and DuckDB's own grammar reader does not accept it, so neither do we.

Two more forms carry the leaves. A bracketed character class, `[ \t\n\r]` or `[^\']`, and an angle bracketed capture holding one, `<[a-z_]i[a-z0-9_]i*>`, where the trailing `i` makes the preceding class case insensitive. Both are lexical rather than structural, both appear in a handful of rules, and both belong to the tokenizer rather than to the matcher, which is document 04's problem.

The two conveniences are what keep 1,087 rules down to 1,421 lines. Parameterized rules, invoked like macros:

```
List(D)   <- D (',' D)* ','?
Parens(D) <- '(' D ')'
```

so `Parens(List(Expression))` is a whole argument list, trailing comma included. And keyword lists as first class token classes, so the grammar refers to `ReservedKeyword` and the tokenizer resolves it against a sorted table. That reference is not defined in any `.gram` file. Upstream's build turns each `.list` file into a rule named after the file, so `reserved_keyword.list` becomes `ReservedKeyword <- 'all' / 'analyse' / ...`, and our generator has to do the same thing before the grammar is even well formed.

Two rules are provided by the runtime rather than by the grammar text: `%whitespace`, whose definition `[ \t\n\r]*` is in `common.gram` but which the matcher applies implicitly between tokens, and `EndOfInput`, which is referenced and never defined.

Uppercase bare words are keywords. Single quoted strings are literal punctuation. Rule references are `CamelCase`. Comments run from `#` to end of line. A rule ends at the first newline that is outside brackets and not preceded by a trailing `/`, which is the only piece of the notation that is whitespace sensitive and the only place a naive reader gets it wrong. That is the entire notation, and a recursive descent parser for it fits in a few hundred lines, which matters because we have to parse the grammar itself to generate anything from it.

## 3. What we take, exactly

**Take verbatim, byte for byte:** `src/parser/peg/grammar/statements/*.gram` and `src/parser/peg/grammar/keywords/*.list`. These are checked into `firepanda/sql/grammar/` under a `VENDOR` file recording the upstream ref, the commit SHA, the retrieval date and the SHA-256 of each file.

**Take the two lists we cannot derive**, both vendored beside the grammar with their own checksums.

`packrat_memoized_rules`, out of `scripts/parser/grammar_types.yml`, is the twenty two rules DuckDB memoizes, all of them on the expression chain from `Expression` down to `FunctionExpression`. That list is a performance decision somebody made with a profiler, and document 04 copies it rather than choosing its own.

The rule overrides, out of the generated block in `src/parser/peg/compiled_grammar.cpp`, are the twenty four rules whose bodies the matcher does not walk, because it matches them itself. This one is not an optimization and skipping it would be a correctness bug. `OperatorLiteral <- Identifier` is what the grammar text says, so a matcher that believed the body would read a bare `+` as an identifier, and the same goes for `NumberLiteral`, `StringLiteral` and the twenty one name rules. The bodies stay in the generated table even though nothing walks them, because the round trip check in section 5 is what proves we read the grammar text correctly and it can only check text that is still there. The generated table records the override as a short section of its own listing the twenty four rules, their matcher and the suggestion each was built with.

The overrides come out of the C++ and not out of `grammar_types.yml`, even though the yml has a `matcher_rule_overrides` block that looks like the same table with the same twenty four names in it. The yml is input to the transformer generator, and its `matcher` field is a result type rather than a matcher class. For twenty three rules the two names coincide. For `ReservedKeyword` the yml says `identifier_string` and the parser installs a `ReservedIdentifierMatcher`, so reading the yml would have given that rule a matcher DuckDB does not use. The C++ is what runs.

The suggestion looks like autocomplete trivia and is not. `IdentifierMatcher` reads it twice: once to pick which keyword class the position tolerates, and once to decide whether a single quoted string counts as a name there. A type name accepts a type name keyword, a function name accepts a type or function name keyword, everything else accepts a column name keyword, and only a table name accepts `'path.csv'`. Dropping the suggestion and keeping only the matcher would have made all twelve identifier rules behave like a column name.

**Take as reference, do not vendor:** the rest of `grammar_types.yml`, which is DuckDB's map from rule name to the C++ node type its transformer produces, plus its `excluded_rules` list. It is a C++ artifact and useless to us directly, but it is the best available index of which rules a transformer actually has to handle and which are pure syntax. Document 05 uses it to order the work.

**Do not take:** the generator, the matcher, the tokenizer, the transformer, the binder. All C++, all ours to write. `scripts/parser/inline_grammar.py` is read for its behaviour, because it is the definition of how the keyword lists and the `.gram` files become one grammar, and then discarded.

The vendored directory is never edited. Not for a fix, not for a workaround, not to add a rule we would like. The moment a local edit exists, compatible by construction becomes compatible except for the edits, and nobody will remember what they were. If upstream's grammar is wrong, the fix goes upstream and we pin the next tag. Document 13 records the one scenario that could force this and what we would do instead.

## 4. How much it moves

The load bearing question for a vendoring strategy is churn, so it was measured across four upstream points rather than assumed.

| ref | rules | `.gram` lines changed against previous |
| --- | --- | --- |
| v1.4.0 | 526 | |
| v1.5.0 | 763 | 1,092 |
| v1.5.5 | 779 | 78 |
| v2.0-cyanoptera, branch head | 1,087 | 691 |

Read it in two parts. Within a release series the grammar is nearly static, at 78 lines across the whole of v1.5.0 to v1.5.5, most of it new function syntax. Across a major release it moves several hundred lines, and the v1.5.0 jump is inflated because that was the series where the PEG grammar was still being brought to parity with the Bison one.

The rule count roughly doubled in a year, which sounds alarming until you look at what the new rules are. The great majority are leaf rules naming keywords and small option clauses, added because a declarative grammar makes it cheap to name things that a Bison grammar would have inlined. The rules that a transformer must actually handle grew far more slowly.

The operational consequence is that a grammar bump is a routine, small, mechanical task done once per upstream release, not a rewrite. Budget a day for a patch bump and a week for a major, where the week is transformer work for genuinely new syntax and not grammar work.

## 5. The generator

A Python script, `tools/gen_grammar.py`, run at build time, with checked in output.

Input is the vendored `.gram` and `.list` files. Output is `firepanda/sql/generated/rules.mojo`, a table of rule descriptors that the matcher in document 04 interprets, plus `keywords.mojo`, which holds every keyword and the classes it belongs to. The output sits beside the vendored directory rather than inside it, because a Mojo subpackage needs an `__init__.mojo` and "the vendored tree is byte for byte upstream" is worth more than the tidier path.

Four decisions about it, all made for reasons that will otherwise be relitigated.

**The output is data, not code.** The generator emits a flat array of rule nodes, covering sequence, choice, repeat, optional, reference, keyword, literal and lookahead, and the matcher walks it. It does not emit one Mojo function per rule. Two reasons: 1,087 generated functions is a compile time problem in a language whose compile times firepanda already tracks with `tools/compile_budget.py`, and a data table can be regenerated and diffed by a human. Interpreting a rule table costs an indirect branch per node, which document 04's budget shows is affordable.

The table goes further than data and is emitted as one string rather than as a list of structs. A list literal with tens of thousands of entries is a compile time cost paid by everyone who builds firepanda whether or not they ever run a query, whereas a string literal costs one entry no matter how long it is, and it stays a readable diff. The price is a reader, `firepanda/sql/table.mojo`, and one pass over ninety kilobytes the first time a program asks for a `Grammar`.

**The keyword table is one sorted list with a class mask, not five tables.** The five classes are not disjoint, twenty six words are both a function name and a type name keyword, so five tables means storing those words twice and deciding which answer wins. Worse, most words in a query are not keywords at all, and the common case with five tables is five misses. One bisection over 499 words answers the whole question, and the classes are bits in a mask that the grammar's keyword nodes carry as their payload, so matching a word against a class is an and.

**Parameterized rules are expanded at generation time.** `List(Expression)` becomes a concrete rule with a synthesized name. This costs a few hundred extra table entries and buys a matcher with no environment to thread through it.

**The generated files are checked in.** A contributor with no Python and no network can build firepanda. CI regenerates and fails on any diff, which is what actually enforces that nobody hand edits the output.

## 6. The bump procedure

Written down because it will be run by someone who has not read this document.

1. `tools/vendor_grammar.sh <ref>` fetches the grammar directory at the ref, resolves the ref to a commit SHA, rewrites `VENDOR`, and stops if nothing changed or if the fetch was partial.
2. `git diff` on the vendored tree is the complete syntactic change in that release. Read it. It is tens of lines for a patch release.
3. Regenerate. The table diff should be proportionate to the grammar diff, and if it is not then the generator has a bug.
4. Run the differential parse harness from document 11 against the new DuckDB. Accept and reject must agree on the whole corpus. New syntax that we now parse and cannot transform shows up here as unsupported, never as a syntax error.
5. New rules with no transformer case get a refusal by name and an issue. That is the entire cost of falling behind on semantics, and it is bounded.

CI runs the first step once a week against the upstream default branch and opens an issue when it moves. That is the mechanism that makes one hundred per cent compatible with DuckDB a maintained property instead of a claim that was true once.

## 7. Where fidelity actually leaks

The grammar guarantees less than it appears to, and it is worth being precise about the three gaps, because each is a place where a bug can hide behind the words but we vendored the grammar.

**The tokenizer is not in the grammar.** String literals, dollar quoting, numeric literal forms, comments, identifier quoting and the case folding rules live in DuckDB's C++ tokenizer. That is a hand written component we must match by behaviour, and document 04 treats it as the primary risk in the front end. It is small, a few hundred lines, and it is where the differential fuzzer earns its cost.

**Ordered choice makes the matcher's semantics load bearing.** The same rule table can be walked with subtly different backtracking or memoization behaviour and accept a different language. This is why document 04 copies DuckDB's memoized rule list rather than choosing its own, and why the differential harness compares parse trees on the corpus and not just accept against reject.

**Some syntax is accepted by the grammar and rejected by the binder, upstream too.** DuckDB's grammar is deliberately permissive in places, leaving the error to a later stage. Our accept and reject agreement is therefore measured per stage, parser against parser and full pipeline against full pipeline, with the two reported separately so that a binder difference never gets counted as a grammar success.
