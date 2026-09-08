# The grammar

This is the mechanism that makes the compatibility half of document 01 cheap. Everything else in this specification is work we would have to do anyway for a lazy dataframe engine. This document is the part that is only possible because of what DuckDB shipped in August 2026, and it is the reason to start now rather than at M11.

## 1. The artifact

At `v2.0-cyanoptera`, `duckdb/grammar/` contains:

| | |
| --- | --- |
| `.gram` files | 40 |
| total bytes | 61,190 |
| non blank, non comment lines | 1,421 |
| rules (`Name <- ...`) | 1,087 |
| keyword lists | 5 files: 75 reserved, 55 unreserved, 30 column name, 32 type name, 339 other |

Sixty one kilobytes of declarative text is the entire syntactic surface of the dialect that beat Postgres compatibility at its own game. For comparison, DuckDB's hand written transformer, which is the layer after the parser and which we still have to write ourselves, is 46 files and 2,817,088 bytes, of which about 2.3 MB is generated. The grammar is two per cent of the front end and it is the two per cent that defines compatibility.

The license is MIT. We vendor it.

## 2. The notation

PEG, with the ordinary operators and two conveniences.

```
SelectStatement <- WithClause? SelectOrParens SetopClause* OrderByClause? LimitClause?
```

`<-` defines a rule. Juxtaposition is sequence. `/` is ordered choice, meaning the first alternative that matches wins, and this is the whole reason the dialect stopped fighting its parser, because there are no conflicts to resolve, only an order to get right. `?`, `*` and `+` are the usual. `&` and `!` are lookahead, and `!` is parsed and, in DuckDB's own matcher, treated as advisory rather than enforced in some positions, which document 04 handles explicitly rather than by guessing.

The two conveniences are what keep 1,087 rules down to 1,421 lines. Parameterized rules, invoked like macros:

```
List(D)   <- D (',' D)*
Parens(D) <- '(' D ')'
```

so `Parens(List(Expression))` is a whole argument list. And keyword lists as first class token classes, so the grammar refers to `ReservedKeyword` and the tokenizer resolves it against a sorted table.

Uppercase bare words are keywords. Single quoted strings are literal punctuation. Rule references are `CamelCase`. That is the entire notation, and a recursive descent parser for it fits in a few hundred lines, which matters because we have to parse the grammar itself to generate anything from it.

## 3. What we take, exactly

**Take verbatim, byte for byte:** `grammar/**/*.gram` and `grammar/keywords/*.list`. These are checked into `firepanda/sql/grammar/` under a `VENDOR` file recording the upstream tag, the commit SHA, the retrieval date and the SHA-256 of each file.

**Take as reference, do not vendor:** `scripts/parser/grammar_types.yml`, which is DuckDB's map from rule name to the C++ node type its transformer produces. It is a C++ artifact and useless to us directly, but it is the best available index of which rules a transformer actually has to handle and which are pure syntax. Document 05 uses it to order the work.

**Do not take:** the generator, the matcher, the tokenizer, the transformer, the binder. All C++, all ours to write. `build_grammar.sh` is read for its behaviour and then discarded.

The vendored directory is never edited. Not for a fix, not for a workaround, not to add a rule we would like. The moment a local edit exists, compatible by construction becomes compatible except for the edits, and nobody will remember what they were. If upstream's grammar is wrong, the fix goes upstream and we pin the next tag. Document 13 records the one scenario that could force this and what we would do instead.

## 4. How much it moves

The load bearing question for a vendoring strategy is churn, so it was measured across four upstream tags rather than assumed.

| tag | rules | `.gram` lines changed against previous |
| --- | --- | --- |
| v1.4.0 | 526 | |
| v1.5.0 | 763 | 1,092 |
| v1.5.5 | 779 | 78 |
| v2.0-cyanoptera | 1,087 | 691 |

Read it in two parts. Within a release series the grammar is nearly static, at 78 lines across the whole of v1.5.0 to v1.5.5, most of it new function syntax. Across a major release it moves several hundred lines, and the v1.5.0 jump is inflated because that was the series where the PEG grammar was still being brought to parity with the Bison one.

The rule count roughly doubled in a year, which sounds alarming until you look at what the new rules are. The great majority are leaf rules naming keywords and small option clauses, added because a declarative grammar makes it cheap to name things that a Bison grammar would have inlined. The rules that a transformer must actually handle grew far more slowly.

The operational consequence is that a grammar bump is a routine, small, mechanical task done once per upstream release, not a rewrite. Budget a day for a patch bump and a week for a major, where the week is transformer work for genuinely new syntax and not grammar work.

## 5. The generator

A Python script, `tools/gen_grammar.py`, run at build time, with checked in output.

Input is the vendored `.gram` and `.list` files. Output is `firepanda/sql/grammar/generated/rules.mojo`, a table of rule descriptors that the matcher in document 04 interprets, plus `keywords.mojo`, which holds perfect hash lookup tables for the five keyword classes.

Three decisions about it, all made for reasons that will otherwise be relitigated.

**The output is data, not code.** The generator emits a flat array of rule nodes, covering sequence, choice, repeat, optional, reference, keyword, literal and lookahead, and the matcher walks it. It does not emit one Mojo function per rule. Two reasons: 1,087 generated functions is a compile time problem in a language whose compile times firepanda already tracks with `tools/compile_budget.py`, and a data table can be regenerated and diffed by a human. Interpreting a rule table costs an indirect branch per node, which document 04's budget shows is affordable.

**Parameterized rules are expanded at generation time.** `List(Expression)` becomes a concrete rule with a synthesized name. This costs a few hundred extra table entries and buys a matcher with no environment to thread through it.

**The generated files are checked in.** A contributor with no Python and no network can build firepanda. CI regenerates and fails on any diff, which is what actually enforces that nobody hand edits the output.

## 6. The bump procedure

Written down because it will be run by someone who has not read this document.

1. `tools/vendor_grammar.sh <tag>` fetches `grammar/` at the tag, rewrites `VENDOR`, and stops if any checksum is unchanged, meaning there is nothing to do, or if the fetch was partial.
2. `git diff` on the vendored tree is the complete syntactic change in that release. Read it. It is tens of lines for a patch release.
3. Regenerate. The table diff should be proportionate to the grammar diff, and if it is not then the generator has a bug.
4. Run the differential parse harness from document 11 against the new DuckDB. Accept and reject must agree on the whole corpus. New syntax that we now parse and cannot transform shows up here as unsupported, never as a syntax error.
5. New rules with no transformer case get a refusal by name and an issue. That is the entire cost of falling behind on semantics, and it is bounded.

CI runs the first step once a week against the latest upstream tag and opens an issue when it moves. That is the mechanism that makes one hundred per cent compatible with DuckDB a maintained property instead of a claim that was true once.

## 7. Where fidelity actually leaks

The grammar guarantees less than it appears to, and it is worth being precise about the three gaps, because each is a place where a bug can hide behind the words but we vendored the grammar.

**The tokenizer is not in the grammar.** String literals, dollar quoting, numeric literal forms, comments, identifier quoting and the case folding rules live in DuckDB's C++ tokenizer. That is a hand written component we must match by behaviour, and document 04 treats it as the primary risk in the front end. It is small, a few hundred lines, and it is where the differential fuzzer earns its cost.

**Ordered choice makes the matcher's semantics load bearing.** The same rule table can be walked with subtly different backtracking or memoization behaviour and accept a different language. This is why document 04 copies DuckDB's memoized rule list rather than choosing its own, and why the differential harness compares parse trees on the corpus and not just accept against reject.

**Some syntax is accepted by the grammar and rejected by the binder, upstream too.** DuckDB's grammar is deliberately permissive in places, leaving the error to a later stage. Our accept and reject agreement is therefore measured per stage, parser against parser and full pipeline against full pipeline, with the two reported separately so that a binder difference never gets counted as a grammar success.
