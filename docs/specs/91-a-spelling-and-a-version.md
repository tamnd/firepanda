# 91. A spelling and a version

## 1. What this is

Document 90 measured every one of the held-out corpus's 30052 patterns against 3.12, 3.13 and 3.14 and found 280 cells that move, in exactly two families. It fixed the first family, which is `\B` on an empty subject, and named the second one and left it. This is the second family.

The rule is short. RE2 spells the end of the string `\z`. Python spells the same position `\Z` and read `\z` as a `bad escape \z` until 3.14, which added it as a synonym for `\Z`. So nothing became sayable in 3.14 that was not sayable before. A spelling stopped being an error.

That makes it a smaller change than the `\B` one and a more awkward one to implement, and the awkwardness is the whole reason it is a document. The `\B` rule is about what a pattern means and the compiler is holding the pattern's meaning by the time it has to decide. This rule is about what the caller typed, and by the time the compiler sees a pattern the caller's typing has already been thrown away, twice, once by the parser and once by this library's own rewriting.

## 2. What actually moved, measured

Asked of running interpreters rather than of a changelog. On 3.13 and every version before it, `\z`, `a\z`, `(?i)\z`, `a\zb`, `\A(a)\z` and `(?:\z)` are each `bad escape \z at position N`. On 3.14 every one of them compiles, and matches exactly where the same pattern with `\Z` matches.

Two things did not move. `\Z` is legal in every version and is the absolute end of the string in all of them, which is worth saying because Perl's `\Z` is not and a reader coming from there will expect a trailing newline to be allowed. And inside a character class both spellings are still an error on 3.14: `[\z]` and `[a\z]` are a bad escape on 3.13 and on 3.14 alike, because 3.14 added an anchor and a character class holds characters. That second fact keeps the version question out of the class parser entirely, which is the one piece of luck in the slice.

150 of the corpus's 280 moving cells are this family. The other 130 were document 90's.

## 3. What pandas does with it, measured

The route decides, and the route is decided by whether a flag was passed, which has nothing to do with escapes. On pandas 3.0.6 under 3.13:

`Series.str.contains(r"a\z")` with no flags answers a column, because the call goes to Arrow and RE2 has always had the spelling. `Series.str.contains(r"a\z", flags=re.MULTILINE)` raises `re.PatternError: bad escape \z at position 1` on the same interpreter, on the same pattern, over the same column. `count` and `replace` and `fullmatch` behave the same way on both sides of that line.

`Series.str.extract(r"(a)\z")` raises with no flags at all, because `extract` is the one pattern method upstream never routes.

`Series.str.contains(r"[\z]")` with no flags is an `ArrowInvalid` rather than an `re.error`, because RE2 refuses the escape in a class too and the refusal arrives from the wrong layer. That is the shape document 78 already records and this slice does not change it.

So upstream's answer to this pattern on one interpreter depends on a keyword argument that says nothing about escapes, and that is the behaviour being copied rather than a defect being worked around. A caller who reads the pandas documentation and concludes that `flags` is an addition to a call is wrong in a way that only shows up on patterns like this one.

## 4. Why the refusal cannot be driven off the tree

`\Z` and `\z` build the same node. They have to, because they are the same position, and any other arrangement would mean two instructions that a matcher has to treat identically and a reader has to learn to. So the parse tree that reaches the compiler cannot tell which spelling produced it, and the compiler is where the version is known.

This is the shape `re2_refuses` has and the shape `scoped` has. Both are facts about the pattern text rather than about the tree, both are recorded by the parser because the parser is the last thing that sees the text, and both are read by the compiler because the compiler is the thing that knows what to do with them. So the parse gained a third field of that kind, `zed`, set when a `\z` is read outside a class and never set inside one.

A reader who wants the general rule: a field on the parse is right when the compiler needs to know something the tree does not hold, and a node is right when a matcher needs to behave differently. This is the first kind. The alternative, which is a distinct node for `\z` that compiles to the same instruction, would put a difference into the tree that nothing downstream may act on, and the first person to write a tree walk would have to remember to treat the two alike.

## 5. The anchor this library writes, which was RE2's spelling in a Python pattern

A call that landed on Python's engine and asked `match` or `fullmatch` is answered by rewriting the caller's pattern with anchors around it, because upstream answers those two with `regex.match` and `regex.fullmatch` and those anchor from outside the pattern. The function that does that rewriting wrote `\A(` and `)\z`.

That is RE2's spelling, in a pattern written for Python's engine, and it was wrong before this slice for a reason that had nothing to do with this slice. It never showed. This library's own parser reads both spellings, and the pattern written there is never handed to a real interpreter, so the wrong spelling compiled to the right position every time and no test could have caught it by asking for an answer.

It blocked the fix completely, though, and in the worst available way. With the anchoring writing a `\z`, every anchored pattern carries one, so a refusal driven off `tree.zed` would refuse every `fullmatch` call with a flag beside it on 3.13, for an escape the caller never wrote. The failure would have been loud rather than subtle, which is the only good thing about it.

So the anchoring writes `\Z` now. Same position, legal in every version of Python, and the only `\z` in a pattern reaching the compiler by the Python route is one the caller put there.

The general rule this is an instance of: a pattern this library writes for an engine has to be spelled in that engine's dialect, and it is not enough that this library's parser reads both, because the dialect is the thing being copied. The Arrow rewrite in the same file goes the other way on purpose, turning a trailing `\Z` into `\z`, and it is right to, because it is writing for RE2. The two rewrites sitting in the same file with opposite spellings is not an inconsistency, it is two engines.

## 6. Where the refusal goes

In `compile_program`, beside the others, in a specific place in the order. After the grammar refusal, after RE2's two, and before the one for a named character that is not resolved yet:

```
if tree.zed and python and minor < PYTHON_ZED_ESCAPE:
```

`python` is there because RE2 has always had the spelling and has no version of Python to ask about. `minor` is the interpreter number document 90 put on the builder, read at the door off `sys.version_info.minor`. `PYTHON_ZED_ESCAPE` is 14 and sits beside `PYTHON_PLAIN_NON_BOUNDARY`, which is also 14 and is a different rule that happens to have landed in the same release, and keeping them as two constants rather than one is the whole argument document 90 made for carrying a number instead of a flag.

The refusal is not a gap. A gap in this library means a caller asked for something pandas answers and this library does not, and it comes back as `NotImplementedError`. Upstream refuses this pattern too, on the same interpreter, for the same reason, so this is a `ValueError` and it is the ordinary kind of refusal. The one difference from upstream is the class: `re.PatternError` has `Exception` as its only base and this library raises `ValueError` for every bad pattern on either engine, which document 86 already records as a divergence.

## 7. The order the three steps happen in

`program_for` routes first, rewrites for Arrow second, anchors third, and the version check happens inside the compile that comes after all three. That order is upstream's and this slice did not choose it, but it is what makes the answer right, so it is worth writing down why.

Routing first means an unflagged call has already left for Arrow before anything asks about a version, which is upstream's answer. The Arrow rewrite, which turns a trailing `\Z` into `\z`, only runs on a call that stayed on Arrow, so it cannot manufacture a `\z` on a pattern headed for Python. And the anchoring, now that it writes `\Z`, cannot either. So by the time the compiler reads `tree.zed`, a set bit means the caller wrote one, on every path.

That is three separate things that had to be true, and two of them were already true by accident and one of them was false. A reader auditing this later should check all three rather than the one this slice changed.

## 8. What it looks like compiled

`\z` beside 13 on Python's engine is `this Python has no \z escape`. Beside 14 it compiles, and so does `\Z` beside anything. On RE2 both spellings compile beside every number, because the number is not consulted.

`[\z]` is `Python's grammar cannot read this pattern` beside every number and on both engines, which is the ordinary bad escape path and not this rule at all. It is a row in the tests anyway, because the fact that no version moves it is the thing a reader would want to check.

`fullmatch` with a flag and a pattern holding no `\z` compiles beside 12, 13 and 14, which is the row the whole slice turns on and would have failed before section 5's change.

## 9. What moved

The parser gained `zed` on `Parsed` and on its cursor and sets it in the escape reader. The compiler gained `PYTHON_ZED_ESCAPE` and one refusal. `python_anchored` writes `\Z` where it wrote `\z`, and the test file that pinned its output now pins the new spelling. There is a new Mojo test file for the rule and a new accessor test file, and the accessor one computes whether the escape is legal by asking the running `re` at import rather than spelling an answer, which is document 90's rule and is what lets the same file be green under 3.13 and under 3.14.

## 10. What is not done

The job that fails when the interpreter running the build is newer than `PYTHON_NEWEST` still does not exist. Document 90 deferred it on the grounds that a mechanism for keeping a list of one rule honest is a mechanism nobody maintains. The list is two now, which is the condition that document set, so it is the next thing rather than a someday thing, and it is not in this slice because it is a change to the build rather than to the engine and those are two reviews.

Nothing here asks 3.15. The two rules found were found by measuring three versions that exist, and the corpus sweep that found them is a thing somebody has to run again rather than a thing that runs itself.

And the sweep still only asks sixteen texts. A corpus of texts built the way the corpus of patterns was built does not exist, which document 90 also named, and a version rule that shows up only on a text nobody thought to include would be invisible to both of these slices.
