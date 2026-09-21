# 114. A name only one engine can use

## 1. What this is

After document 113 the largest remaining regular expression gap was `\N{NAME}`, at 374 patterns held out of `str.contains` and much the same of three others. The reason given was `a named character is not resolved yet`, and the reason behind the reason was that resolving a name means carrying the Unicode name table and the table has never been worth it.

This document does not add the table. It observes that three quarters of the patterns wanting it are patterns no table would help, because the engine they are going to be run on has never heard of the construct.

## 2. What RE2 does with it

Nothing, and it says so.

```
    \N{LATIN SMALL LETTER A}    invalid escape sequence: \N
    \N{BULLET}                  invalid escape sequence: \N
    \N{NO SUCH NAME}            invalid escape sequence: \N
    \N{}                        invalid escape sequence: \N
    \N                          invalid escape sequence: \N
    [\N{BULLET}]                invalid escape sequence: \N
```

Measured against pyarrow 24.0.0 and against pandas 3.0.5, which gives the same `ArrowInvalid` for every one of them. RE2 stops at the `N` and never looks at the brace, so the name makes no difference, and neither does whether the name exists, and neither does being inside a character class. Three of those six are patterns `re.compile` also refuses, and the other three are patterns Python reads perfectly well, and pandas raises on all six alike.

That last part is the whole of it. A caller who writes `Series.str.contains(r"\N{BULLET}")` and passes no flag argument gets an exception from pandas today. Answering them a column would be wrong, and holding them out as something this library cannot do yet is wrong in a quieter way, because the thing it cannot do is not the thing standing between it and the answer.

## 3. The flag that already existed

The parser has carried `re2_refuses` since long before this, for syntax Python reads and RE2 has never had: a comment group, `\uXXXX`, a non trailing `\Z`, a backslash in front of a character outside ASCII, a one digit octal escape in a class, and `\b` inside a class. The compiler reads it on the RE2 side only and refuses with `RE2 has no such syntax`, as a refusal rather than a gap, because a refusal is what upstream gives.

`\N{NAME}` belongs on that list and was not on it. Adding it is one line, and the branch that reads it already runs before the branch that refuses an unresolved name, so nothing else had to move.

The placeholder the parser leaves behind is untouched. It reads the braces, notes that it could not resolve what is between them, and adds a node holding U+FFFD, which is what lets the router answer the only question the router asks, which is whether the pattern parses. That was already right and is still right.

## 4. Where the table is still wanted

The engine that could use the answer, and nowhere else.

Fifty four patterns are still held out of `str.contains` for this reason and three hundred and nineteen out of `str.findall`. Those are the patterns that route to Python's engine, where `\N{NAME}` is real syntax, where a name that exists means a particular character and a name that does not is a `re.error`, and where a placeholder standing in for either one would be a column of wrong answers rather than a refusal.

So the table is not cancelled, it is narrowed. It was being asked for by four comparisons and is now asked for by the two that can use it, which also means that when somebody does build it the thing to check is `findall` rather than `contains`.

## 5. What the differential says

```
                    held out before   after
    routing               831           831
    str.contains          554           234
    str.match             514           196
    str.fullmatch         514           196
    str.count             568           248
    str.replace           248           248
    str.findall           474           474
```

One thousand two hundred and seventy six held out patterns answered across four comparisons, all seven still at ten thousand agreements in ten thousand with no disagreements.

`str.replace` does not move because its named character count was already fifty four: the replace path refuses more patterns earlier for other reasons, so the ones this would have reached were not reaching it. `str.findall` does not move because it is Python's engine. Routing does not move because the router holds these out for a reason of its own, which is that pandas raises and there is no routing decision to compare.

The RE2 grammar reader needed nothing. `\N` was already not one of the escape letters it knows, so it was already refusing these patterns and already counting them under `invalid escape sequence`, and this slice is the parser catching up with the reader rather than the other way round. That is the first time in thirteen slices it has gone in that direction.

## 6. What is left

The four largest reasons on `str.contains` are now the byte non boundary at 106, the three lookaround pairings at 60 between them, the named character at 54 and the repeat that is too large at 10.

The gap that was three hundred and seventy four is fifty four, and the remaining fifty four want a table of about thirty thousand names. That is still not worth it, and it is worth a great deal less than it was this morning, which is the point of writing this down: the argument for building the table has to be made on `findall` now, where the number is three hundred and nineteen, rather than on a number that was mostly patterns that raise.
