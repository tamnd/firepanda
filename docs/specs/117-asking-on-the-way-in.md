# 117. Asking on the way in

## 1. What this is

The last row of a list that has been getting shorter for nine slices. `\p{L}` was on it until the name table landed, `\Qa+b\E` until the quoted run landed, `[\d-a]` until the dash after a set landed, `{,3}` until the braceless count landed and `(?<n>a)` until the other spelling of a named group landed. What was left was four patterns in thirty thousand, and document 111 named the change that would reach them and said it was not that document's change.

This is that change. It is three lines of wiring and no new grammar at all, which is why it is worth writing down: the reason those four patterns were held out had stopped being a reason about regular expressions some time ago and had become a reason about the order two questions were asked in.

## 2. The branch that turned them away

`program_for` in `method.mojo` starts by reading the pattern with Python's grammar, because that is the tree pandas routes on. When that read fails there is a branch for it, and what the branch did was compile the failed tree for RE2, which produces a refusal, and then ask the grammar reader whether RE2 would have taken the pattern. If RE2 would not, the reader's own sentence replaces the refusal and the caller gets the `ValueError` pandas gives them. If RE2 would, nothing was replaced and the caller got a gap.

For two patterns in three of the generated corpus the reader says no and the branch does the right thing. For the rest it says yes, and those are patterns pandas answers with a column and this library answered with `Python's grammar cannot read this pattern`, which is a true sentence about a reading nobody had asked for.

The second reading existed all the same. Document 111 added it and wired it into the one place that compiles for RE2 on the path where Python's grammar succeeded. The branch above that path never called it, and the reason given at the time was that the branch is also the one that keeps `)a` refused, and what made `)a` safe was that nothing got past.

## 3. What changed

The grammar reader is asked first instead of last.

It was already being asked, on the same text, in the same branch. Moving the call above the compile rather than below it costs nothing and turns its answer from a correction into a gate. When the reader says RE2 reads the pattern, the second reading is asked for a tree, and if there is one it is compiled and returned. Everything else is exactly as it was: the reader saying no still writes the refusal, and a second reading that does not read still leaves the gap where it was.

So the reader decides and the reading only supplies the tree. That is the division the branch needed. A tree is not evidence that RE2 would take a pattern, because the second reading is this library's statement of RE2's grammar and the whole point of having a separate reader is that a statement of a grammar can be wrong. The reader is the one that has been compared against a running RE2 on thirty thousand patterns in every slice since document 110, and it is the one whose answer is allowed to open the gate.

## 4. The four patterns

```
    {}{,2}{1,3}
    {}{,2}{1,3}
    [abc]{,2}{1,3}?
    (?P<1n>\n){,2}{2,}
```

Four occurrences of three texts, and all three are one shape. A braceless count is a count of zero to n to Python and four ordinary characters to RE2, and a second count written after it is a count on a count to Python and a count on a closing brace to RE2. Python's grammar dies at the second count. RE2 reads `a{,2}{1,3}` as an `a`, then the three characters `{`, `,` and `2`, then between one and three closing braces, which is why it matches the text `a{,2}}}` and does not match `aa`.

The third one carries a group name Python will not take either, which makes no difference here: the pattern was already past saving as far as Python's grammar was concerned, and RE2 takes both the name and the counts.

## 5. What the corpus gained

Four patterns is too few to see a shape in, so eighteen more of the same shape were measured and added: the count with nothing to repeat it on, the count after a class and after an escape and after a group, the count with a star and with a plus after it, a third count stacked on the second, and the shape written beside an anchor, an alternation, a flag group and a bracket. The corpus goes from 30435 to 30453.

```
                    before   after
    routing           831      831
    str.contains      234      230
    str.match         196      191
    str.fullmatch     196      191
    str.count         248      244
    str.replace       248      244
    str.findall       474      474
```

Twenty two held out patterns answered across five comparisons, and every one of the seven still at ten thousand agreements in ten thousand with no disagreements. The RE2 grammar reader reads 8693 and refuses 21760 and reads nothing RE2 refuses.

`str.findall` does not move because it is Python's engine, where a pattern Python's grammar cannot read is not a gap at all.

## 6. The reason that did not disappear

It is gone from `str.contains`, from `str.count` and from `str.replace`. It is still on `str.match` and on `str.fullmatch`, at 67 patterns each, and that number is worth looking at because it is not what it appears to be.

Sixty seven occurrences, two texts. `?` and `^?`, both of which pandas rewrites to `^(?)` before Arrow sees them, because `match` wraps the pattern in a bracket and strips a caret the caller wrote. So the construct nobody can read here is `(?)`, an empty flag group, and it was not written by anybody. It is what a caller's `?` becomes.

RE2 reads it. Measured against pyarrow 24.0.0, `(?)` matches at every position, `(?)a` is `a` and `a(?)b` is `ab`, and `(?)*` is refused with `no argument for repetition operator`, so it is a flag setting that happens to name no flags rather than an empty group that can be repeated. Python refuses it and so do both readings here, which is why the wrap turns a pattern `contains` refuses into a pattern `match` answers, and why this library holds out for `match` exactly what it answers for `contains`.

That is a construct and not a wiring change, it wants the parser and the reader to agree about it, and `a(?)*` is a question that has to be put to RE2 before either of them is written. It is the next slice and it is 134 held out patterns across the two methods.

## 7. What is left

The byte non boundary is the largest thing left at 106 patterns on three comparisons, and it is not a front end question. The named character is 54 on `str.contains` and 319 on `str.findall`, and document 114 said the argument for the thirty thousand name table has to be made on the second number. The three lookaround pairings are 60 between them and the repeat that is too large is 10.

The list this document closes is closed. There is no pattern left in thirty thousand that RE2 reads, that Python's grammar refuses, and that this library turns away for the reason that there was no reading to run, and the branch that used to turn them away now turns away only what the grammar reader says RE2 turns away too.

## 8. What this did not need

No new grammar, no change to the parser, no change to the reader, no change to either engine. The second reading already read all three of the texts in section 4 before this document started, and nothing had ever asked it to.

That is the useful thing here. A slice that adds a construct is measured by the construct. This one is measured by how long a capability sat behind a branch that had been written when it did not exist, which was six slices, and the thing that found it was a canary test whose whole job was to name the pattern that would find it.
