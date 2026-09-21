# 110. The other spelling of a name

## 1. What this is

The ninth and last slice of the patterns RE2 reads and this library did not. Documents 102 through 109 took the flag group written in a place, the Unicode name, the repeat on a position test, the run of characters taken literally, the backslash followed by a digit, the name for a group, the dash after a set and the count with no lower bound, and what those eight left behind was three patterns spelled `(?<name>a)` and four the last section of this document is about.

`(?<name>a)` is a named group to RE2 and nothing at all to Python.

## 2. The rule

Python reads `(?<` and then looks at one character. If it is `=` the group is a lookbehind and if it is `!` it is a negative lookbehind, and if it is anything else Python stops there and says `unknown extension ?<` followed by that character.

RE2 reads `(?<` and then looks at the same character. If it is `=` or `!` RE2 refuses the group outright, because RE2 has no lookbehind. If it is anything else RE2 reads a name up to a `>` and opens a capturing group, which is exactly what RE2 does for `(?P<name>`.

So the two grammars use the same three characters to open two constructs that have nothing in common, and each one refuses the other's.

```
    (?<n>a)     Python refuses, RE2 opens a group
    (?<name>a)  Python refuses, RE2 opens a group
    (?<1n>a)    Python refuses, RE2 opens a group
    (?<=a)b     Python reads a lookbehind, RE2 refuses
    (?<!a)b     Python reads a lookbehind, RE2 refuses
    (?<n.m>a)   both refuse
    (?<>a)      both refuse
    (?<n        both refuse
    (?<         both refuse
```

Every row was measured against the RE2 inside pyarrow 24.0.0 and against a running CPython before anything was written.

## 3. Why it is a slice at all

Python has no reading here to lose. That is the same argument documents 101 through 109 made and it is at its cleanest in this one, because Python does not get as far as the name: it decides on the character after the angle bracket and stops. So there is no Python tree for the pattern, RE2's reading is the only reading, and the tree can hold it with Python's own sentence recorded beside it for a caller who reached Python's engine by passing a `flags` argument.

Worth saying plainly, because it was measured rather than assumed: pandas hands these patterns to Arrow. `Series.str.contains("(?<n>a)")` answers a column today rather than raising, so the reading that reaches a caller is RE2's, and a library that refused the pattern would be answering a question pandas does not ask.

## 4. What Python's sentence is

`unknown extension ?<` and then the character Python stopped on, so `(?<n>a)` is `unknown extension ?<n` and `(?<1n>a)` is `unknown extension ?<1`. It names the character rather than the name because Python never read a name. `(?<é>a)` is `unknown extension ?<é`, with the character written out as itself.

The sentence is recorded before the name is read, which is what makes it the one a caller sees when the name is also the second binding of a name already used. Python's first complaint about `(?P<n>a)(?<n>b)` is the `?<` at position nine rather than the redefinition, and recording it first is how the first complaint wins.

## 5. Where the name rule comes from

Document 107 measured which characters RE2 will take in a group name against every code point below U+11000, and the answer was a general category test rather than an ASCII one. That rule is already in this library and it is the same rule under both spellings, so a name RE2 takes after `(?P<` is a name RE2 takes after `(?<` and a name it will not take is refused the same way.

What changes under the new spelling is only whose rule gets asked. Under `(?P<` there are two rules, Python's identifier test and RE2's category test, and the four ways they can come out are what document 107 was about. Under `(?<` there is one rule, because Python's never runs, so the reader is told which spelling it is reading and asks RE2 only.

## 6. Where it lands

The group reader in `parse.mojo` had a branch that took the two lookbehind characters and gave up on everything else. It now records Python's sentence and falls through to the group definition reader.

That reader is new only in the sense that it has a name. The body of `(?P<name>a)` was already written out inline and it is now a function both spellings call, with one argument saying which of them it is. The name reader takes the same argument and skips Python's identifier test when it is set.

The compiler was not touched, the engines were not touched and the router was not touched. A named group is a numbered group once it is read, so `(?<n>a)` compiles to exactly the program `(?P<n>a)` compiles to.

## 7. What the reader needed

One line, which is the first thing any of the nine slices has asked of it that it did not already have.

`re2.mojo` has read `(?<name>` since it was written, which is visible in the sentence in its own group reader that says RE2 has five bracket forms and names this one as part of the third, and the corpus had been confirming that on every run. What it had not been asked was `(?<` with nothing after it, and there the reader ran off the end looking for a name while RE2 never started looking, so the two refused the pattern for different reasons. The line says that RE2 decides what `(?<` is by the character after it and with nothing there the complaint is about the bracket.

That is a small thing and it is the argument for the two statements of the grammar being separate, made from the other side. Eight slices found the reader already right and the ninth found it wrong in a corner nobody had thought to write down, which is what a second statement checked against a running RE2 is for.

## 8. The differential

The corpus gains 40 patterns of this shape and goes from 30276 to 30316.

The count of patterns RE2 reads and this library does not falls from 7 to 4, and the `unknown extension ?<` line goes to nothing rather than to a smaller number.

On the widened corpus the seven comparisons read:

```
    routing         compared 29485, held out 831
    str.contains    compared 29496, held out 820
    str.match       compared 29251, held out 1065
    str.fullmatch   compared 29251, held out 1065
    str.count       compared 29482, held out 834
    str.replace     compared 29839, held out 477
    str.findall     compared 29842, held out 474
```

Every one of the seven answered ten thousand agreements in ten thousand with no disagreements, and the RE2 grammar reader read 8570, refused 21746 and read nothing RE2 refuses.

The reader is 8570 rather than 8544 because the corpus found it a disagreement worth having. `(?<` with nothing after it was the one pattern in thirty thousand the reader and RE2 refused for different reasons, RE2 complaining about the bracket and the reader about the name, because RE2 decides what `(?<` is by the character after it and with nothing there has not got as far as a name. That is one line in the reader and is the first thing the corpus has caught it out on in three slices.

## 9. What is left

Four patterns in thirty thousand, and they are one shape: a count with no lower bound that was read Python's way because there was something in front of it to repeat, followed by a count Python will not put on the result, as in `{}{,2}{1,3}` and `\.{,2}{,2}`. RE2 reads the whole of each as characters and a repeat of the last brace. Reaching them means unwinding a reading the parser has already committed to, which is a different kind of change from any of the nine slices, and it buys nothing measurable: the first count in each of them has already marked the pattern as one the two grammars do not agree about, so the pattern is held out either way.

So the list this document has been working down since document 101 is finished. On this corpus every pattern RE2 reads is a pattern this library reads.

What remains is not about whether a pattern reads but about what it means, and it is two families. The larger is what RE2 reads differently, which the POSIX class and `a{,3}` are both in and which is 255 patterns. The other is where RE2 puts a non boundary between two bytes of one character, which is 124. Neither is a front end question and both want the same thing first, which is the corpus of texts document 90 section 10 asked for and still does not exist.

The table from document 103 still holds 163 script names the corpus exercises two of.
