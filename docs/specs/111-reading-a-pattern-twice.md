# 111. Reading a pattern twice

## 1. What this is

Documents 102 through 110 worked down a list of patterns RE2 reads and this library did not, and the last of them said the list was finished. What it did not say was finished is the other list, and this document is the first slice off it.

The difference between the two is the difference between a pattern that will not read and a pattern that reads to the wrong thing. Every slice up to here had the same shape underneath: Python had no reading, so RE2's was the only reading there was, and a tree that held it was not throwing anything away. That argument runs out at exactly two constructs, where both grammars have a whole reading of the same characters and the two readings have nothing in common.

```
    [[:alpha:]]   a set of letters to RE2, a set of brackets and colons and
                  letters to Python
    a{,2}         zero to two of `a` to Python, the five characters `a{,2}` to
                  RE2
```

Both were measured against the RE2 inside pyarrow 24.0.0 and against a running CPython, and both are patterns pandas hands straight to Arrow, so the reading a caller gets today is RE2's and the reading this library had was Python's.

## 2. Why one tree was not enough

A tree holds one reading. The parser has carried two flags since long before this document, `re2_refuses` for syntax RE2 has never had and `re2_differs` for syntax both grammars read and read differently, and the second of those is an admission rather than a solution: it says the tree in hand is Python's, that RE2 would have built a different one, and that the compiler should therefore refuse to build an RE2 program out of it. Three hundred and sixty one patterns in the corpus carried it and every one of them was a pattern pandas answers and firepanda held out.

The way out is not a cleverer tree. It is a second parse. The characters are the same and the grammar is not, so the parse is told which grammar it is reading, and the caller that tells it is the one that has already decided which engine is going to run the answer.

## 3. Where the flag lives

`parse_pattern` gained a third argument, `for_re2`, defaulting to false, and the cursor carries it. False is Python's grammar and is what every existing caller gets without changing a line, which matters because the router is one of those callers and has to stay on Python's tree.

Only two places in the parser read the flag. Everywhere else the two grammars either agree or one of them has no reading at all, and in the second case the existing arrangement is still the right one: the tree holds the reading that exists and the flags say who else would have refused it.

## 4. The POSIX class

RE2 has fourteen names and Python has none.

The rule for finding one is not the rule anybody would guess. RE2 sees `[:` inside a character class and then scans forward for the next `:]` anywhere at all, without stopping at the `]` that would have closed the class. Whatever lies between is the name, and if it is not one of the fourteen the whole pattern is refused with `invalid character class range`. If there is no `:]` anywhere after it then the `[` was an ordinary member and both grammars read the pattern the same way.

```
    [[:alpha:]]    a class of letters
    [[:a]b:]]      refused, because the name is `a]b`
    [a[:b]c:]d]    refused, because the name is `b]c`
    [[:alpha]]     a set of brackets and letters to both grammars
    [x[:y]]        a set of five characters to both grammars
    [[:]]          a set of two characters to both grammars
    [[::]]         refused, because the name is empty
```

`[[:]]` and `[[::]]` are the pair worth reading twice. The scan starts after the `[:`, so in the first of them it finds nothing and the brackets are members, and in the second it finds the `:]` immediately and the name between is empty, which is a name RE2 has not got.

A name may open with a `^`, and the complement is taken over every code point there is rather than over the ASCII range the name itself lives in, so `[[:^digit:]]` matches a letter and also matches `é`. All fourteen names were measured over the first seven hundred and seventy code points and every one of them is ASCII only:

```
    alnum   30-39 41-5A 61-7A
    alpha   41-5A 61-7A
    ascii   00-7F
    blank   09 20
    cntrl   00-1F 7F
    digit   30-39
    graph   21-7E
    lower   61-7A
    print   20-7E
    punct   21-2F 3A-40 5B-60 7B-7E
    space   09-0D 20
    upper   41-5A
    word    30-39 41-5A 5F 61-7A
    xdigit  30-39 41-46 61-66
```

A class read this way goes into the tree as the ranges it stands for, which is why nothing below the parser had to change. A folded pattern folds the ranges the way it folds any other range, so `(?i)[[:lower:]]` matches an upper case letter, and that is the compiler doing what it already did.

## 5. The count with no lower bound

Document 109 established the half of this that could be established without a second parse. Python has read `{,n}` as `{0,n}` since 3.11 and RE2 never has, because to RE2 a count opens with a digit and `{` followed by a comma is just a brace. Where Python refused the count outright, which is where there was nothing in front of the brace or a position test or another repeat, there was one reading left and document 109 took it.

What was left is the ordinary case, `a{,2}`, where Python counts and RE2 spells. The branch that used to set `re2_differs` now asks the flag first, and a parse reading RE2's grammar takes the same path the refusal cases took, which is one literal node per character from the opening brace to wherever the count reader stopped.

The stack rule falls out of that for free. `a{,2}{1,3}` puts five literals in and then meets a real count, and a real count repeats whatever is on top, which is the closing brace. That is RE2's answer and it is nobody's special case here.

## 6. Who asks for the second reading

`program_for` in `method.mojo`, in the one place it compiles for RE2, and nowhere else.

That is the whole of the wiring and the position of it is the argument. Routing happens first and happens on Python's tree, because the tree pandas had is Python's and a pattern is sent to Python's engine for what Python can see in it. A lookaround written beside a POSIX class is still a lookaround, the pattern still goes to Python, and on that engine Python's reading of the class is the right reading. So the second parse happens after the engine is settled, which means it happens exactly when it is correct and never when it is not.

The branch in the compiler that refused a tree marked `re2_differs` is untouched and is still reachable, because a tree read Python's way does not stop being Python's tree. What has changed is that the one caller who used to hand it that tree now hands it the other one.

## 7. A refusal that could not happen before

There is a shape here that did not exist until there were two readings: a pattern Python reads and RE2 does not. `[[:bogus:]]` is a perfectly good character class to Python and is a refusal to RE2, and before this document the parse that reached the RE2 compile had always already read.

So the RE2 compile now has a failing branch, and it is worded the way the branch above it is worded, by asking the grammar reader what RE2 would have said. A caller gets `RE2 has no such character class` and a `ValueError`, which is the class pandas raises with the message pandas carries, rather than a `NotImplementedError` about a gap the library does not actually have.

## 8. What the reader needed

The same correction the parser needed, which is the second time in eleven slices the reader has been caught out and the second time the corpus was what caught it.

`re2.mojo` has read `[:name:]` since it was written, but it read the name as letters and required the `:]` to come straight after them. That is right for every name RE2 has and wrong for `[[:a]b:]]`, where RE2 scans past the bracket and refuses and the reader stopped at the bracket and read a set. The fix is the scan described in section 4, which the parser and the reader now state separately and identically, and the corpus checks both against a running RE2.

This is the argument for two statements of the grammar made from the same side twice. The reader is not a shortcut for the parser and the parser is not a shortcut for the reader, and where they agreed by accident for eleven slices a new pattern shape found where they did not.

## 9. The differential

The corpus gains 81 patterns, fifty nine POSIX and twenty two counts, and goes from 30316 to 30397.

The line that mattered is gone. `RE2 reads this syntax differently` was a held out reason on six of the seven comparisons and appears on none of them now.

```
    routing         compared 29566, held out 831
    str.contains    compared 29825, held out 572
    str.match       compared 29759, held out 638
    str.fullmatch   compared 29759, held out 638
    str.count       compared 29811, held out 586
    str.replace     compared 30137, held out 260
    str.findall     compared 29923, held out 474
```

Every one of the seven answered ten thousand agreements in ten thousand with no disagreements, and the RE2 grammar reader read 8638, refused 21759 and read nothing RE2 refuses.

## 10. What is left

The four patterns document 110 ended on are still four, and they are now four for a different reason. `{}{,2}{1,3}` is a pattern Python cannot read at all, so it never reaches the RE2 compile, it is turned away at the branch that handles a tree that did not parse. The second reading would read it, and pandas answers it, and reaching it means letting a pattern Python's grammar refused through to the second parse. That is a change worth making and it is not this one, because the branch it changes is the branch that keeps `)a` refused, and what makes `)a` safe today is that nothing gets past it. Letting patterns past means asking the grammar reader on the way in rather than only on the way out, which is a fourth caller for the reader and a measurement of its own.

The other meaning family is untouched and is the larger one: the hundred and twenty four patterns where RE2 puts a non boundary between two bytes of one character. That is not a front end question at all.

Both of them want the same thing first, which is the corpus of texts document 90 section 10 asked for and which still does not exist. Sixteen handpicked texts is what every one of the seven comparisons runs against, and a reading that is wrong only on a text nobody picked is a reading this differential calls right.

The table from document 103 still holds 163 script names the corpus exercises two of.
