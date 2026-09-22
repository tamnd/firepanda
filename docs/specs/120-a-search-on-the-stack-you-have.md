# 120. A search on the stack you have

## 1. What this is

A lookaround standing beside a backreference, an atomic group or a conditional group. `(?=(a))a\1`, `(?=ab)(?>a+)b` and `(?=(a))(?(1)a|b)` are three patterns upstream answers and this library refused, each with a sentence of its own, and they are the last three refusals the Python engine had that were about a construct rather than about a table nobody has written down.

The three refusals arrived one at a time and always for the same reason. Document 95 landed the backreference, document 99 the atomic group and document 100 the conditional, and each of them ended by saying that a pattern holding a lookaround as well has nowhere to go. A lookaround is a search inside a search, the Pike machine was the only engine that could run the inner one, and a backreference, a cut and a test are the three shapes that machine cannot be handed. So the compiler refused the pairing rather than sending the program to an engine that would walk off the end of it.

This is that sentence being wrong. The backtracker can run a search inside a search, on the stack it already has, from the height the stack stood at when the body started. Saying so took one new method, one extra parameter on the walk, and a branch.

## 2. What upstream does with it

Nothing unusual, which is the point. The two constructs do not interact, they are simply both in the pattern:

```
    (?=(a))a\1          on aa    match aa, group 1 a
    (a)\1(?=b)          on aab   match aa, group 1 a
    (?!(a))(b)\2        on bb    match bb, group 1 unset, group 2 b
    (?<=(a))b\1         on aba   match ba, group 1 a
    (?=ab)(?>a+)b       on ab    match ab
    (?=aa)(?>a+)a       on aa    no match
    (?<=(?>a))b         on ab    match b
    (?=(a))(?(1)a|b)    on a     match a,  group 1 a
    (?<=(a))(?(1)b|c)   on zc    no match
```

Two rows are worth naming. `(?=aa)(?>a+)a` does not match, because the cut throws away the choice of taking one `a` instead of two and there is nothing left for the trailing `a` to read, and the assertion in front changes none of that. And `(?<=(a))(?(1)b|c)` does not match `zc`, because the lookbehind fails at every position, so the group never takes part and the test never gets asked. Neither of those needs a rule about the pairing. They are the two constructs each doing what it already did.

## 3. Where the refusals were

All three in one block at the end of `compile_program`, after the instructions are written, reading a flag per construct:

```
    if saw_look and (out.refs or out.cuts or out.asks):
        ok = False, gap = True
        refs ? "this engine has no lookaround beside a backreference yet"
        cuts ? "this engine has no lookaround beside an atomic group yet"
        else "this engine has no lookaround beside a conditional group yet"
```

The sentences were in the order the constructs landed rather than in the order they appear in the pattern, so a pattern holding two of them was named by the older one, on the grounds that a person told about either has been told what to take out. The whole block is gone, along with the `saw_look` flag and the three lines that turned `refs`, `cuts` and `asks` back off on the way out.

The scan itself stays. Those three flags are what picks the engine and they were never only about the refusal. It is a scan over the instructions rather than over the tree, because a construct under a repeat of zero is in the tree and not in the program, so `(?=a){0}(b)\1` is a program with a backreference in it and no assertion.

## 4. The stack the body runs on

`_attempt` used to clear the stacks, push the entry instruction and then walk until the stack was empty. It is now two methods. `_attempt` does the clearing and the pushing, and `_walk` does the walking and takes the height to stop at, so `while len(self.jobs_pc) > 0` becomes `while len(self.jobs_pc) > base`.

That one parameter is the whole of the recursion. `_body` is the new method between them:

```mojo
    var base = len(self.jobs_pc)
    self._push(entry, Int32(at))
    var end = self._walk(program, points, lead, length, at, base, inner, False)
    while len(self.jobs_pc) > base:
        var pc = self.jobs_pc.pop()
        var value = self.jobs_at.pop()
        if pc < 0 and pc != MARKED:
            self._write(Int(-pc - 1), value)
    return end != NO_MATCH
```

The body's own part of the stack is everything above the height the stack stood at when it started. That is not a new idea here. An atomic group already knows which choices are its own by where its mark is, and a mark is a height written down. What is new is that the height is a parameter rather than a constant, and the walk is otherwise the same walk, because the body is ordinary instructions and there is nothing about them that says they are a body.

The unwind is the second half. Whatever the body left on the stack is thrown away, and every save above the height is paid, so the slots are put back the way the path outside left them. A body that matched and a body that failed leave the same thing behind, which is nothing, and that is what makes the caller free to decide what to keep.

## 5. What the body leaves behind

The branch in `_walk` is twelve lines and nine of them are working out where to start:

```mojo
    var behind = instruction.op == IN_BEHIND
    var width = Int(instruction.b) >> 1 if behind else 0
    var want = (instruction.b & 1) == 1 if behind else instruction.b == 1
    var into = Int(at) - width
    var inner = List[Int32]()
    var got = into >= 0 and self._body(program, instruction.a, points, lead, length, into, inner)
    if self.overrun:
        return NO_MATCH
    if got == want:
        if got:
            for k in range(self.nslots):
                if self.slots[k] != inner[k]:
                    self._push(Int32(-k - 1), self.slots[k])
                    self._write(k, inner[k])
        self._push(pc + 1, at)
```

A lookahead starts the body where the path is standing and a lookbehind that many characters further back, which lands the end of the body on the path because the compiler refuses a body whose width is not always the same number. That is document 94's rule and this branch inherits it rather than restating it.

`got == want` is the four forms in one comparison, and the groups are kept only when `got` is true. That is upstream's rule from document 119 read the other way round: a group inside a positive assertion keeps what it matched, a group inside a negative one is unset, and a negative assertion that succeeded is one whose body failed and wrote nothing. So there is no case for it, there is a place where the write is.

The write is pushed back onto the stack as a save to be put back, `-k - 1` with the old value beside it, which is exactly what `IN_SAVE` does. The path outside this assertion does not go through it, and a slot written on one path and left there is a slot the other path reads by accident. The groups a lookaround leaves are owed back the same way any other groups are.

The `overrun` check is there because `_body` can run out of steps, and a body that ran out has not answered no, it has not answered. Returning `NO_MATCH` without it would turn a step count into a wrong column.

## 6. The marks a body leaves in the bitmap

The bitmap says that arriving twice at an instruction and a position is worth nothing the second time, and everything else in this engine rests on that sentence being true. It is true for a cut, which document 99 found, because what a group matches from a position does not depend on the path that arrived. It is false for a backreference and for a conditional, which documents 95 and 100 found, because those are questions about the arriving path.

A lookaround leaves it true and breaks it in one place. Inside a body the sentence holds, because a body is ordinary instructions. Between two separate askings of the same body it does not. A body that matched marked every pair on the way to matching, and the next time the path outside tries a second way through and asks the same assertion at the same position, the walk reads its own marks from last time as pairs already visited and finds nothing. That is a failure the body did not have.

The first version of this slice answered that by turning the bitmap off for any program holding a lookaround, on the grounds that undoing the marks is a third list to keep and the step count is already there as a bound. The corpus said no. Seven patterns of thirty thousand ran out of steps, all of them an assertion under a repeat, and `(?!\b)*+` is the shortest of them. A star over something that reads nothing is a loop, and the bitmap was the thing that stopped it, so taking the bitmap away turned seven answers into seven raises. Three of the seven comparison drivers do not catch a step count and stopped on the first one.

So the marks are undone, and the list to keep with was already here. `stamped` is the mode a program holding a backreference runs in: every cell the walk sets is appended to `marks`, and changing a slot forgets all of them. A program holding a lookaround turns the same mode on, `_body` writes down how long the list was before it started, and on the way out it forgets everything above that. `_forget_from` is `_forget` with a floor. Its one subtlety is a list shorter than the height it was asked about, which means a slot changed inside the body and took the whole list with it, and then everything left was written afterwards and all of it goes. Forgetting too much is free, because the bitmap only ever says that a pair is not worth arriving at twice, so forgetting a cell costs a second arrival rather than an answer.

What that leaves is a bitmap that bounds everything outside a body and everything inside one, and a step count underneath. `tests/test_regex_backtrack.mojo` went from eighty one seconds to thirty three.

## 7. Which engine takes which program

A lookaround alone still goes to the machine next door. `Bounded.__init__` refuses it:

```mojo
    if self.nests and not self.alone:
        self.ok = False
```

`alone` is the flag that already meant this engine is the only one that can run this program, which is `refs or cuts or asks`. So a program whose only unusual thing is an assertion is handed straight back, the way it always was, and a program that holds one of the other three is walked here, assertion and all.

That is deliberate and it is the conservative half of this slice. The machine next door runs a lookaround by starting a second machine with buffers of its own, which is document 93 section 8 and is still allocated per call, and this engine runs one by using stack it already has. Neither of those is obviously cheaper. What is certain is that no pattern that works today changes engines, so nothing that agrees today can stop agreeing because of where it ran.

The other thing that falls out of it is in `program.mojo`. A lookbehind holding a conditional was measured for width and then refused for the pairing, so the width walk was kept only to tell a `ValueError` apart from a gap. Both halves of that walk are visible now: `(?P<n>b)(?<=(?(1)b|c))` is a column and `(?P<n>a)(?<=(?(1)b|cc))` is the error upstream gives.

## 8. The differential

The corpus goes from 30493 patterns to 30517, and the held out totals fall for the first time in three documents:

```
                    before   after
    routing           831      831
    str.contains      230      171
    str.match         124       65
    str.fullmatch     124       65
    str.count         230      171
    str.replace       230      171
    str.findall       474      387
```

59 on four comparisons and 87 on `str.findall`, which is the number documents 95, 99, 100 and 119 all quoted as the thing the pairings would buy, arriving exactly as quoted. What is left on the four is 54 for a named character, 106 for the byte non boundary and 11 for a repeat this compiler will not unroll, and on `str.findall` it is 319 and 68, with no third line. The routing sweep does not move, because a pattern is either read or not read there and the pairings were always read.

The 24 hand written patterns are the three pairings in both directions with nothing else in the way, and they are there because the generator writes the first pairing in quantity, the other two almost never, and none of the three with the second construct inside the body of the assertion rather than beside it. `(?<=(?>a))b` and `(?=(?>a+))ab` are the shapes that say the recursion is a recursion.

All seven comparisons are at ten thousand agreements in ten thousand with zero disagreements. They were not, twice, and both of those are worth keeping.

The first is section 6, the seven patterns that ran out of steps.

The second was in the comparison driver rather than in the engine. `tests/differential/regex_match.mojo` picked the engine with `if program.refs`, where the column kernel picks with `refs or cuts or asks`, so a program holding a cut or a test was handed to the machine, which cannot obey either. That line had been wrong since document 99 and nothing could see it: a cut or a test reaches that driver only in a pattern the router sends to Python, the router walks for a lookaround and a backreference and nothing else, and every such pattern was refused for the pairing. Letting the pairing through was what made the pattern arrive. It reported 18, 14 and 11 disagreements on the three methods and every one of them was the driver reading the right answer from the wrong engine.

That is the second time in two documents that closing a refusal has found something the corpus could not reach past it. Document 119's was a bucket that emptied into the bucket next door and moved no total. This one is a line of code in a test.

## 9. What is left

Three lines, and nothing about a construct. The byte non boundary at 106 patterns on three comparisons, which document 113 section 9 says is not a slice, because every character instruction would have to become a sequence of byte instructions. The named character at 54 on four comparisons and 319 on `str.findall`, which is a table of about forty five thousand names. The repeat this compiler will not unroll at 11 and at 68.

Two measurements are owed and both of them are older than this slice. The nested machine next door still allocates its buffers per call, which documents 93 and 94 named and document 119 made slightly more expensive, and this slice gives the same question a second answer to compare against, since the backtracker runs the same construct on a stack it already had. And nobody has measured what the stamp costs a program that holds a lookaround and a backreference, which is the common shape in the corpus, against the same program without an assertion in it.

The other thing worth measuring is which of the two engines should be running a lookaround alone. This slice keeps that where it was on purpose, so that nothing changes engines, and the question of whether a program whose only unusual thing is an assertion is better off here is now a question somebody can answer with a benchmark rather than with an argument.
