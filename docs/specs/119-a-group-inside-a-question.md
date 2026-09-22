# 119. A group inside a question

## 1. What this is

A capturing group written inside a lookaround. `str.extract(r"(?=(ab))a")` is a pattern upstream answers with the column `ab`, and this library refused it from document 93, when the lookahead landed, until now. The refusal was narrow and it was in the compiler rather than in the machine: a program was turned down when the caller had asked for groups and the tree held a group underneath an assertion, and every caller who had not asked for groups was answered all along.

That narrowness is what makes this a small slice. The parser has read the construct since document 93, the router has sent it to Python's engine since document 85, and the only thing missing was that the nested walk threw away what it matched.

## 2. What upstream does with it

A group inside a lookaround keeps the text the body read, and the assertion still has no width:

```
    (?=(ab))a     on ab     match a,  group 1 ab
    (?=(a))a      on ab     match a,  group 1 a
    (a)(?=(b))    on ab     match a,  group 1 a, group 2 b
    (?!(a))(b)    on cb     match b,  group 1 unset, group 2 b
    (?<=(a))b     on ab     match b,  group 1 a
```

Two rules are hiding in that table and both of them are the whole of this slice.

The first is that the group survives the assertion. The body is run, the body is thrown away in the sense that the position does not move, and what the body wrote into its groups stays written. A machine that treated the nested walk as a yes or no question, which is what this one did, gets the match right and the column wrong.

The second is the fourth row. A negative lookaround whose body matched is a lookaround that failed, and a body that failed writes nothing, so group one is unset rather than holding `a`. That is the row that separates a machine which writes the body's groups when it learns the body matched from one which writes them as it goes.

## 3. Where the refusal was

`_check_node` in `program.mojo`, in the branch that walked into a lookaround:

```
    if b.captures and _holds_group(nodes, it.first):
        give_up
```

`_holds_group` walked the body looking for a capture and existed for that one caller, so it goes with the branch. What is left in its place is a comment saying why there is nothing to ask: the second machine carries slots now and hands back what the arm the pattern preferred matched, so the question the builder was being asked has no answer that is not yes.

Nothing else in the compiler moved. The three pairings that are still refused, which are a lookaround beside a backreference, an atomic group or a conditional group, are refused by the walk above this one and are document 95's business rather than this one's.

## 4. What the machine needed

`_looks` in `pike.mojo` is the nested walk, and it took two more arguments: how many slots a thread carries and the slots of the path that reached the assertion.

It reads them in, because a group opened before the assertion is still set inside it and a body that mentions one has to see what it holds. It runs the body with a thread per position the way the machine outside does, each thread carrying its own copy. And it writes the winner's copy back into the caller's list only when the body matched, which is the fourth row of the table above falling out of where the write is rather than out of a case for it.

The two call sites in `_queue` take a copy of the path's slots before the call and put them back after it. That is the same thing `IN_SAVE` already does for the same reason: the walk that fills the list is one path through the program, the path beside it does not go through this assertion, and a slot written on one path and left there is a slot the other path reads by accident.

## 5. Which thread wins

The part that was wrong on the first try, and the reason there is a test with seven shapes in it.

`_looks` used to return the moment any thread reached the end of the body, which is correct for a question whose answer is yes or no and wrong for one whose answer has groups in it. The machine outside does not do that. It records the match, ends every thread behind it in the list, and carries on walking, so a thread the pattern preferred that is still reading can match further along and take the answer away from the one that got there first.

`(?=(ab)|(a))ab` is the pattern that says so. At the position after `a` the second arm reaches the end of the body, and the first arm is a character short of it and still alive. Upstream sets group one to `ab`. A machine that stopped at the first end it saw sets group two to `a` and leaves group one unset, which is a different column with no error anywhere in sight.

So `_looks` now picks the winner the way `_run` picks one, and the only thing it keeps of the old shape is a return on the spot when the caller has no slots at all. That caller cannot tell the two threads apart, because which thread won is a question only the groups answer, so the walk stops where it used to stop and the four methods that never ask for groups pay nothing for this.

## 6. What the tests ask

`test_regex_extract.mojo` grew the seven shapes, every expectation read off a running CPython rather than reasoned about, because the interesting half of this is which group is left unset rather than which text lands in the ones that are set. Two of them are worth naming. `((?=(a))|b)a` is the row that catches a machine writing the body's slots into the path before it knows the body matched, since the arm holding the assertion fails at position zero and the arm beside it wins, so group two is null rather than holding what the failed body read. And `(?=(a(?=(b))))ab` is a lookahead inside a lookahead, which says the carrying works by recursion rather than at one level.

The two lookaround files lost the test that said the construct was refused when somebody asked for groups and gained one saying it keeps what it matched. `test_regex_extract.mojo` lost the one that said `extract` speaks Python's refusal for this pattern, since there is no refusal left to speak.

## 7. The differential

The corpus goes from 30473 patterns to 30493 and not one comparison holds out fewer than it did:

```
                    before   after
    routing           831      831
    str.contains      230      230
    str.match         124      124
    str.fullmatch     124      124
    str.count         230      230
    str.replace       230      230
    str.findall       474      474
```

That is worth writing down rather than hiding. The bucket `this engine has no capture inside a lookahead yet` is gone from all seven, and every pattern that used to be in it is still held out, under the bucket next door. The generated half of the corpus opens a group inside an assertion only in order to write a backreference to it afterwards, so a pattern of this shape in thirty thousand was refused twice over and closing one of the two refusals moves it from one line of the tally to another. The lookaround beside a backreference is now 45 patterns on four comparisons and 46 on `str.findall`, and it is the same three pairings document 95 left and the same ones the backtracker would have to run.

What did move is the number compared, which is twenty higher on every comparison, because the twenty patterns added by hand are all answered. They are the construct with nothing else in the way: the group alone inside each of the four forms, a group each side of the assertion, two arms where only one is taken, a group inside a group, a lookahead inside a lookahead, a repeat inside the body and a repeat on the assertion, a class and a count inside the body, and the arm that fails beside the arm that does not.

All seven comparisons are still at ten thousand agreements in ten thousand with no disagreements.

## 8. What is left

The three lookaround pairings, which are the same three the backtracker would have to run, at 59 patterns between them on four comparisons and 87 on `str.findall`. The byte non boundary at 106 on three comparisons. The named character at 54 on four and 319 on `str.findall`. The repeat this compiler will not unroll at 11 and at 68.

The nested machine still allocates its buffers per call, which document 93 named and document 94 named again and this slice has now made slightly more expensive, since a call with slots copies two lists in and one out. A lookaround can hold a lookaround and the nesting is what a held buffer would have to be indexed by, so it stays measured work rather than guessed work.

And the corpus still cannot see a construct the generator does not write. Twenty patterns by hand is the answer this time, and the question underneath it is whether the generator should be writing a group inside an assertion without a backreference beside it, which it never has.
