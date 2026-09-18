# 98. A limit that could not have stopped anything

## 1. What this is

Issue #682 was opened against q23 of ClickBench, which is the only query in that suite asking for the table rather than for a reduction of it:

```sql
SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 10;
```

At 1M, ninety five rows out of a million survive the filter and ten of those are the answer, with all 105 columns coming back. The issue said the planner route was building the frame of ninety five rows and 105 columns in the middle and that the fix was a rule folding the filter, the sort and the limit into `filter_sort_limit`, which is the call the hand written port in firepanda-bench makes and which carries positions instead.

That reading is half of it. The middle frame is real and it does cost something, which section 4 measures at about five milliseconds of the twenty four the query took. But the larger half is bigger than that and is nowhere near the width. A filter over a million rows was running on one core, and it was running there because of the `LIMIT 10` on the end of a statement whose limit could not have stopped a single chunk from being read.

That half is one line in the pipeline driver. This document is about that line. The width is still open and section 7 says what is now known about it.

## 2. The rule the driver had

`Pipeline._parallel_lead` decides how many of the leading operators run on every core. It gives up and returns zero, which puts the whole line on the calling thread, when any operator in the line can report FINISHED before the source runs out. Only `Limit` can do that, and the reason for the rule is good: the parallel route reads a batch of chunks before it pushes any of them, and reading a batch ahead on behalf of ten cores is reading rows that a limit was about to make unnecessary.

The rule was written for `SELECT ... LIMIT 10` over a file, where it is exactly right. The scan should read one chunk, the limit should say it has enough, and nothing should read the second chunk at all. Document 08 has that as limit pushdown falling out of asking rather than out of a rule.

The question the rule asks is "is there a limit in this line". The question it means to ask is "can a limit stop the source".

## 3. Why a breaker changes the question

A breaker holds every row it is given and emits nothing until its input has run out. A sort cannot do otherwise, since the last row to arrive can belong at the front. A group by and a reduction are the same, and so are a window and a unique.

So a limit with a breaker under it counts its first row after the last chunk has been read. It can still stop the rows above the breaker, and it does, but there is nothing left to stop by then: the source is empty and the reading is done. The batch the parallel route reads ahead is a batch the breaker was going to be handed anyway.

The rule is therefore that the search for a limit stops at the first breaker:

```mojo
for i in range(len(self.operators)):
    if node_is_breaker(self.operators[i]):
        break
    if node_ends_early(self.operators[i]):
        return 0
```

A join is not a breaker for this purpose even though it holds a build side, because rows flow through it while the source is being read, so a limit above a join can still stop the scan and the loop walks past it to find that limit. That is the same distinction `node_is_breaker` already draws for cutting a pipeline into stages, so the predicate did not have to be invented here.

`_run_batched` has no `_finished` check in it, and the comment beside that said the route is only taken when nothing can end early. It is now taken with a limit in the line, so the comment says the narrower thing that is still true: the limit beyond the breaker has emitted no row and cannot report FINISHED until the breaker under it has seen the last chunk, which is after the batched loop has returned.

## 4. Where q23 actually spends its time

Four statements over the same table, the same 1M partition, five runs each, the median of the five, all back to back so the rows compare with each other. The first is the published q23 and the other three change one thing each.

| | statement | rows out | median |
|---|---|---|---|
| A | the published q23 | 10 of 105 columns | 23.81 ms |
| B | A with no `LIMIT 10` | 95 of 105 columns | 11.47 ms |
| C | A projected to `EventTime` alone | 10 of 1 column | 19.39 ms |
| D | A with no `ORDER BY` and no `LIMIT` | 95 of 105 columns | 6.63 ms |

The hand written port, which is one `filter_sort_limit` call, answered the published q23 in 7.49 ms in the same batch. q20 is `COUNT(*)` over the same predicate with no limit anywhere and answered in 7.89 ms through the planner.

Reading the rows. D is the scan and the predicate and nothing else, and it is the floor at 6.6 ms. B adds the sort of the surviving ninety five rows and costs 4.8 ms more, which is the middle frame: `Sort.process` flattens the selection the filter handed it, so ninety five rows across 105 columns are gathered before anything is ordered. A adds the limit and costs 12.3 ms more than B, for an operator that discards eighty five rows. C says the same thing from the other side: one column instead of 105 takes 4.4 ms off A and leaves 19.4, so the width is a fifth of the gap and the limit is the rest.

Twelve milliseconds to throw away eighty five rows is not a cost the limit is paying. It is the cost of everything under the limit moving to one core.

## 5. What it is worth

The same statements paired against a driver built from the commit before the change with nothing else different, both binaries run back to back in one batch on a quiet machine, five runs each, the median of the five.

| | before | after |
|---|---|---|
| q23 through the planner | 23.81 ms | 13.56 ms |
| q22 through the planner | 21.53 ms | 12.03 ms |
| q23 variant C, one column | 19.39 ms | 7.62 ms |
| q20 through the planner, no limit | 7.89 ms | 7.52 ms |
| q24 through the planner | 7.82 ms | 7.50 ms |
| q23 variant B, no limit | 11.47 ms | 11.59 ms |
| q23 variant D, no sort and no limit | 6.63 ms | 6.62 ms |
| q23 hand written | 7.49 ms | 6.93 ms |

The first three rows are the change. q22 is the same shape as q23 with a narrower projection. Variant C is where it shows clearest, since taking the width out of the way leaves the serial prefix as the whole query and the change takes it from 19.4 to 7.6, which is variant D plus a sort.

The last five rows are the controls and the differences in them are the machine rather than the change. q20 and variants B and D have no limit in them at all, and the hand written route is one `filter_sort_limit` call that never reaches the pipeline driver, so none of the four can be touched by this. q24 is `SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime LIMIT 10`, which is the shape the change is for and does take the new route, but it reads two columns where q23 reads 105 and was already at the floor, so there was nothing in it to win.

q23 through the planner is now 13.56 against 6.93 for the hand written route. Section 7 is about the 6.6 milliseconds between them.

## 6. Why this is not only q23

Twenty two of the 43 ClickBench statements end in `LIMIT 10` and almost all of them have a group by or an order by under it. Every one of those was running its scan and its predicates on one core. q23 is where it showed up because q23 has the least work after the breaker and the most before it, so the serial prefix is the whole query rather than a part of it. On a query whose group by dominates, the same change moves a smaller share.

It is also not only ClickBench. The shape is any aggregate or ordered query with a limit on the end, which is most of what a person types at an interactive prompt.

## 7. What is left

The rule in #682 as it was written, which is one operator carrying positions from the filter through the sort so the wide gather happens once at ten rows instead of once at ninety five, is still worth something and is now the only thing left in q23. After the change the planner is at 13.56 and variant C, which is the same query one column wide, is at 7.62, so the 105 columns are costing about six milliseconds and the rest of the query is costing about seven. That six is the middle frame the issue was opened about.

Two things to know before building it. The first is that the gather is ninety five rows wide, not a million, so what costs six milliseconds is 105 allocations and 105 offset walks rather than any volume of data, and a rule that fuses the three operators is not the only way to stop paying it. The second is that the measurement to take is C against A on the same binary in the same batch, since that is the pair that isolates the width, and not a hand written route against a planned one, which mixes the width in with everything else the planner does differently.

What this document does not answer is whether the batch size is right once a pipeline with a limit in it is allowed on the cores. A limit under a breaker still forces one chunk at a time, and that is correct. A limit above one now reads `worker_count()` times `BATCH_CHUNKS_PER_WORKER` chunks ahead, which is the same number every other pipeline reads, and nothing here measured whether that number is right for a line whose second half is serial.
