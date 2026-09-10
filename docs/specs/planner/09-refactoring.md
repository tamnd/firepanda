# Refactoring

What in the existing code changes to make room for a planner, and what deliberately does not.

The rule throughout: no user visible behaviour changes, and there is one execution path rather than two that drift apart.

## The package layout

Today: `firepanda/frame/` is the eager API, `firepanda/kernel/` is the column at a time work, `firepanda/exec/` is a chunked engine nothing calls, and `firepanda/hash/`, `firepanda/join/`, `firepanda/io/` are the rest.

Add one package.

`firepanda/plan/` holds the logical layer. `node.mojo` for the nine logical node kinds, `expr.mojo` for the nine expression kinds and the three analyses, `bind.mojo` for name and type resolution, `pass/` for the rewrites one file per pass, `lower.mojo` for turning a logical plan into the existing `exec` nodes, and `print.mojo` for the two explain forms.

`firepanda/exec/` keeps its name and its meaning. It is the physical layer and the plan package lowers into it. The naming collision between `plan/node.mojo` and `exec/node.mojo` is the right collision, because they are the same concept at two levels and keeping the names parallel is clearer than inventing a second word.

## The eager surface does not change

`DataFrame.filter`, `.group_by`, `.join` and the rest keep their signatures and their behaviour. Internally each one builds a one node plan over an in memory scan and collects it immediately.

That is not a rewrite of the frame package. It is a change to the body of each method, and for many of them the body stays a direct kernel call for a while, because there is nothing for a one node plan to optimize and the indirection costs a little. The methods that should route through the plan first are the ones that are internally several operations: `read_parquet` with a column list and a predicate, `read_csv` the same, and anything that takes a list of aggregate specs.

The lazy surface is a separate type. `LazyFrame` holds a plan and does not run it, `collect()` optimizes and runs. That is milestone M3 and it is the reason this folder exists now rather than after M3.

## What gets deleted

**The by name column accessor's role as a workaround.** `DataFrame.column(name)` copies and flattens; `__getitem__(i)` borrows. The TPC-H driver has `_cmp`, `_cmp2`, `_in`, `_year`, `_product` and `_contains`, six helper functions whose entire purpose is to look a name up and hand the borrowed column to a kernel, because the by name accessor costs ninety six megabytes on a text column of six million rows. Binding resolves names to positions once, so those helpers have nothing to do. `DataFrame.column` stays as a public convenience with its copy documented, because a user asking for a column by name and getting a `Series` they own is correct behaviour.

**Hardcoded operator choices in callers.** The TPC-H driver picks `group_broadcast` for q17 and `group_by` plus a semi join for q18, with docstrings explaining which is which. That knowledge moves into the operator, chosen from the measured input.

**`with_column`'s copy as a thing callers work around.** `add_column` exists because building a wide frame with `with_column` deep copies the frame once per column added. A projection node produces one frame from one pass and the question does not arise. Both methods stay, because both are correct for what they promise.

## What gets extended

**`Chunk` and the column metadata.** Document 07 wants a distinct count, a sortedness flag, a minimum and maximum, and an all valid flag on a column. This is the one change that touches a lot of files, because every kernel that produces a column has to decide what it knows about the result. The discipline is that a flag is either known true, known false, or unknown, and unknown is always safe. Start with sortedness and all valid, which are the two with the clearest producers.

**`exec/node.mojo`'s node set.** Missing against document 01's list: `Sort`, `Distinct`, `Union`, and the semi and anti join kinds as first class rather than through `Materialize`. Each is one pull request and each removes one fallback, which is the incremental property Polars' partial lowering has and which `engine/polar/02-lazy-ir.md` calls the single most important structural idea to copy.

**`firepanda/hash/table.mojo`.** Gains a Bloom filter built from the same 64 bit hashes, for documents 04 and 06. New code beside the table rather than a change to it.

**`firepanda/join/pairs.mojo`.** Build side by size, and dense integer key detection. Both are local.

## What does not change

**The kernels.** All of them. A planner decides what to run and the kernels are what runs. The kernel level thresholds, the SIMD, the parallel morsel structure, the scalar twins, the rule that a null is a zero in the values buffer: none of it is affected.

**The Arrow layout.** No change.

**The tests.** Existing tests test behaviour, and behaviour does not change. New tests are needed for the plan layer, and the shape they should take is a plan in and a plan out, printed and compared as text, which is how every optimizer in the world is tested and is much easier to read than assertions about tree structure.

## Two things to be careful about

**Do not build the planner and the streaming engine as one project.** They are independent and each is useful without the other. A plan that lowers to the existing whole column kernels is already worth most of what document 02 measures, because projection and predicate pushdown are about not doing work rather than about doing it in a better order. The `Materialize` fallback is what keeps them independent.

**Do not let the eager path become a second implementation.** The failure mode is that `DataFrame.filter` keeps its direct kernel call for speed, the plan layer grows its own filter, and six months later they disagree about null handling. The defence is that the plan's filter node calls the same kernel the eager method calls, and that any behaviour rule lives in the kernel rather than in either caller.

## What we should take from this document

One new package, `firepanda/plan/`, lowering into the `exec` package that already exists.

The eager API keeps its signatures and its behaviour, and gains a plan behind it rather than beside it.

Column metadata is the change that touches the most files and it is worth doing carefully and early, because five separate decisions elsewhere in this folder depend on it.

Keep the planner and the streaming engine as separate projects sharing a fallback node.
