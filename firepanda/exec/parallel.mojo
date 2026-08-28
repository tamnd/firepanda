"""Running the same body over a range of indices on every core.

This is the whole of the library's threading. There is one primitive, it takes a
body and a count, it runs the body once per index and it returns when all of them
have finished. Nothing here is a queue, a pool or a scheduler, because a morsel
scheduler that nothing yet feeds is a thing to maintain rather than a thing to
use, and the operators that will want one are M6.

`parallelize` is not in the Mojo 1.0 standard library. It moved to `max.algorithm`
along with the rest of the kernel machinery, and getting at it from there means
depending on `max.gpu.host.DeviceContext` for the sake of running a loop on the
CPU. So this is built directly on `TaskGroup` and `create_task` from
`std.runtime.asyncrt`, which is what `max`'s own version is built on underneath.

The shape of the body is the one detail here that is not obvious, and it is not a
matter of taste. A coroutine that captures a closure crashes the compiler in this
toolchain, with a segmentation fault and no diagnostic; a coroutine that takes the
same closure as an argument compiles and runs. So `_one` takes the body rather
than closing over it, and every caller in the library passes state through the
body's own capture list.

Errors do not abort. `max`'s `sync_parallelize` calls `abort` when a task raises,
which for a compute kernel is defensible and for a file reader is not: a CSV file
with a bad value in it must produce a message, not a core dump. Each index gets a
slot to fail into, the group is waited on either way so that no task is still
running when its captured state goes away, and the first failure is re-raised
after. The slots are written one per index and read only after the wait, so there
is nothing shared between two tasks at the same time and no atomic is needed.
"""

from std.runtime.asyncrt import TaskGroup, parallelism_level


def worker_count() -> Int:
    """Returns how many bodies can usefully run at once.

    This is the runtime's own count, which is logical cores rather than physical
    ones. Splitting work more finely than this is free, splitting it less finely
    leaves cores idle, and neither is worth a heuristic of our own.

    Returns:
        The number of workers, at least one.
    """
    var level = parallelism_level()
    return level if level > 0 else 1


async def _one[
    F: def(Int) raises -> None
](body: F, index: Int, failures: MutPointer[List[String], MutUntrackedOrigin]):
    """Runs one index and records a failure rather than propagating it.

    Args:
        body: What to run.
        index: Which index to run it for.
        failures: One slot per index, written only at this index.

    Parameters:
        F: The body's type.
    """
    try:
        body(index)
    except e:
        failures[][index] = String(e)


def parallel_for[F: def(Int) raises -> None](body: F, count: Int) raises:
    """Runs `body(0)` through `body(count - 1)`, returning when all have finished.

    A count of one runs inline. That is not only an optimisation: it keeps a
    single-block read out of the runtime entirely, so the error it raises is
    raised from the caller's own stack.

    Args:
        body: What to run. It may raise.
        count: How many indices to run it for.

    Raises:
        Error: The first failure, by index, if any index failed. Every index is
            run either way.

    Parameters:
        F: The body's type.
    """
    if count <= 0:
        return
    if count == 1:
        body(0)
        return

    var failures = List[String](length=count, fill=String())
    var slots = Pointer(to=failures).unsafe_origin_cast[MutUntrackedOrigin]()
    var group = TaskGroup()
    for index in range(count):
        group.create_task(_one[F](body, index, slots))
    group.wait()

    for index in range(count):
        if failures[index]:
            raise Error(failures[index])
