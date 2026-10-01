"""`firepanda.sql`, which runs a query in DuckDB's dialect over frames in scope.

A table name in the query is looked for in three places, in order: the frames
handed to `register`, the caller's local variables, and the caller's globals.
Only a `DataFrame` or a `Series` is taken from the caller, and a name bound to
anything else is passed over rather than refused, so a local called `orders`
that is a list does not hide a registered frame called `orders`. A `Series` is
the frame of its one column.

SQL folds a name to lower case and Python does not, so `Orders` in a query
reaches a variable called `orders`. Two variables whose names differ only in
case are then the same name to the query, and naming one is refused as
ambiguous rather than answered with whichever was found first. A registered
frame is not ambiguous with a variable, since a registration comes first by
rule.

`df.sql(query)` runs the same way with the frame under the name `self`,
which comes before a registration of that name as a registration comes before
a variable. `capture=False` leaves the caller's variables out, for a library that runs a
query for its own caller and does not want that caller's names in scope.

Only the frames the query could be naming are handed across: a variable whose
folded name is not a word of the query is never read. The statement runs over
a catalog of those frames and nothing else, and the catalog is gone when the
call returns, so a `CREATE TABLE` in one call is not there in the next.

A parameter is written `?`, `$1` or `$name`, as in a prepared statement, and
takes a value passed with the call: `params=[...]` for `?` and `$n`, and
`params={...}` or a keyword for `$name`. A value is never written into the
statement's text. The values travel as one row of a frame of their own and each
parameter reads its column, so a string passed for one is a string and can
never be read as SQL.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from typing import Any

from ._frame import DataFrame, Series, _sql

_registered: dict[str, DataFrame | Series] = {}

# A word a table name could be: a bare identifier, or the inside of a quoted
# one. A string literal is skipped first, so a word inside one names nothing.
_WORDS = re.compile(r"'(?:[^']|'')*'|\"((?:[^\"]|\"\")*)\"|([A-Za-z_][A-Za-z0-9_$]*)")


def register(name: str, frame: DataFrame | Series) -> None:
    """Puts a frame under a name every later `sql` call can read.

    A registration comes before the caller's variables, and registering a name
    again replaces the frame that was there.

    Args:
        name: The name a query says.
        frame: The frame, or a column to read as a frame of one column.
    """
    if not isinstance(name, str) or not name:
        raise TypeError(f"a frame is registered under a non empty string, not {name!r}")
    if not isinstance(frame, (DataFrame, Series)):
        raise TypeError(
            f"only a DataFrame or a Series can be registered, not {type(frame).__name__}"
        )
    for held in [held for held in _registered if held.lower() == name.lower()]:
        del _registered[held]
    _registered[name] = frame


def unregister(name: str) -> None:
    """Takes a registered frame away, as SQL reads the name, so in any case."""
    for held in [held for held in _registered if held.lower() == name.lower()]:
        del _registered[held]


def _words(query: str) -> set[str]:
    """Every word of the query a table name could be, folded."""
    found: set[str] = set()
    for match in _WORDS.finditer(query):
        quoted, bare = match.group(1), match.group(2)
        if quoted is not None:
            found.add(quoted.replace('""', '"').lower())
        elif bare is not None:
            found.add(bare.lower())
    return found


def _caller() -> dict[str, Any]:
    """The caller's variables, its locals over its globals."""
    # The first frame outside the package is whoever called `sql` or
    # `DataFrame.sql`. How many frames sit between is not fixed, since a method
    # of the frame may be wrapped on its way out.
    caller: Any = sys._getframe(1)
    while caller is not None and str(caller.f_globals.get("__name__", "")).startswith("firepanda."):
        caller = caller.f_back
    found: dict[str, Any] = {}
    if caller is not None:
        found.update(caller.f_globals)
        found.update(caller.f_locals)
    return found


def sql(
    query: str,
    params: Any = None,
    *,
    capture: bool = True,
    **named: Any,
) -> DataFrame:
    """Runs one SQL statement in DuckDB's dialect and answers its rows as a frame.

    Args:
        query: The statement.
        params: The values for the statement's parameters: a list or a tuple
            for `?` and `$1`, or a dict for `$name`.
        capture: Whether the caller's local and global variables are in scope
            behind the registered frames.
        named: More values for `$name` parameters, by keyword.

    Returns:
        What the statement answers: a query's rows, the count an `INSERT` adds,
        or a frame with no columns for a statement that answers nothing.

    Raises:
        NotImplementedError: For a statement firepanda does not run yet, which
            the message names.
        ValueError: For a statement that does not parse or does not bind, with
            DuckDB's message, or for a name two variables answer to, or
            for a parameter no value was passed for.
        TypeError: For `params` that is neither a list, a tuple nor a dict,
            or for values passed both by position and by name.
    """
    return _run(query, capture, None, _parameters(params, named))


def _parameters(params: Any, named: dict[str, Any]) -> dict[str, Any]:
    """The values a call passed, each under what it is for: `1`, `2` and on
    for one passed by position, and its name for one passed by name."""
    out: dict[str, Any] = {}
    if params is None:
        pass
    elif isinstance(params, Mapping):
        for name, value in params.items():
            if not isinstance(name, str):
                raise TypeError(f"a parameter is named by a string, not {name!r}")
            out[name] = value
    elif isinstance(params, (list, tuple)):
        for i, value in enumerate(params):
            out[str(i + 1)] = value
    else:
        raise TypeError(
            "params is a list or a tuple for ? and $1, or a dict for $name,"
            f" not {type(params).__name__}"
        )
    if named and isinstance(params, (list, tuple)) and params:
        raise TypeError("values are passed by position or by name, not both")
    out.update(named)
    return out


def _run(
    query: str,
    capture: bool,
    own: DataFrame | None,
    values: dict[str, Any] | None = None,
) -> DataFrame:
    """Runs `query` for `sql` or for `DataFrame.sql`.

    Args:
        query: The statement.
        capture: Whether the caller's variables are in scope.
        own: The frame `self` names, for `DataFrame.sql`.
        values: The parameters' values, each under what it is for.
    """
    if not isinstance(query, str):
        raise TypeError(f"sql takes the statement as a string, not {type(query).__name__}")
    wanted = _words(query)
    chosen: dict[str, tuple[str, DataFrame | Series]] = {}
    if own is not None and "self" in wanted:
        chosen["self"] = ("self", own)
    for name, frame in _registered.items():
        if name.lower() in wanted and name.lower() not in chosen:
            chosen[name.lower()] = (name, frame)
    if capture:
        taken = set(chosen)
        seen: dict[str, str] = {}
        for name, value in _caller().items():
            folded = name.lower()
            if folded not in wanted or folded in taken:
                continue
            if not isinstance(value, (DataFrame, Series)):
                continue
            if folded in seen:
                raise ValueError(
                    f"the name {folded} is ambiguous: the variables {seen[folded]}"
                    f" and {name} both answer to it"
                )
            seen[folded] = name
            chosen[folded] = (name, value)
    names: list[str] = []
    frames: list[object] = []
    for name, frame in chosen.values():
        names.append(name)
        whole = frame.to_frame() if isinstance(frame, Series) else frame
        frames.append(whole._inner)
    said: list[str] = []
    row: object = None
    if values:
        said = list(values)
        row = DataFrame({f"p{i + 1}": [value] for i, value in enumerate(values.values())})._inner
    return _sql(query, names, frames, said, row)
