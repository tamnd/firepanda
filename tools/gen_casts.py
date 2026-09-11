#!/usr/bin/env python3
"""Writes firepanda/sql/generated/casts.mojo from a live DuckDB.

Two things are read off DuckDB here and neither of them is written down
anywhere DuckDB publishes.

The first is which implicit casts exist. `can_cast_implicitly(NULL::A, NULL::B)`
answers that directly for every pair of types, which is where the matrix comes
from. It is full of things nobody would guess: `TINYINT` does not cast to
`VARCHAR`, so `length` over a `TINYINT` column is a binder error, and
`UTINYINT` casts to `SMALLINT` but not to `TINYINT`.

The second is what a cast costs, and there is no function for that one.
DuckDB picks the overload whose casts total the least, so every call it binds
is an inequality: the signature it chose costs less than every rival that was
still reachable from the same arguments. This script collects those
inequalities by asking DuckDB to `EXPLAIN` every one, two and three argument
tier 1 call, over a real table column per argument, and reading the casts out
of the plan. A literal argument is no good for this, because constant folding
runs before the plan is printed and the cast disappears with it, and two
arguments of the same type have to be two different columns or the two casts
land on one name and the reading is wrong. What the plan shows is the casts and
not the signature, so the signature is recovered as the one candidate the casts
are consistent with, which is also how `DECIMAL` and `ANY` are kept honest:
both are templates, both bind without leaving a cast behind, and a slot with no
cast under it is not the same thing as an exact match.

Then it solves. The costs that come out are a linear program's answer, not
DuckDB's own numbers, and they do not have to be: what matters is that they
order the same way DuckDB's do. That is checked rather than asserted. Every
call collected is replayed through the solved table, and a single disagreement
fails the run. Two thousand of them, which is every decision the tier 1 catalog
can be made to show at these arities.

A cost is a property of the target type alone. That was a hypothesis at the
start and it is now a measured fact: a model with one number per target
satisfies every inequality, and a richer model with one number per source and
target pair buys nothing. Note that this is about cost, not about
reachability, which very much depends on the source.

The matrix covers scalar types only. `LIST`, `ARRAY`, `STRUCT`, `MAP` and
`UNION` cast by their elements, so one row for `INTEGER[]` would be a claim
about every list there is; firepanda/sql/types.mojo draws the same line and for
the same reason.

Run `pixi run gen-casts` and commit the result. CI regenerates it and fails on
any diff.

Usage:
    python tools/gen_casts.py            write the table
    python tools/gen_casts.py --check    fail if the checked in file is stale
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import re
import sys

import duckdb
import numpy as np
from scipy.optimize import linprog

from gen_functions import TIER_ONE

# The scalar half of firepanda/sql/types.mojo, each type paired with the
# identifier that file gives it, the name it goes under there and a spelling
# DuckDB will accept. The identifiers are repeated here rather than parsed out
# of the Mojo, so that a renumbering over there fails the test suite rather
# than silently shifting every row of this table by one. The name is checked
# against `typeof` below, which is what catches a spelling that names a
# different type from the one the identifier means.
SCALARS = [
    (1, '"NULL"', '"NULL"'),
    (2, "BOOLEAN", "BOOLEAN"),
    (3, "TINYINT", "TINYINT"),
    (4, "SMALLINT", "SMALLINT"),
    (5, "INTEGER", "INTEGER"),
    (6, "BIGINT", "BIGINT"),
    (7, "HUGEINT", "HUGEINT"),
    (8, "UTINYINT", "UTINYINT"),
    (9, "USMALLINT", "USMALLINT"),
    (10, "UINTEGER", "UINTEGER"),
    (11, "UBIGINT", "UBIGINT"),
    (12, "UHUGEINT", "UHUGEINT"),
    (13, "FLOAT", "FLOAT"),
    (14, "DOUBLE", "DOUBLE"),
    (15, "DECIMAL", "DECIMAL(18,3)"),
    (16, "VARCHAR", "VARCHAR"),
    (17, "BLOB", "BLOB"),
    (18, "DATE", "DATE"),
    (19, "TIME", "TIME"),
    (20, "TIME WITH TIME ZONE", "TIMETZ"),
    (21, "TIMESTAMP", "TIMESTAMP"),
    (22, "TIMESTAMP WITH TIME ZONE", "TIMESTAMPTZ"),
    (23, "TIMESTAMP_S", "TIMESTAMP_S"),
    (24, "TIMESTAMP_MS", "TIMESTAMP_MS"),
    (25, "TIMESTAMP_NS", "TIMESTAMP_NS"),
    (26, "TIME_NS", "TIME_NS"),
    (27, "INTERVAL", "INTERVAL"),
    (34, "UUID", "UUID"),
    (35, "BIT", "BIT"),
    (36, "BIGNUM", "BIGNUM"),
    (37, "VARIANT", "VARIANT"),
]

# How many identifiers types.mojo has, TYPE_COUNT over there. The generated
# table is written at this width with a row of refusals for every identifier
# not in SCALARS, so that a reader can index it by type identifier and never
# think about the gap.
TYPE_COUNT = 40

# Arities to collect decisions at. DuckDB's catalog has 95 names with several
# one argument overloads, 101 with several two argument ones, 11 at three and
# three at four. The three at four are `equi_width_bins`, `regexp_extract` and
# `regexp_extract_all`, none of them tier 1, and a fourth argument multiplies
# the calls to try by the width of the type list again for no decision this has
# not already seen.
ARITIES = (1, 2, 3)

# `EXPLAIN` prints the plan as a box drawing, so the cast is found by pattern
# rather than by structure. The column names are chosen to be unambiguous.
CAST = re.compile(r"CAST\((p\d+_\d+) AS ([A-Z0-9_\[\]() ,\"]+?)\)")

def canonical(text: str) -> str:
    """What `typeof` would call a type, given any of the ways DuckDB spells it.

    A cast spells a type one way and `duckdb_functions()` spells it another, so
    everything here is keyed on the one name. `TIMETZ` and
    `TIME WITH TIME ZONE` are one type, and comparing the two spellings instead
    of the two types drops candidates without saying so.

    Args:
        text: The spelling, from wherever it came.

    Returns:
        The canonical name.
    """
    # A decimal parameter carries no width, because DuckDB's decimal parameter
    # is a template that takes the width of the argument. The width a column
    # happens to have is not part of the question here.
    if text.startswith("DECIMAL"):
        return "DECIMAL"
    # A bare capital letter is one of DuckDB's template parameters, `T` in
    # `first(T)` and `K` in `contains(MAP(K, V), K)`. It takes an argument of
    # any type without a cast, which is what `ANY` does, so it is counted as
    # `ANY` here. The two cannot both be the cheapest thing on a slot or
    # `first` would be ambiguous over every column there is, and it is not, so
    # they differ; which way round is not something the catalog can be made to
    # say, because the only name that has both has them returning the same type
    # and doing the same thing.
    if len(text) == 1 and text.isalpha():
        return "ANY"
    return text


def signature_of(line: str) -> tuple:
    """The parameter types of one candidate line from an error message.

    DuckDB writes a candidate as `century(INTERVAL) -> BIGINT`, and the
    parameters are what sits between the brackets. They are split by hand
    rather than on commas, because `MAP(K, V)` and
    `STRUCT("year" BIGINT, "month" BIGINT)` are one parameter each and both
    hold a comma.

    Args:
        line: The candidate, with no leading tab.

    Returns:
        The canonical parameter types, or an empty tuple for a line that is not
        a candidate.
    """
    opened = line.find("(")
    if opened == -1:
        return ()
    depth = 0
    closed = -1
    for at in range(opened, len(line)):
        if line[at] == "(":
            depth += 1
        elif line[at] == ")":
            depth -= 1
            if depth == 0:
                closed = at
                break
    if closed == -1:
        return ()
    parameters = []
    depth = 0
    quoted = False
    current = ""
    for character in line[opened + 1 : closed]:
        if character == '"':
            quoted = not quoted
        if not quoted and character == "(":
            depth += 1
        elif not quoted and character == ")":
            depth -= 1
        if character == "," and depth == 0 and not quoted:
            parameters.append(current.strip())
            current = ""
            continue
        current += character
    if current.strip():
        parameters.append(current.strip())
    return tuple(canonical(parameter) for parameter in parameters)


def tied(message: str) -> list[tuple]:
    """The signatures DuckDB says it could not choose between.

    Its ambiguity error lists the candidates that tied and not the ones that
    lost, which makes it a direct reading of which signatures cost the same.
    Nothing else DuckDB prints says that.

    Args:
        message: The error text.

    Returns:
        The tied signatures, or an empty list for any other error.
    """
    if "Could not choose a best candidate" not in message:
        return []
    if "Candidate functions:\n" not in message:
        return []
    listed = []
    for line in message.split("Candidate functions:\n", 1)[1].split("\n"):
        if not line.startswith("\t"):
            break
        signature = signature_of(line.strip())
        if signature:
            listed.append(signature)
    return listed


HEADER = '''"""Which implicit casts exist and what they cost.

Generated by tools/gen_casts.py from DuckDB {version}.
Do not edit. Run `pixi run gen-casts` and commit the result.

{scalars} scalar types, {edges} implicit casts between them, {costs} distinct
costs, checked against {decisions} overload decisions read off DuckDB.

DuckDB resolves a call by totalling what it would cost to cast each argument
to each candidate signature and taking the cheapest. The numbers below are not
DuckDB's own, because DuckDB publishes none; they are the answer to the
inequalities its choices imply, and they order the same way. The generator
replays every decision it collected through this table and fails if one comes
out differently, so the claim that they order the same way is checked on every
CI run rather than believed.

{ambiguities} of those decisions are DuckDB refusing to decide. It lists the
candidates that tied when it refuses, and that list is the only place it ever
says two signatures cost the same, so those are the equalities the solver gets
and every other decision is an inequality per candidate that lost.

{guessed} of the {edges} costs are a floor rather than a reading. No tier 1
call turns on them, so nothing DuckDB does pins them down and they are written
as the cheapest a cast is allowed to be. The rest are held in place by at
least one decision.

A cost belongs to the target type alone, except from `NULL`. A `NULL` argument
is not a value being converted and DuckDB does not price it as one, so the
`NULL` column is a set of numbers of its own. Whether the cast exists at all
very much belongs to the pair: `UTINYINT` reaches `SMALLINT` and not
`TINYINT`, and `TINYINT` reaches neither `VARCHAR` nor any unsigned type.

Scalar types only. A list, an array, a struct, a map and a union cast by their
elements, so a row here for one of them would be a claim about every list
there is. See docs/specs/sql/06-types-and-semantics.md.
"""


# No implicit cast from the one type to the other. Not a cost of zero, which
# is what an exact match costs.
comptime NO_CAST: Int16 = -1

comptime SCALAR_COUNT: Int = {scalars}
comptime CAST_COUNT: Int = {edges}

# What an `ANY` parameter costs. It takes the argument as it stands and
# inserts no cast, so it is not in the matrix, but it is not free either:
# `first` has an `ANY` overload and a `DECIMAL` one and picks the `ANY` one
# for an integer column, which only says something because both cost
# something. A single letter template is counted as `ANY` here, which is not
# the whole truth: `first` has one of each and DuckDB binds it without
# complaint, so the two cannot cost the same. Which of them is dearer is not
# something the catalog can be made to say, because the name that has both has
# them returning the same type, and resolve.mojo breaks the tie the one way
# that leaves `first` bindable.
comptime ANY_COST: Int16 = {any_cost}

# The version the table was read off. DuckDB has changed which casts are
# implicit between releases, and a query that binds one way on 1.5 and another
# way on 1.6 is a compatibility break we would want to see rather than inherit.
comptime DUCKDB_VERSION: StaticString = "{version}"

# One line per type identifier in firepanda/sql/types.mojo, line `n` for
# identifier `n` starting at `TYPE_INVALID`, and including the identifiers
# this table does not speak for, whose name is written `-` and whose costs are
# all `NO_CAST`. The name is the one `type_name` gives the identifier over
# there, and the reader holds every line against it, which is what catches a
# table generated against a different numbering from the one it is read under.
# After the name comes one number per identifier: what it costs
# to cast a value of that identifier's type to this line's type, `NO_CAST`
# where there is no implicit cast, and zero on the diagonal because an exact
# match is not a cast.
#
# The line is the target and the position within it is the source, which is
# the way round a binder wants it: it has a signature to price and a column of
# arguments to price against.
comptime TABLE: StaticString = """
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="fail if the output is stale"
    )
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    out = root / "firepanda" / "sql" / "generated" / "casts.mojo"

    connection = duckdb.connect()
    version = connection.execute("select version()").fetchone()[0].lstrip("v")
    # One thread, so that a plan is printed the same way twice.
    connection.execute("set threads = 1")

    spellings = [spelling for _, _, spelling in SCALARS]
    names = {
        spelling: canonical(
            connection.execute(f"select typeof(NULL::{spelling})").fetchone()[0]
        )
        for spelling in spellings
    }
    for _, name, spelling in SCALARS:
        if names[spelling] != name:
            print(
                f"{spelling} is a {names[spelling]} and not a {name}",
                file=sys.stderr,
            )
            return 1
    # A column per type per argument position. Two arguments of the same type
    # have to name two columns, or the two casts collide on one column name
    # and the pair is read back as whatever the second one said.
    columns = [
        {names[spelling]: f"p{slot}_{i}" for i, spelling in enumerate(spellings)}
        for slot in range(max(ARITIES))
    ]
    connection.execute(
        "create table probe("
        + ", ".join(
            f"{columns[slot][names[spelling]]} {spelling}"
            for slot in range(max(ARITIES))
            for spelling in spellings
        )
        + ")"
    )
    # One row in it, every column of it NULL, because `typeof` over a table
    # with no rows in it has no row to give the type back on.
    connection.execute(
        f"insert into probe ({columns[0][names[spellings[0]]]}) values (NULL)"
    )

    reaches = {}
    for source in spellings:
        for target in spellings:
            reaches[(names[source], names[target])] = connection.execute(
                f"select can_cast_implicitly(NULL::{source}, NULL::{target})"
            ).fetchone()[0]
    sources = [names[spelling] for spelling in spellings]

    # `ANY` takes an argument of any type without a cast and is not free, so
    # it is priced like a target even though nothing is cast to it. Everything
    # else a signature can say that is not a plain type name, which is the
    # single letter templates and the container spellings, cannot be reached
    # from a scalar argument at all.
    def reachable(source: str, want: str) -> bool:
        return want == "ANY" or want == source or bool(reaches.get((source, want)))

    # Every tier 1 name and arity DuckDB has more than one overload for. A name
    # with one overload decides nothing, and varargs never come with a second
    # overload to choose against.
    #
    # Tier 1 and not the whole catalog, which is a scope and not a shortcut.
    # Nothing outside tier 1 binds in firepanda, and the rest of the catalog
    # does not all fit an additive cost anyway: `json_extract_path`,
    # `json_extract_path_text` and `json_extract_string` want `VARIANT` to
    # reach `VARCHAR` more cheaply than `BIGINT` while `abs` wants the
    # opposite, and no cost per pair, let alone per target, has both. Whatever
    # the json functions are doing, they are doing it in a bind function and
    # not in the scoring, and the day tier 2 arrives with them in it is the day
    # to find out what.
    rows = connection.execute(
        """
        select function_name, parameter_types, return_type
        from duckdb_functions()
        where schema_name = 'main'
          and function_type in ('scalar', 'aggregate')
          and varargs is null
        """
    ).fetchall()
    groups: dict[tuple[str, int], dict[tuple, str]] = {}
    for name, parameters, returns in rows:
        if not parameters or len(parameters) not in ARITIES:
            continue
        if name not in TIER_ONE:
            continue
        signature = tuple(canonical(parameter) for parameter in parameters)
        groups.setdefault((name, len(parameters)), {}).setdefault(
            signature, canonical(returns)
        )
    groups = {key: value for key, value in groups.items() if len(value) > 1}

    # Two readings per call, because neither one answers on its own.
    #
    # The plan says which casts went in, which names the signature whenever
    # every slot that mattered was cast. It says nothing about a slot that
    # needed no cast, and nothing at all about a `NULL` argument, because a
    # call whose arguments are all constant is folded before any plan is
    # printed.
    #
    # `typeof` says what the call came out as, which finishes the job wherever
    # the candidates return different types. It is only believed for a name
    # whose returns have already been seen to be the ones the catalog declares,
    # since an aggregate is allowed a bind function that computes a return type
    # of its own and several of them have one.
    observed = []
    for (name, arity), catalog in sorted(groups.items()):
        candidates = sorted(catalog)
        wanted = {parameter for signature in candidates for parameter in signature}
        live = [s for s in sources if any(reachable(s, w) for w in wanted)]
        for combination in itertools.product(live, repeat=arity):
            rivals = [
                signature
                for signature in candidates
                if all(
                    reachable(have, want)
                    for have, want in zip(combination, signature)
                )
            ]
            if len(rivals) < 2:
                continue
            call = f"{name}(" + ", ".join(
                columns[slot][source] for slot, source in enumerate(combination)
            ) + ")"
            try:
                returns = canonical(
                    connection.execute(
                        f"select typeof({call}) from probe"
                    ).fetchall()[0][0]
                )
            except Exception as error:
                # An ambiguity error lists the candidates that tied, which is
                # the one place DuckDB says out loud that two signatures cost
                # the same. Any other error is a call that binds to nothing and
                # says nothing about cost.
                equal = [s for s in tied(str(error)) if s in rivals]
                if len(equal) > 1:
                    observed.append((name, combination, equal, rivals))
                continue
            plan = " ".join(
                connection.execute(f"explain select {call} from probe")
                .fetchall()[0][1]
                .split()
            )
            seen = dict.fromkeys(
                (columns[slot][source] for slot, source in enumerate(combination))
            )
            for column, target in CAST.findall(plan):
                if column in seen:
                    seen[column] = canonical(target.strip())
            # A plan that mentions none of the call's columns is a call that
            # was folded away, which happens to every call whose arguments are
            # all `NULL` and to a null propagating one that has a `NULL` among
            # them. Nothing about it can be read off the plan, and reading a
            # slot with no cast on it as an exact match would be reading the
            # fold rather than the binder.
            folded = not any(column in plan for column in seen)
            casts = [
                seen[columns[slot][source]]
                for slot, source in enumerate(combination)
            ]
            fits = [
                signature
                for signature in rivals
                if all(
                    want == cast
                    if cast is not None
                    else folded or have == '"NULL"' or want in (have, "ANY")
                    for have, cast, want in zip(combination, casts, signature)
                )
            ]
            observed.append((name, combination, fits, rivals, returns, catalog))

    # Which names `typeof` can be believed about. A name whose plan named the
    # signature outright and whose return then did not match what the catalog
    # says that signature returns is a name with a bind function, and its
    # return is not a reading of anything.
    honest: dict[str, bool] = {}
    for entry in observed:
        if len(entry) != 6:
            continue
        name, _, fits, _, returns, catalog = entry
        if len(fits) != 1 or catalog[fits[0]] == "ANY":
            continue
        honest[name] = honest.get(name, True) and catalog[fits[0]] == returns

    # One observation is one call, the signatures DuckDB took and the ones it
    # passed over. A bind has one winner and an ambiguity has several, and the
    # difference is the whole reason the ambiguity errors are collected: they
    # are the only equalities there are.
    decisions = []
    ambiguities = 0
    for entry in observed:
        if len(entry) != 6:
            name, combination, winners, rivals = entry
            ambiguities += 1
        else:
            name, combination, fits, rivals, returns, catalog = entry
            if len(fits) > 1 and honest.get(name):
                fits = [
                    signature
                    for signature in fits
                    if catalog[signature] in ("ANY", returns)
                ]
            if len(fits) != 1:
                continue
            winners = fits
        losers = [s for s in rivals if s not in winners]
        if losers or len(winners) > 1:
            decisions.append((name, combination, winners, losers))

    if len(decisions) < 2000:
        print(
            f"only {len(decisions)} decisions to solve against, which is too"
            " few to trust the answer",
            file=sys.stderr,
        )
        return 1

    # One variable per target type, and a second one for every target reached
    # from `NULL`, because `NULL` is not a value that gets converted and DuckDB
    # does not price it as though it were. Everything else shares a column:
    # what a cast costs is the target and not the pair, which is measured here
    # and not assumed, since a model this thin either fits all of it or fails.
    def weight_of(have: str, want: str):
        return ("NULL", want) if have == '"NULL"' and want != "ANY" else want

    keys = sorted(
        {
            weight_of(have, want)
            for _, combination, winners, losers in decisions
            for signature in winners + losers
            for have, want in zip(combination, signature)
            if have != want
        },
        key=str,
    )
    index = {key: at for at, key in enumerate(keys)}

    def weigh(combination, signature):
        row = np.zeros(len(keys))
        for have, want in zip(combination, signature):
            if have != want:
                row[index[weight_of(have, want)]] += 1
        return row

    # A winner is cheaper than a loser by at least one, and two winners cost
    # the same. A difference of nothing at all is a call the model cannot
    # explain whichever numbers go in it, so it is reported rather than
    # dropped.
    inequalities = []
    equalities = []
    for name, combination, winners, losers in decisions:
        rows = [weigh(combination, signature) for signature in winners]
        for row in rows[1:]:
            if (rows[0] - row).any():
                equalities.append(rows[0] - row)
        for loser in losers:
            difference = rows[0] - weigh(combination, loser)
            if not difference.any():
                print(
                    f"nothing tells {winners[0]} from {loser} in"
                    f" {name}{combination}, which duckdb decided anyway",
                    file=sys.stderr,
                )
                return 1
            inequalities.append(difference)

    answer = linprog(
        np.ones(len(keys)),
        A_ub=np.array(inequalities),
        b_ub=-np.ones(len(inequalities)),
        A_eq=np.array(equalities) if equalities else None,
        b_eq=np.zeros(len(equalities)) if equalities else None,
        bounds=[(1, 1000)] * len(keys),
        method="highs",
    )
    if answer.status != 0:
        print(
            "no cost on the target type alone explains DuckDB's choices:"
            f" {answer.message.strip()}",
            file=sys.stderr,
        )
        return 1
    weights = {key: int(round(value)) for key, value in zip(keys, answer.x)}

    def cost(source, target):
        if source == target:
            return 0
        if target != "ANY" and not reaches.get((source, target)):
            return None
        return weights.get(weight_of(source, target), 1)

    # The check the whole file rests on. Resolve every collected call the way
    # the generated table says to, and insist DuckDB agreed, down to which
    # signatures tied.
    def total(combination, signature):
        return sum(
            cost(have, want) or 0 for have, want in zip(combination, signature)
        )

    for name, combination, winners, losers in decisions:
        prices = {
            signature: total(combination, signature)
            for signature in winners + losers
        }
        cheapest = min(prices.values())
        ours = sorted(s for s in prices if prices[s] == cheapest)
        if ours != sorted(winners):
            print(
                f"the solved costs do not resolve {name}{combination}:"
                f" duckdb took {sorted(winners)} and this table would take"
                f" {ours}",
                file=sys.stderr,
            )
            return 1

    # What `typeof` calls each one, which is the key everything above is on.
    # Not the spelling, which for a decimal carries a width and for a zoned
    # time is a different word altogether. It is also what the line is named
    # after, so that the reader can hold each line against `type_name` in
    # types.mojo and catch a renumbering on either side.
    known = {identifier: names[spelling] for identifier, _, spelling in SCALARS}

    lines = [f"X {TYPE_COUNT}"]
    edges = 0
    guessed = 0
    levels = set()
    for identifier in range(TYPE_COUNT):
        target = known.get(identifier)
        if target is None:
            lines.append("- " + " ".join(["-1"] * TYPE_COUNT))
            continue
        fields = []
        for other in range(TYPE_COUNT):
            source = known.get(other)
            price = cost(source, target) if source else None
            fields.append("-1" if price is None else str(price))
            if price is not None and price > 0:
                edges += 1
                levels.add(price)
                # A pair no decision ever weighed. The solver was never asked
                # about it and the number in it is the bound and not an answer,
                # which is worth counting rather than hiding.
                if weight_of(source, target) not in weights:
                    guessed += 1
        lines.append(f"{target} " + " ".join(fields))

    text = HEADER.format(
        version=version,
        scalars=len(SCALARS),
        edges=edges,
        costs=len(levels),
        decisions=len(decisions),
        ambiguities=ambiguities,
        guessed=guessed,
        any_cost=weights.get("ANY", 1),
    )
    text += "\n".join(lines) + '\n"""\n'

    if args.check:
        if not out.exists():
            print(f"{out.relative_to(root)} does not exist", file=sys.stderr)
            return 1
        if out.read_text() != text:
            print(
                f"{out.relative_to(root)} is stale, run `pixi run gen-casts`",
                file=sys.stderr,
            )
            return 1
        print(
            f"{out.relative_to(root)} is current,"
            f" {len(decisions)} decisions agree"
        )
        return 0

    out.write_text(text)
    print(
        f"{out.relative_to(root)}: {len(SCALARS)} scalar types, {edges} implicit"
        f" casts, {len(levels)} distinct costs, {len(decisions)} decisions"
        f" agree of which {ambiguities} are ties, {guessed} costs nothing"
        f" constrains, from DuckDB {version}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
