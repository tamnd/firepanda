"""SQL, the DuckDB dialect.

`grammar/` is DuckDB's PEG grammar, vendored verbatim and never edited. It is
data rather than source: 40 `.gram` files, 5 keyword lists, two short lists
lifted out of upstream's `grammar_types.yml` and `compiled_grammar.cpp` because
they are decisions the grammar text does not record, the MIT license they come
under, and a `VENDOR` file recording the commit they came from and the SHA-256
of each one. `tools/vendor_grammar.sh` is the only thing allowed to write into
it.

`generated/` is what `tools/gen_grammar.py` makes of that: one flat node array
covering every rule, and one sorted keyword table. It is checked in, so a
contributor with no Python and no network can still build firepanda, and CI
regenerates it and fails on any diff, which is what actually stops somebody hand
editing a table.

`table.mojo` reads the generated tables back into a `Grammar`. `token.mojo` cuts
a query into tokens and `matcher.mojo` walks the rule table over them, so
`parse(sql, grammar)` is a whole query in and a tree of rule indices out.

`ast.mojo` is the shape the rest of the engine binds against, three arenas of
fixed size nodes that hold no grammar rule names, and `printer.mojo` turns one
back into SQL text. `transform.mojo` is what sits between the parse tree and
the AST, and it is the only file that knows a grammar rule name.

`unsupported.mojo` is the line between what the grammar accepts and what
firepanda runs. Every refusal is an entry in its table rather than a `raise`
written where the cases ran out, which is what lets `sql_support()` list the
whole set. See docs/specs/sql/00-README.md.
"""

from .ast import Ast, Expr, Ref, Stmt
from .matcher import Parse, ParseNode, parse, parse_rule, parse_unfiltered
from .printer import (
    needs_quoting,
    print_expr,
    print_ref,
    print_stmt,
    quote_name,
    quote_string,
)
from .table import Grammar, GrammarNode, memoized_rules, overridden_rules
from .token import Token, tokenize, token_text
from .transform import Transform
from .unsupported import NO_REFUSAL, Refusal, feature_of, refusal, sql_support
