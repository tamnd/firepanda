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

`table.mojo` reads the generated tables back into a `Grammar`. That is the whole
of this package for now. The matcher that walks a `Grammar`, the tokenizer under
it, and the transformer above it are the rest of M4b. See
docs/specs/sql/00-README.md.
"""

from .table import Grammar, GrammarNode, memoized_rules, overridden_rules
from .token import Token, tokenize, token_text
