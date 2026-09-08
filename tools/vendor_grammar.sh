#!/usr/bin/env bash
# Fetches DuckDB's PEG grammar and rewrites firepanda/sql/grammar/ from it.
#
# The grammar is the definition of the dialect we claim to be compatible with,
# so it is vendored verbatim and never edited. This script is the only thing
# allowed to write into that directory. See docs/specs/sql/03-the-grammar.md.
#
# Usage: tools/vendor_grammar.sh [ref]
#
# The ref is a tag or a branch. With no argument it re-fetches the ref already
# recorded in VENDOR, which is how you check whether upstream has moved. The ref
# is resolved to a commit SHA and it is the SHA that goes in VENDOR, because
# v2.0-cyanoptera is DuckDB's default branch rather than a tag and a branch name
# on its own pins nothing.
#
# A partial fetch must never produce a half rewritten grammar, so everything is
# assembled in a temporary directory and moved into place in one step at the end.

set -euo pipefail

cd "$(dirname "$0")/.."

readonly UPSTREAM=https://github.com/duckdb/duckdb.git
readonly DEST=firepanda/sql/grammar
# Sparse checkout of just these two directories. The grammar is 61 KB and the
# repository is not, and a blobless partial clone of the whole tree still walks
# it. src/parser/peg is here for compiled_grammar.cpp and scripts/parser for
# grammar_types.yml, which between them carry the two lists that are not
# derivable from the grammar itself.
readonly PATHS=(src/parser/peg scripts/parser)

ref=${1:-}
if [ -z "$ref" ]; then
  if [ -f "$DEST/VENDOR" ]; then
    ref=$(awk '/^ref:/ {print $2}' "$DEST/VENDOR")
  fi
  ref=${ref:-v2.0-cyanoptera}
  echo "no ref given, re-fetching $ref"
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/firepanda-grammar.XXXXXXXX")
trap 'rm -rf "$work"' EXIT

echo "fetching $UPSTREAM at $ref"
git clone --quiet --depth 1 --branch "$ref" --filter=blob:none --sparse \
  "$UPSTREAM" "$work/duckdb"
git -C "$work/duckdb" sparse-checkout set "${PATHS[@]}"

sha=$(git -C "$work/duckdb" rev-parse HEAD)
src=$work/duckdb/src/parser/peg/grammar

for required in "$src/statements" "$src/keywords" \
  "$work/duckdb/scripts/parser/grammar_types.yml" \
  "$work/duckdb/src/parser/peg/compiled_grammar.cpp" "$work/duckdb/LICENSE"; do
  if [ ! -e "$required" ]; then
    echo "upstream layout changed: $required is missing at $ref" >&2
    echo "read docs/specs/sql/03-the-grammar.md section 6 before touching this" >&2
    exit 1
  fi
done

staged=$work/staged
mkdir -p "$staged/statements" "$staged/keywords"
cp "$src"/statements/*.gram "$staged/statements/"
cp "$src"/keywords/*.list "$staged/keywords/"
cp "$work/duckdb/LICENSE" "$staged/LICENSE.duckdb"

# packrat_memoized_rules is a performance decision somebody made with a profiler
# and it is not derivable from the grammar, so it is vendored beside it rather
# than reinvented. It is a flat YAML list of bare names, which is why this is awk
# and not a YAML parser: taking on a parser dependency to read twenty two lines
# would be the more fragile choice, and the guard below fails loudly if the shape
# ever stops being a flat list.
awk '
  /^packrat_memoized_rules:/ { collecting = 1; next }
  collecting && /^[[:space:]]*-[[:space:]]/ { sub(/^[[:space:]]*-[[:space:]]*/, ""); print; next }
  collecting && /^[[:alpha:]]/ { collecting = 0 }
' "$work/duckdb/scripts/parser/grammar_types.yml" > "$staged/memoized_rules.list"

if [ ! -s "$staged/memoized_rules.list" ]; then
  echo "packrat_memoized_rules is empty or has moved in grammar_types.yml" >&2
  exit 1
fi

# The rule overrides name the rules whose bodies the matcher ignores in favour
# of a hand written matcher. They are not derivable from the grammar and they
# are not optional: OperatorLiteral reads `Identifier` in the grammar text, so
# without the override a bare `+` parses as an identifier.
#
# These come from compiled_grammar.cpp and not from grammar_types.yml, even
# though the yml has a matcher_rule_overrides block that looks like the same
# table. The yml is input to the transformer generator and its `matcher` field
# is a result type rather than a matcher class, so for ReservedKeyword it says
# identifier_string while the parser installs a ReservedIdentifierMatcher. The
# C++ is what runs, so the C++ is what gets read.
#
# The second column is the matcher class in snake case and the third is the
# suggestion it was constructed with. The suggestion is not autocomplete
# trivia: IdentifierMatcher reads it to decide which keyword category the rule
# tolerates and whether a single quoted string counts as a name in that
# position, so a table name and a type name behave differently.
awk '
  function snake(s,   out, i, c) {
    sub(/Matcher$/, "", s)
    for (i = 1; i <= length(s); i++) {
      c = substr(s, i, 1)
      if (c ~ /[A-Z]/ && i > 1) out = out "_"
      out = out tolower(c)
    }
    return out
  }
  /START GENERATED RULE OVERRIDES/ { collecting = 1; next }
  /END GENERATED RULE OVERRIDES/ { collecting = 0 }
  collecting { body = body $0 }
  END {
    gsub(/[ \t]/, "", body)
    n = split(body, calls, ";")
    for (i = 1; i <= n; i++) {
      if (calls[i] !~ /AddTerminalRuleOverride\(overrides,"/) continue
      if (!match(calls[i], /"[A-Za-z_]+"/)) continue
      rule = substr(calls[i], RSTART + 1, RLENGTH - 2)
      if (!match(calls[i], /make_uniq<[A-Za-z]+>/)) continue
      matcher = snake(substr(calls[i], RSTART + 10, RLENGTH - 11))
      suggestion = "none"
      if (match(calls[i], /SuggestionState::SUGGEST_[A-Z_]+/))
        suggestion = tolower(substr(calls[i], RSTART + 25, RLENGTH - 25))
      print rule, matcher, suggestion
    }
  }
' "$work/duckdb/src/parser/peg/compiled_grammar.cpp" |
  sort > "$staged/matcher_overrides.list"

if [ ! -s "$staged/matcher_overrides.list" ]; then
  echo "the generated rule overrides block is empty or has moved in" \
    "compiled_grammar.cpp" >&2
  exit 1
fi

if awk 'NF != 3 { exit 1 }' "$staged/matcher_overrides.list"; then :; else
  echo "a rule override line is not a rule, a matcher and a suggestion" >&2
  exit 1
fi

{
  echo "# DuckDB's PEG grammar, vendored verbatim. Do not edit anything in this"
  echo "# directory. Run tools/vendor_grammar.sh to change it."
  echo "#"
  echo "# License: MIT, see LICENSE.duckdb. memoized_rules.list is extracted from"
  echo "# scripts/parser/grammar_types.yml and matcher_overrides.list from"
  echo "# src/parser/peg/compiled_grammar.cpp, in the same repository at the same"
  echo "# commit."
  echo
  echo "upstream: https://github.com/duckdb/duckdb"
  echo "ref: $ref"
  echo "commit: $sha"
  echo "retrieved: $(date -u +%Y-%m-%d)"
  echo
  echo "# sha256 of every vendored file, relative to this directory."
  (cd "$staged" && find . -type f ! -name VENDOR | sed 's|^\./||' | sort |
    while IFS= read -r f; do
      echo "$(shasum -a 256 "$f" | cut -d' ' -f1)  $f"
    done)
} > "$staged/VENDOR"

if [ -d "$DEST" ] && diff -qr "$DEST" "$staged" \
  --exclude=VENDOR --exclude=generated > /dev/null 2>&1; then
  echo "grammar is unchanged at $ref ($sha)"
  # VENDOR still gets the new commit, because "unchanged content at a newer
  # commit" is a fact worth recording and is not the same as not having looked.
  cp "$staged/VENDOR" "$DEST/VENDOR"
  exit 0
fi

# generated/ is written by tools/gen_grammar.py and is not part of the vendored
# input, so it survives the swap and is regenerated in the next step.
mkdir -p "$DEST"
rm -rf "$DEST/statements" "$DEST/keywords"
cp -R "$staged/statements" "$staged/keywords" "$DEST/"
cp "$staged/LICENSE.duckdb" "$staged/memoized_rules.list" \
  "$staged/matcher_overrides.list" "$staged/VENDOR" "$DEST/"

echo "vendored $ref ($sha) into $DEST"
echo "next: python tools/gen_grammar.py, then read the diff"
