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
# it. scripts/parser is here for grammar_types.yml, which carries the one list
# that is not derivable from the grammar itself.
readonly PATHS=(src/parser/peg/grammar scripts/parser)

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
  "$work/duckdb/scripts/parser/grammar_types.yml" "$work/duckdb/LICENSE"; do
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

# matcher_rule_overrides names the rules whose bodies the matcher ignores in
# favour of a hand written matcher. It is not derivable from the grammar and it
# is not optional: OperatorLiteral reads `Identifier` in the grammar text, so
# without the override a bare `+` parses as an identifier. Same argument as
# above for awk over a YAML parser, and the same guard below.
awk '
  /^matcher_rule_overrides:/ { collecting = 1; next }
  collecting && /^  [A-Za-z_][A-Za-z0-9_]*:[[:space:]]*$/ {
    rule = $1; sub(/:$/, "", rule); next
  }
  collecting && /^    matcher:[[:space:]]/ { print rule, $2; next }
  collecting && /^[[:alpha:]]/ { collecting = 0 }
' "$work/duckdb/scripts/parser/grammar_types.yml" | sort > "$staged/matcher_overrides.list"

if [ ! -s "$staged/matcher_overrides.list" ]; then
  echo "matcher_rule_overrides is empty or has moved in grammar_types.yml" >&2
  exit 1
fi

{
  echo "# DuckDB's PEG grammar, vendored verbatim. Do not edit anything in this"
  echo "# directory. Run tools/vendor_grammar.sh to change it."
  echo "#"
  echo "# License: MIT, see LICENSE.duckdb. memoized_rules.list is extracted from"
  echo "# scripts/parser/grammar_types.yml in the same repository at the same commit."
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
