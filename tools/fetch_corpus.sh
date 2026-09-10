#!/usr/bin/env bash
# Fetches DuckDB's SQL test corpus at the commit the grammar is pinned to.
#
# The corpus is the oracle for the compatibility claim, so it has to be the one
# that belongs to the vendored grammar. The commit is read out of the grammar's
# VENDOR file rather than passed in, which means the two can never drift: move
# the grammar and the next fetch moves the corpus with it.
#
# Unlike the grammar this is not checked in. test/sql is 33 MB against a 5 MB
# repository, so vendoring it would make every clone six times larger forever to
# carry something only the differential harness reads. See
# docs/specs/sql/13-open-questions.md question 10.
#
# Usage: tools/fetch_corpus.sh
#
# The result lands in $FIREPANDA_CORPUS, or .cache/duckdb-corpus if that is not
# set, under a directory named after the commit. Two commits can therefore sit
# side by side, which is what makes a bisect across a grammar bump work.

set -euo pipefail

cd "$(dirname "$0")/.."

readonly UPSTREAM=https://github.com/duckdb/duckdb.git
readonly VENDOR=firepanda/sql/grammar/VENDOR
readonly WANTED=test/sql

if [ ! -f "$VENDOR" ]; then
  echo "$VENDOR is missing, so there is no commit to fetch" >&2
  exit 1
fi

sha=$(awk '/^commit:/ {print $2}' "$VENDOR")
if [ -z "$sha" ]; then
  echo "$VENDOR has no commit line" >&2
  exit 1
fi

cache=${FIREPANDA_CORPUS:-$PWD/.cache/duckdb-corpus}
dest=$cache/$sha

if [ -f "$dest/.complete" ]; then
  echo "corpus for $sha is already at $dest"
  exit 0
fi

# A fetch that dies halfway must not leave something that looks finished, so the
# work happens beside the destination and the marker file is written last.
work=$dest.partial
rm -rf "$work"
mkdir -p "$work"

echo "fetching $WANTED from $UPSTREAM at $sha"
git -C "$work" init --quiet
git -C "$work" remote add origin "$UPSTREAM"
git -C "$work" config core.sparseCheckout true
git -C "$work" sparse-checkout set --no-cone "$WANTED"
# By commit and not by tag, because the grammar is pinned by commit and a tag can
# be moved. GitHub serves an arbitrary reachable commit to a client that asks for
# one, so this needs no branch name at all.
git -C "$work" fetch --quiet --depth 1 --filter=blob:none origin "$sha"
git -C "$work" checkout --quiet FETCH_HEAD

if [ ! -d "$work/$WANTED" ]; then
  echo "$WANTED is missing at $sha, so upstream has moved its tests" >&2
  exit 1
fi

files=$(find "$work/$WANTED" -type f -name '*.test*' | wc -l | tr -d ' ')
if [ "$files" -lt 1000 ]; then
  echo "only $files test files came back, which is too few to be the corpus" >&2
  exit 1
fi

touch "$work/.complete"
rm -rf "$dest"
mv "$work" "$dest"

echo "corpus for $sha is at $dest, $files test files"
