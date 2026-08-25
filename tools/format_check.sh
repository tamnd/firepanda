#!/usr/bin/env bash
# Fails if anything under the given roots is not formatted the way `mojo format`
# would format it.
#
# Mojo 1.0's formatter has no --check and no --diff; it rewrites files in place
# and that is all it does. So the check is done the long way round: copy the
# sources somewhere else, format the copy, diff the two. Formatting the working
# tree and reading `git diff` would be three lines shorter and would also report
# every unrelated edit a developer happens to have in flight, which is a bad
# trade for a script whose whole job is to be believed.

set -uo pipefail

cd "$(dirname "$0")/.."

roots=("$@")
if [ "${#roots[@]}" -eq 0 ]; then
  roots=(firepanda tests benchmarks tools)
fi

sources=()
while IFS= read -r file; do
  sources+=("$file")
done < <(find "${roots[@]}" -name '*.mojo' | sort)

if [ "${#sources[@]}" -eq 0 ]; then
  echo "no Mojo sources under: ${roots[*]}" >&2
  exit 1
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

for file in "${sources[@]}"; do
  mkdir -p "$work/$(dirname "$file")"
  cp "$file" "$work/$file"
done

if ! (cd "$work" && mojo format --quiet "${roots[@]}"); then
  echo "mojo format failed on a copy of the tree" >&2
  exit 1
fi

status=0
for file in "${sources[@]}"; do
  if ! cmp -s "$file" "$work/$file"; then
    diff -u --label "$file" --label "$file (formatted)" "$file" "$work/$file"
    status=1
  fi
done

if [ "$status" -ne 0 ]; then
  echo >&2
  echo "run 'pixi run format' and commit the result" >&2
  exit 1
fi

echo "${#sources[@]} files are formatted"
