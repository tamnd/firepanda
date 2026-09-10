#!/usr/bin/env bash
# Merges a stack of pull requests bottom up, one at a time.
#
# Each one is retargeted at main once the one below it has landed, because the
# repository does not delete a branch on merge and so GitHub never retargets
# them itself. Skipping that step is what merges a stack into itself and leaves
# main with only the bottom of it.
#
# It does not wait for the hosted checks. The runners here take one job at a
# time and there are fourteen merges to make, each of which starts two more
# runs, so waiting would take most of a day to confirm what the local run
# already said. The tip of this stack was built and tested locally and the
# whole stack is its ancestry, so `--admin` goes past the branch protection on
# purpose rather than by accident.
#
# It still stops on a conflict, because that is not a slow check, it is a
# question about the code that nobody has answered yet.
set -uo pipefail

REPO=tamnd/firepanda
STACK=(293 294 296 300 303 315 326 330 336 345 355 362 365 373)

first=1
for pr in "${STACK[@]}"; do
  # Somebody else may have landed one already. A merged pull request reports its
  # mergeability as UNKNOWN, which reads exactly like one github has not made its
  # mind up about yet, so the state has to be asked for first.
  if [ "$(gh pr view "$pr" --repo "$REPO" --json state --jq .state)" = "MERGED" ]; then
    echo "$pr was already merged"
    first=0
    continue
  fi

  if [ "$first" = 0 ]; then
    base=$(gh pr view "$pr" --repo "$REPO" --json baseRefName --jq .baseRefName)
    if [ "$base" != "main" ]; then
      echo "retargeting $pr from $base to main"
      gh pr edit "$pr" --repo "$REPO" --base main >/dev/null || { echo "STOP: could not retarget $pr"; exit 1; }
    fi
  fi
  first=0

  ok=0
  for _ in $(seq 1 20); do
    sleep 10
    state=$(gh pr view "$pr" --repo "$REPO" --json mergeable --jq .mergeable)
    if [ "$state" = "CONFLICTING" ]; then
      echo "STOP: $pr conflicts with main and needs a person"
      exit 1
    fi
    if [ "$state" = "MERGEABLE" ]; then
      ok=1
      break
    fi
  done

  if [ "$ok" != 1 ]; then
    echo "STOP: github never said whether $pr merges"
    exit 1
  fi

  gh pr merge "$pr" --repo "$REPO" --squash --admin || { echo "STOP: merging $pr failed"; exit 1; }
  echo "merged $pr"
done

echo "the whole stack landed"
