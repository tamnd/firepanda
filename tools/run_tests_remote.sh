#!/usr/bin/env bash
# Runs the test suite across several machines, one shard each.
#
# `tools/run_tests.sh` already knows how to run a shard. What it cannot do is
# find more machines, and more machines is the only lever that works on this
# suite. Its own header records why: a test file spends most of its wall clock
# compiling the library again, precompiling the package does not save anything,
# and splitting a file does not either, because the price is instantiating the
# engine at the point of use and one call pays all of it.
#
# So this rsyncs the working tree to each host, runs one shard there, and brings
# the logs back. The hosts come from `FIREPANDA_TEST_HOSTS` or default to the
# three in `~/.ssh/config`. There is one shard per host and the shards are
# balanced by `tools/test_costs.txt`, which is what `run_tests.sh` does already.
#
# Why this exists at all: the laptop these worktrees live on runs several other
# projects at once, and when it does, a suite that costs half an hour idle costs
# two hours. `run_tests.sh` backs its own width off when memory is short, which
# keeps the run honest but does not make it quick. The servers are not idle
# either, but they are three, and three loaded machines beat one loaded machine
# that is also running the editor.
#
# The working tree is copied rather than pulled, on purpose. The whole point is
# to test the change in front of you, which is usually not committed yet.
#
# `--fast` is passed through. `FIREPANDA_TEST_JOBS` is passed through too, per
# host, which is worth setting when a host is shared with something heavy.

set -uo pipefail

cd "$(dirname "$0")/.."

hosts_default="server1 server2 server3"
read -r -a hosts <<< "${FIREPANDA_TEST_HOSTS:-$hosts_default}"

# A shard does not finish until its host does, so one saturated host holds the
# whole run. This is not hypothetical: the first time this ran, server1 was
# carrying somebody else's build at a load average of 137 on four cores, and
# forty minutes in it had not finished precompiling the library while the other
# two were most of the way through their shards.
#
# So each host is asked what it is already carrying and dropped if the answer is
# more than eight times its cores. Eight rather than one because these machines
# are shared on purpose and are never idle: measured over an afternoon, the two
# that are worth using sit between two and six times their cores and the one
# that is not sits between ten and forty. The threshold has to separate those
# two populations and nothing finer, because a stricter one waits for an idle
# machine that never arrives. A host that is dropped is named, since the
# alternative is a run that is quietly a third smaller than it looks.
#
# Memory is asked about for the same reason and is the one that actually bit.
# `run_tests.sh` backs its own width off to fit what is spare, so a host with
# half a gigabyte left does not thrash, it runs one file at a time, and one file
# at a time through a third of the suite is slower than not using the host at
# all. Two files' worth is the floor, which at the three quarters of a gigabyte
# a compile wants is a gigabyte and a half.
#
# `FIREPANDA_TEST_ALL_HOSTS` skips the check, for when the load is known to be
# about to go away.
if [ -z "${FIREPANDA_TEST_ALL_HOSTS:-}" ]; then
  usable=()
  for host in "${hosts[@]}"; do
    reading=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$host" \
      'echo $(nproc) $(cut -d" " -f1 /proc/loadavg) \
       $(awk "/^MemAvailable:/ { print int(\$2 / 1024) }" /proc/meminfo)' 2>/dev/null)
    if [ -z "$reading" ]; then
      echo "$host did not answer, so it is left out"
      continue
    fi
    read -r cores load spare <<< "$reading"
    if [ "${load%%.*}" -gt $((cores * 8)) ]; then
      echo "$host is at a load of $load on $cores cores, so it is left out"
      continue
    fi
    if [ "${spare:-0}" -lt 1536 ]; then
      echo "$host has ${spare}MB of memory going spare, so it is left out"
      continue
    fi
    usable[${#usable[@]}]=$host
  done
  hosts=("${usable[@]}")
fi

shards=${#hosts[@]}
if [ "$shards" -lt 1 ]; then
  echo "no hosts to run on" >&2
  exit 1
fi

remote_dir=${FIREPANDA_TEST_REMOTE_DIR:-fp-sql-ci}
passthrough=("$@")

# The logs go under `.cache/` rather than under `build/`, because `build/` is
# what gets deleted first when this machine runs out of disk, which it does.
logs=.cache/runs/remote.$$
rm -rf "$logs"
mkdir -p "$logs"

echo "running $shards shards on: ${hosts[*]}"

for i in "${!hosts[@]}"; do
  host=${hosts[$i]}
  shard=$((i + 1))
  (
    # `--delete` so that a file deleted locally is deleted there too, which
    # matters because a stale test file left behind would be run and counted.
    if ! rsync -a --delete \
        --exclude '.pixi' --exclude 'build' --exclude '.cache' \
        --exclude '.git' --exclude 'target' \
        -e ssh ./ "$host:$remote_dir/" > "$logs/$host.rsync" 2>&1; then
      echo "rsync to $host failed" > "$logs/$host.log"
      cat "$logs/$host.rsync" >> "$logs/$host.log"
      exit 1
    fi
    ssh "$host" "cd \$HOME/$remote_dir && \
      FIREPANDA_TEST_SHARDS=$shards FIREPANDA_TEST_SHARD=$shard \
      ${FIREPANDA_TEST_JOBS:+FIREPANDA_TEST_JOBS=$FIREPANDA_TEST_JOBS} \
      \$HOME/.pixi/bin/pixi run test ${passthrough[*]:-}" \
      > "$logs/$host.log" 2>&1
    echo $? > "$logs/$host.status"
  ) &
done
wait

failed=0
for i in "${!hosts[@]}"; do
  host=${hosts[$i]}
  echo
  echo "======== $host (shard $((i + 1)) of $shards)"
  if [ -e "$logs/$host.log" ]; then
    cat "$logs/$host.log"
  else
    echo "no output came back from $host"
    failed=$((failed + 1))
    continue
  fi
  status=$(cat "$logs/$host.status" 2>/dev/null || echo 1)
  [ "$status" -eq 0 ] || failed=$((failed + 1))
done

echo
if [ "$failed" -ne 0 ]; then
  echo "$failed of $shards shards failed"
  echo "the logs are in $logs"
  exit 1
fi
echo "all $shards shards passed"
rm -rf "$logs"
