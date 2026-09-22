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
# One host holds one copy of the tree and one pixi environment, which is two and
# a half gigabytes, so several worktrees share the one directory rather than
# each having their own. That makes two runs against one host a collision, and
# the way they collide is the worst one available: the second one's `rsync
# --delete` takes files out from under the first one's compiler, and the first
# one then reports failures that are nobody's change. So a run takes a lock on
# the directory first and waits its turn. This is not hypothetical either: it
# cost an afternoon's measurement and left two suites running on one host for
# well over an hour each.
#
# Which is the other half of the same story. A run started here used to outlive
# the thing that started it, because closing an ssh connection does not stop
# what is on the far end of it. Asking for a terminal with `-tt` was supposed to
# be enough, on the theory that a terminal going away hangs up what is on it,
# and measured it is not: with the connection gone and killed, the run carried
# on for another half a minute and was still compiling when the machine was
# asked a third time. So the run is stopped by name on the way out instead.
#
# By name means by session, not by the command line, because the command line of
# a test is `mojo run -I . tests/whatever.mojo` and says nothing about which
# directory or which run it belongs to. Killing that by pattern would kill
# everybody's. Everything ssh starts shares one session id, so the session is
# both the exact set of processes this run owns and nothing else's, and the far
# end writes it into the lock where the near end can read it back.
#
# `--fast` is passed through. `FIREPANDA_TEST_JOBS` is passed through too, per
# host, which is worth setting when a host is shared with something heavy.

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

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

# Stops whatever a lock says is running and takes the lock away with it. The
# session id is read on the far end rather than brought back here, so that a
# lock with nothing in it still gets removed.
stop_run() {
  ssh -o BatchMode=yes "$1" \
    "session=\$(cat \$HOME/$remote_dir.lock/session 2> /dev/null); \
     if [ -n \"\$session\" ]; then pkill -s \"\$session\"; fi; \
     rm -rf \$HOME/$remote_dir.lock" 2> /dev/null
  return 0
}

# A lock is a directory rather than a file because `mkdir` either makes one or
# says it could not, in one step, with no flag to get wrong. A lock nobody
# released is a lock whose owner went away, which is what a closed laptop looks
# like from here, so one older than the longest a suite has ever taken is
# broken rather than waited on, and whatever it left running is stopped first.
#
# Taking one leaves a note in the log directory saying so, because a lock is
# only ever released by the run that took it. Without the note, a run that gave
# up waiting would release the lock on its way out and hand the directory to a
# third run while the second one was still compiling in it, which is the
# collision this is here to stop and is exactly what happened the first time.
take_lock() {
  local host=$1 waited=0 age
  while ! ssh -o BatchMode=yes "$host" \
      "mkdir \$HOME/$remote_dir.lock" 2> /dev/null; do
    age=$(ssh -o BatchMode=yes "$host" \
      "echo \$(( \$(date +%s) - \$(stat -c %Y \$HOME/$remote_dir.lock) ))" \
      2> /dev/null)
    if [ "${age:-0}" -gt 10800 ]; then
      echo "$host holds a lock ${age}s old, which is nobody's, so it is taken"
      stop_run "$host"
      continue
    fi
    if [ "$waited" -eq 0 ]; then
      echo "$host is already running a suite, so this waits for it"
    fi
    sleep 30
    waited=$((waited + 30))
    if [ "$waited" -gt 7200 ]; then
      echo "$host did not come free in two hours"
      return 1
    fi
  done
  : > "$logs/$host.held"
  return 0
}

drop_lock() {
  local host=$1
  [ -e "$logs/$host.held" ] || return 0
  ssh -o BatchMode=yes "$host" "rm -rf \$HOME/$remote_dir.lock" 2> /dev/null
  rm -f "$logs/$host.held"
  return 0
}

# The connections go first and the runs behind them go second. `pkill -P` is
# what reaches a connection, since the job here is the subshell around it
# rather than the connection itself.
#
# A host only gets stopped if this run still holds its lock, which on the way
# out of a finished run it does not, because the subshell released it as soon
# as its shard came back. So this only ever fires on the way out of a run that
# was interrupted, which is the only time there is anything to stop.
finish() {
  trap - EXIT
  local child host
  for child in $(jobs -p); do
    pkill -P "$child" 2> /dev/null
    kill "$child" 2> /dev/null
  done
  wait 2> /dev/null
  for host in "${hosts[@]}"; do
    if [ -e "$logs/$host.held" ]; then
      stop_run "$host"
      rm -f "$logs/$host.held"
    fi
  done
}
trap finish EXIT
# An untrapped signal kills the shell without running its exit trap, and the
# thing that gets skipped is the part that stops the runs. Exiting from the
# handler runs the exit trap the ordinary way.
trap 'exit 130' INT TERM

echo "running $shards shards on: ${hosts[*]}"

for i in "${!hosts[@]}"; do
  host=${hosts[$i]}
  shard=$((i + 1))
  (
    # The traps above belong to the shell that set them. A copy of them in here
    # would stop every other host the moment this one host finished.
    trap - EXIT INT TERM
    if ! take_lock "$host" > "$logs/$host.log" 2>&1; then
      exit 1
    fi
    # `--delete` so that a file deleted locally is deleted there too, which
    # matters because a stale test file left behind would be run and counted.
    if ! rsync -a --delete \
        --exclude '.pixi' --exclude 'build' --exclude '.cache' \
        --exclude '.git' --exclude 'target' \
        -e ssh ./ "$host:$remote_dir/" > "$logs/$host.rsync" 2>&1; then
      echo "rsync to $host failed" >> "$logs/$host.log"
      cat "$logs/$host.rsync" >> "$logs/$host.log"
      exit 1
    fi
    # The session id goes into the lock before the run starts, so that whoever
    # has to stop this knows what to stop. It is written rather than reported
    # back because the thing that most often needs it is a later run finding a
    # lock nobody released, and that run has no connection to this one at all.
    ssh "$host" "cd \$HOME/$remote_dir && \
      ps -o sid= -p \$\$ | tr -d ' ' > \$HOME/$remote_dir.lock/session && \
      FIREPANDA_TEST_SHARDS=$shards FIREPANDA_TEST_SHARD=$shard \
      ${FIREPANDA_TEST_JOBS:+FIREPANDA_TEST_JOBS=$FIREPANDA_TEST_JOBS} \
      \$HOME/.pixi/bin/pixi run test ${passthrough[*]:-}" \
      >> "$logs/$host.log" 2>&1
    echo "$?" > "$logs/$host.status"
    drop_lock "$host"
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
