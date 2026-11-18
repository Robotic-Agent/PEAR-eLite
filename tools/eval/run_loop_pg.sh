#!/usr/bin/env bash
# usage: run_loop_pg.sh <model name> <episodes json> <out name> [extra args]: run_loop.sh for several instances on one
# machine: each run_eval.sh in its own process group, and a crash kills that group only (its vLLM and simulator
# workers), not the other instances'. Restarts while it exits 3 (engine dead), at most MAX_RESTARTS times.
EVAL_DIR=$(cd "$(dirname "$0")" && pwd -P)
PEAR_ROOT=${PEAR_ROOT:-$(cd "$EVAL_DIR/../.." && pwd -P)}
WORK_DIR=${WORK_DIR:?the directory scripts/install.sh built}
RUNS_DIR=${RUNS_DIR:-$WORK_DIR/runs}
MAX_RESTARTS=${MAX_RESTARTS:-20}
for i in $(seq 0 "$MAX_RESTARTS"); do
  echo "== run $i $(date -u)"
  [ -f "$RUNS_DIR/$3/STOP" ] && { echo "STOP file: not restarting"; exit 0; }
  setsid "$EVAL_DIR/run_eval.sh" "$@" &
  pid=$!
  wait "$pid"
  code=$?
  kill -9 -- "-$pid" 2> /dev/null
  sleep 15
  echo "== run $i exit $code $(date -u)"
  [ "$code" = 3 ] || exit "$code"
done
