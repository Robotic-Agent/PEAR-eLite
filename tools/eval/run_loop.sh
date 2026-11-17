#!/usr/bin/env bash
# usage: run_loop.sh <model name> <episodes json> <out name> [extra args]: run_eval.sh until it ends without an engine
# crash (exit 3 = engine dead: restart, resuming the episodes left), at most MAX_RESTARTS times
EVAL_DIR=$(cd "$(dirname "$0")" && pwd -P)
PEAR_ROOT=${PEAR_ROOT:-$(cd "$EVAL_DIR/../.." && pwd -P)}
WORK_DIR=${WORK_DIR:?the directory scripts/install.sh built}
RUNS_DIR=${RUNS_DIR:-$WORK_DIR/runs}
MAX_RESTARTS=${MAX_RESTARTS:-20}
for i in $(seq 0 "$MAX_RESTARTS"); do
  echo "== run $i $(date -u)"
  [ -f "$RUNS_DIR/$3/STOP" ] && { echo "STOP file: not restarting"; exit 0; }
  "$EVAL_DIR/run_eval.sh" "$@"
  code=$?
  # what a crashed engine leaves: its workers and the simulator workers of this run
  pkill -9 -f "[V]LLM::" 2> /dev/null
  pkill -f "[l]ibero_worker.py" 2> /dev/null
  sleep 15
  echo "== run $i exit $code $(date -u)"
  [ "$code" = 3 ] || exit "$code"
done
