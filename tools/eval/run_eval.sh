#!/usr/bin/env bash
# usage: run_eval.sh <model name> <episodes json> <out name> [extra args of pear_libero_eval.py]
set -euo pipefail
EVAL_DIR=$(cd "$(dirname "$0")" && pwd -P)
PEAR_ROOT=${PEAR_ROOT:-$(cd "$EVAL_DIR/../.." && pwd -P)}
WORK_DIR=${WORK_DIR:?the directory scripts/install.sh built}
RUNS_DIR=${RUNS_DIR:-$WORK_DIR/runs}
MODEL=$1; [ -d "$MODEL" ] || MODEL=$WORK_DIR/models/$1
EPIS=$EVAL_DIR/episodes/$2
OUT=$RUNS_DIR/$3
shift 3
mkdir -p "$OUT"
export PEAR_REPO=$PEAR_ROOT VLLM_PLUGINS=none
export WORK_DIR LOCAL_DIR=$WORK_DIR/local
mkdir -p "$LOCAL_DIR"
unset PYTHONPATH
# CUDA_HOME = the environment's CUDA 13 toolkit, the caches and threads, as PEAR's run_server.sh sets them
source $PEAR_ROOT/scripts/gpu_process_variables.sh $WORK_DIR/envs/server
exec $WORK_DIR/envs/server/bin/python -u $EVAL_DIR/pear_libero_eval.py --model "$MODEL" --episodes "$EPIS" --out "$OUT" "$@"
