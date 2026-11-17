#!/usr/bin/env bash
# Top-32 prompt log-probs of the BF16 and the NVFP4 teacher on the texts of an OPD rollout dump, then their agreement.
# GPUs 0-1 run the NVFP4 teacher (TP2), GPUs 2-5 the BF16 teacher (TP4).
#   WORK_DIR=<work dir> bash tools/teacher/teacher_cmp.sh <bf16 dir> <nvfp4 dir> <rollout dump jsonl> <out dir>
set -uo pipefail
T=$(cd "$(dirname "$0")" && pwd -P); ROOT=$(cd "$T/../.." && pwd -P)
WORK_DIR=${WORK_DIR:?the directory scripts/install.sh built}
BF16=${1:?bf16 teacher}; NVFP4=${2:?nvfp4 teacher}; DUMP=${3:?rollout dump jsonl}; O=${4:?out dir}
mkdir -p "$O"
PY=$WORK_DIR/envs/verl/bin/python
LD_LIBRARY_PATH=$(printf %s "${LD_LIBRARY_PATH:-}" | tr : '\n' | { grep -v '/dist-packages/torch' || true; } | paste -sd: -); export LD_LIBRARY_PATH
export WORK_DIR LOCAL_DIR=${LOCAL_DIR:-$WORK_DIR/local}
mkdir -p "$LOCAL_DIR"
source "$ROOT/scripts/gpu_process_variables.sh" "$WORK_DIR/envs/verl"
cd /tmp
CUDA_VISIBLE_DEVICES=0,1 $PY "$T/teacher_prompt_logprobs.py" "$NVFP4" 2 "$DUMP" "$O/nvfp4.json" > "$O/nvfp4.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=2,3,4,5 $PY "$T/teacher_prompt_logprobs.py" "$BF16" 4 "$DUMP" "$O/bf16.json" > "$O/bf16.log" 2>&1 &
B=$!
wait $A; echo "nvfp4 exit $?"; wait $B; echo "bf16 exit $?"
[ -f "$O/nvfp4.json" ] && [ -f "$O/bf16.json" ] && python3 "$T/compare_teacher_logprobs.py" "$O/bf16.json" "$O/nvfp4.json" | tee "$O/compare.txt"
