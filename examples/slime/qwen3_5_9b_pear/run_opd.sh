#!/usr/bin/env bash
# Replayed on-policy distillation (ReOPD) of Qwen3.8-27B-PEAR-LIBERO_only into a Qwen3.5-9B-pear student on slime,
# one 8 x H100 node, with a per-turn thinking curriculum (reasoning effort xhigh -> medium -> low, a response cap per
# stage).
#
# A prompt is a recorded 27B episode cut before one of its assistant turns (pear_slime/opd_data.py); the student
# generates that turn (pear_slime/opd_generate.py, slime's SGLang engines colocated with training), the teacher SGLang
# server returns its token log-probs (pear_slime/opd_teacher.py), and the loss is slime's OPD: advantage = -kl_coef x
# (log p_student - log p_teacher) per token (reverse KL), no task reward.
#
#   source $SLIME_PREFIX/activate.sh      # env/slime
#   STUDENT=<HF dir, e.g. the slime SFT's hf/iter_71> TEACHER=<27B HF dir> TRAIN_DATA=<opd_data.py output> RUN=<dir> \
#     bash examples/slime/qwen3_5_9b_pear/run_opd.sh [extra slime args]
#
# STAGES="effort:steps:thinking_budget ..." (default "xhigh:40:32768 medium:40:8192 low:40:8192"): each stage resumes
# from $RUN/ckpt and runs to its cumulative step count; a rerun continues where the last one stopped. The budgets are
# ceilings near the 99th percentile of the 27B's turn lengths at that effort (xhigh p99 32k, medium 7.2k, low 5.8k);
# the shortening comes from the effort itself, the teacher being scored under the same effort. A turn that reaches the
# budget gets its thinking closed and writes its action (pear_slime/opd_generate.py, PEAR_OPD_SOLUTION_BUDGET).
# Smoke: STAGES="xhigh:2:4096 low:4:2048" SAVE_INTERVAL=2 BATCH=6
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd -P)
SLIME_PREFIX=${SLIME_PREFIX:?source the env/slime activate.sh}
SLIME=$SLIME_PREFIX/src/slime
STUDENT=${STUDENT:?student HF directory}
TEACHER=${TEACHER:?teacher HF directory}
TRAIN_DATA=${TRAIN_DATA:?parquet from pear_slime/opd_data.py}
RUN=${RUN:?output directory}
STAGES=${STAGES:-"xhigh:40:32768 medium:40:8192 low:40:8192"}
ACTOR_GPUS=${ACTOR_GPUS:-0,1,2,3,4,5}
TEACHER_GPUS=${TEACHER_GPUS:-6,7}
TP=${TP:-2}
CP=${CP:-3}
BATCH=${BATCH:-32}
LR=${LR:-1e-6}
KL_COEF=${KL_COEF:-1.0}
SAVE_INTERVAL=${SAVE_INTERVAL:-20}
SGLANG_MEM=${SGLANG_MEM:-0.5}
TEACHER_PORT=${TEACHER_PORT:-13141}
TEACHER_CONTEXT=${TEACHER_CONTEXT:-98304}
RAY_PORT=${RAY_PORT:-6390}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8275}
NUM_ACTOR_GPUS=$(awk -F, '{print NF}' <<< "$ACTOR_GPUS")
TEACHER_TP=$(awk -F, '{print NF}' <<< "$TEACHER_GPUS")

mkdir -p "$RUN"
LOG=$RUN/train.log
export PYTHONPATH=$ROOT:$SLIME:$SLIME_PREFIX/src/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1 CUDA_DEVICE_MAX_CONNECTIONS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0 no_proxy=127.0.0.1 RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray-pear-slime-$RAY_PORT}
mkdir -p "$RAY_TMPDIR"

PIDS=()
cleanup() {
  for p in "${PIDS[@]}"; do kill -TERM -- -"$p" 2> /dev/null || true; done
  sleep 5
  for p in "${PIDS[@]}"; do kill -KILL -- -"$p" 2> /dev/null || true; done
}
trap cleanup EXIT

# the teacher: prefill only (max_new_tokens 0, log-probs of the whole sequence)
CUDA_VISIBLE_DEVICES=$TEACHER_GPUS setsid python3 -m sglang.launch_server --model-path "$TEACHER" \
  --host 127.0.0.1 --port "$TEACHER_PORT" --tp "$TEACHER_TP" --mem-fraction-static 0.85 \
  --context-length "$TEACHER_CONTEXT" --chunked-prefill-size 8192 --trust-remote-code \
  >> "$RUN/teacher.log" 2>&1 &
PIDS+=($!)
for i in $(seq 360); do
  curl -sf "http://127.0.0.1:$TEACHER_PORT/health_generate" > /dev/null && break
  kill -0 "${PIDS[0]}" 2> /dev/null || { echo "teacher server exited"; tail -30 "$RUN/teacher.log"; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:$TEACHER_PORT/get_model_info" | tee -a "$LOG"
echo

# A private Ray instance on its own ports and temp dir, on the actor GPUs only
CUDA_VISIBLE_DEVICES=$ACTOR_GPUS setsid ray start --head --block --node-ip-address 127.0.0.1 --port "$RAY_PORT" \
  --num-gpus "$NUM_ACTOR_GPUS" --temp-dir "$RAY_TMPDIR" --dashboard-port "$RAY_DASHBOARD_PORT" \
  --include-dashboard false --disable-usage-stats >> "$RUN/ray.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 60); do
  ray status --address "127.0.0.1:$RAY_PORT" > /dev/null 2>&1 && break
  sleep 2
done

# shellcheck source=/dev/null
source "$SLIME/scripts/models/qwen3.5-9B.sh"
MODEL_ARGS[1]="pear_slime.model"
MODEL_ARGS[2]="get_qwen3_5_vl_model_provider"

PERF_ARGS=(
  --tensor-model-parallel-size "$TP" --sequence-parallel --pipeline-model-parallel-size 1
  --context-parallel-size "$CP"
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --micro-batch-size 1 --log-probs-chunk-size 1024
  # nominal (slime trains variable-length packed sequences), but Megatron wants a multiple of 2 x CP
  --seq-length $((2 * CP * 1024))
)
OPD_ARGS=(
  --custom-generate-function-path pear_slime.opd_generate.generate
  --prompt-data "$TRAIN_DATA" --input-key messages --metadata-key metadata --rollout-shuffle
  --rollout-batch-size "$BATCH" --n-samples-per-prompt 1 --global-batch-size "$BATCH"
  --rollout-temperature 1.0
  --custom-rm-path pear_slime.opd_teacher.reward_func
  --custom-reward-post-process-path pear_slime.opd_teacher.post_process_rewards
  --rm-url "http://127.0.0.1:$TEACHER_PORT/generate"
  --advantage-estimator grpo --use-opd --opd-type sglang --opd-kl-coef "$KL_COEF"
  --kl-loss-coef 0.0 --kl-coef 0.0 --entropy-coef 0.0 --calculate-per-token-loss
)
OPTIMIZER_ARGS=(
  --optimizer adam --lr "$LR" --lr-decay-style constant --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98
  --clip-grad 1.0 --use-distributed-optimizer
  --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer
)
SGLANG_ARGS=(--colocate --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static "$SGLANG_MEM")
MISC_ARGS=(--attention-dropout 0.0 --hidden-dropout 0.0 --attention-softmax-in-fp32 --attention-backend auto)

# slime numbers a checkpoint by its 0-based rollout id: the steps done are that id + 1
steps_done() {
  local f=$RUN/ckpt/latest_checkpointed_iteration.txt
  if [ -f "$f" ]; then echo $(($(cat "$f") + 1)); else echo 0; fi
}
total=0
for stage in $STAGES; do
  IFS=: read -r effort steps max_response <<< "$stage"
  total=$((total + steps))
  done_steps=$(steps_done)
  if [ "$done_steps" -ge "$total" ]; then
    echo "== stage $effort done ($done_steps >= $total)" | tee -a "$LOG"
    continue
  fi
  LOAD=$STUDENT
  [ -f "$RUN/ckpt/latest_checkpointed_iteration.txt" ] && LOAD=$RUN/ckpt
  echo "== stage $effort: steps to $total, max response $max_response, from $LOAD $(date -u)" | tee -a "$LOG"
  (
    cd "$SLIME"
    RAY_ADDRESS=127.0.0.1:$RAY_PORT python3 train.py \
      --actor-num-nodes 1 --actor-num-gpus-per-node "$NUM_ACTOR_GPUS" \
      "${MODEL_ARGS[@]}" \
      --hf-checkpoint "$STUDENT" --load "$LOAD" --save "$RUN/ckpt" --save-interval "$SAVE_INTERVAL" \
      --save-hf "$RUN/hf/iter_{rollout_id}" \
      --num-rollout "$total" --rollout-max-response-len "$max_response" \
      --apply-chat-template-kwargs "{\"reasoning_effort\": \"$effort\"}" \
      "${OPD_ARGS[@]}" "${PERF_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}" \
      "$@"
  ) 2>&1 | tee -a "$LOG"
  # the stage's end is a checkpoint: the next stage resumes from it
  [ "$(steps_done)" -ge "$total" ] || {
    echo "== stage $effort ended without its checkpoint at $total" | tee -a "$LOG"
    exit 1
  }
done
echo "== all stages done $(date -u)" | tee -a "$LOG"
