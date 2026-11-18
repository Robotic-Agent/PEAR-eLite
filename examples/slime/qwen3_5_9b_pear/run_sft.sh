#!/usr/bin/env bash
# Full-parameter SFT of Qwen3.5-9B (PEAR chat template, vision tower included) on successful multi-turn LIBERO
# trajectories, on slime's Megatron backend: the language model in Megatron (TP x CP), the Transformers ViT replicated.
# Same data and schedule as examples/models/qwen3_5_4b_pear/train/run_sft_full_h100.sh (verl).
#
#   source $SLIME_PREFIX/activate.sh      # env/slime
#   MODEL=.../Qwen3.5-9B-pear TRAIN_DATA=<pear_slime/convert_sft_data.py output> RUN=<out dir> \
#     bash examples/slime/qwen3_5_9b_pear/run_sft.sh [extra slime args]
#
# The first run loads the HF model directly; a rerun resumes from $RUN/ckpt. HF checkpoints go to $RUN/hf/iter_<rollout id>.
# Smoke: NUM_ROLLOUT=2 SAVE_INTERVAL=2 ... (TP is at most 4: 4 query groups)
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd -P)
SLIME_PREFIX=${SLIME_PREFIX:?source the env/slime activate.sh}
SLIME=$SLIME_PREFIX/src/slime
MODEL=${MODEL:?HF model directory}
TRAIN_DATA=${TRAIN_DATA:?parquet from pear_slime/convert_sft_data.py}
RUN=${RUN:?output directory}
NUM_GPUS=${NUM_GPUS:-8}
TP=${TP:-4}
CP=${CP:-2}
GLOBAL_BATCH=${GLOBAL_BATCH:-16}
EPOCHS=${EPOCHS:-3}
NUM_ROLLOUT=${NUM_ROLLOUT:-}           # caps the number of steps (one rollout = one global batch); empty = EPOCHS
LR=${LR:-1e-5}
SAVE_INTERVAL=${SAVE_INTERVAL:-24}
REASONING_EFFORT=${REASONING_EFFORT:-xhigh}
# 1 keeps the gradient buffer in fp32. On 32 GB cards it does not fit: each TP rank holds 3.9 B parameters (the Gated
# DeltaNet layers and the ViT are not split by TP), so bf16 parameters + fp32 gradients take 23.4 GB before activations.
GRAD_FP32=${GRAD_FP32:-0}
# 1 keeps the optimizer state on the host (needed on 32 GB cards, not on 80 GB)
OPT_OFFLOAD=${OPT_OFFLOAD:-1}
RAY_PORT=${RAY_PORT:-6390}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8275}

mkdir -p "$RUN"
LOG=$RUN/train.log

# shellcheck source=/dev/null
source "$SLIME/scripts/models/qwen3.5-9B.sh"
# slime's Qwen3.5-VL provider with activation checkpointing in the ViT (pear_slime/model.py)
MODEL_ARGS[1]="pear_slime.model"
MODEL_ARGS[2]="get_qwen3_5_vl_model_provider"

LOAD=$MODEL
[ -f "$RUN/ckpt/latest_checkpointed_iteration.txt" ] && LOAD=$RUN/ckpt
CKPT_ARGS=(
  --hf-checkpoint "$MODEL"
  --load "$LOAD"
  --save "$RUN/ckpt"
  --save-interval "$SAVE_INTERVAL"
  --save-hf "$RUN/hf/iter_{rollout_id}"
)

if [ -n "$NUM_ROLLOUT" ]; then
  LENGTH_ARGS=(--num-rollout "$NUM_ROLLOUT")
else
  LENGTH_ARGS=(--num-epoch "$EPOCHS")
fi
SFT_ARGS=(
  --rollout-function-path pear_slime.sft_rollout.generate_rollout
  --prompt-data "$TRAIN_DATA"
  --input-key messages
  --metadata-key metadata
  --apply-chat-template-kwargs "{\"reasoning_effort\": \"$REASONING_EFFORT\"}"
  --rollout-shuffle
  "${LENGTH_ARGS[@]}"
  --rollout-batch-size "$GLOBAL_BATCH"
  --global-batch-size "$GLOBAL_BATCH"
  --loss-type sft_loss
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
)

# TE picks the attention backend. Qwen3.5-9B attention has head dim 256: on RTX 5090 (sm120) neither cuDNN 9.20 nor
# FlashAttention 4 trains it with packed sequences, so this recipe runs on H100.
# Qwen3.5 Gated DeltaNet: micro batch 1, no dynamic batch size (examples/geo3k_vlm/README.md)
PERF_ARGS=(
  --tensor-model-parallel-size "$TP"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size "$CP"
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --micro-batch-size 1
  --log-probs-chunk-size 1024
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "$LR"
  --lr-decay-style cosine
  --min-lr 0
  --lr-warmup-fraction 0.1
  --weight-decay 0.01
  --adam-beta1 0.9
  --adam-beta2 0.95
  --clip-grad 1.0
  --use-distributed-optimizer
)
[ "$OPT_OFFLOAD" = 1 ] && OPTIMIZER_ARGS+=(--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --attention-softmax-in-fp32
  --attention-backend auto
)
[ "$GRAD_FP32" = 1 ] && MISC_ARGS+=(--accumulate-allreduce-grads-in-fp32)

# A private Ray instance on its own ports and temp dir (the dev machine is shared); its workers inherit this env.
export PYTHONPATH=$ROOT:$SLIME:$SLIME_PREFIX/src/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1 CUDA_DEVICE_MAX_CONNECTIONS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0 no_proxy=127.0.0.1 RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray-pear-slime-$RAY_PORT}
mkdir -p "$RAY_TMPDIR"
# `ray stop` would stop every Ray instance on the machine, so this head runs in its own process group and is stopped
# by killing that group
setsid ray start --head --block --node-ip-address 127.0.0.1 --port "$RAY_PORT" --num-gpus "$NUM_GPUS" \
  --temp-dir "$RAY_TMPDIR" --dashboard-port "$RAY_DASHBOARD_PORT" --include-dashboard false --disable-usage-stats \
  >> "$RUN/ray.log" 2>&1 &
RAY_PGID=$!
trap 'kill -TERM -- -$RAY_PGID 2> /dev/null || true; sleep 5; kill -KILL -- -$RAY_PGID 2> /dev/null || true' EXIT
for _ in $(seq 60); do
  ray status --address "127.0.0.1:$RAY_PORT" > /dev/null 2>&1 && break
  sleep 2
done
ray status --address "127.0.0.1:$RAY_PORT" | tee -a "$LOG"

cd "$SLIME"
RAY_ADDRESS=127.0.0.1:$RAY_PORT python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "$NUM_GPUS" \
  "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${SFT_ARGS[@]}" "${PERF_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" "${MISC_ARGS[@]}" \
  "$@" 2>&1 | tee -a "$LOG"
