#!/usr/bin/env bash
# LoRA SFT of PEAR-27B on successful multi-turn LIBERO trajectories, on one 8 x RTX 5090 node.
# The vision tower is excluded from LoRA; MultiTurnSFTDataset computes loss on assistant messages only.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../../.." && pwd -P)
cd "$ROOT"

WORK_DIR=${WORK_DIR:?the directory scripts/install.sh built}
RUN=${RUN:?the output directory}
TRAIN_DATA=${TRAIN_DATA:?the parquet made by scripts/prepare_effort_sft.py}
LOCAL_DIR=${LOCAL_DIR:-$WORK_DIR/local}
MODEL=${MODEL:-$WORK_DIR/models/Qwen3.8-27B-PEAR-LIBERO_only}
VERL_ENV=${VERL_ENV:-$WORK_DIR/envs/verl}
STEPS=${STEPS:-1}
EPOCHS=${EPOCHS:-1}
GLOBAL_BATCH=${GLOBAL_BATCH:-1}
MAX_LENGTH=${MAX_LENGTH:-40960}
TOKENS_PER_GPU=${TOKENS_PER_GPU:-6144}
LR=${LR:-1e-5}
REASONING_EFFORT=${REASONING_EFFORT:-medium}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
LOGGER=${LOGGER:-'["console","file","tensorboard"]'}
EXPORT_LORA=${EXPORT_LORA:-1}

case "$REASONING_EFFORT" in
  low|medium|xhigh) ;;
  *) echo "unsupported REASONING_EFFORT: $REASONING_EFFORT (expected low, medium, or xhigh)" >&2; exit 2 ;;
esac

mkdir -p "$RUN"
source "$ROOT/scripts/gpu_process_variables.sh" "$VERL_ENV"
export PYTHONPATH="$ROOT/pear/utils/sitecustomize:$ROOT:$ROOT/3rdparty/verl/src${PYTHONPATH:+:$PYTHONPATH}"
export VERL_FILE_LOGGER_PATH="$RUN/metrics.jsonl"

bash "$ROOT/3rdparty/verl/scripts/checkout_verl.sh" --check | tee -a "$RUN/train.log"
torchrun --standalone --nnodes=1 --nproc_per_node=8 -m verl.trainer.sft_trainer \
  data.train_files="$TRAIN_DATA" \
  data.val_files=null \
  data.train_batch_size="$GLOBAL_BATCH" \
  data.micro_batch_size_per_gpu=1 \
  data.max_token_len_per_gpu="$TOKENS_PER_GPU" \
  data.max_length="$MAX_LENGTH" \
  data.pad_mode=no_padding \
  data.truncation=error \
  data.ignore_input_ids_mismatch=true \
  data.num_workers=2 \
  data.custom_cls.path="$ROOT/scripts/pear_episode_sft_dataset.py" \
  data.custom_cls.name=PearEpisodeSFTDataset \
  +data.apply_chat_template_kwargs.reasoning_effort="$REASONING_EFFORT" \
  optim.lr="$LR" \
  model.path="$MODEL" \
  model.trust_remote_code=true \
  model.use_remove_padding=true \
  model.enable_gradient_checkpointing=true \
  model.use_liger=false \
  '+model.override_config.attn_implementation={text_config:sdpa,vision_config:sdpa}' \
  model.lora_rank="$LORA_RANK" \
  model.lora_alpha="$LORA_ALPHA" \
  model.target_modules=all-linear \
  'model.exclude_modules=.*visual.*' \
  engine.strategy=fsdp2 \
  engine.fsdp_size=8 \
  engine.ulysses_sequence_parallel_size=8 \
  engine.model_dtype=bfloat16 \
  engine.dtype=bfloat16 \
  engine.use_torch_compile=false \
  engine.optimizer_offload=true \
  engine.optimizer_offload_step=true \
  +checkpoint.save_lora_only=true \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.total_training_steps="$STEPS" \
  trainer.total_epochs="$EPOCHS" \
  trainer.default_local_dir="$RUN/ckpt" \
  trainer.project_name=pear_effort_anneal \
  trainer.experiment_name="$(basename "$RUN")" \
  trainer.logger="$LOGGER" \
  trainer.resume_mode=auto \
  hydra.run.dir="$RUN/hydra" \
  "$@" 2>&1 | tee -a "$RUN/train.log"

if [ "$EXPORT_LORA" = 1 ]; then
  CHECKPOINT=$(find "$RUN/ckpt" -maxdepth 1 -type d -name 'global_step_*' -print | sort -V | tail -n 1)
  if [ -z "$CHECKPOINT" ]; then
    echo "no FSDP actor checkpoint found under $RUN/ckpt" >&2
    exit 3
  fi
  CUDA_VISIBLE_DEVICES= python "$ROOT/scripts/export_lora_adapter.py" \
    --checkpoint-dir "$CHECKPOINT" \
    --output-root "$RUN/export" \
    --trust-remote-code 2>&1 | tee -a "$RUN/train.log"
fi
