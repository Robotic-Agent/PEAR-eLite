#!/usr/bin/env bash
# Full-parameter SFT of Qwen3.5-4B (PEAR chat template) on successful multi-turn LIBERO trajectories,
# one 8 x H100 80GB node. MultiTurnSFTDataset computes loss on assistant messages only.
# Saves an HF model with every checkpoint.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../../.." && pwd -P)
cd "$ROOT"

WORK_DIR=${WORK_DIR:?the directory scripts/install.sh built}
RUN=${RUN:?the output directory}
TRAIN_DATA=${TRAIN_DATA:?the parquet made by scripts/prepare_effort_sft.py}
MODEL=${MODEL:-$WORK_DIR/models/Qwen3.5-4B-pear}
VERL_ENV=${VERL_ENV:-$WORK_DIR/envs/verl}
STEPS=${STEPS:-null}
EPOCHS=${EPOCHS:-3}
GLOBAL_BATCH=${GLOBAL_BATCH:-16}
MAX_LENGTH=${MAX_LENGTH:-49152}
SP=${SP:-4}
TOKENS_PER_GPU=${TOKENS_PER_GPU:-$((MAX_LENGTH / SP))}
LR=${LR:-1e-5}
REASONING_EFFORT=${REASONING_EFFORT:-xhigh}
LOGGER=${LOGGER:-'["console","file","tensorboard"]'}

case "$REASONING_EFFORT" in
  low|medium|xhigh) ;;
  *) echo "unsupported REASONING_EFFORT: $REASONING_EFFORT" >&2; exit 2 ;;
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
  engine.strategy=fsdp2 \
  engine.fsdp_size=8 \
  engine.ulysses_sequence_parallel_size="$SP" \
  engine.model_dtype=bfloat16 \
  engine.dtype=bfloat16 \
  engine.use_torch_compile=false \
  'checkpoint.save_contents=["model","optimizer","extra","hf_model"]' \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.total_training_steps="$STEPS" \
  trainer.total_epochs="$EPOCHS" \
  trainer.default_local_dir="$RUN/ckpt" \
  trainer.project_name=pear_effort_anneal_4b \
  trainer.experiment_name="$(basename "$RUN")" \
  trainer.logger="$LOGGER" \
  trainer.resume_mode=auto \
  hydra.run.dir="$RUN/hydra" \
  "$@" 2>&1 | tee -a "$RUN/train.log"
