#!/usr/bin/env bash
# Distill PEAR-27B into Qwen3.5-4B on one 8 x RTX 5090 node. The allocation is 2 GPUs for the FSDP student actor,
# 2 GPUs for two TP1 student rollout replicas and 4 GPUs for one TP4 teacher. Extra arguments are Hydra overrides.
#
# A one-update smoke test:
#   WORK_DIR=<work dir> RUN=<run dir> \
#     bash examples/models/qwen3_5_4b_pear/distill/run_opd_5090.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../../.." && pwd -P)
cd "$ROOT"

# ---- user-adjustable ----
WORK_DIR=${WORK_DIR:?the directory scripts/install.sh built}
RUN=${RUN:?the run directory}
STUDENT_MODEL=${STUDENT_MODEL:-$WORK_DIR/models/Qwen3.5-4B-pear}
TEACHER_MODEL=${TEACHER_MODEL:-$WORK_DIR/models/Qwen3.8-27B-PEAR-LIBERO_only}
VERL_ENV=${VERL_ENV:-$WORK_DIR/envs/verl}
LOCAL_DIR=${LOCAL_DIR:-/var/tmp/${USER:-$(id -un)}/pear}
EPISODES=${EPISODES:-$ROOT/pear/envs/libero/configs/libero_opd_smoke.json}

STEPS=${STEPS:-1}
BATCH=${BATCH:-2}
N=${N:-1}
MAX_PROMPT=${MAX_PROMPT:-4096}
MAX_RESPONSE=${MAX_RESPONSE:-8192}
TOPK=${TOPK:-32}
ACTOR_LR=${ACTOR_LR:-1e-6}

TRAINER_GPUS=${TRAINER_GPUS:-2}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}
TEACHER_GPUS=${TEACHER_GPUS:-4}
TEACHER_TP=${TEACHER_TP:-4}
LOGGER=${LOGGER:-'["console","file","tensorboard"]'}
# ---- end user-adjustable ----

if (( TRAINER_GPUS + ROLLOUT_GPUS + TEACHER_GPUS != 8 )); then
  echo "TRAINER_GPUS + ROLLOUT_GPUS + TEACHER_GPUS must equal 8"
  exit 2
fi
if (( TEACHER_GPUS % TEACHER_TP != 0 )); then
  echo "TEACHER_TP must divide TEACHER_GPUS"
  exit 2
fi

MAX_MODEL_LEN=$((MAX_PROMPT + MAX_RESPONSE + 1))
TOOL_CONFIG=$ROOT/pear/envs/libero/configs/tool_config.yaml
VERL_SRC=$ROOT/3rdparty/verl/src
RUN_DIR=$(readlink -m "$RUN")
START_TIME=$(date -u +%Y%m%dT%H%M%SZ)

DATA_ARGS=(
  "data.train_files=[$EPISODES]"
  "data.val_files=[$EPISODES]"
  data.max_prompt_length="$MAX_PROMPT"
  data.max_response_length="$MAX_RESPONSE"
  data.train_batch_size="$BATCH"
  +data.apply_chat_template_kwargs.enable_thinking=true
  +data.apply_chat_template_kwargs.reasoning_effort=xhigh
)

MODEL_ARGS=(
  +qwen3_5=config
  actor_rollout_ref.model.path="$STUDENT_MODEL"
  actor_rollout_ref.model.trust_remote_code=True
  actor_rollout_ref.model.use_remove_padding=False
  actor_rollout_ref.model.use_fused_kernels=False
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  '+actor_rollout_ref.model.override_config.attn_implementation={text_config:sdpa,vision_config:sdpa}'
)

ACTOR_ARGS=(
  actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH"
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.use_torch_compile=False
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_MODEL_LEN"
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR"
  actor_rollout_ref.actor.fsdp_config.kernels=null
  actor_rollout_ref.actor.fsdp_config.use_gpu_backend=False
  actor_rollout_ref.actor.fsdp_config.mixed_precision.fp32_residual=False
  actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1
  actor_rollout_ref.actor.fsdp_config.fsdp_size="$TRAINER_GPUS"
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload_step=True
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
  actor_rollout_ref.actor.fsdp_config.hsdp_all_reduce_on_last_micro_batch=False
)

ROLLOUT_ARGS=(
  actor_rollout_ref.rollout.nnodes=1
  actor_rollout_ref.rollout.n_gpus_per_node="$ROLLOUT_GPUS"
  actor_rollout_ref.rollout.n="$N"
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN"
  actor_rollout_ref.rollout.max_num_seqs=2
  actor_rollout_ref.rollout.max_num_batched_tokens=4096
  actor_rollout_ref.rollout.standalone_gpu_memory_utilization=0.80
  actor_rollout_ref.rollout.kernels=null
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$MAX_MODEL_LEN"
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG"
  +actor_rollout_ref.rollout.limit_images=24
  actor_rollout_ref.rollout.custom.node_label=rollout
)

DISTILLATION_ARGS=(
  distillation.enabled=True
  distillation.nnodes=1
  distillation.n_gpus_per_node="$TEACHER_GPUS"
  distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
  distillation.teacher_models.teacher_model.inference.name=vllm
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="$TEACHER_TP"
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.80
  distillation.teacher_models.teacher_model.inference.max_model_len="$MAX_MODEL_LEN"
  distillation.teacher_models.teacher_model.inference.max_num_seqs="$BATCH"
  distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=4096
  +distillation.teacher_models.teacher_model.inference.limit_images=24
  distillation.distillation_loss.loss_mode=k3
  distillation.distillation_loss.topk="$TOPK"
  distillation.distillation_loss.use_task_rewards=False
  distillation.distillation_loss.use_policy_gradient=False
  distillation.distillation_loss.loss_max_clamp=10.0
  distillation.distillation_loss.log_prob_min_clamp=-10.0
)

TRAINER_ARGS=(
  run_dir="$RUN_DIR"
  trainer.nnodes=1
  trainer.n_gpus_per_node="$TRAINER_GPUS"
  trainer.node_label=trainer
  trainer.total_training_steps="$STEPS"
  trainer.total_epochs=1
  trainer.val_before_train=False
  trainer.test_freq=-1
  trainer.save_freq=1
  trainer.logger="$LOGGER"
  trainer.project_name=pear_opd
  trainer.experiment_name="$(basename "$RUN")"
  trainer.default_local_dir="$RUN/ckpt"
  trainer.rollout_data_dir="$RUN/rollout_dumps"
  trainer.validation_data_dir="$RUN/val_dumps"
  hydra.run.dir="$RUN/hydra/$START_TIME"
  run_sizes.sessions_per_replica=1
)

ALGORITHM_ARGS=(
  algorithm.filter_groups.enable=False
  algorithm.rollout_correction.score_centering=False
  actor_rollout_ref.actor.policy_loss.rollout_correction.score_centering=False
  actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode=False
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
  algorithm.rollout_correction.bypass_mode=False
)

ENV_VAR=+ray_kwargs.ray_init.runtime_env.env_vars
EXTERNAL_MODULES=pear.rollout.server,pear.models.qwen3_5.tool_parser
RAY_ARGS=(
  "$ENV_VAR.VERL_USE_EXTERNAL_MODULES='$EXTERNAL_MODULES'"
  "$ENV_VAR.PYTHONPATH='$ROOT/pear/utils/sitecustomize:$ROOT:$VERL_SRC'"
  "$ENV_VAR.VERL_FILE_LOGGER_PATH=$RUN/metrics_$START_TIME.jsonl"
  "$ENV_VAR.LIBERO_CONFIG_PATH=$WORK_DIR/envs/libero/etc/libero"
  "$ENV_VAR.LIBERO_PYTHON=$WORK_DIR/envs/libero/bin/python"
  "$ENV_VAR.LIBERO_MESA_DIR=$WORK_DIR/envs/mesa"
)

mkdir -p "$RUN"
source "$ROOT/scripts/gpu_process_variables.sh" "$VERL_ENV"
export RAY_ADDRESS=${RAY_ADDRESS:-auto}
export PEAR_LOG_DIR=$RUN/sim
export WANDB_DIR=$RUN
export VERL_USE_EXTERNAL_MODULES=$EXTERNAL_MODULES
export PYTHONPATH=$ROOT/pear/utils/sitecustomize:$ROOT:$VERL_SRC${PYTHONPATH:+:$PYTHONPATH}

bash "$ROOT/3rdparty/verl/scripts/checkout_verl.sh" --check | tee -a "$RUN/train.log"
python -m pear.trainer.main \
  "${DATA_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  "${ACTOR_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${DISTILLATION_ARGS[@]}" \
  "${TRAINER_ARGS[@]}" \
  "${ALGORITHM_ARGS[@]}" \
  "${RAY_ARGS[@]}" \
  "$@" 2>&1 | tee -a "$RUN/train.log"
